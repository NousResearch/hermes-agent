"""Explicit Kanban review policy: optional | required | disabled.

What these tests pin down (all new behaviour on top of the reviewer gate):

* Create: ``optional`` is the default; a reviewer without a policy implies
  ``required``; ``required`` needs a valid reviewer; ``optional``/``disabled``
  with a reviewer is a conflict. Every refusal is zero-write, and a conflict is
  reported as a conflict even when the reviewer name is also unknown.
* Legacy rows (NULL policy) read back their historical meaning, and the
  migration adds the column without disturbing existing cards.
* ``disabled``: the kernel refuses ``request_review`` on every route (including
  ``force=True``) with no state, event, claim or assignment change. The
  dispatcher stamps the env, the worker tool surface hides ``kanban_request_review``,
  the worker context and stop-gate nudge describe the normal completion path,
  and ``kanban_complete`` still closes the card.
* ``required``: the reviewer argument is dropped from the implementer's schema,
  a reviewer that has lost its review skill is refused at handoff with no state
  change and no fallback to completion, and the saved reviewer still routes.
* ``optional``: native review is unchanged, including the changes-requested cycle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_validation as kv
from hermes_cli import profiles as profiles_mod


PROFILES = ("default", "qa", "reviewer", "worker")


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME + empty board, with the profile roster pinned."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in (
        "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_REQUIRED_REVIEWER",
        "HERMES_KANBAN_RUN_PHASE", "HERMES_KANBAN_REVIEW_POLICY",
    ):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: name in PROFILES)
    monkeypatch.setattr(profiles_mod, "list_profile_names", lambda: list(PROFILES))
    return home


def _counts(conn) -> dict:
    return {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("tasks", "task_links", "task_events", "task_runs")
    }


def _frozen(conn, tid: str) -> dict:
    """Everything a refused handoff must leave byte-for-byte alone."""
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
    return {
        "task": tuple(row) if row is not None else None,
        "events": [tuple(r) for r in conn.execute(
            "SELECT kind, payload, run_id FROM task_events WHERE task_id = ? ORDER BY id",
            (tid,),
        )],
        "runs": [tuple(r) for r in conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (tid,),
        )],
    }


def _events(conn, tid, kind=None):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
    ).fetchall()
    out = [(r["kind"], json.loads(r["payload"]) if r["payload"] else None) for r in rows]
    return [e for e in out if e[0] == kind] if kind else out


def _seed_skill(home: Path, name: str) -> None:
    skill_dir = home / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test double\n---\n\nbody\n", encoding="utf-8",
    )


def _parse_kanban(argv: list) -> argparse.Namespace:
    root = argparse.ArgumentParser(prog="hermes")
    kc.build_parser(root.add_subparsers())
    return root.parse_args(["kanban", *argv])


def _offered(name: str) -> dict:
    from tools.registry import registry

    defs = registry.get_definitions({name}, quiet=True)
    return {d["function"]["name"]: d["function"] for d in defs}.get(name, {})


def _worker_env(monkeypatch, tid: str, *, policy=None, reviewer=None, phase=None) -> None:
    from tools.registry import invalidate_check_fn_cache

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    if policy is not None:
        monkeypatch.setenv("HERMES_KANBAN_REVIEW_POLICY", policy)
    if reviewer is not None:
        monkeypatch.setenv("HERMES_KANBAN_REQUIRED_REVIEWER", reviewer)
    if phase is not None:
        monkeypatch.setenv("HERMES_KANBAN_RUN_PHASE", phase)
    invalidate_check_fn_cache()


# ---------------------------------------------------------------------------
# Create: defaults, implied policy, conflicts. All refusals are zero-write.
# ---------------------------------------------------------------------------


def test_default_policy_is_optional_and_ungated(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain", assignee="worker")
        task = kb.get_task(conn, tid)
        assert task.review_policy == "optional"
        assert task.required_reviewer is None
        created = _events(conn, tid, "created")[0][1]
        assert created["review_policy"] == "optional"


def test_reviewer_without_policy_implies_required(board: Path) -> None:
    """Preserves the saved-reviewer behaviour: a reviewer means the gate holds."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker", reviewer="reviewer")
        task = kb.get_task(conn, tid)
        assert task.review_policy == "required"
        assert task.required_reviewer == "reviewer"


def test_explicit_required_with_valid_reviewer_persists(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated", assignee="worker", reviewer="reviewer",
            review_policy="required",
        )
        assert kb.get_task(conn, tid).review_policy == "required"


def test_required_without_reviewer_is_zero_writes(board: Path) -> None:
    with kbc.connect() as conn:
        before = _counts(conn)
        with pytest.raises(ValueError) as excinfo:
            kb.create_task(conn, title="x", assignee="worker", review_policy="required")
        assert "needs a reviewer" in str(excinfo.value)
        assert str(excinfo.value).endswith("Nothing changed.")
        assert _counts(conn) == before


@pytest.mark.parametrize("policy", ["optional", "disabled"])
def test_non_required_policy_with_reviewer_is_a_conflict(board: Path, policy: str) -> None:
    with kbc.connect() as conn:
        before = _counts(conn)
        with pytest.raises(ValueError) as excinfo:
            kb.create_task(
                conn, title="x", assignee="worker", reviewer="reviewer",
                review_policy=policy,
            )
        message = str(excinfo.value)
        assert "conflicts with a reviewer" in message
        assert message.endswith("Nothing changed.")
        assert _counts(conn) == before


def test_conflict_is_reported_before_an_unknown_reviewer(board: Path) -> None:
    """The verdict on raw inputs comes first: a contradictory request reads as a
    conflict, not as a profile-lookup error."""
    with kbc.connect() as conn:
        before = _counts(conn)
        with pytest.raises(ValueError) as excinfo:
            kb.create_task(
                conn, title="x", assignee="worker", reviewer="ghost",
                review_policy="disabled",
            )
        assert "conflicts with a reviewer" in str(excinfo.value)
        assert "not found" not in str(excinfo.value)
        assert _counts(conn) == before


def test_unknown_policy_value_is_zero_writes(board: Path) -> None:
    with kbc.connect() as conn:
        before = _counts(conn)
        with pytest.raises(ValueError) as excinfo:
            kb.create_task(conn, title="x", assignee="worker", review_policy="maybe")
        assert "review_policy must be one of optional, required, disabled" in str(excinfo.value)
        assert _counts(conn) == before


def test_disabled_card_is_ungated_for_completion(board: Path) -> None:
    """disabled means no native review; the implementer closes the card itself."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="solo", assignee="worker", review_policy="disabled")
        run = kb.claim_task(conn, tid)
        assert kb.complete_task(
            conn, tid, summary="done", expected_run_id=run.current_run_id,
        ) is True
        assert kb.get_task(conn, tid).status == "done"


# ---------------------------------------------------------------------------
# Legacy rows and migration
# ---------------------------------------------------------------------------


def test_legacy_null_policy_reads_back_its_historical_meaning(board: Path) -> None:
    with kbc.connect() as conn:
        gated = kb.create_task(conn, title="g", assignee="worker", reviewer="reviewer")
        plain = kb.create_task(conn, title="p", assignee="worker")
        conn.execute("UPDATE tasks SET review_policy = NULL WHERE id IN (?, ?)", (gated, plain))
        assert kb.get_task(conn, gated).review_policy == "required"
        assert kb.get_task(conn, plain).review_policy == "optional"


def test_migration_adds_review_policy_to_a_legacy_board(board: Path) -> None:
    import sqlite3

    db_path = board / "kanban.db"
    conn = sqlite3.connect(db_path)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "review_policy" in cols
        conn.execute("ALTER TABLE tasks DROP COLUMN review_policy")
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at, workspace_kind, required_reviewer) "
            "VALUES ('t_legacy', 'legacy gated', 'ready', 0, 'scratch', 'reviewer')"
        )
        conn.commit()
    finally:
        conn.close()

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db(db_path)
    with kbc.connect() as conn:
        task = kb.get_task(conn, "t_legacy")
        assert task.required_reviewer == "reviewer"
        assert task.review_policy == "required"
        assert "review_policy" in {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}


# ---------------------------------------------------------------------------
# Disabled: kernel refusal on every route, no writes
# ---------------------------------------------------------------------------


def test_disabled_request_review_is_refused_with_no_writes(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="solo", assignee="worker", review_policy="disabled")
        run = kb.claim_task(conn, tid)
        before = _frozen(conn, tid)
        for kwargs in (
            {"expected_run_id": run.current_run_id},
            {"expected_run_id": run.current_run_id, "force": True},
            {"expected_run_id": run.current_run_id, "reviewer": "reviewer"},
        ):
            ok, reason = kb.request_review(conn, tid, summary="done", with_reason=True, **kwargs)
            assert ok is False
            assert "review_policy=disabled" in reason and reason.endswith("Nothing changed.")
            assert _frozen(conn, tid) == before, f"refused handoff changed state for {kwargs}"
        assert kb.get_task(conn, tid).status == "running"


def test_disabled_card_cannot_be_forced_into_review_from_ready(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="solo", assignee="worker", review_policy="disabled")
        before = _frozen(conn, tid)
        ok, reason = kb.request_review(conn, tid, summary="done", force=True, with_reason=True)
        assert ok is False and "disabled" in reason
        assert _frozen(conn, tid) == before


def test_disabled_tool_handler_refuses_and_writes_nothing(
    board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="solo", assignee="worker", review_policy="disabled")
        run = kb.claim_task(conn, tid)
        before = _frozen(conn, tid)
    _worker_env(monkeypatch, tid, policy="disabled")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.current_run_id))

    out = json.loads(kt._handle_request_review({"task_id": tid, "summary": "done"}))
    assert out.get("ok") is not True
    assert "review_policy=disabled" in out["error"]
    with kbc.connect() as conn:
        assert _frozen(conn, tid) == before


def test_disabled_card_still_offers_kanban_complete_and_hides_request_review(
    board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as kt
    from tools.registry import invalidate_check_fn_cache, registry

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="solo", assignee="worker", review_policy="disabled")
    _worker_env(monkeypatch, tid, policy="disabled")
    invalidate_check_fn_cache()
    assert "kanban_request_review" not in {
        d["function"]["name"] for d in registry.get_definitions({"kanban_request_review"}, quiet=True)
    }
    assert "kanban_complete" in {
        d["function"]["name"] for d in registry.get_definitions({"kanban_complete"}, quiet=True)
    }


def test_optional_worker_keeps_request_review_and_its_reviewer_argument(
    board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.kanban_tools

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain", assignee="worker")
    _worker_env(monkeypatch, tid, policy="optional")
    schema = _offered("kanban_request_review")
    assert schema, "optional worker lost kanban_request_review"
    assert "reviewer" in schema["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Required: reviewer argument dropped for the implementer; capability refusal
# ---------------------------------------------------------------------------


def test_required_implementer_schema_omits_reviewer_argument(
    board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.kanban_tools

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker", reviewer="reviewer")
    _worker_env(monkeypatch, tid, reviewer="reviewer", phase="implementation")
    schema = _offered("kanban_request_review")
    assert schema, "required implementer lost kanban_request_review"
    properties = schema["parameters"]["properties"]
    assert "reviewer" not in properties
    assert "summary" in properties, "summary must survive the schema override"


def test_required_reviewer_run_keeps_reviewer_argument(
    board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.kanban_tools

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker", reviewer="reviewer")
    _worker_env(monkeypatch, tid, reviewer="reviewer", phase="review")
    schema = _offered("kanban_request_review")
    assert "reviewer" in schema["parameters"]["properties"]


def test_required_routes_to_saved_reviewer_and_refuses_override(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated", assignee="worker", reviewer="reviewer",
            review_policy="required",
        )
        run = kb.claim_task(conn, tid)
        ok, why = kb.request_review(
            conn, tid, summary="done", expected_run_id=run.current_run_id,
            reviewer="qa", with_reason=True,
        )
        assert ok is False and "reviewer override is not allowed" in why
        assert kb.get_task(conn, tid).status == "running"
        ok, why = kb.request_review(
            conn, tid, summary="done", expected_run_id=run.current_run_id, with_reason=True,
        )
        assert ok is True, why
        task = kb.get_task(conn, tid)
        assert task.status == "review" and task.assignee == "reviewer"


def test_required_reviewer_losing_its_skill_is_a_capability_refusal(
    board: Path,
) -> None:
    """A saved reviewer that can no longer run review is refused at handoff with no
    state change, and the refusal routes the issue to the orchestrator. The required
    gate never falls back to completion."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker", reviewer="reviewer")
        run = kb.claim_task(conn, tid)
    # The reviewer's library now exists but no longer carries sdlc-review.
    _seed_skill(board, "some-other-skill")
    with kbc.connect() as conn:
        before = _frozen(conn, tid)
        ok, reason = kb.request_review(
            conn, tid, summary="done", expected_run_id=run.current_run_id, with_reason=True,
        )
        assert ok is False
        assert "sdlc-review" in reason
        assert "Routing/capability issue for the orchestrator" in reason
        assert "never falls back to completion" in reason
        assert _frozen(conn, tid) == before
        assert kb.get_task(conn, tid).status == "running"
        assert kb.get_task(conn, tid).assignee == "worker"


def test_required_reviewer_deleted_after_create_is_refused_with_no_writes(
    board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saved reviewer whose profile is deleted after creation must not be silently
    accepted: the handoff is refused with no state change, the card is never routed to
    a profile the dispatcher cannot spawn, and it is never completed in its place."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker", reviewer="reviewer")
        run = kb.claim_task(conn, tid)
        before = _frozen(conn, tid)
    monkeypatch.setattr(
        profiles_mod, "profile_exists", lambda name: name in ("default", "qa", "worker"),
    )
    with kbc.connect() as conn:
        ok, reason = kb.request_review(
            conn, tid, summary="done", expected_run_id=run.current_run_id, with_reason=True,
        )
        assert ok is False, "a deleted reviewer profile was accepted for routing"
        assert "reviewer" in reason and "Routing/capability issue" in reason
        assert _frozen(conn, tid) == before
        assert kb.get_task(conn, tid).assignee == "worker"


def test_required_card_cannot_be_completed_by_its_implementer(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker", reviewer="reviewer")
        run = kb.claim_task(conn, tid)
        before = _frozen(conn, tid)
        with pytest.raises(kb.ReviewerGateError):
            kb.complete_task(conn, tid, summary="done", expected_run_id=run.current_run_id)
        assert _frozen(conn, tid) == before


# ---------------------------------------------------------------------------
# Optional: native review unchanged, including the changes-requested cycle
# ---------------------------------------------------------------------------


def test_optional_native_review_cycle_keeps_policy(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain", assignee="worker", reviewer=None)
        run = kb.claim_task(conn, tid)
        ok, why = kb.request_review(
            conn, tid, reviewer="reviewer", summary="first", expected_run_id=run.current_run_id, with_reason=True,
        )
        assert ok is True, why
        assert kb.get_task(conn, tid).review_policy == "optional"
        review_run = kb.claim_review_task(conn, tid)
        assert review_run is not None
        ok, implementer = kb.request_changes(conn, tid, reason="more tests")
        assert ok is True and implementer == "worker"
        assert kb.get_task(conn, tid).review_policy == "optional"


# ---------------------------------------------------------------------------
# Worker context, stop-gate nudge, goal-loop prompts, dispatcher env
# ---------------------------------------------------------------------------


def test_disabled_worker_context_names_the_normal_completion_path(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="solo", assignee="worker", review_policy="disabled")
        context = kb.build_worker_context(conn, tid)
    assert "Review policy: disabled" in context
    assert "finish with kanban_complete" in context
    assert "kanban_request_review is not available" in context


def test_required_worker_context_keeps_the_reviewer_header(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker", reviewer="reviewer")
        context = kb.build_worker_context(conn, tid)
    assert "Required reviewer: reviewer" in context
    assert "Review policy: disabled" not in context


def test_stop_nudge_offers_no_request_review_on_a_disabled_card(
    board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.kanban_stop import build_kanban_stop_nudge

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="solo", assignee="worker", review_policy="disabled")
    _worker_env(monkeypatch, tid, policy="disabled")
    nudge = build_kanban_stop_nudge(messages=[], attempts=0, task_id=tid)
    assert nudge is not None
    assert "kanban_request_review" not in nudge
    assert "kanban_complete" in nudge


def test_goal_loop_prompts_do_not_point_disabled_workers_at_request_review(
    board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import goals

    monkeypatch.setenv("HERMES_KANBAN_REVIEW_POLICY", "disabled")
    step = goals._goal_review_step()
    assert "kanban_request_review" not in step and "kanban_complete" in step
    assert "kanban_request_review" not in goals._goal_review_step(finalize=True)
    monkeypatch.delenv("HERMES_KANBAN_REVIEW_POLICY")
    assert "kanban_request_review" in goals._goal_review_step()


def test_dispatcher_stamps_the_disabled_policy_into_the_worker_env(board: Path) -> None:
    with kbc.connect() as conn:
        disabled = kb.create_task(conn, title="d", assignee="worker", review_policy="disabled")
        plain = kb.create_task(conn, title="p", assignee="worker")
        gated = kb.create_task(conn, title="g", assignee="worker", reviewer="reviewer")
        assert kbd.review_gate_env(kb.get_task(conn, disabled)) == {
            "HERMES_KANBAN_REVIEW_POLICY": "disabled",
        }
        assert kbd.review_gate_env(kb.get_task(conn, plain)) == {}
        assert kbd.review_gate_env(kb.get_task(conn, gated))["HERMES_KANBAN_REQUIRED_REVIEWER"] == "reviewer"


# ---------------------------------------------------------------------------
# Agent tool, CLI and read surfaces
# ---------------------------------------------------------------------------


def test_tool_create_conflict_is_zero_writes(board: Path) -> None:
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        before = _counts(conn)
    out = json.loads(kt._handle_create({
        "title": "x", "assignee": "worker", "reviewer": "reviewer",
        "review_policy": "disabled",
    }))
    assert out.get("ok") is not True
    assert "conflicts with a reviewer" in out["error"]
    with kbc.connect() as conn:
        assert _counts(conn) == before


def test_tool_create_persists_the_requested_policy(board: Path) -> None:
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({
        "title": "solo", "assignee": "worker", "review_policy": "disabled",
    }))
    assert out.get("ok") is True
    with kbc.connect() as conn:
        assert kb.get_task(conn, out["task_id"]).review_policy == "disabled"


def test_create_schema_declares_the_policy_enum() -> None:
    from tools.registry import registry

    prop = registry.get_schema("kanban_create")["parameters"]["properties"]["review_policy"]
    assert prop["enum"] == ["optional", "required", "disabled"]


def test_cli_create_disabled_with_reviewer_is_refused_with_no_writes(
    board: Path, capsys,
) -> None:
    with kbc.connect() as conn:
        before = _counts(conn)
    rc = kc._cmd_create(_parse_kanban([
        "create", "T", "--assignee", "worker", "--reviewer", "reviewer",
        "--review-policy", "disabled",
    ]))
    assert rc == 2
    assert "conflicts with a reviewer" in capsys.readouterr().err
    with kbc.connect() as conn:
        assert _counts(conn) == before


def test_cli_create_disabled_round_trips(board: Path, capsys) -> None:
    rc = kc._cmd_create(_parse_kanban([
        "create", "T", "--assignee", "worker", "--review-policy", "disabled",
    ]))
    assert rc == 0
    tid = capsys.readouterr().out.split()[1]
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).review_policy == "disabled"


def test_cli_request_review_on_disabled_card_is_refused_with_no_writes(
    board: Path, capsys,
) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="solo", assignee="worker", review_policy="disabled")
        kb.claim_task(conn, tid)
        before = _frozen(conn, tid)
    rc = kc._cmd_request_review(_parse_kanban(["request-review", tid, "--summary", "done"]))
    assert rc != 0
    assert "review_policy=disabled" in capsys.readouterr().err
    with kbc.connect() as conn:
        assert _frozen(conn, tid) == before


def test_dashboard_review_verb_on_disabled_card_is_refused_with_no_writes(board: Path) -> None:
    """The dashboard drag/PATCH ``review`` verb forces the handoff (force=True); the
    kernel refusal must still hold and leave the card untouched."""
    from types import SimpleNamespace

    from plugins.kanban.dashboard import plugin_api

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="solo", assignee="worker", review_policy="disabled")
        kb.claim_task(conn, tid)
        before = _frozen(conn, tid)
        payload = SimpleNamespace(summary="done", metadata=None, assignee=None)
        assert not plugin_api._STATUS_HANDLERS["review"](conn, tid, payload)
        assert _frozen(conn, tid) == before


def test_dashboard_create_body_carries_the_policy(board: Path) -> None:
    from plugins.kanban.dashboard.plugin_api import CreateTaskBody

    assert CreateTaskBody(title="legacy").review_policy is None

    with kbc.connect() as conn:
        before = _counts(conn)
        conflict = CreateTaskBody(
            title="x", assignee="worker", reviewer="reviewer", review_policy="disabled",
        )
        with pytest.raises(ValueError, match="conflicts with a reviewer"):
            kb.create_task(conn, created_by="dashboard", **conflict.model_dump())
        assert _counts(conn) == before

        ok = CreateTaskBody(title="solo", assignee="worker", review_policy="disabled")
        tid = kb.create_task(conn, created_by="dashboard", **ok.model_dump())
        assert kb.get_task(conn, tid).review_policy == "disabled"


@pytest.mark.parametrize("review_policy", ["maybe", "", "OPTIONAL"])
def test_dashboard_create_body_rejects_unknown_review_policy(
    review_policy: str,
) -> None:
    from plugins.kanban.dashboard.plugin_api import CreateTaskBody

    with pytest.raises(ValidationError, match="review_policy"):
        CreateTaskBody(title="invalid policy", review_policy=review_policy)
