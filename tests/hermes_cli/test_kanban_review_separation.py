"""Reviewer/implementer separation guards.

* ``request_review`` refuses a reviewer who is the implementer of the card:
  self-review is no review.
* ``claim_task`` refuses to start a profile listed in
  ``kanban.review_only_profiles`` on a ``ready`` card and parks the card
  ``blocked`` for a human to reassign; ``claim_review_task`` (the review lane)
  still starts it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def review_only(monkeypatch: pytest.MonkeyPatch):
    import hermes_cli.config as cfgmod

    def _set(profiles):
        monkeypatch.setattr(
            cfgmod, "load_config",
            lambda *a, **k: {"kanban": {"review_only_profiles": profiles}},
        )
    return _set


def _events(conn, tid, kind):
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (tid, kind),
    ).fetchall()
    return [json.loads(r["payload"]) if r["payload"] else None for r in rows]


# ---------------------------------------------------------------------------
# request_review: reviewer must not be the implementer
# ---------------------------------------------------------------------------


def test_request_review_refuses_implementer_as_reviewer(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="code-monkey")
        claimed = kb.claim_task(conn, tid)
        ok, reason = kb.request_review(
            conn, tid, summary="done", reviewer="code-monkey",
            expected_run_id=claimed.current_run_id, with_reason=True,
        )
        assert ok is False
        assert "cannot review its own work" in reason
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.assignee == "code-monkey"
        assert _events(conn, tid, "review_requested") == []


def test_request_review_refuses_implementer_case_insensitively(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="code-monkey")
        claimed = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="done", reviewer="Code-Monkey",
            expected_run_id=claimed.current_run_id,
        ) is False


def test_request_review_refuses_self_review_after_own_run(kanban_home: Path) -> None:
    """The live run names the implementer; naming that profile as reviewer is refused."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        # assignee == reviewer here is the existing "assigned to its reviewer"
        # case (no implementer recorded) and stays allowed; the guard is about
        # a recorded implementer, which a live run provides.
        claimed = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="x", reviewer="worker",
            expected_run_id=claimed.current_run_id,
        ) is False


def test_request_review_with_distinct_reviewer_still_works(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="code-monkey")
        claimed = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="done", reviewer="principal-engineer",
            expected_run_id=claimed.current_run_id,
        ) is True
        assert kb.get_task(conn, tid).assignee == "principal-engineer"
        ev = _events(conn, tid, "review_requested")[0]
        assert (ev["implementer"], ev["reviewer"]) == ("code-monkey", "principal-engineer")


def test_re_review_defaulting_to_prior_reviewer_is_allowed(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="code-monkey")
        claimed = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="v1", reviewer="principal-engineer",
            expected_run_id=claimed.current_run_id,
        ) is True
        rv = kb.claim_review_task(conn, tid, claimer="principal-engineer")
        ok, _ = kb.request_changes(conn, tid, reason="fix it", expected_run_id=rv.current_run_id)
        assert ok is True
        assert kb.get_task(conn, tid).assignee == "code-monkey"
        claimed = kb.claim_task(conn, tid)
        # reviewer omitted -> prior reviewer (principal-engineer), not the implementer
        assert kb.request_review(
            conn, tid, summary="v2", expected_run_id=claimed.current_run_id,
        ) is True
        assert kb.get_task(conn, tid).assignee == "principal-engineer"


# ---------------------------------------------------------------------------
# claim_task: review-only profiles never implement
# ---------------------------------------------------------------------------


def test_claim_task_refuses_review_only_profile_and_parks_blocked(
    kanban_home: Path, review_only,
) -> None:
    review_only(["principal-engineer"])
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="code me", assignee="principal-engineer")
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.claim_task(conn, tid) is None
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.current_run_id is None
        assert _events(conn, tid, "claim_rejected")[-1]["reason"] == "review_only_profile"
        assert "review-only" in _events(conn, tid, "blocked")[-1]["reason"]


def test_claim_task_review_only_matching_is_case_insensitive(
    kanban_home: Path, review_only,
) -> None:
    review_only("Principal-Engineer")  # a bare string is accepted too
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="code me", assignee="principal-engineer")
        assert kb.claim_task(conn, tid) is None
        assert kb.get_task(conn, tid).status == "blocked"


def test_claim_task_other_profiles_unaffected(kanban_home: Path, review_only) -> None:
    review_only(["principal-engineer"])
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="code me", assignee="code-monkey")
        assert kb.claim_task(conn, tid) is not None
        assert kb.get_task(conn, tid).status == "running"


def test_claim_task_guard_off_by_default(kanban_home: Path, review_only) -> None:
    review_only([])
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="code me", assignee="principal-engineer")
        assert kb.claim_task(conn, tid) is not None


def test_review_only_profile_still_claims_review_lane(
    kanban_home: Path, review_only,
) -> None:
    review_only(["principal-engineer"])
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="code-monkey")
        claimed = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="done", reviewer="principal-engineer",
            expected_run_id=claimed.current_run_id,
        ) is True
        rv = kb.claim_review_task(conn, tid, claimer="principal-engineer")
        assert rv is not None and rv.status == "running"


def test_review_only_profiles_survives_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.config as cfgmod

    def _boom(*a, **k):
        raise RuntimeError("bad yaml")
    monkeypatch.setattr(cfgmod, "load_config", _boom)
    assert kb.review_only_profiles() == frozenset()


def test_dispatcher_does_not_spawn_review_only_profile_on_ready_card(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_only_profiles": ["principal-engineer"]}},
    )
    spawned = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="code me", assignee="principal-engineer")
        res = kbd.dispatch_once(
            conn, spawn_fn=lambda *a, **k: spawned.append(a) or 4242,
        )
        assert tid not in [s[0] for s in res.spawned]
        assert spawned == []
        assert kb.get_task(conn, tid).status == "blocked"
