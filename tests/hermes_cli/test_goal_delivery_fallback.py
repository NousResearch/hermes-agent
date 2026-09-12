"""Goal-judge delivery fallback (#100954): a judge *transport* failure on a
terminal handoff is infra, not content. ``kanban_complete`` may hand verification
to live downstream children owned by another profile; everything else stays
fail-closed (content rejections, review handoffs, no-eligible-child cases)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


# ---------------------------------------------------------------------------
# eligible_delivery_children (DB layer)
# ---------------------------------------------------------------------------

@pytest.fixture
def board(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


def test_eligible_delivery_children_filters(board) -> None:
    conn = board
    parent = kb.create_task(conn, title="p", assignee="builder")
    live = kb.create_task(conn, title="qa-child", assignee="qa", parents=[parent])
    # A done child must have reached a completable state first; complete_task
    # requires satisfied parents, so complete it BEFORE linking the edge.
    done = kb.create_task(conn, title="done-child", assignee="reviewer",
                          initial_status="running")
    kb.claim_task(conn, done)
    assert kb.complete_task(conn, done, summary="done")
    kb.link_tasks(conn, parent, done)
    same = kb.create_task(conn, title="self-child", assignee="builder", parents=[parent])
    unassigned = kb.create_task(conn, title="orphan-child", parents=[parent])

    eligible = kb.eligible_delivery_children(conn, parent, "builder")
    assert live in eligible
    assert done not in eligible          # terminal: cannot carry verification
    assert same not in eligible          # same assignee: self-approval
    assert unassigned not in eligible    # no owner to run it


def test_eligible_delivery_children_no_children(board) -> None:
    conn = board
    parent = kb.create_task(conn, title="lonely", assignee="builder")
    assert kb.eligible_delivery_children(conn, parent, "builder") == []


def test_eligible_delivery_children_archived(board) -> None:
    conn = board
    parent = kb.create_task(conn, title="p", assignee="builder")
    child = kb.create_task(conn, title="c", assignee="qa", parents=[parent])
    kb.archive_task(conn, child)
    assert kb.eligible_delivery_children(conn, parent, "builder") == []


# ---------------------------------------------------------------------------
# Tool surface (kanban_complete / kanban_request_review gates)
# ---------------------------------------------------------------------------

def _make_goal_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *,
                      child_assignee: str | None = "qa") -> tuple[str, str | None]:
    """Isolated HERMES_HOME, claimed goal_mode parent (+ optional child)."""
    home = tmp_path / ".home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="ship X", assignee="test-worker",
                                body="Must achieve X.", goal_mode=True)
        kb.claim_task(conn, parent)
        child = None
        if child_assignee is not None:
            child = kb.create_task(conn, title="verify X", assignee=child_assignee,
                                   parents=[parent])
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", parent)
    return parent, child


def _stub_tool_judge(monkeypatch: pytest.MonkeyPatch, *, transport_failed: bool,
                     verdict: str = "continue", reason: str = "judge error: BadRequestError") -> None:
    monkeypatch.setattr(
        "tools.kanban_tools.judge_goal",
        lambda *a, **k: (verdict, reason, False, None, transport_failed))
    monkeypatch.setattr("tools.kanban_tools._goal_judge_available", lambda: True)


def test_tool_complete_transport_failure_with_child_releases(monkeypatch, tmp_path) -> None:
    from tools import kanban_tools as kt

    parent, child = _make_goal_worker(monkeypatch, tmp_path)
    _stub_tool_judge(monkeypatch, transport_failed=True)

    out = json.loads(kt._handle_complete({"summary": "X shipped, evidence attached."}))
    assert out.get("ok") is True
    assert "goal_delivery_fallback" in out
    assert child in out["goal_delivery_fallback"]

    conn = kbc.connect()
    try:
        task = kb.get_task(conn, parent)
        assert task.status == "done"
        run = kb.latest_run(conn, parent)
        stamp = (run.metadata or {})["goal_delivery_fallback"]
        assert stamp["trigger"] == "goal_judge_transport_failed"
        assert stamp["judge_error"] == "judge error: BadRequestError"
        assert stamp["released_children"] == [child]
        # The child was RELEASED for verification, not marked done by the fallback.
        assert kb.get_task(conn, child).status not in ("done", "archived")
    finally:
        conn.close()


def test_tool_complete_transport_failure_without_child_rejects(monkeypatch, tmp_path) -> None:
    from tools import kanban_tools as kt

    parent, _ = _make_goal_worker(monkeypatch, tmp_path, child_assignee=None)
    _stub_tool_judge(monkeypatch, transport_failed=True)

    out = json.loads(kt._handle_complete({"summary": "X shipped."}))
    assert "error" in out
    assert "transport failure" in out["error"]

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, parent).status == "running"
    finally:
        conn.close()


def test_tool_complete_same_assignee_child_is_not_a_fallback(monkeypatch, tmp_path) -> None:
    from tools import kanban_tools as kt

    parent, _ = _make_goal_worker(monkeypatch, tmp_path, child_assignee="test-worker")
    _stub_tool_judge(monkeypatch, transport_failed=True)

    out = json.loads(kt._handle_complete({"summary": "X shipped."}))
    assert "error" in out and "transport failure" in out["error"]
    conn = kbc.connect()
    try:
        assert kb.get_task(conn, parent).status == "running"
    finally:
        conn.close()


def test_tool_complete_content_rejection_is_not_masked(monkeypatch, tmp_path) -> None:
    """A judge that answers ("continue", transport ok) rejects even with an
    eligible child — the fallback routes around infra, never around verdicts."""
    from tools import kanban_tools as kt

    parent, child = _make_goal_worker(monkeypatch, tmp_path)
    _stub_tool_judge(monkeypatch, transport_failed=False, verdict="continue",
                     reason="missing verification evidence")

    out = json.loads(kt._handle_complete({"summary": "I did some stuff."}))
    assert "error" in out
    assert "rejected by judge" in out["error"]
    assert "missing verification evidence" in out["error"]
    assert f"parents=[{parent}]" in out["error"]
    conn = kbc.connect()
    try:
        assert kb.get_task(conn, parent).status == "running"
    finally:
        conn.close()


def test_tool_request_review_transport_failure_stays_closed(monkeypatch, tmp_path) -> None:
    """The fallback exists only on ``kanban_complete``; a review handoff on a
    dead judge rejects (the review lane would bypass verification entirely)."""
    from tools import kanban_tools as kt

    _make_goal_worker(monkeypatch, tmp_path)
    _stub_tool_judge(monkeypatch, transport_failed=True)

    out = json.loads(kt._handle_request_review({"summary": "Looks ready."}))
    assert "error" in out
    assert "rejected" in out["error"]


# ---------------------------------------------------------------------------
# CLI surface (hermes kanban complete / request-review)
# ---------------------------------------------------------------------------

def _stub_cli_judge(monkeypatch: pytest.MonkeyPatch, *, transport_failed: bool,
                    reason: str = "judge error: PermissionDenied") -> None:
    import agent.auxiliary_client as auxiliary_client
    from hermes_cli import goals

    monkeypatch.setattr(auxiliary_client, "get_text_auxiliary_client",
                        lambda purpose: (object(), "judge-model"))
    monkeypatch.setattr(
        goals, "judge_goal",
        lambda *a, **k: ("continue", reason, False, None, transport_failed))


def test_cli_complete_transport_failure_with_child_releases(monkeypatch, tmp_path) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="ship Y", assignee="builder",
                                body="Must achieve Y.", goal_mode=True)
        kb.claim_task(conn, parent)
        child = kb.create_task(conn, title="verify Y", assignee="qa", parents=[parent])
    _stub_cli_judge(monkeypatch, transport_failed=True)

    output = kc.run_slash(f"complete {parent} --summary 'Y shipped'")
    assert "Completed" in output

    with kbc.connect() as conn:
        assert kb.get_task(conn, parent).status == "done"
        run = kb.latest_run(conn, parent)
        stamp = (run.metadata or {})["goal_delivery_fallback"]
        assert stamp["trigger"] == "goal_judge_transport_failed"
        assert stamp["released_children"] == [child]
        assert kb.get_task(conn, child).status not in ("done", "archived")


def test_cli_complete_transport_failure_without_child_rejects(monkeypatch, tmp_path) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="ship Z", assignee="builder",
                                body="Must achieve Z.", goal_mode=True)
        kb.claim_task(conn, parent)
    _stub_cli_judge(monkeypatch, transport_failed=True)

    output = kc.run_slash(f"complete {parent} --summary 'Z shipped'")
    assert "rejected" in output
    assert "transport failure" in output

    with kbc.connect() as conn:
        assert kb.get_task(conn, parent).status == "running"


def test_cli_complete_content_rejection_is_not_masked(monkeypatch, tmp_path) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="ship W", assignee="builder",
                                body="Must achieve W.", goal_mode=True)
        kb.claim_task(conn, parent)
        kb.create_task(conn, title="verify W", assignee="qa", parents=[parent])
    _stub_cli_judge(monkeypatch, transport_failed=False, reason="tests are missing")

    output = kc.run_slash(f"complete {parent} --summary 'W shipped'")
    assert "rejected by judge" in output
    assert "tests are missing" in output

    with kbc.connect() as conn:
        assert kb.get_task(conn, parent).status == "running"


# ---------------------------------------------------------------------------
# Fail-closed invariants the fallback must NOT weaken
# ---------------------------------------------------------------------------

def test_tool_fallback_pr_contract_still_enforced(monkeypatch, tmp_path) -> None:
    """A PR-contract task with a dead judge + eligible child still cannot
    complete without exact-head acceptance evidence — the fallback routes the
    decision around the judge, never around the PR contract."""
    from tools import kanban_tools as kt

    home = tmp_path / ".home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="ship PR", assignee="test-worker",
                                body="Must publish.", goal_mode=True,
                                completion_contract="acme/repo",
                                initial_status="running")
        kb.claim_task(conn, parent)
        kb.create_task(conn, title="verify PR", assignee="qa", parents=[parent])
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", parent)
    _stub_tool_judge(monkeypatch, transport_failed=True)

    # No published_pr / no green CI: acceptance fails, card stays in-flight.
    out = json.loads(kt._handle_complete({"summary": "PR ready."}))
    assert "error" in out
    conn = kbc.connect()
    try:
        assert kb.get_task(conn, parent).status != "done"
    finally:
        conn.close()


def test_tool_fallback_cas_mismatch_rejected(monkeypatch, tmp_path) -> None:
    """A stale/replaced run (CAS via expected_run_id) still rejects on the
    fallback path — a reclaimed run cannot complete the card."""
    from tools import kanban_tools as kt

    parent, _child = _make_goal_worker(monkeypatch, tmp_path)
    # Worker env pins run 999999 while the DB carries the real claimed run id.
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "999999")
    _stub_tool_judge(monkeypatch, transport_failed=True)

    out = json.loads(kt._handle_complete({"summary": "X shipped."}))
    assert "error" in out
    conn = kbc.connect()
    try:
        assert kb.get_task(conn, parent).status == "running"
    finally:
        conn.close()


def test_tool_fallback_stamp_overrides_caller_supplied(monkeypatch, tmp_path) -> None:
    """The audit stamp is system-owned: a model-supplied goal_delivery_fallback
    key is overwritten by the real trigger/judge_error/children, never trusted."""
    from tools import kanban_tools as kt

    parent, child = _make_goal_worker(monkeypatch, tmp_path)
    _stub_tool_judge(monkeypatch, transport_failed=True,
                     reason="judge error: RealReason")

    out = json.loads(kt._handle_complete({
        "summary": "X shipped.",
        "metadata": {"goal_delivery_fallback": {"trigger": "forged"}, "own": "kept"}}))
    assert out.get("ok") is True
    conn = kbc.connect()
    try:
        run = kb.latest_run(conn, parent)
        stamp = (run.metadata or {})["goal_delivery_fallback"]
        assert stamp["trigger"] == "goal_judge_transport_failed"
        assert stamp["judge_error"] == "judge error: RealReason"
        assert stamp["released_children"] == [child]
        assert (run.metadata or {})["own"] == "kept"
    finally:
        conn.close()
