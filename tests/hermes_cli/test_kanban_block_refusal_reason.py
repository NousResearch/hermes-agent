"""``block_task`` refusal reasons: the stale-worker case names its cause.

``kanban_block`` used to report every refusal as
``could not block <id> (unknown id or not in running/ready)``, so a worker
whose run had already ended (card back to ``ready``, ``current_run_id``
NULL) read the message as "the card vanished". These tests pin the
``with_reason`` contract that mirrors ``request_review``: each refusing
branch of :func:`block_task <hermes_cli.kanban_db.block_task>` returns a
specific reason, and the tool surface prints it (Issue #133358).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _release_to_pool(conn, tid: str) -> None:
    """The run ended and the dispatcher returned the card to the pool.

    Stand-in for the reclaim path: claim fields cleared, no run holds the
    card, status back to ``ready`` — exactly the state a stale worker
    process still in its turn sees (#133358 repro step 2).
    """
    conn.execute("UPDATE tasks SET status = 'ready', current_run_id = NULL WHERE id = ?", (tid,))
    conn.execute("UPDATE tasks SET claim_lock = NULL, claim_expires = NULL, worker_pid = NULL WHERE id = ?", (tid,))


def test_block_task_stale_run_reason_names_the_ended_run(kanban_home: Path) -> None:
    """The reported bug: refusal says 'stale run', not 'unknown id'."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stale worker", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = kb._current_run_id(conn, tid)
        _release_to_pool(conn, tid)

        ok, reason = kb.block_task(
            conn, tid, reason="needs input", kind="needs_input",
            expected_run_id=run_id, with_reason=True,
        )
        assert ok is False
        assert reason is not None and "stale run" in reason
        assert str(run_id) in reason          # this worker's ended run
        assert "none" in reason               # no run holds the card now
        assert "'ready'" in reason            # the card did not vanish
        # The card is untouched by the refused call.
        assert kb.get_task(conn, tid).status == "ready"


def test_block_task_unknown_task_reason(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        ok, reason = kb.block_task(conn, "t_deadbeefcafe", with_reason=True)
        assert ok is False
        assert reason == "task not found"


def test_block_task_status_refusal_reason(kanban_home: Path) -> None:
    """A card sitting in ``review`` refuses with its status, not 'unknown id'."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="in review", assignee="worker")
        assert kb.claim_task(conn, tid) is not None
        assert kb.request_review(conn, tid, summary="done") is True

        ok, reason = kb.block_task(conn, tid, reason="late block", with_reason=True)
        assert ok is False
        assert reason is not None and "review" in reason
        assert "not running/ready" in reason


def test_block_task_already_blocked_typed_reason(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="typed", assignee="worker")
        assert kb.block_task(conn, tid, reason="first", kind="capability") is True

        ok, reason = kb.block_task(
            conn, tid, reason="again", kind="capability", with_reason=True)
        assert ok is False
        assert reason is not None and "already 'blocked'" in reason
        assert "capability" in reason


def test_block_task_bool_mode_callers_unchanged(kanban_home: Path) -> None:
    """Without with_reason every path still returns a plain bool."""
    with kbc.connect() as conn:
        assert kb.block_task(conn, "t_deadbeefcafe") is False
        tid = kb.create_task(conn, title="fresh", assignee="worker")
        assert kb.block_task(conn, tid, reason="first") is True
        assert kb.get_task(conn, tid).status == "blocked"


def test_kanban_block_tool_reports_stale_run_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tool surface prints the DB reason instead of the generic guess."""
    from tools import kanban_tools as kt

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="tool stale", assignee="test-worker")
        assert kb.claim_task(conn, tid) is not None
        run_id = kb._current_run_id(conn, tid)
        _release_to_pool(conn, tid)
    finally:
        conn.close()
    # The stale worker process still carries its ended run id.
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    out = kt._handle_block({"task_id": tid, "reason": "needs input"})
    d = json.loads(out)
    assert "error" in d
    assert "stale run" in d["error"]
    assert "unknown id or not in running/ready" not in d["error"]
