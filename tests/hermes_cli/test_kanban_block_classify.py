"""block_task() classifies an untyped breaker block in place (#117363)."""
import os
import time
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


def _time_out_once(conn, tid: str) -> None:
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    started = int(time.time()) - 30
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (started, tid))
        conn.execute(
            "UPDATE task_runs SET started_at = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (started, tid),
        )
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: None)


def test_typed_block_classifies_untyped_breaker_block(kanban_home, monkeypatch):
    """Two timeouts trip the breaker untyped; the supervisor's typed block then
    attaches the kind without a status flap, a synthetic run, or lost evidence."""
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ceiling", assignee="worker", max_runtime_seconds=1)
        _time_out_once(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"
        _time_out_once(conn, tid)
        parked = kb.get_task(conn, tid)
        assert (parked.status, parked.block_kind, parked.block_recurrences) == ("blocked", None, 0)

        assert kb.block_task(conn, tid, reason="needs a human decision", kind="needs_input") is True

        after = kb.get_task(conn, tid)
        assert (after.status, after.block_kind, after.block_recurrences) == ("blocked", "needs_input", 1)
        assert after.consecutive_failures == parked.consecutive_failures
        assert after.last_failure_error == parked.last_failure_error
        assert after.current_run_id is None
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert (kinds.count("blocked"), kinds.count("gave_up"), kinds.count("timed_out")) == (1, 1, 2)
        assert len(kb.list_runs(conn, tid)) == 2


def test_typed_block_still_refuses_typed_or_live_blocked_cards(kanban_home):
    with kbc.connect() as conn:
        typed = kb.create_task(conn, title="typed once", assignee="worker")
        kb.claim_task(conn, typed)
        assert kb.block_task(conn, typed, reason="decision", kind="needs_input") is True
        assert kb.block_task(conn, typed, reason="again", kind="capability") is False
        assert kb.get_task(conn, typed).block_kind == "needs_input"

        live = kb.create_task(conn, title="live run", assignee="worker")
        kb.claim_task(conn, live)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (live,))
        assert kb.block_task(conn, live, reason="race", kind="needs_input") is False
        assert kb.block_task(conn, live, reason="no kind") is False
        assert kb.get_task(conn, live).block_kind is None
