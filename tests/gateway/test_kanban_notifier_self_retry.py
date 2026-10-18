"""A crash/timeout the dispatcher already re-queued needs nobody: no ping, no wake turn.

The escalation (``gave_up`` after the failure limit) still reaches the subscriber.
"""
import asyncio

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from tests.gateway.test_kanban_notifier import RecordingAdapter, _make_runner, _run_one_notifier_tick


def _sub_with_events(monkeypatch, tmp_path, mode, events):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "self-retry.db"))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="retry me", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1", delivery_mode=mode)
        for kind, payload in events:
            kb._append_event(conn, tid, kind=kind, payload=payload)
        return tid
    finally:
        conn.close()


def _cursor_drained(tid):
    conn = kbc.connect()
    try:
        _, unseen = kbn.unseen_events_for_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1",
                                              kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"])
        return unseen == []
    finally:
        conn.close()


def test_self_retried_crash_and_timeout_neither_ping_nor_wake(tmp_path, monkeypatch):
    tid = _sub_with_events(monkeypatch, tmp_path, "notify+wake", [
        ("crashed", {"exit_kind": "nonzero_exit", "retry_status": "ready"}),
        ("timed_out", {"limit_seconds": 5400, "retry_status": "ready"}),
    ])
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert adapter.sent == [] and adapter.handled == []
    assert _cursor_drained(tid), "silent events must still advance the cursor, never wedge it"


def test_escalation_after_retries_still_wakes(tmp_path, monkeypatch):
    _sub_with_events(monkeypatch, tmp_path, "wake", [
        ("crashed", {"retry_status": "ready"}),
        ("gave_up", {"failures": 2, "effective_limit": 2, "error": "pid exited 1"}),
    ])
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.handled) == 1
    assert "crashed" not in adapter.handled[0].text.split("\n")[0]


def test_needs_input_block_still_wakes(tmp_path, monkeypatch):
    _sub_with_events(monkeypatch, tmp_path, "wake", [("blocked", {"kind": "needs_input", "reason": "which key?"})])
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.handled) == 1
