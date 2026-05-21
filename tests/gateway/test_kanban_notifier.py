import asyncio
import json
from pathlib import Path

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
        return SendResult(success=True, message_id=f"msg-{len(self.sent)}")


class DisconnectedAdapters(dict):
    """Expose a platform during collection, then simulate disconnect on get()."""

    def get(self, key, default=None):
        return None


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    return runner


def _create_completed_subscription(summary="done once"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify once", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary=summary)
        return tid
    finally:
        conn.close()


def _unseen_terminal_events(tid):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_dedupes_board_slugs_pointing_to_same_db(tmp_path, monkeypatch):
    db_path = tmp_path / "shared-kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    kb.write_board_metadata("alias-a", name="Alias A")
    kb.write_board_metadata("alias-b", name="Alias B")

    tid = _create_completed_subscription()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 2
    assert "Kanban 已派单" in adapter.sent[0]["text"]
    assert "Issue/任务完成" in adapter.sent[1]["text"]
    assert all(tid in item["text"] for item in adapter.sent)


def test_kanban_notifier_claim_prevents_second_watcher_send(tmp_path, monkeypatch):
    db_path = tmp_path / "single-owner.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    tid = _create_completed_subscription()

    adapter1 = RecordingAdapter()
    adapter2 = RecordingAdapter()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter1)))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter2)))

    assert len(adapter1.sent) == 2
    assert adapter2.sent == []


def test_kanban_notifier_rewinds_claim_if_adapter_disconnects(tmp_path, monkeypatch):
    db_path = tmp_path / "adapter-disconnect.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = DisconnectedAdapters({Platform.TELEGRAM: RecordingAdapter()})
    runner._kanban_sub_fail_counts = {}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_kanban_db_path_is_test_isolated_from_real_home():
    hermes_home = Path(kb.kanban_home())
    production_db = Path.home() / ".hermes" / "kanban.db"
    assert kb.kanban_db_path().resolve() != production_db.resolve()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
    finally:
        conn.close()

    assert kb.kanban_db_path().resolve().is_relative_to(hermes_home.resolve())
    assert kb.kanban_db_path().resolve() != production_db.resolve()


class FailingAdapter:
    """Adapter whose send() always raises, simulating a transient send error."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        raise RuntimeError("simulated send failure")


class FailedResultAdapter:
    """Adapter whose send() returns an explicit non-delivery result."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        return SendResult(success=False, error="simulated non-delivery")


def test_kanban_notifier_rewinds_claim_on_send_exception(tmp_path, monkeypatch):
    """A raising adapter rewinds the claim so the next tick can retry.

    This is the second rewind path (distinct from the adapter-disconnect path
    in test_kanban_notifier_rewinds_claim_if_adapter_disconnects). Here the
    adapter is connected and the send call actually fires; the claim must
    still rewind so the event isn't lost when send() raises mid-tick.
    """
    db_path = tmp_path / "send-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = FailingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Send was attempted (so we exercised the failure path, not just the
    # disconnect path) and the claim was rewound — the unseen-events query
    # still returns the event for retry on the next tick.
    assert adapter.attempts >= 1, "send should have been attempted at least once"
    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_kanban_notifier_treats_failed_send_result_as_failed_delivery(tmp_path, monkeypatch):
    """SendResult(success=False) must not ledger or advance as delivered."""
    db_path = tmp_path / "failed-result.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = FailedResultAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.attempts >= 1, "send should have been attempted at least once"
    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]

    conn = kb.connect()
    try:
        evidence_rows = conn.execute(
            "SELECT 1 FROM task_events "
            "WHERE task_id = ? AND kind = 'notify_delivery_evidence'",
            (tid,),
        ).fetchall()
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()

    assert evidence_rows == []
    assert len(subs) == 1, "terminal subscription must not be removed on failed send"
    assert runner._kanban_sub_fail_counts[(tid, "telegram", "chat-1", "")] == 1


def test_notifier_redelivers_same_kind_on_dispatch_cycle(tmp_path, monkeypatch):
    """A retry cycle (crashed → reclaimed → crashed) notifies the user twice.

    Before #21398 the notifier auto-unsubscribed on any terminal event kind
    (gave_up / crashed / timed_out), so the second crash in a respawn cycle
    silently dropped — the subscription was already gone. This test pins the
    new contract: subscription survives non-final terminal events; the
    cursor handles dedup.

    Two crashes ten seconds apart on the same task — both should land on
    the adapter.
    """
    db_path = tmp_path / "redeliver-cycle.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="cycle test", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        # First crash — fired by the dispatcher when the worker PID dies.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Initial assignment and first crash delivered.
    assert len(adapter.sent) == 2
    assert "Kanban 已派单" in adapter.sent[0]["text"]
    assert "崩溃" in adapter.sent[1]["text"]

    # Subscription survives — the cursor advanced past event #1, but the
    # row is still there.
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1, (
            "Subscription must survive a crashed event so a respawn-cycle "
            "second crash also notifies the user (issue #21398)."
        )

        # Second crash — same task, same dispatcher (or a respawn). Append
        # another event to simulate the dispatcher firing crashed a second
        # time during retry.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    # New tick: the second event has a fresh id past the cursor advance,
    # so it gets claimed and delivered.
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 3, (
        f"Second crashed event should also notify; got {len(adapter.sent)} "
        f"deliveries (texts: {[d['text'] for d in adapter.sent]})"
    )
    assert "崩溃" in adapter.sent[2]["text"]


def test_kanban_notifier_sends_runtime_lifecycle_receipts(tmp_path, monkeypatch):
    db_path = tmp_path / "runtime-lifecycle.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="runtime receipts", assignee="coder")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.assign_task(conn, tid, "reviewer")
        kb._append_event(conn, tid, "spawned", {"pid": 12345})
        kb._append_event(conn, tid, "heartbeat", {"note": "halfway done"})
        kb.complete_task(conn, tid, summary="finished handoff")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    texts = [item["text"] for item in adapter.sent]
    assert [text.splitlines()[0] for text in texts] == [
        "📌 Kanban 已派单",
        "📌 Kanban 已分配",
        "▶️ Runtime 已开始",
        "🔄 Runtime 进展",
        "✅ Issue/任务完成",
    ]
    assert "@reviewer" in texts[-1]
    assert "halfway done" in texts[3]

    conn = kb.connect()
    try:
        rows = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'notify_delivery_evidence' "
            "ORDER BY id ASC",
            (tid,),
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 5
    payloads = [json.loads(row["payload"]) for row in rows]
    assert payloads[-1]["work_item_id"] == tid
    assert payloads[-1]["task_id"] == tid
    assert payloads[-1]["runtime_profile"] == "reviewer"
    assert payloads[-1]["assignee"] == "reviewer"
    assert payloads[-1]["notifier_profile"] == "default"
    assert payloads[-1]["chat_id"] == "chat-1"
    assert payloads[-1]["thread_id"] == ""
    assert payloads[-1]["message_id"] == "msg-5"
    assert payloads[-1]["source_event_kind"] == "completed"
