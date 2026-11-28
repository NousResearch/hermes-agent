"""ADR-0020 §2.4 delivery parity — a chain break must reach the chat like a completion.

Cursor-level parity is pinned in ``tests/hermes_cli/test_kanban_replay_chainbreak.py``;
this is the end-to-end receipt: one notifier tick over a card that completed with
promised-but-uncreated successor work must actually deliver the CHAIN-BROKE ping
on the same route ``completed`` uses, without waking the agent for a kind that is
not in ``_WAKE_KINDS``.
"""
import asyncio

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn

Q8_SUMMARY = "preparing handoff successor — filing t_next tomorrow"


class RecordingAdapter:
    def __init__(self):
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})

    async def handle_message(self, event):
        self.handled.append(event)
        event._gateway_accepted = True


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _warn_phase(monkeypatch):
    """Shipped posture: chain-break warn phase, wake off (the estate default)."""
    def _cfg():
        return {
            "gate": True, "wake": False, "wake_require_idle_queue": True,
            "fresh_window_s": 21600, "wake_patterns": [], "routers": {},
            "chain_break": True, "intent_gate": False, "structural_break_check": False,
            "intent_patterns": None, "intent_denylist": None,
        }
    monkeypatch.setattr(kb, "_successor_guard_config", _cfg)


def test_chain_break_ping_is_delivered_like_a_completion(tmp_path, monkeypatch):
    db_path = tmp_path / "chain-break-notify.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb.init_db()
    _warn_phase(monkeypatch)

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="Q8 bisect", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        assert kb.complete_task(conn, tid, summary=Q8_SUMMARY) is True
        broke = [e for e in kb.list_events(conn, tid) if e.kind == "chain_broke"]
        assert len(broke) == 1
        triage_id = str((broke[0].payload or {})["triage_task"])
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    text = "\n".join(msg["text"] for msg in adapter.sent)
    assert "CHAIN BROKE" in text, f"no chain-break ping in {text!r}"
    assert tid in text and triage_id in text
    # The completion still notified on the same tick — parity, not replacement.
    assert "done" in text
