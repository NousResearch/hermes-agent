"""Replay real collector/delivery with durable subscriptions and a fake Telegram transport."""
import asyncio
from types import SimpleNamespace

from gateway.config import Platform
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn


class Transport:
    def __init__(self):
        self.sent = []
        self.edits = []
        self.error = None
        self.send_error = False
        self.documents = []

    async def send_document(self, chat, path, **kwargs):
        from pathlib import Path
        self.documents.append(Path(path).read_text())
        return SendResult(success=True, message_id="doc")

    async def send(self, chat, text, metadata=None):
        self.sent.append((chat, text, metadata))
        if self.send_error:
            raise TimeoutError("ambiguous timeout")
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def edit_message(self, chat, message_id, text, **kwargs):
        self.edits.append((chat, message_id, text))
        return SendResult(success=not self.error, message_id=message_id, error=self.error)


def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    monkeypatch.setattr("gateway.warning_notifications.present_notification", present)
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Retry fixture", assignee="worker")
        conn.execute("UPDATE tasks SET id='t_5c0f4f61' WHERE id=?", (tid,))
        kbn.add_notify_sub(conn, task_id="t_5c0f4f61", platform="telegram", chat_id="chat",
                           delivery_metadata={"thread_id": "42", "approval_marker": "preserve"})
    return "t_5c0f4f61"


async def present(send, **kwargs):
    await send()
    return True


def runner(adapter):
    r = GatewayRunner.__new__(GatewayRunner)
    r.adapters = {Platform.TELEGRAM: adapter}
    r._authorization_adapter = lambda *args: adapter
    r._owns_kanban_dispatcher_lock = lambda: True
    r._kanban_sub_fail_counts = {}
    return r


def tick(r):
    batches = _notifier_collect(r, kb, notifier_profile=None, gc_due=False, gc_retention_days=30)
    for batch in batches:
        asyncio.run(_KanbanNotification(r, batch, platform_cls=Platform,
                                       sub_fail_counts=r._kanban_sub_fail_counts).deliver())


def emit(tid, kind, payload=None, status="running"):
    with kbc.connect() as conn:
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, tid))
        kb._append_event(conn, tid, kind=kind, payload=payload or {})


def test_retry_recovery_done_one_send_then_edits_survives_runner_restart(monkeypatch, tmp_path):
    tid = setup(monkeypatch, tmp_path)
    adapter = Transport()
    sequence = [("crashed", {}, "running"), ("status", {"status": "ready"}, "ready"),
                ("gave_up", {"error": "pid: 123\nsession: secret\nlast_output: noisy"}, "blocked"),
                ("unblocked", {}, "ready"),
                ("completed", {"summary": "Fixed the notifier. Preserved existing pending approvals.",
                               "metadata": {"completed_work": "Native task status delivery",
                                            "tests_run": "Focused replay passed",
                                            "verification_limits": "Live Telegram delivery not exercised."}}, "done")]
    for kind, payload, status in sequence:
        emit(tid, kind, payload, status)
        tick(runner(adapter))
    assert len(adapter.sent) == 1
    assert len(adapter.edits) == 4
    final = adapter.edits[-1][2]
    assert final.startswith("Fixed the notifier. Preserved existing pending approvals.")
    assert "Done" in final and "Live Telegram delivery not exercised." in final
    assert "Decision needed" not in final
    assert "success not yet verified" in adapter.edits[-2][2]
    assert all("last_output" not in text and "pid:" not in text for _, _, text in adapter.edits)
    tick(runner(adapter))
    assert len(adapter.edits) == 4
    with kbc.connect() as conn:
        sub = kbn.list_notify_subs(conn, tid)[0]
        assert sub["delivery_metadata"]["task_status_message_id"] == "1"
        assert sub["delivery_metadata"]["approval_marker"] == "preserve"


def test_edit_failure_backoff_and_missing_message_fallback(monkeypatch, tmp_path):
    tid = setup(monkeypatch, tmp_path)
    a = Transport()
    emit(tid, "crashed")
    tick(runner(a))
    a.error = "temporary transport timeout"
    emit(tid, "gave_up", status="blocked")
    for _ in range(20):
        tick(runner(a))
    assert len(a.sent) == 1 and len(a.edits) == 1
    with kbc.connect() as conn:
        kbn.record_notify_status(conn, task_id=tid, platform="telegram", chat_id="chat",
                                 state={"task_status_retry_at": 0})
    a.error = "Bad Request: message to edit not found"
    tick(runner(a))
    assert len(a.sent) == 2
    for _ in range(20):
        tick(runner(a))
    assert len(a.sent) == 2


def test_ambiguous_replacement_does_not_send_storm(monkeypatch, tmp_path):
    tid = setup(monkeypatch, tmp_path)
    a = Transport()
    emit(tid, "crashed")
    tick(runner(a))
    a.error = "message to edit not found"
    a.send_error = True
    async def wake(self):
        return None
    monkeypatch.setattr(_KanbanNotification, "wake", wake)
    emit(tid, "completed", {"summary": "Completed fixture."}, "done")
    for _ in range(15):
        tick(runner(a))
    assert len(a.sent) == 2 and len(a.edits) == 1
    with kbc.connect() as conn:
        assert kbn.list_notify_subs(conn, tid)[0]["delivery_metadata"]["task_status_send_pending"]


def test_real_completion_hydrates_unclipped_summary_and_verification(monkeypatch, tmp_path):
    tid = setup(monkeypatch, tmp_path)
    summary = "First sentence. " * 30 + "The final outcome remains complete."
    with kbc.connect() as conn:
        assert kb.complete_task(conn, tid, summary=summary,
                                metadata={"verification_limits": "Hardware was not exercised."})
    a = Transport()
    tick(runner(a))
    assert a.sent[0][1].startswith(summary)
    assert "Hardware was not exercised." in a.sent[0][1]


def test_complete_sentences_and_no_false_done():
    from gateway.kanban_task_status import render_status
    n = SimpleNamespace(task_id="fixture", title="Fixture", task=SimpleNamespace(status="ready", result=""), d={})
    ev = SimpleNamespace(id=1, kind="completed", payload={"summary": "An entire sentence. " + "x" * 3000})
    text = render_status(n, ev)
    assert text.startswith("An entire sentence.")
    assert "task is now ready" in text
    assert len(text) < 4096


def test_real_telegram_adapter_overflow_and_markdown_stay_one_bubble(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter
    tid = setup(monkeypatch, tmp_path)
    # Real send/edit/format/document methods, only Bot network I/O is mocked.
    a = TelegramAdapter(PlatformConfig(enabled=True, token="123:fixture"))
    a._rich_messages_enabled = False
    a._telegram_chat_outbound_slot_secs = 0
    a._bot = AsyncMock()
    a._bot.send_message.return_value = SimpleNamespace(message_id=17)
    a._bot.send_document.return_value = SimpleNamespace(message_id=18)
    documents = []
    async def document(**kwargs):
        documents.append(kwargs["document"].read().decode())
        return SimpleNamespace(message_id=18)
    a._bot.send_document.side_effect = document
    emit(tid, "crashed")
    tick(runner(a))
    for kind, payload, status in [("status", {"status": "ready"}, "ready"),
                                   ("gave_up", {}, "blocked"), ("unblocked", {}, "ready")]:
        emit(tid, kind, payload, status)
        tick(runner(a))
    summary = "x" * 3000
    limits = "No live verification. " + "😀" * 3000
    payload = {"summary": summary, "metadata": {"verification_limits": limits,
               "completed_work": "_*[]().!" * 700, "tests_run": "😀" * 1000}}
    emit(tid, "completed", payload, "done")
    tick(runner(a))
    # Same overflowing handoff on a later event must not upload the details again.
    emit(tid, "completed", payload, "done")
    tick(runner(a))
    assert a._bot.send_message.await_count == 1
    assert a._bot.edit_message_text.await_count == 5
    assert len(documents) == 1
    assert summary in documents[0] and limits in documents[0]
    for call in a._bot.edit_message_text.call_args_list[-2:]:
        text = call.kwargs["text"]
        assert len(text.encode("utf-16-le")) // 2 < 4096
        assert "Done" in text and "Verification limits" in text
        assert "details attached" in text
    assert a._bot.edit_message_text.call_args.kwargs["text"].startswith("Done")


def test_ambiguous_initial_send_preserves_decision_wake_and_reconciles(monkeypatch, tmp_path, caplog):
    tid = setup(monkeypatch, tmp_path)
    a = Transport()
    a.send_error = True
    wakes = []
    async def wake(self):
        wakes.append(self.synth)
    monkeypatch.setattr(_KanbanNotification, "wake", wake)
    emit(tid, "crashed")
    for _ in range(15):
        tick(runner(a))
    assert len(a.sent) == 1 and len(wakes) == 1
    emit(tid, "blocked", {"reason": "Approve deleting the obsolete volume?", "kind": "needs_input"}, "blocked")
    for _ in range(15):
        tick(runner(a))
    assert len(a.sent) == 1 and len(wakes) == 2
    assert "Approve deleting the obsolete volume?" in wakes[-1]
    assert "reconcile_notify_status" in wakes[-1]
    with kbc.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_comments WHERE author='kanban-notifier'").fetchone()[0] == 2
        sub = kbn.list_notify_subs(conn, tid)[0]
        assert sub["delivery_metadata"]["approval_marker"] == "preserve"
        kbn.reconcile_notify_status(conn, task_id=tid, platform="telegram", chat_id="chat", message_id="99")
    a.send_error = False
    tick(runner(a))
    assert len(a.sent) == 1
    assert a.edits[-1][1] == "99"
    assert "Approve deleting" in a.edits[-1][2]
    assert "ambiguous send requires reconciliation" not in caplog.text


def test_uncertain_send_failed_wake_retries_admission_not_transport(monkeypatch, tmp_path):
    from gateway.wake import WakeNotAccepted
    tid = setup(monkeypatch, tmp_path)
    a = Transport()
    async def uncertain(*args, **kwargs):
        a.sent.append(args)
        return SendResult(success=False, retryable=False, error="read timeout")
    a.send = uncertain
    attempts = []
    async def wake(self):
        attempts.append(self.synth)
        if len(attempts) == 1:
            raise WakeNotAccepted("queue busy")
    monkeypatch.setattr(_KanbanNotification, "wake", wake)
    emit(tid, "blocked", {"kind": "needs_input", "reason": "Approve removal?"}, "blocked")
    for _ in range(12):
        tick(runner(a))
    assert len(a.sent) == 1 and len(attempts) == 2
    assert "Approve removal?" in attempts[-1]
    with kbc.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_comments WHERE author='kanban-notifier'").fetchone()[0] == 1
        kbn.reconcile_notify_status(conn, task_id=tid, platform="telegram", chat_id="chat", confirmed_absent=True)
    a.send = Transport.send.__get__(a)
    tick(runner(a))
    assert len(a.sent) == 2
