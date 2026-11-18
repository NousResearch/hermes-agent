"""Completion summaries survive the real Telegram sender with a fake Bot API."""
import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from plugins.platforms.telegram.adapter import TelegramAdapter


async def summary_flow(tmp_path, monkeypatch, summary):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "summary.db"))
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="Synthetic handoff", assignee="builder")
        claimed = kb.claim_task(conn, tid, claimer="test:builder")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="42", delivery_mode="notify+wake")
        assert kb.complete_task(conn, tid, summary=summary, expected_run_id=claimed.current_run_id)
        event = [e for e in kb.list_events(conn, tid) if e.kind == "completed"][0]
        assert event.payload["summary"] == summary
    finally:
        conn.close()
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic-token"))
    # Test legacy MarkdownV2 transport explicitly, not Bot API capability negotiation.
    adapter._rich_send_disabled = True
    adapter._bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=7)))
    adapter._retrigger_typing = AsyncMock()
    wakes = []

    async def handle(event):
        wakes.append(event.text)
        event._gateway_accepted = True

    adapter.handle_message = handle
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_dispatcher_lock_handle = object()
    ds = _notifier_collect(runner, kb, notifier_profile=None, gc_due=False, gc_retention_days=30)
    for d in ds:
        await _KanbanNotification(runner, d, platform_cls=Platform, sub_fail_counts={}).deliver()
    chunks = [c.kwargs["text"] for c in adapter._bot.send_message.await_args_list]
    assert chunks
    assert all(len(c.encode("utf-16-le")) // 2 <= adapter.MAX_MESSAGE_LENGTH for c in chunks)
    # Decode transport escaping and presentation-only continuation numbering.
    text = [re.sub(r"\\(.)", r"\1", c) for c in chunks]
    if len(text) > 1:
        text = [re.sub(r" \(\d+/\d+\)$", "", c) for c in text]
    reconstructed = " ".join(text).split("\n", 1)[1]
    assert reconstructed == summary
    assert len(wakes) == 1 and summary in wakes[0]
    assert not _notifier_collect(runner, kb, notifier_profile=None, gc_due=False, gc_retention_days=30)


@pytest.mark.parametrize("summary", [
    "Короткий результат 🐾.",
    "Проверено: " + "полный полезный текст 🐾 " * 30 + "\nВторая строка: сохранена полностью.",
    "Длинный результат 🐾 " * 700 + "Конец.",
])
def test_summary_reaches_bot_and_wake_losslessly(tmp_path, monkeypatch, summary):
    asyncio.run(summary_flow(tmp_path, monkeypatch, summary))
