from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.kanban_telegram_forum import render_task_card, sync_telegram_forum_tasks
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


@pytest.mark.asyncio
async def test_sync_creates_one_topic_subscription_and_pinned_card(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "forum.db"))
    kb.init_db()
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="Telegram control room", assignee="gustave")
    finally:
        conn.close()

    adapter = SimpleNamespace(
        create_handoff_thread=AsyncMock(return_value="812"),
        send=AsyncMock(return_value=SimpleNamespace(success=True, message_id="99")),
        pin_message=AsyncMock(),
    )
    runner = SimpleNamespace(_authorization_adapter=lambda platform, profile=None: adapter)
    config = {"kanban": {"telegram_forum": {
        "enabled": True,
        "chat_id": "-100123",
        "board": "default",
        "profile": "default",
        "profile_mentions": {"gustave": "@GustaveBot"},
    }}}

    await sync_telegram_forum_tasks(runner, kb, config, notifier_profile="default")
    await sync_telegram_forum_tasks(runner, kb, config, notifier_profile="default")

    adapter.create_handoff_thread.assert_awaited_once_with(
        "-100123", f"Telegram control room · {task_id}",
    )
    adapter.pin_message.assert_awaited_once_with("-100123", "99", thread_id="812")
    sent_text = adapter.send.await_args.args[1]
    assert sent_text.startswith("📌 Telegram control room\n")
    assert "@GustaveBot (gustave)" in sent_text
    conn = kbc.connect()
    try:
        subs = kbn.list_notify_subs(conn, task_id)
    finally:
        conn.close()
    assert len(subs) == 1
    assert subs[0]["thread_id"] == "812"
    assert subs[0]["delivery_metadata"]["kanban_task_message_id"] == "99"


def test_render_task_card_exposes_next_step_and_blocker():
    task = SimpleNamespace(
        id="t_1", title="Repair service", status="blocked", assignee="gustave",
        last_failure_error="Waiting for operator access",
    )
    text = render_task_card(task, {"gustave": "@GustaveBot"})
    assert "Prochaine étape : Lever le blocage indiqué." in text
    assert "Blocage : Waiting for operator access" in text
