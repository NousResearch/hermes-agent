"""#147616: a Telegram send accepted with no ``message_id`` is not a delivery.

``sendRichMessage`` answered without an id used to come back as ``success=True``; the delivery ledger
finalizes on ``success`` and prunes a delivered row, so a reply that never reached the chat was never
redelivered. The delivery is unknown, so it must stay recoverable AND must not be resent on the spot
(a blind resend would duplicate a message Telegram may already hold).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_agent.gateway import delivery_ledger as dl
from hermes_agent.gateway.config import Platform, PlatformConfig
from hermes_agent.gateway.platforms.event import MessageEvent, MessageType
from hermes_agent.gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter

pytest.importorskip("telegram")

TABLE_REPLY = "Results:\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"


@pytest.mark.asyncio
async def test_rich_send_without_message_id_stays_recoverable_and_is_sent_once(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token", extra={"rich_messages": True}))
    bot = MagicMock()
    bot.do_api_request = AsyncMock(return_value={})  # Telegram accepted the call; the result has no id
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=7))
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, message_id="in-1",
                         source=SessionSource(platform=Platform.TELEGRAM, chat_id="111", chat_type="dm"))

    result, _ = await adapter.send_final_ledgered(event, "agent:main:telegram:dm:111", TABLE_REPLY, {}, reply_to=None)

    assert result.success is False, "a send with no message id is no proof of delivery"
    with dl._connect() as conn:
        states = [row[0] for row in conn.execute("SELECT state FROM delivery_obligations")]
    assert states == ["failed"], "the row must stay for redelivery, not be finalized as delivered and pruned"
    assert bot.do_api_request.await_count == 1 and bot.send_message.await_count == 0, (
        "delivery is unknown: neither a retry nor a legacy resend may duplicate it")
