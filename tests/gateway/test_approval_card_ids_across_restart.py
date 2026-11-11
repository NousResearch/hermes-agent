"""An approval card still visible after a gateway restart must not answer the first new card.

Card ids ride in Telegram's ``callback_data``; a fresh adapter (a restart) used to number its cards from
1 again, so the old card's button carried the new card's id and resolved the new request.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

_SESSION = "agent:main:telegram:dm:12345"


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    adapter._bot = AsyncMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    adapter._app = MagicMock()
    return adapter


def _once_data(adapter: TelegramAdapter) -> str:
    (card_id,) = adapter._approval_state  # the id the card's buttons carry in callback_data
    return f"ea:once:{card_id}"


def _tap(data: str) -> MagicMock:
    query = AsyncMock(data=data)
    query.message = MagicMock(chat_id=12345, text_html="<b>Dangerous command</b>")
    query.from_user = MagicMock(first_name="Ana", id="12345")
    return MagicMock(callback_query=query)


@pytest.mark.asyncio
async def test_old_card_tapped_after_restart_does_not_resolve_the_new_card():
    before = _adapter()
    await before.send_exec_approval(chat_id="12345", command="ls", session_key=_SESSION)
    old_card = _once_data(before)

    after = _adapter()  # the gateway restarted: a new adapter, the old card still in the chat
    await after.send_exec_approval(chat_id="12345", command="rm -rf /srv", session_key=_SESSION)
    new_card = _once_data(after)
    assert old_card != new_card, "a restarted adapter reissued the old card's id"

    resolve = MagicMock(return_value=1)
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False), \
            patch("tools.approval.resolve_gateway_approval_card", resolve):
        await after._handle_callback_query(_tap(old_card), MagicMock())
        resolve.assert_not_called()
        await after._handle_callback_query(_tap(new_card), MagicMock())  # control: the new card answers
    assert resolve.call_count == 1 and resolve.call_args.args[:2] == (_SESSION, "once")
