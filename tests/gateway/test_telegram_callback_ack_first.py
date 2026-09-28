"""A button tap clears the spinner without hiding later visible feedback.

Telegram keeps only the first answerCallbackQuery. The spinner ack therefore
goes out first and must finish before the handler returns. Denial, expiry and
success text is written onto the message, because a second answer never shows.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import unauthorized_action_notice
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _query(data: str, user_id: str = "777"):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.chat.type = "private"
    query.message.text = "Pick"
    query.from_user = MagicMock()
    query.from_user.id = user_id
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return query


def _wire_ack_completion(query):
    """Record when the empty ack actually finishes."""
    done = []

    async def answer(*_args, **kwargs):
        done.append(kwargs.get("text"))
        return True

    query.answer = AsyncMock(side_effect=answer)
    return done


@pytest.mark.asyncio
async def test_empty_ack_completes_and_denial_stays_visible():
    """The spinner ack finishes, and the denial is still on the message."""
    adapter = _adapter()
    adapter._clarify_state["cidX"] = "skX"
    query = _query("cl:cidX:0", user_id="999")
    done = _wire_ack_completion(query)
    update = MagicMock()
    update.callback_query = query
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "777"}, clear=False):
        await adapter._handle_callback_query(update, MagicMock())

    notice = unauthorized_action_notice("telegram")
    assert done == [None]
    assert query.answer.await_count == 1
    edited = query.edit_message_text.await_args.kwargs["text"]
    assert notice in edited


@pytest.mark.asyncio
async def test_empty_ack_completes_and_success_stays_visible():
    """The spinner ack finishes, and the chosen answer is still on the message."""
    adapter = _adapter()
    adapter._clarify_state["cidOk"] = "skOk"
    query = _query("cl:cidOk:0", user_id="777")
    done = _wire_ack_completion(query)
    update = MagicMock()
    update.callback_query = query

    class _Entry:
        choices = ["green"]
        response = None
        event = MagicMock()

    entry = _Entry()
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False), patch(
        "tools.clarify_gateway._entries", {"cidOk": entry}
    ), patch("tools.clarify_gateway.resolve_gateway_clarify", return_value=True):
        await adapter._handle_callback_query(update, MagicMock())

    assert done == [None]
    edited = query.edit_message_text.await_args.kwargs["text"]
    assert "green" in edited


@pytest.mark.asyncio
async def test_spinner_ack_starts_before_authorization():
    """The spinner ack is created before the allowlist gate runs."""
    adapter = _adapter()
    order = []
    query = _query("update_prompt:y", user_id="222")
    query.answer = AsyncMock(side_effect=lambda *a, **k: order.append("answer"))
    started = adapter._start_callback_ack

    def _start(query):
        order.append("start")
        return started(query)

    adapter._start_callback_ack = _start
    real_authorized = adapter._is_callback_user_authorized

    def _gate(*args, **kwargs):
        order.append("auth")
        return real_authorized(*args, **kwargs)

    adapter._is_callback_user_authorized = _gate
    update = MagicMock()
    update.callback_query = query
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "111"}, clear=False):
        await adapter._handle_callback_query(update, MagicMock())

    assert order[0] == "start"
    assert order.index("start") < order.index("auth")
    assert "answer" in order
