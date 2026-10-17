"""Telegram ``retire_clarify_card``: a clarify that ends without a click retires the card.

Mirrors the Slack contract (plugins/platforms/slack/adapter.py ``retire_clarify_card``):
on timeout / cancellation the gateway calls ``adapter.retire_clarify_card(clarify_id,
notice)`` and the card must drop its inline keyboard (``reply_markup=None``) with the
notice html-escaped. Unknown ids are a no-op; repeat calls are idempotent.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


async def _seed_card(adapter, clarify_id="cid1", chat_id=12345, question="Pick one?"):
    """Deliver a clarify via the real send_clarify so _clarify_messages fills organically."""
    msg = MagicMock()
    msg.message_id = 100
    adapter._bot.send_message = AsyncMock(return_value=msg)
    result = await adapter.send_clarify(
        chat_id=str(chat_id), question=question, choices=["a", "b"],
        clarify_id=clarify_id, session_key="sk1")
    assert result.success is True
    return result


@pytest.mark.asyncio
async def test_retire_edits_message_without_markup_and_escapes_notice():
    adapter = _make_adapter()
    await _seed_card(adapter, clarify_id="cid1")
    assert "cid1" in adapter._clarify_messages

    await adapter.retire_clarify_card("cid1", "<b>expired</b>")

    kwargs = adapter._bot.edit_message_text.call_args[1]
    assert str(kwargs["chat_id"]) == "12345"
    assert kwargs["message_id"] == 100
    assert kwargs["reply_markup"] is None
    # Notice must be html-escaped, not interpreted as markup.
    assert "&lt;b&gt;expired&lt;/b&gt;" in kwargs["text"]
    # Both card registries are cleaned up (no leak, late clicks are dead).
    assert "cid1" not in adapter._clarify_messages
    assert "cid1" not in adapter._clarify_state


@pytest.mark.asyncio
async def test_retire_unknown_clarify_id_is_noop():
    adapter = _make_adapter()
    await adapter.retire_clarify_card("ghost", "notice")
    adapter._bot.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_retire_is_idempotent():
    adapter = _make_adapter()
    await _seed_card(adapter, clarify_id="cid1")

    await adapter.retire_clarify_card("cid1", "gone")
    adapter._bot.edit_message_text.reset_mock()
    await adapter.retire_clarify_card("cid1", "gone again")

    adapter._bot.edit_message_text.assert_not_called()
