"""TelegramAdapter.send_draft uses direct_messages_topic_id for private DM topics.

Regression tests for issue #45770: draft streaming in private DM topics must
use ``direct_messages_topic_id`` (not ``message_thread_id``) so Telegram
renders the draft in the correct topic lane.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


def _make_adapter(extra=None):
    """Build a TelegramAdapter with a mock bot wired for draft sends."""
    config = PlatformConfig(enabled=True, token="fake-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    bot.do_api_request = AsyncMock(return_value=True)
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.send_chat_action = AsyncMock()
    bot.send_message_draft = AsyncMock(return_value=True)
    adapter._bot = bot
    return adapter


# ── Rich draft (sendRichMessageDraft) ─────────────────────────────────


@pytest.mark.asyncio
async def test_rich_draft_uses_direct_messages_topic_id_for_dm_topics():
    """sendRichMessageDraft must use direct_messages_topic_id, not
    message_thread_id, when metadata carries a DM topic identifier."""
    adapter = _make_adapter()
    metadata = {
        "direct_messages_topic_id": "999",
        "telegram_dm_topic_reply_fallback": True,
    }

    result = await adapter.send_draft(
        "12345", draft_id=7, content="hello", metadata=metadata,
    )

    assert result.success is True
    call = adapter._bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessageDraft"
    api_kwargs = call.kwargs["api_kwargs"]
    assert api_kwargs.get("direct_messages_topic_id") == 999
    assert api_kwargs.get("message_thread_id") is None


@pytest.mark.asyncio
async def test_rich_draft_uses_message_thread_id_for_forum_topics():
    """Forum/supergroup topics should still use message_thread_id."""
    adapter = _make_adapter()
    metadata = {"thread_id": "42"}

    result = await adapter.send_draft(
        "12345", draft_id=7, content="hello", metadata=metadata,
    )

    assert result.success is True
    call = adapter._bot.do_api_request.call_args
    api_kwargs = call.kwargs["api_kwargs"]
    assert api_kwargs.get("message_thread_id") == 42
    assert "direct_messages_topic_id" not in api_kwargs


# ── Legacy plain-text draft (sendMessageDraft) ───────────────────────


@pytest.mark.asyncio
async def test_plain_draft_uses_direct_messages_topic_id_for_dm_topics():
    """sendMessageDraft fallback must also use direct_messages_topic_id
    for private DM topics."""
    adapter = _make_adapter()
    # Disable rich drafts so the legacy path runs.
    adapter._rich_draft_disabled = True
    metadata = {
        "direct_messages_topic_id": "888",
        "telegram_dm_topic_reply_fallback": True,
    }

    result = await adapter.send_draft(
        "12345", draft_id=7, content="hello", metadata=metadata,
    )

    assert result.success is True
    adapter._bot.send_message_draft.assert_awaited_once()
    call = adapter._bot.send_message_draft.call_args
    kwargs = call.kwargs
    assert kwargs.get("direct_messages_topic_id") == 888
    assert kwargs.get("message_thread_id") is None


@pytest.mark.asyncio
async def test_plain_draft_uses_message_thread_id_for_forum_topics():
    """Forum topics should still use message_thread_id in legacy drafts."""
    adapter = _make_adapter()
    adapter._rich_draft_disabled = True
    metadata = {"thread_id": "42"}

    result = await adapter.send_draft(
        "12345", draft_id=7, content="hello", metadata=metadata,
    )

    assert result.success is True
    adapter._bot.send_message_draft.assert_awaited_once()
    call = adapter._bot.send_message_draft.call_args
    kwargs = call.kwargs
    assert kwargs.get("message_thread_id") == 42
    assert "direct_messages_topic_id" not in kwargs


@pytest.mark.asyncio
async def test_plain_draft_omits_thread_kwargs_when_no_metadata():
    """No thread metadata → no thread kwargs at all."""
    adapter = _make_adapter()
    adapter._rich_draft_disabled = True

    result = await adapter.send_draft(
        "12345", draft_id=7, content="hello", metadata=None,
    )

    assert result.success is True
    call = adapter._bot.send_message_draft.call_args
    kwargs = call.kwargs
    assert "message_thread_id" not in kwargs
    assert "direct_messages_topic_id" not in kwargs
