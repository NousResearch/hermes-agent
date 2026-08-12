"""Regression tests for top-level Telegram Bot API 10.1 rich-message ingress."""

from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter():
    return TelegramAdapter(PlatformConfig(enabled=True, token="fake", extra={}))


def _message(rich, *, origin=None):
    return SimpleNamespace(
        text=None, caption=None, api_kwargs=MappingProxyType({"rich_message": rich}),
        rich_message=None,
        chat=SimpleNamespace(id=123, type="private", title=None, full_name="Private"),
        from_user=SimpleNamespace(id=456, full_name="Maxim", username="qwinty"),
        message_id=77, date=None, message_thread_id=None, direct_messages_topic=None,
        is_topic_message=False, reply_to_message=None, forward_origin=origin,
        forum_topic_created=None,
    )


@pytest.mark.asyncio
async def test_rich_only_update_routes_through_text_ingress_with_hidden_link():
    adapter = _adapter()
    rich = MappingProxyType({"blocks": [MappingProxyType({
        "type": "paragraph", "text": MappingProxyType({
            "type": "url", "text": "Why Agentic Systems Need Ontologies",
            "url": "https://www.youtube.com/watch?v=Sir59K8ZDPU",
        })
    })]})
    msg = _message(rich)
    update = SimpleNamespace(effective_message=msg, message=msg, update_id=91)
    adapter._should_process_message = MagicMock(return_value=True)
    adapter._is_user_authorized_from_message = MagicMock(return_value=True)
    adapter._ensure_forum_commands = AsyncMock()
    adapter._cache_replied_media = AsyncMock()
    adapter._enqueue_text_event = MagicMock()

    await adapter._handle_rich_message(update, None)

    event = adapter._enqueue_text_event.call_args.args[0]
    assert event.message_type is MessageType.TEXT
    assert event.text == "Why Agentic Systems Need Ontologies (https://www.youtube.com/watch?v=Sir59K8ZDPU)"


@pytest.mark.asyncio
async def test_rich_only_update_preserves_forward_origin():
    adapter = _adapter()
    origin = SimpleNamespace(type="channel", chat=SimpleNamespace(title="Vibe Coding"), date=None)
    msg = _message({"blocks": [{"type": "paragraph", "text": "Forwarded body"}]}, origin=origin)
    update = SimpleNamespace(effective_message=msg, message=msg, update_id=92)
    adapter._should_process_message = MagicMock(return_value=True)
    adapter._is_user_authorized_from_message = MagicMock(return_value=True)
    adapter._ensure_forum_commands = AsyncMock()
    adapter._cache_replied_media = AsyncMock()
    adapter._enqueue_text_event = MagicMock()

    await adapter._handle_rich_message(update, None)

    event = adapter._enqueue_text_event.call_args.args[0]
    assert event.text == "Forwarded body"
    assert event.forward_origin["chat_name"] == "Vibe Coding"


@pytest.mark.asyncio
async def test_rich_only_update_obeys_authorization_gate():
    adapter = _adapter()
    msg = _message({"blocks": [{"type": "paragraph", "text": "Secret"}]})
    update = SimpleNamespace(effective_message=msg, message=msg, update_id=93)
    adapter._is_user_authorized_from_message = MagicMock(return_value=False)
    adapter._enqueue_text_event = MagicMock()

    await adapter._handle_rich_message(update, None)
    adapter._enqueue_text_event.assert_not_called()


def test_rich_filter_matches_only_payload_bearing_messages():
    adapter = _adapter()
    rich = _message({"blocks": [{"type": "paragraph", "text": "Body"}]})
    ordinary = _message(None)
    ordinary.text = "ordinary"
    assert adapter._is_rich_message_update(rich) is True
    assert adapter._is_rich_message_update(ordinary) is False


def test_rich_block_flattener_handles_nested_structures():
    blocks = [
        {"type": "section_heading", "text": {"type": "bold", "text": "Heading"}},
        {"type": "list", "items": [
            {"label": "1.", "blocks": [{"type": "paragraph", "text": "First"}]},
            {"blocks": [{"type": "block_quotation", "text": "Second"}]},
        ]},
        {"type": "details", "title": "More", "blocks": [{"type": "paragraph", "text": "Details body"}]},
        {"type": "future_unknown", "children": [{"text": "Fallback"}]},
    ]
    text = TelegramAdapter._flatten_rich_blocks(blocks)
    for expected in ("Heading", "1. First", "Second", "More", "Details body", "Fallback"):
        assert expected in text
