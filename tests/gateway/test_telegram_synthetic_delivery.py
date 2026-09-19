"""Synthetic replies retain Telegram routes without requiring an unrelated quote."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import BadRequest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, _reply_anchor_for_event, _thread_metadata_for_event
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter(*, rich=False):
    adapter = TelegramAdapter(PlatformConfig(
        enabled=True, token="test-token", reply_to_mode="first", extra={"rich_messages": rich},
    ))
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=99))
    return adapter


def _event(thread_id, reply_anchor):
    return MessageEvent(
        text="Background process completed", internal=True, message_id=reply_anchor,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="123", chat_type="dm",
            user_id="1", thread_id=thread_id,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["text", "rich", "document"])
@pytest.mark.parametrize("thread_id", [None, "1", "24296"])
@pytest.mark.parametrize("reply_anchor", [None, "777"])
async def test_synthetic_reply_preserves_its_dm_route(tmp_path, transport, thread_id, reply_anchor):
    """General uses the normal DM destination; named topics retain their thread."""
    event = _event(thread_id, reply_anchor)
    adapter = _adapter(rich=transport == "rich")
    adapter._bot.do_api_request = AsyncMock(return_value=MagicMock(message_id=99))
    adapter._bot.send_document = AsyncMock(return_value=MagicMock(message_id=99))
    kwargs = dict(reply_to=_reply_anchor_for_event(event), metadata=_thread_metadata_for_event(event))
    if transport == "document":
        path = tmp_path / "report.txt"
        path.write_text("Task completed", encoding="utf-8")
        result = await adapter.send_document(event.source.chat_id, str(path), **kwargs)
        send_method = adapter._bot.send_document
    else:
        result = await adapter.send(
            event.source.chat_id, "| Task | Result |\n|---|---|\n| Build | Complete |", **kwargs,
        )
        send_method = adapter._bot.do_api_request if transport == "rich" else adapter._bot.send_message

    assert result.success, result.error
    send_method.assert_awaited_once()
    sent = send_method.await_args.kwargs
    if transport == "rich":
        sent = sent["api_kwargs"]
    assert sent["chat_id"] == int(event.source.chat_id)
    sent_reply = (sent.get("reply_parameters") or {}).get("message_id") if transport == "rich" else (
        sent.get("reply_to_message_id")
    )
    assert sent_reply == (int(reply_anchor) if reply_anchor else None)
    assert sent.get("message_thread_id") == (
        int(thread_id) if thread_id not in (None, "1") else None
    )
    assert sent.get("direct_messages_topic_id") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("transport", "reply_anchor", "failure"), [
    ("text", None, "Message thread not found"),
    ("rich", None, "Message thread not found"),
    ("document", None, "Message thread not found"),
    ("text", "777", "Message thread not found"),
    ("rich", "777", "Message thread not found"),
    ("text", "777", "Message to be replied not found"),
    ("rich", "777", "Message to be replied not found"),
])
async def test_named_dm_topic_failure_does_not_redirect_to_general(tmp_path, transport, reply_anchor, failure):
    """A fake server accepting root sends exposes successful-looking misdelivery."""
    event = _event("24296", reply_anchor)
    adapter = _adapter(rich=transport == "rich")
    sent = []

    async def reject_topic_but_accept_root(**kwargs):
        sent.append(kwargs)
        if kwargs.get("message_thread_id") == int(event.source.thread_id):
            raise BadRequest(failure)
        return MagicMock(message_id=99)

    async def rich_send(_endpoint, *, api_kwargs):
        return await reject_topic_but_accept_root(**api_kwargs)

    adapter._bot.send_message.side_effect = reject_topic_but_accept_root
    adapter._bot.do_api_request = AsyncMock(side_effect=rich_send)
    adapter._bot.send_document = AsyncMock(side_effect=reject_topic_but_accept_root)
    adapter._prune_stale_dm_topic_binding = MagicMock()
    kwargs = dict(reply_to=_reply_anchor_for_event(event), metadata=_thread_metadata_for_event(event))
    if transport == "document":
        path = tmp_path / "report.txt"
        path.write_text("Task completed", encoding="utf-8")
        result = await adapter.send_document(event.source.chat_id, str(path), **kwargs)
    else:
        result = await adapter.send(
            event.source.chat_id, "| Task | Result |\n|---|---|\n| Build | Complete |", **kwargs,
        )

    assert not result.success
    assert sent
    assert all(call.get("message_thread_id") == int(event.source.thread_id) for call in sent)
    assert all(call.get("direct_messages_topic_id") is None for call in sent)
    adapter._prune_stale_dm_topic_binding.assert_not_called()
