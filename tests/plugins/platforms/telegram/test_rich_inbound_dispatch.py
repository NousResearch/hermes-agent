"""Rich inbound regression through real PTB admission and the adapter pipeline."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("telegram")
from telegram import Update

from gateway.platforms.event import MessageType
from tests.plugins.test_telegram_update_admission import connected, update


def rich_update(bot, uid=10, *, blocks=None, fields=None, **kwargs):
    payload = update(bot, uid, **kwargs).to_dict()
    msg = payload.get("message", payload.get("edited_message"))
    msg.pop("text", None)
    msg["rich_message"] = {"blocks": blocks if blocks is not None else [
        {"type": "table", "cells": [[{"text": "Name"}, {"text": "Status"}],
                                   [{"text": "A"}, {"text": "Ready"}]]},
    ]}
    msg.update(fields or {})
    return Update.de_json(payload, bot)


async def drain(adapter):
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())


@pytest.mark.asyncio
async def test_rich_table_is_admitted_once_with_original_identity(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, delivered):
        incoming = rich_update(app.bot, fields={"message_thread_id": 7, "is_topic_message": True})
        before = incoming.to_dict()
        await app.process_update(incoming)
        await app.process_update(incoming)
        await drain(adapter)
        assert len(delivered) == 1
        event = delivered[0]
        assert event.text == "| Name | Status |\n| A | Ready |"
        assert event.source.user_id == "88" and event.source.chat_id == "42"
        assert event.source.thread_id == "7"
        assert event.message_id == "472" and event.platform_update_id == 10
        assert event.raw_message is incoming.message
        assert not event.allow_gateway_control
        assert incoming.to_dict() == before
        assert "111:10" in adapter._seen_update_ids
        assert not adapter._inflight_update_ids


@pytest.mark.asyncio
async def test_reply_table_uses_same_projection_but_plain_control_is_unchanged(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, _):
        replied = rich_update(app.bot).message.to_dict()
        incoming = rich_update(app.bot, 11, fields={"text": "/stop", "reply_to_message": replied})
        event = adapter._build_message_event(incoming.message, MessageType.TEXT)
        assert event.reply_to_text == "| Name | Status |\n| A | Ready |"
        assert event.text == "/stop" and event.allow_gateway_control
        assert event.get_command() == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["user", "chat", "topic", "mention_only"])
async def test_rejected_rich_never_projects_observes_or_enqueues(monkeypatch, mode):
    from plugins.platforms.telegram import telegram_rich_inbound as rich
    extra = {"allowed_chats": ["42"], "require_mention": True, "guest_mode": True}
    if mode == "topic":
        extra["allowed_topics"] = ["7"]
    async with connected(monkeypatch, extra=extra) as (adapter, app, delivered):
        project = Mock(side_effect=AssertionError("Rejected content must not be parsed"))
        monkeypatch.setattr(rich, "rich_message_text", project)
        observe = Mock()
        monkeypatch.setattr(adapter, "_observe_unmentioned_group_message", observe)
        fields = {"message_thread_id": 8, "is_topic_message": True} if mode == "topic" else {}
        incoming = rich_update(app.bot, user=99 if mode == "user" else 88,
                               chat=43 if mode == "chat" else 42, group=mode != "user",
                               fields=fields, blocks=[{"type": "paragraph", "text": "@offline_bot /stop"}])
        await app.process_update(incoming)
        assert not delivered and not adapter._pending_text_batches
        project.assert_not_called()
        observe.assert_not_called()


@pytest.mark.asyncio
async def test_rich_observation_records_projection_once_without_dispatch(monkeypatch):
    async with connected(monkeypatch, extra={"allowed_chats": ["42"], "require_mention": True,
                                           "group_allowed_chats": ["42"],
                                           "observe_unmentioned_group_messages": True}) as (adapter, app, delivered):
        store = SimpleNamespace(get_or_create_session=Mock(return_value=SimpleNamespace(session_id="offline")),
                                append_to_transcript=Mock(), close_all_db_handles=Mock())
        adapter._session_store = store
        incoming = rich_update(app.bot, group=True)
        await app.process_update(incoming)
        await app.process_update(incoming)
        assert not delivered
        store.append_to_transcript.assert_called_once()
        row = store.append_to_transcript.call_args.args[1]
        assert row["observed"] is True and "| A | Ready |" in row["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "cancel"])
async def test_prequeue_failure_remains_retryable_through_real_admission(monkeypatch, failure):
    async with connected(monkeypatch) as (adapter, app, delivered):
        incoming = rich_update(app.bot)
        original = adapter._cache_replied_media
        fault = OSError("before queue") if failure == "error" else asyncio.CancelledError()
        with monkeypatch.context() as broken:
            broken.setattr(adapter, "_cache_replied_media", AsyncMock(side_effect=fault))
            if failure == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await app.process_update(incoming)
            else:
                await app.process_update(incoming)
        assert not adapter._seen_update_ids and not adapter._inflight_update_ids
        assert not delivered
        assert adapter._cache_replied_media == original
        await app.process_update(incoming)
        await drain(adapter)
        assert len(delivered) == 1 and "111:10" in adapter._seen_update_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["/stop", "/approve", "yes", "1", "@offline_bot /stop"])
@pytest.mark.parametrize("kind", ["paragraph", "table", "blockquote", "pre"])
async def test_rich_control_stays_disabled_after_cleanup_batching_and_plain_command(monkeypatch, text, kind):
    async with connected(monkeypatch) as (adapter, app, delivered):
        block = {"type": kind, "text": text}
        if kind == "table":
            block = {"type": kind, "cells": [[{"text": text}]]}
        elif kind == "blockquote":
            block = {"type": kind, "blocks": [{"type": "paragraph", "text": text}]}
        for uid in (10, 11):
            await app.process_update(rich_update(app.bot, uid, blocks=[block]))
        # Same topic/user, but plaintext and rich chunks cannot upgrade each other's control rights.
        await app.process_update(update(app.bot, 12, text="plain text"))
        assert len(adapter._pending_text_batches) == 2
        await drain(adapter)
        assert len(delivered) == 2
        rich = next(event for event in delivered if not event.allow_gateway_control)
        assert not rich.is_command() and rich.get_command() is None
        if kind == "paragraph":
            assert "@offline_bot" not in rich.text
        plain = next(event for event in delivered if event.allow_gateway_control)
        assert plain.text == "plain text"
        from gateway.run_inbound import GatewayInboundMixin
        prompts = SimpleNamespace(_hm_update_prompt_reply=Mock(return_value="confirmed"),
                                  _hm_clarify_reply=AsyncMock(), _hm_slash_confirm_reply=AsyncMock())
        assert await GatewayInboundMixin._hm_pending_reply_intercepts(prompts, rich, rich.source, "offline") is None
        prompts._hm_update_prompt_reply.assert_not_called()
        prompts._hm_clarify_reply.assert_not_awaited()
        prompts._hm_slash_confirm_reply.assert_not_awaited()
        assert await GatewayInboundMixin._hm_pending_reply_intercepts(prompts, plain, plain.source, "offline") == "confirmed"
        # Ordinary commands keep their immediate path, including after a rich batch.
        await app.process_update(update(app.bot, 13, kind="command"))
        assert delivered[-1].get_command() == "status"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["location", "photo", "command"])
async def test_rich_field_does_not_steal_existing_handlers(monkeypatch, kind):
    async with connected(monkeypatch) as (_adapter, app, _):
        incoming = update(app.bot, kind=kind).to_dict()
        incoming["message"]["rich_message"] = {"blocks": [{"type": "paragraph", "text": "extra"}]}
        incoming = Update.de_json(incoming, app.bot)
        selected = next(handler for handler in app.handlers[0] if handler.check_update(incoming))
        assert selected.callback.__name__ == {"location": "_handle_location_message", "photo": "_handle_media_message",
                                              "command": "_handle_command"}[kind]
        observers = [handler for handler in app.handlers[99] if handler.check_update(incoming)]
        assert len(observers) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fields", [{"text": "plain"}, {"caption": "caption"}])
async def test_normal_carrier_and_partial_quote_keep_priority(monkeypatch, fields):
    async with connected(monkeypatch) as (adapter, app, _):
        replied = rich_update(app.bot).message.to_dict()
        incoming = rich_update(app.bot, fields={**fields, "reply_to_message": replied,
                                               "quote": {"text": "selected", "position": 0}})
        event = adapter._build_message_event(incoming.message, MessageType.TEXT)
        assert event.text == next(iter(fields.values()))
        assert event.reply_to_text == "selected" and event.allow_gateway_control


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,expected", [(None, "Invalid"), ({"blocks": []}, "Empty"),
                                             ({"blocks": [{"type": "future", "text": "secret"}]}, "Unsupported"),
                                             ({"blocks": [{"type": "paragraph", "text": "x" * 40000}]}, "truncated")])
async def test_invalid_unknown_empty_and_oversized_rich_are_visible_and_settled(monkeypatch, payload, expected):
    async with connected(monkeypatch) as (adapter, app, delivered):
        raw = update(app.bot).to_dict()
        raw["message"].pop("text")
        raw["message"]["rich_message"] = payload
        incoming = Update.de_json(raw, app.bot)
        await app.process_update(incoming)
        await app.process_update(incoming)
        await drain(adapter)
        assert len(delivered) == 1
        assert expected in delivered[0].text and len(delivered[0].text) <= 32768
        assert not delivered[0].allow_gateway_control
        assert "111:10" in adapter._seen_update_ids


@pytest.mark.asyncio
async def test_rich_hold_and_redispatch_preserve_control_and_admission(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, delivered):
        adapter._mark_disconnected()
        incoming = rich_update(app.bot)
        await app.process_update(incoming)
        await app.process_update(incoming)
        assert not delivered and len(adapter._held_inbound_events) == 1
        adapter._mark_connected()
        await adapter._held_inbound_redispatch_task
        assert len(delivered) == 1 and not delivered[0].allow_gateway_control
        assert "| A | Ready |" in delivered[0].text


@pytest.mark.asyncio
async def test_media_merge_cannot_restore_gateway_control(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, _):
        rich = adapter._build_message_event(rich_update(app.bot).message, MessageType.PHOTO)
        plain = adapter._build_message_event(update(app.bot).message, MessageType.PHOTO)
        for events in ((rich, plain), (plain, rich)):
            pending = {}
            for event in events:
                adapter._merge_into_pending(pending, "album", event)
            assert pending["album"].allow_gateway_control is False
