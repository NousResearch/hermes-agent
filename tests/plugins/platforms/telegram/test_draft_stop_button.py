"""Bot API 10.3 native Stop button for streaming drafts (opt-in ``rich_draft_stop``).

When the extra is on, draft sends carry ``can_stop``/``keep_on_stop`` so Telegram shows a
native Stop button while the draft streams. The installed PTB's typed ``send_message_draft``
has no such field, so the enabled plain path goes through raw ``do_api_request``
("sendMessageDraft"), mirroring the rich path. With the extra absent the payloads must stay
byte-identical to the pre-10.3 behaviour (upstream tests pin the current payloads).
A ``stopped_message_generation`` update (the user tapped Stop) is re-dispatched as a
synthetic ``/stop`` through the normal inbound path, so the running turn is interrupted
with the same runner verdict as a typed ``/stop``. Private chats only: the payload carries
no separate user identity, so a group tap must not pass as the chat owner (fail-closed).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.constants import ParseMode
from telegram.error import BadRequest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(**extra) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra=extra))
    adapter._bot = MagicMock()
    adapter._bot.send_message_draft = AsyncMock(return_value=True)
    adapter._bot.do_api_request = AsyncMock(return_value=True)
    adapter.format_message = lambda content, **kwargs: content
    return adapter


# ---------------------------------------------------------------------------
# plain sendMessageDraft transport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_off_plain_draft_payload_unchanged():
    """Extra absent: typed send_message_draft, exact pre-10.3 kwargs, no raw call."""
    adapter = _make_adapter()
    assert adapter._rich_draft_stop is False

    result = await adapter.send_draft("123", 7, "hello")

    assert result.success is True
    adapter._bot.send_message_draft.assert_awaited_once_with(
        chat_id=123, draft_id=7, text="hello", parse_mode=ParseMode.MARKDOWN_V2)
    adapter._bot.do_api_request.assert_not_called()


@pytest.mark.asyncio
async def test_stop_on_routes_raw_send_message_draft_with_can_stop():
    """Extra on: raw sendMessageDraft frame carries can_stop + keep_on_stop."""
    adapter = _make_adapter(rich_draft_stop=True)
    assert adapter._rich_draft_stop is True

    result = await adapter.send_draft("123", 7, "hello")

    assert result.success is True
    adapter._bot.send_message_draft.assert_not_called()
    adapter._bot.do_api_request.assert_awaited_once_with(
        "sendMessageDraft",
        api_kwargs={
            "chat_id": 123, "draft_id": 7, "text": "hello",
            "parse_mode": ParseMode.MARKDOWN_V2, "can_stop": True, "keep_on_stop": True,
        },
    )


@pytest.mark.asyncio
async def test_stop_on_markdownv2_bad_request_falls_back_plain_keeps_can_stop():
    """The MarkdownV2 -> plain retry keeps the raw transport and the stop fields."""
    adapter = _make_adapter(rich_draft_stop="true")  # string coercion via _coerce_bool_extra
    assert adapter._rich_draft_stop is True
    payloads: list[dict] = []

    async def _do_api_request(method, api_kwargs=None, **kwargs):
        payloads.append(dict(api_kwargs or {}))
        if "parse_mode" in api_kwargs:
            raise BadRequest("can't parse entities")
        return True

    adapter._bot.do_api_request = AsyncMock(side_effect=_do_api_request)

    result = await adapter.send_draft("123", 7, "weird _text")

    assert result.success is True
    assert len(payloads) == 2
    assert payloads[0]["can_stop"] is True and payloads[0]["keep_on_stop"] is True
    assert "parse_mode" not in payloads[1]
    assert payloads[1]["can_stop"] is True and payloads[1]["keep_on_stop"] is True


# ---------------------------------------------------------------------------
# rich sendRichMessageDraft transport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rich_draft_payload_carries_stop_fields_when_enabled():
    adapter = _make_adapter(rich_messages=True, rich_drafts=True, rich_draft_stop=True)

    ok = await adapter._try_send_rich_draft("123", 5, "|a|\n|---|\n|b|", None)

    assert ok is True
    call = adapter._bot.do_api_request.await_args
    assert call.args[0] == "sendRichMessageDraft"
    payload = call.kwargs["api_kwargs"]
    assert payload["can_stop"] is True
    assert payload["keep_on_stop"] is True


@pytest.mark.asyncio
async def test_rich_draft_payload_unchanged_when_stop_disabled():
    adapter = _make_adapter(rich_messages=True, rich_drafts=True)

    ok = await adapter._try_send_rich_draft("123", 5, "|a|\n|---|\n|b|", None)

    assert ok is True
    payload = adapter._bot.do_api_request.await_args.kwargs["api_kwargs"]
    assert "can_stop" not in payload
    assert "keep_on_stop" not in payload


# ---------------------------------------------------------------------------
# stopped_message_generation updates → synthetic /stop
# ---------------------------------------------------------------------------
# Duck-typed updates (SimpleNamespace) on purpose: the gateway conftest may
# install a MagicMock ``telegram`` package session-wide, so ``Update.de_json``
# is not dependable. PTB 22.8 exposes unknown Bot API fields via
# ``Update.api_kwargs`` (a mapping); a future PTB may expose a native attribute.


def _stop_update_api_kwargs(payload=None) -> SimpleNamespace:
    return SimpleNamespace(
        update_id=1,
        api_kwargs={"stopped_message_generation": payload or {"chat_id": 123, "message_id": 9}})


@pytest.mark.asyncio
async def test_stopped_generation_update_consumed_without_event_handler_call(monkeypatch):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
    adapter = _make_adapter(rich_draft_stop=True)
    adapter._platform_event_handler = AsyncMock()
    handled = AsyncMock()
    monkeypatch.setattr(adapter, "_handle_stopped_message_generation", handled)

    await adapter._on_platform_update(_stop_update_api_kwargs(), None)  # must not raise

    handled.assert_awaited_once_with({"chat_id": 123, "message_id": 9})
    adapter._platform_event_handler.assert_not_called()


@pytest.mark.asyncio
async def test_stop_tap_dispatches_synthetic_stop(monkeypatch):
    """A private-chat tap re-enters the inbound path as a /stop command event."""
    from gateway.platforms.event import MessageType

    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
    adapter = _make_adapter(rich_draft_stop=True)
    adapter.handle_message = AsyncMock()

    await adapter._on_platform_update(_stop_update_api_kwargs(), None)

    adapter.handle_message.assert_awaited_once()
    call = adapter.handle_message.await_args
    assert call is not None
    event = call.args[0]
    assert event.text == "/stop"
    assert event.message_type == MessageType.COMMAND
    assert event.source.chat_id == "123"
    assert event.source.user_id == "123"  # DM: the tap author is the chat owner
    assert event.source.chat_type == "dm"


@pytest.mark.asyncio
async def test_stop_tap_native_attribute_shape_dispatches_stop():
    """A future PTB may expose the field natively instead of via api_kwargs."""
    adapter = _make_adapter(rich_draft_stop=True)
    adapter.handle_message = AsyncMock()
    update = SimpleNamespace(update_id=1, stopped_message_generation={"chat": {"id": 55}})

    await adapter._on_platform_update(update, None)

    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].text == "/stop"


@pytest.mark.asyncio
async def test_stop_tap_ignored_outside_private_chats():
    """Fail-closed: the payload has no separate sender, so a group tap must not act as the owner."""
    adapter = _make_adapter(rich_draft_stop=True)
    adapter.handle_message = AsyncMock()
    payload = {"chat": {"id": -100, "type": "group"}, "message_id": 9}

    await adapter._handle_stopped_message_generation(payload)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_tap_without_chat_id_is_dropped():
    adapter = _make_adapter(rich_draft_stop=True)
    adapter.handle_message = AsyncMock()

    await adapter._handle_stopped_message_generation({"message_id": 9})

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_tap_dispatch_failure_does_not_escape():
    """A failing /stop dispatch must never crash the update loop."""
    adapter = _make_adapter(rich_draft_stop=True)
    adapter.handle_message = AsyncMock(side_effect=RuntimeError("boom"))

    await adapter._handle_stopped_message_generation({"chat_id": 123})  # must not raise


@pytest.mark.asyncio
async def test_stopped_generation_update_harmless_when_feature_disabled(monkeypatch):
    """Extra absent: the unknown update type must simply be ignored, never crash."""
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
    adapter = _make_adapter()
    adapter._platform_event_handler = AsyncMock()
    adapter.handle_message = AsyncMock()

    await adapter._on_platform_update(_stop_update_api_kwargs(), None)

    adapter._platform_event_handler.assert_not_called()
    adapter.handle_message.assert_not_awaited()


def _edited_update() -> SimpleNamespace:
    chat = SimpleNamespace(id=123, type="private", is_forum=False)
    user = SimpleNamespace(id=777, is_bot=False, first_name="Editor", username="editor", full_name="Editor")
    message = SimpleNamespace(chat=chat, from_user=user, sender_chat=None, message_id=456,
                              text="fixed typo", caption=None, message_thread_id=None,
                              is_topic_message=False, edit_date=None)
    return SimpleNamespace(update_id=2, message_reaction=None, edited_message=message)


@pytest.mark.asyncio
async def test_normal_updates_still_flow_with_stop_enabled(monkeypatch):
    """The stop branch must not swallow ordinary edited-message updates."""
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
    adapter = _make_adapter(rich_draft_stop=True)
    seen: list = []

    async def observe(event, source):
        seen.append((event, source))

    adapter.set_platform_event_handler(observe)

    await adapter._on_platform_update(_edited_update(), None)

    assert len(seen) == 1
    assert seen[0][0]["event_type"] == "message_edited"
