"""Real PTB dispatch and authenticated media extraction for the delegated inbox."""

import asyncio
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from telegram import Update
from telegram.ext import Application, ExtBot
from telegram.request import BaseRequest

from gateway.config import PlatformConfig
from gateway.platforms.base import _thread_metadata_for_event
from gateway.platforms.event import MessageType
from gateway.session import build_session_key
from plugins.platforms.telegram.adapter import TelegramAdapter


class TelegramTransport(BaseRequest):
    """Exercise the SDK parsing/dispatch/download path without an external Telegram account."""

    def __init__(self):
        self.calls = []
        self.enabled = True
        self.can_reply = True
        buf = io.BytesIO()
        Image.new("RGB", (2, 2), "blue").save(buf, format="PNG")
        self.image = buf.getvalue()

    @property
    def read_timeout(self):
        return 5

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(self, url, method, request_data=None, **kwargs):
        params = request_data.parameters if request_data else {}
        operation = url.rsplit("/", 1)[-1]
        self.calls.append((operation, params))
        if "/file/" in url:
            return 200, self.image if operation.endswith(".png") else b"customer attachment"
        if operation == "getMe":
            result = {"id": 999, "is_bot": True, "first_name": "Hermes", "username": "hermes_bot"}
        elif operation == "getBusinessConnection":
            result = {"id": params["business_connection_id"], "user": {
                "id": 777, "is_bot": False, "first_name": "Owner"}, "user_chat_id": 777,
                "date": 1, "is_enabled": self.enabled, "rights": {"can_reply": self.can_reply}}
        elif operation == "getFile":
            result = {"file_id": params["file_id"], "file_unique_id": params["file_id"],
                      "file_size": 20, "file_path": f"files/{params['file_id']}"}
        elif operation == "sendChatAction":
            result = True
        else:
            result = {"message_id": 90, "date": 1, "chat": {"id": 123, "type": "private"}, "text": "reply"}
        return 200, json.dumps({"ok": True, "result": result}).encode()


@asynccontextmanager
async def inbox(monkeypatch, **policy):
    monkeypatch.setenv("TELEGRAM_ALLOW_ALL_USERS", "true")
    transport = TelegramTransport()
    bot = ExtBot("123:TEST", request=transport, get_updates_request=TelegramTransport())
    app = Application.builder().bot(bot).updater(None).build()
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:TEST", extra={"business": {
        "enabled": True, "allow_business_send_as_account": True, "trigger_words": ["Hermes"],
        "allowed_owner_ids": ["777"], **policy}}))
    adapter._bot = bot
    adapter._media_batch_delay_seconds = 0
    adapter.MEDIA_GROUP_WAIT_SECONDS = 60  # tests flush explicitly, without timing assumptions
    adapter._telegram_chat_outbound_slot_secs = 0
    accepted = []
    adapter.handle_message = AsyncMock(side_effect=accepted.append)
    adapter._enqueue_text_event = accepted.append
    adapter._retrigger_typing = AsyncMock()
    adapter._register_handlers(app)
    errors = []

    async def error_handler(update, context):
        errors.append(context.error)

    app.add_error_handler(error_handler)
    await app.initialize()
    try:
        yield adapter, app, transport, accepted, errors
    finally:
        tasks = list(adapter._media_group_tasks.values()) + list(adapter._pending_photo_batch_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await app.shutdown()


def update(bot, *, kind="business_message", text="Hermes hello", media=None, connection="account-a",
           album=None, message_id=42, actor=123):
    message = {"message_id": message_id, "date": 1, "chat": {"id": 123, "type": "private", "first_name": "Customer"},
               "from": {"id": actor, "is_bot": False, "first_name": "Customer"}, "business_connection_id": connection}
    if media:
        name, suffix = {"photo": ("photo", "png"), "voice": ("voice", "ogg"), "audio": ("audio", "mp3"),
                        "video": ("video", "mp4"), "document": ("document", "txt")}[media]
        attachment = {"file_id": f"{message_id}.{suffix}", "file_unique_id": str(message_id), "file_size": 20,
                      "width": 2, "height": 2, "duration": 1, "file_name": f"note.{suffix}",
                      "mime_type": "text/plain" if media == "document" else f"{media}/{suffix}"}
        message[name] = [attachment] if media == "photo" else attachment
        if text is not None:
            message["caption"] = text
    else:
        message["text"] = text
        if text and text.startswith("/"):
            message["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text)}]
    if album:
        message["media_group_id"] = album
    return Update.de_json({"update_id": message_id, kind: message}, bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["business_message", "edited_business_message"])
@pytest.mark.parametrize("text", ["Hermes hello", "/reset"])
@pytest.mark.parametrize("enabled", [False, True])
async def test_ptb_dispatch_never_admits_business_through_ordinary_allow_all(monkeypatch, kind, text, enabled):
    async with inbox(monkeypatch, enabled=enabled) as (adapter, app, transport, accepted, errors):
        await app.process_update(update(app.bot, kind=kind, text=text))
        assert not errors
        admitted = kind == "business_message" and enabled and text == "Hermes hello"
        assert bool(accepted) is admitted
        if admitted:
            event = accepted[0]
            assert event.text == "hello"
            assert event.allow_gateway_control is False
            assert event.source.scope_id == "telegram-business:account-a"
            assert event.source._transport_adapter_ref() is adapter
        assert not any(op.startswith("send") for op, _ in transport.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("media,expected", [("photo", MessageType.PHOTO), ("voice", MessageType.VOICE),
    ("audio", MessageType.AUDIO), ("video", MessageType.VIDEO), ("document", MessageType.DOCUMENT)])
async def test_business_media_downloads_only_after_admission(monkeypatch, media, expected):
    async with inbox(monkeypatch) as (adapter, app, transport, accepted, errors):
        await app.process_update(update(app.bot, media=media, text=None))
        await app.process_update(update(app.bot, media=media, text="Hermes inspect", actor=777))
        assert not accepted
        assert not any(op == "getFile" for op, _ in transport.calls)
        await app.process_update(update(app.bot, media=media, text="Hermes inspect"))
        tasks = list(adapter._pending_photo_batch_tasks.values())
        if tasks:
            await asyncio.gather(*tasks)
        assert not errors
        event = accepted[0]
        assert event.message_type == expected
        assert len(event.media_urls) == 1
        assert Path(event.media_urls[0]).read_bytes()
        assert event.allow_gateway_control is False
        assert "inspect" in event.text
        assert (await adapter.send(event.source.chat_id, "reply", metadata=_thread_metadata_for_event(event))).success
        sends = [params for op, params in transport.calls if op == "sendMessage"]
        assert sends[-1]["business_connection_id"] == "account-a"


@pytest.mark.asyncio
async def test_business_album_captionless_policy_and_connection_isolation(monkeypatch):
    async with inbox(monkeypatch) as (adapter, app, transport, accepted, errors):
        # Identical Telegram album IDs in two connected accounts must not merge.
        for connection in ("account-a", "account-b"):
            await app.process_update(update(app.bot, media="photo", text="Hermes inspect", album="same", connection=connection))
            await app.process_update(update(app.bot, media="photo", text=None, album="same", connection=connection, message_id=43))
        # Neither an untriggered account nor another sender inherits the pending trigger.
        downloads = sum(op == "getFile" for op, _ in transport.calls)
        await app.process_update(update(app.bot, media="photo", text=None, album="same", connection="account-c"))
        await app.process_update(update(app.bot, media="photo", text=None, album="same", actor=124))
        assert sum(op == "getFile" for op, _ in transport.calls) == downloads
        assert not errors
        pending = list(adapter._media_group_events.values())
        assert len(pending) == 2
        assert all(len(event.media_urls) == 2 for event in pending)
        assert len({build_session_key(event.source) for event in pending}) == 2
        assert {event.source.scope_id for event in pending} == {"telegram-business:account-a", "telegram-business:account-b"}
        assert not accepted


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["split", "markdown"])
async def test_each_business_text_rpc_rechecks_live_authority(monkeypatch, failure):
    async with inbox(monkeypatch) as (adapter, app, transport, accepted, errors):
        await app.process_update(update(app.bot))
        event = accepted[0]
        original_request = transport.do_request

        async def revoke_after_first_send(url, method, **kwargs):
            response = await original_request(url, method, **kwargs)
            if url.endswith("/sendMessage"):
                transport.enabled = False
                if failure == "markdown":
                    return 400, json.dumps({"ok": False, "error_code": 400,
                                            "description": "Bad Request: can't parse entities"}).encode()
            return response

        monkeypatch.setattr(transport, "do_request", revoke_after_first_send)
        result = await adapter.send("123", "reply " * (1600 if failure == "split" else 1),
                                    metadata=_thread_metadata_for_event(event))
        assert not result.success
        sends = [params for op, params in transport.calls if op == "sendMessage"]
        assert len(sends) == 1
        assert sends[0]["business_connection_id"] == "account-a"
        assert not errors


@pytest.mark.asyncio
async def test_slow_captionless_album_download_cannot_start_a_second_turn(monkeypatch):
    async with inbox(monkeypatch) as (adapter, app, transport, accepted, errors):
        await app.process_update(update(app.bot, media="photo", album="album"))
        parent = next(iter(adapter._media_group_events.values()))
        original_request = transport.do_request

        async def flush_trigger_during_download(url, method, **kwargs):
            if "/file/" in url:
                adapter._media_group_events.clear()
                await adapter.handle_message(parent)
            return await original_request(url, method, **kwargs)

        monkeypatch.setattr(transport, "do_request", flush_trigger_during_download)
        await app.process_update(update(app.bot, media="photo", text=None, album="album", message_id=43))
        assert accepted == [parent]
        assert not adapter._media_group_events
        assert not errors
