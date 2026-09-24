"""Telegram replay admission through real PTB dispatch (Refs #68502).

The four-handler replay cases build on @smfworks' #68906 tests. Transport
and model work are stand-ins; registration, gating, batching and PTB are real.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("telegram")
from telegram import InlineQuery, Message, PhotoSize, Sticker, Update
from telegram.ext import ApplicationHandlerStop, ConversationHandler, Defaults, MessageHandler, TypeHandler, filters
from telegram.request import BaseRequest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter


class NoNetwork(BaseRequest):
    def __init__(self, bot_id=111):
        self.bot_id = bot_id
        self.answers = []

    @property
    def read_timeout(self):
        return 1

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(self, url, method, request_data=None, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        if endpoint in ("answerCallbackQuery", "answerInlineQuery"):
            self.answers.append(endpoint)
            return 200, b'{"ok":true,"result":true}'
        assert endpoint == "getMe", "No Telegram polls, sends or downloads allowed"
        return 200, json.dumps({"ok": True, "result": {
            "id": self.bot_id, "is_bot": True, "first_name": "Offline", "username": "offline_bot",
        }}).encode()


def update(bot, uid=10, kind="text", *, edited=False, chat=42, text="hello", group=False, user=88):
    sender = {"id": user, "is_bot": False, "first_name": "Human"}
    message = {
        "message_id": 472, "date": 1800000000,
        "chat": {"id": chat, "type": "supergroup" if group else "private"},
        "from": sender,
    }
    payloads = {
        "text": {"text": text},
        "command": {"text": "/status", "entities": [{"type": "bot_command", "offset": 0, "length": 7}]},
        "location": {"location": {"latitude": 37.77, "longitude": -122.42}},
        "media": {"sticker": {"file_id": "offline", "file_unique_id": "offline", "type": "regular",
                              "width": 128, "height": 128, "is_animated": True, "is_video": False}},
        "photo": {"photo": [{"file_id": "offline", "file_unique_id": "offline", "width": 1, "height": 1}]},
    }
    if kind == "album":
        message["media_group_id"] = "album"
        kind = "photo"
    special = {
        "callback": {"callback_query": {"id": "query", "from": sender, "chat_instance": "chat",
                                        "data": "ea:yes:expired", "message": message}},
        "inline": {"inline_query": {"id": "query", "from": sender, "query": "", "offset": ""}},
        "reaction": {"message_reaction": {"chat": message["chat"], "message_id": 472, "date": 1800000010,
                                          "user": sender, "old_reaction": [],
                                          "new_reaction": [{"type": "emoji", "emoji": "👍"}]}},
    }
    if kind in special:
        return Update.de_json({"update_id": uid, **special[kind]}, bot)
    message.update(payloads[kind])
    if edited:
        message["edit_date"] = 1800000010
    return Update.de_json({"update_id": uid, "edited_message" if edited else "message": message}, bot)


@asynccontextmanager
async def connected(monkeypatch, *, extra=None, bot_id=111, is_reconnect=False):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=f"{bot_id}:offline-test", extra=extra or {}))
    # Only transport/lifecycle services and the final model-work boundary are replaced.
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "88")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "")
    monkeypatch.setattr(adapter, "_build_ptb_requests", AsyncMock(return_value=(NoNetwork(bot_id), NoNetwork(bot_id))))
    monkeypatch.setattr(adapter, "_start_polling_mode", AsyncMock())
    monkeypatch.setattr(adapter, "_start_post_connect_housekeeping", lambda: None)
    monkeypatch.setattr(adapter, "_restart_task_attr", lambda name, coroutine: coroutine.close())
    monkeypatch.setattr(adapter, "_set_status_indicator", AsyncMock())
    delivered = []
    adapter._message_handler = AsyncMock()

    def start(event, session_key):
        delivered.append(event)
        return True

    monkeypatch.setattr(adapter, "_start_session_processing", start)
    assert await adapter.connect(is_reconnect=is_reconnect)
    try:
        yield adapter, adapter._app, delivered
    finally:
        await adapter.disconnect()
        store = getattr(adapter, "_session_store", None)
        if store is not None:
            store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,mode", [
    *[(kind, "normal") for kind in ("text", "command", "location", "media", "photo", "album", "callback", "inline", "reaction")],
    ("text", "edit"), ("text", "observe"), ("media", "observe"), ("text", "hold"),
    ("photo", "hold"), ("album", "hold"), ("text", "unauthorized"),
    ("command", "retention"), ("text", "owners"),
])
@pytest.mark.parametrize("concurrent", [False, True])
async def test_replay_is_admitted_once_before_dispatch(monkeypatch, tmp_path, kind, mode, concurrent):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    import hermes_cli.lifecycle

    monkeypatch.setattr(hermes_cli.lifecycle, "has_hook", lambda name: True)
    # Transport-only photo stand-in: no file or Telegram network operation.
    download = AsyncMock(return_value=SimpleNamespace(
        file_path="offline.png", download_as_bytearray=AsyncMock(return_value=bytearray(b"offline"))))
    monkeypatch.setattr(PhotoSize, "get_file", download)
    monkeypatch.setattr("plugins.platforms.telegram.adapter.cache_image_from_bytes_async", AsyncMock(return_value=str(tmp_path / "photo.png")))
    async with connected(monkeypatch, extra={"observe_unmentioned_group_messages": True, "require_mention": True,
                                            "allowed_chats": ["42"], "group_allowed_chats": ["42"]}) as (adapter, app, delivered):
        observer = adapter._platform_event_handler = AsyncMock()
        adapter._media_batch_delay_seconds = adapter.MEDIA_GROUP_WAIT_SECONDS = 0.01
        if mode == "observe":
            store = SessionStore(tmp_path / "sessions", GatewayConfig())
            append = Mock(wraps=store.append_to_transcript)
            monkeypatch.setattr(store, "append_to_transcript", append)
            adapter._session_store = store
        if mode == "hold":
            adapter._mark_disconnected()
        # Unauthorized inline queries test the real non-disclosing empty-answer path.
        args = dict(kind=kind, edited=mode == "edit", group=mode == "observe", user=99 if mode == "unauthorized" or kind == "inline" else 88)
        if concurrent:
            await asyncio.gather(*(app.process_update(update(app.bot, **args)) for _ in range(2)))
        else:
            for _ in range(2):
                await app.process_update(update(app.bot, **args))
        if mode == "hold":
            assert not delivered and len(adapter._held_inbound_events) == 1
            adapter._mark_connected()
            await adapter._held_inbound_redispatch_task
        tasks = [*adapter._pending_text_batch_tasks.values(), *adapter._pending_photo_batch_tasks.values(), *adapter._media_group_tasks.values()]
        await asyncio.gather(*tasks)
        expected = int(mode not in ("observe", "unauthorized") and kind not in ("callback", "inline", "reaction"))
        assert len(delivered) == expected
        if kind == "text" and expected:
            assert delivered[0].text == "hello"
        if kind in ("photo", "album"):
            assert download.await_count == 1
            assert len(delivered[0].media_urls) == 1
        assert observer.await_count == int(mode == "edit" or kind == "reaction")
        assert len(app.bot.request.answers) == int(kind in ("callback", "inline"))
        assert adapter._updates_dispatched_total == 2
        if mode == "observe":
            assert append.call_count == 1  # before persistence can mask a repeated writer
            source = adapter._build_message_event(update(app.bot, **args).effective_message, MessageType.TEXT).source
            entry = store.get_or_create_session(adapter._telegram_group_observe_shared_source(source))
            rows = store.load_transcript(entry.session_id)
            assert len(rows) == 1 and rows[0]["observed"] is True
        if mode == "edit":
            await app.process_update(update(app.bot, 11, edited=True, text="changed"))
            await app.process_update(update(app.bot, 12, chat=43))
            await asyncio.gather(*adapter._pending_text_batch_tasks.values())
            assert [event.message_id for event in delivered] == ["472"] * 3
            assert [event.text for event in delivered] == ["hello", "changed", "hello"]
            assert observer.await_count == 2
        if mode == "retention":
            now = time.time()
            with monkeypatch.context() as clock:
                clock.setattr(time, "time", lambda: now + 6.5 * 60 * 60)
                await app.process_update(update(app.bot, **args))
            assert len(delivered) == 1  # no TTL: elapsed time alone must not admit replay
            # Accepted-history eviction, not a numeric high-watermark. IDs may decrease after idle.
            for uid in range(100, 4196):
                await app.process_update(Update(update_id=uid))
            assert len(adapter._seen_update_ids) <= 4096
            await app.process_update(update(app.bot, **args))
            assert len(delivered) == 2
        if mode == "owners":
            for profile, bot_id, fresh in (("alpha", 222, 1), ("beta", 333, 1), ("alpha", 222, 0)):
                token = set_hermes_home_override(tmp_path / profile)
                try:
                    async with connected(monkeypatch, bot_id=bot_id) as (other, other_app, other_delivered):
                        await other_app.process_update(update(other_app.bot))
                        await asyncio.gather(*other._pending_text_batch_tasks.values())
                        # A rebuilt adapter for the same bot and home reads that home's receipt.
                        assert len(other_delivered) == fresh
                finally:
                    reset_hermes_home_override(token)
            await app.process_update(update(app.bot))
            assert len(delivered) == 1
            # Same owner can reconnect with a different bot; its update-ID space is independent.
            await adapter.disconnect()
            adapter.config.token = "444:offline-test"
            monkeypatch.setattr(adapter, "_build_ptb_requests", AsyncMock(return_value=(NoNetwork(444), NoNetwork(444))))
            assert await adapter.connect(is_reconnect=True)
            await adapter._app.process_update(update(adapter._app.bot))
            await asyncio.gather(*adapter._pending_text_batch_tasks.values())
            assert len(delivered) == 2


async def _check_sticker_handoff(monkeypatch, adapter, app, delivered, stage):
    from gateway import sticker_cache
    from tools import vision_tools

    entered = asyncio.Event()
    calls = []
    monkeypatch.setattr(sticker_cache, "get_cached_description", lambda _: None)
    monkeypatch.setattr(sticker_cache, "cache_sticker_description", Mock())
    monkeypatch.setattr(Sticker, "get_file", AsyncMock(return_value=SimpleNamespace(
        download_as_bytearray=AsyncMock(return_value=bytearray(b"offline")))))

    async def cache_image(*args, **kwargs):
        if stage == "sticker_prepare" and not entered.is_set():
            entered.set()
            await asyncio.Event().wait()
        return "offline.webp"

    async def vision(**kwargs):
        calls.append(kwargs)
        if stage == "sticker_vision" and not entered.is_set():
            entered.set()
            await asyncio.Event().wait()
        return json.dumps({"success": True, "analysis": "an offline sticker"})

    monkeypatch.setattr("plugins.platforms.telegram.adapter.cache_image_from_bytes_async", cache_image)
    monkeypatch.setattr(vision_tools, "vision_analyze_tool", vision)
    payload = update(app.bot, kind="media").to_dict()
    payload["message"]["sticker"]["is_animated"] = False
    first = asyncio.create_task(app.process_update(Update.de_json(payload, app.bot)))
    try:
        await asyncio.wait_for(entered.wait(), 2)
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
    for _ in range(2):
        await asyncio.wait_for(app.process_update(Update.de_json(payload, app.bot)), 2)
    assert len(calls) == 1
    assert len(delivered) == int(stage == "sticker_prepare")
    assert not adapter._inflight_update_ids


async def _check_caught_preparation(monkeypatch, tmp_path, adapter, app, stage):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from plugins.platforms.telegram import inline_picker
    import sqlite3

    if stage == "observed_prepare":
        adapter.config.extra.update(observe_unmentioned_group_messages=True, require_mention=True,
                                    allowed_chats=["42"], group_allowed_chats=["42"])
        store = adapter._session_store = SessionStore(tmp_path / "sessions", GatewayConfig())
        effect = Mock(wraps=store.append_to_transcript)
        monkeypatch.setattr(store, "append_to_transcript", effect)
        target, method = store, "get_or_create_session"
        args = {"group": True}
    elif stage == "inline_prepare":
        effect = AsyncMock()
        monkeypatch.setattr(InlineQuery, "answer", effect)
        monkeypatch.setattr(inline_picker, "build_inline_results", lambda *a, **kw: ([], ""))
        target, method = inline_picker, "build_inline_results"
        args = {"kind": "inline"}
    else:
        effect = adapter._platform_event_handler = AsyncMock()
        target = adapter
        method = "_normalize_platform_event" if stage == "normalize_prepare" else "_source_for_platform_event_auth"
        args = {"kind": "reaction"}
    with monkeypatch.context() as broken:
        prepare = Mock(side_effect=sqlite3.OperationalError("preparation unavailable"))
        broken.setattr(target, method, prepare)
        await app.process_update(update(app.bot, **args))
        assert prepare.call_count == 1
        assert effect.call_count == 0
    await app.process_update(update(app.bot, **args))
    await app.process_update(update(app.bot, **args))
    assert effect.call_count == 1
    assert not adapter._inflight_update_ids


async def _check_nonblocking_plugin(monkeypatch, adapter, app, stage, failure):
    entered, release = asyncio.Event(), asyncio.Event()
    calls, tasks = [], []

    async def native(incoming, context):
        if incoming.effective_message is None:
            return
        calls.append(incoming.update_id)
        tasks.append(asyncio.current_task())
        entered.set()
        await release.wait()
        raise RuntimeError("plugin failed after effect")

    if stage.endswith("default"):
        monkeypatch.setattr(app.bot, "_defaults", Defaults(block=False))
        # Explicit core block=True must still win over bot defaults.
        for handlers in app.handlers.values():
            for handler in handlers:
                handler.block = True
        handler = TypeHandler(Update, native)
    else:
        handler = TypeHandler(Update, native, block=False)
    if stage.startswith("conversation"):
        handler = ConversationHandler(entry_points=[handler], states={}, fallbacks=[])
    app.add_handler(handler, group=-1)
    monkeypatch.setattr(adapter, "_cache_replied_media", AsyncMock(side_effect=OSError("preparation failed")))
    errors = []

    async def record_error(incoming, context):
        errors.append(context.error)

    app.add_error_handler(record_error, block=True)
    for error_handler in app.error_handlers:
        app.error_handlers[error_handler] = True
    try:
        await asyncio.wait_for(app.process_update(update(app.bot)), 2)
        await asyncio.wait_for(entered.wait(), 2)
        assert any(isinstance(error, OSError) for error in errors)
        # A completed-receipt shortcut would reopen the still-active plugin on eviction.
        for uid in range(100, 4196):
            await app.process_update(Update(uid))
        await app.create_task(asyncio.sleep(0))
        await app.process_update(update(app.bot))
    finally:
        if failure == "cancel":
            for task in tasks:
                task.cancel()
        release.set()
        await app.stop()  # PTB awaits scheduled handlers, including ones not yet entered.
    await app.start()
    await app.process_update(update(app.bot))
    await app.stop()
    assert calls == [10]
    assert not adapter._inflight_update_ids
    assert len(adapter._seen_update_ids) <= 4096


async def _check_error_registration(monkeypatch, adapter, app):
    effects = []

    async def first(incoming, context):
        effects.append((context.bot.id, "first"))
        raise RuntimeError("error handler failed")

    class Recorder:
        async def stop(self, incoming, context):
            effects.append((context.bot.id, "stop"))
            raise ApplicationHandlerStop

    async def last(incoming, context):
        effects.append((context.bot.id, "last"))

    recorder = Recorder()
    async with connected(monkeypatch, bot_id=222) as (other, other_app, _):
        for owner, native in ((adapter, app), (other, other_app)):
            monkeypatch.setattr(owner, "_cache_replied_media", AsyncMock(side_effect=OSError("before enqueue")))
            native.add_error_handler(first, block=True)
            native.add_error_handler(recorder.stop, block=True)
            native.add_error_handler(last, block=True)
            native.add_error_handler(first, block=False)  # duplicate must not alter block/order
            assert native.error_handlers[first] is True
            assert list(native.error_handlers) == [first, recorder.stop, last]
            for _ in range(2):
                await native.process_update(update(native.bot))
            assert not owner._inflight_update_ids
            assert f"{native.bot.id}:10" in owner._seen_update_ids
        assert effects == [(111, "first"), (111, "stop"), (222, "first"), (222, "stop")]
        app.remove_error_handler(recorder.stop)  # fresh bound-method object, same registration
        app.remove_error_handler(recorder.stop)  # removing an absent handler is still a no-op
        assert recorder.stop not in app.error_handlers
        app.add_error_handler(recorder.stop, block=True)
        assert list(app.error_handlers) == [first, last, recorder.stop]
        await app.process_update(update(app.bot, 11))
        assert effects[-3:] == [(111, "first"), (111, "last"), (111, "stop")]
        # Poll/job errors outside update dispatch still run; no synthetic admission is minted.
        before = dict(adapter._seen_update_ids)
        assert await app.process_error(None, OSError("outside update")) is True
        assert adapter._seen_update_ids == before and not adapter._inflight_update_ids


async def _check_error_cancel_before_entry(monkeypatch, adapter, app, delivered):
    scheduled, effects = [], []
    create = app._Application__create_task

    def cancel_on_schedule(coroutine, *args, **kwargs):
        task = create(coroutine, *args, **kwargs)
        if kwargs.get("is_error_handler"):
            scheduled.append((task, coroutine))
            task.cancel()
        return task

    async def notify(incoming, context):
        effects.append(incoming.update_id)

    app.add_error_handler(notify, block=False)
    with monkeypatch.context() as broken:
        broken.setattr(type(app), "_Application__create_task", lambda self, *a, **kw: cancel_on_schedule(*a, **kw))
        broken.setattr(adapter, "_cache_replied_media", AsyncMock(side_effect=OSError("before enqueue")))
        await app.process_update(update(app.bot))
        assert "111:10" in adapter._inflight_update_ids
        await app.process_update(update(app.bot))
        await app.stop()
    assert len(scheduled) == 1 and not effects
    assert not adapter._inflight_update_ids and not adapter._seen_update_ids
    # The cancelled PTB task never awaited the callback coroutine: it must be closed too.
    try:
        assert scheduled[0][1].cr_frame is None
    finally:
        scheduled[0][1].close()
    app.remove_error_handler(notify)
    await app.start()
    await app.process_update(update(app.bot))
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())
    assert len(delivered) == 1


async def _check_native_error_callback(monkeypatch, adapter, app, stage, failure):
    entered, release = asyncio.Event(), asyncio.Event()
    effects, tasks = [], []
    block = stage in ("native_error_block", "native_error_default")
    if stage in ("native_error_block", "native_error_default_async"):
        monkeypatch.setattr(app.bot, "_defaults", Defaults(block=False))
        for handlers in app.handlers.values():
            for handler in handlers:
                handler.block = True

    async def native_error(incoming, context):
        effects.append(incoming.update_id)
        tasks.append(asyncio.current_task())
        entered.set()
        await release.wait()
        if failure == "error":
            raise RuntimeError("notification already sent")

    def factory(native, owner):
        assert native is app and owner is adapter
        if stage in ("native_error_default", "native_error_default_async"):
            native.add_error_handler(native_error)
        else:
            native.add_error_handler(native_error, block=block)

    manager = SimpleNamespace(get_platform_handler_factories=lambda platform: [(factory, "offline-error")])
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    adapter._wire_plugin_handlers(app)
    assert native_error in app.error_handlers
    monkeypatch.setattr(adapter, "_cache_replied_media", AsyncMock(side_effect=OSError("before enqueue")))
    first = asyncio.create_task(app.process_update(update(app.bot)))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.wait_for(app.process_update(update(app.bot)), 2)
        assert effects == [10]
        assert "111:10" in adapter._inflight_update_ids
        assert (tasks[0] is first) == block  # PTB still owns explicit/default block resolution.
        for uid in range(100, 4196):
            await app.process_update(Update(uid))
        await app.process_update(update(app.bot))
        assert effects == [10]
    finally:
        if failure == "cancel":
            for task in tasks:
                task.cancel()
        release.set()
        await asyncio.gather(first, return_exceptions=True)
        await app.stop()
    await app.start()
    await asyncio.wait_for(app.process_update(update(app.bot)), 2)
    await app.stop()
    assert effects == [10]
    assert not adapter._inflight_update_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("stage,failure", [
    *[(stage, failure) for stage in ("prepare", "batch_prepare", "observer", "enqueued", "pressure", "native", "dispatch", "callback", "inline", "media_warning")
      for failure in ("error", "cancel")],
    *[(stage, failure) for stage in ("native_error_block", "native_error_async", "native_error_default", "native_error_default_async")
      for failure in ("error", "cancel", "success")],
    ("error_before_entry", "cancel"), ("error_registration", "error"), ("error_context", "error"),
    ("context", "error"), ("conversation", "error"), ("conversation_owner", "error"), ("ingress", "error"),
    ("sticker_vision", "cancel"), ("sticker_prepare", "cancel"),
    *[(stage, "error") for stage in ("observed_prepare", "inline_prepare", "normalize_prepare", "source_prepare")],
    *[(stage, failure) for stage in ("core_default", "native_async", "native_default", "conversation_async", "conversation_default")
      for failure in ("error", "cancel")],
])
async def test_only_pre_handoff_failure_reopens_admission(monkeypatch, tmp_path, caplog, stage, failure):
    import hermes_cli.lifecycle

    monkeypatch.setattr(hermes_cli.lifecycle, "has_hook", lambda name: True)
    async with connected(monkeypatch) as (adapter, app, delivered):
        if stage == "ingress":
            monkeypatch.setattr(adapter, "_cache_replied_media", AsyncMock(side_effect=OSError("before enqueue")))
            adapter._updates_received_total = 2
            for _ in range(2):
                await app.process_update(update(app.bot))
            assert not adapter._seen_update_ids and not adapter._inflight_update_ids
            assert adapter._updates_dispatched_total == 2
            for _ in range(3):
                adapter._check_ingress_dispatch_stall()
            assert not any("healthy but deaf" in record.message for record in caplog.records)
            return
        if stage.startswith("native_error_"):
            await _check_native_error_callback(monkeypatch, adapter, app, stage, failure)
            return
        if stage == "error_before_entry":
            await _check_error_cancel_before_entry(monkeypatch, adapter, app, delivered)
            return
        if stage == "error_registration":
            await _check_error_registration(monkeypatch, adapter, app)
            return
        if stage == "error_context":
            effect = AsyncMock()
            app.add_error_handler(effect)
            with monkeypatch.context() as broken:
                broken.setattr(adapter, "_cache_replied_media", AsyncMock(side_effect=OSError("before enqueue")))
                broken.setattr(app.context_types.context, "from_error", Mock(side_effect=OSError("no context")))
                await app.process_update(update(app.bot))
            assert not adapter._inflight_update_ids and not adapter._seen_update_ids
            app.remove_error_handler(effect)
            await app.process_update(update(app.bot))
            await asyncio.gather(*adapter._pending_text_batch_tasks.values())
            assert len(delivered) == 1 and effect.await_count == 0
            return
        if stage in ("sticker_vision", "sticker_prepare"):
            await _check_sticker_handoff(monkeypatch, adapter, app, delivered, stage)
            return
        if stage == "core_default":
            monkeypatch.setattr(app.bot, "_defaults", Defaults(block=False))
            for error_handler in app.error_handlers:
                app.error_handlers[error_handler] = True
            entered, release = asyncio.Event(), asyncio.Event()
            preparing = []
            prepare = adapter._cache_replied_media

            async def fail_preparation(*args):
                preparing.append(asyncio.current_task())
                entered.set()
                await release.wait()
                raise OSError("core preparation unavailable")

            monkeypatch.setattr(adapter, "_cache_replied_media", fail_preparation)
            try:
                await asyncio.wait_for(app.process_update(update(app.bot)), 2)
                await asyncio.wait_for(entered.wait(), 2)
                # The no-op observer completes while core preparation remains suspended.
                await app.create_task(asyncio.sleep(0))
                await app.process_update(update(app.bot))
            finally:
                if failure == "cancel":
                    for task in preparing:
                        task.cancel()
                release.set()
                await app.stop()
            assert len(preparing) == 1
            assert not delivered
            monkeypatch.setattr(adapter, "_cache_replied_media", prepare)
            await app.start()
            await app.process_update(update(app.bot))
            await app.process_update(update(app.bot))
            await app.stop()
            await asyncio.gather(*adapter._pending_text_batch_tasks.values())
            assert len(delivered) == 1
            assert not adapter._inflight_update_ids
            return
        if stage in ("observed_prepare", "inline_prepare", "normalize_prepare", "source_prepare"):
            await _check_caught_preparation(monkeypatch, tmp_path, adapter, app, stage)
            return
        if stage in ("native_async", "native_default", "conversation_async", "conversation_default"):
            await _check_nonblocking_plugin(monkeypatch, adapter, app, stage, failure)
            return
        if stage == "conversation_owner":
            effects = []

            async def entry(incoming, context):
                effects.append(context.bot.id)
                raise OSError("plugin failed after handoff")

            conversation = ConversationHandler(entry_points=[TypeHandler(Update, entry)], states={}, fallbacks=[])
            app.add_handler(conversation, group=-1)
            async with connected(monkeypatch, bot_id=222) as (other, other_app, _):
                other_app.add_handler(conversation, group=-1)
                for _ in range(2):
                    await other_app.process_update(update(other_app.bot))
                assert effects == [222]
            return
        if stage == "conversation":
            transitions = []

            async def enter(incoming, context):
                transitions.append("enter")
                return 1

            async def finish(incoming, context):
                transitions.append("finish")
                return ConversationHandler.END

            conversation = ConversationHandler(
                entry_points=[MessageHandler(filters.Regex("^hello$"), enter)],
                states={1: [MessageHandler(filters.Regex("^done$"), finish)]}, fallbacks=[])
            def factory(native, owner):
                assert native is app and owner is adapter
                native.add_handler(conversation, group=-1)

            manager = SimpleNamespace(get_platform_handler_factories=lambda platform: [(factory, "offline-conversation")])
            monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
            adapter._wire_plugin_handlers(app)
            assert -1 in app.handlers  # Factory errors are logged/swallowed, not connection failures.
            assert app.handlers[-1][0] is conversation
            for uid, text in ((10, "hello"), (10, "hello"), (11, "done"), (11, "done"), (12, "hello")):
                await app.process_update(update(app.bot, uid, text=text))
            assert transitions == ["enter", "finish", "enter"]
            app.remove_handler(conversation, group=-1)
            await app.process_update(update(app.bot, 13, text="done"))
            assert transitions == ["enter", "finish", "enter"]
            return
        if stage == "context":
            context_type = app.context_types.context
            with monkeypatch.context() as patch:
                def unavailable_context(*args):
                    raise RuntimeError("context unavailable before any callback")
                patch.setattr(context_type, "from_update", unavailable_context)
                await app.process_update(update(app.bot))
            await app.process_update(update(app.bot))
            await asyncio.gather(*adapter._pending_text_batch_tasks.values())
            assert len(delivered) == 1
            return
        entered, release = asyncio.Event(), asyncio.Event()
        observed = []
        native_calls = []
        args = {"kind": {"dispatch": "command", "callback": "callback", "inline": "inline", "media_warning": "photo"}.get(stage, "text"),
                "edited": stage != "media_warning", "user": 99 if stage == "inline" else 88}

        async def observer(event, source):
            observed.append(event)
            if stage == "observer":
                await fail()

        async def fail(*args):
            entered.set()
            await release.wait()
            if stage not in ("enqueued", "dispatch", "batch_prepare"):
                raise RuntimeError("pre-handoff test failure")

        enqueue = adapter._enqueue_text_event
        batch_key = adapter._text_batch_key

        def key_failure(event):
            raise RuntimeError("no batch key yet") if failure == "error" else asyncio.CancelledError()

        def enqueue_then_fail(event):
            enqueue(event)
            # Inject the failure immediately after the real batch mutation, before callback return.
            raise RuntimeError("accepted test failure") if failure == "error" else asyncio.CancelledError()

        start = adapter._start_session_processing

        def start_then_fail(event, session_key):
            start(event, session_key)
            raise RuntimeError("accepted dispatch failure") if failure == "error" else asyncio.CancelledError()

        async def native(update, context):
            native_calls.append(update.update_id)
            await fail()

        async def effect_then_fail(*args, **kwargs):
            native_calls.append(10)
            await fail()

        # Observe PTB's logger, not a registered plugin callback (itself a handoff).
        adapter._platform_event_handler = observer
        prepare = adapter._cache_replied_media
        if stage in ("prepare", "enqueued", "pressure", "dispatch", "batch_prepare"):
            monkeypatch.setattr(adapter, "_cache_replied_media", fail)
        if stage == "batch_prepare":
            monkeypatch.setattr(adapter, "_text_batch_key", key_failure)
        if stage == "enqueued":
            monkeypatch.setattr(adapter, "_enqueue_text_event", enqueue_then_fail)
        if stage == "native":
            app.add_handler(TypeHandler(Update, native), group=-10)
        if stage == "dispatch":
            monkeypatch.setattr(adapter, "_start_session_processing", start_then_fail)
        if stage == "callback":
            monkeypatch.setattr(adapter, "_handle_exec_approval_callback", effect_then_fail)
        if stage == "inline":
            monkeypatch.setattr(InlineQuery, "answer", effect_then_fail)
        if stage == "media_warning":
            monkeypatch.setattr(PhotoSize, "get_file", AsyncMock(side_effect=OSError("download unavailable")))
            monkeypatch.setattr(Message, "reply_text", effect_then_fail)
        first = asyncio.create_task(app.process_update(update(app.bot, **args)))
        await asyncio.wait_for(entered.wait(), 2)
        if stage == "pressure":
            for uid in range(100, 4196):
                await app.process_update(Update(update_id=uid))
        # Independent SDK objects, same update ID. A duplicate must not run another group.
        try:
            await asyncio.wait_for(app.process_update(update(app.bot, **args)), 2)
        except BaseException:
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            raise
        if failure == "cancel":
            if stage in ("enqueued", "dispatch", "batch_prepare"):
                release.set()
            else:
                first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            release.set()
            await first  # PTB swallows callback exceptions through process_error.
        if stage in ("prepare", "pressure", "batch_prepare"):
            assert not observed
            logged = any(record.name == "telegram.ext.Application" and record.exc_info for record in caplog.records)
            assert logged == (failure == "error")
        monkeypatch.setattr(adapter, "_cache_replied_media", prepare)
        monkeypatch.setattr(adapter, "_enqueue_text_event", enqueue)
        monkeypatch.setattr(adapter, "_text_batch_key", batch_key)
        monkeypatch.setattr(adapter, "_start_session_processing", start)
        release.set()
        adapter._platform_event_handler = AsyncMock()
        await app.process_update(update(app.bot, **args))
        await app.process_update(update(app.bot, **args))
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())
        if stage in ("native", "callback", "inline", "media_warning"):
            assert native_calls == [10]
            return
        assert len(delivered) == 1
        assert delivered[0].text == ("/status" if stage == "dispatch" else "hello")
        assert adapter._platform_event_handler.await_count == (1 if stage in ("prepare", "pressure", "batch_prepare") else 0)


@pytest.mark.asyncio
async def test_redelivery_to_rebuilt_adapter_is_dropped(monkeypatch, tmp_path):
    """The reconnect watcher and a gateway restart both build a new adapter, and a new PTB
    Updater polls from offset 0: Telegram resends every update whose acknowledgement never
    landed. The receipt must outlive the adapter that completed the update."""
    from hermes_constants import get_hermes_home
    from plugins.platforms.telegram.update_admission import RECEIPT_TTL_SECONDS

    receipts = get_hermes_home() / "telegram_update_receipts_111.json"
    async with connected(monkeypatch) as (adapter, app, delivered):
        await app.process_update(update(app.bot, 10))
        with monkeypatch.context() as broken:
            broken.setattr(adapter, "_cache_replied_media", AsyncMock(side_effect=OSError("before enqueue")))
            await app.process_update(update(app.bot, 20, text="unaccepted"))
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())
        assert [event.text for event in delivered] == ["hello"]
    # disconnect() waits for the receipt write; the failed preparation is not a receipt.
    assert set(json.loads(receipts.read_text())["update_ids"]) == {"10"}

    # A fresh adapter, connected the way the gateway reconnect watcher does it.
    async with connected(monkeypatch, is_reconnect=True) as (adapter, app, delivered):
        await app.process_update(update(app.bot, 10))
        await app.process_update(update(app.bot, 20, text="unaccepted"))
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())
        await app.process_update(update(app.bot, 21, edited=True, text="changed"))
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())
        # Old update dropped; the retry of an unaccepted one and an edit of message 472 still run.
        assert [event.text for event in delivered] == ["unaccepted", "changed"]
        assert adapter._updates_dispatched_total == 3

    # Receipts are bot-scoped, and older than Telegram's 24h retention they cannot match.
    async with connected(monkeypatch, bot_id=222) as (adapter, app, delivered):
        await app.process_update(update(app.bot, 10))
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())
        assert len(delivered) == 1
    stale = time.time() - RECEIPT_TTL_SECONDS - 1
    receipts.write_text(json.dumps({"update_ids": {"10": stale, "20": "bad", "x": time.time()}}))
    async with connected(monkeypatch) as (adapter, app, delivered):
        await app.process_update(update(app.bot, 10))
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())
        assert len(delivered) == 1
    assert set(json.loads(receipts.read_text())["update_ids"]) == {"10"}

    receipts.write_text("{not json")
    async with connected(monkeypatch) as (adapter, app, delivered):
        await app.process_update(update(app.bot, 30))
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())
        assert len(delivered) == 1
