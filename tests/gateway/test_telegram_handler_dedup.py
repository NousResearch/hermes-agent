"""Behavioral duplicate-delivery tests for registered Telegram handlers."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
)
from plugins.platforms.telegram import adapter as telegram_module
from plugins.platforms.telegram.adapter import TelegramAdapter


class _RecordingApp:
    def __init__(self):
        self.handlers = []

    def add_handler(self, handler, group=0):
        self.handlers.append((handler, group))


class _RegisteredHandler:
    """Small PTB handler stand-in that retains the registered callback."""

    def __init__(self, *args):
        self.callback = args[-1]


def _message(*, message_id=10, text=None, location=None, photo=None, chat_type="private"):
    chat = SimpleNamespace(id=123, type=chat_type, is_forum=False)
    sender = SimpleNamespace(id=456, is_bot=False, username="sender", first_name="Sender")
    return SimpleNamespace(
        message_id=message_id,
        text=text,
        caption=None,
        chat=chat,
        from_user=sender,
        sender_chat=None,
        date=None,
        edit_date=None,
        message_thread_id=None,
        is_topic_message=False,
        location=location,
        venue=None,
        photo=photo,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
        media_group_id=None,
        entities=None,
        caption_entities=None,
        reply_to_message=None,
        quote=None,
        reply_markup=None,
    )


def _update(update_id, message, *, effective_message=None, channel_post=None, edited_channel_post=None):
    if effective_message is None:
        effective_message = message
    return SimpleNamespace(
        update_id=update_id,
        message=message,
        effective_message=effective_message,
        callback_query=None,
        channel_post=channel_post,
        edited_channel_post=edited_channel_post,
    )


def _event_for(message, message_type, update_id=None):
    return SimpleNamespace(
        text=message.text or message.caption or "",
        message_type=message_type,
        source=SimpleNamespace(
            platform=Platform.TELEGRAM,
            chat_id=str(message.chat.id),
            user_id=str(message.from_user.id),
            chat_type="dm",
            profile=None,
            thread_id=None,
            scope_id=None,
            user_id_alt=None,
            prospective_thread_id=None,
        ),
        message_id=str(message.message_id),
        media_urls=[],
        media_types=[],
        platform_update_id=update_id,
        metadata={"telegram_durable_update_ids": [update_id]},
    )


@pytest.fixture
def registered_handler_fixture(monkeypatch):
    if not telegram_module.TELEGRAM_AVAILABLE:
        pytest.skip("python-telegram-bot is not installed")

    adapter = object.__new__(TelegramAdapter)
    adapter.config = PlatformConfig(enabled=True, token="test", extra={})
    adapter._seen_update_ids = {}
    adapter._seen_update_ids_max = 4096
    adapter._seen_platform_update_ids = {}
    adapter._mention_patterns = []
    adapter._forum_lock = __import__("asyncio").Lock()
    adapter._forum_command_registered = set()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._pending_photo_batches = {}
    adapter._pending_photo_batch_tasks = {}
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._drop_delayed_deliveries = False

    enqueue_text = []
    enqueue_photo = []
    observed = []
    dispatched = []
    adapter._is_user_authorized_from_message = lambda _message: True
    adapter._should_process_message = lambda *_args, **_kwargs: True
    adapter._should_observe_unmentioned_group_message = lambda _message: True
    adapter._observe_unmentioned_group_message = lambda *args, **kwargs: observed.append((args, kwargs))
    adapter._ensure_forum_commands = AsyncMock()
    adapter._build_message_event = _event_for
    adapter._clean_bot_trigger_text = lambda text: text
    adapter._cache_replied_media = AsyncMock()
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    adapter._enqueue_text_event = enqueue_text.append
    adapter._enqueue_photo_event = lambda key, event: enqueue_photo.append((key, event))
    adapter._photo_batch_key = lambda event, message: "telegram:photos"
    adapter._should_drop_delayed_delivery = lambda: False
    adapter._media_message_type = lambda _message: MessageType.PHOTO
    adapter._cache_observed_media = AsyncMock()
    adapter._surface_media_cache_failure = AsyncMock()
    adapter._handle_sticker = AsyncMock()

    monkeypatch.setattr(telegram_module, "TelegramMessageHandler", _RegisteredHandler)
    monkeypatch.setattr(telegram_module, "CallbackQueryHandler", _RegisteredHandler)
    monkeypatch.setattr(telegram_module, "TypeHandler", _RegisteredHandler)

    async def dispatch(event):
        dispatched.append(event)

    adapter.handle_message = dispatch

    app = _RecordingApp()
    adapter._register_handlers(app)
    callbacks = {
        handler.callback.__name__: handler.callback
        for handler, _group in app.handlers
        if getattr(handler, "callback", None) is not None
        and hasattr(handler.callback, "__name__")
    }
    return SimpleNamespace(
        adapter=adapter,
        callbacks=callbacks,
        enqueue_text=enqueue_text,
        enqueue_photo=enqueue_photo,
        observed=observed,
        dispatched=dispatched,
    )


async def _make_durable_dispatch_harness(
    tmp_path, update_id, process, *, update_kind="message"
):
    from plugins.platforms.telegram.inbound_store import (
        CaptureDecision,
        DurableTelegramUpdateQueue,
        TelegramInboundStore,
    )

    session_key = "agent:main:telegram:dm:123"
    payload = {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "date": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 456, "is_bot": False, "first_name": "Sender"},
            "text": "durable test",
        },
    }
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")

    def classify(item):
        return CaptureDecision(
            actionable=True,
            account_id="telegram",
            update_kind=update_kind,
            chat_id="123",
            message_id=str(item["message"]["message_id"]),
            session_key=session_key,
            payload=item,
        )

    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=classify,
        lease_owner=f"gateway:durable-test-{update_id}",
        active_limit=1,
    )
    await queue.put(payload)
    queued = await queue.get()
    assert queued["update_id"] == update_id
    claim = queue.claim_for_update(update_id)
    assert claim is not None
    queue.task_done()

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    BasePlatformAdapter.__init__(
        adapter,
        PlatformConfig(enabled=True, token="111:test-token", extra={}),
        Platform.TELEGRAM,
    )
    adapter.config.typing_indicator = False
    adapter._bot = SimpleNamespace(id=111)
    adapter._inbound_queue = queue
    adapter._message_handler = process
    adapter.send = AsyncMock()
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="123",
        user_id="456",
        chat_type="dm",
        profile=None,
        thread_id=None,
        scope_id=None,
        user_id_alt=None,
        prospective_thread_id=None,
    )
    event = MessageEvent(
        text="durable test",
        message_type=MessageType.TEXT,
        source=source,
        raw_message=claim,
        message_id=str(update_id),
        platform_update_id=update_id,
        metadata={
            "telegram_durable_update_ids": [update_id],
            "gateway_session_key": session_key,
        },
    )
    return adapter, queue, store, event


@pytest.mark.asyncio
async def test_registered_text_handler_deduplicates_before_enqueue_or_observe(
    registered_handler_fixture,
):
    fixture = registered_handler_fixture
    callback = fixture.callbacks["_handle_text_message"]
    update = _update(1001, _message(text="hello"))

    await callback(update, None)
    await callback(update, None)

    assert len(fixture.enqueue_text) == 1
    assert fixture.observed == []
    assert fixture.dispatched == []


@pytest.mark.asyncio
async def test_registered_command_handler_deduplicates_immediate_dispatch(
    registered_handler_fixture,
):
    fixture = registered_handler_fixture
    callback = fixture.callbacks["_handle_command"]
    update = _update(1002, _message(text="/stop"))

    await callback(update, None)
    await callback(update, None)

    assert len(fixture.dispatched) == 1
    assert fixture.enqueue_text == []
    assert fixture.observed == []


@pytest.mark.asyncio
async def test_registered_location_handler_deduplicates_dispatch(
    registered_handler_fixture,
):
    fixture = registered_handler_fixture
    callback = fixture.callbacks["_handle_location_message"]
    location = SimpleNamespace(latitude=34.7, longitude=-86.6)
    update = _update(1003, _message(location=location))

    await callback(update, None)
    await callback(update, None)

    assert len(fixture.dispatched) == 1
    assert fixture.enqueue_text == []
    assert fixture.observed == []


@pytest.mark.asyncio
async def test_registered_media_handler_deduplicates_before_download_and_enqueue(
    registered_handler_fixture, monkeypatch
):
    fixture = registered_handler_fixture
    callback = fixture.callbacks["_handle_media_message"]
    calls = {"get_file": 0, "download": 0, "cache": 0}

    class FakeFile:
        file_path = "photo.jpg"

        async def download_as_bytearray(self):
            calls["download"] += 1
            return bytearray(b"image")

    class FakePhoto:
        async def get_file(self):
            calls["get_file"] += 1
            return FakeFile()

    async def cache_image(data, *, ext):
        calls["cache"] += 1
        assert data == b"image"
        assert ext == ".jpg"
        return "/tmp/telegram-test.jpg"

    monkeypatch.setattr(telegram_module, "cache_image_from_bytes_async", cache_image)
    update = _update(1004, _message(photo=[FakePhoto()]))

    await callback(update, None)
    await callback(update, None)

    assert calls == {"get_file": 1, "download": 1, "cache": 1}
    assert len(fixture.enqueue_photo) == 1
    assert fixture.observed == []
    assert fixture.dispatched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("update_field", ["channel_post", "edited_channel_post"])
async def test_registered_media_handler_uses_effective_channel_message(
    registered_handler_fixture, monkeypatch, update_field
):
    fixture = registered_handler_fixture
    callback = fixture.callbacks["_handle_media_message"]
    calls = {"get_file": 0, "download": 0, "cache": 0}

    class FakeFile:
        file_path = "photo.jpg"

        async def download_as_bytearray(self):
            calls["download"] += 1
            return bytearray(b"image")

    class FakePhoto:
        async def get_file(self):
            calls["get_file"] += 1
            return FakeFile()

    async def cache_image(data, *, ext):
        calls["cache"] += 1
        assert data == b"image"
        assert ext == ".jpg"
        return "/tmp/telegram-effective-media.jpg"

    monkeypatch.setattr(telegram_module, "cache_image_from_bytes_async", cache_image)
    message = _message(message_id=1010, photo=[FakePhoto()])
    update = _update(1010, None, effective_message=message, **{update_field: message})

    await callback(update, None)

    assert calls == {"get_file": 1, "download": 1, "cache": 1}
    assert len(fixture.enqueue_photo) == 1
    assert fixture.dispatched == []


@pytest.mark.asyncio
async def test_observation_path_evaluates_should_process_false_without_forced_processing(
    registered_handler_fixture,
):
    fixture = registered_handler_fixture
    fixture.adapter._should_process_message = lambda *_args, **_kwargs: False
    callback = fixture.callbacks["_handle_text_message"]

    await callback(_update(1011, _message(text="unmentioned", chat_type="group")), None)

    assert len(fixture.observed) == 1
    assert fixture.enqueue_text == []
    assert fixture.dispatched == []


@pytest.mark.asyncio
async def test_registered_text_handler_keeps_equal_text_with_distinct_update_ids(
    registered_handler_fixture,
):
    fixture = registered_handler_fixture
    callback = fixture.callbacks["_handle_text_message"]

    await callback(_update(1005, _message(text="same")), None)
    await callback(_update(1006, _message(text="same")), None)

    assert len(fixture.enqueue_text) == 2
    assert fixture.observed == []
    assert fixture.dispatched == []


@pytest.mark.asyncio
async def test_registered_handler_failure_allows_durable_replay_after_requeue(
    registered_handler_fixture,
):
    fixture = registered_handler_fixture
    callback = fixture.callbacks["_handle_text_message"]
    update = _update(1007, _message(text="retry"))

    def fail_once(_event):
        raise RuntimeError("transient handler failure")

    fixture.adapter._enqueue_text_event = fail_once
    with pytest.raises(RuntimeError, match="transient handler failure"):
        await callback(update, None)

    fixture.adapter._enqueue_text_event = fixture.enqueue_text.append
    await callback(update, None)

    assert len(fixture.enqueue_text) == 1


@pytest.mark.asyncio
async def test_inflight_durable_claim_blocks_replay_after_local_cache_eviction(
    registered_handler_fixture,
):
    fixture = registered_handler_fixture
    callback = fixture.callbacks["_handle_text_message"]
    update = _update(1008, _message(text="in flight"))
    fixture.adapter._inbound_queue = SimpleNamespace(
        handler_claimed=lambda _update_id: True,
        mark_handler_claim=lambda _update_id: None,
    )

    await callback(update, None)

    assert fixture.enqueue_text == []


@pytest.mark.asyncio
async def test_delayed_text_does_not_complete_durable_claim_before_dispatch(
    registered_handler_fixture,
):
    fixture = registered_handler_fixture
    adapter = fixture.adapter
    callback = fixture.callbacks["_handle_text_message"]
    completed = []

    class Queue:
        def handler_claimed(self, _update_id):
            return False

        def mark_handler_claim(self, update_id):
            return SimpleNamespace(event_id=f"telegram:111:{update_id}", update_kind="message")

        async def complete_update(self, update_id, *, success):
            completed.append((update_id, success))
            return True

    adapter._inbound_queue = Queue()
    adapter._enqueue_text_event = TelegramAdapter._enqueue_text_event.__get__(adapter)
    adapter._text_batch_delay_seconds = 60.0
    update = _update(1012, _message(text="delayed"))

    await callback(update, None)

    assert completed == []
    assert 1012 in adapter._deferred_inbound_update_ids

    key = next(iter(adapter._pending_text_batch_tasks))
    pending_task = adapter._pending_text_batch_tasks.pop(key)
    pending_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_task
    adapter._text_batch_delay_seconds = 0.0
    await adapter._flush_text_batch(key)
    await asyncio.gather(*tuple(adapter._durable_batch_dispatch_tasks))

    assert completed == [(1012, True)]
    assert 1012 not in adapter._deferred_inbound_update_ids


@pytest.mark.asyncio
async def test_durable_dispatch_returns_before_inherited_background_processing():
    """PTB admission returns promptly; durable completion follows the task boundary."""
    started = asyncio.Event()
    release = asyncio.Event()
    processed = []
    completed = []

    class Queue:
        async def complete_update(self, update_id, *, success):
            completed.append((update_id, success, bool(processed)))
            return True

    class InheritedDispatchAdapter(TelegramAdapter):
        def __init__(self):
            BasePlatformAdapter.__init__(
                self,
                PlatformConfig(enabled=True, token="test", extra={}),
                Platform.TELEGRAM,
            )
            self.config.typing_indicator = False
            self._bot = SimpleNamespace(id=999)
            self._inbound_queue = Queue()

            async def process(event):
                processed.append(event)
                started.set()
                await release.wait()
                return None

            self._message_handler = process

    adapter = InheritedDispatchAdapter()
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="123",
        user_id="456",
        chat_type="dm",
        profile=None,
        thread_id=None,
        scope_id=None,
        user_id_alt=None,
        prospective_thread_id=None,
    )
    event = MessageEvent(
        text="durable background work",
        message_type=MessageType.TEXT,
        source=source,
        message_id="durable-1016",
        platform_update_id=1016,
        metadata={"telegram_durable_update_ids": [1016]},
    )

    dispatch_task = asyncio.create_task(
        adapter._dispatch_and_complete_durable_event(event)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        assert processed == [event]
        await asyncio.wait_for(asyncio.shield(dispatch_task), timeout=0.2)
        assert completed == []
    finally:
        release.set()

    for _ in range(50):
        if completed:
            break
        await asyncio.sleep(0.01)

    assert completed == [(1016, True, True)]


@pytest.mark.asyncio
async def test_cancelling_batch_timer_does_not_cancel_started_durable_dispatch():
    """A new batch window must not cancel the prior event after dispatch starts."""
    adapter = object.__new__(TelegramAdapter)
    event = SimpleNamespace(text="one", metadata={"telegram_durable_update_ids": [1016]})
    adapter._pending_text_batches = {"session": event}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.0
    adapter._text_batch_split_delay_seconds = 0.0
    adapter._should_drop_delayed_delivery = lambda: False
    adapter._hold_inbound_event = lambda _event, **_kwargs: None
    started = asyncio.Event()
    release = asyncio.Event()
    completed = []

    async def dispatch(dispatched):
        started.set()
        await release.wait()
        completed.append(dispatched)

    adapter._dispatch_and_complete_durable_event = dispatch
    flush = asyncio.create_task(adapter._flush_text_batch("session"))
    try:
        await asyncio.wait_for(started.wait(), timeout=0.5)
        flush.cancel()
        await asyncio.gather(flush, return_exceptions=True)
        release.set()
        for _ in range(50):
            if completed:
                break
            await asyncio.sleep(0.01)
        assert completed == [event]
    finally:
        release.set()
        if not flush.done():
            flush.cancel()
            await asyncio.gather(flush, return_exceptions=True)


@pytest.mark.asyncio
async def test_teardown_cancels_and_holds_started_durable_batch_dispatch():
    """Teardown must await detached durable dispatch before returning."""
    adapter = object.__new__(TelegramAdapter)
    event = SimpleNamespace(
        text="one",
        metadata={"telegram_durable_update_ids": [1017]},
    )
    adapter._inbound_queue = SimpleNamespace(_lifecycle_retired=False)
    adapter._media_group_tasks = {}
    adapter._media_group_events = {}
    adapter._pending_photo_batch_tasks = {}
    adapter._pending_photo_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._pending_text_batches = {}
    adapter._polling_error_task = None
    adapter._polling_progress_verifier_task = None
    adapter._held_inbound_redispatch_task = None
    adapter._drop_delayed_deliveries = True
    adapter._is_permanent_fatal = lambda: False
    adapter._inbound_queue_handoff_target = lambda: None
    adapter._mark_durable_event_active = lambda _event: True
    adapter._inbound_session_key = lambda _event: "session"
    adapter._remove_durable_event_from_gateway_fifo = lambda *_args: False
    held = []
    adapter._hold_inbound_event = lambda held_event, *, where: held.append(
        (held_event, where)
    )
    started = asyncio.Event()

    async def dispatch(_event):
        started.set()
        await asyncio.Event().wait()

    adapter._dispatch_inbound_event = dispatch
    task = adapter._start_batched_durable_dispatch(event)
    try:
        await asyncio.wait_for(started.wait(), timeout=0.5)
        await asyncio.wait_for(adapter._cancel_pending_delivery_tasks(), timeout=0.5)

        assert task.done()
        assert task.cancelled()
        assert held == [(event, "durable-dispatch-cancelled")]
        assert adapter._durable_batch_dispatch_tasks == set()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_durable_background_failure_requeues_claim(tmp_path):
    """A contained BasePlatformAdapter failure must leave the row replayable."""
    started = asyncio.Event()

    async def process(_event):
        started.set()
        raise RuntimeError("background failure")

    adapter, queue, store, event = await _make_durable_dispatch_harness(
        tmp_path, 1017, process
    )

    await asyncio.wait_for(
        adapter._dispatch_and_complete_durable_event(event), timeout=1.0
    )
    completion = getattr(event, "_telegram_durable_completion_task", None)
    assert completion is not None
    await asyncio.wait_for(asyncio.shield(completion), timeout=1.0)

    assert started.is_set()
    row = store.get("telegram:111:1017")
    assert row is not None
    assert row.work_state == "queued"
    assert row.dispatch_state == "pending"
    assert row.lease_owner is None
    assert row.consumed_at is None
    assert row.last_error_class == "handler_failed"
    assert queue.claim_for_update(1017) is None


@pytest.mark.asyncio
async def test_plugin_handlers_are_durable_wrapped_before_core_process_update(
    tmp_path, monkeypatch
):
    """PTB plugin precedence must not bypass the durable inbound wrapper."""
    del tmp_path, monkeypatch
    if not telegram_module.TELEGRAM_AVAILABLE:
        pytest.skip("python-telegram-bot is not installed")

    import json
    import os
    import sys
    import textwrap

    probe = textwrap.dedent(
        """
        import asyncio
        import json
        import sys
        import tempfile
        from pathlib import Path
        from types import ModuleType, SimpleNamespace

        class _Filter:
            def __and__(self, _other):
                return self

            def __or__(self, _other):
                return self

            def __invert__(self):
                return self

        filter_value = _Filter()
        filters = SimpleNamespace(
            TEXT=filter_value,
            COMMAND=filter_value,
            LOCATION=filter_value,
            VENUE=filter_value,
            PHOTO=filter_value,
            VIDEO=filter_value,
            AUDIO=filter_value,
            VOICE=filter_value,
            Document=SimpleNamespace(ALL=filter_value),
            Sticker=SimpleNamespace(ALL=filter_value),
        )

        class BaseHandler:
            def __init__(self, *args):
                self.callback = args[-1]

            def check_update(self, _update):
                return True

        class MessageHandler(BaseHandler):
            pass

        class CallbackQueryHandler(BaseHandler):
            pass

        class InlineQueryHandler(BaseHandler):
            pass

        class TypeHandler(BaseHandler):
            def check_update(self, _update):
                return False

        class ConversationHandler(BaseHandler):
            pass

        class Update:
            ALL_TYPES = []

            def __init__(self, update_id):
                self.update_id = update_id

            @classmethod
            def de_json(cls, payload, _bot):
                return cls(payload["update_id"])

        class _ApplicationBuilder:
            def token(self, _token):
                return self

            def build(self):
                return Application()

        class Application:
            def __init__(self):
                self.handlers = {}
                self.bot = SimpleNamespace(id=111)
                self._error_handler = None

            @classmethod
            def builder(cls):
                return _ApplicationBuilder()

            def add_handler(self, handler, group=0):
                self.handlers.setdefault(group, []).append(handler)

            def add_error_handler(self, callback):
                self._error_handler = callback

            async def process_update(self, update):
                for group in sorted(self.handlers):
                    for handler in self.handlers[group]:
                        if not handler.check_update(update):
                            continue
                        try:
                            await handler.callback(update, None)
                        except BaseException as exc:
                            if self._error_handler is None:
                                raise
                            await self._error_handler(
                                update, SimpleNamespace(error=exc)
                            )
                        break

        telegram_module = ModuleType("telegram")
        telegram_module.__path__ = []
        telegram_module.Update = Update
        telegram_module.Bot = object
        telegram_module.Message = object
        telegram_module.InlineKeyboardButton = object
        telegram_module.InlineKeyboardMarkup = object
        telegram_module.LinkPreviewOptions = object

        ext_module = ModuleType("telegram.ext")
        ext_module.__path__ = []
        ext_module.Application = Application
        ext_module.CommandHandler = MessageHandler
        ext_module.CallbackQueryHandler = CallbackQueryHandler
        ext_module.InlineQueryHandler = InlineQueryHandler
        ext_module.MessageHandler = MessageHandler
        ext_module.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
        ext_module.TypeHandler = TypeHandler
        ext_module.filters = filters

        constants_module = ModuleType("telegram.constants")
        constants_module.ParseMode = SimpleNamespace()
        constants_module.ChatType = SimpleNamespace()
        request_module = ModuleType("telegram.request")
        request_module.HTTPXRequest = object
        handlers_module = ModuleType("telegram.ext._handlers")
        handlers_module.__path__ = []
        base_module = ModuleType("telegram.ext._handlers.basehandler")
        base_module.BaseHandler = BaseHandler
        conversation_module = ModuleType(
            "telegram.ext._handlers.conversationhandler"
        )
        conversation_module.ConversationHandler = ConversationHandler
        sys.modules.update(
            {
                "telegram": telegram_module,
                "telegram.ext": ext_module,
                "telegram.constants": constants_module,
                "telegram.request": request_module,
                "telegram.ext._handlers": handlers_module,
                "telegram.ext._handlers.basehandler": base_module,
                "telegram.ext._handlers.conversationhandler": conversation_module,
            }
        )

        from telegram import Update
        from telegram.ext import Application, MessageHandler, filters

        import hermes_cli.plugins as plugins_module
        from gateway.config import Platform, PlatformConfig
        from gateway.platforms.base import BasePlatformAdapter
        from plugins.platforms.telegram.adapter import TelegramAdapter
        from plugins.platforms.telegram.inbound_store import (
            CaptureDecision,
            DurableTelegramUpdateQueue,
            TelegramInboundStore,
        )

        async def main():
            with tempfile.TemporaryDirectory() as directory:
                store = TelegramInboundStore(Path(directory) / "telegram.db")
                payload = {
                    "update_id": 1040,
                    "message": {
                        "message_id": 1040,
                        "date": 1,
                        "chat": {"id": 123, "type": "private"},
                        "from": {"id": 456, "is_bot": False, "first_name": "Sender"},
                        "text": "plugin dispatch",
                    },
                }

                def classify(item):
                    return CaptureDecision(
                        actionable=True,
                        account_id="telegram",
                        update_kind="message",
                        chat_id="123",
                        message_id="1040",
                        session_key="agent:main:telegram:dm:123",
                        payload=item,
                    )

                queue = DurableTelegramUpdateQueue(
                    store=store,
                    bot_account_id=111,
                    classifier=classify,
                    lease_owner="gateway:plugin-probe",
                    active_limit=1,
                )
                await queue.put(payload)
                queued = await queue.get()
                assert queued["update_id"] == 1040
                assert queue.claim_for_update(1040) is not None
                queue.task_done()

                adapter = TelegramAdapter.__new__(TelegramAdapter)
                BasePlatformAdapter.__init__(
                    adapter,
                    PlatformConfig(enabled=True, token="111:test-token", extra={}),
                    Platform.TELEGRAM,
                )
                adapter.config.typing_indicator = False
                adapter._inbound_queue = queue
                adapter._bot_account_id = "111"
                adapter._message_handler = lambda _event: None
                plugin_calls = []
                core_calls = []
                factory_calls = []
                errors = []

                async def plugin_callback(update, _context):
                    plugin_calls.append(update.update_id)

                async def core_callback(update, _context):
                    core_calls.append(update.update_id)

                async def error_callback(_update, context):
                    errors.append(f"{type(context.error).__name__}: {context.error}")

                adapter._handle_text_message = core_callback
                app = Application.builder().token("111:test-token").build()
                app.add_error_handler(error_callback)
                adapter._bot = app.bot

                def factory(native, _adapter):
                    factory_calls.append(type(native).__name__)
                    assert type(native).__name__ == "_TelegramPluginApplicationProxy"
                    native.add_handler(
                        MessageHandler(filters.TEXT & ~filters.COMMAND, plugin_callback)
                    )

                class PluginManager:
                    def get_platform_handler_factories(self, platform_name):
                        assert platform_name == "telegram"
                        return [(factory, "durable-plugin-test")]

                plugins_module.get_plugin_manager = lambda: PluginManager()
                adapter._wire_plugin_handlers(
                    adapter._telegram_plugin_application_proxy(app)
                )
                adapter._register_handlers(app)
                update = Update.de_json(payload, app.bot)
                checks = [
                    repr(handler.check_update(update))
                    for handler in app.handlers.get(0, [])
                ]
                app._initialized = True
                await app.process_update(update)
                await app.process_update(update)

                row = store.get("telegram:111:1040")
                plugin_handler = app.handlers[0][0]
                print(json.dumps({
                    "plugin_calls": plugin_calls,
                    "core_calls": core_calls,
                    "factory_calls": factory_calls,
                    "handler_types": [
                        type(handler).__name__
                        for handler in app.handlers.get(0, [])
                    ],
                    "handler_markers": [
                        bool(getattr(handler.callback, "_hermes_telegram_durable_wrapper", False))
                        for handler in app.handlers.get(0, [])
                        if hasattr(handler, "callback")
                    ],
                    "checks": checks,
                    "errors": errors,
                    "work_state": row.work_state if row else None,
                    "wrapped": (
                        getattr(plugin_handler.callback, "__wrapped__", None)
                        is plugin_callback
                    ),
                    "wrapped_once": adapter._wrap_inbound_handler(
                        plugin_handler.callback
                    ) is plugin_handler.callback,
                }))

        asyncio.run(main())
        """
    )
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - the child runs a fresh interpreter
        try:
            os.close(read_fd)
            os.dup2(write_fd, 1)
            os.dup2(write_fd, 2)
            os.close(write_fd)
            os.execve(
                sys.executable,
                [sys.executable, "-c", probe],
                os.environ.copy(),
            )
        finally:
            os._exit(127)
    os.close(write_fd)
    _, wait_status = os.waitpid(child_pid, 0)
    output = os.read(read_fd, 1 << 20).decode("utf-8", errors="replace")
    os.close(read_fd)
    return_code = os.waitstatus_to_exitcode(wait_status)
    assert return_code == 0, output
    result = json.loads(output.strip().splitlines()[-1])
    assert result["plugin_calls"] == [1040]
    assert result["core_calls"] == []
    assert result["factory_calls"] == ["_TelegramPluginApplicationProxy"]
    assert result["handler_types"][0] == "MessageHandler"
    assert result["handler_markers"]
    assert all(result["handler_markers"])
    assert result["checks"][0] == "True"
    assert result["errors"] == []
    assert result["work_state"] == "consumed"
    assert result["wrapped"] is True
    assert result["wrapped_once"] is True


@pytest.mark.asyncio
async def test_transient_init_rebuild_rewires_plugin_handlers_through_durable_proxy(
    tmp_path, monkeypatch
):
    """A rebuilt PTB application must not restore unwrapped plugin callbacks."""
    if not telegram_module.TELEGRAM_AVAILABLE:
        pytest.skip("python-telegram-bot is not installed")

    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="111:test-token", extra={})
    )
    first_app = MagicMock()
    first_app.bot = MagicMock()
    first_app.initialize = MagicMock(side_effect=OSError("transient"))
    rebuilt_app = MagicMock()
    rebuilt_app.bot = MagicMock()
    rebuilt_app.initialize = MagicMock(side_effect=RuntimeError("stop after rebuild"))

    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.update_queue.return_value = builder
    builder.build.side_effect = [first_app, rebuilt_app]

    async def no_sleep(_delay):
        return None

    async def no_fallback_ips():
        return []

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        telegram_module,
        "Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )
    monkeypatch.setattr(
        telegram_module, "HTTPXRequest", lambda **_kwargs: MagicMock()
    )
    monkeypatch.setattr(telegram_module, "discover_fallback_ips", no_fallback_ips)
    monkeypatch.setattr(telegram_module, "resolve_proxy_url", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(telegram_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        telegram_module, "_shutdown_abandoned_app", AsyncMock()
    )
    adapter._fallback_ips = MagicMock(return_value=[])
    adapter._instrument_polling_request = MagicMock(side_effect=lambda request: request)
    adapter._acquire_platform_lock = MagicMock(return_value=True)
    adapter._release_platform_lock = MagicMock()
    adapter._ensure_inbound_queue = MagicMock(return_value=MagicMock())
    adapter._recover_inbound_queue = AsyncMock()
    adapter._register_handlers = MagicMock()
    adapter._wire_plugin_handlers = MagicMock()
    adapter._looks_like_network_error = MagicMock(return_value=False)

    assert await adapter.connect() is False

    proxies = [call.args[0] for call in adapter._wire_plugin_handlers.call_args_list]
    assert [proxy._application for proxy in proxies] == [first_app, rebuilt_app]
    assert all(
        type(proxy).__name__ == "_TelegramPluginApplicationProxy"
        for proxy in proxies
    )
    assert adapter._register_handlers.call_args_list == [
        ((first_app,), {}),
        ((rebuilt_app,), {}),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("update_kind", ["command", "callback_query"])
async def test_busy_cap_defer_requeues_control_updates_without_using_budget(
    tmp_path, monkeypatch, update_kind
):
    """A not-admitted control callback must return to the due-time queue."""
    monkeypatch.setattr(telegram_module, "_DURABLE_BUSY_RETRY_DELAY_SECONDS", 0.01)

    async def deferred(_update, _context):
        return telegram_module._DURABLE_DISPATCH_DEFERRED

    update_id = 1041 if update_kind == "command" else 1042
    adapter, queue, store, _event = await _make_durable_dispatch_harness(
        tmp_path, update_id, lambda _event: None, update_kind=update_kind
    )
    wrapped = adapter._wrap_inbound_handler(deferred)
    update = SimpleNamespace(
        update_id=update_id,
        message=SimpleNamespace(text="/busy" if update_kind == "command" else None),
        effective_message=None,
        callback_query=(SimpleNamespace() if update_kind == "callback_query" else None),
    )

    assert await wrapped(update, None) is telegram_module._DURABLE_DISPATCH_DEFERRED

    row = store.get(f"telegram:111:{update_id}")
    assert row is not None
    assert row.work_state == "queued"
    assert row.dispatch_state == "pending"
    assert row.attempt_count == 0
    assert row.terminal_reason is None
    assert row.last_error_class == "busy_cap"
    assert store.next_pending_dispatch_at(bot_account_id=111) is not None
    assert queue.claim_for_update(update_id) is None

    replay = await asyncio.wait_for(queue.get(), timeout=0.5)
    assert replay["update_id"] == update_id
    replay_claim = queue.claim_for_update(update_id)
    assert replay_claim is not None
    assert replay_claim.attempt_count == 1
    queue.task_done()
    assert await queue.complete_update(update_id, success=True)


@pytest.mark.asyncio
async def test_started_callback_failure_remains_quarantined(tmp_path):
    """A callback that was admitted before failing must not be replayed."""

    async def failed(_update, _context):
        raise RuntimeError("callback effect failed")

    adapter, queue, store, _event = await _make_durable_dispatch_harness(
        tmp_path, 1043, lambda _event: None, update_kind="callback_query"
    )
    wrapped = adapter._wrap_inbound_handler(failed)
    update = SimpleNamespace(
        update_id=1043,
        message=None,
        effective_message=None,
        callback_query=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="callback effect failed"):
        await wrapped(update, None)

    row = store.get("telegram:111:1043")
    assert row is not None
    assert row.work_state == "dead_letter"
    assert row.terminal_reason == "control_effect_failed"
    assert row.dispatch_state == "pending"
    assert row.attempt_count == 1
    assert queue.claim_for_update(1043) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("callback_name", "update_kind"),
    [
        pytest.param("_handle_command", "command", id="command"),
        pytest.param("_handle_location_message", "location", id="location"),
        pytest.param("_handle_media_message", "message", id="sticker-media"),
    ],
)
async def test_registered_immediate_handler_failure_does_not_ack_real_claim(
    tmp_path, monkeypatch, callback_name, update_kind
):
    """Registered immediate handlers must not ACK contained owner failures."""
    if not telegram_module.TELEGRAM_AVAILABLE:
        pytest.skip("python-telegram-bot is not installed")

    started = asyncio.Event()

    async def process(_event):
        started.set()
        raise RuntimeError("registered immediate handler failure")

    adapter, queue, store, event = await _make_durable_dispatch_harness(
        tmp_path, 1020, process, update_kind=update_kind
    )
    event.message_type = {
        "_handle_command": MessageType.COMMAND,
        "_handle_location_message": MessageType.LOCATION,
        "_handle_media_message": MessageType.STICKER,
    }[callback_name]
    adapter._is_user_authorized_from_message = lambda message: True
    adapter._should_process_message = lambda *_args, **_kwargs: True
    adapter._ensure_forum_commands = AsyncMock()
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    adapter._build_message_event = lambda *_args, **_kwargs: event
    adapter._clean_bot_trigger_text = lambda text: text
    adapter._cache_replied_media = AsyncMock()
    adapter._handle_sticker = AsyncMock()

    if callback_name == "_handle_command":
        message = SimpleNamespace(text="/probe")
    elif callback_name == "_handle_location_message":
        message = SimpleNamespace(
            location=SimpleNamespace(latitude=34.7, longitude=-86.6),
            venue=None,
        )
    else:
        message = SimpleNamespace(sticker=SimpleNamespace(), caption=None)

    monkeypatch.setattr(telegram_module, "TelegramMessageHandler", _RegisteredHandler)
    monkeypatch.setattr(telegram_module, "CallbackQueryHandler", _RegisteredHandler)
    monkeypatch.setattr(telegram_module, "TypeHandler", _RegisteredHandler)
    app = _RecordingApp()
    adapter._register_handlers(app)
    callbacks = {
        handler.callback.__name__: handler.callback
        for handler, _group in app.handlers
        if getattr(handler, "callback", None) is not None
        and hasattr(handler.callback, "__name__")
    }

    update = SimpleNamespace(
        update_id=1020,
        message=message,
        effective_message=None,
    )
    await callbacks[callback_name](update, None)
    await asyncio.wait_for(started.wait(), timeout=1.0)
    completion = getattr(event, "_telegram_durable_completion_task", None)
    assert completion is not None
    await asyncio.wait_for(asyncio.shield(completion), timeout=1.0)

    assert started.is_set()
    row = store.get(event.raw_message.event_id)
    assert row is not None
    assert row.work_state == "queued"
    assert row.terminal_reason is None
    assert row.dispatch_state == "pending"
    assert row.lease_owner is None
    assert row.consumed_at is None
    assert row.last_error_class == "handler_failed"
    assert queue.claim_for_update(1020) is None
    assert ("111", 1020) not in adapter._seen_update_ids
    await queue.suspend_projection()


@pytest.mark.asyncio
async def test_expected_durable_background_cancellation_requeues_claim(tmp_path):
    """Expected owner cancellation must requeue, without holding a duplicate."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def process(_event):
        started.set()
        await release.wait()

    adapter, queue, store, event = await _make_durable_dispatch_harness(
        tmp_path, 1018, process
    )
    dispatch_task = asyncio.create_task(
        adapter._dispatch_and_complete_durable_event(event)
    )

    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        owner = adapter._session_tasks.get("agent:main:telegram:dm:123")
        assert owner is not None
        adapter._expected_cancelled_tasks.add(owner)
        owner.cancel()
        await asyncio.wait_for(dispatch_task, timeout=1.0)
        completion = getattr(event, "_telegram_durable_completion_task", None)
        assert completion is not None
        await asyncio.wait_for(asyncio.shield(completion), timeout=1.0)
    finally:
        release.set()
        if not dispatch_task.done():
            dispatch_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await dispatch_task

    row = store.get("telegram:111:1018")
    assert row is not None
    assert row.work_state == "queued"
    assert row.dispatch_state == "pending"
    assert row.lease_owner is None
    assert row.consumed_at is None
    assert row.last_error_class == "handler_failed"
    assert row.attempt_count == 1
    assert queue.claim_for_update(1018) is None
    assert not getattr(adapter, "_held_inbound_events", [])
    await queue.suspend_projection()


@pytest.mark.asyncio
async def test_control_fence_failure_allows_same_process_durable_retry(
    registered_handler_fixture,
):
    """A stale control fence must not poison the process-local dedup cache."""
    fixture = registered_handler_fixture
    adapter = fixture.adapter
    completed = []

    class Queue:
        def handler_claimed(self, _update_id):
            return False

        def mark_handler_claim(self, update_id):
            return SimpleNamespace(
                event_id=f"telegram:111:{update_id}",
                update_kind="callback_query",
            )

        def mark_control_started(self, _update_id):
            return False

        async def complete_update(self, update_id, *, success):
            completed.append((update_id, success))
            return True

    adapter._inbound_queue = Queue()
    callback = fixture.callbacks["_handle_command"]
    update = _update(1019, _message(text="/stop"))

    await callback(update, None)
    await callback(update, None)

    assert completed == [(1019, False), (1019, False)]


def test_deduplication_is_serialized_and_lazily_initialized(registered_handler_fixture):
    adapter = registered_handler_fixture.adapter
    update = _update(1013, _message(text="race"))

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(adapter._is_duplicate_update, [update] * 50))

    assert results.count(False) == 1
    assert results.count(True) == 49


def test_uninitialized_adapter_dedup_state_is_safe():
    adapter = object.__new__(TelegramAdapter)
    adapter._bot = SimpleNamespace(id=111)
    update = _update(1014, _message(text="lazy"))

    assert adapter._is_duplicate_update(update) is False
    assert adapter._is_duplicate_update(update) is True


@pytest.mark.asyncio
async def test_near_limit_command_uses_real_text_batch_flush(
    registered_handler_fixture,
):
    fixture = registered_handler_fixture
    adapter = fixture.adapter
    callback = fixture.callbacks["_handle_command"]
    adapter._enqueue_text_event = TelegramAdapter._enqueue_text_event.__get__(adapter)
    adapter._text_batch_delay_seconds = 0.0
    adapter._text_batch_split_delay_seconds = 0.0
    text = "/queue " + ("x" * (adapter._SPLIT_THRESHOLD - len("/queue ")))

    await callback(_update(1015, _message(text=text)), None)
    tasks = list(adapter._pending_text_batch_tasks.values())
    assert tasks
    await asyncio.gather(*tasks)

    assert len(fixture.dispatched) == 1
    assert fixture.dispatched[0].text == text


@pytest.mark.asyncio
async def test_processing_hooks_tolerate_sparse_adapter_without_bot(monkeypatch):
    """Lifecycle hooks must remain safe before the Telegram bot is attached."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = object.__new__(TelegramAdapter)
    event = SimpleNamespace(
        source=SimpleNamespace(chat_id="123"),
        message_id="10",
    )

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert event._telegram_processing_started is True
    assert event._telegram_processing_outcome is ProcessingOutcome.SUCCESS


def test_inline_picker_is_registered_through_durable_inbound_fence(monkeypatch):
    """Inline-query ingress must claim the durable record before dispatch."""
    if not telegram_module.TELEGRAM_AVAILABLE:
        pytest.skip("python-telegram-bot is not installed")

    adapter = object.__new__(TelegramAdapter)
    wrapped = []

    async def handle_text(*_args, **_kwargs):
        return None

    async def handle_command(*_args, **_kwargs):
        return None

    async def handle_location(*_args, **_kwargs):
        return None

    async def handle_media(*_args, **_kwargs):
        return None

    async def handle_callback(*_args, **_kwargs):
        return None

    async def handle_inline(*_args, **_kwargs):
        return None

    async def handle_platform_event(*_args, **_kwargs):
        return None

    def wrap(callback):
        wrapped.append(callback.__name__)
        return callback

    adapter._wrap_inbound_handler = wrap
    adapter._wrap_platform_event_handler = lambda callback: callback
    adapter._handle_text_message = handle_text
    adapter._handle_command = handle_command
    adapter._handle_location_message = handle_location
    adapter._handle_media_message = handle_media
    adapter._handle_callback_query = handle_callback
    adapter._handle_inline_query = handle_inline
    adapter._on_platform_update = handle_platform_event

    monkeypatch.setattr(telegram_module, "TelegramMessageHandler", _RegisteredHandler)
    monkeypatch.setattr(telegram_module, "CallbackQueryHandler", _RegisteredHandler)
    monkeypatch.setattr(telegram_module, "InlineQueryHandler", _RegisteredHandler)
    monkeypatch.setattr(telegram_module, "TypeHandler", _RegisteredHandler)

    TelegramAdapter._register_handlers(adapter, _RecordingApp())

    assert wrapped == [
        "handle_text",
        "handle_command",
        "handle_location",
        "handle_media",
        "handle_callback",
        "handle_inline",
    ]
