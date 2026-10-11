"""Tests for text message batching across all gateway adapters.

When a user sends a long message, the messaging client splits it at the
platform's character limit.  Each adapter should buffer rapid successive
text messages from the same session and aggregate them before dispatching.

Covers: Discord, Matrix, WeCom, and the adaptive delay logic for
Telegram and Feishu.
"""

import asyncio
from dataclasses import replace
from importlib import import_module
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType


# =====================================================================
# Helpers
# =====================================================================

def _make_event(
    text: str,
    platform: Platform,
    chat_id: str = "12345",
    msg_type: MessageType = MessageType.TEXT,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=msg_type,
        source=SessionSource(platform=platform, chat_id=chat_id, chat_type="dm"),
    )


# =====================================================================
# Discord text batching
# =====================================================================

def _make_discord_adapter():
    """Create a minimal DiscordAdapter for testing text batching."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(DiscordAdapter)
    adapter._platform = adapter.platform = Platform.DISCORD
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1  # fast for tests
    adapter._text_batch_split_delay_seconds = 0.3  # fast for tests
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestDiscordTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_discord_adapter()
        event = _make_event("hello world", Platform.DISCORD)

        adapter._enqueue_text_event(event)

        # Not dispatched yet
        adapter.handle_message.assert_not_called()

        # Wait for flush
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        """Two rapid messages from the same chat should be merged."""
        adapter = _make_discord_adapter()

        adapter._enqueue_text_event(_make_event("Part one of a long", Platform.DISCORD))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("message that was split.", Platform.DISCORD))

        adapter.handle_message.assert_not_called()

        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "Part one" in text
        assert "split" in text


# =====================================================================
# Matrix text batching
# =====================================================================

def _make_matrix_adapter():
    """Create a minimal MatrixAdapter for testing text batching."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(MatrixAdapter)
    adapter._platform = adapter.platform = Platform.MATRIX
    adapter.config = config
    adapter._client = None
    adapter._text_batch_intakes = {}
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.0
    adapter._text_batch_split_delay_seconds = 0.0
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestMatrixTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_matrix_adapter()
        event = _make_event("hello world", Platform.MATRIX)

        adapter._enqueue_text_event(event)

        adapter.handle_message.assert_not_called()
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())

        adapter.handle_message.assert_called_once()
        assert adapter.handle_message.call_args[0][0].text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        adapter = _make_matrix_adapter()

        adapter._enqueue_text_event(_make_event("first part", Platform.MATRIX))
        adapter._enqueue_text_event(_make_event("second part", Platform.MATRIX))

        adapter.handle_message.assert_not_called()
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "first part" in text
        assert "second part" in text


# =====================================================================
# WeCom text batching
# =====================================================================

def _make_wecom_adapter():
    """Create a minimal WeComAdapter for testing text batching."""
    from plugins.platforms.wecom.adapter import WeComAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(WeComAdapter)
    adapter._platform = adapter.platform = Platform.WECOM
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.3
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestWeComTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_wecom_adapter()
        event = _make_event("hello world", Platform.WECOM)

        adapter._enqueue_text_event(event)

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        assert adapter.handle_message.call_args[0][0].text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        adapter = _make_wecom_adapter()

        adapter._enqueue_text_event(_make_event("first part", Platform.WECOM))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("second part", Platform.WECOM))

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "first part" in text
        assert "second part" in text


_ADAPTER_TYPES = {
    Platform.DISCORD: ("plugins.platforms.discord.adapter", "DiscordAdapter"),
    Platform.MATRIX: ("plugins.platforms.matrix.adapter", "MatrixAdapter"),
    Platform.WHATSAPP: ("plugins.platforms.whatsapp.adapter", "WhatsAppAdapter"),
    Platform("simplex"): ("plugins.platforms.simplex.adapter", "SimplexAdapter"),
    Platform.WECOM: ("plugins.platforms.wecom.adapter", "WeComAdapter"),
}


def _make_reply_batch_adapter(platform: Platform):
    module, adapter_type = _ADAPTER_TYPES[platform]
    adapter = object.__new__(getattr(import_module(module), adapter_type))
    if platform == Platform.MATRIX:
        adapter._client = None
        adapter._text_batch_intakes = {}
        adapter._buffered_intakes = {}
    adapter._platform = adapter.platform = platform
    adapter.config = PlatformConfig(enabled=True, token="test-token")
    adapter._background_tasks = set()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0
    adapter._text_batch_split_delay_seconds = 0
    adapter._attachment_text_merge_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", tuple(_ADAPTER_TYPES))
@pytest.mark.parametrize(
    "boundary",
    (
        "same", "plain-first", "plain-last", "reply_to_message_id",
        "reply_to_text", "reply_to_author_id", "reply_to_author_name",
        "reply_to_is_own_message", "attachment-caption",
    ),
)
async def test_shared_batches_preserve_reply_context_and_attachment_positions(
    platform, boundary
):
    adapter = _make_reply_batch_adapter(platform)
    first = _make_event("first", platform)
    first.reply_to_message_id = "reply-a"
    first.reply_to_text = "quoted text"
    first.reply_to_author_id = "author-a"
    first.reply_to_author_name = "Author A"
    first.reply_to_is_own_message = False
    first.media_urls = ["/tmp/first.png"]
    first.media_types = ["image/png"]
    first.media_text_inlined = []
    second = replace(
        first, text="second", media_urls=["/tmp/second.txt"],
        media_types=["text/plain"], media_text_inlined=[False],
    )
    if boundary.startswith("reply_to_"):
        setattr(second, boundary, True if boundary == "reply_to_is_own_message" else "other")
    if boundary in {"plain-first", "plain-last"}:
        plain = first if boundary == "plain-first" else second
        plain.reply_to_message_id = plain.reply_to_text = None
        plain.reply_to_author_id = plain.reply_to_author_name = None
        plain.reply_to_is_own_message = False
    if boundary == "attachment-caption":
        first.text = ""
        first.message_type = MessageType.PHOTO
        second.message_type = MessageType.TEXT
        second.reply_to_message_id = "reply-b"
    separated = boundary.startswith("reply_to_") or boundary == "attachment-caption"
    if separated:
        expected = [replace(first), replace(second)]
    else:
        context = second if boundary == "plain-first" else first
        expected = [replace(
            first, text="first\nsecond",
            reply_to_message_id=context.reply_to_message_id,
            reply_to_text=context.reply_to_text,
            reply_to_author_id=context.reply_to_author_id,
            reply_to_author_name=context.reply_to_author_name,
            reply_to_is_own_message=context.reply_to_is_own_message,
            media_urls=first.media_urls + second.media_urls,
            media_types=first.media_types + second.media_types,
            media_text_inlined=[None, False],
        )]

    adapter._enqueue_text_event(first)
    adapter._enqueue_text_event(second)
    await asyncio.wait_for(asyncio.gather(
        *adapter._background_tasks, *adapter._pending_text_batch_tasks.values()
    ), timeout=5)
    actual = [call.args[0] for call in adapter.handle_message.await_args_list]
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", tuple(_ADAPTER_TYPES))
async def test_shared_reply_boundary_preserves_an_in_flight_dispatch(platform):
    adapter = _make_reply_batch_adapter(platform)
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    first = _make_event("first", platform)
    first.reply_to_message_id = "reply-a"
    second = _make_event("second", platform)
    second.reply_to_message_id = "reply-b"
    expected = [replace(first), replace(second)]

    async def dispatch(event):
        if event is first:
            started.set()
            await release.wait()
            finished.set()

    adapter.handle_message.side_effect = dispatch
    try:
        adapter._enqueue_text_event(first)
        adapter._enqueue_text_event(second)
        boundary_tasks = tuple(adapter._background_tasks)
        await asyncio.wait_for(started.wait(), timeout=5)
        for task in boundary_tasks:
            task.cancel()
        await asyncio.wait_for(asyncio.gather(*boundary_tasks), timeout=5)
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=5)
        await asyncio.wait_for(asyncio.gather(
            *adapter._pending_text_batch_tasks.values()
        ), timeout=5)
    finally:
        release.set()
        tasks = (*adapter._background_tasks, *adapter._pending_text_batch_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)
    actual = [call.args[0] for call in adapter.handle_message.await_args_list]
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('route', ['base', 'telegram-photo', 'telegram-album'])
@pytest.mark.parametrize('relation', ['matching', 'first-quote', 'second-quote', 'conflicting', 'conflicting-failure', 'authorization', 'sender', 'control'])
async def test_entry_batch_keeps_each_reply_target(route, relation):
    from dataclasses import replace
    from gateway.platforms.base_pending import pending_part, withdraw_from_event
    from plugins.platforms.telegram.adapter import TelegramAdapter

    delivered = []
    if route == 'base':
        adapter = _make_matrix_adapter()
        maps = [adapter._pending_text_batch_tasks]
        platform = Platform.MATRIX
    else:
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token='1234:dummy'))
        adapter._drop_delayed_deliveries = False
        adapter._media_batch_delay_seconds = 0
        adapter.MEDIA_GROUP_WAIT_SECONDS = 0
        maps = [adapter._pending_photo_batch_tasks, adapter._media_group_tasks]
        platform = Platform.TELEGRAM

    async def capture(event):
        delivered.append(event)
        if relation == 'conflicting-failure' and event.message_id == 'm-one':
            raise ValueError('controlled admission failure')

    adapter.handle_message = capture
    source = SessionSource(platform=platform, chat_id='12345', chat_type='dm', user_id='sender')
    first = MessageEvent(text='one', source=source, message_id='m-one',
                         message_type=MessageType.TEXT if route == 'base' else MessageType.PHOTO)
    second = MessageEvent(text='two', source=source, message_id='m-two',
                          message_type=first.message_type)
    if relation != 'second-quote':
        first.reply_to_message_id, first.reply_to_text = 'q-one', 'quoted one'
    if relation != 'first-quote':
        second.reply_to_message_id, second.reply_to_text = (
            ('q-two', 'quoted two') if relation.startswith('conflicting') else ('q-one', 'quoted one'))
    if route != 'base':
        first.media_urls, first.media_types, first.media_text_inlined = ['one.png'], ['image/png'], [True]
        second.media_urls, second.media_types, second.media_text_inlined = ['two.png'], ['image/png'], [False]
    first.reply_to_author_authorized = second.reply_to_author_authorized = True
    if relation == 'authorization':
        second.reply_to_author_authorized = False
    if relation == 'sender':
        second.source = replace(source, user_id='other-sender')
    if relation == 'control':
        second.allow_gateway_control = False
    originals = [pending_part(event) for event in (first, second)]
    if route == 'base':
        adapter._enqueue_text_event(first)
        adapter._enqueue_text_event(second)
    elif route == 'telegram-photo':
        adapter._enqueue_photo_event('lane', first)
        adapter._enqueue_photo_event('lane', second)
    else:
        await adapter._queue_media_group_event('album', first)
        await adapter._queue_media_group_event('album', second)
    while tasks := [task for pending in maps for task in pending.values()]:
        await asyncio.gather(*tasks, return_exceptions=True)
    held = getattr(adapter, '_held_inbound_redispatch_task', None)
    if held is not None:
        await held
    states = [(event.text, event.reply_context(), event.media_urls,
               event.media_types, event.media_text_inlined) for event in delivered]
    first_media = (['one.png'], ['image/png'], [True]) if route != 'base' else ([], [], [])
    second_media = (['two.png'], ['image/png'], [False]) if route != 'base' else ([], [], [])
    if relation in {'conflicting', 'conflicting-failure', 'authorization', 'sender', 'control'}:
        expected = [('one', originals[0].reply_context(), *first_media),
                    ('two', originals[1].reply_context(), *second_media)]
    else:
        combined_media = (['one.png', 'two.png'], ['image/png', 'image/png'], [True, False]) if route != 'base' else ([], [], [])
        reply = originals[1].reply_context() if relation == 'second-quote' else originals[0].reply_context()
        expected = [('one\ntwo' if route == 'base' else 'one\n\ntwo', reply, *combined_media)]
    assert states == expected
    if len(delivered) == 1:
        matched, remaining = withdraw_from_event(delivered[0], lambda event: event.message_id == 'm-one')
        assert (matched, remaining) == (True, originals[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["buffered", "preparing", "telegram-photo", "telegram-album",
                                   "mixed-pending-text", "mixed-pending-photo", "mixed-ingress-reservation",
                                   "mixed-reservation-ingress", "mixed-pending-reservation", "mixed-fifo-ingress"])
@pytest.mark.parametrize("write_fails", [False, True])
async def test_ingress_shutdown_preserves_split_inputs_before_consumption(tmp_path, monkeypatch, stage, write_fails):
    import json
    from dataclasses import fields

    from plugins.platforms.matrix.adapter import MatrixAdapter
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from gateway.shutdown_pending import PendingQueueSnapshot

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = (TelegramAdapter(PlatformConfig(enabled=True, token="1234:dummy"))
               if stage.startswith("telegram") or stage == "mixed-pending-photo" else MatrixAdapter(PlatformConfig(enabled=True)))
    adapter._text_batch_delay_seconds = 0 if stage == "preparing" else 60
    adapter._media_batch_delay_seconds = adapter.MEDIA_GROUP_WAIT_SECONDS = 60
    adapter._drop_delayed_deliveries = False
    source = SessionSource(platform=adapter.platform, chat_id="12345", chat_type="dm", user_id="sender")
    events = [MessageEvent(text=text, source=source, message_id=f"m-{text}",
                           reply_to_message_id=f"q-{text}", reply_to_text=f"quoted {text}",
                           reply_to_author_id="author", reply_to_author_name="Quoted author",
                           reply_to_is_own_message=False, reply_to_author_authorized=True)
              for text in (("one", "two", "three") if stage == "mixed-fifo-ingress" else ("one", "two"))]
    entered = asyncio.Event()
    consumed = []

    async def prepare(event):
        entered.set()
        await asyncio.Event().wait()
        consumed.append(event)

    adapter.handle_message = prepare
    adapter.set_message_handler(prepare)
    adapter._busy_text_mode = "interrupt"
    key = adapter._event_session_key(events[0])
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.platforms.base_pending import reserve_pending_dispatch
    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig

    async def park(event):
        adapter._active_sessions[key] = asyncio.Event()
        await BasePlatformAdapter.handle_message(adapter, event)

    async def reserve(event):
        reserve_pending_dispatch(adapter, key, event, accepted=False)

    async def text(event):
        adapter._enqueue_text_event(event)

    async def photo(event):
        event.message_type = MessageType.PHOTO
        event.media_urls, event.media_types, event.media_text_inlined = [event.text + ".png"], ["image/png"], [False]
        adapter._enqueue_photo_event("lane", event)

    async def fifo(event):
        assert runner._enqueue_fifo(key, event, adapter)

    mixed = {
        "mixed-pending-text": (park, text),
        "mixed-pending-photo": (park, photo),
        "mixed-ingress-reservation": (text, reserve),
        "mixed-reservation-ingress": (reserve, text),
        "mixed-pending-reservation": (park, reserve),
        "mixed-fifo-ingress": (fifo, fifo, text),
    }
    runner = GatewayRunner(config=GatewayConfig())
    if stage == "mixed-fifo-ingress":
        adapter.gateway_runner = runner
        runner.adapters[adapter.platform] = adapter
    from gateway.shutdown_flush import _write_payload
    if write_fails:
        def fail(directory, payload):
            raise OSError("controlled full disk")
        monkeypatch.setattr("gateway.shutdown_flush._write_payload", fail)
    try:
        for index, event in enumerate(events):
            if stage in mixed:
                await mixed[stage][index](event)
            elif stage.startswith("telegram"):
                event.message_type = MessageType.PHOTO
                event.media_urls, event.media_types, event.media_text_inlined = [event.text + ".png"], ["image/png"], [False]
                if stage == "telegram-photo":
                    adapter._enqueue_photo_event("lane", event)
                else:
                    await adapter._queue_media_group_event("album", event)
            else:
                adapter._enqueue_text_event(event)
        if stage == "preparing":
            await asyncio.wait_for(entered.wait(), 2)
        expected = [{item.name: getattr(event, item.name) for item in fields(event)
                     if item.init and not item.name.startswith("_") and item.name not in {"raw_message", "source"}}
                    for event in events]
        for value, event in zip(expected, events):
            value.update(message_type=event.message_type.value, timestamp=event.timestamp.isoformat(),
                         source={item.name: getattr(source, item.name) for item in fields(source)})
            value["source"]["platform"] = source.platform.value
        await adapter.cancel_background_tasks()
        payloads = [json.loads(path.read_text()) for path in (tmp_path / "pending_messages").glob("*.json")]
        if write_fails:
            assert (payloads, consumed) == ([], [])
            from gateway.shutdown_pending import flush_adapter_pending
            monkeypatch.setattr("gateway.shutdown_flush._write_payload", _write_payload)
            flush_adapter_pending(adapter, {})
            payloads = [json.loads(path.read_text()) for path in (tmp_path / "pending_messages").glob("*.json")]
        snapshot, = [PendingQueueSnapshot.from_payload(payload) for payload in payloads]
        assert (snapshot.session_key, snapshot.runtime_home, [record["event"] for record in snapshot.events], consumed) == (
            adapter._event_session_key(events[0]), str(tmp_path), expected, [])
        if not write_fails:
            assert (adapter._pending_text_batches, getattr(adapter, "_pending_photo_batches", {}),
                    getattr(adapter, "_media_group_events", {}), getattr(adapter, "_held_inbound_events", [])) == ({}, {}, {}, [])
    finally:
        tasks = [task for attr in ("_pending_text_batch_tasks", "_pending_photo_batch_tasks", "_media_group_tasks")
                 for task in getattr(adapter, attr, {}).values()]
        tasks.extend(getattr(adapter, "_background_tasks", ()))
        held = getattr(adapter, "_held_inbound_redispatch_task", None)
        if held is not None:
            tasks.append(held)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
