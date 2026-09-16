"""Supervisor consult replies must not dump into Discord text by default.

The realtime voice model already speaks the summary in the VC. The bound
text channel keeps session history and media, but the full agent reply is
not posted unless ``voice.realtime.discord_text_mirror`` is on.
"""

import asyncio
import logging
from queue import Queue
from types import SimpleNamespace
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource, build_session_key


class _DummyAdapter(BasePlatformAdapter):
    def __init__(self, platform: Platform = Platform.DISCORD):
        super().__init__(
            PlatformConfig(enabled=True, token="fake-token", typing_indicator=False),
            platform,
        )
        self.sent = []
        self.documents = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="1")

    async def send_document(
        self, chat_id, file_path, caption=None, file_name=None, reply_to=None,
        metadata=None, **kwargs,
    ) -> SendResult:
        self.documents.append({"chat_id": chat_id, "file_path": file_path})
        return SendResult(success=True, message_id="doc-1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _hold_typing():
    async def hold(*_args, **_kwargs):
        await asyncio.Event().wait()

    return hold


def _make_event() -> MessageEvent:
    return MessageEvent(
        text="list the repos",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="900",
            chat_type="channel",
        ),
        message_id="consult-1",
    )


async def _deliver(adapter: _DummyAdapter, event: MessageEvent, reply: str):
    # Belt-and-suspenders: even if typing_indicator is False, a custom
    # _keep_typing must not spin forever if a future change re-enables it.
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda _chat_id: False
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result=reply))
    await adapter._process_message_background(event, build_session_key(event.source))
    return adapter


@pytest.mark.asyncio
async def test_consumed_consult_skips_text_send():
    adapter = _DummyAdapter()
    event = _make_event()
    event.voice_reply_consumed = True
    await _deliver(adapter, event, "x" * 400)
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_consumed_consult_does_not_log_delivery_dropped(caplog):
    adapter = _DummyAdapter()
    event = _make_event()
    event.voice_reply_consumed = True
    with caplog.at_level(logging.ERROR):
        await _deliver(adapter, event, "x" * 400)
    assert not any("response_delivery_dropped" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_unconsumed_reply_still_sends_text():
    adapter = _DummyAdapter()
    event = _make_event()
    await _deliver(adapter, event, "hello from hermes")
    assert adapter.sent and adapter.sent[0]["content"] == "hello from hermes"


@pytest.mark.asyncio
async def test_text_mirror_opt_in_sends_consumed_reply():
    adapter = _DummyAdapter()
    event = _make_event()
    event.voice_reply_consumed = True
    event.voice_text_mirror = True
    await _deliver(adapter, event, "mirrored consult reply")
    assert adapter.sent and adapter.sent[0]["content"] == "mirrored consult reply"


@pytest.mark.asyncio
async def test_consumed_consult_still_delivers_media():
    adapter = _DummyAdapter()
    event = _make_event()
    event.voice_reply_consumed = True
    adapter.extract_media = (
        lambda content: ([("/tmp/consult-note.pdf", False)], "summary text")
    )
    adapter.filter_media_delivery_paths = lambda files, session_key="": files
    await _deliver(adapter, event, "summary text\nMEDIA:/tmp/consult-note.pdf")
    assert adapter.sent == []
    assert adapter.documents and adapter.documents[0]["file_path"] == "/tmp/consult-note.pdf"


def test_consult_flags_survive_dispatch_hook_rewrite():
    """A plugin ``pre_gateway_dispatch`` rewrite rebuilds the event with ``dataclasses.replace``;
    the consult contract must ride along or the turn silently degrades to a typed message."""
    import dataclasses

    consult = _make_event()
    consult.voice_consult = True
    consult.voice_text_mirror = True
    rewritten = dataclasses.replace(consult, text="rewritten by plugin")
    assert rewritten.voice_consult is True
    assert rewritten.voice_text_mirror is True
    assert MessageEvent(text="typed").voice_consult is False


def test_consult_coalesced_with_typed_message_keeps_both_contracts():
    """The pending slot holds one event per session. When a consult and a typed message from the
    same sender coalesce (either order), the merged turn must still complete the consult AND post
    its text reply — never turn the typed half voice-only, never drop either input."""
    from gateway.platforms.base import merge_pending_message_event

    for first_is_consult in (True, False):
        pending = {}
        first = _make_event()
        first.text = "first"
        first.voice_consult = first_is_consult
        second = _make_event()
        second.text = "second"
        second.voice_consult = not first_is_consult
        merge_pending_message_event(pending, "k", first, merge_text=True)
        merge_pending_message_event(pending, "k", second, merge_text=True)
        merged = pending["k"]
        assert "first" in merged.text and "second" in merged.text
        assert merged.voice_consult is True
        assert merged.voice_text_mirror is True

    # Two typed messages stay a plain typed turn.
    pending = {}
    merge_pending_message_event(pending, "k", _make_event(), merge_text=True)
    merge_pending_message_event(pending, "k", _make_event(), merge_text=True)
    assert pending["k"].voice_consult is False
    assert pending["k"].voice_text_mirror is False


def _progress_ctx(guild_id=5, queue=None):
    return SimpleNamespace(
        source=SimpleNamespace(platform=Platform.DISCORD, chat_id="900"),
        _voice_ack_guild=[guild_id],
        _live_status_adapter=None,
        _live_status_mode="off",
        log_queue=None,
        progress_queue=queue if queue is not None else Queue(),
        _run_still_current=lambda: True,
        long_tool_hint_fired=[False],
        progress_mode="all",
        tool_progress_enabled=True,
        _thinking_enabled=False,
        _native_slack_task_cards=False,
        agent_holder=[None],
        last_tool=[None],
        last_was_terminal_block=[False],
        last_progress_msg=[None],
        repeat_count=[0],
        stream_consumer_holder=[None],
    )


@pytest.mark.asyncio
async def test_voice_only_consult_shows_no_typing_indicator():
    """A consult answered by the voice model never posts text, so the bound channel must not
    show "typing…" for it; with text mirror on, the reply lands there and typing is fine."""
    typing_started = []

    class _TypingAdapter(_DummyAdapter):
        def __init__(self):
            super().__init__()
            self.config.typing_indicator = True

        async def _keep_typing(self, chat_id, metadata=None, stop_event=None):
            typing_started.append(chat_id)

    adapter = _TypingAdapter()
    adapter._should_auto_tts_for_chat = lambda _chat_id: False
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="done"))

    consult = _make_event()
    consult.voice_consult = True
    await adapter._process_message_background(consult, build_session_key(consult.source))
    assert typing_started == []

    mirrored = _make_event()
    mirrored.voice_consult = True
    mirrored.voice_text_mirror = True
    await adapter._process_message_background(mirrored, build_session_key(mirrored.source))
    assert typing_started == ["900"]

    typed = _make_event()
    await adapter._process_message_background(typed, build_session_key(typed.source))
    assert typing_started == ["900", "900"]


def test_consult_active_skips_progress_queue():
    runner = object.__new__(GatewayRunner)
    controller = SimpleNamespace(
        consult_active=True,
        session=SimpleNamespace(alive=True, _cfg=SimpleNamespace(discord_text_mirror=False)),
    )
    runner._voice_realtime_controllers = {(None, 5): controller}
    q = Queue()
    TurnRunner(runner, _progress_ctx(queue=q)).progress_callback(
        "tool.started", "terminal", preview="ls"
    )
    assert q.empty()


def test_progress_queue_still_fills_when_text_mirror_on():
    runner = object.__new__(GatewayRunner)
    controller = SimpleNamespace(
        consult_active=True,
        session=SimpleNamespace(alive=True, _cfg=SimpleNamespace(discord_text_mirror=True)),
    )
    runner._voice_realtime_controllers = {(None, 5): controller}
    q = Queue()
    TurnRunner(runner, _progress_ctx(queue=q)).progress_callback(
        "tool.started", "terminal", preview="ls"
    )
    assert not q.empty()


def test_progress_queue_fills_when_no_consult():
    runner = object.__new__(GatewayRunner)
    controller = SimpleNamespace(
        consult_active=False,
        session=SimpleNamespace(alive=True, _cfg=SimpleNamespace(discord_text_mirror=False)),
    )
    runner._voice_realtime_controllers = {(None, 5): controller}
    q = Queue()
    TurnRunner(runner, _progress_ctx(queue=q)).progress_callback(
        "tool.started", "terminal", preview="ls"
    )
    assert not q.empty()


def test_progress_queue_fills_without_ack_guild():
    """No resolved voice guild (a chat with no VC binding) means no consult can own the turn —
    tool progress posts as usual."""
    runner = object.__new__(GatewayRunner)
    controller = SimpleNamespace(
        consult_active=True,
        session=SimpleNamespace(alive=True, _cfg=SimpleNamespace(discord_text_mirror=False)),
    )
    runner._voice_realtime_controllers = {(None, 5): controller}
    q = Queue()
    ctx = _progress_ctx(queue=q)
    ctx._voice_ack_guild = [None]
    TurnRunner(runner, ctx).progress_callback(
        "tool.started", "terminal", preview="ls"
    )
    assert not q.empty()
