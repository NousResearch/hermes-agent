"""Adapter-declared session scope must survive the live TTS delivery handoff (#144148)."""

import asyncio
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.platforms.base import BasePlatformAdapter, SendResult, StreamingTTSHandle
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore, build_session_context
from gateway.session_identity import RoutingIdentity
from gateway.streaming_tts_consumer import StreamingTTSConsumer
from gateway.turn_context import TurnContext


class VoiceAdapter(BasePlatformAdapter):
    async def connect(self, **kwargs):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, **kwargs):
        return SendResult(success=True, message_id="reply")

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "group"}

    def _should_auto_tts_for_chat(self, chat_id):
        return True

    def supports_streaming_tts(self, chat_id, audio_format):
        return True

    async def begin_streaming_tts(self, chat_id, audio_format, metadata=None):
        return StreamingTTSHandle(chat_id=chat_id, audio_format=audio_format)

    async def write_streaming_tts(self, handle, chunk):
        handle.audible = True

    async def finish_streaming_tts(self, handle, **kwargs):
        pass

    async def abort_streaming_tts(self, handle, **kwargs):
        handle.aborted = True


def wire(runner, adapter):
    runner._wire_adapter_handlers(
        adapter, message_handler=AsyncMock(), fatal_error_handler=Mock(),
        busy_session_handler=AsyncMock(), authorization_check=lambda *_: True,
        platform_event_handler=AsyncMock(), busy_text_mode="queue", human_delay=None)


@pytest.fixture
def rig(tmp_path):
    name = "session-scope-probe"
    platform_registry.register(PlatformEntry(
        name=name, label="Scope probe", adapter_factory=lambda cfg: None, check_fn=lambda: True))
    platform = Platform(name)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False)
    runner._primary_profile_name = "default"
    runner._profile_configs, runner._profile_adapters = {}, {}
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    adapter = VoiceAdapter(PlatformConfig(typing_indicator=False, extra={
        "group_sessions_per_user": False}), platform)
    adapter.gateway_runner = runner
    runner.adapters = {platform: adapter}
    wire(runner, adapter)
    try:
        yield SimpleNamespace(runner=runner, adapter=adapter, platform=platform)
    finally:
        platform_registry.unregister(name)


def test_plugin_scope_agrees_across_store_runner_context_and_controls(rig):
    runner, adapter = rig.runner, rig.adapter
    first = adapter.build_source(chat_id="room", chat_type="group", user_id="one")
    second = adapter.build_source(chat_id="room", chat_type="group", user_id="two")
    key = adapter._source_session_key(first)
    assert runner.session_store._generate_session_key(first) == key
    assert runner._session_key_for_source(first) == key
    assert runner._session_key_for_source(second) == key
    assert runner._is_shared_session_source(first)
    context = build_session_context(first, runner.config, session_store=runner.session_store)
    assert context.shared_multi_user_session
    # An unrelated platform retains the gateway's participant isolation.
    ordinary = SessionSource(Platform.TELEGRAM, "room", chat_type="group", user_id="one")
    other = SessionSource(Platform.TELEGRAM, "room", chat_type="group", user_id="two")
    assert runner._session_key_for_source(ordinary) != runner._session_key_for_source(other)


@pytest.mark.parametrize("thread", [False, True])
@pytest.mark.parametrize("per_user", [False, True])
def test_opposite_platform_and_gateway_policies_keep_keys_and_guards_aligned(rig, thread, per_user):
    runner, adapter = rig.runner, rig.adapter
    flag = "thread_sessions_per_user" if thread else "group_sessions_per_user"
    setattr(runner.config, flag, not per_user)
    adapter.config.extra = {flag: per_user}
    wire(runner, adapter)
    source = adapter.build_source(chat_id="room", chat_type="group", user_id="speaker",
                                  thread_id="thread" if thread else None)
    key = runner._session_key_for_source(source)
    assert key == adapter._source_session_key(source)
    assert key.endswith(":speaker") is per_user
    assert runner._is_shared_session_source(source) is not per_user
    assert runner._same_origin_chat(source, SessionSource(
        rig.platform, "room", chat_type="group", user_id="another",
        thread_id=source.thread_id)) is not per_user


def test_secondary_uses_own_defaults_and_routed_transport_policy(rig, tmp_path):
    runner, primary = rig.runner, rig.adapter
    runner.config.multiplex_profiles = True
    primary.config.extra = {"group_sessions_per_user": True}
    wire(runner, primary)
    runner._profile_configs["secondary"] = GatewayConfig(
        group_sessions_per_user=False, thread_sessions_per_user=True)
    secondary = VoiceAdapter(PlatformConfig(extra={"group_sessions_per_user": None}), rig.platform)
    secondary.set_owner_profile("secondary")
    wire(runner, secondary)
    assert secondary.config.extra == {
        "group_sessions_per_user": False, "thread_sessions_per_user": True}
    source = SessionSource(rig.platform, "room", chat_type="group", user_id="speaker", profile="runtime")
    source._identity = RoutingIdentity(
        transport_profile="secondary", runtime_profile="runtime",
        authorization_home=tmp_path / "secondary", runtime_home=tmp_path / "runtime",
        transport=weakref.ref(secondary))
    key = runner._session_key_for_source(source)
    assert key == "agent:runtime:session-scope-probe:group:room"
    # A restored source has no live adapter but retains its receiving-profile identity.
    source._identity = RoutingIdentity(
        transport_profile="secondary", runtime_profile="runtime",
        authorization_home=tmp_path / "secondary", runtime_home=tmp_path / "runtime")
    assert runner._session_key_for_source(source) == key
    ordinary = SessionSource(rig.platform, "room", chat_type="group", user_id="speaker")
    assert runner._session_key_for_source(ordinary).endswith(":speaker")


def test_runner_fallback_keeps_the_live_adapter_policy(rig):
    source = rig.adapter.build_source(chat_id="room", chat_type="group", user_id="speaker")
    expected = rig.adapter._source_session_key(source)
    rig.runner.session_store = None
    assert rig.runner._session_key_for_source(source) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["complete", "partial", "silent"])
async def test_audible_stream_is_not_synthesized_again_at_final_delivery(rig, monkeypatch, outcome):
    runner, adapter = rig.runner, rig.adapter

    class Streamer:
        sample_rate, channels, sample_width = 24000, 1, 2

        def stream(self, text):
            if outcome != "silent":
                yield b"\x01\x00" * 240
            if outcome == "partial":
                raise RuntimeError("synthetic provider failure after audible PCM")

    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda _: Streamer())
    source = adapter.build_source(chat_id="room", chat_type="group", user_id="speaker")
    event = MessageEvent(text="Synthetic input", source=source,
                         message_type=MessageType.VOICE, message_id="input")
    adapter_key = adapter._event_session_key(event)
    guard = asyncio.Event()
    adapter._active_sessions[adapter_key] = guard
    adapter._run_processing_hook = AsyncMock()
    adapter._synthesize_auto_tts = AsyncMock(return_value=([], None))
    async def final_text(*args):
        args[-1](SendResult(success=True, message_id="reply"))

    adapter._send_final_text = AsyncMock(side_effect=final_text)

    async def reply(_event):
        runner_key = runner._session_key_for_source(source)
        runner._bind_adapter_run_generation(adapter, runner_key, 1)
        consumer = StreamingTTSConsumer(adapter, source.chat_id, {}, asyncio.get_running_loop())
        consumer.start()
        consumer.on_delta("One synthetic reply.")
        await runner._run_agent_finalize_streaming_tts(TurnContext(
            session_key=runner_key, run_generation=1, streaming_tts_consumer_holder=[consumer]), adapter)
        return "One synthetic reply."

    adapter.set_message_handler(reply)
    await adapter._process_message_background(event, adapter_key)
    assert adapter._synthesize_auto_tts.await_count == (1 if outcome == "silent" else 0)
    adapter._send_final_text.assert_awaited_once()
    assert not adapter._streaming_tts_completed_turns
