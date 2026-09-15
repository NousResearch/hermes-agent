import asyncio
import base64
import copy
import json
import logging
import struct

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG
from plugins.platforms.discord import live_transcription
from plugins.platforms.discord.live_transcription import (
    DiscordLiveTranscriptionController,
    LiveTranscriptionConfig,
    LiveTranscriptionError,
    OpenAIRealtimeTranscriptionSession,
    build_live_session_update,
    normalize_discord_stt_mode,
    pcm48_stereo_to_pcm24_mono,
    resolve_openai_realtime_api_key,
)


class FakeWebSocket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def recv(self):
        return json.dumps(await self.incoming.get())

    async def close(self):
        self.closed = True


@pytest.mark.parametrize("playback_outcome", ["complete", "error", "cancel"])
@pytest.mark.asyncio
async def test_first_live_utterance_after_playback_contains_only_new_audio(monkeypatch, playback_outcome):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from plugins.platforms.discord import adapter as discord_adapter

    receiver = discord_adapter.VoiceReceiver(SimpleNamespace())
    receiver._running = True
    receiver.require_explicit_mapping = True
    receiver.map_ssrc(100, 42)
    adapter = object.__new__(discord_adapter.DiscordAdapter)
    adapter._voice_receivers = {7: receiver}
    adapter._voice_input_callback = AsyncMock()
    adapter._is_allowed_user = lambda user_id, **kwargs: user_id == "42"
    adapter._voice_timeout_tasks = {}
    adapter._voice_timeout_limit = lambda: 0
    adapter._playback_timeout_for_audio = AsyncMock(return_value=2)
    sockets = []

    class Socket(FakeWebSocket):
        async def send(self, payload):
            await super().send(payload)
            if json.loads(payload)["type"] == "input_audio_buffer.commit":
                await self.incoming.put({"type": "input_audio_buffer.committed", "item_id": "new-turn"})
                await self.incoming.put({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "new-turn", "transcript": "new speech after playback",
                })

        async def close(self):
            if self is sockets[0] and not self.closed:
                assert receiver._paused, "old provider state must retire before capture resumes"
            await super().close()

    async def connect(*args, **kwargs):
        ws = Socket()
        sockets.append(ws)
        await ws.incoming.put({"type": "session.updated"})
        return ws

    config = LiveTranscriptionConfig()
    controller = DiscordLiveTranscriptionController(
        api_key="synthetic", config=config,
        session_factory=lambda: OpenAIRealtimeTranscriptionSession(
            api_key="synthetic", config=config, websocket_connect=connect),
    )
    adapter._voice_live_transcribers = {7: controller}

    def play(source, *, after):
        assert receiver._paused
        receiver._buffer_decoded_pcm(100, b"\x09\x00" * 4800)
        if playback_outcome == "error":
            raise RuntimeError("synthetic playback failure")
        if playback_outcome == "cancel":
            raise asyncio.CancelledError()
        after(None)

    adapter._voice_clients = {7: SimpleNamespace(
        is_connected=lambda: True, is_playing=lambda: False, play=play,
    )}
    monkeypatch.setattr(discord_adapter, "_read_runtime_stt_enabled", lambda: True)
    monkeypatch.setattr(discord_adapter, "resolve_ffmpeg_executable", lambda: "unused-ffmpeg")
    monkeypatch.setattr(discord_adapter.discord, "FFmpegPCMAudio", lambda *args, **kwargs: object())
    monkeypatch.setattr(discord_adapter.discord, "PCMVolumeTransformer", lambda source, **kwargs: source)

    async def tick():
        await adapter._process_voice_listener_tick(
            guild_id=7, receiver=receiver, stt_mode="openai_live_high", guild=None)

    old_pcm = b"\x01\x00\x02\x00" * 4800
    new_pcm = b"\x03\x00\x04\x00" * 24000
    try:
        receiver._buffer_decoded_pcm(100, old_pcm)
        await tick()
        assert controller._source_bytes[42] == len(old_pcm)
        assert any(event["type"] == "input_audio_buffer.append" for event in sockets[0].sent)
        adapter._voice_input_callback.assert_not_awaited()

        if playback_outcome == "complete":
            assert await adapter.play_in_voice_channel(7, "synthetic.wav")
        else:
            error = RuntimeError if playback_outcome == "error" else asyncio.CancelledError
            with pytest.raises(error):
                await adapter.play_in_voice_channel(7, "synthetic.wav")
        assert not receiver._paused

        receiver._buffer_decoded_pcm(100, new_pcm)
        receiver._last_packet_time[100] = 0
        await tick()
        adapter._voice_input_callback.assert_awaited_once_with(
            guild_id=7, user_id=42, transcript="new speech after playback")
        assert sockets[0].closed
        assert not any(event["type"] == "input_audio_buffer.commit" for event in sockets[0].sent)
        wire_pcm = b"".join(base64.b64decode(event["audio"]) for event in sockets[-1].sent
                            if event["type"] == "input_audio_buffer.append")
        assert wire_pcm == pcm48_stereo_to_pcm24_mono(new_pcm)
        assert not controller._source_bytes
        assert not controller._turn_admissions
    finally:
        receiver.pause()
        await controller.close()


async def _wait_for_sent_type(ws, event_type, count=1):
    for _ in range(100):
        if sum(event.get("type") == event_type for event in ws.sent) >= count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"timed out waiting for {count} {event_type} event(s)")


@pytest.fixture
def delayed_live_wire():
    sockets = []

    class Socket(FakeWebSocket):
        def __init__(self):
            super().__init__()
            self.commit_entered = asyncio.Event()
            self.release_commit = asyncio.Event()
            self.release_commit.set()
            self.close_entered = asyncio.Event()
            self.release_close = asyncio.Event()
            self.release_close.set()

        async def send(self, payload):
            if json.loads(payload)["type"] == "input_audio_buffer.commit":
                self.commit_entered.set()
                await self.release_commit.wait()
            await super().send(payload)

        async def close(self):
            self.close_entered.set()
            await self.release_close.wait()
            await super().close()

    async def connect(*args, **kwargs):
        ws = Socket()
        sockets.append(ws)
        await ws.incoming.put({"type": "session.updated"})
        return ws

    config = LiveTranscriptionConfig()
    controller = DiscordLiveTranscriptionController(
        api_key="synthetic", config=config,
        session_factory=lambda: OpenAIRealtimeTranscriptionSession(
            api_key="synthetic", config=config, websocket_connect=connect),
    )
    return controller, sockets


@pytest.mark.parametrize("outcome", ["complete", "failed", "revoked", "cancel"])
@pytest.mark.asyncio
async def test_delayed_completion_releases_controller_after_commit_send(delayed_live_wire, outcome):
    controller, sockets = delayed_live_wire
    first_pcm, next_pcm, third_pcm, other_pcm = [bytes([n, 0]) * 480 for n in range(1, 5)]
    allowed = [True]
    tasks = []
    try:
        await controller.append_pcm48(42, first_pcm, admit=lambda: allowed[0])
        ws = sockets[0]
        ws.release_commit.clear()
        first = asyncio.create_task(controller.finish_utterance(
            42, expected_source_bytes=len(first_pcm), admit=lambda: allowed[0]))
        tasks.append(first)
        await asyncio.wait_for(ws.commit_entered.wait(), 2)
        next_append = asyncio.create_task(controller.append_pcm48(42, next_pcm, admit=lambda: True))
        other_append = asyncio.create_task(controller.append_pcm48(43, other_pcm, admit=lambda: True))
        tasks.extend([next_append, other_append])
        await asyncio.sleep(0)
        assert [event["type"] for event in ws.sent] == ["session.update", "input_audio_buffer.append"]
        ws.release_commit.set()

        # Neither provider ACK nor completion has arrived; both speakers must reach the wire.
        await _wait_for_sent_type(ws, "input_audio_buffer.append", 2)
        await asyncio.wait_for(asyncio.gather(next_append, other_append), 2)
        assert not first.done()
        assert [event["type"] for event in ws.sent] == [
            "session.update", "input_audio_buffer.append", "input_audio_buffer.commit", "input_audio_buffer.append"]
        assert [base64.b64decode(e["audio"]) for e in ws.sent if "audio" in e] == [
            pcm48_stereo_to_pcm24_mono(first_pcm), pcm48_stereo_to_pcm24_mono(next_pcm)]
        assert [base64.b64decode(e["audio"]) for e in sockets[1].sent if "audio" in e] == [
            pcm48_stereo_to_pcm24_mono(other_pcm)]

        second = asyncio.create_task(controller.finish_utterance(
            42, expected_source_bytes=len(next_pcm), admit=lambda: True))
        tasks.append(second)
        await _wait_for_sent_type(ws, "input_audio_buffer.commit", 2)
        await controller.append_pcm48(42, third_pcm, admit=lambda: True)
        for item_id in ("first", "second"):
            await ws.incoming.put({"type": "input_audio_buffer.committed", "item_id": item_id})
        await ws.incoming.put({"type": "conversation.item.input_audio_transcription.completed",
                               "item_id": "second", "transcript": "second turn"})
        assert (await asyncio.wait_for(second, 2))["transcript"] == "second turn"
        assert not first.done()
        if outcome == "cancel":
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            allowed[0] = outcome != "revoked"
            event_type = "failed" if outcome == "failed" else "completed"
            await ws.incoming.put({"type": f"conversation.item.input_audio_transcription.{event_type}",
                                   "item_id": "first", "transcript": "first turn"})
            result = await asyncio.wait_for(first, 2)
            assert result["success"] is (outcome == "complete")
            if outcome == "complete":
                assert result["transcript"] == "first turn"
        assert controller._source_bytes == {42: len(third_pcm), 43: len(other_pcm)}
        assert not sockets[1].closed
        third = asyncio.create_task(controller.finish_utterance(
            42, expected_source_bytes=len(third_pcm), admit=lambda: True))
        tasks.append(third)
        if outcome == "complete":
            await _wait_for_sent_type(ws, "input_audio_buffer.commit", 3)
            await ws.incoming.put({"type": "input_audio_buffer.committed", "item_id": "third"})
            await ws.incoming.put({"type": "conversation.item.input_audio_transcription.completed",
                                   "item_id": "third", "transcript": "third turn"})
            assert (await asyncio.wait_for(third, 2))["transcript"] == "third turn"
        else:
            assert (await asyncio.wait_for(third, 2))["success"] is False
            assert ws.closed
        assert [base64.b64decode(e["audio"]) for e in ws.sent if "audio" in e][-1] == pcm48_stereo_to_pcm24_mono(third_pcm)
    finally:
        for ws in sockets:
            ws.release_commit.set()
            ws.release_close.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await controller.close()


@pytest.mark.parametrize("retire", ["abort_user", "abort_pending", "close"])
@pytest.mark.asyncio
async def test_delayed_completion_retirement_preserves_replacement(delayed_live_wire, retire):
    controller, sockets = delayed_live_wire
    pcm = b"\x01\x00" * 480
    replacement_pcm = b"\x02\x00" * 480
    tasks = []
    try:
        await controller.append_pcm48(42, pcm, admit=lambda: True)
        old = controller._sessions[42]
        ws = sockets[0]
        first = asyncio.create_task(controller.finish_utterance(42, expected_source_bytes=len(pcm)))
        tasks.append(first)
        await _wait_for_sent_type(ws, "input_audio_buffer.commit")
        ws.release_close.clear()
        retiring = asyncio.create_task(controller.abort_user(42) if retire == "abort_user"
                                       else getattr(controller, retire)())
        tasks.append(retiring)
        await asyncio.wait_for(ws.close_entered.wait(), 2)
        replacement = None
        if retire != "close":
            # Queue replacement before close wakes the old completion's failure cleanup.
            replacement = asyncio.create_task(controller.append_pcm48(42, replacement_pcm, admit=lambda: True))
            tasks.append(replacement)
            await asyncio.sleep(0)
        ws.release_close.set()
        await asyncio.wait_for(retiring, 2)
        if replacement is not None:
            await asyncio.wait_for(replacement, 2)
        assert (await asyncio.wait_for(first, 2))["success"] is False
        assert ws.closed
        assert not old._pending_unassigned and not old._pending_by_item
        if retire == "close":
            assert not controller._sessions and not controller._source_bytes
        else:
            assert controller._sessions[42] is not old
            assert controller._source_bytes == {42: len(replacement_pcm)}
            assert not sockets[1].closed
            final = asyncio.create_task(controller.finish_utterance(42, expected_source_bytes=len(replacement_pcm)))
            tasks.append(final)
            await _wait_for_sent_type(sockets[1], "input_audio_buffer.commit")
            await sockets[1].incoming.put({"type": "input_audio_buffer.committed", "item_id": "replacement"})
            await sockets[1].incoming.put({"type": "conversation.item.input_audio_transcription.completed",
                                           "item_id": "replacement", "transcript": "replacement turn"})
            assert (await asyncio.wait_for(final, 2))["transcript"] == "replacement turn"
            assert [base64.b64decode(e["audio"]) for e in sockets[1].sent if "audio" in e] == [
                pcm48_stereo_to_pcm24_mono(replacement_pcm)]
    finally:
        for ws in sockets:
            ws.release_commit.set()
            ws.release_close.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await controller.close()


@pytest.fixture
def listener_live_wire(delayed_live_wire, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock
    from plugins.platforms.discord import adapter as discord_adapter

    controller, sockets = delayed_live_wire
    receiver = discord_adapter.VoiceReceiver(SimpleNamespace())
    receiver._running = True
    receiver.require_explicit_mapping = True
    receiver.map_ssrc(100, 42)
    receiver.map_ssrc(101, 43)
    adapter = object.__new__(discord_adapter.DiscordAdapter)
    adapter._client = None
    adapter._allowed_user_ids = {"42", "43"}
    adapter._allowed_role_ids = set()
    adapter._voice_input_callback = AsyncMock()
    adapter._voice_receivers = {7: receiver}
    adapter._voice_live_transcribers = {7: controller}
    adapter._voice_stt_modes = {7: "openai_live_high"}
    adapter._voice_locks = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_timeout_limit = lambda: 0
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    vc = SimpleNamespace(is_connected=lambda: True, is_playing=lambda: False,
                         disconnect=AsyncMock(), _connection=MagicMock())
    adapter._voice_clients = {7: vc}
    clock = [100.0]
    ticks = asyncio.Queue()

    async def sleep(delay):
        if delay == 0.2:
            await ticks.get()
        else:
            await asyncio.sleep(delay)

    class AsyncioProxy:
        def __getattr__(self, name):
            return sleep if name == "sleep" else getattr(asyncio, name)

    monkeypatch.setattr(discord_adapter, "asyncio", AsyncioProxy())
    monkeypatch.setattr(discord_adapter, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(discord_adapter, "_read_runtime_stt_enabled", lambda: True)

    async def settle():
        for _ in range(100):
            await asyncio.sleep(0)

    async def tick():
        ticks.put_nowait(None)
        await settle()

    def packet(ssrc, sample):
        # One decoded Discord packet: 20ms, stereo PCM16 at 48kHz.
        pcm = struct.pack("<h", sample) * 1920
        receiver._buffer_decoded_pcm(ssrc, pcm)
        return pcm

    async def answer(ws, item, text):
        await ws.incoming.put({"type": "input_audio_buffer.committed", "item_id": item})
        await ws.incoming.put({"type": "conversation.item.input_audio_transcription.completed",
                               "item_id": item, "transcript": text})
        await settle()

    return SimpleNamespace(adapter=adapter, receiver=receiver, controller=controller,
                           sockets=sockets, clock=clock, tick=tick, settle=settle,
                           packet=packet, answer=answer, vc=vc)


@pytest.mark.asyncio
async def test_real_listener_pumps_six_seconds_before_first_answer_and_during_dispatch(listener_live_wire):
    r = listener_live_wire
    first_pcm = b"".join(r.packet(100, 1) for _ in range(25))
    r.clock[0] += r.receiver.SILENCE_THRESHOLD + 0.2
    listener = asyncio.create_task(r.adapter._voice_listen_loop(7))
    r.adapter._voice_listen_tasks[7] = listener
    callback_entered, release_callback = asyncio.Event(), asyncio.Event()

    async def callback(**kwargs):
        if kwargs["transcript"] == "first phrase":
            callback_entered.set()
            await release_callback.wait()

    r.adapter._voice_input_callback.side_effect = callback
    sessions = []
    try:
        await r.tick()
        ws = r.sockets[0]
        await _wait_for_sent_type(ws, "input_audio_buffer.commit")
        later, other = [], []
        for number in range(300):
            r.clock[0] += 0.02
            later.append(r.packet(100, number + 10))
            other.append(r.packet(101, number + 1000))
            if number % 10 == 9:
                await r.tick()
        wire = b"".join(base64.b64decode(e["audio"]) for e in ws.sent if "audio" in e)
        assert wire == pcm48_stereo_to_pcm24_mono(first_pcm + b"".join(later)), (
            "all six packet-paced seconds must reach the wire BEFORE the first completion")
        assert len(r.sockets) == 2
        assert b"".join(base64.b64decode(e["audio"]) for e in r.sockets[1].sent if "audio" in e) == pcm48_stereo_to_pcm24_mono(b"".join(other))
        assert sum(e["type"] == "input_audio_buffer.commit" for e in ws.sent) == 1
        r.adapter._voice_input_callback.assert_not_awaited()
        assert r.controller._source_bytes == {42: len(b"".join(later)), 43: len(b"".join(other))}
        r.clock[0] += 15
        await r.tick()
        r.vc._connection.send_packet.assert_called_with(b'\xf8\xff\xfe')

        await r.answer(ws, "first", "first phrase")
        assert callback_entered.is_set()
        for _ in range(10):
            r.clock[0] += 0.02
            later.append(r.packet(100, 400))
        await r.tick()
        assert b"".join(base64.b64decode(e["audio"]) for e in ws.sent if "audio" in e) == pcm48_stereo_to_pcm24_mono(first_pcm + b"".join(later))
        release_callback.set()
        await r.settle()
        r.clock[0] += r.receiver.SILENCE_THRESHOLD + 0.2
        await r.tick()
        await _wait_for_sent_type(ws, "input_audio_buffer.commit", 2)
        await _wait_for_sent_type(r.sockets[1], "input_audio_buffer.commit")
        await r.answer(r.sockets[1], "other", "other speaker")
        assert r.adapter._voice_input_callback.await_count == 1
        await r.answer(ws, "later", "later phrases coalesced")
        assert [(c.kwargs["user_id"], c.kwargs["transcript"]) for c in r.adapter._voice_input_callback.await_args_list] == [
            (42, "first phrase"), (42, "later phrases coalesced"), (43, "other speaker")]
        assert not r.controller._source_bytes
        sessions = list(r.controller._sessions.values())
    finally:
        release_callback.set()
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)
        await r.adapter.leave_voice_channel(7)
    assert all(ws.closed for ws in r.sockets)
    assert all(s._reader_task is None and not s._pending_by_item and not s._pending_unassigned for s in sessions)
    assert not r.adapter._voice_listen_tasks and not r.controller._sessions


@pytest.mark.parametrize("exit_kind", ["release", "cancel_send", "cancel_wait", "revoke", "flush"])
@pytest.mark.asyncio
async def test_listener_seals_batch_before_pump_and_retires_children(listener_live_wire, exit_kind):
    r = listener_live_wire
    for _ in range(25):
        r.packet(100, 1)
        r.packet(101, 2)
    # First tick appends only; next tick extracts both completed records.
    listener = asyncio.create_task(r.adapter._voice_listen_loop(7))
    r.adapter._voice_listen_tasks[7] = listener
    tasks = [listener]
    sessions = []
    try:
        await r.tick()
        assert len(r.sockets) == 2
        first, second = r.sockets
        sessions = list(r.controller._sessions.values())
        second.release_commit.clear()
        r.clock[0] += r.receiver.SILENCE_THRESHOLD + 0.2
        await r.tick()
        assert first.commit_entered.is_set()
        assert second.commit_entered.is_set(), "seal the second extracted record while the first answer waits"
        # Later audio must stay in the receiver until ALL extracted records seal.
        r.packet(101, 3)
        await r.tick()
        assert sum("audio" in e for e in second.sent) == 1
        if exit_kind == "cancel_send":
            listener.cancel()
            await asyncio.gather(listener, return_exceptions=True)
            assert all(ws.closed for ws in r.sockets)
        else:
            second.release_commit.set()
            await r.settle()
            await r.tick()
            assert sum("audio" in e for e in second.sent) == 2
            if exit_kind in {"cancel_wait", "flush"}:
                listener.cancel()
                await asyncio.gather(listener, return_exceptions=True)
                assert all(ws.closed for ws in r.sockets)
                if exit_kind == "flush":
                    # Cancellation loses in-flight partial accounting. Fresh captured audio
                    # on another source still flushes through the native leave path.
                    r.receiver.discard_pending()
                    for _ in range(25):
                        r.packet(100, 4)
                    leaving = asyncio.create_task(r.adapter.leave_voice_channel(7))
                    tasks.append(leaving)
                    await r.settle()
                    final = r.sockets[-1]
                    assert final is not first
                    await _wait_for_sent_type(final, "input_audio_buffer.commit")
                    r.packet(100, 5)
                    await r.tick()
                    assert sum("audio" in e for e in final.sent) == 1, "flush cannot start an ingress pump"
                    await r.answer(final, "final", "final captured speech")
                    await asyncio.wait_for(leaving, 2)
                    assert r.adapter._voice_input_callback.await_count == 1
            else:
                await r.answer(second, "second", "second captured speech")
                r.adapter._voice_input_callback.assert_not_awaited()
                if exit_kind == "revoke":
                    r.adapter._allowed_user_ids.remove("43")
                    await r.adapter._revoke_voice_stt_user(7, 43)
                await r.answer(first, "first", "first captured speech")
                assert [c.kwargs["user_id"] for c in r.adapter._voice_input_callback.await_args_list] == (
                    [42] if exit_kind == "revoke" else [42, 43])
    finally:
        for ws in r.sockets:
            ws.release_commit.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        r.receiver.stop()
        await r.controller.close()
    assert all(s._reader_task is None and not s._pending_unassigned and not s._pending_by_item for s in sessions)
    assert all(ws.closed for ws in r.sockets)


def test_mode_aliases_are_strict_and_backward_safe():
    assert normalize_discord_stt_mode(None) == "configured"
    assert normalize_discord_stt_mode("default") == "configured"
    assert normalize_discord_stt_mode("configured") == "configured"
    assert normalize_discord_stt_mode("contextual") == "openai_contextual"
    assert normalize_discord_stt_mode("openai-contextual") == "openai_contextual"
    assert normalize_discord_stt_mode("live-high") == "openai_live_high"
    assert normalize_discord_stt_mode("openai_live_high") == "openai_live_high"
    with pytest.raises(ValueError, match="unsupported Discord STT mode"):
        normalize_discord_stt_mode("surprise-paid-backend")


def test_pcm48_stereo_to_pcm24_mono_downmixes_and_decimates():
    source = struct.pack(
        "<12h",
        1000,
        3000,
        30000,
        30000,
        -2000,
        -4000,
        -30000,
        -30000,
        200,
        400,
        1234,
        5678,
    )
    converted = pcm48_stereo_to_pcm24_mono(source)
    assert struct.unpack("<3h", converted) == (16000, -16500, 1878)


def test_pcm_converter_rejects_partial_stereo_frame():
    with pytest.raises(ValueError, match="whole pairs of 48 kHz stereo frames"):
        pcm48_stereo_to_pcm24_mono(b"\x00\x01")


def test_session_update_matches_gpt_live_transcribe_contract():
    config = LiveTranscriptionConfig(
        prompt="German private voice chat about AI.",
        keywords=("Hermes Agent", "Mac mini"),
        languages=("de", "en"),
    )
    update = build_live_session_update(config)
    transcription = update["session"]["audio"]["input"]["transcription"]
    assert update["type"] == "session.update"
    assert update["session"]["type"] == "transcription"
    assert update["session"]["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }
    assert update["session"]["audio"]["input"]["turn_detection"] is None
    assert transcription == {
        "model": "gpt-live-transcribe",
        "prompt": "German private voice chat about AI.",
        "keywords": ["Hermes Agent", "Mac mini"],
        "languages": ["de", "en"],
        "delay": "high",
    }


def test_invalid_context_is_rejected_before_provider_call():
    with pytest.raises(ValueError, match="keyword"):
        LiveTranscriptionConfig(keywords=("bad\nkeyword",))
    with pytest.raises(ValueError, match="delay"):
        LiveTranscriptionConfig(delay="turbo")
    with pytest.raises(ValueError, match="requires delay=high"):
        LiveTranscriptionConfig(delay="low")
    with pytest.raises(ValueError, match="api.openai.com"):
        LiveTranscriptionConfig(
            endpoint="wss://attacker.example/v1/realtime?intent=transcription"
        )


@pytest.mark.parametrize("field", ["languages", "keywords"])
@pytest.mark.parametrize("raw", [None, "", "  ", [], (), " [] "])
def test_empty_live_context_is_omitted_from_session_update(field, raw):
    config = LiveTranscriptionConfig(**{field: raw})
    transcription = build_live_session_update(config)["session"]["audio"]["input"]["transcription"]
    assert getattr(config, field) == ()
    assert field not in transcription


@pytest.mark.parametrize(
    "field,raw,expected",
    [
        ("languages", " de-DE ", ("de-DE",)),
        ("languages", '["de", " en ", "", "de"]', ("de", "en", "de")),
        ("languages", (" de ", "en"), ("de", "en")),
        ("keywords", " Hermes Agent ", ("Hermes Agent",)),
        ("keywords", '["Hermes Agent", " Mac mini ", ""]', ("Hermes Agent", "Mac mini")),
        ("keywords", ["[literal keyword]", " A,B "], ("[literal keyword]", "A,B")),
    ],
)
def test_live_context_decoding_preserves_wire_values_and_input(field, raw, expected):
    before = copy.deepcopy(raw)
    config = LiveTranscriptionConfig(**{field: raw})
    transcription = build_live_session_update(config)["session"]["audio"]["input"]["transcription"]
    assert getattr(config, field) == expected
    assert transcription[field] == list(expected)
    assert raw == before


@pytest.mark.parametrize(
    "field,raw,error",
    [
        ("languages", '["de"', "valid JSON list"),
        ("keywords", "[literal keyword]", "valid JSON list"),
        ("languages", ["de", 1], "string or list of strings"),
        ("keywords", ("Hermes", None), "string or list of strings"),
        ("languages", {"de": True}, "string or list of strings"),
        ("keywords", 42, "string or list of strings"),
        ("languages", "de,en", "invalid language code format"),
        ("languages", "en_US", "invalid language code format"),
        ("languages", '["english"]', "invalid language code format"),
        ("keywords", "Hermes\n", "forbidden character"),
        ("keywords", ["\rHermes"], "forbidden character"),
        ("keywords", '["Hermes\\n"]', "forbidden character"),
        ("keywords", "[\n\"Hermes\"]", "forbidden character"),
        ("keywords", ("<Hermes>",), "forbidden character"),
    ],
)
def test_live_context_rejects_invalid_raw_values_before_normalization(field, raw, error):
    with pytest.raises(ValueError, match=error):
        LiveTranscriptionConfig(**{field: raw})


def test_live_config_inherits_context_from_openai_stt_section():
    config = LiveTranscriptionConfig.from_hermes_config(
        {
            "stt": {
                "openai": {
                    "prompt": "German Discord chat.",
                    "keywords": ["Hermes Agent"],
                    "languages": ["de", "en"],
                }
            },
            "discord": {
                "voice_stt": {
                    "openai_live": {
                        "delay": "high",
                    }
                }
            },
        }
    )
    assert config.prompt == "German Discord chat."
    assert config.keywords == ("Hermes Agent",)
    assert config.languages == ("de", "en")


def test_live_config_uses_global_language_fallback_and_cli_json_lists():
    fallback = LiveTranscriptionConfig.from_hermes_config(
        {"stt": {"language": "de"}}
    )
    assert fallback.languages == ("de",)

    cli = LiveTranscriptionConfig.from_hermes_config(
        {
            "stt": {
                "openai": {
                    "keywords": '["Hermes Agent","gpt-live-transcribe"]',
                    "languages": '["de","en"]',
                }
            }
        }
    )
    assert cli.keywords == ("Hermes Agent", "gpt-live-transcribe")
    assert cli.languages == ("de", "en")


def test_live_config_uses_global_language_with_merged_empty_openai_defaults():
    merged = copy.deepcopy(DEFAULT_CONFIG)
    merged["stt"]["language"] = "de"
    merged["stt"]["openai"]["language"] = ""
    merged["stt"]["openai"]["languages"] = "[]"

    config = LiveTranscriptionConfig.from_hermes_config(merged)

    assert config.languages == ("de",)


def test_live_config_rejects_malformed_cli_json_list():
    with pytest.raises(ValueError, match="JSON list"):
        LiveTranscriptionConfig.from_hermes_config(
            {"stt": {"openai": {"languages": '["de"'}}}
        )


def test_realtime_key_resolution_uses_only_direct_openai_credentials(monkeypatch):
    values = {
        "VOICE_TOOLS_OPENAI_KEY": "voice-key",
        "OPENAI_API_KEY": "general-key",
    }
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda name: values.get(name),
    )
    assert resolve_openai_realtime_api_key({}) == "voice-key"
    assert resolve_openai_realtime_api_key(
        {"stt": {"openai": {"api_key": "config-key"}}}
    ) == "config-key"


def test_realtime_key_resolution_fails_without_direct_key(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda _name: None)
    with pytest.raises(ValueError, match="direct OpenAI API key"):
        resolve_openai_realtime_api_key({"stt": {"use_gateway": True}})


@pytest.mark.asyncio
async def test_default_websocket_connector_rejects_every_redirect(monkeypatch):
    from websockets.asyncio.client import connect
    from websockets.exceptions import SecurityError

    monkeypatch.setattr(
        connect,
        "process_redirect",
        lambda _self, _exc: "wss://attacker.invalid/capture",
    )
    connector = live_transcription._create_no_redirect_websocket_connect(
        LiveTranscriptionConfig().endpoint
    )

    assert isinstance(connector.process_redirect(Exception("redirect")), SecurityError)


@pytest.mark.asyncio
async def test_realtime_connection_uses_non_debug_non_propagating_logger(caplog):
    ws = FakeWebSocket()
    await ws.incoming.put({"type": "session.updated"})
    captured = {}

    async def connect(*args, **kwargs):
        captured.update(kwargs)
        return ws

    session = OpenAIRealtimeTranscriptionSession(
        api_key="synthetic-test-key",
        config=LiveTranscriptionConfig(),
        websocket_connect=connect,
    )
    await session.start()

    transport_logger = captured["logger"]
    assert transport_logger.isEnabledFor(logging.DEBUG) is False
    assert transport_logger.propagate is False
    assert any(isinstance(handler, logging.NullHandler) for handler in transport_logger.handlers)
    with caplog.at_level(logging.DEBUG):
        transport_logger.debug("Authorization: Bearer synthetic-marker")
        transport_logger.debug("secret transcript marker")
    assert "synthetic-marker" not in caplog.text
    assert "secret transcript marker" not in caplog.text

    await session.close()


@pytest.mark.asyncio
async def test_persistent_session_reconciles_out_of_order_completions_by_item_id():
    ws = FakeWebSocket()
    await ws.incoming.put({"type": "session.created"})
    await ws.incoming.put({"type": "session.updated"})

    async def connect(*args, **kwargs):
        assert kwargs["additional_headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["compression"] is None
        return ws

    session = OpenAIRealtimeTranscriptionSession(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        websocket_connect=connect,
    )
    await session.start()
    assert ws.sent[0]["type"] == "session.update"

    await session.append_pcm24(b"\x01\x00" * 240)
    commit_one = asyncio.create_task(session.commit())
    await _wait_for_sent_type(ws, "input_audio_buffer.commit", 1)

    await session.append_pcm24(b"\x02\x00" * 240)
    commit_two = asyncio.create_task(session.commit())
    await _wait_for_sent_type(ws, "input_audio_buffer.commit", 2)

    await ws.incoming.put(
        {"type": "input_audio_buffer.committed", "item_id": "item_one"}
    )
    await ws.incoming.put(
        {"type": "input_audio_buffer.committed", "item_id": "item_two"}
    )
    await ws.incoming.put(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_two",
            "transcript": "zweiter Turn",
            "usage": {"type": "duration", "seconds": 0.01},
        }
    )
    await ws.incoming.put(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_one",
            "transcript": "erster Turn",
            "usage": {"type": "duration", "seconds": 0.01},
        }
    )

    first, second = await asyncio.gather(commit_one, commit_two)
    assert first["item_id"] == "item_one"
    assert first["transcript"] == "erster Turn"
    assert second["item_id"] == "item_two"
    assert second["transcript"] == "zweiter Turn"
    assert first["provider"] == "openai_realtime"
    assert first["model"] == "gpt-live-transcribe"

    append_events = [
        event for event in ws.sent if event["type"] == "input_audio_buffer.append"
    ]
    assert len(append_events) == 2
    assert base64.b64decode(append_events[0]["audio"]) == b"\x01\x00" * 240

    await session.close()
    assert ws.closed is True


@pytest.mark.asyncio
async def test_rejected_session_setup_closes_half_open_websocket():
    ws = FakeWebSocket()
    await ws.incoming.put(
        {
            "type": "error",
            "error": {
                "code": "invalid_session",
                "message": "must not leak provider detail secret-value",
            },
        }
    )

    async def connect(*args, **kwargs):
        return ws

    session = OpenAIRealtimeTranscriptionSession(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        websocket_connect=connect,
    )
    with pytest.raises(LiveTranscriptionError, match="invalid_session") as exc:
        await session.start()
    assert "secret-value" not in str(exc.value)
    assert ws.closed is True


@pytest.mark.asyncio
async def test_close_linearizes_against_blocked_start_and_leaves_no_reader():
    ws = FakeWebSocket()
    await ws.incoming.put({"type": "session.updated"})
    release_connect = asyncio.Event()

    async def blocked_connect(_endpoint, **_kwargs):
        await release_connect.wait()
        return ws

    session = OpenAIRealtimeTranscriptionSession(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        websocket_connect=blocked_connect,
    )
    start_task = asyncio.create_task(session.start())
    await asyncio.sleep(0)
    close_task = asyncio.create_task(session.close())
    release_connect.set()
    await asyncio.gather(start_task, close_task)

    assert ws.closed is True
    assert session._ws is None
    assert session._reader_task is None


@pytest.mark.asyncio
async def test_close_linearizes_against_blocked_append_send():
    ws = FakeWebSocket()
    await ws.incoming.put({"type": "transcription_session.updated"})
    send_started = asyncio.Event()
    release_send = asyncio.Event()
    original_send = ws.send

    async def send(payload):
        event = json.loads(payload)
        if event.get("type") == "input_audio_buffer.append":
            send_started.set()
            await release_send.wait()
        await original_send(payload)

    ws.send = send

    async def connect(*_args, **_kwargs):
        return ws

    session = OpenAIRealtimeTranscriptionSession(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        websocket_connect=connect,
    )
    await session.start()
    append_task = asyncio.create_task(session.append_pcm24(b"\x00\x00" * 240))
    await send_started.wait()
    close_task = asyncio.create_task(session.close())
    await asyncio.sleep(0)
    assert ws.closed is False

    release_send.set()
    with pytest.raises(LiveTranscriptionError, match="closed"):
        await append_task
    await close_task
    assert ws.closed is True


@pytest.mark.asyncio
async def test_websocket_send_timeout_bounds_blocked_provider():
    ws = FakeWebSocket()
    await ws.incoming.put({"type": "session.updated"})
    block_send = [False]
    original_send = ws.send

    async def send(payload):
        if block_send[0]:
            await asyncio.Event().wait()
        await original_send(payload)

    ws.send = send

    async def connect(*_args, **_kwargs):
        return ws

    session = OpenAIRealtimeTranscriptionSession(
        api_key="test-key",
        config=LiveTranscriptionConfig(send_timeout_seconds=0.01),
        websocket_connect=connect,
    )
    await session.start()
    block_send[0] = True
    with pytest.raises(asyncio.TimeoutError):
        await session.append_pcm24(b"\x00\x00" * 240)
    await session.close()


@pytest.mark.asyncio
async def test_session_rolls_over_before_provider_sixty_minute_limit_between_turns():
    now = [0.0]
    sockets = [FakeWebSocket(), FakeWebSocket()]
    for ws in sockets:
        await ws.incoming.put({"type": "transcription_session.updated"})
    calls = []

    async def connect(*args, **kwargs):
        calls.append((args, kwargs))
        return sockets[len(calls) - 1]

    session = OpenAIRealtimeTranscriptionSession(
        api_key="test-key",
        config=LiveTranscriptionConfig(max_session_seconds=3300),
        websocket_connect=connect,
        clock=lambda: now[0],
    )
    await session.start()
    now[0] = 3301.0

    assert await session.rollover_if_due() is True
    assert sockets[0].closed is True
    assert len(calls) == 2
    assert session.rollover_count == 1
    await session.close()


@pytest.mark.asyncio
async def test_async_provider_error_without_pending_commit_fails_next_append_immediately():
    ws = FakeWebSocket()
    await ws.incoming.put({"type": "transcription_session.updated"})

    async def connect(*args, **kwargs):
        return ws

    session = OpenAIRealtimeTranscriptionSession(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        websocket_connect=connect,
    )
    await session.start()
    await ws.incoming.put(
        {
            "type": "error",
            "error": {"code": "bad_audio", "message": "secret-value"},
        }
    )
    await asyncio.sleep(0)

    with pytest.raises(LiveTranscriptionError, match="bad_audio") as exc:
        await session.append_pcm24(struct.pack("<h", 1))
    assert "secret-value" not in str(exc.value)
    await session.close()


@pytest.mark.asyncio
async def test_duplicate_commit_ack_fails_closed_without_turn_misrouting():
    ws = FakeWebSocket()
    await ws.incoming.put({"type": "transcription_session.updated"})

    async def connect(*_args, **_kwargs):
        return ws

    session = OpenAIRealtimeTranscriptionSession(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        websocket_connect=connect,
    )
    await session.start()
    await session.append_pcm24(b"\x01\x00" * 240)
    first = asyncio.create_task(session.commit())
    await _wait_for_sent_type(ws, "input_audio_buffer.commit", 1)
    await session.append_pcm24(b"\x02\x00" * 240)
    second = asyncio.create_task(session.commit())
    await _wait_for_sent_type(ws, "input_audio_buffer.commit", 2)

    await ws.incoming.put(
        {"type": "input_audio_buffer.committed", "item_id": "duplicate"}
    )
    await ws.incoming.put(
        {"type": "input_audio_buffer.committed", "item_id": "duplicate"}
    )

    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(result, LiveTranscriptionError) for result in results)
    assert session._pending_by_item == {}
    await session.close()


@pytest.mark.asyncio
async def test_unknown_early_items_are_bounded_and_fail_closed():
    ws = FakeWebSocket()
    await ws.incoming.put({"type": "transcription_session.updated"})

    async def connect(*_args, **_kwargs):
        return ws

    session = OpenAIRealtimeTranscriptionSession(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        websocket_connect=connect,
    )
    await session.start()
    for index in range(300):
        await ws.incoming.put(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": f"unknown-{index}",
                "delta": "x",
            }
        )
    for _ in range(100):
        if session._terminal_error is not None:
            break
        await asyncio.sleep(0)

    assert session._terminal_error is not None
    assert len(session._early_events) <= session.MAX_EARLY_ITEM_IDS
    await session.close()


class FakeRealtimeSession:
    def __init__(self, result=None):
        self.appended = []
        self.append_attempts = 0
        self.generation = 1
        self.commit_calls = 0
        self.clear_calls = 0
        self.close_calls = 0
        self.append_error: Exception | None = None
        self.result = result or {
            "success": True,
            "transcript": "Hallo Hermes",
            "provider": "openai_realtime",
            "model": "gpt-live-transcribe",
            "delay": "high",
            "item_id": "item_1",
        }

    async def append_pcm24(self, pcm):
        self.append_attempts += 1
        if self.append_error is not None:
            raise self.append_error
        self.appended.append(pcm)

    async def commit(self):
        self.commit_calls += 1
        return dict(self.result)

    async def clear(self):
        self.clear_calls += 1

    async def close(self):
        self.close_calls += 1


@pytest.mark.asyncio
async def test_controller_commits_only_when_full_source_utterance_was_streamed():
    fake = FakeRealtimeSession()
    controller = DiscordLiveTranscriptionController(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        session_factory=lambda: fake,
    )
    source = struct.pack("<8h", 100, 300, 0, 0, -200, -400, 0, 0)
    await controller.append_pcm48(42, source)
    result = await controller.finish_utterance(42, expected_source_bytes=len(source))
    assert result["success"] is True
    assert result["transcript"] == "Hallo Hermes"
    assert fake.commit_calls == 1
    assert len(fake.appended) == 1
    assert struct.unpack("<2h", fake.appended[0]) == (100, -150)


@pytest.mark.asyncio
async def test_controller_fails_closed_when_initial_pcm_was_not_streamed():
    fake = FakeRealtimeSession()
    controller = DiscordLiveTranscriptionController(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        session_factory=lambda: fake,
    )
    source = struct.pack("<8h", 100, 300, 0, 0, -200, -400, 0, 0)
    await controller.append_pcm48(42, source)
    result = await controller.finish_utterance(
        42,
        expected_source_bytes=len(source) + 3840,
    )
    assert result["success"] is False
    assert result["error"] == "incomplete_live_stream"
    assert result["provider"] == "openai_realtime"
    assert fake.commit_calls == 0
    assert fake.clear_calls == 1


@pytest.mark.asyncio
async def test_controller_drops_failed_stream_session_before_next_turn():
    sessions = [FakeRealtimeSession(), FakeRealtimeSession()]
    first = sessions[0]
    first.append_error = LiveTranscriptionError("bad_audio")

    controller = DiscordLiveTranscriptionController(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        session_factory=lambda: sessions.pop(0),
    )
    source = struct.pack("<8h", 100, 300, 0, 0, -200, -400, 0, 0)
    await controller.append_pcm48(42, source)
    await controller.append_pcm48(42, source)
    result = await controller.finish_utterance(
        42,
        expected_source_bytes=len(source),
    )

    assert result["success"] is False
    assert result["error"] == "live_stream_failed"
    assert first.append_attempts == 1
    assert first.close_calls == 1
    assert controller._sessions == {}


@pytest.mark.asyncio
async def test_controller_rejects_pcm_split_across_session_generations():
    session = FakeRealtimeSession()
    controller = DiscordLiveTranscriptionController(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        session_factory=lambda: session,
    )
    source = struct.pack("<8h", 100, 300, 0, 0, -200, -400, 0, 0)

    await controller.append_pcm48(42, source)
    session.generation = 2
    await controller.append_pcm48(42, source)
    result = await controller.finish_utterance(
        42,
        expected_source_bytes=len(source) * 2,
    )

    assert result["success"] is False
    assert result["error"] == "live_stream_failed"
    assert session.close_calls == 1


@pytest.mark.asyncio
async def test_controller_closes_all_per_user_sessions():
    sessions = []

    def factory():
        session = FakeRealtimeSession()
        sessions.append(session)
        return session

    controller = DiscordLiveTranscriptionController(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        session_factory=factory,
    )
    source = struct.pack("<4h", 100, 300, 0, 0)
    await controller.append_pcm48(1, source)
    await controller.append_pcm48(2, source)
    await controller.close()
    assert len(sessions) == 2
    assert [session.close_calls for session in sessions] == [1, 1]


@pytest.mark.asyncio
async def test_controller_close_linearizes_against_blocked_append():
    append_started = asyncio.Event()
    release_append = asyncio.Event()

    class BlockingSession(FakeRealtimeSession):
        async def append_pcm24(self, pcm):
            append_started.set()
            await release_append.wait()
            await super().append_pcm24(pcm)

    session = BlockingSession()
    controller = DiscordLiveTranscriptionController(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        session_factory=lambda: session,
    )
    source = struct.pack("<4h", 100, 300, 0, 0)
    append_task = asyncio.create_task(controller.append_pcm48(42, source))
    await append_started.wait()
    close_task = asyncio.create_task(controller.close())
    await asyncio.sleep(0)
    release_append.set()

    with pytest.raises(LiveTranscriptionError, match="closed"):
        await append_task
    await close_task
    assert controller._sessions == {}
    assert controller._source_bytes == {}


@pytest.mark.asyncio
async def test_closed_controller_cannot_create_new_user_sessions():
    sessions = []

    def factory():
        session = FakeRealtimeSession()
        sessions.append(session)
        return session

    controller = DiscordLiveTranscriptionController(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        session_factory=factory,
    )
    await controller.close()

    with pytest.raises(LiveTranscriptionError, match="closed"):
        await controller.append_pcm48(42, b"\x00" * 8)
    assert sessions == []


@pytest.mark.asyncio
async def test_controller_caps_concurrent_user_sessions():
    controller = DiscordLiveTranscriptionController(
        api_key="test-key",
        config=LiveTranscriptionConfig(),
        session_factory=FakeRealtimeSession,
    )
    source = struct.pack("<4h", 100, 300, 0, 0)
    for user_id in range(1, 6):
        await controller.append_pcm48(user_id, source)

    assert len(controller._sessions) == controller.MAX_SESSIONS
    result = await controller.finish_utterance(
        5,
        expected_source_bytes=len(source),
    )
    assert result["success"] is False
    assert result["error"] == "live_stream_failed"
    await controller.close()


@pytest.mark.parametrize("base_url", [
    "https://foreign.invalid/v1", "http://api.openai.com/v1", "https://api.openai.com/V1",
    "https://user@api.openai.com/v1", "https://api.openai.com:444/v1",
    "https://api.openai.com/v1?other=1", "https://api.openai.com/v1#other",
])
def test_foreign_configured_endpoint_never_repurposes_key(monkeypatch, base_url):
    reads = []
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda name: reads.append(name) or "other-route-key")
    with pytest.raises(ValueError, match="endpoint"):
        resolve_openai_realtime_api_key({"stt": {"openai": {"api_key": "foreign-key", "base_url": base_url}}})
    assert reads == []


@pytest.mark.parametrize("endpoint", [
    "wss://user@api.openai.com/v1/realtime?intent=transcription",
    "wss://api.openai.com:444/v1/realtime?intent=transcription",
    "wss://api.openai.com/v1/realtime?intent=transcription&extra=1",
    "wss://api.openai.com/v1/realtime?intent=transcription#fragment",
])
def test_live_endpoint_has_one_fixed_authority(endpoint):
    with pytest.raises(ValueError, match="endpoint"):
        LiveTranscriptionConfig(endpoint=endpoint)


@pytest.mark.asyncio
async def test_installed_websocket_redirect_headers_and_protocol_logging(caplog, record_property, tmp_path):
    import inspect
    import websockets
    from websockets.asyncio.client import connect
    from websockets.client import ClientProtocol
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus, SecurityError
    from websockets.http11 import Response
    from websockets.uri import parse_uri
    from websockets.protocol import Protocol, CLIENT
    from websockets.frames import Frame, OP_TEXT

    record_property("websockets_version", websockets.__version__)
    record_property("websockets_path", inspect.getfile(connect))
    assert "additional_headers" in inspect.signature(connect).parameters
    headers = {"Authorization": "Bearer OFFLINE_HEADER_SENTINEL"}
    connector = live_transcription._create_no_redirect_websocket_connect(DEFAULT_ENDPOINT := LiveTranscriptionConfig().endpoint,
                                                                       additional_headers=headers)
    for target in [DEFAULT_ENDPOINT, "wss://foreign.invalid/capture"]:
        response = Response(302, "Found", Headers({"Location": target}))
        assert isinstance(connector.process_redirect(InvalidStatus(response)), SecurityError)
        assert connector.uri == DEFAULT_ENDPOINT
        assert connector.additional_headers is headers
    logger = live_transcription._OPENAI_WEBSOCKET_LOGGER
    with caplog.at_level(logging.DEBUG):
        protocol = ClientProtocol(parse_uri(DEFAULT_ENDPOINT), logger=logger)
        request = protocol.connect()
        request.headers.update(headers)
        protocol.send_request(request)
        frames = Protocol(CLIENT, logger=logger)
        frames.send_text(b"OFFLINE_OUTGOING_SENTINEL")
        frames.receive_data(Frame(OP_TEXT, b"OFFLINE_INCOMING_SENTINEL").serialize(mask=False))
    assert "OFFLINE_HEADER_SENTINEL" not in caplog.text
    assert "OFFLINE_OUTGOING_SENTINEL" not in caplog.text
    assert "OFFLINE_INCOMING_SENTINEL" not in caplog.text
    (tmp_path / "websockets-contract.json").write_text(json.dumps({
        "version": websockets.__version__, "path": inspect.getfile(connect),
        "additional_headers_supported": True, "same_and_cross_origin_redirects_rejected": True,
        "header_identity_preserved_without_redirect": True, "protocol_markers_not_logged": True,
    }))


@pytest.mark.parametrize("operation", ["append", "commit"])
@pytest.mark.asyncio
async def test_session_admission_rechecked_after_send_lock(operation):
    ws = FakeWebSocket()
    await ws.incoming.put({"type": "session.updated"})
    async def connect(*args, **kwargs):
        return ws
    session = OpenAIRealtimeTranscriptionSession(api_key="synthetic", config=LiveTranscriptionConfig(), websocket_connect=connect)
    await session.start()
    allowed = [True]
    await session._send_lock.acquire()
    task = asyncio.create_task(session.append_pcm24(b"\x00" * 8, admit=lambda: allowed[0]) if operation == "append"
                               else session.commit(admit=lambda: allowed[0]))
    await asyncio.sleep(0)
    allowed[0] = False
    session._send_lock.release()
    try:
        with pytest.raises(LiveTranscriptionError, match="admi"):
            await task
        assert [e["type"] for e in ws.sent] == ["session.update"]
    finally:
        await session.close()


@pytest.mark.parametrize("field,value", [
    ("keywords", ["\nword"]), ("keywords", [1]), ("languages", [None]),
])
def test_live_context_rejects_raw_invalid_values(field, value):
    with pytest.raises(ValueError):
        LiveTranscriptionConfig.from_hermes_config({"stt": {"openai": {field: value}}})


@pytest.mark.parametrize("field,value", [
    ("send_timeout_seconds", float("nan")), ("send_timeout_seconds", 6),
    ("completion_timeout_seconds", float("inf")), ("completion_timeout_seconds", 21),
    ("max_session_seconds", 3400),
])
def test_config_cannot_raise_live_resource_caps(field, value):
    with pytest.raises(ValueError):
        LiveTranscriptionConfig(**{field: value})
