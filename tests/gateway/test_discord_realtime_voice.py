import base64
import json
import queue

from gateway.platforms.discord_realtime import (
    DISCORD_FRAME_BYTES,
    OpenAIRealtimeDiscordBridge,
    QueuePCMAudioSource,
    _make_discord_audio_source,
    discord_pcm_to_realtime,
    realtime_pcm_to_discord,
)


def test_discord_pcm_to_realtime_halves_rate_and_mixes_to_mono():
    # 20 ms of 48 kHz stereo PCM16: 960 frames * 2 channels * 2 bytes.
    discord_pcm = b"\x00\x00\x01\x00" * 960
    realtime_pcm = discord_pcm_to_realtime(discord_pcm)
    # 20 ms of 24 kHz mono PCM16: 480 frames * 1 channel * 2 bytes.
    assert len(realtime_pcm) == 960


def test_realtime_pcm_to_discord_upsamples_and_expands_to_stereo():
    # 20 ms of 24 kHz mono PCM16: 480 frames * 2 bytes. audioop.ratecv may
    # produce one frame less for a stateless chunk; QueuePCMAudioSource pads the
    # final Discord frame boundary.
    realtime_pcm = b"\x01\x00" * 480
    discord_pcm = realtime_pcm_to_discord(realtime_pcm)
    assert DISCORD_FRAME_BYTES - 4 <= len(discord_pcm) <= DISCORD_FRAME_BYTES
    assert len(discord_pcm) % 4 == 0


def test_queue_pcm_audio_source_returns_20ms_frames_and_pads_short_reads():
    q = queue.Queue()
    q.put(b"\x01" * 100)
    source = QueuePCMAudioSource(q)
    frame = source.read()
    assert len(frame) == DISCORD_FRAME_BYTES
    assert frame.startswith(b"\x01" * 100)
    assert frame[100:] == b"\x00" * (DISCORD_FRAME_BYTES - 100)
    assert source.is_opus() is False


def test_queue_pcm_audio_source_ends_when_idle_instead_of_speaking_silence():
    source = QueuePCMAudioSource(queue.Queue())

    assert source.read() == b""


def test_gpt_realtime_2_uses_ga_session_shape():
    bridge = OpenAIRealtimeDiscordBridge(
        api_key="test",
        voice_client=None,
        model="gpt-realtime-2",
        voice="alloy",
        instructions="hi",
    )
    payload = bridge._session_update_payload()
    assert bridge._uses_ga_api() is True
    assert payload["session"]["type"] == "realtime"
    assert payload["session"]["model"] == "gpt-realtime-2"
    assert payload["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert payload["session"]["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert "modalities" not in payload["session"]
    assert "input_audio_format" not in payload["session"]


def test_legacy_realtime_models_use_beta_session_shape():
    bridge = OpenAIRealtimeDiscordBridge(
        api_key="test",
        voice_client=None,
        model="gpt-realtime",
        voice="alloy",
        instructions="hi",
    )
    payload = bridge._session_update_payload()
    assert bridge._uses_ga_api() is False
    assert payload["session"]["modalities"] == ["audio", "text"]
    assert payload["session"]["input_audio_format"] == "pcm16"
    assert payload["session"]["output_audio_format"] == "pcm16"
    assert "audio" not in payload["session"]


def test_make_discord_audio_source_passes_discord_py_audio_source_check():
    import discord

    q = queue.Queue()
    source = _make_discord_audio_source(q)

    audio_source_type = getattr(discord, "AudioSource", None)
    if isinstance(audio_source_type, type):
        assert isinstance(source, audio_source_type)
    assert source.is_opus() is False


class _FakeVoiceClient:
    def __init__(self):
        self.played_source = None
        self.stopped = False

    def is_connected(self):
        return True

    def is_playing(self):
        return False

    def stop(self):
        self.stopped = True

    def play(self, source, after=None):
        import discord

        audio_source_type = getattr(discord, "AudioSource", None)
        if isinstance(audio_source_type, type):
            assert isinstance(source, audio_source_type)
        self.played_source = source
        self.after = after


def test_start_discord_playback_uses_real_discord_audio_source():
    voice_client = _FakeVoiceClient()
    bridge = OpenAIRealtimeDiscordBridge(api_key="test", voice_client=voice_client)

    bridge._start_discord_playback()

    assert voice_client.played_source is not None


def test_enqueue_output_lazily_starts_discord_playback():
    voice_client = _FakeVoiceClient()
    bridge = OpenAIRealtimeDiscordBridge(api_key="test", voice_client=voice_client)

    assert voice_client.played_source is None

    bridge._enqueue_output(b"\x01" * DISCORD_FRAME_BYTES)

    assert voice_client.played_source is not None


class _FakeRealtimeWebSocket:
    def __init__(self, frames):
        self._frames = queue.Queue()
        for frame in frames:
            self._frames.put(json.dumps(frame))
        self.closed = False

    def recv(self, timeout=None):
        return self._frames.get_nowait()

    def close(self):
        self.closed = True


def test_recv_loop_accepts_ga_output_audio_delta_event_name():
    voice_client = _FakeVoiceClient()
    bridge = OpenAIRealtimeDiscordBridge(api_key="test", voice_client=voice_client)
    bridge._ws = _FakeRealtimeWebSocket([
        {"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x01\x00" * 480).decode()},
        {"type": "response.done"},
    ])

    bridge._recv_loop()

    assert bridge.output_bytes > 0
    assert bridge.last_output_at is not None
    assert voice_client.played_source is not None
