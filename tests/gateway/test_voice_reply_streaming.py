"""Background-completion voice parity: a typed turn whose reply lands in the runner's
auto voice-reply lane (``GatewayVoiceMixin._send_voice_reply``) must STREAM the text into
the bound voice channel instead of synthesizing one whole file and sending it as a voice
message. Background task / delegation completions arrive as TEXT events, so the adapter's
voice-input auto-TTS gate never fires; the runner lane is the only voice surface they get.
When streaming takes ownership the pre-synthesized file audio is never delivered; any miss
(unbound chat, streaming disabled, exception) falls back to the legacy file path fail-open.
"""

import asyncio
from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_event():
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="999",
        user_id="42",
        chat_type="dm",
    )
    return MessageEvent(
        text="Background task finished: report ready.",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )


def _runner_with_adapter(adapter):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    return runner


def test_voice_reply_streams_into_bound_vc_instead_of_voice_message(monkeypatch):
    """Bound + in-VC + streaming available: text streams, no file audio is ever synthesized."""
    streams = []
    synth_calls = []

    class _Adapter:
        _voice_text_channels = {123: "999"}

        def is_in_voice_channel(self, gid):
            return gid == 123

        async def play_reply_streaming_in_voice(self, gid, text):
            streams.append((gid, text))
            return True

    def _fail_synth(*_a, **_k):
        raise AssertionError("whole-file synthesis must not run when streaming takes ownership")

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", _fail_synth)
    runner = _runner_with_adapter(_Adapter())
    event = _make_event()
    asyncio.run(runner._send_voice_reply(event, event.text))
    assert streams == [(123, "Background task finished: report ready.")]


def test_voice_reply_falls_back_to_file_when_stream_declines(monkeypatch):
    """Streaming enabled but unavailable (returns False): legacy voice-message path runs."""
    streams = []

    class _Adapter:
        _voice_text_channels = {123: "999"}

        def is_in_voice_channel(self, gid):
            return gid == 123

        async def play_reply_streaming_in_voice(self, gid, text):
            streams.append((gid, text))
            return False

        async def send_voice(self, chat_id, audio_path, reply_to=None, metadata=None):
            return SimpleNamespace(success=True)

    synth_calls = []

    def _fake_synth(*, text, output_path, **_kwargs):
        synth_calls.append(text)
        return SimpleNamespace(success=False, error="no provider")

    monkeypatch.setattr(
        "gateway.run_voice.build_auto_tts_output_path",
        lambda platform: "/tmp/hermes_voice_test_reply.ogg",
    )
    # The real _send_voice_reply imports text_to_speech_tool inside the method; patch the source.
    import tools.tts_tool as tts_tool_mod
    monkeypatch.setattr(tts_tool_mod, "text_to_speech_tool",
                        lambda *, text, output_path, **_k: '{"success": false, "error": "no provider"}')
    runner = _runner_with_adapter(_Adapter())
    event = _make_event()
    # Must not raise: synthesis "fails" (no provider), the legacy lane handled it fail-open.
    asyncio.run(runner._send_voice_reply(event, event.text))
    assert len(streams) == 1  # exactly one streaming attempt, made BEFORE synthesis


def test_voice_reply_falls_back_when_streaming_raises(monkeypatch):
    """A streaming crash is fail-open: the legacy file lane still runs."""

    class _Adapter:
        _voice_text_channels = {123: "999"}

        def is_in_voice_channel(self, gid):
            return gid == 123

        async def play_reply_streaming_in_voice(self, gid, text):
            raise RuntimeError("boom")

        async def send_voice(self, chat_id, audio_path, reply_to=None, metadata=None):
            return SimpleNamespace(success=True)

    monkeypatch.setattr(
        "gateway.run_voice.build_auto_tts_output_path",
        lambda platform: "/tmp/hermes_voice_test_reply.ogg",
    )
    import tools.tts_tool as tts_tool_mod
    monkeypatch.setattr(tts_tool_mod, "text_to_speech_tool",
                        lambda *, text, output_path, **_k: '{"success": false, "error": "no provider"}')
    runner = _runner_with_adapter(_Adapter())
    event = _make_event()
    # The streaming failure is swallowed; the legacy lane runs (and fails open on synthesis).
    asyncio.run(runner._send_voice_reply(event, event.text))


def test_voice_reply_skips_streaming_when_chat_not_bound(monkeypatch):
    """Chat not in _voice_text_channels: no streaming call, straight to the legacy lane."""
    streams = []

    class _Adapter:
        _voice_text_channels = {123: "555"}

        def is_in_voice_channel(self, gid):
            return True

        async def play_reply_streaming_in_voice(self, gid, text):
            streams.append((gid, text))
            return True

        async def send_voice(self, chat_id, audio_path, reply_to=None, metadata=None):
            return SimpleNamespace(success=True)

    monkeypatch.setattr(
        "gateway.run_voice.build_auto_tts_output_path",
        lambda platform: "/tmp/hermes_voice_test_reply.ogg",
    )
    import tools.tts_tool as tts_tool_mod
    monkeypatch.setattr(tts_tool_mod, "text_to_speech_tool",
                        lambda *, text, output_path, **_k: '{"success": false, "error": "no provider"}')
    runner = _runner_with_adapter(_Adapter())
    event = _make_event()
    asyncio.run(runner._send_voice_reply(event, event.text))
    assert streams == []


def test_voice_reply_skips_streaming_without_voice_surface(monkeypatch):
    """Adapter without any voice attributes (e.g. Telegram): behaves exactly as before."""

    class _Adapter:
        async def send_voice(self, chat_id, audio_path, reply_to=None, metadata=None):
            return SimpleNamespace(success=True)

    monkeypatch.setattr(
        "gateway.run_voice.build_auto_tts_output_path",
        lambda platform: "/tmp/hermes_voice_test_reply.ogg",
    )
    import tools.tts_tool as tts_tool_mod
    monkeypatch.setattr(tts_tool_mod, "text_to_speech_tool",
                        lambda *, text, output_path, **_k: '{"success": false, "error": "no provider"}')
    runner = _runner_with_adapter(_Adapter())
    event = _make_event()
    # Must not raise from the streaming probe; legacy lane runs.
    asyncio.run(runner._send_voice_reply(event, event.text))