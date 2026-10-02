"""Native inbound-audio buffer: per-session isolation, one-shot consumption, mode gate.

Mirrors tests/gateway/test_native_image_buffer_isolation.py for the audio half of
_prepare_inbound_message_text → TurnRunner._native_media_run_message.
"""

from __future__ import annotations

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource, build_session_key


def _make_runner(media_cfg: dict | None = None) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda **_: "native"
    # Keep tests off the live config file: default = auto (buffer), overridable per test.
    cfg = {"media": {"native_audio": "auto"}} if media_cfg is None else media_cfg
    runner._hermes_config_readonly = lambda: cfg

    async def _no_enrich(event, source, message_text, audio_paths):
        return message_text

    runner._enrich_inbound_voice = _no_enrich  # STT is exercised elsewhere; buffer is under test
    return runner


def _source(chat_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="private",
        user_name=f"user-{chat_id}",
    )


def _voice_event(source: SessionSource, path: str) -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=[path],
        media_types=["audio/ogg"],
    )


async def _prepare(runner: GatewayRunner, event: MessageEvent, source: SessionSource, **kw):
    return await runner._prepare_inbound_message_text(event=event, source=source, history=[], **kw)


# ─── buffering ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_voice_paths_buffered_per_session():
    runner = _make_runner()
    source_a, source_b = _source("chat-a"), _source("chat-b")

    await _prepare(runner, _voice_event(source_a, "/tmp/a.ogg"), source_a)
    await _prepare(runner, _voice_event(source_b, "/tmp/b.ogg"), source_b)

    assert runner._consume_pending_native_audio_paths(build_session_key(source_a)) == ["/tmp/a.ogg"]
    assert runner._consume_pending_native_audio_paths(build_session_key(source_b)) == ["/tmp/b.ogg"]


@pytest.mark.asyncio
async def test_buffer_not_cleared_by_other_sessions_without_voice():
    runner = _make_runner()
    source_a, source_b = _source("chat-a"), _source("chat-b")

    await _prepare(runner, _voice_event(source_a, "/tmp/a.ogg"), source_a)
    await _prepare(runner, MessageEvent(text="plain text", source=source_b), source_b)

    assert runner._consume_pending_native_audio_paths(build_session_key(source_a)) == ["/tmp/a.ogg"]
    assert runner._consume_pending_native_audio_paths(build_session_key(source_b)) == []


@pytest.mark.asyncio
async def test_buffer_consumed_once_not_reattached_next_turn():
    runner = _make_runner()
    source = _source("chat-a")
    await _prepare(runner, _voice_event(source, "/tmp/a.ogg"), source)

    key = build_session_key(source)
    assert runner._consume_pending_native_audio_paths(key) == ["/tmp/a.ogg"]
    assert runner._consume_pending_native_audio_paths(key) == []  # one-shot


@pytest.mark.asyncio
async def test_new_prepare_resets_previous_buffer_for_same_session():
    runner = _make_runner()
    source = _source("chat-a")
    await _prepare(runner, _voice_event(source, "/tmp/stale.ogg"), source)
    await _prepare(runner, MessageEvent(text="next turn", source=source), source)

    assert runner._consume_pending_native_audio_paths(build_session_key(source)) == []


@pytest.mark.asyncio
async def test_uses_resolved_session_key_when_provided():
    runner = _make_runner()
    source = _source("chat-a")
    runner._session_key_for_source = lambda _source: "source-derived-key"

    await _prepare(runner, _voice_event(source, "/tmp/a.ogg"), source, session_key="canonical-session-key")

    assert runner._consume_pending_native_audio_paths("source-derived-key") == []
    assert runner._consume_pending_native_audio_paths("canonical-session-key") == ["/tmp/a.ogg"]


@pytest.mark.asyncio
async def test_already_transcribed_event_falls_back_to_event_paths():
    """STT ran earlier (``_gateway_pending_stt_text``), so classify yields no audio_paths —
    the clip must still buffer from the event's own STT-eligible paths."""
    runner = _make_runner()
    source = _source("chat-a")
    event = _voice_event(source, "/tmp/pending.ogg")
    event._gateway_pending_stt_text = '"transcribed earlier"'

    await _prepare(runner, event, source)

    assert runner._consume_pending_native_audio_paths(build_session_key(source)) == ["/tmp/pending.ogg"]


@pytest.mark.asyncio
async def test_native_audio_off_does_not_buffer():
    runner = _make_runner({"media": {"native_audio": "off"}})
    source = _source("chat-a")

    await _prepare(runner, _voice_event(source, "/tmp/a.ogg"), source)

    assert runner._consume_pending_native_audio_paths(build_session_key(source)) == []


@pytest.mark.asyncio
async def test_audio_file_attachment_does_not_buffer():
    """MessageType.AUDIO is a file attachment (audio_file_paths), never native STT audio."""
    runner = _make_runner()
    source = _source("chat-a")
    event = MessageEvent(
        text="song",
        message_type=MessageType.AUDIO,
        source=source,
        media_urls=["/tmp/song.mp3"],
        media_types=["audio/mpeg"],
    )

    await _prepare(runner, event, source)

    assert runner._consume_pending_native_audio_paths(build_session_key(source)) == []


# ─── TurnRunner._native_media_run_message ───────────────────────────────────


class _Ctx:
    def __init__(self, message, session_key="k"):
        self.message = message
        self.session_key = session_key


def _turn_runner(imgs, auds, message="listen to this", agent=None) -> tuple:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._consume_pending_native_image_paths = lambda _key: list(imgs)
    runner._consume_pending_native_audio_paths = lambda _key: list(auds)
    tr = TurnRunner(runner, _Ctx(message))
    if agent is None:
        from types import SimpleNamespace

        agent = SimpleNamespace(api_mode="chat_completions", provider="openai")
    return tr, agent


@pytest.mark.asyncio
async def test_media_message_merges_image_and_audio(tmp_path, monkeypatch):
    import base64
    import wave

    wav = tmp_path / "v.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(b"\x00\x00" * 80)
    png = tmp_path / "i.png"
    png.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"media": {"native_audio": "auto"}})

    tr, agent = _turn_runner([str(png)], [str(wav)])
    out = tr._native_media_run_message(agent)

    assert [p["type"] for p in out] == ["text", "image_url", "input_audio"]
    assert str(wav.name) in out[0]["text"]
    assert base64.b64decode(out[2]["input_audio"]["data"]) == wav.read_bytes()
    # both buffers consumed even though the runner double returns fresh lists each call
    assert tr._ctx.message == "listen to this"  # ctx untouched


def test_audio_only_message(monkeypatch, tmp_path):
    import wave

    wav = tmp_path / "v.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(b"\x00\x00" * 80)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"media": {"native_audio": "auto"}})

    tr, agent = _turn_runner([], [str(wav)])
    out = tr._native_media_run_message(agent)

    assert [p["type"] for p in out] == ["text", "input_audio"]
    assert "[Voice message attached as audio:" in out[0]["text"]


def test_unsupported_backend_keeps_text_note(monkeypatch, tmp_path):
    from types import SimpleNamespace

    wav = tmp_path / "v.wav"
    wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVE")  # never read: gate rejects first
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"media": {"native_audio": "auto"}})

    tr, agent = _turn_runner([], [str(wav)], agent=SimpleNamespace(api_mode="anthropic_messages", provider="anthropic"))
    assert tr._native_media_run_message(agent) == "listen to this"


def test_no_buffered_media_returns_plain_text():
    tr, agent = _turn_runner([], [])
    assert tr._native_media_run_message(agent) == "listen to this"


def test_mode_off_skips_audio_but_keeps_images(monkeypatch, tmp_path):
    import wave

    wav = tmp_path / "v.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"media": {"native_audio": "off"}})

    tr, agent = _turn_runner([], [str(wav)])
    assert tr._native_media_run_message(agent) == "listen to this"
