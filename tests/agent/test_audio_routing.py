"""Tests for agent/audio_routing.py — native inbound audio (input_audio) routing.

Covers the mode gate (auto/on/off × api_mode × provider), format sniffing, the
clip-preparation ceilings (8 MB / 600 s) and ffmpeg normalization, the native
content-part build, and the send-path strip for backends that cannot take audio.
"""

from __future__ import annotations

import base64
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import audio_routing
from agent.audio_routing import (
    AUDIO_PART_TYPES,
    MAX_NATIVE_AUDIO_BYTES,
    MAX_NATIVE_AUDIO_SECONDS,
    audio_input_mode,
    build_native_audio_parts,
    is_audio_part,
    native_audio_supported,
    sniff_audio_format,
    strip_unsupported_audio_parts,
)


def _cfg(mode: object) -> dict:
    return {"media": {"native_audio": mode}}


def _make_wav(path: Path, frames: int = 160, rate: int = 8000) -> Path:
    """Tiny real PCM wav (0.02 s) — passes sniffing and the stdlib duration probe."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * frames)
    return path


def _agent(api_mode: str = "chat_completions", provider: str = "openai",
           model: str = "gpt-4o", rejecting: set | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        api_mode=api_mode, provider=provider, model=model,
        _audio_rejecting_models=set() if rejecting is None else rejecting,
    )


def _audio_part(fmt: str = "wav") -> dict:
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(b"RIFF").decode("ascii"), "format": fmt}}


# ─── mode normalization ─────────────────────────────────────────────────────


class TestAudioInputMode:
    @pytest.mark.parametrize("raw,expected", [
        ("on", "on"), ("OFF", "off"), ("Auto", "auto"),
        ("native", "on"), ("text", "off"),  # image-routing vocabulary aliases
        ("true", "on"), ("no", "off"), ("1", "on"), ("0", "off"),
        (True, "on"), (False, "off"),
        ("nonsense", "auto"), ("", "auto"), (None, "auto"), (42, "auto"),
    ])
    def test_coercion(self, raw, expected):
        assert audio_input_mode(_cfg(raw)) == expected

    @pytest.mark.parametrize("cfg", [None, {}, {"media": None}, {"media": "on"}, {}])
    def test_absent_or_malformed_is_auto(self, cfg):
        assert audio_input_mode(cfg) == "auto"

    def test_absent_media_key_is_auto(self):
        assert audio_input_mode({"stt": {"enabled": True}}) == "auto"


# ─── native_audio_supported: mode × api_mode × provider ─────────────────────


class TestNativeAudioSupported:
    # ``on`` attaches on every wire — the explicit override.
    @pytest.mark.parametrize("api_mode", [
        "chat_completions", "anthropic_messages", "codex_responses",
        "bedrock_converse", "",
    ])
    def test_on_always_supported(self, api_mode):
        assert native_audio_supported(_cfg("on"), api_mode=api_mode, provider="anthropic") is True

    # ``off`` never attaches, on any wire.
    @pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
    def test_off_never_supported(self, api_mode):
        assert native_audio_supported(_cfg("off"), api_mode=api_mode, provider="openai") is False

    def test_auto_chat_completions_openai_supported(self):
        assert native_audio_supported(_cfg("auto"), api_mode="chat_completions", provider="openai") is True

    @pytest.mark.parametrize("provider", [
        "anthropic", "gemini", "google", "bedrock", "vertex", "vertexai", "openai-codex",
    ])
    def test_auto_excluded_providers(self, provider):
        assert native_audio_supported(_cfg("auto"), api_mode="chat_completions", provider=provider) is False

    @pytest.mark.parametrize("api_mode", [
        "anthropic_messages", "codex_responses", "bedrock_converse", "codex_app_server", "",
    ])
    def test_auto_non_chat_completions_wires_unsupported(self, api_mode):
        assert native_audio_supported(_cfg("auto"), api_mode=api_mode, provider="openai") is False

    def test_auto_defaults_when_kwargs_absent(self):
        # No api_mode at all counts as unknown → text note (guessing wrong is a 4xx).
        assert native_audio_supported(_cfg("auto")) is False

    def test_provider_matching_is_case_insensitive(self):
        assert native_audio_supported(_cfg("auto"), api_mode="chat_completions", provider="Anthropic") is False


# ─── format sniffing ────────────────────────────────────────────────────────


class TestSniffAudioFormat:
    def test_riff_wave_magic(self, tmp_path: Path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"RIFF$\x00\x00\x00WAVEfmt ")
        assert sniff_audio_format(p) == "wav"

    def test_mp3_id3_tag(self, tmp_path: Path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00junk")
        assert sniff_audio_format(p) == "mp3"

    def test_mp3_frame_sync(self, tmp_path: Path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 16)
        assert sniff_audio_format(p) == "mp3"

    def test_extension_fallback_when_magic_unknown(self, tmp_path: Path):
        (tmp_path / "a.wav").write_bytes(b"\x00\x01\x02\x03")
        (tmp_path / "a.mp3").write_bytes(b"\x00\x01\x02\x03")
        assert sniff_audio_format(tmp_path / "a.wav") == "wav"
        assert sniff_audio_format(tmp_path / "a.mp3") == "mp3"

    def test_unrecognized_returns_none(self, tmp_path: Path):
        p = tmp_path / "a.amr"
        p.write_bytes(b"#!AMR\n" + b"\x00" * 8)
        assert sniff_audio_format(p) is None

    def test_missing_file_without_known_extension_returns_none(self, tmp_path: Path):
        # Extension fallback fires even for an absent path; _prepare_clip's is_file()
        # gate is what keeps a missing clip out of the parts list.
        assert sniff_audio_format(tmp_path / "nope.amr") is None
        assert sniff_audio_format(tmp_path / "nope.wav") == "wav"

    def test_real_wav_roundtrip(self, tmp_path: Path):
        assert sniff_audio_format(_make_wav(tmp_path / "v.wav")) == "wav"


# ─── clip preparation: ceilings + normalization ─────────────────────────────


class TestPrepareClipGates:
    def test_missing_file_is_skipped(self, tmp_path: Path):
        parts, skipped = build_native_audio_parts("hi", [str(tmp_path / "gone.wav")])
        assert skipped == [str(tmp_path / "gone.wav")]
        assert parts == [{"type": "text", "text": "hi"}]

    def test_empty_file_is_skipped(self, tmp_path: Path):
        p = tmp_path / "empty.wav"
        p.write_bytes(b"")
        _parts, skipped = build_native_audio_parts("hi", [str(p)])
        assert skipped == [str(p)]

    def test_over_8mb_is_skipped(self, tmp_path: Path):
        assert MAX_NATIVE_AUDIO_BYTES == 8 * 1024 * 1024
        p = tmp_path / "big.wav"
        p.write_bytes(b"\x00" * (MAX_NATIVE_AUDIO_BYTES + 1))
        parts, skipped = build_native_audio_parts("hi", [str(p)])
        assert skipped == [str(p)]
        assert not any(is_audio_part(x) for x in parts)

    def test_over_600s_is_skipped(self, tmp_path: Path, monkeypatch):
        assert MAX_NATIVE_AUDIO_SECONDS == 600.0
        p = _make_wav(tmp_path / "long.wav")
        monkeypatch.setattr(audio_routing, "probe_duration_seconds",
                            lambda _p: MAX_NATIVE_AUDIO_SECONDS + 1)
        _parts, skipped = build_native_audio_parts("hi", [str(p)])
        assert skipped == [str(p)]

    def test_at_duration_limit_still_attaches(self, tmp_path: Path, monkeypatch):
        p = _make_wav(tmp_path / "ok.wav")
        monkeypatch.setattr(audio_routing, "probe_duration_seconds",
                            lambda _p: MAX_NATIVE_AUDIO_SECONDS - 1)
        parts, skipped = build_native_audio_parts("hi", [str(p)])
        assert skipped == []
        assert sum(is_audio_part(x) for x in parts) == 1

    def test_duration_probe_failure_does_not_block(self, tmp_path: Path, monkeypatch):
        p = _make_wav(tmp_path / "ok.wav")
        monkeypatch.setattr(audio_routing, "probe_duration_seconds", lambda _p: None)
        _parts, skipped = build_native_audio_parts("hi", [str(p)])
        assert skipped == []

    def test_blocked_read_path_is_skipped(self, tmp_path: Path, monkeypatch):
        p = _make_wav(tmp_path / "v.wav")

        def _block(_path):
            raise ValueError("read blocked")

        monkeypatch.setattr("agent.file_safety.raise_if_read_blocked", _block)
        _parts, skipped = build_native_audio_parts("hi", [str(p)])
        assert skipped == [str(p)]


class TestNormalizationSelection:
    def test_wav_bypasses_ffmpeg(self, tmp_path: Path, monkeypatch):
        p = _make_wav(tmp_path / "v.wav")

        def _boom(_path):
            raise AssertionError("wav must not be transcoded")

        monkeypatch.setattr(audio_routing, "transcode_with_ffmpeg", _boom)
        parts, skipped = build_native_audio_parts("hi", [str(p)])
        assert skipped == []
        audio = [x for x in parts if is_audio_part(x)]
        assert audio[0]["input_audio"]["format"] == "wav"

    def test_mp3_bypasses_ffmpeg(self, tmp_path: Path, monkeypatch):
        p = tmp_path / "v.mp3"
        p.write_bytes(b"ID3\x04\x00" + b"\x00" * 32)
        monkeypatch.setattr(audio_routing, "transcode_with_ffmpeg",
                            lambda _p: (_ for _ in ()).throw(AssertionError("mp3 must not be transcoded")))
        parts, skipped = build_native_audio_parts("hi", [str(p)])
        assert skipped == []
        assert next(x for x in parts if is_audio_part(x))["input_audio"]["format"] == "mp3"

    def test_ogg_is_transcoded_to_mp3(self, tmp_path: Path, monkeypatch):
        p = tmp_path / "voice.ogg"
        p.write_bytes(b"OggS\x00\x02" + b"\x00" * 32)
        seen: dict = {}
        monkeypatch.setattr(audio_routing, "transcode_with_ffmpeg",
                            lambda path: seen.setdefault("path", str(path)) and None or b"FAKEMP3")
        parts, skipped = build_native_audio_parts("hi", [str(p)])
        assert skipped == []
        audio = [x for x in parts if is_audio_part(x)]
        assert len(audio) == 1
        assert audio[0]["input_audio"]["format"] == "mp3"
        assert base64.b64decode(audio[0]["input_audio"]["data"]) == b"FAKEMP3"
        assert seen["path"] == str(p)

    def test_untranscodable_clip_degrades_to_text(self, tmp_path: Path, monkeypatch):
        p = tmp_path / "voice.amr"
        p.write_bytes(b"#!AMR\n" + b"\x00" * 8)
        monkeypatch.setattr(audio_routing, "transcode_with_ffmpeg", lambda _p: None)
        parts, skipped = build_native_audio_parts("hi", [str(p)])
        assert skipped == [str(p)]
        assert parts == [{"type": "text", "text": "hi"}]

    def test_transcode_result_over_ceiling_is_skipped(self, tmp_path: Path, monkeypatch):
        p = tmp_path / "voice.ogg"
        p.write_bytes(b"OggS\x00\x02" + b"\x00" * 8)
        monkeypatch.setattr(audio_routing, "transcode_with_ffmpeg",
                            lambda _p: b"x" * (MAX_NATIVE_AUDIO_BYTES + 1))
        _parts, skipped = build_native_audio_parts("hi", [str(p)])
        assert skipped == [str(p)]

    def test_ffmpeg_missing_is_degrade_not_error(self, tmp_path: Path, monkeypatch):
        p = tmp_path / "voice.ogg"
        p.write_bytes(b"OggS\x00\x02" + b"\x00" * 8)
        monkeypatch.setattr(audio_routing.shutil, "which", lambda _name: None)
        assert audio_routing.transcode_with_ffmpeg(p) is None


# ─── build_native_audio_parts ───────────────────────────────────────────────


class TestBuildNativeAudioParts:
    def test_text_then_hint_then_audio(self, tmp_path: Path):
        p = str(_make_wav(tmp_path / "v.wav"))
        parts, skipped = build_native_audio_parts("what did I say?", [p])
        assert skipped == []
        assert parts[0] == {
            "type": "text",
            "text": f"what did I say?\n\n[Voice message attached as audio: {p}]",
        }
        assert [x["type"] for x in parts] == ["text", "input_audio"]
        assert base64.b64decode(parts[1]["input_audio"]["data"]) == Path(p).read_bytes()
        assert parts[1]["input_audio"]["format"] == "wav"

    def test_duplicate_paths_deduped(self, tmp_path: Path):
        p = str(_make_wav(tmp_path / "v.wav"))
        parts, skipped = build_native_audio_parts("hi", [p, p])
        assert skipped == []
        assert sum(is_audio_part(x) for x in parts) == 1

    def test_audio_only_turn_gets_hint_as_text(self, tmp_path: Path):
        p = str(_make_wav(tmp_path / "v.wav"))
        parts, _ = build_native_audio_parts("", [p])
        assert parts[0] == {"type": "text", "text": f"[Voice message attached as audio: {p}]"}

    def test_no_text_and_no_clip_yields_empty(self):
        parts, skipped = build_native_audio_parts("", [])
        assert parts == [] and skipped == []

    def test_part_types_recognized(self, tmp_path: Path):
        p = str(_make_wav(tmp_path / "v.wav"))
        parts, _ = build_native_audio_parts("hi", [p])
        assert all(is_audio_part(x) for x in parts[1:])
        assert AUDIO_PART_TYPES == frozenset({"input_audio", "audio"})
        assert is_audio_part({"type": "text", "text": "x"}) is False
        assert is_audio_part("input_audio") is False
        assert is_audio_part({"type": "audio"}) is True  # generic spelling


# ─── strip_unsupported_audio_parts ──────────────────────────────────────────


class TestStripUnsupportedAudio:
    @staticmethod
    def _msgs():
        return [{"role": "user", "content": [{"type": "text", "text": "listen"}, _audio_part()]}]

    def test_supported_backend_untouched(self, monkeypatch):
        monkeypatch.setattr(audio_routing, "_load_cfg_readonly", lambda: _cfg("auto"))
        msgs = self._msgs()
        assert strip_unsupported_audio_parts(_agent("chat_completions", "openai"), msgs) == 0
        assert is_audio_part(msgs[0]["content"][1])

    @pytest.mark.parametrize("api_mode", ["anthropic_messages", "codex_responses", "bedrock_converse"])
    def test_unsupported_wire_strips(self, monkeypatch, api_mode):
        monkeypatch.setattr(audio_routing, "_load_cfg_readonly", lambda: _cfg("auto"))
        msgs = self._msgs()
        assert strip_unsupported_audio_parts(_agent(api_mode, "openai"), msgs) == 1
        assert msgs[0]["content"] == [{"type": "text", "text": "listen"}]

    def test_excluded_provider_strips_even_on_chat_completions(self, monkeypatch):
        monkeypatch.setattr(audio_routing, "_load_cfg_readonly", lambda: _cfg("auto"))
        msgs = self._msgs()
        assert strip_unsupported_audio_parts(_agent("chat_completions", "gemini"), msgs) == 1

    def test_mode_off_strips(self, monkeypatch):
        monkeypatch.setattr(audio_routing, "_load_cfg_readonly", lambda: _cfg("off"))
        msgs = self._msgs()
        assert strip_unsupported_audio_parts(_agent("chat_completions", "openai"), msgs) == 1
        assert msgs[0]["content"] == [{"type": "text", "text": "listen"}]

    def test_mode_on_overrides_unsupported_wire(self, monkeypatch):
        monkeypatch.setattr(audio_routing, "_load_cfg_readonly", lambda: _cfg("on"))
        msgs = self._msgs()
        assert strip_unsupported_audio_parts(_agent("anthropic_messages", "anthropic"), msgs) == 0

    def test_rejecting_model_strips_even_when_supported(self, monkeypatch):
        monkeypatch.setattr(audio_routing, "_load_cfg_readonly", lambda: _cfg("on"))
        msgs = self._msgs()
        agent = _agent("chat_completions", "openai", "gpt-4o", rejecting={("openai", "gpt-4o")})
        assert strip_unsupported_audio_parts(agent, msgs) == 1
        assert msgs[0]["content"] == [{"type": "text", "text": "listen"}]

    def test_other_model_same_provider_not_stripped(self, monkeypatch):
        monkeypatch.setattr(audio_routing, "_load_cfg_readonly", lambda: _cfg("on"))
        msgs = self._msgs()
        agent = _agent("chat_completions", "openai", "gpt-4o-mini", rejecting={("openai", "gpt-4o")})
        assert strip_unsupported_audio_parts(agent, msgs) == 0

    def test_audio_only_row_gets_placeholder(self, monkeypatch):
        monkeypatch.setattr(audio_routing, "_load_cfg_readonly", lambda: _cfg("off"))
        msgs = [{"role": "user", "content": [_audio_part()]}]
        assert strip_unsupported_audio_parts(_agent(), msgs) == 1
        (row,) = msgs
        assert len(row["content"]) == 1
        assert row["content"][0]["type"] == "text"
        assert "does not accept audio" in row["content"][0]["text"]

    def test_row_without_text_gets_placeholder_prepend(self, monkeypatch):
        monkeypatch.setattr(audio_routing, "_load_cfg_readonly", lambda: _cfg("off"))
        msgs = [{"role": "user", "content": [_audio_part(), {"type": "text", "text": " "}]}]
        strip_unsupported_audio_parts(_agent(), msgs)
        first = msgs[0]["content"][0]
        assert first["type"] == "text" and "does not accept audio" in first["text"]

    def test_non_list_or_empty_returns_zero(self, monkeypatch):
        monkeypatch.setattr(audio_routing, "_load_cfg_readonly", lambda: _cfg("off"))
        assert strip_unsupported_audio_parts(_agent(), None) == 0
        assert strip_unsupported_audio_parts(_agent(), []) == 0
        assert strip_unsupported_audio_parts(_agent(), "nope") == 0

    def test_rows_without_audio_untouched(self, monkeypatch):
        monkeypatch.setattr(audio_routing, "_load_cfg_readonly", lambda: _cfg("off"))
        msgs = [{"role": "user", "content": "plain"}]
        assert strip_unsupported_audio_parts(_agent(), msgs) == 0
        assert msgs[0]["content"] == "plain"

    def test_audio_part_types_both_stripped(self, monkeypatch):
        monkeypatch.setattr(audio_routing, "_load_cfg_readonly", lambda: _cfg("off"))
        msgs = [{"role": "user", "content": [{"type": "audio", "audio": {"data": "AA=="}}, _audio_part()]}]
        assert strip_unsupported_audio_parts(_agent(), msgs) == 1
        assert msgs[0]["content"] == [
            {"type": "text", "text": "[audio attachment omitted: this backend does not accept audio input]"}
        ]
