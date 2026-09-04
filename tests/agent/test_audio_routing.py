"""Tests for agent/audio_routing.py — the per-turn audio input mode decision and parts builder."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.audio_routing import (
    _coerce_capability_bool,
    _coerce_mode,
    _guess_audio_mime,
    _is_known_audio_model_or_provider,
    _lookup_supports_audio,
    _mime_to_audio_format,
    _sniff_audio_mime_from_bytes,
    _supports_audio_override,
    build_native_audio_content_parts,
    decide_audio_input_mode,
    extract_audio_refs,
)
from agent.gemini_native_adapter import _extract_multimodal_parts


class TestCoerceMode:
    def test_case_insensitive(self):
        assert _coerce_mode("NATIVE") == "native"
        assert _coerce_mode("Auto") == "auto"
        assert _coerce_mode("STT") == "stt"

    def test_text_alias_maps_to_stt(self):
        assert _coerce_mode("text") == "stt"

    def test_invalid_falls_back_to_auto(self):
        assert _coerce_mode("nonsense") == "auto"
        assert _coerce_mode("") == "auto"
        assert _coerce_mode(None) == "auto"
        assert _coerce_mode(42) == "auto"


class TestCoerceCapabilityBool:
    def test_bool_literals(self):
        assert _coerce_capability_bool(True) is True
        assert _coerce_capability_bool(False) is False

    def test_integer_flags(self):
        assert _coerce_capability_bool(1) is True
        assert _coerce_capability_bool(0) is False
        assert _coerce_capability_bool(2) is None

    def test_string_tokens(self):
        for token in ("true", "TRUE", "yes", "on", "1"):
            assert _coerce_capability_bool(token) is True
        for token in ("false", "FALSE", "no", "off", "0"):
            assert _coerce_capability_bool(token) is False

    def test_invalid_tokens(self):
        assert _coerce_capability_bool("maybe") is None
        assert _coerce_capability_bool("") is None
        assert _coerce_capability_bool({}) is None


class TestDecideAudioInputMode:
    def test_mode_native_override(self):
        cfg = {"agent": {"audio_input_mode": "native"}}
        assert decide_audio_input_mode("any-provider", "any-model", cfg) == "native"

    def test_mode_stt_override(self):
        cfg = {"agent": {"audio_input_mode": "stt"}}
        assert decide_audio_input_mode("gemini", "gemini-2.5-flash", cfg) == "stt"

    def test_mode_text_alias_override(self):
        cfg = {"agent": {"audio_input_mode": "text"}}
        assert decide_audio_input_mode("gemini", "gemini-2.5-flash", cfg) == "stt"

    def test_auto_with_gemini_provider(self):
        assert decide_audio_input_mode("gemini", "gemini-2.0-flash", {}) == "native"
        assert decide_audio_input_mode("google", "gemini-pro", {}) == "native"

    def test_auto_with_gemini_model_slug(self):
        assert decide_audio_input_mode("openrouter", "google/gemini-2.5-flash", {}) == "native"
        assert decide_audio_input_mode("custom", "gemini-1.5-pro", {}) == "native"

    def test_auto_with_gpt4o_audio_model(self):
        assert decide_audio_input_mode("openai", "gpt-4o-audio-preview", {}) == "native"
        assert decide_audio_input_mode("openai", "gpt-4o-mini-audio-preview", {}) == "native"

    def test_auto_with_text_only_model(self):
        assert decide_audio_input_mode("anthropic", "claude-sonnet-4", {}) == "stt"
        assert decide_audio_input_mode("openai", "gpt-4o", {}) == "stt"

    def test_auto_with_models_dev_capability(self):
        mock_cap = MagicMock()
        mock_cap.supports_audio_input.return_value = True
        with patch("agent.models_dev.get_model_capabilities", return_value=mock_cap):
            assert decide_audio_input_mode("custom_provider", "some-multimodal-model", {}) == "native"

    def test_config_model_supports_audio_override(self):
        cfg = {"model": {"supports_audio": True}}
        assert decide_audio_input_mode("custom", "any-text-model", cfg) == "native"

        cfg_false = {"model": {"supports_audio": False}}
        assert decide_audio_input_mode("gemini", "gemini-2.5-flash", cfg_false) == "stt"

    def test_config_providers_model_override(self):
        cfg = {
            "providers": {
                "my-vllm": {
                    "models": {
                        "whisper-llm": {"supports_audio": True}
                    }
                }
            }
        }
        assert decide_audio_input_mode("my-vllm", "whisper-llm", cfg) == "native"


class TestSniffAndGuessMime:
    def test_sniff_audio_wav(self):
        wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt "
        assert _sniff_audio_mime_from_bytes(wav_header) == "audio/wav"

    def test_sniff_audio_ogg(self):
        ogg_header = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00"
        assert _sniff_audio_mime_from_bytes(ogg_header) == "audio/ogg"

    def test_sniff_audio_mp3_id3(self):
        mp3_header = b"ID3\x04\x00\x00\x00\x00\x00"
        assert _sniff_audio_mime_from_bytes(mp3_header) == "audio/mp3"

    def test_sniff_audio_flac(self):
        flac_header = b"fLaC\x00\x00\x00\x22"
        assert _sniff_audio_mime_from_bytes(flac_header) == "audio/flac"

    def test_guess_audio_mime_by_extension(self):
        assert _guess_audio_mime(Path("voice.oga")) == "audio/ogg"
        assert _guess_audio_mime(Path("note.ogg")) == "audio/ogg"
        assert _guess_audio_mime(Path("clip.mp3")) in ("audio/mp3", "audio/mpeg")
        assert _mime_to_audio_format(_guess_audio_mime(Path("clip.mp3"))) == "mp3"
        assert _guess_audio_mime(Path("recording.wav")) == "audio/wav"
        assert _guess_audio_mime(Path("voice.m4a")) == "audio/m4a"


class TestMimeToAudioFormat:
    def test_mime_formats(self):
        assert _mime_to_audio_format("audio/wav") == "wav"
        assert _mime_to_audio_format("audio/mp3") == "mp3"
        assert _mime_to_audio_format("audio/ogg") == "ogg"
        assert _mime_to_audio_format("audio/oga") == "ogg"
        assert _mime_to_audio_format("audio/m4a") == "m4a"


class TestBuildNativeAudioContentParts:
    def test_build_parts_from_valid_audio(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        raw_content = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        audio_file.write_bytes(raw_content)

        parts, skipped = build_native_audio_content_parts(
            "Hello, listen to this:",
            [str(audio_file)],
        )

        assert skipped == []
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert "Hello, listen to this:" in parts[0]["text"]
        assert f"[Audio attached at: {audio_file}]" in parts[0]["text"]

        assert parts[1]["type"] == "input_audio"
        assert parts[1]["input_audio"]["format"] == "wav"
        assert parts[1]["input_audio"]["data"] == base64.b64encode(raw_content).decode("ascii")

    def test_build_parts_empty_text_default_prompt(self, tmp_path):
        audio_file = tmp_path / "voice.oga"
        raw_content = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00fakeopuspayload"
        audio_file.write_bytes(raw_content)

        parts, skipped = build_native_audio_content_parts(
            "",
            [str(audio_file)],
        )

        assert skipped == []
        assert len(parts) == 2
        assert "Listen to the attached audio" in parts[0]["text"]
        assert f"[Audio attached at: {audio_file}]" in parts[0]["text"]
        assert parts[1]["type"] == "input_audio"
        assert parts[1]["input_audio"]["format"] == "ogg"

    def test_build_parts_missing_file_skipped(self, tmp_path):
        missing = tmp_path / "non_existent.mp3"
        parts, skipped = build_native_audio_content_parts(
            "Check audio",
            [str(missing)],
        )
        assert skipped == [str(missing)]
        assert parts == [{"type": "text", "text": "Check audio"}]


class TestGeminiNativeAdapterAudioParts:
    def test_extract_input_audio_part_to_gemini_inlinedata(self):
        b64_data = base64.b64encode(b"fake-audio-bytes").decode("ascii")
        content = [
            {"type": "text", "text": "Listen to this voice clip"},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": b64_data,
                    "format": "ogg",
                },
            },
        ]

        parts = _extract_multimodal_parts(content)
        assert len(parts) == 2
        assert parts[0] == {"text": "Listen to this voice clip"}
        assert parts[1] == {
            "inlineData": {
                "mimeType": "audio/ogg",
                "data": b64_data,
            }
        }

    def test_extract_input_audio_wav_to_gemini(self):
        b64_data = base64.b64encode(b"RIFFwavefake").decode("ascii")
        content = [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": b64_data,
                    "format": "wav",
                },
            }
        ]

        parts = _extract_multimodal_parts(content)
        assert len(parts) == 1
        assert parts[0]["inlineData"]["mimeType"] == "audio/wav"
        assert parts[0]["inlineData"]["data"] == b64_data

    def test_extract_audio_url_data_uri_to_gemini(self):
        b64_data = base64.b64encode(b"fake-mp3").decode("ascii")
        content = [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": f"data:audio/mp3;base64,{b64_data}",
                },
            }
        ]

        parts = _extract_multimodal_parts(content)
        assert len(parts) == 1
        assert parts[0]["inlineData"]["mimeType"] == "audio/mp3"
        assert parts[0]["inlineData"]["data"] == b64_data


class TestExtractAudioRefs:
    def test_extract_audio_paths(self, tmp_path):
        f1 = tmp_path / "voice.oga"
        f1.write_bytes(b"123")
        f2 = tmp_path / "song.mp3"
        f2.write_bytes(b"456")

        text = f"Check {f1} and {f2} and https://example.com/audio.wav"
        local_paths, urls = extract_audio_refs(text)

        assert str(f1) in local_paths
        assert str(f2) in local_paths
        assert "https://example.com/audio.wav" in urls

    def test_ignore_code_blocks(self, tmp_path):
        f = tmp_path / "hidden.wav"
        f.write_bytes(b"test")
        text = f"Here is code:\n```\n{f}\n```\nand inline `{f}`"
        local_paths, urls = extract_audio_refs(text)
        assert local_paths == []
        assert urls == []
