# -*- coding: utf-8 -*-
"""Tests for gateway.platforms.qqbot.audio — audio conversion & STT utilities."""

from __future__ import annotations

import os
import tempfile

import pytest

from gateway.platforms.qqbot.core.audio import (
    _STT_PROVIDER_URLS,
    guess_audio_ext,
    is_voice_content_type,
    looks_like_silk,
    resolve_stt_config,
)


# ── is_voice_content_type ─────────────────────────────────────────────

class TestIsVoiceContentType:
    def test_voice_string(self):
        assert is_voice_content_type("voice", "") is True

    def test_audio_mime(self):
        assert is_voice_content_type("audio/mpeg", "") is True
        assert is_voice_content_type("audio/amr", "") is True

    def test_silk_extension(self):
        assert is_voice_content_type("", "message.silk") is True

    def test_amr_extension(self):
        assert is_voice_content_type("", "voice.amr") is True

    def test_image_not_voice(self):
        assert is_voice_content_type("image/jpeg", "photo.jpg") is False

    def test_empty(self):
        assert is_voice_content_type("", "") is False

    def test_case_insensitive(self):
        assert is_voice_content_type("Audio/WAV", "") is True
        assert is_voice_content_type("VOICE", "") is True


# ── guess_audio_ext ───────────────────────────────────────────────────

class TestGuessAudioExt:
    def test_silk_v3(self):
        assert guess_audio_ext(b"#!SILK_V3" + b"\x00" * 20) == ".silk"

    def test_silk_short(self):
        assert guess_audio_ext(b"#!SILK" + b"\x00" * 20) == ".silk"

    def test_silk_binary(self):
        assert guess_audio_ext(b"\x02!" + b"\x00" * 20) == ".silk"

    def test_wav(self):
        assert guess_audio_ext(b"RIFF" + b"\x00" * 20) == ".wav"

    def test_flac(self):
        assert guess_audio_ext(b"fLaC" + b"\x00" * 20) == ".flac"

    def test_mp3(self):
        assert guess_audio_ext(b"\xff\xfb" + b"\x00" * 20) == ".mp3"
        assert guess_audio_ext(b"\xff\xf3" + b"\x00" * 20) == ".mp3"

    def test_ogg(self):
        assert guess_audio_ext(b"\x4f\x67\x67\x53" + b"\x00" * 20) == ".ogg"

    def test_unknown_defaults_amr(self):
        assert guess_audio_ext(b"\x00" * 20) == ".amr"

    def test_empty(self):
        assert guess_audio_ext(b"") == ".amr"


# ── looks_like_silk ───────────────────────────────────────────────────

class TestLooksLikeSilk:
    def test_silk_v3_header(self):
        assert looks_like_silk(b"#!SILK_V3" + b"\x00" * 20) is True

    def test_silk_short_header(self):
        assert looks_like_silk(b"#!SILK" + b"\x00" * 20) is True

    def test_binary_header(self):
        assert looks_like_silk(b"\x02!" + b"\x00" * 20) is True

    def test_not_silk(self):
        assert looks_like_silk(b"RIFF" + b"\x00" * 20) is False

    def test_empty(self):
        assert looks_like_silk(b"") is False


# ── resolve_stt_config ────────────────────────────────────────────────

class TestResolveSttConfig:
    def test_no_config(self):
        assert resolve_stt_config({}) is None

    def test_full_config(self):
        extra = {
            "stt": {
                "baseUrl": "https://api.example.com/v1",
                "apiKey": "key-123",
                "model": "whisper-1",
            }
        }
        cfg = resolve_stt_config(extra)
        assert cfg is not None
        assert cfg["base_url"] == "https://api.example.com/v1"
        assert cfg["api_key"] == "key-123"
        assert cfg["model"] == "whisper-1"

    def test_provider_only(self):
        extra = {
            "stt": {
                "provider": "zai",
                "apiKey": "key-456",
            }
        }
        cfg = resolve_stt_config(extra)
        assert cfg is not None
        assert cfg["base_url"] == _STT_PROVIDER_URLS["zai"]
        assert cfg["model"] == "glm-asr"

    def test_openai_provider(self):
        extra = {
            "stt": {
                "provider": "openai",
                "apiKey": "key-789",
            }
        }
        cfg = resolve_stt_config(extra)
        assert cfg is not None
        assert cfg["model"] == "whisper-1"

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("QQ_STT_API_KEY", "env-key")
        monkeypatch.setenv("QQ_STT_MODEL", "env-model")
        cfg = resolve_stt_config({})
        assert cfg is not None
        assert cfg["api_key"] == "env-key"
        assert cfg["model"] == "env-model"

    def test_disabled(self):
        extra = {"stt": {"enabled": False, "apiKey": "key"}}
        assert resolve_stt_config(extra) is None

    def test_base_url_trailing_slash(self):
        extra = {
            "stt": {
                "baseUrl": "https://api.example.com/v1/",
                "apiKey": "key",
            }
        }
        cfg = resolve_stt_config(extra)
        assert cfg["base_url"] == "https://api.example.com/v1"
