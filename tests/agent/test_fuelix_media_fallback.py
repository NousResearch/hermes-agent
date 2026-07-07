"""E2E coverage for the Phase 4 FuelIX media fallback providers.

Regression test for a real bug found during the Phase 6 merge audit:
``load_config()`` stores the FuelIX credential under the flat key
``providers['api.fuelix.ai']['api_key']`` (confirmed against the real
config.yaml shape), not the nested ``providers['api']['fuelix.ai']['api_key']``
the original Phase 4 implementation assumed. That bug meant every FuelIX
fallback path silently reported "API key missing" and never actually fired.
These tests mock ``load_config()`` with the REAL flat-key shape so this
class of bug cannot regress silently again.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.transcription_provider import FuelIXTranscriptionProvider
from agent.tts_provider import FuelIXTTSProvider

REAL_SHAPE_CONFIG = {"providers": {"api.fuelix.ai": {"api_key": "ak-test-key-123"}}}
WRONG_SHAPE_CONFIG = {"providers": {"api": {"fuelix.ai": {"api_key": "ak-test-key-123"}}}}
EMPTY_CONFIG: dict = {"providers": {}}


class TestFuelIXTranscriptionProvider:
    def test_is_available_true_with_real_config_shape(self):
        provider = FuelIXTranscriptionProvider()
        with patch("hermes_cli.config.load_config", return_value=REAL_SHAPE_CONFIG):
            assert provider.is_available() is True

    def test_is_available_false_with_wrong_nested_shape(self):
        """Guards the exact bug found in Phase 6 audit: nested api/fuelix.ai
        lookup must NOT match, proving the flat-key fix is actually load-bearing."""
        provider = FuelIXTranscriptionProvider()
        with patch("hermes_cli.config.load_config", return_value=WRONG_SHAPE_CONFIG):
            assert provider.is_available() is False

    def test_is_available_false_when_key_missing(self):
        provider = FuelIXTranscriptionProvider()
        with patch("hermes_cli.config.load_config", return_value=EMPTY_CONFIG):
            assert provider.is_available() is False

    def test_transcribe_missing_key_returns_error_envelope(self, tmp_path):
        provider = FuelIXTranscriptionProvider()
        audio_file = tmp_path / "clip.wav"
        audio_file.write_bytes(b"fake-audio")
        with patch("hermes_cli.config.load_config", return_value=EMPTY_CONFIG):
            result = provider.transcribe(str(audio_file))
        assert result["success"] is False
        assert "API key missing" in result["error"]
        assert result["provider"] == "fuelix"

    def test_transcribe_success(self, tmp_path):
        provider = FuelIXTranscriptionProvider()
        audio_file = tmp_path / "clip.wav"
        audio_file.write_bytes(b"fake-audio")

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"text": "hello world"}

        with patch("hermes_cli.config.load_config", return_value=REAL_SHAPE_CONFIG), \
             patch("requests.post", return_value=mock_response) as mock_post:
            result = provider.transcribe(str(audio_file), model="whisper-1")

        assert result == {"success": True, "transcript": "hello world", "provider": "fuelix"}
        called_url = mock_post.call_args.args[0]
        assert called_url == "https://api.fuelix.ai/v1/audio/transcriptions"
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer ak-test-key-123"

    def test_transcribe_http_error_returns_error_envelope(self, tmp_path):
        provider = FuelIXTranscriptionProvider()
        audio_file = tmp_path / "clip.wav"
        audio_file.write_bytes(b"fake-audio")

        with patch("hermes_cli.config.load_config", return_value=REAL_SHAPE_CONFIG), \
             patch("requests.post", side_effect=RuntimeError("connection reset")):
            result = provider.transcribe(str(audio_file))

        assert result["success"] is False
        assert "connection reset" in result["error"]
        assert result["provider"] == "fuelix"


class TestFuelIXTTSProvider:
    def test_is_available_true_with_real_config_shape(self):
        provider = FuelIXTTSProvider()
        with patch("hermes_cli.config.load_config", return_value=REAL_SHAPE_CONFIG):
            assert provider.is_available() is True

    def test_is_available_false_with_wrong_nested_shape(self):
        provider = FuelIXTTSProvider()
        with patch("hermes_cli.config.load_config", return_value=WRONG_SHAPE_CONFIG):
            assert provider.is_available() is False

    def test_synthesize_missing_key_raises(self, tmp_path):
        provider = FuelIXTTSProvider()
        out_path = tmp_path / "out.mp3"
        with patch("hermes_cli.config.load_config", return_value=EMPTY_CONFIG):
            with pytest.raises(ValueError, match="API key missing"):
                provider.synthesize("hello", str(out_path))

    def test_synthesize_success_writes_audio_bytes(self, tmp_path):
        provider = FuelIXTTSProvider()
        out_path = tmp_path / "out.mp3"

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b"fake-mp3-bytes"

        with patch("hermes_cli.config.load_config", return_value=REAL_SHAPE_CONFIG), \
             patch("requests.post", return_value=mock_response) as mock_post:
            result_path = provider.synthesize("hello world", str(out_path), voice="alloy", model="tts-1")

        assert result_path == str(out_path)
        assert out_path.read_bytes() == b"fake-mp3-bytes"
        called_url = mock_post.call_args.args[0]
        assert called_url == "https://api.fuelix.ai/v1/audio/speech"
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer ak-test-key-123"
        assert mock_post.call_args.kwargs["json"] == {
            "model": "tts-1",
            "input": "hello world",
            "voice": "alloy",
        }
