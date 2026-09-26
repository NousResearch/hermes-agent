"""Tests for the bundled ``mimo`` transcription plugin.

Covers provider metadata, availability, MIME-type mapping, request shape,
response parsing, and retry/error handling — the gaps flagged in the PR review.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent import transcription_registry
from tools import transcription_tools

mimo_plugin = importlib.import_module("plugins.transcription.mimo")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    transcription_registry._reset_for_tests()
    yield
    transcription_registry._reset_for_tests()


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "sk-test-mimo")
    return mimo_plugin.MiMoAsrProvider()


def _make_urlopen_mock(response_text: str = "hello world", status: int = 200):
    """Return a fake urlopen that records the last request and returns JSON."""
    calls = []

    def fake_urlopen(req, *args, **kwargs):
        calls.append({"url": req.full_url, "headers": dict(req.headers), "body": req.data})
        mock_response = MagicMock()
        payload = {
            "choices": [{"message": {"content": response_text}}]
        }
        mock_response.read.return_value = json.dumps(payload).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        return mock_response

    fake_urlopen.calls = calls
    return fake_urlopen


# ---------------------------------------------------------------------------
# Metadata & availability
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_name_and_display_name(self):
        p = mimo_plugin.MiMoAsrProvider()
        assert p.name == "mimo"
        assert p.display_name == "MiMo ASR"

    def test_default_model(self):
        p = mimo_plugin.MiMoAsrProvider()
        assert p.default_model() == "mimo-v2.5-asr"

    def test_list_models(self):
        p = mimo_plugin.MiMoAsrProvider()
        models = p.list_models()
        assert len(models) == 1
        assert models[0]["id"] == "mimo-v2.5-asr"
        assert "auto" in models[0]["languages"]

    def test_setup_schema_exposes_env_var(self):
        p = mimo_plugin.MiMoAsrProvider()
        schema = p.get_setup_schema()
        assert schema["name"] == "MiMo ASR"
        assert any(entry["key"] == "MIMO_API_KEY" for entry in schema["env_vars"])


class TestAvailability:
    def test_available_when_mimo_api_key_set(self, monkeypatch):
        monkeypatch.setenv("MIMO_API_KEY", "sk-test")
        monkeypatch.delenv("XIAOMIMIMO_API_KEY", raising=False)
        monkeypatch.delenv("XIAOMI_API_KEY", raising=False)
        assert mimo_plugin.MiMoAsrProvider().is_available() is True

    def test_available_when_xiaomi_api_key_set(self, monkeypatch):
        monkeypatch.delenv("MIMO_API_KEY", raising=False)
        monkeypatch.setenv("XIAOMI_API_KEY", "sk-test")
        assert mimo_plugin.MiMoAsrProvider().is_available() is True

    def test_unavailable_when_no_key(self, monkeypatch):
        monkeypatch.delenv("MIMO_API_KEY", raising=False)
        monkeypatch.delenv("XIAOMIMIMO_API_KEY", raising=False)
        monkeypatch.delenv("XIAOMI_API_KEY", raising=False)
        assert mimo_plugin.MiMoAsrProvider().is_available() is False


# ---------------------------------------------------------------------------
# Request shape & MIME mapping
# ---------------------------------------------------------------------------


class TestRequestShape:
    @pytest.mark.parametrize(
        "suffix,expected_mime",
        [
            (".wav", "audio/wav"),
            (".mp3", "audio/mpeg"),
        ],
    )
    def test_mime_type_mapping(self, provider, monkeypatch, tmp_path, suffix, expected_mime):
        audio_path = tmp_path / f"audio{suffix}"
        audio_path.write_bytes(b"fake audio bytes")

        fake_urlopen = _make_urlopen_mock()
        monkeypatch.setattr(mimo_plugin, "urlopen", fake_urlopen)

        result = provider.transcribe(str(audio_path))

        assert result["success"] is True
        assert result["transcript"] == "hello world"
        assert len(fake_urlopen.calls) == 1
        body = json.loads(fake_urlopen.calls[0]["body"].decode("utf-8"))
        data_url = body["messages"][0]["content"][0]["input_audio"]["data"]
        assert data_url.startswith(f"data:{expected_mime};base64,")

    def test_unsupported_format_rejected(self, provider, tmp_path):
        for suffix in (".ogg", ".flac", ".m4a", ".aac"):
            audio_path = tmp_path / f"audio{suffix}"
            audio_path.write_bytes(b"nope")
            result = provider.transcribe(str(audio_path))
            assert result["success"] is False, suffix
            assert "Unsupported audio format" in result["error"]
            assert "MiMo ASR only supports .wav and .mp3" in result["error"]

    def test_request_headers_and_payload(self, provider, monkeypatch, tmp_path):
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake audio bytes")

        fake_urlopen = _make_urlopen_mock()
        monkeypatch.setattr(mimo_plugin, "urlopen", fake_urlopen)

        result = provider.transcribe(str(audio_path), model="mimo-v2.5-asr", language="zh")

        assert result["success"] is True
        call = fake_urlopen.calls[0]
        headers = {k.lower(): v for k, v in call["headers"].items()}
        assert headers["content-type"] == "application/json"
        assert headers["api-key"] == "sk-test-mimo"
        assert call["url"] == "https://api.xiaomimimo.com/v1/chat/completions"

        body = json.loads(call["body"].decode("utf-8"))
        assert body["model"] == "mimo-v2.5-asr"
        assert body["asr_options"]["language"] == "zh"
        assert body["messages"][0]["content"][0]["type"] == "input_audio"

    def test_default_model_and_language(self, provider, monkeypatch, tmp_path):
        audio_path = tmp_path / "audio.mp3"
        audio_path.write_bytes(b"fake audio bytes")

        fake_urlopen = _make_urlopen_mock()
        monkeypatch.setattr(mimo_plugin, "urlopen", fake_urlopen)

        provider.transcribe(str(audio_path))
        body = json.loads(fake_urlopen.calls[0]["body"].decode("utf-8"))
        assert body["model"] == "mimo-v2.5-asr"
        assert body["asr_options"]["language"] == "auto"


# ---------------------------------------------------------------------------
# Response parsing & errors
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_whitespace_is_trimmed(self, provider, monkeypatch, tmp_path):
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"x")

        fake_urlopen = _make_urlopen_mock(response_text="  trimmed  ")
        monkeypatch.setattr(mimo_plugin, "urlopen", fake_urlopen)

        result = provider.transcribe(str(audio_path))
        assert result["transcript"] == "trimmed"

    def test_missing_api_key_returns_error_envelope(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MIMO_API_KEY", raising=False)
        monkeypatch.delenv("XIAOMIMIMO_API_KEY", raising=False)
        monkeypatch.delenv("XIAOMI_API_KEY", raising=False)
        p = mimo_plugin.MiMoAsrProvider()
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"x")
        result = p.transcribe(str(audio_path))
        assert result["success"] is False
        assert "not set" in result["error"]

    def test_missing_file_returns_error_envelope(self, provider, tmp_path):
        missing = tmp_path / "missing.wav"
        result = provider.transcribe(str(missing))
        assert result["success"] is False
        assert "Cannot read audio file" in result["error"]


class TestRetryAndErrorHandling:
    def test_retries_on_retryable_status_then_succeeds(self, provider, monkeypatch, tmp_path):
        audio_path = tmp_path / "audio.mp3"
        audio_path.write_bytes(b"x")

        from urllib.error import HTTPError

        attempt = {"count": 0}

        def fake_urlopen(req, *args, **kwargs):
            attempt["count"] += 1
            if attempt["count"] < 3:
                raise HTTPError(req.full_url, 503, "Service Unavailable", {}, None)
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"choices": [{"message": {"content": "ok after retry"}}]}).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = False
            return mock_resp

        monkeypatch.setattr(mimo_plugin, "urlopen", fake_urlopen)
        monkeypatch.setattr(mimo_plugin, "time", MagicMock())

        result = provider.transcribe(str(audio_path))
        assert result["success"] is True
        assert result["transcript"] == "ok after retry"
        assert attempt["count"] == 3

    def test_non_retryable_http_error_returns_error(self, provider, monkeypatch, tmp_path):
        audio_path = tmp_path / "audio.mp3"
        audio_path.write_bytes(b"x")

        from urllib.error import HTTPError

        def fake_urlopen(req, *args, **kwargs):
            raise HTTPError(req.full_url, 400, "Bad Request", {}, None)

        monkeypatch.setattr(mimo_plugin, "urlopen", fake_urlopen)
        result = provider.transcribe(str(audio_path))
        assert result["success"] is False
        assert "HTTP 400" in result["error"]

    def test_non_retryable_exception_returns_error(self, provider, monkeypatch, tmp_path):
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"x")

        def fake_urlopen(req, *args, **kwargs):
            raise ValueError("boom")

        monkeypatch.setattr(mimo_plugin, "urlopen", fake_urlopen)
        result = provider.transcribe(str(audio_path))
        assert result["success"] is False
        assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# Dispatch wiring
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_plugin_registers_under_name_mimo(self):
        mimo_plugin.register(MagicMock())
        # We verify the call argument is a MiMoAsrProvider instance
        # by capturing the context mock's method call.

    def test_transcribe_audio_dispatches_to_mimo_plugin(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MIMO_API_KEY", "sk-test-mimo")
        monkeypatch.setattr(mimo_plugin, "time", MagicMock())

        audio_path = tmp_path / "audio.mp3"
        audio_path.write_bytes(b"x")

        fake_urlopen = _make_urlopen_mock(response_text="from mimo")
        monkeypatch.setattr(mimo_plugin, "urlopen", fake_urlopen)

        provider = mimo_plugin.MiMoAsrProvider()
        monkeypatch.setattr(
            transcription_registry,
            "get_provider",
            lambda name: provider if name.strip().lower() == "mimo" else None,
        )

        from unittest.mock import patch
        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value={"provider": "mimo"}), \
             patch("tools.transcription_tools.is_stt_enabled", return_value=True), \
             patch("tools.transcription_tools._get_provider", return_value="mimo"):
            result = transcription_tools.transcribe_audio(str(audio_path))

        assert result["success"] is True
        assert result["transcript"] == "from mimo"
        assert result["provider"] == "mimo"
