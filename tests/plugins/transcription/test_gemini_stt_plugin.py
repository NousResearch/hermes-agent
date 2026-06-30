from __future__ import annotations

from unittest.mock import Mock

from plugins.transcription.gemini import GeminiTranscriptionProvider, register


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


def test_provider_metadata_and_availability(monkeypatch):
    monkeypatch.delenv("GEMINI_STT_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    provider = GeminiTranscriptionProvider()
    assert provider.name == "gemini"
    assert provider.display_name == "Gemini STT"
    assert provider.default_model() == "gemini-3.1-flash-lite-preview"
    assert provider.is_available() is False

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert provider.is_available() is True


def test_transcribe_success(monkeypatch, tmp_path):
    import plugins.transcription.gemini as gemini

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    audio_path = tmp_path / "voice.mp3"
    audio_path.write_bytes(b"fake audio")

    post = Mock(
        return_value=_Response(
            {"candidates": [{"content": {"parts": [{"text": "hello world"}]}}]}
        )
    )
    monkeypatch.setattr(gemini.requests, "post", post)

    result = GeminiTranscriptionProvider().transcribe(
        str(audio_path),
        model="gemini-2.5-flash",
        language="en",
    )

    assert result == {
        "success": True,
        "transcript": "hello world",
        "provider": "gemini",
    }

    _, kwargs = post.call_args
    assert kwargs["params"] == {"key": "test-key"}
    assert kwargs["headers"] == {"Content-Type": "application/json"}
    assert post.call_args.args[0].endswith("/models/gemini-2.5-flash:generateContent")
    parts = kwargs["json"]["contents"][0]["parts"]
    assert parts[0]["text"].endswith("The expected language is en.")
    assert parts[1]["inline_data"]["data"]
    assert parts[1]["inline_data"]["mime_type"] in {"audio/mp3", "audio/mpeg"}


def test_transcribe_rejects_audio_over_inline_limit(monkeypatch, tmp_path):
    import plugins.transcription.gemini as gemini

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(gemini, "INLINE_AUDIO_LIMIT_BYTES", 8)

    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"original audio larger than limit")

    post = Mock()
    monkeypatch.setattr(gemini.requests, "post", post)

    result = GeminiTranscriptionProvider().transcribe(str(audio_path))

    assert result["success"] is False
    assert result["transcript"] == ""
    assert result["provider"] == "gemini"
    assert "inline audio is limited to 20 MB" in result["error"]
    post.assert_not_called()


def test_transcribe_returns_http_error_detail(monkeypatch, tmp_path):
    import plugins.transcription.gemini as gemini

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"fake audio")

    post = Mock(
        return_value=_Response(
            {"error": {"message": "bad request"}},
            status_code=400,
            text="fallback",
        )
    )
    monkeypatch.setattr(gemini.requests, "post", post)

    result = GeminiTranscriptionProvider().transcribe(str(audio_path))

    assert result["success"] is False
    assert result["transcript"] == ""
    assert result["provider"] == "gemini"
    assert result["error"] == "Gemini STT API error (HTTP 400): bad request"


def test_transcribe_returns_error_envelope(monkeypatch, tmp_path):
    import plugins.transcription.gemini as gemini

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"fake audio")

    post = Mock(side_effect=RuntimeError("network down"))
    monkeypatch.setattr(gemini.requests, "post", post)

    result = GeminiTranscriptionProvider().transcribe(str(audio_path))

    assert result["success"] is False
    assert result["transcript"] == ""
    assert result["provider"] == "gemini"
    assert result["error"] == "Gemini STT failed: network down"


def test_register_adds_provider():
    ctx = Mock()

    register(ctx)

    ctx.register_transcription_provider.assert_called_once()
    provider = ctx.register_transcription_provider.call_args.args[0]
    assert isinstance(provider, GeminiTranscriptionProvider)
