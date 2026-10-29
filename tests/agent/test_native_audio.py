"""Gemini requests retain actual audio; malformed blocks cannot degrade to text-only."""

import base64

import pytest

from agent.gemini_native_adapter import _build_gemini_contents


@pytest.mark.parametrize("fmt,mime", [("wav", "audio/wav"), ("m4a", "audio/m4a")])
def test_gemini_request_contains_audio_and_question(fmt, mime):
    encoded = base64.b64encode(b"RIFF-audio-fixture").decode()
    contents, _ = _build_gemini_contents([{"role": "user", "content": [
        {"type": "text", "text": "Which instruments?"},
        {"type": "input_audio", "input_audio": {"data": encoded, "format": fmt}},
    ]}])
    assert contents == [{"role": "user", "parts": [
        {"text": "Which instruments?"}, {"inlineData": {"mimeType": mime, "data": encoded}},
    ]}]


@pytest.mark.parametrize("audio", [None, {"data": "***", "format": "wav"},
                                   {"data": "YQ==", "format": "unknown"}])
def test_invalid_audio_refuses_instead_of_dropping_the_attachment(audio):
    with pytest.raises(ValueError):
        _build_gemini_contents([{"role": "user", "content": [
            {"type": "text", "text": "Describe this"}, {"type": "input_audio", "input_audio": audio},
        ]}])
