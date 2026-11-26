"""Gemini requests retain actual audio; malformed blocks cannot degrade to text-only."""

import base64
import importlib

import pytest


def _native_adapter():
    """The Google plugin's native adapter, which the provider loader republishes.

    The adapter lives in ``plugins/model-providers/google`` — a directory name no module path
    can spell — and the loader exposes it as ``plugins.model_providers.google`` once it has run,
    so resolve the module after loading instead of importing a fixed path at collection time.
    """
    import providers

    providers.get_provider_profile("gemini")
    return importlib.import_module("plugins.model_providers.google.native_adapter")


@pytest.mark.parametrize("fmt,mime", [("wav", "audio/wav"), ("m4a", "audio/m4a")])
def test_gemini_request_contains_audio_and_question(fmt, mime):
    encoded = base64.b64encode(b"RIFF-audio-fixture").decode()
    contents, _ = _native_adapter()._build_gemini_contents([{"role": "user", "content": [
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
        _native_adapter()._build_gemini_contents([{"role": "user", "content": [
            {"type": "text", "text": "Describe this"}, {"type": "input_audio", "input_audio": audio},
        ]}])
