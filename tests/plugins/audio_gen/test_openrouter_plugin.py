"""Tests for the OpenRouter audio gen plugin — register, payload, response handling."""

from __future__ import annotations

import base64
import json

import pytest

from agent import audio_gen_registry
from plugins.audio_gen import openrouter as _ora_module

# Capture the genuine _fetch_models at import time, before the autouse stub
# fixture can replace it — so the catalog-filter test exercises real code.
_REAL_FETCH_MODELS = _ora_module._fetch_models


@pytest.fixture(autouse=True)
def _reset_registry():
    audio_gen_registry._reset_for_tests()
    yield
    audio_gen_registry._reset_for_tests()


@pytest.fixture(autouse=True)
def _stub_models(monkeypatch):
    """Avoid live /models network calls in unit tests."""
    from plugins.audio_gen import openrouter as ora

    models = [
        {
            "id": "google/lyria-3-pro-preview",
            "display": "Lyria 3 Pro",
            "strengths": "Music with vocals",
            "kinds": ["music"],
            "supports_lyrics": True,
        },
        {
            "id": "google/lyria-3-clip-preview",
            "display": "Lyria 3 Clip",
            "strengths": "Short music clips",
            "kinds": ["music"],
            "supports_lyrics": True,
        },
    ]
    monkeypatch.setattr(ora, "_fetch_models", lambda: models)
    return models


def _provider(monkeypatch, api_key="sk-or-test"):
    from plugins.audio_gen import openrouter as ora

    monkeypatch.setattr(
        ora, "_resolve_credentials",
        lambda: (api_key, "https://openrouter.ai/api/v1"),
    )
    return ora.OpenRouterAudioGenProvider()


def test_registers_and_basic_surface(monkeypatch):
    from plugins.audio_gen.openrouter import OpenRouterAudioGenProvider, DEFAULT_MODEL

    provider = _provider(monkeypatch)
    audio_gen_registry.register_provider(provider)

    assert audio_gen_registry.get_provider("openrouter") is provider
    assert provider.name == "openrouter"
    assert provider.display_name == "OpenRouter"
    assert provider.default_model() == DEFAULT_MODEL == "google/lyria-3-pro-preview"


def test_is_available_requires_key(monkeypatch):
    assert _provider(monkeypatch, api_key="").is_available() is False
    assert _provider(monkeypatch, api_key="sk-or-x").is_available() is True


def test_setup_schema_declares_openrouter_key(monkeypatch):
    schema = _provider(monkeypatch).get_setup_schema()
    assert schema["name"] == "OpenRouter Audio"
    assert [e["key"] for e in schema["env_vars"]] == ["OPENROUTER_API_KEY"]


def test_capabilities_report_lyrics(monkeypatch):
    caps = _provider(monkeypatch).capabilities()
    assert "music" in caps["kinds"]
    assert caps["supports_lyrics"] is True  # Lyria in catalog
    assert set(caps["formats"]) == {"mp3", "wav"}


def test_generate_requires_key(monkeypatch):
    out = _provider(monkeypatch, api_key="").generate("a jingle")
    assert out["success"] is False
    assert out["error_type"] == "auth_required"


def test_generate_requires_prompt(monkeypatch):
    out = _provider(monkeypatch).generate("   ")
    assert out["success"] is False
    assert out["error_type"] == "missing_prompt"


# ---------------------------------------------------------------------------
# Streaming chat/completions flow — mock httpx.stream (SSE)
# ---------------------------------------------------------------------------
#
# Verified live against OpenRouter: audio models require modalities=
# ["text","audio"] + stream:true, and deliver base64 across streamed
# delta.audio.data chunks. These tests pin that contract.


def _sse(obj) -> str:
    return "data: " + json.dumps(obj)


class _FakeStream:
    """Context-manager stand-in for httpx.stream()."""

    def __init__(self, *, status_code=200, lines=None, err_body=b""):
        self.status_code = status_code
        self._lines = lines or []
        self._err_body = err_body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._err_body

    def iter_lines(self):
        for ln in self._lines:
            yield ln


def test_generate_success_writes_audio(monkeypatch, tmp_path):
    from plugins.audio_gen import openrouter as ora

    provider = _provider(monkeypatch)
    captured: dict = {}
    # Split the base64 across two delta chunks to exercise accumulation.
    full_b64 = base64.b64encode(b"FAKEAUDIOBYTES-LONGER-PAYLOAD").decode()
    half = len(full_b64) // 2
    part1, part2 = full_b64[:half], full_b64[half:]

    lines = [
        _sse({"choices": [{"delta": {"audio": {"data": part1}}}]}),
        _sse({"choices": [{"delta": {"audio": {"data": part2, "transcript": "la la"}}}]}),
        _sse({"usage": {"cost": 0.08}, "choices": [{"delta": {}}]}),
        "data: [DONE]",
    ]

    def _fake_stream(method, url, headers=None, json=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        return _FakeStream(lines=lines)

    saved = {}

    def _fake_save(data, *, prefix="audio", extension="mp3"):
        saved["data"] = data
        saved["ext"] = extension
        out = tmp_path / f"{prefix}.{extension}"
        out.write_bytes(base64.b64decode(data))
        return out

    monkeypatch.setattr(ora.httpx, "stream", _fake_stream)
    monkeypatch.setattr(ora, "save_b64_audio", _fake_save)

    out = provider.generate("upbeat lo-fi beat", duration=20, lyrics="hello world", audio_format="mp3")
    assert out["success"] is True
    assert out["provider"] == "openrouter"
    assert out["format"] == "mp3"
    assert out["duration"] == 20
    assert out["transcript"] == "la la"
    assert out["usage"] == {"cost": 0.08}
    assert out["audio"].endswith(".mp3")
    # The two chunks were concatenated and decode to the original bytes.
    assert saved["data"] == full_b64

    # Payload contract: streamed POST, modalities=[text,audio], format, lyrics.
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/chat/completions")
    assert captured["json"]["modalities"] == ["text", "audio"]
    assert captured["json"]["stream"] is True
    assert captured["json"]["audio"] == {"format": "mp3"}
    assert captured["json"]["model"] == "google/lyria-3-pro-preview"
    msg = captured["json"]["messages"][0]["content"]
    assert "upbeat lo-fi beat" in msg
    assert "hello world" in msg
    assert saved["ext"] == "mp3"


def test_generate_no_audio_in_response_errors(monkeypatch):
    from plugins.audio_gen import openrouter as ora

    provider = _provider(monkeypatch)
    lines = [
        _sse({"choices": [{"delta": {"content": "no audio here"}}]}),
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        ora.httpx, "stream",
        lambda *a, **k: _FakeStream(lines=lines),
    )

    out = provider.generate("a song")
    assert out["success"] is False
    assert out["error_type"] == "empty_response"


def test_generate_http_error_surfaces(monkeypatch):
    from plugins.audio_gen import openrouter as ora

    provider = _provider(monkeypatch)
    monkeypatch.setattr(
        ora.httpx, "stream",
        lambda *a, **k: _FakeStream(status_code=429, err_body=b'{"error":"rate limited"}'),
    )

    out = provider.generate("a song")
    assert out["success"] is False
    assert out["error_type"] == "api_error"
    assert "429" in out["error"]


def test_unsupported_format_clamps_to_mp3(monkeypatch):
    from plugins.audio_gen import openrouter as ora

    provider = _provider(monkeypatch)
    captured: dict = {}
    b64 = base64.b64encode(b"x").decode()
    lines = [
        _sse({"choices": [{"delta": {"audio": {"data": b64}}}]}),
        "data: [DONE]",
    ]

    def _fake_stream(method, url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return _FakeStream(lines=lines)

    monkeypatch.setattr(ora.httpx, "stream", _fake_stream)
    monkeypatch.setattr(
        ora, "save_b64_audio",
        lambda data, *, prefix="a", extension="mp3": __import__("pathlib").Path(f"/tmp/x.{extension}"),
    )

    out = provider.generate("a song", audio_format="ogg")
    assert out["success"] is True
    assert captured["json"]["audio"] == {"format": "mp3"}


def test_catalog_scoped_to_lyria(monkeypatch):
    """gpt-audio speech models must NOT appear in the audio-gen catalog —
    they require audio.voice and belong to text_to_speech. This exercises
    the real _fetch_models filter against a mixed /models response."""
    from plugins.audio_gen import openrouter as ora

    ora._models_cache = None

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"data": [
                {"id": "google/lyria-3-pro-preview", "name": "Lyria 3 Pro",
                 "architecture": {"output_modalities": ["audio"]}},
                {"id": "openai/gpt-audio", "name": "GPT-Audio",
                 "architecture": {"output_modalities": ["audio"]}},
                {"id": "openai/gpt-4o", "name": "GPT-4o",
                 "architecture": {"output_modalities": ["text"]}},
            ]}

    monkeypatch.setattr(ora, "_resolve_credentials", lambda: ("sk-or-x", "https://openrouter.ai/api/v1"))
    monkeypatch.setattr(ora.httpx, "get", lambda *a, **k: _Resp())

    # Call the genuine (un-stubbed) implementation captured at import time.
    ids = [m["id"] for m in _REAL_FETCH_MODELS()]
    assert ids == ["google/lyria-3-pro-preview"]
    ora._models_cache = None


def test_rejects_non_lyria_model_override(monkeypatch):
    """An explicit gpt-audio (or any non-Lyria) model must be rejected before
    any HTTP call — gpt-audio needs audio.voice and belongs to TTS."""
    from plugins.audio_gen import openrouter as ora

    provider = _provider(monkeypatch)
    called = {"stream": False}
    monkeypatch.setattr(ora.httpx, "stream", lambda *a, **k: called.__setitem__("stream", True))

    out = provider.generate("a song", model="openai/gpt-audio")
    assert out["success"] is False
    assert out["error_type"] == "unsupported_model"
    assert called["stream"] is False  # never hit the network


def test_incomplete_stream_without_done_is_error(monkeypatch):
    """Partial audio with no [DONE] must NOT be saved as success."""
    from plugins.audio_gen import openrouter as ora

    provider = _provider(monkeypatch)
    b64 = base64.b64encode(b"partial").decode()
    lines = [_sse({"choices": [{"delta": {"audio": {"data": b64}}}]})]  # no [DONE]

    saved = {"called": False}
    monkeypatch.setattr(ora.httpx, "stream", lambda *a, **k: _FakeStream(lines=lines))
    monkeypatch.setattr(ora, "save_b64_audio",
                        lambda *a, **k: saved.__setitem__("called", True))

    out = provider.generate("a song")
    assert out["success"] is False
    assert out["error_type"] == "incomplete_stream"
    assert saved["called"] is False  # never wrote a truncated file


def test_stream_error_object_surfaces(monkeypatch):
    """An error object inside a 200 stream is surfaced, not swallowed."""
    from plugins.audio_gen import openrouter as ora

    provider = _provider(monkeypatch)
    lines = [_sse({"error": {"message": "model overloaded"}}), "data: [DONE]"]
    monkeypatch.setattr(ora.httpx, "stream", lambda *a, **k: _FakeStream(lines=lines))

    out = provider.generate("a song")
    assert out["success"] is False
    assert "model overloaded" in out["error"]


def test_duration_is_clamped(monkeypatch):
    """Out-of-range duration is clamped to the advertised 1..60 range."""
    from plugins.audio_gen import openrouter as ora

    provider = _provider(monkeypatch)
    b64 = base64.b64encode(b"x").decode()
    lines = [_sse({"choices": [{"delta": {"audio": {"data": b64}}}]}), "data: [DONE]"]
    monkeypatch.setattr(ora.httpx, "stream", lambda *a, **k: _FakeStream(lines=lines))
    monkeypatch.setattr(ora, "save_b64_audio",
                        lambda d, *, prefix="a", extension="mp3": __import__("pathlib").Path(f"/tmp/x.{extension}"))

    out = provider.generate("a song", duration=999999)
    assert out["success"] is True
    assert out["duration"] == 60  # clamped to max

    out2 = provider.generate("a song", duration=-5)
    assert out2["success"] is True
    assert out2["duration"] == 1  # clamped to min
