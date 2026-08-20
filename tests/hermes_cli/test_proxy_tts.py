"""Tests for xAI TTS path/body transforms and optional adapter fallback arity."""

from __future__ import annotations

import json

from hermes_cli.proxy.adapters.xai import XAIGrokAdapter


def test_xai_adapter_allows_tts_paths():
    adapter = XAIGrokAdapter()
    assert "/audio/speech" in adapter.allowed_paths
    assert "/tts" in adapter.allowed_paths


def test_xai_adapter_map_path_rewrites_openai_tts():
    adapter = XAIGrokAdapter()
    assert adapter.map_path("/audio/speech") == "/tts"
    assert adapter.map_path("/audio/speech/") == "/tts"
    assert adapter.map_path("/Audio/Speech") == "/tts"
    assert adapter.map_path("/tts") == "/tts"
    assert adapter.map_path("/chat/completions") == "/chat/completions"


def test_xai_adapter_transform_request_body_openai_to_xai_tts():
    adapter = XAIGrokAdapter()
    original = json.dumps({
        "model": "tts-1",
        "input": "Hello from Hermes",
        "voice": "alloy",
    }).encode("utf-8")

    transformed = adapter.transform_request_body("/audio/speech", original)
    data = json.loads(transformed)

    assert data["text"] == "Hello from Hermes"
    assert data["voice_id"] == "alloy"
    assert "input" not in data
    assert "voice" not in data
    assert "model" not in data
    assert data["text_normalization"] is True
    assert data["language"] == "auto"


def test_xai_adapter_transform_request_body_passthrough_non_tts():
    adapter = XAIGrokAdapter()
    original = b'{"model":"grok","messages":[]}'
    assert adapter.transform_request_body("/chat/completions", original) is original


def test_transform_request_body_noop_fallback_accepts_path_and_body():
    """Mirrors server.py getattr fallback: must accept (path, body), not body alone."""
    noop = lambda _path, b: b  # noqa: E731 — matches hermes_cli.proxy.server fallback
    body = b'{"hello":"world"}'
    assert noop("/chat/completions", body) is body
