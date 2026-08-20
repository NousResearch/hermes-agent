"""Tests for hermes_cli.azure_detect — transport & model auto-detection."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import azure_detect


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

class _FakeHTTPResponse:
    """Minimal stand-in for urllib.request.urlopen's context manager."""

    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body
        self.read_calls: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size: int = -1) -> bytes:
        self.read_calls.append(size)
        if size is None or size < 0:
            return self._body
        return self._body[:size]


def _openai_models_body(*ids: str) -> bytes:
    return json.dumps({
        "object": "list",
        "data": [{"id": i, "object": "model"} for i in ids],
    }).encode()


def _anthropic_error_body(msg: str = "model not found") -> bytes:
    return json.dumps({
        "type": "error",
        "error": {"type": "invalid_request_error", "message": msg},
    }).encode()


# ----------------------------------------------------------------------
# _looks_like_anthropic_path
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
# _extract_model_ids
# ----------------------------------------------------------------------





# ----------------------------------------------------------------------
# detect() integration
# ----------------------------------------------------------------------



def test_detect_openai_models_probe_success():
    """/models probe returning a model list → chat_completions."""
    def _fake_get(url, api_key, timeout=6.0, **kwargs):
        assert "key-abc" == api_key
        return 200, json.loads(_openai_models_body("gpt-5.4", "claude-opus-4-6"))

    with patch.object(azure_detect, "_http_get_json", side_effect=_fake_get):
        result = azure_detect.detect(
            "https://my.openai.azure.com/openai/v1", "key-abc",
        )
    assert result.api_mode == "chat_completions"
    assert result.models_probe_ok is True
    assert result.models == ["gpt-5.4", "claude-opus-4-6"]
    assert "/models" in result.reason


# ----------------------------------------------------------------------
# _probe_openai_models URL list (Azure vs v1 api-version)
# ----------------------------------------------------------------------

def test_probe_openai_models_tries_multiple_api_versions():
    """First call (no api-version) fails, api-version fallback succeeds."""
    calls = []

    def _fake_get(url, api_key, timeout=6.0, **kwargs):
        calls.append(url)
        if "api-version" not in url:
            return 404, None
        return 200, json.loads(_openai_models_body("gpt-4.1"))

    with patch.object(azure_detect, "_http_get_json", side_effect=_fake_get):
        ok, models = azure_detect._probe_openai_models(
            "https://my.openai.azure.com/openai/v1", "k",
        )
    assert ok is True
    assert models == ["gpt-4.1"]
    # Should have tried without api-version first, then with at least one
    assert any("api-version" not in u for u in calls)
    assert any("api-version" in u for u in calls)


# ----------------------------------------------------------------------
# _http_get_json error handling
# ----------------------------------------------------------------------



def test_http_get_json_bounds_success_body(monkeypatch):
    """Oversized model-list responses preserve status but skip JSON parsing."""
    monkeypatch.setattr(azure_detect, "_AZURE_DETECT_JSON_BODY_MAX_BYTES", 8)
    response = _FakeHTTPResponse(200, b"x" * 9)

    with patch.object(azure_detect, "open_credentialed_url", return_value=response):
        status, body = azure_detect._http_get_json("https://x/models", "k")

    assert status == 200
    assert body is None
    assert response.read_calls == [9]


def test_probe_anthropic_messages_bounds_error_body(monkeypatch):
    """Oversized HTTP error bodies are not scanned to detect Anthropic mode."""
    import urllib.error

    monkeypatch.setattr(azure_detect, "_AZURE_DETECT_ERROR_BODY_MAX_BYTES", 8)
    response = _FakeHTTPResponse(400, b"anthropic" + b"x")
    err = urllib.error.HTTPError("https://x/", 400, "Bad Request", {}, response)

    with patch.object(azure_detect, "open_credentialed_url", side_effect=err):
        assert azure_detect._probe_anthropic_messages("https://x/", "k") is False

    assert response.read_calls == [9]


# ----------------------------------------------------------------------
# lookup_context_length
# ----------------------------------------------------------------------



