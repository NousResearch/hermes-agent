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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


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

@pytest.mark.parametrize("url, expected", [
    ("https://foo.services.ai.azure.com/anthropic", True),
    ("https://foo.services.ai.azure.com/anthropic/", True),
    ("https://foo.services.ai.azure.com/anthropic/v1", True),
    ("https://foo.openai.azure.com/openai/v1", False),
    ("https://foo.openai.azure.com/", False),
    ("https://openrouter.ai/api/v1", False),
])
def test_looks_like_anthropic_path(url, expected):
    assert azure_detect._looks_like_anthropic_path(url) is expected


# ----------------------------------------------------------------------
# _extract_model_ids
# ----------------------------------------------------------------------

def test_extract_model_ids_openai_shape():
    body = {
        "object": "list",
        "data": [
            {"id": "gpt-4.1-mini", "object": "model"},
            {"id": "claude-sonnet-4-6", "object": "model"},
        ],
    }
    assert azure_detect._extract_model_ids(body) == ["gpt-4.1-mini", "claude-sonnet-4-6"]


def test_extract_model_ids_bad_shape_returns_empty():
    assert azure_detect._extract_model_ids({}) == []
    assert azure_detect._extract_model_ids({"data": "not-a-list"}) == []
    assert azure_detect._extract_model_ids({"data": [{"no-id": True}]}) == []


# ----------------------------------------------------------------------
# detect() integration
# ----------------------------------------------------------------------

def test_detect_anthropic_path_wins_without_http():
    """URL path sniff short-circuits — no HTTP call happens."""
    with patch.object(azure_detect, "_http_get_json") as fake_get, \
         patch.object(azure_detect, "_probe_anthropic_messages") as fake_probe:
        result = azure_detect.detect(
            "https://foo.services.ai.azure.com/anthropic", "key-abc",
        )
        assert result.api_mode == "anthropic_messages"
        assert result.is_anthropic is True
        assert "path" in result.reason.lower()
        fake_get.assert_not_called()
        fake_probe.assert_not_called()


def test_detect_openai_models_probe_success():
    """/models probe returning a model list → chat_completions."""
    def _fake_get(url, api_key, timeout=6.0):
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


def test_detect_openai_models_probe_empty_list_still_counts():
    """Endpoint returned OpenAI shape but no models → still chat_completions."""
    def _fake_get(url, api_key, timeout=6.0):
        return 200, {"object": "list", "data": []}

    with patch.object(azure_detect, "_http_get_json", side_effect=_fake_get):
        result = azure_detect.detect(
            "https://my.openai.azure.com/openai/v1", "key-abc",
        )
    assert result.api_mode == "chat_completions"
    assert result.models == []
    assert result.models_probe_ok is True


def test_detect_falls_back_to_anthropic_probe():
    """/models fails but Anthropic Messages probe succeeds."""
    def _fake_get(url, api_key, timeout=6.0):
        return 401, None  # /models forbidden

    with patch.object(azure_detect, "_http_get_json", side_effect=_fake_get), \
         patch.object(azure_detect, "_probe_anthropic_messages", return_value=True):
        result = azure_detect.detect(
            "https://my.services.ai.azure.com/v1", "key-abc",
        )
    assert result.api_mode == "anthropic_messages"
    assert result.is_anthropic is True


def test_detect_all_probes_fail_returns_none():
    """Every probe fails → api_mode is None and caller falls back to manual."""
    with patch.object(azure_detect, "_http_get_json", return_value=(500, None)), \
         patch.object(azure_detect, "_probe_anthropic_messages", return_value=False):
        result = azure_detect.detect(
            "https://some-private.example.com/", "key-abc",
        )
    assert result.api_mode is None
    assert result.models == []
    assert "manual" in result.reason.lower()


# ----------------------------------------------------------------------
# _probe_openai_models URL list (Azure vs v1 api-version)
# ----------------------------------------------------------------------

def test_probe_openai_models_tries_multiple_api_versions():
    """First call (no api-version) fails, api-version fallback succeeds."""
    calls = []

    def _fake_get(url, api_key, timeout=6.0):
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

def test_http_get_json_on_urlerror_returns_zero_none():
    """Network failure returns (0, None), never raises."""
    import urllib.error
    with patch("hermes_cli.azure_detect.urllib_request.urlopen",
               side_effect=urllib.error.URLError("dns fail")):
        status, body = azure_detect._http_get_json("https://bad.example/", "k")
    assert status == 0
    assert body is None


def test_http_get_json_on_http_error_returns_code_none():
    """HTTP 4xx/5xx returns (code, None)."""
    import urllib.error
    err = urllib.error.HTTPError("https://x/", 403, "Forbidden", {}, None)
    with patch("hermes_cli.azure_detect.urllib_request.urlopen", side_effect=err):
        status, body = azure_detect._http_get_json("https://x/", "k")
    assert status == 403
    assert body is None


# ----------------------------------------------------------------------
# lookup_context_length
# ----------------------------------------------------------------------

def test_lookup_context_length_returns_known():
    """When model_metadata returns a non-fallback value, we pass it through."""
    fake = MagicMock(return_value=400000)
    with patch("agent.model_metadata.get_model_context_length", fake), \
         patch("agent.model_metadata.DEFAULT_FALLBACK_CONTEXT", 128000):
        n = azure_detect.lookup_context_length(
            "gpt-5.4", "https://x.openai.azure.com/openai/v1", "k",
        )
    assert n == 400000


def test_lookup_context_length_returns_none_on_fallback():
    """When resolver falls through to DEFAULT_FALLBACK_CONTEXT, we return None."""
    with patch("agent.model_metadata.get_model_context_length", return_value=128000), \
         patch("agent.model_metadata.DEFAULT_FALLBACK_CONTEXT", 128000):
        n = azure_detect.lookup_context_length(
            "totally-unknown-model", "https://x.openai.azure.com/openai/v1", "k",
        )
    assert n is None


def test_lookup_context_length_swallows_exceptions():
    """Resolver raising must not crash the wizard."""
    with patch("agent.model_metadata.get_model_context_length",
               side_effect=RuntimeError("boom")):
        assert azure_detect.lookup_context_length("m", "https://x/", "k") is None
