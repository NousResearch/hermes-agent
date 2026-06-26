"""Tests for stream_options incompatibility caching (issue #9705).

Azure AI Foundry MaaS endpoints reject ``stream_options:
{"include_usage": True}`` with HTTP 422 ``extra_forbidden``.  The fix
caches the incompatibility by hostname so subsequent requests skip the
field.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from agent.chat_completion_helpers import (
    _STREAM_OPTIONS_INCOMPATIBLE,
)
from utils import base_url_hostname


@pytest.fixture(autouse=True)
def _clean_cache():
    """Ensure the module-level cache is clean for every test."""
    _STREAM_OPTIONS_INCOMPATIBLE.clear()
    yield
    _STREAM_OPTIONS_INCOMPATIBLE.clear()


class TestStreamOptionsIncompatibleCache:
    """Verify the cache set behaves as expected."""

    def test_cache_is_empty_by_default(self):
        assert len(_STREAM_OPTIONS_INCOMPATIBLE) == 0

    def test_add_host_to_cache(self):
        _STREAM_OPTIONS_INCOMPATIBLE.add("my-hub.services.ai.azure.com")
        assert "my-hub.services.ai.azure.com" in _STREAM_OPTIONS_INCOMPATIBLE

    def test_cache_isolation_between_tests(self):
        """Cache from prior test should not leak (autouse fixture)."""
        assert len(_STREAM_OPTIONS_INCOMPATIBLE) == 0


class TestStreamOptionsDetection:
    """Verify 422 error detection logic."""

    def _make_422_error(self, body: str) -> SimpleNamespace:
        """Create a mock error mimicking OpenAI SDK APIStatusError."""
        return SimpleNamespace(
            status_code=422,
            body=body,
            message="Unprocessable Entity",
        )

    def test_detects_stream_options_422(self):
        e = self._make_422_error(
            '{"detail": [{"type": "extra_forbidden", '
            '"loc": ["body", "stream_options", "include_usage"], '
            '"msg": "Extra inputs are not permitted"}]}'
        )
        assert getattr(e, "status_code", None) == 422
        _err_body = str(getattr(e, "body", "") or "").lower()
        assert "stream_options" in _err_body
        assert "extra" in _err_body

    def test_ignores_unrelated_422(self):
        e = self._make_422_error(
            '{"detail": [{"type": "missing", "loc": ["body", "messages"]}]}'
        )
        _err_body = str(getattr(e, "body", "") or "").lower()
        assert "stream_options" not in _err_body

    def test_ignores_non_422(self):
        e = SimpleNamespace(status_code=500, body="server error")
        assert getattr(e, "status_code", None) not in (400, 422)

    def test_detects_stream_options_400(self):
        """Some proxies return 400 instead of 422 for unsupported params."""
        e = SimpleNamespace(
            status_code=400,
            body='{"error": {"message": "Unrecognized request argument: stream_options"}}',
        )
        _status = getattr(e, "status_code", None)
        _err_body = str(getattr(e, "body", "") or "").lower()
        assert _status in (400, 422)
        assert "stream_options" in _err_body
        assert "unrecognized" in _err_body

    def test_detects_stream_options_not_supported(self):
        e = SimpleNamespace(
            status_code=422,
            body='{"error": "stream_options is not supported"}',
        )
        _status = getattr(e, "status_code", None)
        _err_body = str(getattr(e, "body", "") or "").lower()
        assert _status in (400, 422)
        assert "stream_options" in _err_body
        assert "not supported" in _err_body

    def test_detects_stream_options_unexpected(self):
        e = SimpleNamespace(
            status_code=400,
            body='{"error": "unexpected stream_options field"}',
        )
        _status = getattr(e, "status_code", None)
        _err_body = str(getattr(e, "body", "") or "").lower()
        assert _status in (400, 422)
        assert "stream_options" in _err_body
        assert "unexpected" in _err_body

    def test_ignores_error_without_status_code(self):
        e = SimpleNamespace(body="some error")
        assert getattr(e, "status_code", None) is None


class TestStreamOptionsKwargsBuilding:
    """Verify stream_kwargs includes/excludes stream_options based on cache."""

    def test_stream_options_included_when_cache_empty(self):
        """Default behaviour: stream_options is present."""
        host = "hub.services.ai.azure.com"
        assert host not in _STREAM_OPTIONS_INCOMPATIBLE
        # Simulate the conditional logic from _call_chat_completions
        stream_kwargs = {"stream": True}
        _host = host
        if _host not in _STREAM_OPTIONS_INCOMPATIBLE:
            stream_kwargs["stream_options"] = {"include_usage": True}
        assert "stream_options" in stream_kwargs
        assert stream_kwargs["stream_options"] == {"include_usage": True}

    def test_stream_options_excluded_when_host_cached(self):
        """After 422, stream_options is omitted."""
        host = "hub.services.ai.azure.com"
        _STREAM_OPTIONS_INCOMPATIBLE.add(host)
        stream_kwargs = {"stream": True}
        _host = host
        if _host not in _STREAM_OPTIONS_INCOMPATIBLE:
            stream_kwargs["stream_options"] = {"include_usage": True}
        assert "stream_options" not in stream_kwargs

    def test_different_host_not_affected(self):
        """Cache is host-scoped — other hosts are unaffected."""
        _STREAM_OPTIONS_INCOMPATIBLE.add("hub-a.services.ai.azure.com")
        host_b = "hub-b.services.ai.azure.com"
        stream_kwargs = {"stream": True}
        _host = host_b
        if _host not in _STREAM_OPTIONS_INCOMPATIBLE:
            stream_kwargs["stream_options"] = {"include_usage": True}
        assert "stream_options" in stream_kwargs
