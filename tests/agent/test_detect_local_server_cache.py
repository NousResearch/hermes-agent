"""Unit tests for detect_local_server_type cache + LM Studio v0 fallback (#24510).

The cache and the multi-version LM Studio probe are the two
mechanically-observable parts of the fix.  These tests pin both so a
future refactor that drops them surfaces the regression with an
explicit "#24510" message.

Every test mocks ``httpx.Client`` -- no real network traffic, runs on
every CI box.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_detect_cache():
    """Drop the module-level cache before and after each test so cases
    don't poison each other."""
    from agent.model_metadata import clear_local_server_type_cache
    clear_local_server_type_cache()
    yield
    clear_local_server_type_cache()


def _mock_client(get_side_effect):
    """Build an ``httpx.Client`` mock whose .get() drives the test."""
    client = MagicMock()
    client.__enter__ = lambda s: client
    client.__exit__ = MagicMock(return_value=False)
    client.get.side_effect = get_side_effect
    return client


def _resp(status_code: int, body=None, text: str = ""):
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    if body is not None:
        r.json.return_value = body
    return r


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


class TestDetectLocalServerTypeCache:
    """The cache is the heart of the #24510 fix -- pin it explicitly."""

    def test_second_call_hits_cache_does_not_reprobe(self):
        """First call probes; second call must return cached value
        without issuing any HTTP requests."""
        from agent.model_metadata import detect_local_server_type

        client = _mock_client([_resp(200)])  # LM Studio v1 path -- one probe
        with patch("httpx.Client", return_value=client) as mock_factory:
            first = detect_local_server_type("http://localhost:1234/v1")
            second = detect_local_server_type("http://localhost:1234/v1")

        assert first == "lm-studio"
        assert second == "lm-studio"
        # Crucial: only ONE httpx.Client constructed across both calls.
        assert mock_factory.call_count == 1, (
            "#24510 regression: detect_local_server_type re-probed on "
            "the second call -- the cache is not wired"
        )

    def test_negative_result_is_cached_too(self):
        """A "no server detected" answer must also be cached so a
        misconfigured base_url doesn't drag every subsequent message
        through 4 useless probes."""
        from agent.model_metadata import detect_local_server_type

        # All probes raise -- everything goes through the except branches
        # and the function returns None.
        client = _mock_client(Exception("unreachable"))
        with patch("httpx.Client", return_value=client) as mock_factory:
            first = detect_local_server_type("http://localhost:9999/v1")
            second = detect_local_server_type("http://localhost:9999/v1")

        assert first is None
        assert second is None
        assert mock_factory.call_count == 1, (
            "#24510 regression: negative results must be cached too"
        )

    def test_cache_keyed_by_url_plus_api_key(self):
        """Two different (url, api_key) pairs MUST not share a cache
        entry -- otherwise switching between local servers leaks state."""
        from agent.model_metadata import detect_local_server_type

        client = _mock_client([_resp(200)])
        with patch("httpx.Client", return_value=client) as mock_factory:
            detect_local_server_type("http://localhost:1234/v1", api_key="k1")
            detect_local_server_type("http://localhost:1234/v1", api_key="k2")
            detect_local_server_type("http://localhost:11434/v1", api_key="k1")

        assert mock_factory.call_count == 3, (
            "Each unique (url, api_key) must trigger its own probe"
        )

    def test_clear_cache_forces_reprobe(self):
        """``clear_local_server_type_cache`` must invalidate so the
        next call re-probes -- the public escape hatch for users who
        restart their local server mid-session."""
        from agent.model_metadata import (
            clear_local_server_type_cache,
            detect_local_server_type,
        )

        client = _mock_client([_resp(200)])
        with patch("httpx.Client", return_value=client) as mock_factory:
            detect_local_server_type("http://localhost:1234/v1")
            clear_local_server_type_cache()
            detect_local_server_type("http://localhost:1234/v1")

        assert mock_factory.call_count == 2

    def test_cache_expires_after_ttl(self, monkeypatch):
        """A stale cache entry self-heals after the TTL -- so a user
        who restarts their server with a different backend doesn't
        get the wrong answer forever."""
        from agent import model_metadata
        from agent.model_metadata import detect_local_server_type

        client = _mock_client([_resp(200)])
        with patch("httpx.Client", return_value=client) as mock_factory:
            detect_local_server_type("http://localhost:1234/v1")
            # Wind time past the TTL.
            future = model_metadata.time.monotonic() + model_metadata._DETECT_CACHE_TTL_S + 1.0
            monkeypatch.setattr(
                model_metadata.time, "monotonic", lambda: future,
            )
            detect_local_server_type("http://localhost:1234/v1")

        assert mock_factory.call_count == 2


# ---------------------------------------------------------------------------
# LM Studio multi-version probe (#24510)
# ---------------------------------------------------------------------------


class TestLmStudioVersionFallback:
    """LM Studio >= 0.4.0 ships /api/v1/models; 0.3.6 - 0.3.x only
    ships /api/v0/models.  Both must detect."""

    def test_detects_modern_lm_studio_via_api_v1(self):
        from agent.model_metadata import detect_local_server_type

        seen_paths: list[str] = []

        def fake_get(url, *a, **kw):
            seen_paths.append(url)
            return _resp(200) if "/api/v1/models" in url else _resp(404)

        client = _mock_client(fake_get)
        with patch("httpx.Client", return_value=client):
            result = detect_local_server_type("http://localhost:1234/v1")

        assert result == "lm-studio"
        assert any("/api/v1/models" in p for p in seen_paths)

    def test_detects_legacy_lm_studio_via_api_v0(self):
        """The actual #24510 win: older LM Studio used to detect as
        None, then the gateway fell through to non-LM-Studio code
        paths.  Now /api/v0/models also wins."""
        from agent.model_metadata import detect_local_server_type

        seen_paths: list[str] = []

        def fake_get(url, *a, **kw):
            seen_paths.append(url)
            # /api/v1/models 404s on legacy LM Studio.
            if "/api/v1/models" in url:
                return _resp(404)
            # /api/v0/models works on >= 0.3.6.
            if "/api/v0/models" in url:
                return _resp(200)
            return _resp(404)

        client = _mock_client(fake_get)
        with patch("httpx.Client", return_value=client):
            result = detect_local_server_type("http://localhost:1234/v1")

        assert result == "lm-studio", (
            "#24510 regression: legacy LM Studio (< 0.4.0) detection "
            "must succeed via /api/v0/models fallback"
        )
        # We must have actually probed v1 first (more specific) before
        # falling back to v0.
        v1_idx = next(i for i, p in enumerate(seen_paths) if "/api/v1/models" in p)
        v0_idx = next(i for i, p in enumerate(seen_paths) if "/api/v0/models" in p)
        assert v1_idx < v0_idx, "v1 must be probed before v0"

    def test_no_lm_studio_then_continues_to_other_probes(self):
        """When neither LM Studio path responds, detection must keep
        going (Ollama, llama.cpp, vLLM) instead of short-circuiting."""
        from agent.model_metadata import detect_local_server_type

        seen_paths: list[str] = []

        def fake_get(url, *a, **kw):
            seen_paths.append(url)
            if "/api/tags" in url:
                return _resp(200, body={"models": []})
            return _resp(404)

        client = _mock_client(fake_get)
        with patch("httpx.Client", return_value=client):
            result = detect_local_server_type("http://localhost:11434/v1")

        assert result == "ollama"
        assert any("/api/v1/models" in p for p in seen_paths)
        assert any("/api/v0/models" in p for p in seen_paths)
        assert any("/api/tags" in p for p in seen_paths)


# ---------------------------------------------------------------------------
# Probe timeout (#24510)
# ---------------------------------------------------------------------------


class TestProbeTimeout:
    """The 1.0s per-probe timeout caps the worst-case stall when a
    local server is wedged.  Pin the value -- tightening it further
    is fine, loosening it back to 2.0s reintroduces the heartbeat
    blocking."""

    def test_httpx_client_uses_one_second_timeout(self):
        from agent.model_metadata import detect_local_server_type

        client = _mock_client([_resp(200)])
        with patch("httpx.Client", return_value=client) as mock_factory:
            detect_local_server_type("http://localhost:1234/v1")

        timeout = mock_factory.call_args.kwargs.get("timeout")
        assert timeout == 1.0, (
            "#24510 regression: per-probe timeout regressed from 1.0s "
            f"-- got {timeout}.  Loosening this reintroduces the "
            "Discord heartbeat blocking the issue was filed about."
        )
