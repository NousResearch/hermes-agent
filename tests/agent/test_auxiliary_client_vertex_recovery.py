"""Tests for the auxiliary Vertex 401 recovery path.

The Gemini/openapi Vertex path bakes a frozen OAuth2 bearer token into the
OpenAI client at build time. After the ~1h token lifetime, every auxiliary
call (compression, title_generation, background_review, ...) fails with
401 UNAUTHENTICATED / ACCESS_TOKEN_TYPE_UNSUPPORTED until the process
restarts. Seen live Jul 2026: hours of context_compressor failures in a
long-lived desktop session.

Recovery contract:
  1. _auth_refresh_provider_for_route maps aiplatform.googleapis.com hosts
     (global AND regional "{region}-aiplatform...") to "vertex".
  2. _refresh_provider_credentials("vertex") force re-mints the token
     (via agent.vertex_adapter.refresh_vertex_credentials) and evicts the
     stale cached clients so the retry builds against a fresh token.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.auxiliary_client import (
    _auth_refresh_provider_for_route,
    _is_vertex_host,
    _refresh_provider_credentials,
)


# ── Host routing ─────────────────────────────────────────────────────────────

class TestVertexHostRouting:
    def test_global_endpoint_maps_to_vertex(self):
        url = ("https://aiplatform.googleapis.com/v1beta1/projects/p/"
               "locations/global/endpoints/openapi")
        assert _auth_refresh_provider_for_route("auto", url) == "vertex"

    def test_regional_endpoint_maps_to_vertex(self):
        url = ("https://us-east4-aiplatform.googleapis.com/v1beta1/projects/p/"
               "locations/us-east4/endpoints/openapi")
        assert _auth_refresh_provider_for_route("auto", url) == "vertex"

    def test_lookalike_host_does_not_match(self):
        # Substring lookalikes must not be treated as Vertex.
        url = "https://evilaiplatform.googleapis.com.example/v1"
        assert _auth_refresh_provider_for_route("auto", url) != "vertex"

    def test_empty_and_garbage_urls_do_not_crash(self):
        for url in ("", None, "not a url", "://"):
            assert _is_vertex_host(url or "") is False
        assert _auth_refresh_provider_for_route("auto", "") == "auto"

    def test_explicit_vertex_provider_passes_through(self):
        assert _auth_refresh_provider_for_route("vertex", "") == "vertex"


# ── Host-based cache sweep ───────────────────────────────────────────────────

class TestVertexClientCacheSweep:
    def test_sweep_evicts_vertex_clients_under_any_label(self):
        """Clients cached under 'auto'/task labels (not 'vertex') must still be
        evicted — they hold the same dead frozen token."""
        import agent.auxiliary_client as ac

        class _FakeClient:
            def __init__(self, base_url):
                self.base_url = base_url

        vertex_url = ("https://aiplatform.googleapis.com/v1beta1/projects/p/"
                      "locations/global/endpoints/openapi")
        other_url = "https://api.openai.com/v1"
        k_auto = ("auto", False, vertex_url, "tok", None, None, False, None, "m")
        k_task = ("compression-alias", False, vertex_url, "tok", None, None, False, None, "m")
        k_keep = ("openai", False, other_url, "sk", None, None, False, None, "m")
        with ac._client_cache_lock:
            ac._client_cache[k_auto] = (_FakeClient(vertex_url), "m", None)
            ac._client_cache[k_task] = (_FakeClient(vertex_url), "m", None)
            ac._client_cache[k_keep] = (_FakeClient(other_url), "m", None)
        try:
            ac._evict_cached_vertex_clients()
            with ac._client_cache_lock:
                assert k_auto not in ac._client_cache
                assert k_task not in ac._client_cache
                assert k_keep in ac._client_cache
        finally:
            with ac._client_cache_lock:
                for k in (k_auto, k_task, k_keep):
                    ac._client_cache.pop(k, None)


# ── Credential refresh branch ────────────────────────────────────────────────

class TestVertexCredentialRefresh:
    @pytest.mark.parametrize("name", ["vertex", "google-vertex", "vertex-ai",
                                      "gcp-vertex", "vertexai"])
    def test_vertex_spellings_hit_remint_and_evict(self, name):
        with patch("agent.vertex_adapter.refresh_vertex_credentials",
                   return_value=True) as mock_remint, \
             patch("agent.auxiliary_client._evict_cached_clients") as mock_evict, \
             patch("agent.auxiliary_client._evict_cached_vertex_clients") as mock_sweep:
            assert _refresh_provider_credentials(name) is True
        mock_remint.assert_called_once()
        mock_evict.assert_called_once()
        mock_sweep.assert_called_once()

    def test_remint_failure_returns_false_without_evicting(self):
        with patch("agent.vertex_adapter.refresh_vertex_credentials",
                   return_value=False), \
             patch("agent.auxiliary_client._evict_cached_clients") as mock_evict, \
             patch("agent.auxiliary_client._evict_cached_vertex_clients") as mock_sweep:
            assert _refresh_provider_credentials("vertex") is False
        mock_evict.assert_not_called()
        mock_sweep.assert_not_called()

    def test_remint_exception_returns_false(self):
        """A crash inside the re-mint must be swallowed (best-effort recovery)."""
        with patch("agent.vertex_adapter.refresh_vertex_credentials",
                   side_effect=RuntimeError("boom")):
            assert _refresh_provider_credentials("vertex") is False
