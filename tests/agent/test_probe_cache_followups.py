"""Tests for probe-cache follow-ups on the #29988/#37595/#50572 salvage.

Covers:
- _query_ollama_api_show TTL caching (positive-only, namespaced key)
- persistent context-cache key normalization (trailing-slash dedup)
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """Module-level TTL cache must not leak between tests."""
    from agent import model_metadata
    model_metadata._LOCAL_CTX_PROBE_CACHE.clear()
    yield
    model_metadata._LOCAL_CTX_PROBE_CACHE.clear()


def _mock_show_response(ctx=131072):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "model_info": {"llama.context_length": ctx},
        "parameters": "",
    }
    return resp


def _client_mock(resp):
    client = MagicMock()
    client.__enter__ = lambda s: client
    client.__exit__ = MagicMock(return_value=False)
    client.post.return_value = resp
    return client


class TestOllamaApiShowCaching:
    def test_positive_result_cached_within_ttl(self):
        from agent.model_metadata import _query_ollama_api_show

        client = _client_mock(_mock_show_response(131072))
        with patch("httpx.Client", return_value=client):
            first = _query_ollama_api_show("llama3", "http://127.0.0.1:11434")
            second = _query_ollama_api_show("llama3", "http://127.0.0.1:11434")

        assert first == second == 131072
        assert client.post.call_count == 1  # second call served from cache

    def test_failure_never_memoized(self):
        """A down server must be re-probed on the next call (startup race)."""
        from agent.model_metadata import _query_ollama_api_show

        bad = MagicMock()
        bad.status_code = 404
        client = _client_mock(bad)
        with patch("httpx.Client", return_value=client):
            assert _query_ollama_api_show("llama3", "http://127.0.0.1:11434") is None
            assert _query_ollama_api_show("llama3", "http://127.0.0.1:11434") is None

        assert client.post.call_count == 2  # None was NOT cached

    def test_cache_key_does_not_collide_with_local_ctx_probe(self):
        """The ollama_show namespace must not read _query_local_context_length rows."""
        from agent import model_metadata
        from agent.model_metadata import _query_ollama_api_show
        import time as _time

        # Seed a same-(model,url) entry under the sibling probe's key shape.
        model_metadata._LOCAL_CTX_PROBE_CACHE[("llama3", "http://127.0.0.1:11434")] = (
            999, _time.monotonic(),
        )

        client = _client_mock(_mock_show_response(131072))
        with patch("httpx.Client", return_value=client):
            result = _query_ollama_api_show("llama3", "http://127.0.0.1:11434")

        assert result == 131072  # probed for real, not the sibling's 999
        assert client.post.call_count == 1


class TestLocalhostIPv4SiblingSites:
    """#37595 widened: every probe helper rewrites localhost→127.0.0.1,
    not just detect_local_server_type."""

    def test_helper_rewrites_all_forms(self):
        from agent.model_metadata import _localhost_to_ipv4

        assert _localhost_to_ipv4("http://localhost:1234/v1") == "http://127.0.0.1:1234/v1"
        assert _localhost_to_ipv4("http://localhost/v1") == "http://127.0.0.1/v1"
        assert _localhost_to_ipv4("http://localhost") == "http://127.0.0.1"
        # Non-localhost passes through untouched.
        assert _localhost_to_ipv4("http://192.168.1.10:8080") == "http://192.168.1.10:8080"
        assert _localhost_to_ipv4("https://api.openai.com/v1") == "https://api.openai.com/v1"
        assert _localhost_to_ipv4("") == ""

    def test_ollama_api_show_probes_ipv4(self):
        from agent.model_metadata import _query_ollama_api_show

        client = _client_mock(_mock_show_response(131072))
        with patch("httpx.Client", return_value=client):
            _query_ollama_api_show("llama3", "http://localhost:11434")

        assert client.post.call_args[0][0].startswith("http://127.0.0.1:11434")

    def test_query_ollama_num_ctx_probes_ipv4(self):
        from agent.model_metadata import query_ollama_num_ctx

        client = _client_mock(_mock_show_response(131072))
        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"), \
             patch("httpx.Client", return_value=client):
            query_ollama_num_ctx("llama3", "http://localhost:11434")

        assert client.post.call_args[0][0].startswith("http://127.0.0.1:11434")


class TestContextCacheKeyNormalization:
    def test_trailing_slash_variants_share_one_entry(self, tmp_path, monkeypatch):
        from agent import model_metadata

        monkeypatch.setattr(
            model_metadata, "_get_context_cache_path",
            lambda: tmp_path / "context_lengths.yaml",
        )

        model_metadata.save_context_length("m1", "http://host/v1/", 200_000)
        # Both slash variants resolve to the same row.
        assert model_metadata.get_cached_context_length("m1", "http://host/v1") == 200_000
        assert model_metadata.get_cached_context_length("m1", "http://host/v1/") == 200_000

        cache = model_metadata._load_context_cache()
        assert list(cache.keys()) == ["m1@http://host/v1"]

    def test_legacy_unnormalized_row_still_honored(self, tmp_path, monkeypatch):
        """Rows written pre-normalization (trailing slash in key) must not force a re-probe."""
        import yaml
        from agent import model_metadata

        path = tmp_path / "context_lengths.yaml"
        monkeypatch.setattr(model_metadata, "_get_context_cache_path", lambda: path)
        path.write_text(yaml.dump({"context_lengths": {"m1@http://host/v1/": 128_000}}))

        assert model_metadata.get_cached_context_length("m1", "http://host/v1/") == 128_000

    def test_invalidate_clears_both_key_shapes(self, tmp_path, monkeypatch):
        import yaml
        from agent import model_metadata

        path = tmp_path / "context_lengths.yaml"
        monkeypatch.setattr(model_metadata, "_get_context_cache_path", lambda: path)
        path.write_text(yaml.dump({"context_lengths": {
            "m1@http://host/v1": 128_000,
            "m1@http://host/v1/": 64_000,
        }}))

        model_metadata._invalidate_cached_context_length("m1", "http://host/v1/")
        cache = model_metadata._load_context_cache()
        assert "m1@http://host/v1" not in cache
        assert "m1@http://host/v1/" not in cache
