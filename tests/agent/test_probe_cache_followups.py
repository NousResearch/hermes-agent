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
    """Module-level caches must not leak between tests."""
    from agent import model_metadata
    model_metadata._LOCAL_CTX_PROBE_CACHE.clear()
    model_metadata._endpoint_probe_path_cache.clear()
    yield
    model_metadata._LOCAL_CTX_PROBE_CACHE.clear()
    model_metadata._endpoint_probe_path_cache.clear()


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

    def test_ttl_expiry_reprobes(self):
        """After the 30s TTL lapses, the next call must hit the network again."""
        from agent import model_metadata
        from agent.model_metadata import _query_ollama_api_show
        import time as _time

        client = _client_mock(_mock_show_response(131072))
        with patch("httpx.Client", return_value=client):
            _query_ollama_api_show("llama3", "http://127.0.0.1:11434")
            # Age the entry past the TTL.
            ((key, (val, _ts)),) = list(model_metadata._LOCAL_CTX_PROBE_CACHE.items())
            model_metadata._LOCAL_CTX_PROBE_CACHE[key] = (
                val, _time.monotonic() - model_metadata._LOCAL_CTX_PROBE_TTL_SECONDS - 1,
            )
            _query_ollama_api_show("llama3", "http://127.0.0.1:11434")

        assert client.post.call_count == 2  # expired entry re-probed

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


class TestDetectLocalServerTypeCache:
    """#29988: detect_local_server_type memoized with a bounded TTL."""

    def _get_client(self, server_type="ollama"):
        ollama_resp = MagicMock()
        ollama_resp.status_code = 200
        ollama_resp.json.return_value = {"models": []}
        miss = MagicMock()
        miss.status_code = 404

        client = MagicMock()
        client.__enter__ = lambda s: client
        client.__exit__ = MagicMock(return_value=False)

        def _get(url, *a, **k):
            if url.endswith("/api/tags"):
                return ollama_resp
            return miss

        client.get.side_effect = _get
        return client

    def test_second_call_served_from_cache(self):
        from agent.model_metadata import detect_local_server_type

        client = self._get_client()
        with patch("httpx.Client", return_value=client):
            first = detect_local_server_type("http://127.0.0.1:11434")
            calls_after_first = client.get.call_count
            second = detect_local_server_type("http://127.0.0.1:11434")

        assert first == second == "ollama"
        assert client.get.call_count == calls_after_first  # no new HTTP traffic

    def test_ttl_expiry_allows_server_swap_redetection(self):
        """Stopping Ollama and starting LM Studio on the same port must be
        re-detected once the TTL lapses — the cache is bounded, not
        process-lifetime."""
        from agent import model_metadata
        from agent.model_metadata import detect_local_server_type
        import time as _time

        client = self._get_client()
        with patch("httpx.Client", return_value=client):
            assert detect_local_server_type("http://127.0.0.1:11434") == "ollama"

        # Age the entry past the TTL, then swap the backend behind the URL.
        ((key, (val, _ts)),) = list(model_metadata._endpoint_probe_path_cache.items())
        model_metadata._endpoint_probe_path_cache[key] = (
            val, _time.monotonic() - model_metadata._ENDPOINT_PROBE_TTL_SECONDS - 1,
        )

        lmstudio_resp = MagicMock()
        lmstudio_resp.status_code = 200
        # Realistic LM Studio native /api/v1/models entry (carries LM
        # Studio-specific fields the OpenAI envelope never has). Detection now
        # requires this signature, not a bare 200 (see the misdetection test
        # below), so the mock must look genuinely LM Studio-native.
        lmstudio_resp.json.return_value = {
            "data": [{
                "id": "qwen/qwen3-4b",
                "type": "llm",
                "state": "loaded",
                "max_context_length": 32768,
                "loaded_instances": [{"config": {"context_length": 32768}}],
            }]
        }
        swap_client = MagicMock()
        swap_client.__enter__ = lambda s: swap_client
        swap_client.__exit__ = MagicMock(return_value=False)

        def _get(url, *a, **k):
            if url.endswith("/api/v1/models"):
                return lmstudio_resp
            miss = MagicMock(); miss.status_code = 404
            return miss

        swap_client.get.side_effect = _get
        with patch("httpx.Client", return_value=swap_client):
            assert detect_local_server_type("http://127.0.0.1:11434") == "lm-studio"

    def test_openai_compat_proxy_not_misdetected_as_lmstudio(self):
        """Regression: an OpenAI-compatible server (e.g. a loopback Claude/
        Anthropic proxy) that answers /api/v1/models with a 200 must NOT be
        classified as LM Studio just because the status is 200.

        The bug: such a proxy returns the standard OpenAI listing shape
        ({"object":"list","data":[{"id","object":"model",...}]}); the old
        detector accepted any 200 on /api/v1/models as LM Studio, so the caller
        went down the LM Studio metadata parser, found none of its fields, and
        resolved context_length to None → probe-tier fallback (128K on a 1M
        model → auto-compaction at ~96K). Detection must inspect the entry
        SHAPE. The proxy also answers /api/tags with {"models":[...]}, so it
        legitimately classifies as ollama here — the KEY assertion is simply
        `!= "lm-studio"` (which is what breaks the metadata parse)."""
        from agent import model_metadata
        from agent.model_metadata import detect_local_server_type

        model_metadata._endpoint_probe_path_cache.clear()

        # OpenAI listing shape — exactly what claude-apx / claude-bpx return.
        openai_models = MagicMock()
        openai_models.status_code = 200
        openai_models.json.return_value = {
            "object": "list",
            "data": [
                {"id": "claude-fable-5", "object": "model", "created": 1,
                 "owned_by": "anthropic", "context_length": 1000000},
            ],
        }
        openai_tags = MagicMock()
        openai_tags.status_code = 200
        openai_tags.json.return_value = {
            "models": [{"name": "claude-fable-5", "model": "claude-fable-5",
                        "context_length": 1000000, "details": {}}],
        }

        client = MagicMock()
        client.__enter__ = lambda s: client
        client.__exit__ = MagicMock(return_value=False)

        def _get(url, *a, **k):
            if url.endswith("/api/v1/models"):
                return openai_models
            if url.endswith("/api/tags"):
                return openai_tags
            miss = MagicMock(); miss.status_code = 404
            return miss

        client.get.side_effect = _get
        with patch("httpx.Client", return_value=client):
            detected = detect_local_server_type("http://127.0.0.1:18801/anthropic")

        assert detected != "lm-studio", (
            f"OpenAI-compat proxy must not be misdetected as lm-studio; got {detected!r} "
            "(this is the misdetection that collapsed context_length to a probe-tier default)"
        )

    def test_is_lmstudio_models_payload_shape_discrimination(self):
        """Unit-level: the shape discriminator accepts genuine LM Studio
        payloads and rejects the OpenAI envelope."""
        from agent.model_metadata import _is_lmstudio_models_payload

        def _resp(payload):
            r = MagicMock(); r.json.return_value = payload; return r

        # Genuine LM Studio native entries (any one strong marker suffices).
        assert _is_lmstudio_models_payload(_resp(
            {"data": [{"id": "m", "loaded_instances": []}]}))
        assert _is_lmstudio_models_payload(_resp(
            {"models": [{"key": "pub/m"}]}))
        assert _is_lmstudio_models_payload(_resp(
            {"data": [{"id": "m", "max_context_length": 32768}]}))
        # `key` alone is now a decisive marker (docstring↔tuple consistency).
        assert _is_lmstudio_models_payload(_resp(
            {"data": [{"key": "publisher/model-a"}]}))
        # `type`+`state` together is an acceptable weaker signal.
        assert _is_lmstudio_models_payload(_resp(
            {"data": [{"id": "m", "type": "llm", "state": "loaded"}]}))
        # An idle LM Studio (no models) on the native `models` key is accepted.
        assert _is_lmstudio_models_payload(_resp({"models": []}))
        # OpenAI envelope — must be rejected.
        assert not _is_lmstudio_models_payload(_resp(
            {"object": "list", "data": [{"id": "m", "object": "model", "owned_by": "anthropic"}]}))
        # Even without object=="list", a bare OpenAI-ish entry with no LM Studio
        # markers is rejected (fail-closed).
        assert not _is_lmstudio_models_payload(_resp(
            {"data": [{"id": "m", "object": "model", "created": 1, "owned_by": "x"}]}))
        # A lone generic `type` or `state` (no strong marker, not both) is NOT
        # enough — avoids false-positiving a bespoke proxy.
        assert not _is_lmstudio_models_payload(_resp(
            {"data": [{"id": "m", "type": "chat"}]}))
        assert not _is_lmstudio_models_payload(_resp(
            {"data": [{"id": "m", "state": "ready"}]}))
        # An ambiguous empty `{"data": []}` (no native `models` key) fails closed.
        assert not _is_lmstudio_models_payload(_resp({"data": []}))
        # Malformed / empty → False, never raises.
        assert not _is_lmstudio_models_payload(_resp({}))
        bad = MagicMock(); bad.json.side_effect = ValueError("no json")
        assert not _is_lmstudio_models_payload(bad)


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

    def test_rewrite_is_host_only_not_substring(self):
        """A URL that merely EMBEDS 'http://localhost' in its path/query must
        not be corrupted — only the URL's own host is rewritten."""
        from agent.model_metadata import _localhost_to_ipv4

        proxied = "https://proxy.example.com/route?upstream=http://localhost:11434"
        assert _localhost_to_ipv4(proxied) == proxied
        # Host must be a full label: localhost.example.com is NOT localhost.
        assert _localhost_to_ipv4("http://localhost.example.com/v1") == (
            "http://localhost.example.com/v1"
        )

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

    def test_legacy_slashed_row_found_with_normalized_caller(self, tmp_path, monkeypatch):
        """Reverse migration direction: old row has the slash, current runtime
        passes the normalized no-slash URL — must still hit, not re-probe."""
        import yaml
        from agent import model_metadata

        path = tmp_path / "context_lengths.yaml"
        monkeypatch.setattr(model_metadata, "_get_context_cache_path", lambda: path)
        path.write_text(yaml.dump({"context_lengths": {"m1@http://host/v1/": 128_000}}))

        assert model_metadata.get_cached_context_length("m1", "http://host/v1") == 128_000

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

    def test_invalidate_with_normalized_caller_clears_legacy_row(self, tmp_path, monkeypatch):
        """Reverse direction: invalidating with the no-slash URL must also
        drop a legacy slashed row, or the next lookup resurrects stale data."""
        import yaml
        from agent import model_metadata

        path = tmp_path / "context_lengths.yaml"
        monkeypatch.setattr(model_metadata, "_get_context_cache_path", lambda: path)
        path.write_text(yaml.dump({"context_lengths": {"m1@http://host/v1/": 64_000}}))

        model_metadata._invalidate_cached_context_length("m1", "http://host/v1")
        assert model_metadata.get_cached_context_length("m1", "http://host/v1") is None
        assert model_metadata.get_cached_context_length("m1", "http://host/v1/") is None

    def test_invalidate_also_drops_in_memory_probe_entries(self, tmp_path, monkeypatch):
        """Disk invalidation must clear the in-memory TTL rows too, or the
        next resolution inside the TTL window re-persists the stale value."""
        import time as _time
        from agent import model_metadata

        path = tmp_path / "context_lengths.yaml"
        monkeypatch.setattr(model_metadata, "_get_context_cache_path", lambda: path)

        now = _time.monotonic()
        model_metadata._LOCAL_CTX_PROBE_CACHE[("m1", "http://host/v1")] = (999, now)
        model_metadata._LOCAL_CTX_PROBE_CACHE[("ollama_show", "m1", "http://host/v1")] = (999, now)

        model_metadata._invalidate_cached_context_length("m1", "http://host/v1")

        assert ("m1", "http://host/v1") not in model_metadata._LOCAL_CTX_PROBE_CACHE
        assert ("ollama_show", "m1", "http://host/v1") not in model_metadata._LOCAL_CTX_PROBE_CACHE
