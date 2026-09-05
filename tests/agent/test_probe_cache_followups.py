"""Tests for probe-cache follow-ups on the #29988/#37595/#50572 salvage.

Covers:
- _query_ollama_api_show TTL caching (positive-only, namespaced key)
- persistent context-cache key normalization (trailing-slash dedup)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest



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
        # Age the disk L2 entry too. Its TTL (300s) is much shorter than the
        # in-proc TTL (1h), so in real time-flow it always expires first —
        # this test compresses both expiries into one instant.
        import json as _json
        _disk = model_metadata._local_probe_disk_cache_path()
        if _disk.exists():
            _data = _json.loads(_disk.read_text(encoding="utf-8"))
            for _entry in _data.values():
                if isinstance(_entry, dict):
                    _entry["ts"] = (
                        _time.time() - model_metadata._LOCAL_PROBE_DISK_TTL_SECONDS - 1
                    )
            _disk.write_text(_json.dumps(_data), encoding="utf-8")

        lmstudio_resp = MagicMock()
        lmstudio_resp.status_code = 200
        # LM Studio's native listing shape (entries keyed under "models" with
        # LM Studio-specific fields). detect_local_server_type discriminates on
        # this shape, not on a bare 200, so the fixture has to be a payload a
        # real LM Studio would send.
        lmstudio_resp.json.return_value = {
            "models": [
                {
                    "key": "qwen/qwen3-4b",
                    "type": "llm",
                    "state": "loaded",
                    "loaded_instances": [{"config": {"context_length": 8192}}],
                }
            ]
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


class TestLocalhostIPv4SiblingSites:
    """#37595 widened: every probe helper rewrites localhost→127.0.0.1,
    not just detect_local_server_type."""


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

    def test_fetch_endpoint_model_metadata_generic_probe_uses_ipv4(self):
        """The generic (non-LM-Studio) /models fetch loop must also rewrite
        localhost->127.0.0.1 before probing, like the LM Studio branch above."""
        from agent import model_metadata
        from agent.model_metadata import fetch_endpoint_model_metadata

        model_metadata._endpoint_model_metadata_cache.clear()
        model_metadata._endpoint_model_metadata_cache_time.clear()

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"data": []}

        with patch("agent.model_metadata.detect_local_server_type", return_value=None), \
             patch("agent.model_metadata.requests.get", return_value=resp) as mock_get:
            fetch_endpoint_model_metadata("http://localhost:8000/v1")

        assert mock_get.call_args[0][0].startswith("http://127.0.0.1:8000")

    def test_fetch_endpoint_model_metadata_llamacpp_props_followup_uses_ipv4(self):
        """The llama.cpp /props context-length follow-up must also rewrite
        localhost->127.0.0.1 before probing, not just the initial /models call."""
        from agent import model_metadata
        from agent.model_metadata import fetch_endpoint_model_metadata

        model_metadata._endpoint_model_metadata_cache.clear()
        model_metadata._endpoint_model_metadata_cache_time.clear()

        models_resp = MagicMock()
        models_resp.status_code = 200
        models_resp.raise_for_status = MagicMock()
        models_resp.json.return_value = {
            "data": [{"id": "llama-3-8b", "owned_by": "llamacpp"}],
        }

        props_resp = MagicMock()
        props_resp.ok = True
        props_resp.json.return_value = {
            "default_generation_settings": {"n_ctx": 32768},
            "model_alias": "llama-3-8b",
        }

        with patch("agent.model_metadata.detect_local_server_type", return_value=None), \
             patch(
                 "agent.model_metadata.requests.get",
                 side_effect=[models_resp, props_resp],
             ) as mock_get:
            result = fetch_endpoint_model_metadata("http://localhost:8000/v1")

        assert mock_get.call_count == 2
        props_call_url = mock_get.call_args_list[1][0][0]
        assert props_call_url.startswith("http://127.0.0.1:8000")
        assert result["llama-3-8b"]["context_length"] == 32768



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


class TestDetectServerTypeNegativeCaching:
    """A failed detect_local_server_type verdict is cached briefly (#89863).

    Previously only positive verdicts were memoized, so a remote endpoint
    that answered the whole waterfall with 401s (no recognizable server
    type) was re-probed — 5 requests — on every image-bearing turn.
    """

    @staticmethod
    def _client_all_401():
        client = MagicMock()
        client.__enter__ = lambda s: client
        client.__exit__ = MagicMock(return_value=False)
        resp = MagicMock()
        resp.status_code = 401
        client.get.return_value = resp
        return client

    def test_negative_verdict_is_cached_in_memory(self):
        from agent.model_metadata import detect_local_server_type
        from agent import model_metadata

        client = self._client_all_401()
        with patch("httpx.Client", return_value=client):
            assert detect_local_server_type("http://remote:8080/v1") is None
            assert detect_local_server_type("http://remote:8080/v1") is None

        # Second call served from the in-memory negative entry: the
        # waterfall ran exactly once (5 GETs), not twice.
        assert client.get.call_count == 5
        assert "http://remote:8080" in model_metadata._endpoint_probe_path_cache

    def test_negative_verdict_not_written_to_disk(self):
        from agent.model_metadata import detect_local_server_type
        from agent import model_metadata

        with patch("httpx.Client", return_value=self._client_all_401()), patch.object(
            model_metadata, "_local_probe_disk_put"
        ) as disk_put:
            assert detect_local_server_type("http://remote2:8080/v1") is None
        disk_put.assert_not_called()

    def test_negative_verdict_expires_quickly(self):
        """The short failure TTL keeps a transient failure recoverable."""
        import time as _time
        from agent.model_metadata import detect_local_server_type
        from agent import model_metadata

        client = self._client_all_401()
        with patch("httpx.Client", return_value=client):
            assert detect_local_server_type("http://remote3:8080/v1") is None
            # Age the entry past the failure TTL.
            model_metadata._endpoint_probe_path_cache["http://remote3:8080"] = (
                None,
                _time.monotonic()
                - model_metadata._ENDPOINT_PROBE_FAILURE_TTL_SECONDS
                - 1,
            )
            assert detect_local_server_type("http://remote3:8080/v1") is None

        assert client.get.call_count == 10  # waterfall re-ran after expiry


class TestLMStudioDetectionRequiresNativePayload:
    """A bare 200 on /api/v1/models is not evidence of LM Studio.

    Other OpenAI-compatible local servers serve that path too (e.g. a loopback
    proxy exposing it for model-name validation). Misdetecting one sends the
    caller into the LM Studio metadata parser, which reads ``payload["models"]``
    — a key the OpenAI listing envelope does not have — so ALL advertised
    metadata is discarded and the caller falls back to a probe-tier default.
    """

    def _detect(self, payload):
        from agent import model_metadata
        from agent.model_metadata import detect_local_server_type

        # Each payload below represents a different server. Do not let the
        # detector's intentional per-endpoint cache turn this shape test into
        # an order-dependent assertion about the first example it happened to
        # run.
        model_metadata._endpoint_probe_path_cache.clear()

        hit = MagicMock()
        hit.status_code = 200
        hit.json.return_value = payload
        hit.text = ""
        miss = MagicMock()
        miss.status_code = 404
        miss.text = ""
        miss.json.return_value = {}

        client = MagicMock()
        client.__enter__ = lambda s: client
        client.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = (
            lambda url, *a, **k: hit if url.endswith("/api/v1/models") else miss
        )
        with (
            patch("httpx.Client", return_value=client),
            patch("agent.model_metadata._local_probe_disk_get", return_value=None),
            patch("agent.model_metadata._local_probe_disk_put"),
        ):
            return detect_local_server_type("http://127.0.0.1:8080/v1")

    def test_openai_listing_envelope_is_not_lm_studio(self):
        """The standard OpenAI listing shape must never classify as LM Studio,
        whatever its entries carry."""
        assert self._detect(
            {"object": "list", "data": [{"id": "some-model", "object": "model"}]}
        ) != "lm-studio"
        # Even when the entries carry rich metadata, the envelope decides.
        assert self._detect(
            {
                "object": "list",
                "data": [{"id": "m", "object": "model", "context_length": 1_000_000}],
            }
        ) != "lm-studio"

    def test_lm_studio_native_payload_is_still_detected(self):
        """The feature must survive the fix: LM Studio's own shapes still
        classify, including an idle server with nothing loaded."""
        assert self._detect(
            {
                "models": [
                    {
                        "key": "qwen/qwen3-4b",
                        "loaded_instances": [{"config": {"context_length": 8192}}],
                    }
                ]
            }
        ) == "lm-studio"
        # Idle LM Studio: running, no model loaded.
        assert self._detect({"models": []}) == "lm-studio"
        # A data-keyed response remains OpenAI-compatible even when an entry
        # happens to carry fields also seen in LM Studio's native response.
        assert self._detect(
            {"data": [{"id": "m", "type": "llm", "state": "loaded"}]}
        ) != "lm-studio"

    def test_unparseable_or_ambiguous_payloads_fail_closed(self):
        """Fail closed, never open: an unrecognised body must not classify as
        LM Studio, so the caller continues to the OpenAI-compatible path."""
        assert self._detect("not-a-dict") != "lm-studio"
        assert self._detect({}) != "lm-studio"
        # An empty `data` list is indistinguishable from an idle OpenAI server.
        assert self._detect({"data": []}) != "lm-studio"
        # The prior predicate accepted these rich data-keyed variants even
        # though both LM Studio consumers only read payload["models"].
        assert (
            self._detect({"data": [{"id": "m", "key": "publisher/m"}]})
            != "lm-studio"
        )

    def test_detection_agrees_with_what_the_lm_studio_parser_can_read(self):
        """Contract between the detector and its consumer: this module's LM
        Studio parsers read ``payload["models"]``. Classifying a payload as
        LM Studio when that key is absent guarantees the parser yields nothing,
        which is exactly the bug. So a NON-empty payload may only be called
        LM Studio if the parser could actually extract a model id from it."""
        from agent.model_metadata import _is_lmstudio_models_payload

        def parser_can_read(payload):
            # Mirrors fetch_endpoint_model_metadata's LM Studio branch.
            if not isinstance(payload, dict):
                return False
            return any(
                isinstance(m, dict) and (m.get("key") or m.get("id"))
                for m in payload.get("models", []) or []
            )

        payloads = [
            {"object": "list", "data": [{"id": "m", "object": "model"}]},
            {"data": [{"id": "m", "object": "model", "context_length": 1_000_000}]},
            {"models": [{"key": "qwen/qwen3-4b", "loaded_instances": []}]},
            {"models": [{"id": "m", "max_context_length": 4096}]},
            {"data": [{"id": "m", "type": "llm", "state": "loaded"}]},
            {},
            {"models": []},
        ]
        for payload in payloads:
            resp = MagicMock()
            resp.json.return_value = payload
            classified = _is_lmstudio_models_payload(resp)
            if classified and payload.get("models"):
                assert parser_can_read(payload), (
                    f"classified as LM Studio but the LM Studio parser reads "
                    f"nothing from it: {payload}"
                )


class TestOpenAICompatProxyMetadataSurvivesDetection:
    """End-to-end: an OpenAI-compatible server's advertised context window must
    reach the caller. The LM Studio branch of fetch_endpoint_model_metadata
    returns early, so a misdetection discards the payload outright rather than
    falling through to the working OpenAI parser."""

    def test_advertised_context_window_is_not_discarded(self):
        from agent import model_metadata

        payload = {
            "object": "list",
            "data": [
                {
                    "id": "big-ctx-model",
                    "object": "model",
                    "context_length": 1_000_000,
                    # Gateway extension fields may reuse pricing aliases with
                    # non-scalar values. They must not abort the whole metadata
                    # parse before the real pricing object is reached.
                    "modalities": {"input": [], "output": []},
                    "pricing": {"input": 0, "output": 0},
                }
            ],
        }
        model_metadata._endpoint_model_metadata_cache.clear()
        model_metadata._endpoint_model_metadata_cache_time.clear()

        probe = MagicMock()
        probe.status_code = 200
        probe.json.return_value = payload
        probe.text = ""
        client = MagicMock()
        client.__enter__ = lambda s: client
        client.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = lambda url, *a, **k: probe

        def _requests_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = payload
            resp.raise_for_status = lambda: None
            return resp

        with patch("httpx.Client", return_value=client), patch(
            "requests.get", side_effect=_requests_get
        ):
            metadata = model_metadata.fetch_endpoint_model_metadata(
                "http://127.0.0.1:8080/v1", force_refresh=True
            )

        entry = metadata.get("big-ctx-model", {})
        assert entry.get("context_length") == 1_000_000
        assert entry.get("pricing") == {"prompt": 0, "completion": 0}
