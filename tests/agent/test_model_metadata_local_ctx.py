"""Tests for _query_local_context_length and the local server fallback in
get_model_context_length.

All tests use synthetic inputs — no filesystem or live server required.
"""

from unittest.mock import MagicMock, patch

import pytest



@pytest.fixture(autouse=True)
def _clear_local_ctx_probe_cache():
    """Reset the in-process local-probe TTL cache around every test.

    _query_local_context_length memoizes probes per (model, base_url) for a
    short TTL to bound the probe rate on hot paths. In tests that mock httpx
    to return different responses for the same (model, base_url), a stale
    cache entry would leak across cases — clear it before and after each test.
    """
    import agent.model_metadata as _mm

    _mm._LOCAL_CTX_PROBE_CACHE.clear()
    yield
    _mm._LOCAL_CTX_PROBE_CACHE.clear()



# ---------------------------------------------------------------------------
# _query_local_context_length — unit tests with mocked httpx
# ---------------------------------------------------------------------------

class TestQueryLocalContextLengthOllama:
    """_query_local_context_length with server_type == 'ollama'."""

    def _make_resp(self, status_code, body):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = body
        return resp


    def test_ollama_parameters_num_ctx(self):
        """Falls back to num_ctx in parameters string when model_info lacks context_length."""
        from agent.model_metadata import _query_local_context_length

        show_resp = self._make_resp(200, {
            "model_info": {},
            "parameters": "num_ctx 32768\ntemperature 0.7\n"
        })
        models_resp = self._make_resp(404, {})

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = show_resp
        client_mock.get.return_value = models_resp

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length("some-model", "http://localhost:11434/v1")

        assert result == 32768


    def test_ollama_show_404_falls_through(self):
        """When /api/show returns 404, falls through to /v1/models/{model}."""
        from agent.model_metadata import _query_local_context_length

        show_resp = self._make_resp(404, {})
        model_detail_resp = self._make_resp(200, {"max_model_len": 65536})

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = show_resp
        client_mock.get.return_value = model_detail_resp

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length("some-model", "http://localhost:11434/v1")

        assert result == 65536


class TestQueryLocalContextLengthVllm:
    """_query_local_context_length with vLLM-style /v1/models/{model} response."""

    def _make_resp(self, status_code, body):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = body
        return resp

    def test_vllm_max_model_len(self):
        """Reads max_model_len from /v1/models/{model} response."""
        from agent.model_metadata import _query_local_context_length

        detail_resp = self._make_resp(200, {"id": "omnicoder-9b", "max_model_len": 100000})
        list_resp = self._make_resp(404, {})

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = self._make_resp(404, {})
        client_mock.get.return_value = detail_resp

        with patch("agent.model_metadata.detect_local_server_type", return_value="vllm"), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length("omnicoder-9b", "http://localhost:8000/v1")

        assert result == 100000

    def test_vllm_context_length_key(self):
        """Reads context_length from /v1/models/{model} response."""
        from agent.model_metadata import _query_local_context_length

        detail_resp = self._make_resp(200, {"id": "some-model", "context_length": 32768})

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = self._make_resp(404, {})
        client_mock.get.return_value = detail_resp

        with patch("agent.model_metadata.detect_local_server_type", return_value="vllm"), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length("some-model", "http://localhost:8000/v1")

        assert result == 32768

    def test_detail_branch_reads_context_window_not_output_cap(self):
        """A payload carrying BOTH a context window and an output cap must
        resolve to the context window.

        An OpenAI-compatible ``/v1/models/{id}`` passthrough (LiteLLM, an
        Anthropic-compat shim, a cloud proxy) returns ``max_input_tokens`` —
        the context window — alongside ``max_tokens``, the max *output*
        tokens.  Reading ``max_tokens`` collapses a 1M-context model to its
        128K output cap and drives premature auto-compaction.

        Contract asserted: when a describe payload contains both classes of
        key, the resolver returns the ``_CONTEXT_LENGTH_KEYS`` value, never
        the ``_MAX_COMPLETION_KEYS`` one.
        """
        from agent.model_metadata import _query_local_context_length

        detail_resp = self._make_resp(200, {
            "type": "model",
            "id": "some-model",
            "max_input_tokens": 1000000,   # context window
            "max_tokens": 128000,          # max OUTPUT tokens — not a window
        })

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = self._make_resp(404, {})
        client_mock.get.return_value = detail_resp

        with patch("agent.model_metadata.detect_local_server_type", return_value="vllm"), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length("some-model", "http://localhost:8000/v1")

        assert result == 1000000, (
            f"must resolve the context window, not the output cap; got {result}"
        )

    def test_list_branch_reads_context_window_not_output_cap(self):
        """Same contract on the sibling ``/v1/models`` LIST branch.

        Both probe branches must share one definition of "context window";
        fixing only the detail branch would leave the identical bug reachable
        whenever the per-model describe endpoint 404s.
        """
        from agent.model_metadata import _query_local_context_length

        detail_miss = self._make_resp(404, {})
        list_resp = self._make_resp(200, {"data": [
            {"id": "some-model", "max_input_tokens": 1000000, "max_tokens": 128000},
        ]})

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = self._make_resp(404, {})
        # first GET is /v1/models/{model} (miss), second is /v1/models (list)
        client_mock.get.side_effect = [detail_miss, list_resp]

        with patch("agent.model_metadata.detect_local_server_type", return_value="vllm"), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length("some-model", "http://localhost:8000/v1")

        assert result == 1000000, (
            f"list branch must resolve the context window, not the output cap; got {result}"
        )

    def test_probe_agrees_with_the_module_key_vocabulary(self):
        """Invariant: the probe's notion of a context window is the module's.

        ``_CONTEXT_LENGTH_KEYS`` / ``_MAX_COMPLETION_KEYS`` are the single
        source of truth for this distinction.  Asserting the relation (rather
        than a frozen key list) keeps the guard correct as the vocabulary
        grows, and fails if a probe branch ever re-hardcodes its own keys.
        """
        from agent import model_metadata as mm

        assert "max_tokens" in mm._MAX_COMPLETION_KEYS
        assert "max_tokens" not in mm._CONTEXT_LENGTH_KEYS
        # No key may be classified as both a window and an output cap.
        assert not (set(mm._CONTEXT_LENGTH_KEYS) & set(mm._MAX_COMPLETION_KEYS))

        # Every context key the module recognises is honoured by the flat
        # reader the probe branches use, and no completion key ever is.
        for key in mm._CONTEXT_LENGTH_KEYS:
            assert mm._extract_flat_context_length({key: 123456}) == 123456, key
        for key in mm._MAX_COMPLETION_KEYS:
            assert mm._extract_flat_context_length({key: 123456}) is None, key


class TestQueryLocalContextLengthModelsList:
    """_query_local_context_length: falls back to /v1/models list."""

    def _make_resp(self, status_code, body):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = body
        return resp

    def test_models_list_max_model_len(self):
        """Finds context length for model in /v1/models list."""
        from agent.model_metadata import _query_local_context_length

        detail_resp = self._make_resp(404, {})
        list_resp = self._make_resp(200, {
            "data": [
                {"id": "other-model", "max_model_len": 4096},
                {"id": "omnicoder-9b", "max_model_len": 131072},
            ]
        })

        call_count = [0]
        def side_effect(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return detail_resp  # /v1/models/omnicoder-9b
            return list_resp  # /v1/models

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = self._make_resp(404, {})
        client_mock.get.side_effect = side_effect

        with patch("agent.model_metadata.detect_local_server_type", return_value=None), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length("omnicoder-9b", "http://localhost:1234")

        assert result == 131072

    def test_models_list_model_not_found_returns_none(self):
        """Returns None when the model is absent from a multi-model /v1/models
        list. (Single-model servers are accepted even when the configured name
        doesn't match the reported id — see the llama.cpp tests below.)"""
        from agent.model_metadata import _query_local_context_length

        detail_resp = self._make_resp(404, {})
        list_resp = self._make_resp(200, {
            "data": [
                {"id": "other-model", "max_model_len": 4096},
                {"id": "yet-another-model", "max_model_len": 8192},
            ]
        })

        call_count = [0]
        def side_effect(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return detail_resp
            return list_resp

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = self._make_resp(404, {})
        client_mock.get.side_effect = side_effect

        with patch("agent.model_metadata.detect_local_server_type", return_value=None), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length("omnicoder-9b", "http://localhost:1234")

        assert result is None

    def test_models_list_llamacpp_meta_n_ctx_sole_model(self):
        """llama.cpp nests the runtime context under meta.n_ctx and serves a
        single model whose id (a GGUF path) doesn't match the configured name.

        The sole model should be accepted and meta.n_ctx read, instead of
        returning None and falling back to a family default (e.g. qwen=131072).
        """
        from agent.model_metadata import _query_local_context_length

        detail_resp = self._make_resp(404, {})
        list_resp = self._make_resp(200, {
            "data": [
                {
                    "id": "/app/models/qwen3.6-35b.gguf",
                    "meta": {"n_ctx": 256000, "n_ctx_train": 262144},
                }
            ]
        })

        call_count = [0]
        def side_effect(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return detail_resp  # /v1/models/{model}
            return list_resp  # /v1/models

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = self._make_resp(404, {})
        client_mock.get.side_effect = side_effect

        with patch("agent.model_metadata.detect_local_server_type", return_value=None), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length("qwen3.6-35b", "http://localhost:8080")

        assert result == 256000

    def test_models_list_llamacpp_prefers_runtime_n_ctx_over_train(self):
        """Runtime n_ctx (256000) is preferred over n_ctx_train (262144),
        since the server can only actually serve the runtime value."""
        from agent.model_metadata import _query_local_context_length

        detail_resp = self._make_resp(404, {})
        list_resp = self._make_resp(200, {
            "data": [
                {"id": "/app/models/m.gguf", "meta": {"n_ctx": 256000, "n_ctx_train": 262144}}
            ]
        })

        call_count = [0]
        def side_effect(url, **kwargs):
            call_count[0] += 1
            return detail_resp if call_count[0] == 1 else list_resp

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = self._make_resp(404, {})
        client_mock.get.side_effect = side_effect

        with patch("agent.model_metadata.detect_local_server_type", return_value=None), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length("m", "http://localhost:8080")

        assert result == 256000


class TestContextLengthFromModelPayload:
    """Anthropic / Anthropic-proxy model objects expose max_input_tokens
    (context window) and max_tokens (max OUTPUT). The local probe must not
    treat max_tokens as the context window."""

    def test_prefers_max_input_tokens_over_max_tokens(self):
        from agent.model_metadata import _context_length_from_model_payload

        # Real Anthropic /v1/models shape for claude-fable-5
        payload = {
            "type": "model",
            "id": "claude-fable-5",
            "max_input_tokens": 1_000_000,
            "max_tokens": 128_000,  # output cap, NOT context
        }
        assert _context_length_from_model_payload(payload) == 1_000_000

    def test_prefers_max_model_len_over_max_tokens(self):
        from agent.model_metadata import _context_length_from_model_payload

        payload = {"id": "local-model", "max_model_len": 131072, "max_tokens": 4096}
        assert _context_length_from_model_payload(payload) == 131072

    def test_falls_back_to_max_tokens_when_no_input_window_field(self):
        from agent.model_metadata import _context_length_from_model_payload

        # Some OpenAI-compat servers only expose max_tokens for the window.
        payload = {"id": "odd-server", "max_tokens": 65536}
        assert _context_length_from_model_payload(payload) == 65536

    def test_returns_none_for_empty_payload(self):
        from agent.model_metadata import _context_length_from_model_payload

        assert _context_length_from_model_payload({}) is None
        assert _context_length_from_model_payload(None) is None  # type: ignore[arg-type]


class TestQueryLocalContextLengthAnthropicProxy:
    """Local Anthropic-compatible reverse proxies (e.g. 127.0.0.1:47821)
    return Anthropic-shaped /v1/models entries. The probe must read
    max_input_tokens, not max_tokens."""

    def _make_resp(self, status_code, body):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = body
        return resp

    def test_models_list_prefers_max_input_tokens(self):
        from agent.model_metadata import _query_local_context_length

        detail_resp = self._make_resp(404, {})
        list_resp = self._make_resp(200, {
            "data": [
                {
                    "type": "model",
                    "id": "claude-fable-5",
                    "display_name": "Claude Fable 5",
                    "max_input_tokens": 1_000_000,
                    "max_tokens": 128_000,
                },
                {
                    "type": "model",
                    "id": "claude-haiku-4-5-20251001",
                    "max_input_tokens": 200_000,
                    "max_tokens": 64_000,
                },
            ]
        })

        call_count = [0]

        def side_effect(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return detail_resp  # /v1/models/claude-fable-5
            return list_resp  # /v1/models

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = self._make_resp(404, {})
        client_mock.get.side_effect = side_effect

        with patch("agent.model_metadata.detect_local_server_type", return_value=None), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length(
                "claude-fable-5", "http://127.0.0.1:47821"
            )

        assert result == 1_000_000, (
            f"Expected max_input_tokens (1M), got {result}. "
            "If Hermes uses Anthropic max_tokens (128k), compression fires ~8x early."
        )

    def test_model_detail_prefers_max_input_tokens(self):
        from agent.model_metadata import _query_local_context_length

        detail_resp = self._make_resp(200, {
            "type": "model",
            "id": "claude-fable-5",
            "max_input_tokens": 1_000_000,
            "max_tokens": 128_000,
        })

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = self._make_resp(404, {})
        client_mock.get.return_value = detail_resp

        with patch("agent.model_metadata.detect_local_server_type", return_value=None), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length(
                "claude-fable-5", "http://127.0.0.1:47821/v1"
            )

        assert result == 1_000_000


class TestQueryLocalContextLengthLmStudio:
    """_query_local_context_length with LM Studio native /api/v1/models response."""

    def _make_resp(self, status_code, body):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = body
        return resp

    def _make_client(self, native_resp, detail_resp, list_resp):
        """Build a mock httpx.Client with sequenced GET responses."""
        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = self._make_resp(404, {})

        responses = [native_resp, detail_resp, list_resp]
        call_idx = [0]

        def get_side_effect(url, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx < len(responses):
                return responses[idx]
            return self._make_resp(404, {})

        client_mock.get.side_effect = get_side_effect
        return client_mock

    def test_lmstudio_exact_key_match(self):
        """Resolves loaded ctx when key matches exactly."""
        from agent.model_metadata import _query_local_context_length

        native_resp = self._make_resp(200, {
            "models": [
                {"key": "nvidia/nvidia-nemotron-super-49b-v1",
                 "id": "nvidia/nvidia-nemotron-super-49b-v1",
                 "max_context_length": 1_048_576,
                 "loaded_instances": [{"config": {"context_length": 131072}}]},
            ]
        })
        client_mock = self._make_client(
            native_resp,
            self._make_resp(404, {}),
            self._make_resp(404, {}),
        )

        with patch("agent.model_metadata.detect_local_server_type", return_value="lm-studio"), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length(
                "nvidia/nvidia-nemotron-super-49b-v1", "http://192.168.1.22:1234/v1"
            )

        assert result == 131072





    def test_lmstudio_native_api_base_url_is_not_doubled(self):
        from agent.model_metadata import _query_local_context_length

        native_resp = self._make_resp(200, {
            "models": [
                {
                    "key": "publisher/model-a",
                    "id": "publisher/model-a",
                    "loaded_instances": [{"config": {"context_length": 32768}}],
                },
            ]
        })
        client_mock = self._make_client(
            native_resp,
            self._make_resp(404, {}),
            self._make_resp(404, {}),
        )

        with patch("agent.model_metadata.detect_local_server_type", return_value="lm-studio"), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length("publisher/model-a", "http://localhost:1234/api/v1")

        assert result == 32768
        assert client_mock.get.call_args_list[0].args[0] == "http://127.0.0.1:1234/api/v1/models"


class TestDetectLocalServerTypeAuth:
    def test_passes_bearer_token_to_probe_requests(self):
        from agent.model_metadata import detect_local_server_type

        resp = MagicMock()
        resp.status_code = 200

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.get.return_value = resp

        with patch("httpx.Client", return_value=client_mock) as mock_client:
            result = detect_local_server_type("http://localhost:1234/v1", api_key="lm-token")

        assert result == "lm-studio"
        assert mock_client.call_args.kwargs["headers"] == {
            "Authorization": "Bearer lm-token"
        }

    def test_native_api_base_url_is_not_doubled(self):
        from agent.model_metadata import detect_local_server_type

        resp = MagicMock()
        resp.status_code = 200

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.get.return_value = resp

        result = None
        with patch("httpx.Client", return_value=client_mock):
            result = detect_local_server_type("http://localhost:1234/api/v1")

        assert result == "lm-studio"
        assert client_mock.get.call_args_list[0].args[0] == "http://127.0.0.1:1234/api/v1/models"


class TestDetectLocalServerTypeLocalhostIPv4:
    """detect_local_server_type should resolve localhost to 127.0.0.1."""

    def test_localhost_resolved_to_ipv4(self):
        """Probes should use 127.0.0.1, not localhost, to avoid IPv6 timeout."""
        from agent.model_metadata import detect_local_server_type

        resp = MagicMock()
        resp.status_code = 200

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.get.return_value = resp

        with patch("httpx.Client", return_value=client_mock):
            detect_local_server_type("http://localhost:8317/v1")

        for call in client_mock.get.call_args_list:
            url = call[0][0]
            assert "localhost" not in url, f"Probe URL still uses localhost: {url}"
            assert "127.0.0.1" in url

    def test_non_localhost_urls_unchanged(self):
        """Non-localhost URLs should not be modified."""
        from agent.model_metadata import detect_local_server_type

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        resp = MagicMock()
        resp.status_code = 404
        client_mock.get.return_value = resp

        with patch("httpx.Client", return_value=client_mock):
            detect_local_server_type("http://192.168.1.100:8080")

        for call in client_mock.get.call_args_list:
            url = call[0][0]
            assert "192.168.1.100" in url



class TestFetchEndpointModelMetadataLmStudio:
    """fetch_endpoint_model_metadata should use LM Studio's native models endpoint."""

    def _make_resp(self, body):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = body
        return resp

    def test_uses_native_models_endpoint_only(self):
        from agent.model_metadata import fetch_endpoint_model_metadata

        native_resp = self._make_resp(
            {
                "models": [
                    {
                        "key": "lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q8_0.gguf",
                        "id": "lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q8_0.gguf",
                        "max_context_length": 1_048_576,
                        "loaded_instances": [
                            {"config": {"context_length": 131072}}
                        ],
                    }
                ]
            }
        )

        with patch("agent.model_metadata.detect_local_server_type", return_value="lm-studio"), \
             patch("agent.model_metadata.requests.get", return_value=native_resp) as mock_get:
            result = fetch_endpoint_model_metadata(
                "http://localhost:1234/v1",
                api_key="lm-token",
                force_refresh=True,
            )

        assert mock_get.call_count == 1
        assert mock_get.call_args[0][0] == "http://localhost:1234/api/v1/models"
        assert mock_get.call_args.kwargs["headers"] == {
            "Authorization": "Bearer lm-token"
        }
        assert result["lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q8_0.gguf"]["context_length"] == 131072
        assert result["Qwen3.5-27B-GGUF/Qwen3.5-27B-Q8_0.gguf"]["context_length"] == 131072

    def test_native_api_base_url_is_not_doubled(self):
        from agent.model_metadata import fetch_endpoint_model_metadata

        native_resp = self._make_resp(
            {
                "models": [
                    {
                        "key": "publisher/model-a",
                        "id": "publisher/model-a",
                        "loaded_instances": [
                            {"config": {"context_length": 65536}}
                        ],
                    }
                ]
            }
        )

        with patch("agent.model_metadata.detect_local_server_type", return_value="lm-studio"), \
             patch("agent.model_metadata.requests.get", return_value=native_resp) as mock_get:
            result = fetch_endpoint_model_metadata(
                "http://localhost:1234/api/v1",
                force_refresh=True,
            )

        assert mock_get.call_args[0][0] == "http://localhost:1234/api/v1/models"
        assert result["publisher/model-a"]["context_length"] == 65536


class TestQueryLocalContextLengthNetworkError:
    """_query_local_context_length handles network failures gracefully."""

    def test_connection_error_returns_none(self):
        """Returns None when the server is unreachable."""
        from agent.model_metadata import _query_local_context_length

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.side_effect = Exception("Connection refused")
        client_mock.get.side_effect = Exception("Connection refused")

        with patch("agent.model_metadata.detect_local_server_type", return_value=None), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length("omnicoder-9b", "http://localhost:11434/v1")

        assert result is None


# ---------------------------------------------------------------------------
# get_model_context_length — integration-style tests with mocked helpers
# ---------------------------------------------------------------------------

class TestGetModelContextLengthLocalFallback:
    """get_model_context_length uses local server query before falling back to 2M."""



    def test_local_endpoint_stale_cache_reconciled_from_live_probe(self):
        """Stale disk cache must yield to a live local max_model_len probe."""
        from agent.model_metadata import get_model_context_length

        model = "NousResearch/Hermes-3-Llama-3.1-70B"
        base = "http://192.168.1.50:8000/v1"

        with patch("agent.model_metadata.get_cached_context_length", return_value=131072), \
             patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value={}), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}), \
             patch("agent.model_metadata._query_ollama_api_show", return_value=None), \
             patch("agent.model_metadata._is_custom_endpoint", return_value=False), \
             patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata._query_local_context_length", return_value=32768), \
             patch("agent.model_metadata._invalidate_cached_context_length") as mock_invalidate, \
             patch("agent.model_metadata.save_context_length") as mock_save:
            result = get_model_context_length(model, base, provider="custom")

        assert result == 32768
        mock_invalidate.assert_called_once_with(model, base)
        mock_save.assert_not_called()



    def test_local_endpoint_server_returns_none_falls_back_to_2m(self):
        """When local server returns None, still falls back to 2M probe tier."""
        from agent.model_metadata import get_model_context_length, CONTEXT_PROBE_TIERS

        with patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value={}), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}), \
             patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata._query_local_context_length", return_value=None):
            result = get_model_context_length("omnicoder-9b", "http://localhost:11434/v1")

        assert result == CONTEXT_PROBE_TIERS[0]


    def test_cached_result_skips_local_query(self):
        """Cached context length is returned without querying the local server."""
        from agent.model_metadata import get_model_context_length

        with patch("agent.model_metadata.get_cached_context_length", return_value=65536), \
             patch("agent.model_metadata.is_local_endpoint", return_value=False), \
             patch("agent.model_metadata._query_local_context_length") as mock_query:
            result = get_model_context_length(
                "omnicoder-9b", "https://api.example.com/v1"
            )

        assert result == 65536
        mock_query.assert_not_called()



class TestLocalContextProbeTTLCache:
    """The in-process TTL cache collapses back-to-back probes for the same
    (model, base_url) into one network round-trip (bounds probe rate on hot
    paths like banner + /model switch + compressor update within one startup),
    while a different key still probes."""

    def _make_resp(self, status_code, body):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = body
        return resp

    def test_second_call_within_ttl_does_not_reprobe(self):
        from agent.model_metadata import _query_local_context_length

        show_resp = self._make_resp(200, {"model_info": {"llama.context_length": 32768}})
        models_resp = self._make_resp(404, {})
        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = show_resp
        client_mock.get.return_value = models_resp

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama") as detect, \
             patch("httpx.Client", return_value=client_mock):
            first = _query_local_context_length("m", "http://localhost:11434/v1")
            second = _query_local_context_length("m", "http://localhost:11434/v1")

        assert first == 32768
        assert second == 32768
        # Only the first call hits the network; the second is served from cache.
        assert detect.call_count == 1

    def test_different_key_still_probes(self):
        from agent.model_metadata import _query_local_context_length

        show_resp = self._make_resp(200, {"model_info": {"llama.context_length": 32768}})
        models_resp = self._make_resp(404, {})
        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = show_resp
        client_mock.get.return_value = models_resp

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama") as detect, \
             patch("httpx.Client", return_value=client_mock):
            _query_local_context_length("m1", "http://localhost:11434/v1")
            _query_local_context_length("m2", "http://localhost:11434/v1")

        assert detect.call_count == 2


    def test_none_result_not_cached(self):
        """A failed probe (None) must NOT be memoized — a retry within the TTL
        window must re-probe so a server that comes up mid-startup is caught."""
        from agent.model_metadata import _query_local_context_length

        # First probe: server unreachable -> detect returns None, all queries miss -> None.
        fail_resp = self._make_resp(404, {})
        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = fail_resp
        client_mock.get.return_value = fail_resp

        with patch("agent.model_metadata.detect_local_server_type", return_value=None) as detect, \
             patch("httpx.Client", return_value=client_mock):
            first = _query_local_context_length("m", "http://localhost:11434/v1")
            # Retry within TTL must re-probe (None was not cached).
            second = _query_local_context_length("m", "http://localhost:11434/v1")

        assert first is None
        assert second is None
        assert detect.call_count == 2, "None result was wrongly cached; retry did not re-probe"
