"""Tests for _query_local_context_length and the local server fallback in
get_model_context_length.

All tests use synthetic inputs — no filesystem or live server required.
"""

import sys
import os
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


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

    def test_ollama_model_info_context_length(self):
        """Reads context length from model_info dict in /api/show response."""
        from agent.model_metadata import _query_local_context_length

        show_resp = self._make_resp(200, {
            "model_info": {"llama.context_length": 131072}
        })
        models_resp = self._make_resp(404, {})

        client_mock = MagicMock()
        client_mock.__enter__ = lambda s: client_mock
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.post.return_value = show_resp
        client_mock.get.return_value = models_resp

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"), \
             patch("httpx.Client", return_value=client_mock):
            result = _query_local_context_length("omnicoder-9b", "http://localhost:11434/v1")

        assert result == 131072

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
        """Returns None when model is not in the /v1/models list."""
        from agent.model_metadata import _query_local_context_length

        detail_resp = self._make_resp(404, {})
        list_resp = self._make_resp(200, {
            "data": [{"id": "other-model", "max_model_len": 4096}]
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

    def test_local_endpoint_unknown_model_queries_server(self):
        """Unknown model on local endpoint gets ctx from server, not 2M default."""
        from agent.model_metadata import get_model_context_length

        with patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value={}), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}), \
             patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata._query_local_context_length", return_value=131072), \
             patch("agent.model_metadata.save_context_length") as mock_save:
            result = get_model_context_length("omnicoder-9b", "http://localhost:11434/v1")

        assert result == 131072

    def test_local_endpoint_unknown_model_result_is_cached(self):
        """Context length returned from local server is persisted to cache."""
        from agent.model_metadata import get_model_context_length

        with patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value={}), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}), \
             patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata._query_local_context_length", return_value=131072), \
             patch("agent.model_metadata.save_context_length") as mock_save:
            get_model_context_length("omnicoder-9b", "http://localhost:11434/v1")

        mock_save.assert_called_once_with("omnicoder-9b", "http://localhost:11434/v1", 131072)

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

    def test_non_local_endpoint_does_not_query_local_server(self):
        """For non-local endpoints, _query_local_context_length is not called."""
        from agent.model_metadata import get_model_context_length, CONTEXT_PROBE_TIERS

        with patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value={}), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}), \
             patch("agent.model_metadata.is_local_endpoint", return_value=False), \
             patch("agent.model_metadata._query_local_context_length") as mock_query:
            result = get_model_context_length(
                "unknown-model", "https://some-cloud-api.example.com/v1"
            )

        mock_query.assert_not_called()

    def test_cached_result_skips_local_query(self):
        """Cached context length is returned without querying the local server."""
        from agent.model_metadata import get_model_context_length

        with patch("agent.model_metadata.get_cached_context_length", return_value=65536), \
             patch("agent.model_metadata._query_local_context_length") as mock_query:
            result = get_model_context_length("omnicoder-9b", "http://localhost:11434/v1")

        assert result == 65536
        mock_query.assert_not_called()

    def test_no_base_url_does_not_query_local_server(self):
        """When base_url is empty, local server is not queried."""
        from agent.model_metadata import get_model_context_length

        with patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value={}), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}), \
             patch("agent.model_metadata._query_local_context_length") as mock_query:
            result = get_model_context_length("unknown-xyz-model", "")

        mock_query.assert_not_called()
