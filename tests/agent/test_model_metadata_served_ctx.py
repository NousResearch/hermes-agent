"""Tests for _query_ollama_served_ctx and the /api/ps cap on /api/show values.

/api/show reports the GGUF training maximum and cannot see
OLLAMA_CONTEXT_LENGTH -- a server env var, absent from both model_info and
parameters. When a server is started with a smaller window than the weights
support, every /api/show-derived context length overstates the real limit, and
requests sized to it come back with finish_reason="length" on the first token.

/api/ps reports what each LOADED model is actually being served at, so it is
the authority whenever it has an answer.

All tests use synthetic inputs -- no filesystem or live server required.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_local_ctx_probe_cache():
    """Reset the in-process probe TTL cache around every test.

    _query_ollama_served_ctx memoizes per (model, server_url) for a short TTL.
    Cases below return different /api/ps bodies for the same pair, so a stale
    entry would leak across them.
    """
    import agent.model_metadata as _mm

    _mm._LOCAL_CTX_PROBE_CACHE.clear()
    yield
    _mm._LOCAL_CTX_PROBE_CACHE.clear()


def _resp(status_code, body):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    return resp


def _ps(*models):
    return {"models": [dict(m) for m in models]}


class TestQueryOllamaServedCtx:
    """_query_ollama_served_ctx against a mocked /api/ps."""

    def test_returns_served_context_for_loaded_model(self):
        from agent.model_metadata import _query_ollama_served_ctx

        client = MagicMock()
        client.get.return_value = _resp(
            200, _ps({"name": "gpt-oss:20b", "context_length": 32768})
        )
        with patch("httpx.Client") as C:
            C.return_value.__enter__.return_value = client
            assert (
                _query_ollama_served_ctx("gpt-oss:20b", "http://x:11434", {})
                == 32768
            )

    def test_returns_none_when_model_not_loaded(self):
        """The decisive fallback: an idle model has no knowable served ctx."""
        from agent.model_metadata import _query_ollama_served_ctx

        client = MagicMock()
        client.get.return_value = _resp(
            200, _ps({"name": "other:7b", "context_length": 8192})
        )
        with patch("httpx.Client") as C:
            C.return_value.__enter__.return_value = client
            assert (
                _query_ollama_served_ctx("gpt-oss:20b", "http://x:11434", {})
                is None
            )

    def test_matches_on_model_field_too(self):
        from agent.model_metadata import _query_ollama_served_ctx

        client = MagicMock()
        client.get.return_value = _resp(
            200, _ps({"model": "gpt-oss:20b", "context_length": 16384})
        )
        with patch("httpx.Client") as C:
            C.return_value.__enter__.return_value = client
            assert (
                _query_ollama_served_ctx("gpt-oss:20b", "http://x:11434", {})
                == 16384
            )

    def test_empty_and_error_responses_are_none(self):
        from agent.model_metadata import _query_ollama_served_ctx

        for body, status in (({}, 200), (_ps(), 200), ({}, 404)):
            import agent.model_metadata as _mm

            _mm._LOCAL_CTX_PROBE_CACHE.clear()
            client = MagicMock()
            client.get.return_value = _resp(status, body)
            with patch("httpx.Client") as C:
                C.return_value.__enter__.return_value = client
                assert (
                    _query_ollama_served_ctx("m", "http://x:11434", {}) is None
                )

    def test_nonsense_small_value_ignored(self):
        """Guards against a 0/absurd context_length clamping everything to junk."""
        from agent.model_metadata import _query_ollama_served_ctx

        client = MagicMock()
        client.get.return_value = _resp(
            200, _ps({"name": "m", "context_length": 8})
        )
        with patch("httpx.Client") as C:
            C.return_value.__enter__.return_value = client
            assert _query_ollama_served_ctx("m", "http://x:11434", {}) is None

    def test_connection_error_is_swallowed(self):
        from agent.model_metadata import _query_ollama_served_ctx

        with patch("httpx.Client", side_effect=OSError("refused")):
            assert _query_ollama_served_ctx("m", "http://x:11434", {}) is None


class TestCapCtxByServed:
    """_cap_ctx_by_served only ever lowers, and only on a real signal."""

    def _patched(self, served):
        return patch(
            "agent.model_metadata._query_ollama_served_ctx", return_value=served
        )

    def test_caps_advertised_max_down_to_served(self):
        """The reported case: advertises 131072, actually serving 32768."""
        from agent.model_metadata import _cap_ctx_by_served

        with self._patched(32768):
            assert _cap_ctx_by_served(131072, "m", "http://x", {}) == 32768

    def test_does_not_raise_when_served_is_larger(self):
        """GGUF max is the hard ceiling -- never exceed it on /api/ps's word."""
        from agent.model_metadata import _cap_ctx_by_served

        with self._patched(131072):
            assert _cap_ctx_by_served(32768, "m", "http://x", {}) == 32768

    def test_unloaded_model_keeps_advertised_value(self):
        from agent.model_metadata import _cap_ctx_by_served

        with self._patched(None):
            assert _cap_ctx_by_served(131072, "m", "http://x", {}) == 131072

    def test_none_and_zero_pass_through(self):
        from agent.model_metadata import _cap_ctx_by_served

        with self._patched(32768):
            assert _cap_ctx_by_served(None, "m", "http://x", {}) is None
            assert _cap_ctx_by_served(0, "m", "http://x", {}) == 0
