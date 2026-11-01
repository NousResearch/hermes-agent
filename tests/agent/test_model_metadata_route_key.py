"""Route-credential forwarding for model-metadata probes (#105379).

A `custom_providers` entry (or a `key_env`-configured custom provider) holds the
credential inference already sends — but the model-metadata probe waterfall
(`detect_local_server_type` → `/api/v1/models` → `/api/tags` → `/v1/props` →
`/props` → `/version`, plus the `/models` metadata fetch and the local
context-length probe) received whatever `api_key` the caller happened to pass,
often `""`. Keyed local servers (llama.cpp `--api-key`, oMLX) answered every
unauthenticated leg with a 401, so detection could never succeed and re-failed
per TTL expiry — the same class #89863 fixed for the Ollama vision probe.

These tests pin the contract: when the caller does not pass a key, the probe
credential is resolved from the route-matching custom-provider entry using the
same precedence the request path uses (inline `api_key` → `key_env` → `key_cmd`
→ credential pool), materialized to a plain string, and forwarded to every
probe leg. A truthy caller-passed key is never overridden; a route with no
matching entry behaves exactly as before (no header).

All tests use synthetic entries and mocked HTTP — no live server required.
"""

from unittest.mock import MagicMock, patch

import pytest


def _client_mock_401():
    """httpx.Client mock whose every GET returns 401 (the bug signature)."""
    resp = MagicMock()
    resp.status_code = 401
    client = MagicMock()
    client.__enter__ = lambda s: client
    client.__exit__ = MagicMock(return_value=False)
    client.get.return_value = resp
    return client


ENTRY_KEY_ENV = {
    "name": "omlx",
    "base_url": "http://omlx.lan:8000/v1",
    "key_env": "OMLX_API_KEY",
}
ENTRY_INLINE = {
    "name": "llamacpp",
    "base_url": "http://10.0.0.4:8080/v1",
    "api_key": "sk-inline-route-key",
}


@pytest.fixture(autouse=True)
def _clear_probe_caches():
    """Reset the in-process probe/metadata caches around every test."""
    import agent.model_metadata as _mm

    _mm._endpoint_probe_path_cache.clear()
    _mm._LOCAL_CTX_PROBE_CACHE.clear()
    _mm._endpoint_model_metadata_cache.clear()
    _mm._endpoint_model_metadata_cache_time.clear()
    yield
    _mm._endpoint_probe_path_cache.clear()
    _mm._LOCAL_CTX_PROBE_CACHE.clear()
    _mm._endpoint_model_metadata_cache.clear()
    _mm._endpoint_model_metadata_cache_time.clear()


class TestServerTypeWaterfallKeyForwarding:
    """detect_local_server_type resolves the route credential when the caller passes none."""

    def test_key_env_entry_key_reaches_waterfall_headers(self, monkeypatch):
        """A key_env custom entry's credential is forwarded as the Bearer header."""
        from agent.model_metadata import detect_local_server_type

        monkeypatch.setenv("OMLX_API_KEY", "sk-omlx-from-env")
        client = _client_mock_401()
        with patch("httpx.Client", return_value=client) as client_cls, \
                patch("hermes_cli.config_providers._entries_for_route",
                      return_value=[dict(ENTRY_KEY_ENV)]):
            detect_local_server_type("http://omlx.lan:8000/v1")
        headers = client_cls.call_args.kwargs.get("headers") or {}
        assert headers.get("Authorization") == "Bearer sk-omlx-from-env"

    def test_inline_api_key_entry_reaches_waterfall_headers(self, monkeypatch):
        """An entry's inline api_key is forwarded without any env var."""
        from agent.model_metadata import detect_local_server_type

        monkeypatch.delenv("OMLX_API_KEY", raising=False)
        client = _client_mock_401()
        with patch("httpx.Client", return_value=client) as client_cls, \
                patch("hermes_cli.config_providers._entries_for_route",
                      return_value=[dict(ENTRY_INLINE)]):
            detect_local_server_type("http://10.0.0.4:8080/v1")
        headers = client_cls.call_args.kwargs.get("headers") or {}
        assert headers.get("Authorization") == "Bearer sk-inline-route-key"

    def test_explicit_caller_key_wins_over_entry_lookup(self, monkeypatch):
        """A truthy caller-passed api_key is never overridden by the entry lookup."""
        from agent.model_metadata import detect_local_server_type

        monkeypatch.setenv("OMLX_API_KEY", "sk-omlx-from-env")
        client = _client_mock_401()
        with patch("httpx.Client", return_value=client) as client_cls, \
                patch("hermes_cli.config_providers._entries_for_route",
                      return_value=[dict(ENTRY_KEY_ENV)]) as entries:
            detect_local_server_type("http://omlx.lan:8000/v1", api_key="sk-explicit")
        headers = client_cls.call_args.kwargs.get("headers") or {}
        assert headers.get("Authorization") == "Bearer sk-explicit"
        entries.assert_not_called()

    def test_no_matching_entry_sends_no_header(self, monkeypatch):
        """A route with no matching custom entry probes exactly as before (no header)."""
        from agent.model_metadata import detect_local_server_type

        monkeypatch.delenv("OMLX_API_KEY", raising=False)
        client = _client_mock_401()
        with patch("httpx.Client", return_value=client) as client_cls, \
                patch("hermes_cli.config_providers._entries_for_route", return_value=[]):
            detect_local_server_type("http://unkeyed.lan:11434/v1")
        headers = client_cls.call_args.kwargs.get("headers") or {}
        assert "Authorization" not in headers


class TestContextLengthChainKeyForwarding:
    """get_model_context_length forwards the resolved route key down the whole chain."""

    def test_custom_endpoint_resolution_forwards_key_to_models_fetch(self, monkeypatch):
        """Step 2 (/models for custom endpoints) receives the entry credential when the
        caller passes none — the oMLX case from the 2026-09-21 comment."""
        import agent.model_metadata as mm

        monkeypatch.setenv("OMLX_API_KEY", "sk-omlx-from-env")
        monkeypatch.setattr(mm, "_is_custom_endpoint", lambda b: True)
        monkeypatch.setattr(mm, "_is_known_provider_base_url", lambda b: False)
        monkeypatch.setattr(mm, "_is_bedrock_context", lambda b, p: False)
        monkeypatch.setattr(mm, "_is_codex_route", lambda p, b, cps: False)
        monkeypatch.setattr(mm, "_config_override_context_length", lambda *a, **k: None)
        monkeypatch.setattr(mm, "_endpoint_scoped_context_length", lambda m, b: None)
        monkeypatch.setattr(mm, "get_cached_context_length", lambda m, b: None)
        seen_keys = []

        def _capture_fetch(base_url, api_key="", force_refresh=False):
            seen_keys.append(api_key)
            return {"qwen-mlx": {"context_length": 250000}}

        monkeypatch.setattr(mm, "fetch_endpoint_model_metadata", _capture_fetch)
        with patch("hermes_cli.config_providers._entries_for_route", return_value=[dict(ENTRY_KEY_ENV)]):
            ctx = mm.get_model_context_length(
                "qwen-mlx", base_url="http://omlx.lan:8000/v1", api_key="",
                provider="custom", custom_providers=[dict(ENTRY_KEY_ENV)])
        assert ctx == 250000
        assert seen_keys == ["sk-omlx-from-env"]

    def test_keyless_probe_failure_no_longer_overstates_window(self, monkeypatch):
        """End-to-end contract from the issue: with the key forwarded, detection
        succeeds via /models and the 256K probe-down fallback is never returned."""
        import agent.model_metadata as mm

        monkeypatch.setenv("OMLX_API_KEY", "sk-omlx-from-env")
        monkeypatch.setattr(mm, "_is_custom_endpoint", lambda b: True)
        monkeypatch.setattr(mm, "_is_known_provider_base_url", lambda b: False)
        monkeypatch.setattr(mm, "_is_bedrock_context", lambda b, p: False)
        monkeypatch.setattr(mm, "_is_codex_route", lambda p, b, cps: False)
        monkeypatch.setattr(mm, "_config_override_context_length", lambda *a, **k: None)
        monkeypatch.setattr(mm, "_endpoint_scoped_context_length", lambda m, b: None)
        monkeypatch.setattr(mm, "get_cached_context_length", lambda m, b: None)
        # /models answers 200 with the real window only when authorized.
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": [{"id": "qwen-mlx", "max_model_len": 250000}]}
        resp.headers = {}
        requests_mock = MagicMock()
        requests_mock.get.return_value = resp
        with patch("agent.model_metadata.requests", requests_mock), \
                patch("hermes_cli.config_providers._entries_for_route", return_value=[dict(ENTRY_KEY_ENV)]):
            ctx = mm.get_model_context_length(
                "qwen-mlx", base_url="http://omlx.lan:8000/v1", api_key="",
                provider="custom", custom_providers=[dict(ENTRY_KEY_ENV)])
        assert ctx == 250000
        sent_headers = requests_mock.get.call_args.kwargs.get("headers") or {}
        assert sent_headers.get("Authorization") == "Bearer sk-omlx-from-env"
        assert ctx != mm.DEFAULT_FALLBACK_CONTEXT


class TestCredentialPrecedence:
    """The probe credential uses the request path's precedence — inline beats env."""

    def test_inline_key_beats_key_env(self, monkeypatch):
        from agent.model_metadata import _resolve_route_probe_api_key

        monkeypatch.setenv("OMLX_API_KEY", "sk-from-env")
        entry = dict(ENTRY_KEY_ENV, api_key="sk-inline-wins")
        monkeypatch.setattr(
            "hermes_cli.config_providers._entries_for_route", lambda *a, **k: [entry])
        key = _resolve_route_probe_api_key(
            "http://omlx.lan:8000/v1", "", custom_providers=[entry])
        assert key == "sk-inline-wins"

    def test_key_cmd_caller_materialized_to_str(self, monkeypatch):
        """A key_cmd entry resolves to a materialized string (never a callable) so
        memo/disk cache keys fingerprint consistently."""
        from agent.model_metadata import _resolve_route_probe_api_key

        def _mint():
            return "sk-from-cmd"

        entry = {"name": "vault", "base_url": "http://vault.lan:8000/v1", "key_cmd": "op read x"}
        monkeypatch.setattr(
            "agent.command_token_source.build_command_token_provider", lambda cmd, name: _mint)
        monkeypatch.setattr(
            "hermes_cli.config_providers._entries_for_route", lambda *a, **k: [entry])
        key = _resolve_route_probe_api_key(
            "http://vault.lan:8000/v1", "", custom_providers=[entry])
        assert key == "sk-from-cmd"
        assert not callable(key)

    def test_blank_result_when_no_entry_and_blank_caller_key(self, monkeypatch):
        from agent.model_metadata import _resolve_route_probe_api_key

        monkeypatch.setattr("hermes_cli.config_providers._entries_for_route", lambda *a, **k: [])
        assert _resolve_route_probe_api_key("http://x.lan:8000/v1", "", custom_providers=None) == ""


class TestDirectMetadataFetchForwarding:
    """fetch_endpoint_model_metadata (direct callers, e.g. usage_pricing) forwards too."""

    def test_usage_pricing_style_call_forwards_entry_key(self, monkeypatch):
        import agent.model_metadata as mm

        monkeypatch.setenv("OMLX_API_KEY", "sk-omlx-from-env")
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": [{"id": "qwen-mlx", "max_model_len": 250000}]}
        resp.headers = {}
        requests_mock = MagicMock()
        requests_mock.get.return_value = resp
        with patch("agent.model_metadata.requests", requests_mock), \
                patch("hermes_cli.config_providers._entries_for_route", return_value=[dict(ENTRY_KEY_ENV)]):
            meta = mm.fetch_endpoint_model_metadata("http://omlx.lan:8000/v1", api_key="")
        assert meta.get("qwen-mlx", {}).get("context_length") == 250000
        sent = requests_mock.get.call_args.kwargs.get("headers") or {}
        assert sent.get("Authorization") == "Bearer sk-omlx-from-env"
