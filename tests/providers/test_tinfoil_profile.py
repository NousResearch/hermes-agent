"""Tests for the Tinfoil.sh provider profile."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from providers import get_provider_profile
from providers.base import ProviderProfile

# ── Sample API response ─────────────────────────────────────────────
_TINFOIL_API_RESPONSE = {
    "object": "list",
    "data": [
        {"id": "kimi-k2-6",       "type": "chat",     "tool_calling": True},
        {"id": "glm-5-1",         "type": "chat",     "tool_calling": True},
        {"id": "deepseek-v4-pro", "type": "chat",     "tool_calling": True},
        {"id": "gemma4-31b",      "type": "chat",     "tool_calling": True},
        {"id": "nomic-embed-text",     "type": "embedding", "tool_calling": False},
        {"id": "doc-upload",           "type": "document",  "tool_calling": False},
        {"id": "voxtral-small-24b",    "type": "audio",     "tool_calling": False},
        {"id": "websearch",            "type": "tool",      "tool_calling": False},
        {"id": "whisper-large-v3-turbo", "type": "audio",   "tool_calling": False},
        {"id": "qwen3-tts",            "type": "tts",       "tool_calling": False},
        {"id": "gpt-oss-safeguard-120b", "type": "safety",  "tool_calling": False},
    ],
}

_CHAT_MODEL_IDS = ["kimi-k2-6", "glm-5-1", "deepseek-v4-pro", "gemma4-31b"]

_NON_CHAT_IDS = {
    "nomic-embed-text", "doc-upload", "voxtral-small-24b", "websearch",
    "whisper-large-v3-turbo", "qwen3-tts", "gpt-oss-safeguard-120b",
}


def _mock_httpx_response(data: dict) -> MagicMock:
    """Build a MagicMock that simulates an httpx.Response with JSON data."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = data
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def _make_httpx_client_mock(mock_resp: MagicMock) -> MagicMock:
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    return mock_client


def _clear_tinfoil_cache():
    mod = sys.modules.get("plugins.model_providers.tinfoil")
    if mod is not None:
        mod._TINFOIL_MODEL_CACHE = None


# ── Profile attribute tests ─────────────────────────────────────────

class TestTinfoilProfile:
    def test_name(self):
        p = get_provider_profile("tinfoil")
        assert p.name == "tinfoil"

    def test_aliases(self):
        p = get_provider_profile("tinfoil")
        assert "tinfoil-sh" in p.aliases
        assert "tinfoil.sh" in p.aliases

    def test_base_url(self):
        p = get_provider_profile("tinfoil")
        assert p.base_url == "https://inference.tinfoil.sh/v1"

    def test_env_vars(self):
        p = get_provider_profile("tinfoil")
        assert p.env_vars == ("TINFOIL_API_KEY",)

    def test_auth_type(self):
        p = get_provider_profile("tinfoil")
        assert p.auth_type == "api_key"

    def test_supports_health_check(self):
        p = get_provider_profile("tinfoil")
        assert p.supports_health_check is True

    def test_display_name(self):
        p = get_provider_profile("tinfoil")
        assert p.display_name == "Tinfoil.sh"

    def test_description(self):
        p = get_provider_profile("tinfoil")
        assert "HPKE" in p.description

    def test_signup_url(self):
        p = get_provider_profile("tinfoil")
        assert p.signup_url == "https://tinfoil.sh"

    def test_default_headers_include_content_type(self):
        p = get_provider_profile("tinfoil")
        assert p.default_headers.get("Content-Type") == "application/json"

    def test_fallback_models(self):
        p = get_provider_profile("tinfoil")
        expected = (
            "kimi-k2-6", "glm-5-1", "deepseek-v4-pro",
            "gemma4-31b", "qwen3-vl-30b", "llama3-3-70b", "gpt-oss-120b",
        )
        assert p.fallback_models == expected

    def test_is_subclass_of_provider_profile(self):
        p = get_provider_profile("tinfoil")
        assert isinstance(p, ProviderProfile)

    def test_has_fetch_models_override(self):
        p = get_provider_profile("tinfoil")
        assert type(p).fetch_models is not ProviderProfile.fetch_models

    def test_alias_resolution(self):
        assert get_provider_profile("tinfoil-sh").name == "tinfoil"
        assert get_provider_profile("tinfoil.sh").name == "tinfoil"

    def test_models_url_not_set_explicitly(self):
        p = get_provider_profile("tinfoil")
        assert p.models_url == ""

    def test_hostname_derived_from_base_url(self):
        p = get_provider_profile("tinfoil")
        assert p.get_hostname() == "inference.tinfoil.sh"

    def test_default_max_tokens_not_set(self):
        p = get_provider_profile("tinfoil")
        assert p.default_max_tokens is None

    def test_fixed_temperature_not_set(self):
        p = get_provider_profile("tinfoil")
        assert p.fixed_temperature is None


# ── build_extra_body tests ──────────────────────────────────────────

class TestTinfoilBuildExtraBody:
    def test_known_model_uses_fallback_endpoint(self):
        p = get_provider_profile("tinfoil")
        body = p.build_extra_body(model="kimi-k2-6")
        assert body["tinfoil_endpoint"] == "kimi-k2-6"

    def test_config_endpoint_overrides_fallback(self):
        p = get_provider_profile("tinfoil")
        body = p.build_extra_body(
            model="kimi-k2-6",
            tinfoil_endpoints={"kimi-k2-6": "custom-path"},
        )
        assert body["tinfoil_endpoint"] == "custom-path"

    def test_api_discovered_model_falls_through_to_model_name(self):
        p = get_provider_profile("tinfoil")
        body = p.build_extra_body(model="brand-new-model-v2")
        assert body["tinfoil_endpoint"] == "brand-new-model-v2"

    def test_strips_tinfoil_provider_prefix(self):
        p = get_provider_profile("tinfoil")
        body = p.build_extra_body(model="tinfoil/kimi-k2-6")
        assert body["tinfoil_endpoint"] == "kimi-k2-6"

    def test_empty_model_omits_endpoint(self):
        p = get_provider_profile("tinfoil")
        body = p.build_extra_body(model="")
        assert "tinfoil_endpoint" not in body

    def test_no_args_returns_empty(self):
        p = get_provider_profile("tinfoil")
        body = p.build_extra_body()
        assert body == {}

    def test_case_insensitive_model_match(self):
        p = get_provider_profile("tinfoil")
        body = p.build_extra_body(model="KIMI-K2-6")
        assert body["tinfoil_endpoint"] == "kimi-k2-6"

    def test_config_endpoint_beats_fallback(self):
        p = get_provider_profile("tinfoil")
        body = p.build_extra_body(
            model="kimi-k2-6",
            tinfoil_endpoints={"kimi-k2-6": "user-configured-path"},
        )
        assert body["tinfoil_endpoint"] == "user-configured-path"

    def test_preserves_other_context_keys(self):
        p = get_provider_profile("tinfoil")
        body = p.build_extra_body(model="kimi-k2-6", session_id="sess-123")
        assert body["tinfoil_endpoint"] == "kimi-k2-6"

    def test_empty_endpoints_config_falls_through(self):
        p = get_provider_profile("tinfoil")
        body = p.build_extra_body(model="kimi-k2-6", tinfoil_endpoints={})
        assert body["tinfoil_endpoint"] == "kimi-k2-6"


# ── build_api_kwargs_extras tests ───────────────────────────────────

class TestTinfoilBuildApiKwargsExtras:
    def test_returns_content_type_headers_by_default(self):
        p = get_provider_profile("tinfoil")
        eb, tl = p.build_api_kwargs_extras()
        assert eb == {}
        assert tl["extra_headers"]["Content-Type"] == "application/json"
        assert "X-Tinfoil-Sdk-Version" not in tl["extra_headers"]

    def test_sdk_version_sets_extra_headers(self, monkeypatch):
        monkeypatch.setenv("TINFOIL_SDK_VERSION", "1.2.3")
        p = get_provider_profile("tinfoil")
        eb, tl = p.build_api_kwargs_extras()
        assert tl["extra_headers"]["Content-Type"] == "application/json"
        assert tl["extra_headers"]["X-Tinfoil-Sdk-Version"] == "1.2.3"

    def test_sdk_version_empty_excludes_version(self, monkeypatch):
        monkeypatch.setenv("TINFOIL_SDK_VERSION", "")
        p = get_provider_profile("tinfoil")
        eb, tl = p.build_api_kwargs_extras()
        assert "X-Tinfoil-Sdk-Version" not in tl["extra_headers"]

    def test_sdk_version_unset_excludes_version(self):
        p = get_provider_profile("tinfoil")
        eb, tl = p.build_api_kwargs_extras()
        assert "X-Tinfoil-Sdk-Version" not in tl["extra_headers"]

    def test_reasoning_config_ignored(self):
        p = get_provider_profile("tinfoil")
        eb, tl = p.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            supports_reasoning=True,
        )
        assert eb == {}
        assert tl["extra_headers"]["Content-Type"] == "application/json"


# ── fetch_models tests ──────────────────────────────────────────────

class TestTinfoilFetchModels:
    TINFOIL_MODULE = "plugins.model_providers.tinfoil"

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        _clear_tinfoil_cache()

    def test_returns_only_chat_models(self):
        p = get_provider_profile("tinfoil")
        mock_resp = _mock_httpx_response(_TINFOIL_API_RESPONSE)
        mock_client = _make_httpx_client_mock(mock_resp)
        with patch("httpx.Client", return_value=mock_client):
            models = p.fetch_models(api_key="test-key")
        assert models == _CHAT_MODEL_IDS

    def test_excludes_embedding_audio_tts_tool_document_safety(self):
        p = get_provider_profile("tinfoil")
        mock_resp = _mock_httpx_response(_TINFOIL_API_RESPONSE)
        mock_client = _make_httpx_client_mock(mock_resp)
        with patch("httpx.Client", return_value=mock_client):
            models = p.fetch_models(api_key="test-key")
        returned = set(models)
        assert returned & _NON_CHAT_IDS == set()

    def test_caches_results(self):
        p = get_provider_profile("tinfoil")
        mock_resp = _mock_httpx_response(_TINFOIL_API_RESPONSE)
        mock_client = _make_httpx_client_mock(mock_resp)
        with patch("httpx.Client", return_value=mock_client):
            p.fetch_models(api_key="test-key")
        with patch("httpx.Client", side_effect=Exception("should not be called")):
            models = p.fetch_models(api_key="test-key")
        assert models == _CHAT_MODEL_IDS

    def test_returns_none_on_network_error(self):
        p = get_provider_profile("tinfoil")
        with patch("httpx.Client", side_effect=Exception("connection refused")):
            models = p.fetch_models(api_key="test-key")
        assert models is None

    def test_returns_none_on_invalid_json(self):
        p = get_provider_profile("tinfoil")
        mock_resp = MagicMock()
        mock_resp.json.side_effect = json.JSONDecodeError("not json", "", 0)
        mock_resp.raise_for_status.return_value = None
        mock_client = _make_httpx_client_mock(mock_resp)
        with patch("httpx.Client", return_value=mock_client):
            models = p.fetch_models(api_key="test-key")
        assert models is None

    def test_returns_none_on_timeout(self):
        p = get_provider_profile("tinfoil")
        with patch("httpx.Client", side_effect=TimeoutError("timed out")):
            models = p.fetch_models(api_key="test-key")
        assert models is None

    def test_sends_bearer_auth(self):
        p = get_provider_profile("tinfoil")
        mock_resp = _mock_httpx_response(_TINFOIL_API_RESPONSE)
        captured = {}

        def capture_headers(url, **kw):
            captured["h"] = kw.get("headers", {})
            return mock_resp

        mock_client = MagicMock()
        mock_client.get.side_effect = capture_headers
        mock_client.__enter__.return_value = mock_client
        with patch("httpx.Client", return_value=mock_client):
            p.fetch_models(api_key="my-secret-key")
        assert captured["h"].get("Authorization") == "Bearer my-secret-key"
        assert captured["h"].get("User-Agent", "").startswith("hermes-cli/")
        assert captured["h"].get("Accept") == "application/json"

    def test_no_auth_when_api_key_not_provided(self):
        p = get_provider_profile("tinfoil")
        mock_resp = _mock_httpx_response(_TINFOIL_API_RESPONSE)
        captured = {}

        def capture_headers(url, **kw):
            captured["h"] = kw.get("headers", {})
            return mock_resp

        mock_client = MagicMock()
        mock_client.get.side_effect = capture_headers
        mock_client.__enter__.return_value = mock_client
        with patch("httpx.Client", return_value=mock_client):
            models = p.fetch_models(api_key=None)
        assert "Authorization" not in captured["h"]
        assert models == _CHAT_MODEL_IDS

    def test_correct_url(self):
        p = get_provider_profile("tinfoil")
        mock_resp = _mock_httpx_response({"data": []})
        captured = {}

        def capture_url(url, **kw):
            captured["url"] = str(url)
            return mock_resp

        mock_client = MagicMock()
        mock_client.get.side_effect = capture_url
        mock_client.__enter__.return_value = mock_client
        with patch("httpx.Client", return_value=mock_client):
            p.fetch_models(api_key="test-key")
        assert captured["url"] == "https://inference.tinfoil.sh/v1/models"

    def test_forwards_default_headers(self):
        p = get_provider_profile("tinfoil")
        mock_resp = _mock_httpx_response(_TINFOIL_API_RESPONSE)
        captured = {}

        def capture_headers(url, **kw):
            captured["h"] = kw.get("headers", {})
            return mock_resp

        mock_client = MagicMock()
        mock_client.get.side_effect = capture_headers
        mock_client.__enter__.return_value = mock_client
        with patch("httpx.Client", return_value=mock_client):
            p.fetch_models(api_key="test-key")
        assert captured["h"].get("Content-Type") == "application/json"

    def test_empty_data_list_returns_empty_list(self):
        p = get_provider_profile("tinfoil")
        mock_resp = _mock_httpx_response({"data": []})
        mock_client = _make_httpx_client_mock(mock_resp)
        with patch("httpx.Client", return_value=mock_client):
            models = p.fetch_models(api_key="test-key")
        assert models == []

    def test_no_chat_models_returns_empty(self):
        p = get_provider_profile("tinfoil")
        response = {"data": [{"id": "only-embedding", "type": "embedding", "tool_calling": False}]}
        mock_resp = _mock_httpx_response(response)
        mock_client = _make_httpx_client_mock(mock_resp)
        with patch("httpx.Client", return_value=mock_client):
            models = p.fetch_models(api_key="test-key")
        assert models == []

    def test_response_is_list_directly(self):
        p = get_provider_profile("tinfoil")
        response = [{"id": "model-a", "type": "chat"}, {"id": "model-b", "type": "chat"}]
        mock_resp = _mock_httpx_response(response)
        mock_client = _make_httpx_client_mock(mock_resp)
        with patch("httpx.Client", return_value=mock_client):
            models = p.fetch_models(api_key="test-key")
        assert models == ["model-a", "model-b"]

    def test_cache_not_populated_on_failure(self):
        p = get_provider_profile("tinfoil")
        with patch("httpx.Client", side_effect=Exception("fail")):
            p.fetch_models(api_key="test-key")
        mod = sys.modules.get(self.TINFOIL_MODULE)
        assert mod is not None
        assert mod._TINFOIL_MODEL_CACHE is None


# ── Transport parity tests ──────────────────────────────────────────

class TestTinfoilTransportParity:
    def test_tinfoil_endpoint_in_transport_kwargs(self):
        from agent.transports.chat_completions import ChatCompletionsTransport
        transport = ChatCompletionsTransport()
        p = get_provider_profile("tinfoil")
        kw = transport.build_kwargs(
            model="tinfoil/kimi-k2-6",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            provider_profile=p,
        )
        assert kw["extra_body"]["tinfoil_endpoint"] == "kimi-k2-6"

    def test_no_sdk_extra_headers_by_default_in_transport(self):
        from agent.transports.chat_completions import ChatCompletionsTransport
        transport = ChatCompletionsTransport()
        p = get_provider_profile("tinfoil")
        kw = transport.build_kwargs(
            model="tinfoil/kimi-k2-6",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            provider_profile=p,
        )
        extra_headers = kw.get("extra_headers", {})
        assert extra_headers.get("Content-Type") == "application/json"
        assert "X-Tinfoil-Sdk-Version" not in extra_headers
