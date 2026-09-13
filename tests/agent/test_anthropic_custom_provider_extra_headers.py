"""Custom-provider ``extra_headers`` on the Anthropic wire.

The OpenAI-wire clients pick up ``custom_providers[].extra_headers`` through
``apply_custom_provider_extra_headers_to_client_kwargs``. A custom provider declared with
``api_mode: anthropic_messages`` (a gateway behind Cloudflare Access, a router that wants
attribution headers, ...) goes through ``build_anthropic_client`` instead, so the same lookup
has to happen in ``_new_sdk_client``: that is the single constructor behind the direct client,
the Entra bearer-hook client, the fallback swap and every auxiliary client.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.anthropic_adapter import _merge_extra_headers, build_anthropic_client

_PROXY_URL = "https://llm.internal.example.com/v1"
_EXTRA_HEADERS = {
    "CF-Access-Client-Id": "xxxx.access",
    "CF-Access-Client-Secret": "cf-secret-value",
    "X-Client-Name": "hermes-agent",
}


def _config(extra_headers=None, base_url=_PROXY_URL):
    return {
        "custom_providers": [{
            "name": "my-anthropic-proxy",
            "base_url": base_url,
            "api_key": "proxy-key",
            "api_mode": "anthropic_messages",
            "extra_headers": dict(_EXTRA_HEADERS if extra_headers is None else extra_headers),
        }],
    }


def wire_headers(client) -> dict:
    """Headers the SDK would put on a /v1/messages POST (Omit() defaults are resolved here)."""
    from anthropic._models import FinalRequestOptions

    return dict(client._build_headers(FinalRequestOptions(method="post", url="/v1/messages", json_data={})))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def proxy_config():
    with patch("hermes_cli.config.load_config", return_value=_config()):
        yield


def test_merge_extra_headers_replaces_case_insensitively_and_keeps_order():
    headers = {"anthropic-beta": "beta-a", "Authorization": "omit-sentinel", "X-Title": "Hermes Agent"}
    merged = _merge_extra_headers(headers, {"Anthropic-Beta": "beta-b", "authorization": "Bearer gw"})
    assert merged == {"X-Title": "Hermes Agent", "Anthropic-Beta": "beta-b", "authorization": "Bearer gw"}
    assert len([k for k in merged if k.lower() == "anthropic-beta"]) == 1
    # No-op merge returns the input untouched (no copy, no mutation).
    assert _merge_extra_headers(headers, {}) is headers
    assert headers == {"anthropic-beta": "beta-a", "Authorization": "omit-sentinel", "X-Title": "Hermes Agent"}


def test_direct_client_sends_provider_extra_headers(proxy_config):
    pytest.importorskip("anthropic")
    client = build_anthropic_client("proxy-key", _PROXY_URL)
    for wire_client in (client, client.with_options(timeout=30)):
        headers = wire_headers(wire_client)
        assert headers.get("cf-access-client-id") == "xxxx.access"
        assert headers.get("cf-access-client-secret") == "cf-secret-value"
        assert headers.get("x-client-name") == "hermes-agent"
        # The credential guard from _new_sdk_client is unchanged: one credential, no env bearer.
        assert headers.get("x-api-key") == "proxy-key"
        assert "authorization" not in headers


def test_route_matching_tolerates_trailing_slash_and_host_case(proxy_config):
    pytest.importorskip("anthropic")
    client = build_anthropic_client("proxy-key", "https://LLM.internal.example.com/v1/")
    assert wire_headers(client).get("x-client-name") == "hermes-agent"


def test_non_matching_route_gets_no_extra_headers(proxy_config):
    pytest.importorskip("anthropic")
    client = build_anthropic_client("other-key", "https://other.example.com/v1")
    headers = wire_headers(client)
    assert "cf-access-client-id" not in headers
    assert "x-client-name" not in headers
    assert headers.get("x-api-key") == "other-key"


def test_provider_extra_headers_win_over_hermes_defaults_and_the_omit_sentinel():
    pytest.importorskip("anthropic")
    overrides = {"Anthropic-Beta": "my-gateway-beta", "Authorization": "Bearer gateway-token"}
    with patch("hermes_cli.config.load_config", return_value=_config(overrides)):
        client = build_anthropic_client("proxy-key", _PROXY_URL)
    headers = wire_headers(client)
    # Hermes normally sets anthropic-beta itself; the provider entry is the more specific level.
    assert headers.get("anthropic-beta") == "my-gateway-beta"
    # An Authorization the user named explicitly replaces the Omit() guard instead of being dropped.
    assert headers.get("authorization") == "Bearer gateway-token"
    assert headers.get("x-api-key") == "proxy-key"


def test_bearer_hook_client_sends_provider_extra_headers(proxy_config):
    """Callable token (Entra ID) path: ``_build_anthropic_client_with_bearer_hook`` shares ``_new_sdk_client``."""
    pytest.importorskip("anthropic")
    import httpx

    with patch("agent.azure_identity_adapter.build_bearer_http_client", return_value=httpx.Client()) as hook:
        client = build_anthropic_client(lambda: "entra-jwt", _PROXY_URL)
    assert hook.call_count == 1
    headers = wire_headers(client)
    assert headers.get("cf-access-client-id") == "xxxx.access"
    assert headers.get("x-client-name") == "hermes-agent"
    assert "x-api-key" not in headers


def test_fallback_swap_builds_anthropic_client_with_extra_headers(proxy_config):
    """``_swap_fallback_clients`` rebuilds the Anthropic client from the fallback route."""
    pytest.importorskip("anthropic")
    from agent.client_lifecycle import _swap_fallback_clients

    agent = SimpleNamespace(api_key=None, client=None, _client_kwargs={}, _anthropic_client=None)
    fb_client = SimpleNamespace(api_key="fallback-key")
    _swap_fallback_clients(agent, fb_client, "custom", "my-model", _PROXY_URL, "anthropic_messages")

    assert agent._anthropic_base_url == _PROXY_URL
    assert agent._anthropic_api_key == "fallback-key"
    headers = wire_headers(agent._anthropic_client)
    assert headers.get("cf-access-client-id") == "xxxx.access"
    assert headers.get("x-api-key") == "fallback-key"


def test_auxiliary_custom_endpoint_client_carries_extra_headers(proxy_config):
    """``_try_custom_endpoint`` (compression, titles, ...) builds through ``build_anthropic_client``."""
    pytest.importorskip("anthropic")
    from agent.auxiliary_client import AnthropicAuxiliaryClient, _try_custom_endpoint

    with patch(
        "agent.auxiliary_client._resolve_custom_runtime",
        return_value=(_PROXY_URL, "proxy-key", "anthropic_messages"),
    ), patch("agent.auxiliary_client._read_main_model", return_value="my-model"):
        client, model = _try_custom_endpoint()

    assert isinstance(client, AnthropicAuxiliaryClient)
    assert model == "my-model"
    headers = wire_headers(client._real_client)
    assert headers.get("cf-access-client-id") == "xxxx.access"
    assert headers.get("x-client-name") == "hermes-agent"


def test_extra_header_values_are_never_logged(proxy_config, caplog):
    pytest.importorskip("anthropic")
    with caplog.at_level(logging.DEBUG):
        build_anthropic_client("proxy-key", _PROXY_URL)
    assert "cf-secret-value" not in caplog.text
    assert "xxxx.access" not in caplog.text


def test_lookup_failure_degrades_to_plain_client():
    pytest.importorskip("anthropic")
    with patch("hermes_cli.config.get_custom_provider_extra_headers", side_effect=RuntimeError("boom")):
        client = build_anthropic_client("proxy-key", _PROXY_URL)
    headers = wire_headers(client)
    assert headers.get("x-api-key") == "proxy-key"
    assert "cf-access-client-id" not in headers


def test_direct_lookup_is_routed_through_the_shared_config_helper(proxy_config):
    """Guards against a second, drifting header-lookup implementation for the Anthropic wire."""
    pytest.importorskip("anthropic")
    with patch("hermes_cli.config.get_custom_provider_extra_headers", return_value={"X-Probe": "1"}) as lookup:
        client = build_anthropic_client("proxy-key", _PROXY_URL)
    lookup.assert_called_once_with(_PROXY_URL)
    assert wire_headers(client).get("x-probe") == "1"
    assert isinstance(lookup, MagicMock)
