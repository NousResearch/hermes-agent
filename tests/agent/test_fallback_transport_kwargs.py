"""Fallback activation must adopt the fallback route's own transport policy.

``providers.<name>.ssl_ca_cert`` / ``ssl_verify`` / ``extra_headers`` are keyed by route identity
and are applied by primary init and credential rotation. Fallback activation rebuilt
``_client_kwargs`` from the resolved client object only, so the leg dialed an internal-CA gateway
with the previous route's trust store (``APIConnectionError`` — reported as ``Connection error.``)
and without the provider's ``extra_headers``. These tests pin the route-derived kwargs.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.client_lifecycle import _swap_fallback_clients

GATEWAY_BASE = "https://llm-gateway.internal.example/v1"
GATEWAY_CA = "/tmp/fixture-internal-ca.pem"

ROUTE_CONFIG = {
    "custom_providers": [
        {
            "name": "platform-gateway",
            "base_url": GATEWAY_BASE,
            "ssl_ca_cert": GATEWAY_CA,
            "extra_headers": {"X-Platform-Priority": "normal"},
        }
    ]
}


def _agent(**overrides):
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="stub429",
        model="deepseek-flash",
        api_key="primary-key",
        base_url="http://127.0.0.1:8799/v1",
        client=MagicMock(),
        _client_kwargs={"api_key": "primary-key", "base_url": "http://127.0.0.1:8799/v1"},
        _replace_primary_openai_client=MagicMock(),
    )
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def _fallback_client(headers=None):
    client = SimpleNamespace(api_key="fallback-key", base_url=GATEWAY_BASE)
    if headers:
        client._custom_headers = dict(headers)
    return client


def test_fallback_activation_applies_route_tls_material():
    agent = _agent()

    with patch("hermes_cli.config.load_config_readonly", return_value=ROUTE_CONFIG), patch(
        "agent.client_lifecycle.get_provider_request_timeout", return_value=None
    ):
        _swap_fallback_clients(agent, _fallback_client(), "platform-gateway", "deepseek-flash", GATEWAY_BASE, "chat_completions")

    assert agent._client_kwargs["ssl_ca_cert"] == GATEWAY_CA
    agent._replace_primary_openai_client.assert_called_once_with(reason="fallback_transport_apply")


def test_fallback_activation_merges_route_extra_headers_over_carried_client_headers():
    agent = _agent()

    with patch("hermes_cli.config.load_config_readonly", return_value=ROUTE_CONFIG), patch(
        "agent.client_lifecycle.get_provider_request_timeout", return_value=None
    ):
        _swap_fallback_clients(
            agent, _fallback_client(headers={"User-Agent": "fixture-sentinel"}),
            "platform-gateway", "deepseek-flash", GATEWAY_BASE, "chat_completions",
        )

    headers = agent._client_kwargs["default_headers"]
    # The carrier from resolve_provider_client() survives; the route's own header is added.
    assert headers["User-Agent"] == "fixture-sentinel"
    assert headers["X-Platform-Priority"] == "normal"


def test_fallback_timeout_still_rebuilds_once_with_existing_reason():
    agent = _agent()

    with patch("hermes_cli.config.load_config_readonly", return_value=ROUTE_CONFIG), patch(
        "agent.client_lifecycle.get_provider_request_timeout", return_value=600
    ):
        _swap_fallback_clients(agent, _fallback_client(), "platform-gateway", "deepseek-flash", GATEWAY_BASE, "chat_completions")

    assert agent._client_kwargs["timeout"] == 600
    agent._replace_primary_openai_client.assert_called_once_with(reason="fallback_timeout_apply")


def test_unconfigured_fallback_route_keeps_kwargs_minimal():
    """Neutral control: a route with no config entry must not gain TLS knobs or a rebuild."""
    agent = _agent()

    with patch("hermes_cli.config.load_config_readonly", return_value={"custom_providers": []}), patch(
        "agent.client_lifecycle.get_provider_request_timeout", return_value=None
    ):
        _swap_fallback_clients(agent, _fallback_client(), "unknown", "some-model", "https://api.example/v1", "chat_completions")

    assert set(agent._client_kwargs) == {"api_key", "base_url"}
    agent._replace_primary_openai_client.assert_not_called()
