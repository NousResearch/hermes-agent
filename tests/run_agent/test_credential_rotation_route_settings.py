"""Credential rotation must not carry route-scoped TLS policy."""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def test_credential_rotation_replaces_route_scoped_tls_settings():
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="custom",
        model="shared-model",
        api_key="old",
        base_url="https://a.example/v1",
        _client_kwargs={
            "api_key": "old",
            "base_url": "https://a.example/v1",
            "ssl_verify": False,
            "ssl_ca_cert": "/a.pem",
        },
        _apply_client_headers_for_base_url=MagicMock(),
        _replace_primary_openai_client=MagicMock(),
    )
    entry = SimpleNamespace(
        runtime_api_key="new",
        access_token="",
        runtime_base_url="https://b.example/v1",
        base_url="https://b.example/v1",
    )
    config = {
        "custom_providers": [
            {
                "name": "b",
                "base_url": "https://b.example/v1",
                "ssl_verify": True,
            }
        ]
    }

    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        AIAgent._swap_credential(agent, entry)

    assert agent._client_kwargs["ssl_verify"] is True
    assert "ssl_ca_cert" not in agent._client_kwargs
    agent._replace_primary_openai_client.assert_called_once_with(
        reason="credential_rotation"
    )


def test_credential_rotation_does_not_carry_global_headers_across_routes():
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="custom",
        model="shared-model",
        api_key="old",
        base_url="https://a.example/v1",
        _client_kwargs={
            "api_key": "old",
            "base_url": "https://a.example/v1",
            "default_headers": {"Authorization": "old-secret"},
        },
        _replace_primary_openai_client=MagicMock(),
    )
    agent._apply_client_headers_for_base_url = MethodType(
        AIAgent._apply_client_headers_for_base_url,
        agent,
    )
    agent._apply_user_default_headers = MethodType(
        AIAgent._apply_user_default_headers,
        agent,
    )
    entry = SimpleNamespace(
        runtime_api_key="new",
        access_token="",
        runtime_base_url="https://b.example/v1",
        base_url="https://b.example/v1",
    )
    config = {
        "model": {
            "default_headers": {"Authorization": "global-secret"},
        },
        "custom_providers": [
            {
                "name": "b",
                "base_url": "https://b.example/v1",
                "extra_headers": {"X-Route": "b"},
            }
        ],
    }

    with (
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=config["custom_providers"],
        ),
    ):
        AIAgent._swap_credential(agent, entry)

    headers = agent._client_kwargs["default_headers"]
    assert "Authorization" not in headers
    assert headers["X-Route"] == "b"


def test_credential_rotation_preserves_route_significant_trailing_segments():
    """Route identity comparison uses normalize_route_base_url, but the stored
    base_url is stripped like every other assignment site (__init__, switch_model)."""
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="custom",
        model="shared-model",
        api_key="old",
        base_url="https://a.example/v1",
        _client_kwargs={
            "api_key": "old",
            "base_url": "https://a.example/v1",
        },
        _apply_client_headers_for_base_url=MagicMock(),
        _replace_primary_openai_client=MagicMock(),
    )
    entry = SimpleNamespace(
        runtime_api_key="new",
        access_token="",
        runtime_base_url="https://b.example/v1//",
        base_url="https://b.example/v1//",
    )

    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        AIAgent._swap_credential(agent, entry)

    assert agent.base_url == "https://b.example/v1"
    assert agent._client_kwargs["base_url"] == "https://b.example/v1"
