"""Credential rotation must not carry route-scoped TLS policy."""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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
    agent._reapply_route_client_config = MethodType(
        AIAgent._reapply_route_client_config,
        agent,
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
    agent._reapply_route_client_config = MethodType(
        AIAgent._reapply_route_client_config,
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


@pytest.mark.parametrize(
    "api_mode,current_url,entry_url,expected_url",
    [
        (
            "codex_responses",
            "https://api.deepseek.com",
            "https://api.deepseek.com/v1",
            "https://api.deepseek.com",
        ),
        (
            "chat_completions",
            "https://api.deepseek.com/v1",
            "https://api.deepseek.com",
            "https://api.deepseek.com/v1",
        ),
    ],
)
def test_deepseek_credential_rotation_preserves_wire_specific_official_root(
    api_mode, current_url, entry_url, expected_url
):
    agent = SimpleNamespace(
        api_mode=api_mode,
        provider="deepseek",
        model=(
            "deepseek-v4-flash"
            if api_mode == "codex_responses"
            else "deepseek-v4-pro"
        ),
        api_key="old",
        base_url=current_url,
        _client_kwargs={"api_key": "old", "base_url": current_url},
        _reapply_route_client_config=MagicMock(),
        _replace_primary_openai_client=MagicMock(),
    )
    entry = SimpleNamespace(
        id="pool-entry",
        runtime_api_key="new",
        access_token="",
        runtime_base_url=entry_url,
        base_url=entry_url,
    )

    AIAgent._swap_credential(agent, entry)

    assert agent.base_url == expected_url
    assert agent._client_kwargs["base_url"] == expected_url
    agent._replace_primary_openai_client.assert_called_once_with(
        reason="credential_rotation"
    )
