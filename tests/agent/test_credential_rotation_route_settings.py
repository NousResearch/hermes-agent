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


def test_codex_rotation_keeps_proxy_override(monkeypatch):
    """#40913: a 401/429 rotation adopts the pool row, whose stored URL is the canonical ChatGPT
    endpoint; with HERMES_CODEX_BASE_URL set the rotated client must keep targeting the proxy."""
    from agent.credential_pool import PooledCredential

    monkeypatch.setenv("HERMES_CODEX_BASE_URL", "http://127.0.0.1:8787/backend-api/codex/")
    entry = PooledCredential(provider="openai-codex", id="second", label="second", auth_type="oauth",
                             priority=1, source="manual:device_code", access_token="tok-second",
                             base_url="https://chatgpt.com/backend-api/codex")
    agent = SimpleNamespace(
        api_mode="codex_responses", provider="openai-codex", model="gpt-5.3-codex", api_key="tok-first",
        base_url="http://127.0.0.1:8787/backend-api/codex",
        _client_kwargs={"api_key": "tok-first", "base_url": "http://127.0.0.1:8787/backend-api/codex"},
        _reapply_route_client_config=MagicMock(), _replace_primary_openai_client=MagicMock(),
    )

    assert AIAgent._swap_credential(agent, entry) is True
    assert agent.base_url == "http://127.0.0.1:8787/backend-api/codex"
    assert agent._client_kwargs["base_url"] == "http://127.0.0.1:8787/backend-api/codex"
    assert agent.api_key == "tok-second"


def test_credential_rotation_splits_a_query_bearing_pool_url():
    """The OpenAI SDK appends paths to base_url verbatim: a pool entry at …/t?team=a must become
    base_url …/t + default_query, like agent_init does, or requests go to …/t?team=a with no
    /chat/completions (review 6 of the OAuth-proxy work, finding 6). A later entry without a
    query must not keep the previous tenant."""
    agent = SimpleNamespace(
        api_mode="chat_completions", provider="custom", model="m", api_key="old",
        base_url="https://relay.example.com/t",
        _client_kwargs={"api_key": "old", "base_url": "https://relay.example.com/t"},
        _apply_client_headers_for_base_url=MagicMock(), _replace_primary_openai_client=MagicMock(),
    )
    agent._reapply_route_client_config = MethodType(AIAgent._reapply_route_client_config, agent)
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        AIAgent._swap_credential(agent, SimpleNamespace(
            runtime_api_key="k1", access_token="", runtime_base_url="https://relay.example.com/t?team=a",
            base_url="https://relay.example.com/t?team=a"))
        assert agent._client_kwargs["base_url"] == "https://relay.example.com/t"
        assert agent._client_kwargs["default_query"] == {"team": "a"}
        assert agent.base_url == "https://relay.example.com/t"
        AIAgent._swap_credential(agent, SimpleNamespace(
            runtime_api_key="k2", access_token="", runtime_base_url="https://relay.example.com/t",
            base_url="https://relay.example.com/t"))
        assert "default_query" not in agent._client_kwargs


def test_credential_rotation_onto_a_url_less_entry_keeps_the_live_query():
    """Azure Foundry seeds its pool entry from AZURE_FOUNDRY_API_KEY with no URL when the URL (and
    its ?api-version=) lives in model.base_url. Such an entry does not decide the query: the live
    default_query must stay (review 7 of the OAuth-proxy work, finding 1)."""
    base = "https://res.openai.azure.com/openai/deployments/gpt"
    agent = SimpleNamespace(
        api_mode="chat_completions", provider="azure-foundry", model="gpt", api_key="old", base_url=base,
        _client_kwargs={"api_key": "old", "base_url": base, "default_query": {"api-version": "2024-10-21"}},
        _apply_client_headers_for_base_url=MagicMock(), _replace_primary_openai_client=MagicMock(),
    )
    agent._reapply_route_client_config = MethodType(AIAgent._reapply_route_client_config, agent)
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        assert AIAgent._swap_credential(agent, SimpleNamespace(
            runtime_api_key="k1", access_token="", runtime_base_url=None, base_url="")) is True
    assert agent._client_kwargs["default_query"] == {"api-version": "2024-10-21"}
    assert agent._client_kwargs["base_url"] == base
    # Same route: the user's default headers are kept.
    agent._apply_client_headers_for_base_url.assert_called_once_with(base, apply_user_headers=True)


def test_rotation_onto_the_same_query_bearing_route_is_not_a_route_change():
    """The stored base_url is query-less; comparing it with the raw pool URL called every rotation
    onto a query-bearing entry a route change, dropping model.default_headers each time
    (review 7, finding 2)."""
    agent = SimpleNamespace(
        api_mode="chat_completions", provider="custom", model="m", api_key="old",
        base_url="https://relay.example.com/t",
        _client_kwargs={"api_key": "old", "base_url": "https://relay.example.com/t", "default_query": {"team": "a"}},
        _apply_client_headers_for_base_url=MagicMock(), _replace_primary_openai_client=MagicMock(),
    )
    agent._reapply_route_client_config = MethodType(AIAgent._reapply_route_client_config, agent)
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        AIAgent._swap_credential(agent, SimpleNamespace(
            runtime_api_key="k1", access_token="", runtime_base_url="https://relay.example.com/t?team=a",
            base_url="https://relay.example.com/t?team=a"))
    agent._apply_client_headers_for_base_url.assert_called_once_with(
        "https://relay.example.com/t", apply_user_headers=True)
    assert agent._client_kwargs["default_query"] == {"team": "a"}
