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


def test_nous_rotation_keeps_inference_base_url_override(monkeypatch):
    """#121323: a 401 rotation must not replace the operator's Nous route.

    Pool entries intentionally persist the Portal-provided production URL.  The
    runtime overlay is therefore required both for the initial entry and for a
    refreshed entry adopted by ``_swap_credential``.
    """
    from agent.credential_pool import PooledCredential

    override = "https://staging-inference.example/v1/"
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", override)
    entry = PooledCredential(
        provider="nous", id="refreshed", label="Nous", auth_type="oauth",
        priority=1, source="device_code", access_token="tok-second",
        inference_base_url="https://inference-api.nousresearch.com/v1",
    )
    agent = SimpleNamespace(
        api_mode="chat_completions", provider="nous", model="Hermes-4-70B", api_key="tok-first",
        base_url=override.rstrip("/"),
        _client_kwargs={"api_key": "tok-first", "base_url": override.rstrip("/")},
        _reapply_route_client_config=MagicMock(), _replace_primary_openai_client=MagicMock(),
    )

    assert entry.runtime_base_url == override.rstrip("/")
    assert AIAgent._swap_credential(agent, entry) is True
    assert agent.base_url == override.rstrip("/")
    assert agent._client_kwargs["base_url"] == override.rstrip("/")


def test_nous_rotation_uses_stored_inference_url_without_override(monkeypatch):
    """No override preserves the Portal-provided route for regular Nous users."""
    from agent.credential_pool import PooledCredential

    monkeypatch.delenv("NOUS_INFERENCE_BASE_URL", raising=False)
    stored_url = "https://inference-api.nousresearch.com/v1"
    entry = PooledCredential(
        provider="nous", id="stored", label="Nous", auth_type="oauth",
        priority=0, source="device_code", access_token="tok",
        inference_base_url=stored_url,
    )

    assert entry.runtime_base_url == stored_url
