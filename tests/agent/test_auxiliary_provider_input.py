"""Auxiliary provider/model input resolution regressions."""

from unittest.mock import MagicMock, patch

from agent.auxiliary_client import _resolve_task_provider_model


def test_explicit_copilot_base_url_preserves_provider_identity():
    """Copilot is first-class even though it uses an OpenAI-compatible URL."""
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
        provider="copilot",
        model="gpt-5.4-mini",
        base_url="https://api.githubcopilot.com",
    )

    assert provider == "copilot"
    assert model == "gpt-5.4-mini"
    assert base_url == "https://api.githubcopilot.com"
    assert api_key is None
    assert api_mode is None


def test_openai_alias_with_explicit_base_url_still_routes_as_custom():
    """The direct OpenAI alias remains custom because it is not first-class."""
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
        provider="openai",
        model="gpt-4o-mini",
        base_url="https://proxy.example.com/v1",
        api_key="sk-test",
    )

    assert provider == "custom"
    assert model == "gpt-4o-mini"
    assert base_url == "https://proxy.example.com/v1"
    assert api_key == "sk-test"
    assert api_mode is None


def test_auto_provider_with_explicit_base_url_routes_as_custom():
    """Explicit auto provider must not override an explicit endpoint."""
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
        provider="auto",
        model="gpt-4o-mini",
        base_url="https://proxy.example.com/v1",
        api_key="sk-test",
    )

    assert provider == "custom"
    assert model == "gpt-4o-mini"
    assert base_url == "https://proxy.example.com/v1"
    assert api_key == "sk-test"
    assert api_mode is None


def test_resolved_copilot_client_uses_copilot_credentials_and_headers():
    """The preserved provider identity reaches Copilot auth, not custom auth."""
    from agent.auxiliary_client import resolve_provider_client

    fake_client = MagicMock()
    fake_headers = {
        "Authorization": "Bearer ghu_test",
        "Editor-Version": "Hermes/0.0",
    }

    with patch(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        return_value={
            "api_key": "ghu_test",
            "base_url": "https://api.githubcopilot.com",
        },
    ), patch(
        "hermes_cli.copilot_auth.copilot_request_headers",
        return_value=fake_headers,
    ), patch("agent.auxiliary_client.OpenAI", return_value=fake_client) as openai:
        client, resolved_model = resolve_provider_client(
            "copilot",
            model="gpt-5.4-mini",
            explicit_base_url="https://api.githubcopilot.com",
            raw_codex=True,
        )

    assert client is fake_client
    assert resolved_model == "gpt-5.4-mini"
    assert openai.call_args.kwargs["api_key"] == "ghu_test"
    assert openai.call_args.kwargs["base_url"] == "https://api.githubcopilot.com"
    assert openai.call_args.kwargs["default_headers"] == fake_headers
