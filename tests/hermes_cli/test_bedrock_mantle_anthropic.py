"""Bedrock Mantle Claude routing and credential-boundary regressions."""
from unittest.mock import patch

import pytest

from hermes_cli import runtime_provider as rp

MANTLE_BASE = "https://bedrock-mantle.ap-northeast-1.api.aws/anthropic"


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-test-key")
    monkeypatch.setenv("ANTHROPIC_TOKEN", "native-test-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "native-test-key")
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: pytest.fail("native credentials accessed"))
    monkeypatch.setattr(rp, "load_pool", lambda *a, **k: pytest.fail("native pool accessed"))
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "anthropic", "base_url": MANTLE_BASE,
        "api_mode": "anthropic_messages", "default": "anthropic.claude-opus-4-8",
    })


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_runtime_mantle_credential_boundary(monkeypatch, explicit, missing):
    if missing:
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")
    kwargs = {"explicit_base_url": MANTLE_BASE, "explicit_api_key": "native-test-token"} if explicit else {}
    if missing:
        with pytest.raises(rp.AuthError, match="AWS_BEARER_TOKEN_BEDROCK"):
            rp.resolve_runtime_provider(requested="anthropic", **kwargs)
    else:
        resolved = rp.resolve_runtime_provider(requested="anthropic", **kwargs)
        assert (resolved["provider"], resolved["api_mode"], resolved["base_url"], resolved["api_key"]) == (
            "anthropic", "anthropic_messages", MANTLE_BASE, "bedrock-test-key")
        assert not resolved.get("credential_pool")


def test_mantle_client_ignores_native_key():
    from agent.anthropic_adapter import build_anthropic_client
    with patch("agent.anthropic_adapter._anthropic_sdk") as sdk:
        build_anthropic_client("native-test-token", MANTLE_BASE)
        kwargs = sdk.Anthropic.call_args.kwargs
        assert kwargs["auth_token"] == "bedrock-test-key"
        assert "api_key" not in kwargs


@pytest.mark.parametrize("missing", [False, True])
def test_mantle_token_resolution(monkeypatch, missing):
    from agent.anthropic_credentials import resolve_anthropic_token
    if missing:
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")
    assert resolve_anthropic_token(MANTLE_BASE) == (None if missing else "bedrock-test-key")


@pytest.mark.parametrize("url, expected", [
    (MANTLE_BASE, True),
    ("https://bedrock-mantle.us-west-2.api.aws", True),
    (MANTLE_BASE + "/v1", False),
    ("https://bedrock-mantle.us-east-1.api.aws.evil.example/anthropic", False),
    (MANTLE_BASE.replace("https:", "http:"), False),
])
def test_mantle_endpoint_boundary(url, expected):
    from agent.anthropic_endpoints import _is_bedrock_mantle_endpoint
    assert _is_bedrock_mantle_endpoint(url) is expected


@pytest.mark.parametrize("selected", ["anthropic.claude-opus-4-8", "openai.gpt-5.6"])
def test_bedrock_api_key_setup(monkeypatch, selected):
    from hermes_cli.model_setup_flows_bedrock import _model_flow_bedrock_api_key
    config, saved_env = {}, {}
    monkeypatch.setattr("hermes_cli.auth._resolve_api_key_provider_secret", lambda *a: ("bedrock-test-key", "env"))
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda key, value: saved_env.update({key: value}))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.save_config", lambda value: None)
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a: [selected])
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", lambda *a, **k: selected)
    monkeypatch.setattr("hermes_cli.auth._save_model_choice", lambda *a: None)
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)
    _model_flow_bedrock_api_key(config, "ap-northeast-1")
    model = config["model"]
    if selected.startswith("anthropic."):
        assert model == {"provider": "anthropic", "base_url": MANTLE_BASE,
                         "api_mode": "anthropic_messages", "key_env": "AWS_BEARER_TOKEN_BEDROCK"}
    else:
        assert model["provider"] == "custom:bedrock-mantle"
        assert config["providers"]["bedrock-mantle"]["key_env"] == "AWS_BEARER_TOKEN_BEDROCK"
    assert "ANTHROPIC_API_KEY" not in saved_env
    assert "OPENAI_API_KEY" not in saved_env
