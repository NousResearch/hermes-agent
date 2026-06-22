"""Regression test for issue #50292."""

import agent.bedrock_adapter as bedrock_adapter
from hermes_cli import runtime_provider as rp


def _patch_bedrock(monkeypatch, *, config_default, provider="bedrock"):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "bedrock")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": provider, "default": config_default},
    )
    monkeypatch.setattr(rp, "load_config", lambda: {"bedrock": {}})
    monkeypatch.setattr(rp, "_resolve_explicit_runtime", lambda **k: None)
    monkeypatch.setattr(
        bedrock_adapter, "resolve_bedrock_region", lambda *a, **k: "us-east-1"
    )
    monkeypatch.setattr(
        bedrock_adapter, "resolve_aws_auth_env_var", lambda *a, **k: "AWS_PROFILE"
    )
    monkeypatch.setattr(bedrock_adapter, "has_aws_credentials", lambda *a, **k: True)


def test_bedrock_non_claude_runtime_model_uses_converse(monkeypatch):
    # Config default is Claude, but the runtime --model is a non-Claude model:
    # it must route through the Converse API, not the Anthropic SDK path.
    _patch_bedrock(monkeypatch, config_default="global.anthropic.claude-sonnet-4-6")

    resolved = rp.resolve_runtime_provider(
        requested="bedrock", target_model="deepseek.v3.2"
    )

    assert resolved["provider"] == "bedrock"
    assert resolved["api_mode"] == "bedrock_converse"
    assert resolved.get("bedrock_anthropic") is not True


def test_bedrock_claude_runtime_model_uses_anthropic_sdk(monkeypatch):
    # A Claude runtime model still routes through the AnthropicBedrock SDK.
    _patch_bedrock(monkeypatch, config_default="deepseek.v3.2")

    resolved = rp.resolve_runtime_provider(
        requested="bedrock", target_model="global.anthropic.claude-sonnet-4-6"
    )

    assert resolved["provider"] == "bedrock"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["bedrock_anthropic"] is True


def test_bedrock_falls_back_to_config_default_without_target_model(monkeypatch):
    # No runtime override: the config default still decides routing.
    _patch_bedrock(monkeypatch, config_default="global.anthropic.claude-sonnet-4-6")

    resolved = rp.resolve_runtime_provider(requested="bedrock")

    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["bedrock_anthropic"] is True
