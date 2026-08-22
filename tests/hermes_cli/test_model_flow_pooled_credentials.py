"""Pool-only credentials must be visible to interactive model setup flows."""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.model_setup_flows import _existing_api_key_for_model_flow


class _PoolEntry:
    access_token = "pool-secret"
    runtime_api_key = ""


class _AvailablePool:
    def has_credentials(self) -> bool:
        return True

    def peek(self):
        return _PoolEntry()


class _ExhaustedPool:
    def has_credentials(self) -> bool:
        return True

    def peek(self):
        return None






def test_generic_api_key_flow_passes_pool_key_to_existing_key_prompt(monkeypatch):
    from hermes_cli.model_setup_flows import _model_flow_api_key_provider

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    captured: dict[str, str] = {}

    def capture_prompt(_pconfig, existing_key, **_kwargs):
        captured["existing_key"] = existing_key
        return existing_key, True

    with (
        patch("hermes_cli.config.get_env_value", return_value=""),
        patch("agent.credential_pool.load_pool", return_value=_AvailablePool()),
        patch("hermes_cli.main._prompt_api_key", side_effect=capture_prompt),
    ):
        _model_flow_api_key_provider({}, "deepseek")

    assert captured["existing_key"] == "pool-secret"




def test_bedrock_flow_sees_pool_key_when_no_env(monkeypatch, capsys):
    """Bedrock API-key mode must also see pool-backed credentials."""
    from hermes_cli.model_setup_flows import _model_flow_bedrock_api_key

    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    with (
        patch("hermes_cli.config.get_env_value", return_value=""),
        patch("agent.credential_pool.load_pool", return_value=_AvailablePool()),
        patch("builtins.input", return_value="k"),
    ):
        _model_flow_bedrock_api_key({}, "us-east-1")

    out = capsys.readouterr().out
    # The flow should show the pool-backed key, not prompt for a new one
    assert "pool-secret" in out[:200] or "pool-sec" in out[:200]


def test_bedrock_flow_region_prompt_uses_runtime_resolver(monkeypatch, capsys):
    """The interactive Bedrock setup/reconfigure wizard's region-prompt
    default must come from the config-first resolve_bedrock_runtime_region,
    not the bare env/profile-only resolve_bedrock_region — otherwise
    reconfiguring an already-configured Bedrock provider (bedrock.region
    pinned in config.yaml, but a different ambient AWS_REGION/profile set)
    shows the wrong default, and accepting it with a bare Enter re-saves the
    wrong region, silently regressing the config-pinned one (#199 in
    _model_flow_bedrock writes bedrock_cfg["region"] = region verbatim)."""
    from hermes_cli.model_setup_flows import _model_flow_bedrock

    import agent.bedrock_adapter as ba

    monkeypatch.setattr(ba, "has_aws_credentials", lambda: True)
    monkeypatch.setattr(ba, "resolve_aws_auth_env_var", lambda: "AWS_BEARER_TOKEN_BEDROCK")
    # Deliberately distinct sentinels: the config-first resolver must win.
    monkeypatch.setattr(ba, "resolve_bedrock_runtime_region", lambda: "eu-central-1")
    monkeypatch.setattr(ba, "resolve_bedrock_region", lambda: "us-east-1")

    captured_prompts: list[str] = []

    def fake_input(prompt_text="", *a, **kw):
        captured_prompts.append(prompt_text)
        raise EOFError()  # bail out right after the region prompt renders

    with patch("builtins.input", side_effect=fake_input):
        _model_flow_bedrock({}, "")

    assert any("[eu-central-1]" in p for p in captured_prompts), captured_prompts
    assert not any("us-east-1" in p for p in captured_prompts), captured_prompts
