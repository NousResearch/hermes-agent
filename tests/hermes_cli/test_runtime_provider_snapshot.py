import base64
import json
import time
from types import SimpleNamespace

import pytest

from hermes_cli import runtime_provider as rp


@pytest.fixture
def snapshot_scope(monkeypatch):
    """Real scoped reads, synthetic credentials, and contradictory disk config."""
    from agent import secret_scope
    reads = []
    ambient = {
        "model": {"provider": "custom", "base_url": "https://ambient.invalid/v1",
                  "api_key": "ambient-config-key", "api_mode": "anthropic_messages"},
        "local_runtime": {"enabled": False},
        "bedrock": {"region": "us-west-2"},
    }

    def load_ambient():
        import traceback
        reads.append([(frame.filename, frame.name) for frame in traceback.extract_stack()])
        return ambient

    monkeypatch.setattr(rp, "load_config", load_ambient)
    monkeypatch.setattr(rp._config_mod, "load_config", load_ambient)
    # Unknown-name hints are auth-owned diagnostics, outside the resolver snapshot.
    monkeypatch.setattr(rp.auth_mod, "_get_config_hint_for_unknown_provider", lambda name: "")
    # Native pool discovery owns its own disk reads; keep that credential seam synthetic.
    monkeypatch.setattr(rp, "custom_provider_pool_key_candidates",
                        lambda url, provider_name=None, config=None: ["custom:" + (provider_name or "local")])
    monkeypatch.setattr(rp, "load_pool", lambda name: SimpleNamespace(has_credentials=lambda: False))
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    token = secret_scope.set_secret_scope({"FB_KEY": "scoped-entry-key"})
    try:
        yield reads
    finally:
        secret_scope.reset_secret_scope(token)


def _manual_snapshot_entry(config, index=0):
    from hermes_cli.fallback_config import get_fallback_chain, resolve_entry_api_key
    entry = get_fallback_chain(config)[index]
    return rp.resolve_runtime_provider(
        requested=entry["provider"], target_model=entry["model"],
        explicit_base_url=entry.get("base_url"),
        explicit_api_key=resolve_entry_api_key(entry, strict=True), config=config,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_snapshot_disabled_guard_uses_supplied_config(enabled, snapshot_scope):
    config = {"providers": {"openrouter": {"enabled": enabled}}}
    if enabled:
        assert rp.resolve_runtime_provider(requested="openrouter", explicit_api_key="entry-key", config=config)["provider"] == "openrouter"
    else:
        with pytest.raises(ValueError, match="disabled"):
            rp.resolve_runtime_provider(requested="openrouter", explicit_api_key="entry-key", config=config)
    assert snapshot_scope == []


def test_manual_snapshot_same_url_keeps_named_identity_protocol_headers(snapshot_scope):
    from copy import deepcopy
    config = {
        "providers": {
            "alpha": {"base_url": "https://proxy.invalid/v1", "api_key": "provider-alpha-key",
                      "api_mode": "chat_completions", "extra_headers": {"X-Route": "alpha"}, "default_model": "old-alpha"},
            "beta": {"base_url": "https://proxy.invalid/v1", "api_key": "provider-beta-key",
                     "api_mode": "anthropic_messages", "extra_headers": {"X-Route": "beta"}, "default_model": "old-beta"},
        },
        "fallback_providers": [
            {"provider": "custom:alpha", "model": "target-alpha", "key_env": "FB_KEY"},
            {"provider": "custom:beta", "model": "target-beta", "api_key": "inline-entry-key"},
        ],
    }
    before = deepcopy(config)
    alpha, beta = _manual_snapshot_entry(config), _manual_snapshot_entry(config, 1)
    for result, name, mode, key in [
        (alpha, "alpha", "chat_completions", "scoped-entry-key"),
        (beta, "beta", "anthropic_messages", "inline-entry-key"),
    ]:
        assert result["requested_provider"] == "custom:" + name
        assert result["api_mode"] == mode
        assert result["api_key"] == key
        assert result["model"] == "target-" + name
        assert result["extra_headers"] == {"X-Route": name}
    assert config == before
    assert snapshot_scope == []


def test_snapshot_named_custom_selects_its_real_pool_when_live_config_changed(tmp_path, monkeypatch):
    from hermes_cli.auth import write_credential_pool

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "providers:\n  live-only:\n    api: https://live.invalid/v1\n"
    )
    write_credential_pool("snapshot-endpoint", [{
        "id": "snapshot-manual", "source": "manual", "auth_type": "api_key",
        "access_token": "synthetic-snapshot-pool-key", "base_url": "https://snapshot.invalid/v1",
        "label": "manual",
    }])
    config = {"providers": {"snapshot-endpoint": {
        "api": "https://snapshot.invalid/v1", "transport": "anthropic_messages",
        "extra_headers": {"X-Route": "snapshot"},
    }}}

    result = rp.resolve_runtime_provider(
        requested="custom:snapshot-endpoint", target_model="selected-model", config=config,
    )

    assert result["provider"] == "custom"
    assert result["requested_provider"] == "custom:snapshot-endpoint"
    assert result["api_key"] == "synthetic-snapshot-pool-key"
    assert result["source"] == "pool:snapshot-endpoint"
    assert result["base_url"] == "https://snapshot.invalid/v1"
    assert result["api_mode"] == "anthropic_messages"
    assert result["model"] == "selected-model"
    assert result["extra_headers"] == {"X-Route": "snapshot"}


def test_snapshot_bare_custom_selects_pool_owned_by_snapshot_endpoint(tmp_path, monkeypatch):
    from hermes_cli.auth import write_credential_pool

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "model:\n  provider: custom\n  base_url: https://live.invalid/v1\n"
        "providers:\n  live-only:\n    api: https://live.invalid/v1\n"
    )
    write_credential_pool("snapshot-endpoint", [{
        "id": "snapshot-manual", "source": "manual", "auth_type": "api_key",
        "access_token": "synthetic-snapshot-pool-key", "base_url": "https://snapshot.invalid/v1",
        "label": "manual",
    }])
    config = {
        "model": {
            "provider": "custom",
            "base_url": "https://snapshot.invalid/v1",
            "api_mode": "chat_completions",
        },
        "providers": {"snapshot-endpoint": {"api": "https://snapshot.invalid/v1"}},
    }

    result = rp.resolve_runtime_provider(requested="custom", target_model="selected-model", config=config)

    assert result["provider"] == "custom"
    assert result["api_key"] == "synthetic-snapshot-pool-key"
    assert result["source"] == "pool:snapshot-endpoint"
    assert result["base_url"] == "https://snapshot.invalid/v1"


def test_manual_snapshot_azure_target_beats_default(snapshot_scope):
    config = {
        "model": {"provider": "azure-foundry", "base_url": "https://snapshot.services.ai.azure.com",
                  "api_mode": "chat_completions", "default": "gpt-4o"},
        "fallback_providers": [{"provider": "azure-foundry", "model": "gpt-5.3-codex", "key_env": "FB_KEY"}],
    }
    result = _manual_snapshot_entry(config)
    assert result["api_mode"] == "codex_responses"
    assert result["base_url"] == config["model"]["base_url"]
    assert result["api_key"] == "scoped-entry-key"
    assert snapshot_scope == []


def test_snapshot_openrouter_terminal_uses_mirror_and_protocol(snapshot_scope):
    config = {"model": {"provider": "openrouter", "base_url": "https://mirror.invalid/v1", "api_mode": "codex_responses"}}
    result = rp.resolve_runtime_provider(requested="openrouter", explicit_api_key="entry-key", config=config)
    assert result["base_url"] == "https://mirror.invalid/v1"
    assert result["api_mode"] == "codex_responses"
    assert result["api_key"] == "entry-key"
    assert snapshot_scope == []


@pytest.mark.parametrize("pooled", [False, True])
def test_snapshot_empty_is_authoritative_through_pool_and_terminal(pooled, monkeypatch, snapshot_scope):
    if pooled:
        entry = SimpleNamespace(runtime_api_key="pool-key", base_url="https://openrouter.ai/api/v1", source="manual")
        monkeypatch.setattr(rp, "load_pool", lambda name: SimpleNamespace(
            provider="openrouter", has_credentials=lambda: True, select=lambda model=None: entry))
    result = rp.resolve_runtime_provider(requested="openrouter", config={})
    assert result["api_mode"] == "chat_completions"
    assert result["base_url"] == "https://openrouter.ai/api/v1"
    assert result["api_key"] == ("pool-key" if pooled else "")
    assert snapshot_scope == []


def test_snapshot_bedrock_section_and_target(monkeypatch, snapshot_scope):
    from tools import lazy_deps
    monkeypatch.setattr(lazy_deps, "ensure", lambda *args, **kwargs: True)
    import agent.bedrock_adapter as ba
    monkeypatch.setattr(ba, "resolve_aws_auth_env_var", lambda: "aws-sdk-default-chain")
    config = {"model": {"default": "amazon.nova-pro-v1:0"}, "bedrock": {
        "region": "eu-north-1", "guardrail": {"guardrail_identifier": "synthetic-guardrail", "guardrail_version": "1"}}}
    result = rp.resolve_runtime_provider(requested="bedrock", target_model="global.anthropic.claude-sonnet-4-6", config=config)
    assert result["region"] == "eu-north-1"
    assert result["api_mode"] == "anthropic_messages"
    assert result["guardrail_config"] == {"guardrailIdentifier": "synthetic-guardrail", "guardrailVersion": "1"}
    assert snapshot_scope == []


def test_snapshot_local_bypass_preserves_host_gated_noauth(monkeypatch, snapshot_scope):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-must-not-leak")
    config = {"model": {"provider": "auto", "default": "local-model", "base_url": "http://127.0.0.1:1234/v1"}}
    result = rp.resolve_runtime_provider(requested="auto", config=config)
    assert result["base_url"] == config["model"]["base_url"]
    assert result["api_key"] == "no-key-required"
    # Automatic auth discovery owns its reads; this contract scopes resolver config.
    assert all(any(path.replace("\\", "/").endswith("/hermes_cli/auth.py") and name == "resolve_provider"
                   for path, name in stack) for stack in snapshot_scope)


def test_snapshot_llamacpp_enabled_diagnostic(monkeypatch, snapshot_scope):
    from hermes_cli.local_runtime import endpoint
    monkeypatch.setattr(endpoint, "resolve_llamacpp_endpoint", lambda **kwargs: None)
    with pytest.raises(ValueError, match="isn't running"):
        rp.resolve_runtime_provider(requested="llamacpp", config={"local_runtime": {"enabled": True}})
    assert snapshot_scope == []


def test_snapshot_opencode_custom_default_model(snapshot_scope):
    config = {"model": {"default": "grok-4.5"}, "providers": {
        "relay": {"base_url": "https://opencode.ai/zen/go/v1", "api_key": "relay-key"}}}
    result = rp.resolve_runtime_provider(requested="custom:relay", config=config)
    assert result["api_mode"] == "codex_responses"
    assert snapshot_scope == []


def test_manual_snapshot_absent_key_preserves_native_noauth(monkeypatch, snapshot_scope):
    monkeypatch.setattr(rp, "resolve_api_key_provider_credentials", lambda provider: {
        "api_key": "lmstudio-noauth", "base_url": "http://localhost:1234/v1", "source": "default"})
    config = {"fallback_providers": [{"provider": "lmstudio", "model": "local-model"}]}
    result = _manual_snapshot_entry(config)
    assert result["provider"] == "lmstudio"
    assert result["api_key"] == "lmstudio-noauth"
    assert snapshot_scope == []


def test_snapshot_requested_provider_normalizes_without_mutating(snapshot_scope):
    from copy import deepcopy
    config = {"model": {"default": {"model": "gpt-4o", "provider": "openai"}}}
    before = deepcopy(config)
    assert rp.resolve_requested_provider(config=config) == "openai"
    assert rp.resolve_requested_provider(" OPENROUTER ", config=config) == "openrouter"
    assert rp.resolve_requested_provider(config={}) == "auto"
    assert config == before
    assert snapshot_scope == []
