"""Startup fallback must preserve its own wire on CLI, messaging/oneshot, cron and TUI/Desktop."""

import pytest

from hermes_cli import runtime_provider
from hermes_cli.auth import AuthError
from hermes_cli.config import atomic_config_write


@pytest.mark.parametrize("surface", ["shared", "cli", "cron", "tui"])
@pytest.mark.parametrize("field, value, expected", [
    ("api_mode", "codex_responses", "codex_responses"),
    ("api_mode", "anthropic_messages", "anthropic_messages"),
    ("api_mode", "chat_completions", "chat_completions"),
    ("transport", "responses", "codex_responses"),
    ("api_mode", "invalid-mode", "chat_completions"),
    ("api_mode", None, "chat_completions"),
])
def test_startup_fallback_transport(tmp_path, monkeypatch, surface, field, value, expected):
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "fixture-azure-key")
    entry = {"provider": "azure-foundry", "model": "custom-deployment",
             "base_url": "https://fixture.openai.azure.com/openai/v1", field: value}
    config = {"model": {"provider": "openai-codex", "default": "gpt-6-astra"},
              "fallback_providers": [entry]}
    runtime = _resolve_startup(config, surface, tmp_path, monkeypatch)
    assert runtime["api_mode"] == expected
    assert runtime["provider"] == "azure-foundry"
    expected_url = entry["base_url"].removesuffix("/v1") if expected == "anthropic_messages" else entry["base_url"]
    assert runtime["base_url"] == expected_url
    assert runtime["api_key"] == "fixture-azure-key"


def _resolve_startup(config, surface, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", config)
    entry = config["fallback_providers"][0]
    resolve = runtime_provider.resolve_runtime_provider

    def primary_unavailable(**kwargs):
        if kwargs.get("target_model") != entry["model"]:
            raise AuthError("fixture primary unavailable")
        return resolve(**kwargs)

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", primary_unavailable)
    if surface == "shared":
        runtime, selected = runtime_provider.resolve_runtime_with_fallback(config)
        assert selected["model"] == entry["model"]
    elif surface == "cli":
        from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
        monkeypatch.setattr("cli._cprint", lambda *a, **k: None)
        shell = CLIAgentSetupMixin.__new__(CLIAgentSetupMixin)
        shell._fallback_model = [entry]
        runtime = shell._resolve_fallback_runtime(AuthError("fixture primary unavailable"))
        assert shell.model == entry["model"]
    elif surface == "cron":
        from cron.scheduler import _CronJobConfig, _resolve_job_runtime
        jc = _CronJobConfig(cfg=config, model=config["model"]["default"], model_cfg=config["model"], cron_default_provider="")
        runtime, selected = _resolve_job_runtime({}, "fixture-job", jc)
        assert selected == entry["model"]
    else:
        from tui_gateway import server
        monkeypatch.setattr(server, "_hermes_home", tmp_path)
        resolution = server._resolve_runtime_with_fallback()
        runtime = resolution.runtime
        assert resolution.selected_model == entry["model"]
    return runtime


@pytest.mark.parametrize("surface", ["shared", "cli", "cron", "tui"])
@pytest.mark.parametrize("auth_mode", ["api_key", "entra_id"])
def test_azure_entry_mode_precedes_endpoint_normalization(tmp_path, monkeypatch, surface, auth_mode):
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "fixture-azure-key")
    token_provider = lambda: "fixture-token"
    monkeypatch.setattr("hermes_cli.runtime_provider_backends._azure_entra_credentials", lambda cfg: token_provider)
    entry = {"provider": "azure-foundry", "model": "custom-deployment",
             "base_url": "https://fixture.openai.azure.com/openai/v1", "api_mode": "codex_responses"}
    config = {"model": {"provider": "azure-foundry", "default": "claude-primary",
                        "base_url": "https://fixture.services.ai.azure.com/anthropic",
                        "api_mode": "anthropic_messages", "auth_mode": auth_mode}, "fallback_providers": [entry]}
    runtime = _resolve_startup(config, surface, tmp_path, monkeypatch)
    assert (runtime["api_mode"], runtime["base_url"]) == ("codex_responses", entry["base_url"])
    assert runtime["auth_mode"] == auth_mode
    assert runtime["api_key"] == (token_provider if auth_mode == "entra_id" else "fixture-azure-key")


@pytest.mark.parametrize("surface", ["shared", "cli", "cron", "tui"])
@pytest.mark.parametrize("mode, expected", [
    (None, "anthropic_messages"), ("responses", "codex_responses"), ("chat_completions", "chat_completions"),
])
def test_named_provider_transport_inheritance_and_override(tmp_path, monkeypatch, surface, mode, expected):
    monkeypatch.setenv("FIXTURE_RELAY_KEY", "fixture-relay-key")
    entry = {"provider": "fixture-relay", "model": "fallback-model"}
    if mode:
        entry["transport"] = mode
    base_url = "https://relay.example.test/v1"
    config = {"model": {"provider": "openai-codex", "default": "primary-model"},
              "providers": {"fixture-relay": {"base_url": base_url, "api_mode": "anthropic_messages",
                                              "key_env": "FIXTURE_RELAY_KEY"}},
              "fallback_providers": [entry]}
    runtime = _resolve_startup(config, surface, tmp_path, monkeypatch)
    assert (runtime["api_mode"], runtime["base_url"], runtime["api_key"]) == (expected, base_url, "fixture-relay-key")


@pytest.mark.parametrize("source", ["explicit", "pool", "env"])
@pytest.mark.parametrize("mode", ["codex_responses", "chat_completions", "anthropic_messages"])
def test_registry_entry_mode_precedes_endpoint_normalization(tmp_path, monkeypatch, source, mode):
    from agent.credential_pool import CredentialPool, PooledCredential
    base_url = "https://relay.example.test/v1"
    key = f"fixture-{source}-key"
    entry = {"provider": "opencode-go", "model": "qwen-fixture", "api_mode": mode}
    if source == "explicit":
        entry.update(api_key=key, base_url=base_url)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", key)
    credentials = [PooledCredential(
        provider="opencode-go", id="fixture", label="fixture", auth_type="api_key", priority=0,
        source="fixture-pool", access_token=key, base_url=base_url)] if source == "pool" else []
    pool = CredentialPool("opencode-go", credentials)
    monkeypatch.setattr(runtime_provider, "load_pool", lambda provider: pool)
    config = {"model": {"provider": "opencode-go", "default": "primary-model",
                        "base_url": base_url, "api_mode": "anthropic_messages"},
              "fallback_providers": [entry]}
    runtime = _resolve_startup(config, "shared", tmp_path, monkeypatch)
    expected_url = base_url.removesuffix("/v1") if mode == "anthropic_messages" else base_url
    assert (runtime["api_mode"], runtime["base_url"], runtime["api_key"]) == (mode, expected_url, key)
    if source == "pool":
        assert runtime["credential_pool"] is pool
