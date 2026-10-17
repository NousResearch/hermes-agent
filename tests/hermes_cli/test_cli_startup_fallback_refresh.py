"""A selected startup fallback keeps its route while credentials remain refreshable."""

from types import SimpleNamespace

import pytest

from hermes_cli import runtime_provider
from hermes_cli.auth import AuthError
from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
from hermes_cli.config import atomic_config_write


@pytest.fixture
def fallback_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("FIXTURE_FALLBACK_KEY", "fallback-key")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "ambient-key")
    monkeypatch.setattr("cli._cprint", lambda *args, **kwargs: None)
    entry = {"provider": "azure-foundry", "model": "custom-deployment",
             "base_url": "https://fallback.openai.azure.com/openai/v1",
             "api_mode": "codex_responses", "key_env": "FIXTURE_FALLBACK_KEY"}
    atomic_config_write(tmp_path / "config.yaml", {
        "model": {"provider": "openai-codex", "default": "primary-model"},
        "fallback_providers": [entry],
    })
    resolve = runtime_provider.resolve_runtime_provider

    def unavailable_primary(**kwargs):
        if kwargs.get("requested") == "openai-codex":
            raise AuthError("fixture primary unavailable")
        return resolve(**kwargs)

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", unavailable_primary)
    from cli import HermesCLI
    shell = HermesCLI.__new__(HermesCLI)
    shell.__dict__.update(
        _fallback_model=[entry], requested_provider="openai-codex", model="primary-model",
        _explicit_api_key="primary-key", _explicit_base_url="https://primary.example.test/v1",
        provider="openai-codex", api_mode="codex_responses", acp_command=None, acp_args=[],
        api_key=None, base_url=None, agent=None, _explicit_reasoning_config={"effort": "low"},
        _model_is_default=False, _console_print=lambda *args, **kwargs: None,
    )
    return shell, entry


def test_refresh_retains_entry_transport_endpoint_and_rotates_key(fallback_shell, monkeypatch):
    shell, entry = fallback_shell
    assert shell._ensure_runtime_credentials()
    assert (shell.model, shell.api_mode, shell.base_url, shell.api_key) == (
        entry["model"], "codex_responses", entry["base_url"], "fallback-key")
    monkeypatch.setenv("FIXTURE_FALLBACK_KEY", "rotated-key")
    assert shell._probe_runtime_credentials() == (True, None)
    assert shell._ensure_runtime_credentials()
    assert (shell.api_mode, shell.base_url, shell.api_key) == (
        "codex_responses", entry["base_url"], "rotated-key")


def test_model_normalization_keeps_fallback_transport_on_refresh(fallback_shell, monkeypatch):
    shell, entry = fallback_shell
    entry.update(provider="opencode-go", model="opencode-go/qwen-fixture",
                 base_url="https://fallback.example.test/v1")
    assert shell._ensure_runtime_credentials()
    assert (shell.model, shell.api_mode, shell.base_url) == (
        "qwen-fixture", "codex_responses", entry["base_url"])
    monkeypatch.setenv("FIXTURE_FALLBACK_KEY", "rotated-key")
    assert shell._ensure_runtime_credentials()
    assert (shell.api_mode, shell.base_url, shell.api_key) == (
        "codex_responses", entry["base_url"], "rotated-key")


@pytest.mark.parametrize("accepted", [True, False])
def test_explicit_switch_replaces_fallback_and_cancel_restores_it(fallback_shell, accepted):
    shell, entry = fallback_shell
    assert shell._ensure_runtime_credentials()
    shell.agent = SimpleNamespace(switch_model=lambda **kwargs: accepted)
    result = SimpleNamespace(
        new_model=entry["model"], target_provider=entry["provider"],
        api_key="manual-key", base_url="https://manual.openai.azure.com/openai/v1", api_mode="chat_completions")
    assert CLIModelSwitchMixin._stage_and_swap_model(shell, result, shell.model) is accepted
    assert shell._ensure_runtime_credentials()
    expected = ("chat_completions", result.base_url, "manual-key") if accepted else (
        "codex_responses", entry["base_url"], "fallback-key")
    assert (shell.api_mode, shell.base_url, shell.api_key) == expected


@pytest.mark.parametrize("reset_success", [True, False])
def test_new_session_refresh_follows_the_adopted_route(fallback_shell, monkeypatch, reset_success):
    import cli
    from hermes_cli.model_switch import ModelSwitchResult

    shell, entry = fallback_shell
    assert shell._ensure_runtime_credentials()
    shell.__dict__.update(
        conversation_history=[], session_id="old", _session_db=None,
        _pending_title=None, _resumed=False, _new_session_id=lambda now: "new")
    default_route = ModelSwitchResult(
        success=reset_success, new_model="primary-model", target_provider=entry["provider"],
        api_key="renewed-default-key", base_url="https://primary.openai.azure.com/openai/v1",
        api_mode="chat_completions")
    monkeypatch.setitem(cli.CLI_CONFIG, "model", {
        "provider": default_route.target_provider, "default": default_route.new_model})
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kwargs: default_route)

    shell.new_session(silent=True)
    monkeypatch.setenv("FIXTURE_FALLBACK_KEY", "rotated-fallback-key")
    assert shell._ensure_runtime_credentials()
    expected = (
        default_route.new_model, default_route.api_mode, default_route.base_url, default_route.api_key
    ) if reset_success else (
        entry["model"], entry["api_mode"], entry["base_url"], "rotated-fallback-key")
    assert (shell.model, shell.api_mode, shell.base_url, shell.api_key) == expected
