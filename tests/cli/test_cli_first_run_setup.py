"""First-run onboarding routing for a completely unconfigured install.

Regression tests for the "keyless first run boots into a broken chat" bug:
a fresh install with zero providers accepted a message, spun for ~30s, then
failed with a provider-specific error ("Set OPENROUTER_API_KEY") the user
never chose, and never offered setup.

Covers:
- ``_runtime_credentials_ready()`` silent probe semantics
- ``_offer_first_run_setup()`` routing into the shared provider picker
- the provider-aware (non-OpenRouter-specific) empty-key error message
"""

import importlib
import sys
import types

import pytest

from hermes_cli.auth import AuthError


def _reset_modules(prefixes: tuple[str, ...]):
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _restore_cli_and_tool_modules():
    prefixes = ("tools", "cli", "run_agent")
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == p or name.startswith(p + ".") for p in prefixes)
    }
    try:
        yield
    finally:
        _reset_modules(prefixes)
        sys.modules.update(original_modules)


def _import_cli():
    for name in list(sys.modules):
        if name == "cli" or name == "run_agent" or name == "tools" or name.startswith("tools."):
            sys.modules.pop(name, None)
    if "firecrawl" not in sys.modules:
        sys.modules["firecrawl"] = types.SimpleNamespace(Firecrawl=object)
    return importlib.import_module("cli")


def _make_shell(cli, monkeypatch):
    shell = cli.HermesCLI(compact=True, max_turns=1)
    return shell


# ---------------------------------------------------------------------------
# _runtime_credentials_ready
# ---------------------------------------------------------------------------


def test_credentials_ready_false_when_no_provider(monkeypatch):
    cli = _import_cli()

    def _raise(**kwargs):
        raise AuthError("No inference provider configured.", code="no_provider_configured")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _raise)
    shell = _make_shell(cli, monkeypatch)
    assert shell._runtime_credentials_ready() is False


def test_credentials_ready_false_on_empty_openrouter_key(monkeypatch):
    """The exact broken-chat state: provider resolves but api_key is empty."""
    cli = _import_cli()

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {
            "provider": "openrouter",
            "api_key": "",
            "base_url": "https://openrouter.ai/api/v1",
            "source": "env/config",
        },
    )
    shell = _make_shell(cli, monkeypatch)
    assert shell._runtime_credentials_ready() is False


def test_credentials_ready_true_with_key(monkeypatch):
    cli = _import_cli()

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {
            "provider": "openrouter",
            "api_key": "sk-test",
            "base_url": "https://openrouter.ai/api/v1",
            "source": "env/config",
        },
    )
    shell = _make_shell(cli, monkeypatch)
    assert shell._runtime_credentials_ready() is True


def test_credentials_ready_true_for_external_agent_loop(monkeypatch):
    """External runtimes own auth and intentionally have no HTTP base URL."""
    cli = _import_cli()

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {
            "provider": "claude-agent-sdk",
            "api_mode": "claude_agent_sdk",
            "api_key": "claude-subscription-oauth",
            "base_url": "",
            "source": "claude-agent-sdk",
        },
    )
    shell = _make_shell(cli, monkeypatch)
    assert shell._runtime_credentials_ready() is True


def test_credentials_ready_false_for_http_provider_without_base_url(monkeypatch):
    """The external-loop exemption must not weaken normal HTTP validation."""
    cli = _import_cli()

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "api_key": "sk-test",
            "base_url": "",
            "source": "env/config",
        },
    )
    shell = _make_shell(cli, monkeypatch)
    assert shell._runtime_credentials_ready() is False


def test_credentials_ready_true_for_keyless_local_endpoint(monkeypatch):
    """ollama/llama.cpp-style custom endpoints need no key."""
    cli = _import_cli()

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {
            "provider": "custom",
            "api_key": "",
            "base_url": "http://localhost:11434/v1",
            "source": "custom_provider",
        },
    )
    shell = _make_shell(cli, monkeypatch)
    assert shell._runtime_credentials_ready() is True


def test_credentials_ready_true_for_callable_bearer_provider(monkeypatch):
    cli = _import_cli()

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {
            "provider": "azure-foundry",
            "api_key": lambda: "tok",
            "base_url": "https://foundry.example/v1",
            "source": "entra",
        },
    )
    shell = _make_shell(cli, monkeypatch)
    assert shell._runtime_credentials_ready() is True


def test_credentials_ready_never_prints(monkeypatch, capsys):
    cli = _import_cli()

    def _raise(**kwargs):
        raise AuthError("No inference provider configured.", code="no_provider_configured")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _raise)
    shell = _make_shell(cli, monkeypatch)
    capsys.readouterr()  # drain construction output
    shell._runtime_credentials_ready()
    out = capsys.readouterr()
    assert out.out == ""


# ---------------------------------------------------------------------------
# _offer_first_run_setup
# ---------------------------------------------------------------------------


def test_offer_first_run_setup_routes_into_shared_picker(monkeypatch):
    cli = _import_cli()
    shell = _make_shell(cli, monkeypatch)

    picker_calls = {"count": 0}

    def _fake_picker():
        picker_calls["count"] += 1

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", _fake_picker)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "y")
    # After the picker "runs", config has a provider and creds resolve.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider": "nous", "default": "hermes-4-405b"}},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {
            "provider": "nous",
            "api_key": "portal-token",
            "base_url": "https://inference-api.nousresearch.com/v1",
            "source": "oauth",
        },
    )

    assert shell._offer_first_run_setup() is True
    assert picker_calls["count"] == 1
    assert shell.requested_provider == "nous"
    assert shell.model == "hermes-4-405b"
    # Agent must be rebuilt with the new credentials on next use.
    assert shell.agent is None


def test_offer_first_run_setup_declined(monkeypatch):
    cli = _import_cli()
    shell = _make_shell(cli, monkeypatch)

    def _fail_picker():
        raise AssertionError("picker must not run when declined")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", _fail_picker)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")
    assert shell._offer_first_run_setup() is False


def test_offer_first_run_setup_picker_cancel_is_graceful(monkeypatch):
    cli = _import_cli()
    shell = _make_shell(cli, monkeypatch)

    def _cancel_picker():
        raise KeyboardInterrupt()

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", _cancel_picker)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    # Empty answer defaults to yes -> picker runs -> cancels -> False, no raise.
    assert shell._offer_first_run_setup() is False


# ---------------------------------------------------------------------------
# Provider-aware empty-key error (replaces the OpenRouter-specific one)
# ---------------------------------------------------------------------------


def test_empty_key_error_names_actual_provider(monkeypatch, capsys):
    cli = _import_cli()

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {
            "provider": "fireworks",
            "api_key": "",
            "base_url": "https://api.fireworks.ai/inference/v1/extra",
            "source": "env/config",
        },
    )
    shell = _make_shell(cli, monkeypatch)
    # A custom base_url would get the no-key placeholder; force the
    # openrouter-shaped branch by pointing base_url at openrouter.
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {
            "provider": "fireworks",
            "api_key": "",
            "base_url": "https://openrouter.ai/api/v1",
            "source": "env/config",
        },
    )
    capsys.readouterr()
    assert shell._ensure_runtime_credentials() is False
    out = capsys.readouterr().out
    assert "fireworks" in out
    assert "OPENROUTER_API_KEY" not in out
    assert "hermes model" in out or "hermes setup" in out
