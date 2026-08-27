"""An external-process provider can ship from outside this tree.

An ACP agent CLI is not an HTTP endpoint: it is a subprocess Hermes drives over
stdio. Everything about launching it used to be spelled out for one vendor —
the known-provider gate, the binary, the argv, the env var names. A provider
registered from ``~/.hermes/plugins/model-providers/`` or a pip entry point
therefore could not be reached at all: ``hermes -m <it>`` died with "Unknown
provider" long before any client was built.

This registers a provider the way a standalone package does — before importing
any ``hermes_cli`` module, exactly as plugin discovery does — and walks the real
resolution path end to end. ``copilot-acp`` is asserted alongside it at every
step, so the generalisation cannot quietly change the in-tree provider.
"""

from __future__ import annotations

import os
import stat
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from providers import register_provider  # noqa: E402
from providers.base import ProviderProfile  # noqa: E402


class _AcmeClient:
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")


class _AcmeACPProfile(ProviderProfile):
    def create_client(self, **kwargs):
        return _AcmeClient(**kwargs)

    def fetch_models(self, **kwargs):
        return None


# Registered at import time — the same moment a plugin's ``__init__.py`` or a
# pip entry point runs, i.e. before hermes_cli.auth builds PROVIDER_REGISTRY.
register_provider(
    _AcmeACPProfile(
        name="acme-acp",
        aliases=("acme",),
        display_name="Acme ACP",
        api_mode="chat_completions",
        base_url="acp://acme",
        auth_type="external_process",
        process_command="acme-cli",
        process_args=("--acp",),
        process_command_env_vars=("ACME_CLI_PATH",),
        process_args_env_var="ACME_ACP_ARGS",
    )
)


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    """Put stub `acme-cli` and `copilot` binaries on PATH."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("acme-cli", "copilot"):
        exe = bindir / name
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return bindir


# ── the known-provider gate ──────────────────────────────────────────────────


def test_the_provider_is_absorbed_into_the_cli_registry():
    """Without this the resolver raises "Unknown provider" before anything else."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    assert "acme-acp" in PROVIDER_REGISTRY
    assert PROVIDER_REGISTRY["acme-acp"].auth_type == "external_process"
    assert PROVIDER_REGISTRY["acme-acp"].inference_base_url == "acp://acme"
    assert PROVIDER_REGISTRY["acme-acp"].name == "Acme ACP"
    # Aliases resolve to the same config.
    assert PROVIDER_REGISTRY["acme"] is PROVIDER_REGISTRY["acme-acp"]
    # The in-tree one is declared explicitly and must not be shadowed.
    assert PROVIDER_REGISTRY["copilot-acp"].auth_type == "external_process"


def test_resolve_provider_accepts_the_name_and_its_alias():
    from hermes_cli.auth import resolve_provider

    assert resolve_provider("acme-acp") == "acme-acp"
    assert resolve_provider("acme") == "acme-acp"


# ── launching the CLI ────────────────────────────────────────────────────────


def test_credentials_come_from_the_profile(fake_cli):
    from hermes_cli.auth import resolve_external_process_provider_credentials

    creds = resolve_external_process_provider_credentials("acme-acp")
    assert creds["command"] == str(fake_cli / "acme-cli")
    assert creds["args"] == ["--acp"]
    assert creds["base_url"] == "acp://acme"
    # Placeholder credential keyed on the provider, not a borrowed literal.
    assert creds["api_key"] == "acme-acp"


def test_copilot_launch_details_are_unchanged(fake_cli):
    """These moved from hardcoded resolver code into copilot-acp's profile."""
    from hermes_cli.auth import resolve_external_process_provider_credentials

    creds = resolve_external_process_provider_credentials("copilot-acp")
    assert creds["command"] == str(fake_cli / "copilot")
    assert creds["args"] == ["--acp", "--stdio"]
    assert creds["api_key"] == "copilot-acp"
    assert creds["base_url"] == "acp://copilot"


@pytest.mark.parametrize(
    "provider,env_var,binary",
    [
        ("acme-acp", "ACME_CLI_PATH", "acme-cli"),
        ("copilot-acp", "HERMES_COPILOT_ACP_COMMAND", "copilot"),
        ("copilot-acp", "COPILOT_CLI_PATH", "copilot"),
    ],
)
def test_the_binary_can_be_overridden_by_env(fake_cli, monkeypatch, provider, env_var, binary):
    from hermes_cli.auth import resolve_external_process_provider_credentials

    custom = fake_cli / f"custom-{binary}"
    custom.write_text("#!/bin/sh\nexit 0\n")
    custom.chmod(custom.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv(env_var, str(custom))
    creds = resolve_external_process_provider_credentials(provider)
    assert creds["command"] == str(custom)


@pytest.mark.parametrize(
    "provider,env_var",
    [("acme-acp", "ACME_ACP_ARGS"), ("copilot-acp", "HERMES_COPILOT_ACP_ARGS")],
)
def test_the_args_can_be_overridden_by_env(fake_cli, monkeypatch, provider, env_var):
    from hermes_cli.auth import resolve_external_process_provider_credentials

    monkeypatch.setenv(env_var, "--acp=true --verbose")
    assert resolve_external_process_provider_credentials(provider)["args"] == [
        "--acp=true",
        "--verbose",
    ]


def test_a_missing_binary_names_the_provider_and_its_env_override(monkeypatch, tmp_path):
    from hermes_cli.auth import AuthError, resolve_external_process_provider_credentials

    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(AuthError) as excinfo:
        resolve_external_process_provider_credentials("acme-acp")
    message = str(excinfo.value)
    assert "acme-acp" in message
    assert "ACME_CLI_PATH" in message


def test_a_non_external_process_provider_is_still_rejected():
    from hermes_cli.auth import AuthError, resolve_external_process_provider_credentials

    with pytest.raises(AuthError):
        resolve_external_process_provider_credentials("openai-api")


# ── runtime resolution ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "requested,expected_base_url",
    [
        ("acme-acp", "acp://acme"),
        ("acme", "acp://acme"),
        ("copilot-acp", "acp://copilot"),
    ],
)
def test_runtime_resolution_reaches_the_external_process_branch(
    fake_cli, requested, expected_base_url
):
    """Previously keyed on the literal "copilot-acp", so anything else fell
    through to the OpenRouter default instead of its own runtime."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested=requested, target_model=requested)
    assert runtime["base_url"] == expected_base_url
    assert runtime["api_mode"] == "chat_completions"
    assert runtime["source"] == "process"
    assert runtime["command"]


def test_the_client_is_built_from_the_profile(fake_cli):
    from types import SimpleNamespace

    from agent.agent_runtime_helpers import _provider_supplied_client
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested="acme-acp", target_model="acme-acp")
    agent = SimpleNamespace(provider=runtime["provider"], _client_log_context=lambda: "")
    client = _provider_supplied_client(
        agent, {"api_key": runtime["api_key"], "base_url": runtime["base_url"]}
    )
    assert isinstance(client, _AcmeClient)
    assert client.base_url == "acp://acme"
