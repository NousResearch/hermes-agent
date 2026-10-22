"""An external-process (ACP) provider can ship from outside this tree.

The known-provider gate, the binary, the argv and the env var names used to be
spelled out for one vendor, so a profile registered from a plugin died with
"Unknown provider" before any client was built. This registers a provider the
way a standalone package does — before importing ``hermes_cli`` — and walks the
real resolution path, asserting ``copilot-acp`` is unchanged alongside it.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from providers import register_provider
from providers.base import ProviderProfile


class _AcmeACPProfile(ProviderProfile):
    def create_client(self, **kwargs):
        return ("acme-client", kwargs)

    def fetch_models(self, **kwargs):
        return None


register_provider(
    _AcmeACPProfile(
        name="acme-acp",
        aliases=("acme",),
        display_name="Acme ACP",
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
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("acme-cli", "copilot", "custom-acme"):
        exe = bindir / (name + ".exe" if os.name == "nt" else name)
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return bindir


def test_an_out_of_tree_external_process_provider_resolves_end_to_end(fake_cli, monkeypatch):
    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_external_process_provider_credentials, resolve_provider
    from hermes_cli.runtime_provider import resolve_runtime_provider

    assert PROVIDER_REGISTRY["acme"] is PROVIDER_REGISTRY["acme-acp"]
    assert PROVIDER_REGISTRY["acme-acp"].auth_type == "external_process"
    assert resolve_provider("acme") == "acme-acp"

    creds = resolve_external_process_provider_credentials("acme-acp")
    assert Path(creds["command"]) == fake_cli / ("acme-cli.exe" if os.name == "nt" else "acme-cli")
    assert (creds["args"], creds["api_key"]) == (["--acp"], "acme-acp")

    monkeypatch.setenv("ACME_CLI_PATH", str(fake_cli / ("custom-acme.exe" if os.name == "nt" else "custom-acme")))
    monkeypatch.setenv("ACME_ACP_ARGS", "--acp=true --verbose")
    creds = resolve_external_process_provider_credentials("acme-acp")
    assert (creds["command"], creds["args"]) == (str(fake_cli / ("custom-acme.exe" if os.name == "nt" else "custom-acme")), ["--acp=true", "--verbose"])

    runtime = resolve_runtime_provider(requested="acme", target_model="acme")
    assert (runtime["provider"], runtime["base_url"], runtime["source"]) == ("acme-acp", "acp://acme", "process")


def test_copilot_acp_launch_details_are_unchanged(fake_cli, monkeypatch):
    from hermes_cli.auth import resolve_external_process_provider_credentials
    from hermes_cli.runtime_provider import resolve_runtime_provider

    creds = resolve_external_process_provider_credentials("copilot-acp")
    assert Path(creds["command"]) == fake_cli / ("copilot.exe" if os.name == "nt" else "copilot")
    assert (creds["args"], creds["api_key"], creds["base_url"]) == (["--acp", "--stdio"], "copilot-acp", "acp://copilot")

    monkeypatch.setenv("COPILOT_CLI_PATH", str(fake_cli / ("custom-acme.exe" if os.name == "nt" else "custom-acme")))
    assert resolve_external_process_provider_credentials("copilot-acp")["command"] == str(fake_cli / ("custom-acme.exe" if os.name == "nt" else "custom-acme"))
    assert resolve_runtime_provider(requested="copilot-acp", target_model="x")["base_url"] == "acp://copilot"


@pytest.mark.parametrize("requested", ["acme-acp", "auto"])
def test_missing_external_process_cli_does_not_resolve_paid_fallback(tmp_path, monkeypatch, requested):
    """A real missing CLI must surface its install hint before resolving a paid fallback."""
    from hermes_cli import runtime_provider
    from hermes_cli.auth import AuthError

    missing_cli = str(tmp_path / "missing-acme-cli")
    monkeypatch.setenv("ACME_CLI_PATH", missing_cli)
    config = {
        "model": {"provider": "acme-acp", "default": "acme-model"},
        "fallback_providers": [{
            "provider": "openrouter", "model": "fallback-model", "api_key": "fallback-key",
        }],
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr(runtime_provider, "load_pool", lambda _provider: None)
    original_resolve = runtime_provider.resolve_runtime_provider
    requested_providers = []

    def record_resolve(**kwargs):
        requested_providers.append(kwargs.get("requested"))
        return original_resolve(**kwargs)

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", record_resolve)
    with runtime_provider.frozen_runtime_config(config):
        with pytest.raises(AuthError) as exc_info:
            runtime_provider.resolve_runtime_with_fallback(
                config, requested=requested, target_model="acme-model",
            )

    assert exc_info.value.code == "missing_external_process_cli"
    assert missing_cli in str(exc_info.value)
    assert "Install it" in str(exc_info.value)
    assert requested_providers == [requested]
