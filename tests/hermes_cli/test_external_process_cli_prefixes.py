"""A Claude Code CLI outside PATH is found by every core check, not only by the Anthropic adapter.

A backend started by a macOS LaunchAgent or spawned by Hermes Desktop inherits a bare PATH that
carries none of Claude Code's install prefixes (``~/.local/bin``, ``~/.claude/local``, ...).
``agent.anthropic_adapter`` probes those prefixes after PATH; ``_external_process_spec`` and
``run_oauth_setup_token`` asked PATH only, so a provider whose ``process_command`` is ``claude``
was reported missing ("Could not find the ... CLI command 'claude'") on a machine where the
adapter could run it. Reported against hermes-plugin-claude-subscription-directsdk#32.
"""

from __future__ import annotations

import subprocess

import pytest

from providers import register_provider
from providers.base import ProviderProfile

register_provider(
    ProviderProfile(
        name="claude-cli-acp-test",
        display_name="Claude CLI (test)",
        base_url="process://claude-cli-acp-test",
        auth_type="external_process",
        process_command="claude",
    )
)


def _install(directory) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / "claude"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    return str(exe)


@pytest.fixture
def service_launch(tmp_path, monkeypatch):
    """The user's home holds ``~/.local/bin/claude``; the process PATH carries no install prefix."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    return _install(tmp_path / ".local" / "bin")


@pytest.mark.platforms("posix")  # HOME-relative prefixes and the exec bit are the POSIX layout
def test_external_process_provider_finds_claude_in_an_install_prefix(service_launch, tmp_path, monkeypatch):
    from hermes_cli.auth import get_external_process_provider_status, resolve_external_process_provider_credentials

    assert resolve_external_process_provider_credentials("claude-cli-acp-test")["command"] == service_launch
    status = get_external_process_provider_status("claude-cli-acp-test")
    assert (status["configured"], status["resolved_command"]) == (True, service_launch)

    # A PATH hit is still what runs: the prefix probe only answers a PATH miss.
    on_path = _install(tmp_path / "path-bin")
    monkeypatch.setenv("PATH", str(tmp_path / "path-bin"))
    assert resolve_external_process_provider_credentials("claude-cli-acp-test")["command"] == on_path


@pytest.mark.platforms("posix")
def test_setup_token_runs_claude_from_an_install_prefix(service_launch, monkeypatch):
    import agent.anthropic_credentials as creds

    ran: list[list[str]] = []
    monkeypatch.setattr(creds.subprocess, "run", lambda argv, *a, **k: ran.append(argv) or subprocess.CompletedProcess(argv, 0))
    monkeypatch.setattr(creds, "read_claude_code_credentials", lambda: None)

    creds.run_oauth_setup_token()
    assert ran == [[service_launch, "setup-token"]]
