"""Claude Code version detection must see the CLI from a GUI-launched process.

Anthropic gates OAuth model access on the Claude Code version in the user-agent;
reporting one below the gate returns HTTP 400. GUI launches (the desktop app,
macOS LaunchAgents) inherit a bare ``/usr/bin:/bin:/usr/sbin:/sbin`` PATH, so a
bare-name lookup found nothing and detection fell back to the stale constant.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import agent.anthropic_adapter as adapter
from agent.anthropic_adapter import (
    _CLAUDE_CODE_VERSION_FALLBACK,
    _detect_claude_code_version,
    _get_claude_code_version,
)


def _install(directory, name: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return str(path)


def _cli_versions(monkeypatch, versions: dict[str, str]):
    """Only the given executable paths answer ``--version``; anything else is a bug."""

    def runner(cmd, *args, **kwargs):
        assert cmd[0] in versions, f"probed unexpected path: {cmd[0]}"
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=f"{versions[cmd[0]]} (Claude Code)", stderr=""
        )

    monkeypatch.setattr(adapter.subprocess, "run", runner)


def test_gui_launch_reports_installed_version_not_fallback(monkeypatch, tmp_path):
    """Bare GUI PATH, CLI only under an install prefix → real version, not the fallback."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.setattr(adapter, "_CLAUDE_CODE_PREFIXES", (str(tmp_path / "prefix"),))
    installed = _install(tmp_path / "prefix", "claude")
    _cli_versions(monkeypatch, {installed: "2.1.276"})

    assert _detect_claude_code_version() == "2.1.276" != _CLAUDE_CODE_VERSION_FALLBACK


@pytest.mark.skipif(sys.platform == "win32", reason="PATH lookup of an extensionless shim is POSIX-only")
def test_path_hit_is_probed_first(monkeypatch, tmp_path):
    """A PATH ``claude-code`` beats a stale prefix ``claude``: every PATH hit precedes every prefix."""
    on_path = _install(tmp_path / "path", "claude-code")
    stale = _install(tmp_path / "prefix", "claude")
    monkeypatch.setenv("PATH", str(tmp_path / "path"))
    monkeypatch.setattr(adapter, "_CLAUDE_CODE_PREFIXES", (str(tmp_path / "prefix"),))
    _cli_versions(monkeypatch, {on_path: "2.1.400", stale: "2.1.100"})

    assert _detect_claude_code_version() == "2.1.400"


def test_oauth_identity_clamps_stale_installed_version(monkeypatch):
    monkeypatch.setattr(adapter, "_claude_code_version_cache", None)
    monkeypatch.setattr(adapter, "_detect_claude_code_version", lambda: "2.1.270")

    assert _get_claude_code_version() == "2.1.280"
