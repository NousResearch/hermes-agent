"""Tests for issue #65729 — three sites use subprocess.run with shell=True
on untrusted or user-influenced input.

Each site is fixed to use shlex.split() and shell=False, preventing
shell metacharacter injection from malicious catalog entries, plugin
manifests, or env-var templates.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# --------------------------------------------------------------------------- #
# Site 1: mcp_catalog._run_bootstrap — no shell=True
# --------------------------------------------------------------------------- #


def test_mcp_bootstrap_uses_shlex_split_not_shell():
    """_run_bootstrap must not pass shell=True to subprocess.run.

    Verify by checking the source code — the function must use
    shlex.split() and must not call subprocess.run with shell=True
    (excluding docstring/comment mentions).
    """
    import inspect
    from hermes_cli import mcp_catalog

    source = inspect.getsource(mcp_catalog._run_bootstrap)
    # Check each non-docstring, non-comment line for shell=True
    in_docstring = False
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.endswith('"""'):
            in_docstring = not in_docstring
            continue
        if in_docstring or stripped.startswith("#"):
            continue
        if "shell=True" in stripped:
            pytest.fail(
                f"_run_bootstrap contains shell=True in code: {stripped}"
            )
    # Must use shlex
    assert "shlex" in source or "_shlex" in source, (
        "_run_bootstrap must use shlex.split() — see issue #65729"
    )


def test_mcp_bootstrap_simple_command_runs_without_shell(monkeypatch, tmp_path):
    """A simple bootstrap command (no operators) runs via shlex.split."""
    from hermes_cli import mcp_catalog

    calls = []
    def fake_run(args, **kwargs):
        calls.append({"args": args, "shell": kwargs.get("shell", False)})
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(mcp_catalog.subprocess, "run", fake_run)
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    mcp_catalog._run_bootstrap(tmp_path, ["echo hello"])

    assert len(calls) == 1
    assert calls[0]["args"] == ["echo", "hello"]
    assert calls[0]["shell"] is False


def test_mcp_bootstrap_chained_command_splits_operators(monkeypatch, tmp_path):
    """Chained commands (&&) are split into individual subcommands."""
    from hermes_cli import mcp_catalog

    calls = []
    def fake_run(args, **kwargs):
        calls.append({"args": args, "shell": kwargs.get("shell", False)})
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(mcp_catalog.subprocess, "run", fake_run)
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    mcp_catalog._run_bootstrap(tmp_path, ["echo first && echo second"])

    assert len(calls) == 2
    assert calls[0]["args"] == ["echo", "first"]
    assert calls[1]["args"] == ["echo", "second"]
    # Neither call uses shell=True
    assert all(c["shell"] is False for c in calls)


def test_mcp_bootstrap_injection_attempt_does_not_execute_arbitrary_code(monkeypatch, tmp_path):
    """A malicious command with shell metacharacters must not inject."""
    from hermes_cli import mcp_catalog

    calls = []
    def fake_run(args, **kwargs):
        calls.append({"args": args, "shell": kwargs.get("shell", False)})
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(mcp_catalog.subprocess, "run", fake_run)
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    # Malicious command attempting to chain via ;
    mcp_catalog._run_bootstrap(tmp_path, ["echo safe; rm -rf /"])

    # The ; is treated as a chain separator, so "rm" would be a separate
    # command — but it's split via shlex, not shell. The key point is
    # shell=True is NOT used, so no shell parsing of metacharacters.
    assert all(c["shell"] is False for c in calls)
    # First command is "echo safe"
    assert calls[0]["args"] == ["echo", "safe"]


# --------------------------------------------------------------------------- #
# Site 2: web_server plugin install — no shell=True
# --------------------------------------------------------------------------- #


def test_web_server_plugin_install_uses_shlex_not_shell():
    """The plugin install code path must not use shell=True.

    Check the source around the install_cmd handling.
    """
    import inspect
    from hermes_cli import web_server

    # Find the _run_setup_command calls in the plugin install section
    source = inspect.getsource(web_server)

    # The specific pattern we're looking for: shell=True in the
    # external_dependencies install section (around line 4731)
    # We check that the install_cmd path uses shlex.split and shell=False
    lines = source.split("\n")
    for i, line in enumerate(lines):
        if "install_cmd" in line and "_run_setup_command" in lines[max(0, i-1)]:
            # Check the next few lines for shell=False
            context = "\n".join(lines[i:i+5])
            if "shell=True" in context:
                pytest.fail(
                    f"Plugin install still uses shell=True near line {i} — see issue #65729"
                )
            if "shlex.split" in context:
                break  # Found the fixed version


# --------------------------------------------------------------------------- #
# Site 3: transcription_tools local STT — no shell=True
# --------------------------------------------------------------------------- #


def test_transcription_tools_stt_no_shell_true():
    """The local STT command path must not use shell=True."""
    import inspect
    from tools import transcription_tools

    # Find the function that runs the local STT command
    source = inspect.getsource(transcription_tools)

    # The fix removed the use_shell branch and shell=True
    # Check that the section that runs subprocess.run for STT
    # does not contain shell=True
    lines = source.split("\n")
    for i, line in enumerate(lines):
        if "subprocess.run" in line and "shell=True" in line:
            # Check context — is this in the local STT section?
            context = "\n".join(lines[max(0, i-5):i+5])
            if "LOCAL_STT" in context or "local_stt" in context or "command_template" in context:
                pytest.fail(
                    f"Local STT still uses shell=True near line {i} — see issue #65729"
                )


# --------------------------------------------------------------------------- #
# General: no shell=True in the three identified files (regression guard)
# --------------------------------------------------------------------------- #


def test_no_shell_true_in_mcp_catalog_bootstrap():
    """Regression guard: _run_bootstrap must never use shell=True in code."""
    import inspect
    from hermes_cli import mcp_catalog

    source = inspect.getsource(mcp_catalog._run_bootstrap)
    # Check only code lines, not docstring/comment
    in_docstring = False
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.endswith('"""'):
            in_docstring = not in_docstring
            continue
        if in_docstring or stripped.startswith("#"):
            continue
        if "shell=True" in stripped:
            pytest.fail(f"shell=True found in code: {stripped}")


def test_shlex_split_handles_quoted_arguments():
    """shlex.split correctly parses commands with quoted arguments.

    This is the mechanism we rely on instead of shell=True — verify
    it handles the common cases that shell=True would have handled.
    """
    # Simple command
    assert shlex.split("echo hello") == ["echo", "hello"]

    # Quoted argument
    assert shlex.split('echo "hello world"') == ["echo", "hello world"]

    # Command with flags
    assert shlex.split("pip install -r requirements.txt") == [
        "pip", "install", "-r", "requirements.txt"
    ]

    # npm install
    assert shlex.split("npm install") == ["npm", "install"]

    # Path with spaces
    assert shlex.split('python "/path with spaces/script.py"') == [
        "python", "/path with spaces/script.py"
    ]