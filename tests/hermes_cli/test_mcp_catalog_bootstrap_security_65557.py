"""Tests for _run_bootstrap shell-injection hardening (#65557 Site 1).

`_run_bootstrap` previously called `subprocess.run(cmd, shell=True)` for each
command string in a catalog entry's `bootstrap` list.  This file asserts the
new argv-list path: shell metacharacters in the catalog no longer reach a
shell parser.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.mcp_catalog import _run_bootstrap, CatalogError


def _capture_run_calls():
    calls = []
    def _fake_run(args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
    return calls, _fake_run


def test_run_bootstrap_uses_argv_list_no_shell(monkeypatch, tmp_path):
    """Plain single-command bootstrap → run with shlex.split argv list, no shell."""
    calls, fake_run = _capture_run_calls()
    monkeypatch.setattr("hermes_cli.mcp_catalog.subprocess.run", fake_run)

    _run_bootstrap(tmp_path, ["pip install -r requirements.txt"])

    assert len(calls) == 1
    assert calls[0]["args"] == ["pip", "install", "-r", "requirements.txt"]
    assert calls[0]["kwargs"].get("shell") in (None, False)
    assert calls[0]["kwargs"].get("cwd") == str(tmp_path)


def test_run_bootstrap_metacharacters_not_interpreted(monkeypatch, tmp_path):
    """A semicolon-separated payload is parsed as one argv — no shell execution.

    Previously, ``"pip install x; <payload>"`` executed the payload via the
    system shell.  With shlex.split, the semicolon is just an argument to
    pip — pip will reject it (returncode != 0), but the payload never runs.
    """
    calls, fake_run = _capture_run_calls()
    monkeypatch.setattr("hermes_cli.mcp_catalog.subprocess.run", fake_run)

    _run_bootstrap(tmp_path, ["pip install x; echo PWNED"])

    # Single subprocess.run call, with the entire string shlex.split into one argv list.
    # The "echo" and "PWNED" tokens are passed to pip as additional arguments, NOT
    # executed as a separate shell command.
    assert len(calls) == 1
    assert calls[0]["args"][0] == "pip"
    assert "echo" in calls[0]["args"]
    assert "PWNED" in calls[0]["args"]
    assert calls[0]["kwargs"].get("shell") in (None, False)


def test_run_bootstrap_quoted_arguments_preserved(monkeypatch, tmp_path):
    """shlex.split respects shell quoting — quoted args stay together."""
    calls, fake_run = _capture_run_calls()
    monkeypatch.setattr("hermes_cli.mcp_catalog.subprocess.run", fake_run)

    _run_bootstrap(tmp_path, ['echo "hello world"'])

    assert calls[0]["args"] == ["echo", "hello world"]


def test_run_bootstrap_multiple_commands_run_sequentially(monkeypatch, tmp_path):
    """Catalog entries that split chained steps into list entries run in order."""
    calls, fake_run = _capture_run_calls()
    monkeypatch.setattr("hermes_cli.mcp_catalog.subprocess.run", fake_run)

    _run_bootstrap(
        tmp_path,
        [
            "pip install -r requirements.txt",
            "npm install",
            "npm run build",
        ],
    )

    assert len(calls) == 3
    assert calls[0]["args"] == ["pip", "install", "-r", "requirements.txt"]
    assert calls[1]["args"] == ["npm", "install"]
    assert calls[2]["args"] == ["npm", "run", "build"]


def test_run_bootstrap_aborts_on_first_failure(monkeypatch, tmp_path):
    """A non-zero returncode aborts early — subsequent commands skipped."""
    call_count = {"n": 0}

    def _fake_run(args, **kwargs):
        call_count["n"] += 1
        rc = 1 if call_count["n"] == 1 else 0
        return subprocess.CompletedProcess(args=args, returncode=rc, stdout="", stderr="")

    monkeypatch.setattr("hermes_cli.mcp_catalog.subprocess.run", _fake_run)

    with pytest.raises(CatalogError) as excinfo:
        _run_bootstrap(
            tmp_path,
            [
                "pip install -r requirements.txt",
                "npm install",
            ],
        )

    assert "bootstrap step failed" in str(excinfo.value)
    assert call_count["n"] == 1  # second command never ran


def test_run_bootstrap_skips_empty_or_whitespace_only_strings(monkeypatch, tmp_path):
    """An empty or whitespace-only bootstrap string is a no-op for that step."""
    calls, fake_run = _capture_run_calls()
    monkeypatch.setattr("hermes_cli.mcp_catalog.subprocess.run", fake_run)

    _run_bootstrap(
        tmp_path,
        [
            "",
            "   ",
            "pip install -r requirements.txt",
        ],
    )

    # Only the real command runs.
    assert len(calls) == 1
    assert calls[0]["args"] == ["pip", "install", "-r", "requirements.txt"]


def test_run_bootstrap_handles_no_commands(monkeypatch, tmp_path):
    """Empty list → no subprocess.run invocations."""
    calls, fake_run = _capture_run_calls()
    monkeypatch.setattr("hermes_cli.mcp_catalog.subprocess.run", fake_run)

    _run_bootstrap(tmp_path, [])

    assert calls == []
