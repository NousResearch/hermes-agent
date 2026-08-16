"""CLI coverage for the public Computer Use command surface."""

from __future__ import annotations

import subprocess
import sys


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "computer-use", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_computer_use_help_omits_browser_approve() -> None:
    result = _run("--help")

    assert result.returncode == 0
    assert "browser-approve" not in result.stdout
    assert "doctor" in result.stdout
    assert "permissions" in result.stdout


def test_computer_use_rejects_removed_browser_approve_command() -> None:
    result = _run("browser-approve", "--pid", "123")

    assert result.returncode == 2
    assert "invalid choice: 'browser-approve'" in result.stderr
