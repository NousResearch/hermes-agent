"""Regression tests for Claude Code version detection under a GUI PATH.

Anthropic gates model access on the Claude Code version reported in the OAuth
user-agent. Reporting one below the gate returns HTTP 400:

    Claude Code 2.1.74 does not support this model;
    version 2.1.251 or newer is required.

``_detect_claude_code_version`` looked the CLI up by bare name, which resolves
through PATH only. GUI launches (the Electron desktop app, macOS LaunchAgents)
inherit ``/usr/bin:/bin:/usr/sbin:/sbin`` — none of the CLI's install prefixes
— so detection found nothing there even with a current CLI installed and fell
back to the stale constant. Same machine, same CLI: fine from a terminal,
rejected from the desktop app.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from agent.anthropic_adapter import (
    _CLAUDE_CODE_VERSION_FALLBACK,
    _claude_code_candidates,
    _detect_claude_code_version,
)

BARE_GUI_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
HOME = "/Users/tester"


@pytest.fixture
def gui_launch(monkeypatch):
    """A GUI-launched process: bare PATH, so ``shutil.which`` finds nothing."""
    monkeypatch.setenv("PATH", BARE_GUI_PATH)
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", HOME, 1))
    monkeypatch.setattr("shutil.which", lambda name: None)


def _cli_reporting(version: str, *, installed_at: str):
    """Patch ``subprocess.run`` so only ``installed_at`` answers --version."""
    def runner(cmd, *args, **kwargs):
        assert cmd[0] == installed_at, f"probed unexpected path: {cmd[0]}"
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=f"{version} (Claude Code)", stderr=""
        )

    return patch("subprocess.run", runner)


class TestCandidateProbing:
    def test_bare_gui_path_still_finds_cli_in_user_prefix(self, gui_launch):
        """The desktop-app bug: CLI present, PATH lookup blind to it."""
        installed = f"{HOME}/.local/bin/claude"
        with patch("os.path.isfile", lambda p: p == installed):
            assert installed in _claude_code_candidates()

    def test_path_hit_is_probed_first(self, monkeypatch):
        """A PATH resolution wins over the prefix list — it is what the user invokes."""
        on_path = "/opt/custom/claude"
        monkeypatch.setattr("shutil.which", lambda name: on_path if name == "claude" else None)
        monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", HOME, 1))
        with patch("os.path.isfile", return_value=True):
            assert _claude_code_candidates()[0] == on_path

    def test_candidates_are_deduped(self, monkeypatch):
        """``which`` returning a prefix path must not queue that path twice."""
        installed = f"{HOME}/.local/bin/claude"
        monkeypatch.setattr("shutil.which", lambda name: installed if name == "claude" else None)
        monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", HOME, 1))
        with patch("os.path.isfile", lambda p: p == installed):
            candidates = _claude_code_candidates()
        assert candidates.count(installed) == 1

    def test_missing_files_are_not_probed(self, gui_launch):
        """Nothing installed anywhere — no subprocess spawns to fail on."""
        with patch("os.path.isfile", return_value=False):
            assert _claude_code_candidates() == []


class TestDetectedVersion:
    def test_gui_launch_reports_installed_version_not_fallback(self, gui_launch):
        """End to end: the desktop app reports the real CLI version again."""
        installed = f"{HOME}/.local/bin/claude"
        current = "2.1.276"
        with patch("os.path.isfile", lambda p: p == installed), _cli_reporting(
            current, installed_at=installed
        ):
            assert _detect_claude_code_version() == current

    def test_no_cli_anywhere_falls_back(self, gui_launch):
        with patch("os.path.isfile", return_value=False):
            assert _detect_claude_code_version() == _CLAUDE_CODE_VERSION_FALLBACK

    def test_failing_cli_falls_back(self, gui_launch):
        """A CLI that errors must not yield a garbage version string."""
        installed = f"{HOME}/.local/bin/claude"
        failed = subprocess.CompletedProcess(
            args=[installed, "--version"], returncode=1, stdout="", stderr="boom"
        )
        with patch("os.path.isfile", lambda p: p == installed), patch(
            "subprocess.run", lambda *a, **k: failed
        ):
            assert _detect_claude_code_version() == _CLAUDE_CODE_VERSION_FALLBACK

    def test_non_numeric_output_falls_back(self, gui_launch):
        installed = f"{HOME}/.local/bin/claude"
        with patch("os.path.isfile", lambda p: p == installed), _cli_reporting(
            "unknown", installed_at=installed
        ):
            assert _detect_claude_code_version() == _CLAUDE_CODE_VERSION_FALLBACK
