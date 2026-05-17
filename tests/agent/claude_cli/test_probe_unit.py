"""Unit tests for probe helpers — no real subprocess calls."""

from unittest.mock import patch

import pytest

from agent.claude_cli import errors, probe


def test_discover_binary_returns_path_when_present():
    """When `claude` is on PATH, discover_binary returns the resolved path."""
    with patch("agent.claude_cli.probe.shutil.which", return_value="/usr/local/bin/claude"):
        assert probe.discover_binary() == "/usr/local/bin/claude"


def test_discover_binary_raises_when_missing():
    """When `claude` is not on PATH, discover_binary raises ClaudeCliUnavailable."""
    with patch("agent.claude_cli.probe.shutil.which", return_value=None):
        with pytest.raises(errors.ClaudeCliUnavailable, match="not found"):
            probe.discover_binary()


def test_discover_binary_honors_explicit_path():
    """When an explicit path is given and exists, no PATH lookup is performed."""
    with patch("agent.claude_cli.probe.os.path.isfile", return_value=True), patch(
        "agent.claude_cli.probe.os.access", return_value=True
    ):
        assert probe.discover_binary("/opt/custom/claude") == "/opt/custom/claude"


def test_discover_binary_explicit_path_missing_raises():
    """Explicit path that doesn't exist raises ClaudeCliUnavailable."""
    with patch("agent.claude_cli.probe.os.path.isfile", return_value=False):
        with pytest.raises(errors.ClaudeCliUnavailable, match="/opt/custom/claude"):
            probe.discover_binary("/opt/custom/claude")


def test_parse_version_string_extracts_semver():
    """parse_version_string extracts the numeric version from `claude --version` output."""
    assert probe.parse_version_string("2.1.143 (Claude Code)\n") == (2, 1, 143)
    assert probe.parse_version_string("2.1.143") == (2, 1, 143)
    assert probe.parse_version_string("v2.1.143 (Claude Code)") == (2, 1, 143)


def test_parse_version_string_unparseable_raises():
    """parse_version_string raises on output it can't parse."""
    with pytest.raises(errors.ClaudeCliIncompatible, match="unexpected"):
        probe.parse_version_string("hello world\n")


def test_check_version_passes_when_at_or_above_floor():
    """check_version returns the parsed version tuple when at or above the floor."""
    assert probe.check_version((2, 1, 143), min_version=(2, 1, 143)) == (2, 1, 143)
    assert probe.check_version((2, 2, 0), min_version=(2, 1, 143)) == (2, 2, 0)


def test_check_version_raises_when_below_floor():
    """check_version raises ClaudeCliVersionTooOld when below the floor."""
    with pytest.raises(errors.ClaudeCliVersionTooOld, match="2.1.142.+2.1.143"):
        probe.check_version((2, 1, 142), min_version=(2, 1, 143))
