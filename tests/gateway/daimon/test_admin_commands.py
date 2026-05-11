# tests/gateway/daimon/test_admin_commands.py
"""Tests for /daimon admin command handlers."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from gateway.daimon.admin_commands import (
    CONTAINER_NAME,
    CommandResult,
    handle_daimon_command,
)
from gateway.daimon.session_manager import DaimonSessionManager


@pytest.fixture
def mgr():
    """Create a DaimonSessionManager with minimal config."""
    return DaimonSessionManager({"gateway": {"discord": {"daimon": {"admin_users": ["123"]}}}})


@pytest.fixture
def banned():
    """Create an empty banned users set."""
    return set()


class TestUnknownSubcommand:
    def test_unknown_subcommand(self, mgr, banned):
        result = handle_daimon_command("foobar", "", mgr, banned)
        assert result.success is False
        assert "Unknown subcommand" in result.message
        assert "foobar" in result.message
        # All available commands should be listed
        for cmd in ("restart", "status", "kill", "ban", "limits"):
            assert cmd in result.message


class TestRestart:
    @patch("gateway.daimon.admin_commands.subprocess.run")
    def test_restart_success(self, mock_run, mgr, banned):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = handle_daimon_command("restart", "", mgr, banned)
        assert result.success is True
        assert "restarted" in result.message
        assert CONTAINER_NAME in result.message
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert CONTAINER_NAME in call_args[0][0]

    @patch("gateway.daimon.admin_commands.subprocess.run")
    def test_restart_failure(self, mock_run, mgr, banned):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="No such container"
        )
        result = handle_daimon_command("restart", "", mgr, banned)
        assert result.success is False
        assert "failed" in result.message.lower() or "Restart failed" in result.message
        assert "No such container" in result.message

    @patch("gateway.daimon.admin_commands.subprocess.run")
    def test_restart_timeout(self, mock_run, mgr, banned):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=60)
        result = handle_daimon_command("restart", "", mgr, banned)
        assert result.success is False
        assert "timed out" in result.message.lower()

    @patch("gateway.daimon.admin_commands.subprocess.run")
    def test_restart_exception(self, mock_run, mgr, banned):
        mock_run.side_effect = OSError("docker not found")
        result = handle_daimon_command("restart", "", mgr, banned)
        assert result.success is False
        assert "error" in result.message.lower()


class TestStatus:
    @patch("gateway.daimon.admin_commands.subprocess.run")
    def test_status_format(self, mock_run, mgr, banned):
        # Mock docker stats call
        def side_effect(cmd, **kwargs):
            if "stats" in cmd:
                return MagicMock(
                    returncode=0,
                    stdout="CPU: 5.2%, Mem: 128MiB / 1GiB, PIDs: 42",
                    stderr="",
                )
            elif "inspect" in cmd:
                return MagicMock(
                    returncode=0,
                    stdout="2026-05-09T04:00:00.000Z",
                    stderr="",
                )
            return MagicMock(returncode=1, stdout="", stderr="")

        mock_run.side_effect = side_effect
        banned.add("user999")

        result = handle_daimon_command("status", "", mgr, banned)
        assert result.success is True
        assert "Daimon Status" in result.message
        assert CONTAINER_NAME in result.message
        assert "Active sessions" in result.message
        assert "Queue" in result.message
        assert "Banned users: 1" in result.message
        assert "CPU" in result.message
        assert "since 2026-05-09T04:00:0" in result.message

    @patch("gateway.daimon.admin_commands.subprocess.run")
    def test_status_docker_unavailable(self, mock_run, mgr, banned):
        mock_run.side_effect = Exception("docker not found")
        result = handle_daimon_command("status", "", mgr, banned)
        assert result.success is True
        assert "unavailable" in result.message
        assert "unknown" in result.message


class TestKill:
    def test_kill_session(self, mgr, banned):
        # Start a session first so we can kill it
        mgr.start_session("thread-abc", "user1", {"gateway": {"discord": {"daimon": {"admin_users": ["123"]}}}})
        result = handle_daimon_command("kill", "thread-abc", mgr, banned)
        assert result.success is True
        assert "terminated" in result.message
        assert "thread-abc" in result.message

    def test_kill_no_args(self, mgr, banned):
        result = handle_daimon_command("kill", "", mgr, banned)
        assert result.success is False
        assert "Usage" in result.message
        assert "thread_id" in result.message

    def test_kill_with_whitespace_args(self, mgr, banned):
        result = handle_daimon_command("kill", "   ", mgr, banned)
        assert result.success is False
        assert "Usage" in result.message


class TestBan:
    def test_ban_user(self, mgr, banned):
        result = handle_daimon_command("ban", "user456", mgr, banned)
        assert result.success is True
        assert "user456" in result.message
        assert "Banned" in result.message
        assert "user456" in banned

    def test_ban_no_args(self, mgr, banned):
        result = handle_daimon_command("ban", "", mgr, banned)
        assert result.success is False
        assert "Usage" in result.message
        assert "user_id" in result.message

    def test_ban_multiple_users(self, mgr, banned):
        handle_daimon_command("ban", "user1", mgr, banned)
        handle_daimon_command("ban", "user2", mgr, banned)
        assert "user1" in banned
        assert "user2" in banned
        assert len(banned) == 2


class TestLimits:
    def test_limits_format(self, mgr, banned):
        result = handle_daimon_command("limits", "", mgr, banned)
        assert result.success is True
        msg = result.message
        # Check all expected fields
        assert "Daimon User Limits" in msg
        assert mgr.config.user_model in msg
        assert str(mgr.config.max_iterations) in msg
        assert str(mgr.config.max_threads_per_day) in msg
        assert str(mgr.config.gateway_timeout) in msg
        assert str(mgr.config.max_active_sessions) in msg
        assert "Tool limits" in msg
        # Check some specific tool limits appear
        assert "web_search" in msg
        assert "15/session" in msg
        assert "image_generate" in msg
        assert "3/session" in msg
        # Disabled tools should show as disabled
        assert "text_to_speech" in msg
        assert "disabled" in msg
        # Unlimited tools should NOT appear
        assert "terminal" not in msg
        assert "read_file" not in msg

    def test_limits_custom_config(self, banned):
        custom_mgr = DaimonSessionManager({
            "gateway": {"discord": {"daimon": {
                "admin_users": ["123"],
                "user_model": "custom/model",
                "max_iterations": 10,
                "max_threads_per_day": 3,
            }}}
        })
        result = handle_daimon_command("limits", "", custom_mgr, banned)
        assert result.success is True
        assert "custom/model" in result.message
        assert "10" in result.message
        assert "3" in result.message
