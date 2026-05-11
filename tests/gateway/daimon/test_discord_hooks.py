"""Tests for DaimonDiscordHooks integration layer."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from gateway.daimon.discord_hooks import DaimonDiscordHooks
from gateway.daimon.session_manager import SessionStartResult
from gateway.daimon.admin_commands import CommandResult


def _make_config(admin_users=None, max_active=2, max_threads_per_day=5):
    """Build a raw config dict for testing."""
    return {
        "gateway": {
            "discord": {
                "daimon": {
                    "admin_users": admin_users or [],
                    "max_active_sessions": max_active,
                    "max_threads_per_day": max_threads_per_day,
                    "user_model": "test/user-model",
                    "admin_model": "test/admin-model",
                }
            }
        }
    }


class TestInactiveHooks:
    """Tests for hooks when Daimon is inactive (no admins)."""

    def test_inactive_when_no_admins(self):
        """hooks.active is False when no admin_users configured."""
        hooks = DaimonDiscordHooks(_make_config(admin_users=[]))
        assert hooks.active is False
        assert hooks.manager is None

    def test_should_process_passthrough_when_inactive(self):
        """Returns True for all messages when Daimon not active."""
        hooks = DaimonDiscordHooks(_make_config(admin_users=[]))
        assert hooks.should_process_in_thread("anyone", "any-thread") is True

    def test_on_thread_created_passthrough_when_inactive(self):
        """Returns allowed=True when inactive."""
        hooks = DaimonDiscordHooks(_make_config(admin_users=[]))
        result = hooks.on_thread_created("thread-1", "user-1", {})
        assert result.allowed is True

    def test_on_thread_closed_passthrough_when_inactive(self):
        """Returns None when inactive."""
        hooks = DaimonDiscordHooks(_make_config(admin_users=[]))
        result = hooks.on_thread_closed("thread-1")
        assert result is None

    def test_redact_passthrough(self):
        """No redaction applied when inactive."""
        hooks = DaimonDiscordHooks(_make_config(admin_users=[]))
        text = "My key is sk-proj-12345678901234567890123456789012345678901234567890"
        assert hooks.redact(text) == text

    def test_handle_admin_command_when_inactive(self):
        """Returns failure when Daimon is not active."""
        hooks = DaimonDiscordHooks(_make_config(admin_users=[]))
        result = hooks.handle_admin_command("status", "")
        assert result.success is False
        assert "not active" in result.message


class TestActiveHooks:
    """Tests for hooks when Daimon is active (admins configured)."""

    def test_active_with_admins(self):
        """hooks.active is True when admin_users configured."""
        hooks = DaimonDiscordHooks(_make_config(admin_users=["admin1"]))
        assert hooks.active is True
        assert hooks.manager is not None

    def test_is_banned(self):
        """Banned user detected correctly."""
        hooks = DaimonDiscordHooks(_make_config(admin_users=["admin1"]))
        assert hooks.is_banned("user-1") is False

        # Manually add to banned set
        hooks._banned.add("user-1")
        assert hooks.is_banned("user-1") is True

    @patch("gateway.daimon.workspace.subprocess.run")
    def test_should_process_creator_allowed(self, mock_run):
        """Registered thread creator passes the filter."""
        mock_run.return_value = MagicMock(returncode=0)
        raw_config = _make_config(admin_users=["admin1"])
        hooks = DaimonDiscordHooks(raw_config)

        # Start a session to register the thread
        hooks.on_thread_created("thread-1", "user-1", raw_config)

        assert hooks.should_process_in_thread("user-1", "thread-1") is True

        # Cleanup
        hooks.on_thread_closed("thread-1")

    @patch("gateway.daimon.workspace.subprocess.run")
    def test_should_process_other_blocked(self, mock_run):
        """Non-creator is blocked in a registered thread."""
        mock_run.return_value = MagicMock(returncode=0)
        raw_config = _make_config(admin_users=["admin1"])
        hooks = DaimonDiscordHooks(raw_config)

        # Start a session to register the thread
        hooks.on_thread_created("thread-1", "user-1", raw_config)

        assert hooks.should_process_in_thread("user-2", "thread-1") is False

        # Cleanup
        hooks.on_thread_closed("thread-1")

    @patch("gateway.daimon.workspace.subprocess.run")
    def test_on_thread_created_success(self, mock_run):
        """Session starts, returns allowed=True with overrides."""
        mock_run.return_value = MagicMock(returncode=0)
        raw_config = _make_config(admin_users=["admin1"], max_active=5)
        hooks = DaimonDiscordHooks(raw_config)

        result = hooks.on_thread_created("thread-1", "user-1", raw_config)

        assert result.allowed is True
        assert result.queue_position == 0
        assert result.overrides is not None

        # Cleanup
        hooks.on_thread_closed("thread-1")

    @patch("gateway.daimon.workspace.subprocess.run")
    def test_on_thread_created_banned(self, mock_run):
        """Banned user gets denial."""
        mock_run.return_value = MagicMock(returncode=0)
        raw_config = _make_config(admin_users=["admin1"])
        hooks = DaimonDiscordHooks(raw_config)

        # Ban a user
        hooks._banned.add("banned-user")

        result = hooks.on_thread_created("thread-1", "banned-user", raw_config)

        assert result.allowed is False
        assert "banned" in result.denial_reason.lower()

    @patch("gateway.daimon.workspace.subprocess.run")
    def test_on_thread_closed_cleanup(self, mock_run):
        """on_thread_closed calls end_session and cleans up queued tracking."""
        mock_run.return_value = MagicMock(returncode=0)
        raw_config = _make_config(admin_users=["admin1"], max_active=5)
        hooks = DaimonDiscordHooks(raw_config)

        # Start a session
        hooks.on_thread_created("thread-1", "user-1", raw_config)
        # Add to queued tracking
        hooks.queue_thread("thread-1", "fake-thread-obj")

        # Close
        promoted = hooks.on_thread_closed("thread-1")

        assert promoted is None  # No one in queue to promote
        assert "thread-1" not in hooks._queued

    @patch("gateway.daimon.workspace.subprocess.run")
    def test_handle_admin_command(self, mock_run):
        """Dispatches to admin_commands module."""
        mock_run.return_value = MagicMock(returncode=0, stdout="CPU: 5%", stderr="")
        raw_config = _make_config(admin_users=["admin1"])
        hooks = DaimonDiscordHooks(raw_config)

        result = hooks.handle_admin_command("status", "")

        assert result.success is True
        assert "Daimon Status" in result.message

    def test_redact_active(self):
        """Active hooks apply redaction."""
        hooks = DaimonDiscordHooks(_make_config(admin_users=["admin1"]))
        # Use a pattern that the redaction module will catch
        text = "My key is sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        result = hooks.redact(text)
        assert result != text
        assert "REDACTED" in result


class TestQueueManagement:
    """Tests for queue/pop thread object tracking."""

    def test_queue_and_pop(self):
        """queue_thread stores object, pop_queued retrieves and removes it."""
        hooks = DaimonDiscordHooks(_make_config(admin_users=["admin1"]))

        fake_thread = MagicMock(name="FakeThread")
        hooks.queue_thread("thread-1", fake_thread)

        # Pop returns the object
        result = hooks.pop_queued("thread-1")
        assert result is fake_thread

        # Pop again returns None (already removed)
        result = hooks.pop_queued("thread-1")
        assert result is None

    def test_pop_queued_nonexistent(self):
        """pop_queued returns None for unknown thread_id."""
        hooks = DaimonDiscordHooks(_make_config(admin_users=["admin1"]))
        assert hooks.pop_queued("nonexistent") is None


class TestInitFailure:
    """Tests for graceful handling of init errors."""

    @patch("gateway.daimon.discord_hooks.DaimonSessionManager")
    def test_init_exception_graceful(self, mock_mgr_class):
        """If DaimonSessionManager init raises, hooks gracefully become inactive."""
        mock_mgr_class.side_effect = RuntimeError("config parse failure")

        hooks = DaimonDiscordHooks({"bad": "config"})

        assert hooks.active is False
        assert hooks.manager is None
        # All methods should pass-through gracefully
        assert hooks.should_process_in_thread("user", "thread") is True
        assert hooks.on_thread_closed("thread") is None
        assert hooks.redact("text") == "text"
