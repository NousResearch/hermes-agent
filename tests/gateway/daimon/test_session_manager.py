"""Tests for DaimonSessionManager orchestrator."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from gateway.daimon.session_manager import DaimonSessionManager, SessionStartResult
from gateway.daimon.tool_gate import get_limiter


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


class TestDaimonSessionManagerInactive:
    """Tests for inactive state."""

    def test_inactive_when_no_admins(self):
        """is_active returns False when admin_users is empty."""
        mgr = DaimonSessionManager(_make_config(admin_users=[]))
        assert mgr.is_active is False

    def test_active_when_admins_configured(self):
        """is_active returns True when admin_users is set."""
        mgr = DaimonSessionManager(_make_config(admin_users=["admin1"]))
        assert mgr.is_active is True


class TestStartSession:
    """Tests for start_session lifecycle."""

    @patch("gateway.daimon.workspace.subprocess.run")
    def test_start_session_success(self, mock_run):
        """Session starts successfully with overrides."""
        mock_run.return_value = MagicMock(returncode=0)
        raw_config = _make_config(admin_users=["admin1"], max_active=5)
        mgr = DaimonSessionManager(raw_config)

        result = mgr.start_session("thread-1", "user-1", raw_config)

        assert result.allowed is True
        assert result.queue_position == 0
        assert result.denial_reason == ""
        assert result.overrides is not None
        assert result.overrides.model == "test/user-model"

        # Tool limiter is NOT registered by session_manager anymore —
        # it's handled by gateway_hooks.setup_tool_gate() inside run_sync()
        # (keyed by hermes session_id, not thread_id)

        # Verify workspace was created
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "mkdir" in call_args
        assert "/workspaces/thread-1" in call_args

        # Cleanup
        mgr.end_session("thread-1")

    @patch("gateway.daimon.workspace.subprocess.run")
    def test_start_session_daily_limit(self, mock_run):
        """Session denied when daily limit is hit."""
        mock_run.return_value = MagicMock(returncode=0)
        raw_config = _make_config(admin_users=["admin1"], max_active=10, max_threads_per_day=2)
        mgr = DaimonSessionManager(raw_config)

        # Start 2 sessions (max daily)
        r1 = mgr.start_session("thread-1", "user-1", raw_config)
        r2 = mgr.start_session("thread-2", "user-1", raw_config)
        assert r1.allowed is True
        assert r2.allowed is True

        # Third should be denied
        r3 = mgr.start_session("thread-3", "user-1", raw_config)
        assert r3.allowed is False
        assert "Daily limit" in r3.denial_reason

        # Cleanup
        mgr.end_session("thread-1")
        mgr.end_session("thread-2")

    @patch("gateway.daimon.workspace.subprocess.run")
    def test_start_session_queue(self, mock_run):
        """Session queued when at capacity."""
        mock_run.return_value = MagicMock(returncode=0)
        raw_config = _make_config(admin_users=["admin1"], max_active=1, max_threads_per_day=10)
        mgr = DaimonSessionManager(raw_config)

        # Fill the single slot
        r1 = mgr.start_session("thread-1", "user-1", raw_config)
        assert r1.allowed is True

        # Next should be queued
        r2 = mgr.start_session("thread-2", "user-2", raw_config)
        assert r2.allowed is False
        assert r2.queue_position > 0

        # Cleanup
        mgr.end_session("thread-1")


class TestEndSession:
    """Tests for end_session lifecycle."""

    @patch("gateway.daimon.workspace.subprocess.run")
    def test_end_session_cleanup(self, mock_run):
        """end_session destroys workspace, releases slot (limiter handled by gateway_hooks)."""
        mock_run.return_value = MagicMock(returncode=0)
        raw_config = _make_config(admin_users=["admin1"], max_active=5)
        mgr = DaimonSessionManager(raw_config)

        # Start then end
        mgr.start_session("thread-1", "user-1", raw_config)
        assert mgr.active_sessions == 1

        promoted = mgr.end_session("thread-1")

        assert promoted is None
        assert mgr.active_sessions == 0

        # Verify workspace destroy was called
        destroy_call = mock_run.call_args_list[-1]
        call_args = destroy_call[0][0]
        assert "rm" in call_args
        assert "/workspaces/thread-1" in call_args

    @patch("gateway.daimon.workspace.subprocess.run")
    def test_end_session_promotes_next(self, mock_run):
        """end_session returns promoted thread_id when queue has waiters."""
        mock_run.return_value = MagicMock(returncode=0)
        raw_config = _make_config(admin_users=["admin1"], max_active=1, max_threads_per_day=10)
        mgr = DaimonSessionManager(raw_config)

        # Fill single slot
        mgr.start_session("thread-1", "user-1", raw_config)

        # Queue a second session
        r2 = mgr.start_session("thread-2", "user-2", raw_config)
        assert r2.allowed is False
        assert r2.queue_position == 1

        # End first — should promote second
        promoted = mgr.end_session("thread-1")
        assert promoted == "thread-2"


class TestShouldProcessMessage:
    """Tests for thread ownership filtering."""

    @patch("gateway.daimon.workspace.subprocess.run")
    def test_should_process_message_creator(self, mock_run):
        """Creator of thread passes the filter."""
        mock_run.return_value = MagicMock(returncode=0)
        raw_config = _make_config(admin_users=["admin1"])
        mgr = DaimonSessionManager(raw_config)

        mgr.start_session("thread-1", "user-1", raw_config)

        assert mgr.should_process_message("user-1", "thread-1") is True

        # Cleanup
        mgr.end_session("thread-1")

    @patch("gateway.daimon.workspace.subprocess.run")
    def test_should_process_message_other(self, mock_run):
        """Non-creator of thread is blocked."""
        mock_run.return_value = MagicMock(returncode=0)
        raw_config = _make_config(admin_users=["admin1"])
        mgr = DaimonSessionManager(raw_config)

        mgr.start_session("thread-1", "user-1", raw_config)

        assert mgr.should_process_message("user-2", "thread-1") is False

        # Cleanup
        mgr.end_session("thread-1")

    def test_should_process_message_admin_always_passes(self):
        """Admin passes the filter regardless of thread ownership."""
        raw_config = _make_config(admin_users=["admin1"])
        mgr = DaimonSessionManager(raw_config)

        # Even for an unknown thread, admin should pass
        assert mgr.should_process_message("admin1", "any-thread") is True


class TestRedact:
    """Tests for redaction delegation."""

    def test_redact(self):
        """redact() delegates to redact_response."""
        mgr = DaimonSessionManager(_make_config())
        text = "My key is sk-proj-abcdefghijklmnopqrstuvwx and secret"
        result = mgr.redact(text)
        assert "[REDACTED_OPENAI_KEY]" in result
        assert "sk-proj-" not in result


class TestProperties:
    """Tests for active_sessions and queue_length properties."""

    @patch("gateway.daimon.workspace.subprocess.run")
    def test_active_sessions_and_queue_length(self, mock_run):
        """Properties reflect the internal state."""
        mock_run.return_value = MagicMock(returncode=0)
        raw_config = _make_config(admin_users=["admin1"], max_active=1, max_threads_per_day=10)
        mgr = DaimonSessionManager(raw_config)

        assert mgr.active_sessions == 0
        assert mgr.queue_length == 0

        mgr.start_session("thread-1", "user-1", raw_config)
        assert mgr.active_sessions == 1

        mgr.start_session("thread-2", "user-2", raw_config)
        assert mgr.queue_length == 1

        mgr.end_session("thread-1")
        # thread-2 promoted, queue empty
        assert mgr.active_sessions == 1
        assert mgr.queue_length == 0
