"""Tests for the clean shutdown marker that prevents unwanted session auto-resets.

When the gateway shuts down gracefully (hermes update, gateway restart, /restart),
it writes a .clean_shutdown marker.  On the next startup, if the marker exists,
suspend_recently_active() is skipped so users don't lose their sessions.

After a crash (no marker), suspension still fires as a safety net for stuck sessions.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, SessionResetPolicy
from gateway.session import SessionEntry, SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(platform=Platform.TELEGRAM, chat_id="123", user_id="u1"):
    return SessionSource(platform=platform, chat_id=chat_id, user_id=user_id)


def _make_store(tmp_path, policy=None):
    config = GatewayConfig()
    if policy:
        config.default_reset_policy = policy
    return SessionStore(sessions_dir=tmp_path, config=config)


# ---------------------------------------------------------------------------
# SessionStore.suspend_recently_active
# ---------------------------------------------------------------------------

class TestSuspendRecentlyActive:
    """Verify suspend_recently_active only marks recent sessions."""

    def test_suspends_recently_active_sessions(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        assert not entry.suspended

        count = store.suspend_recently_active()
        assert count == 1

        # Re-fetch — should be suspended now
        refreshed = store.get_or_create_session(source)
        assert refreshed.was_auto_reset

    def test_does_not_suspend_old_sessions(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)

        # Backdate the session's updated_at beyond the cutoff
        with store._lock:
            entry.updated_at = datetime.now() - timedelta(seconds=300)
            store._save()

        count = store.suspend_recently_active(max_age_seconds=120)
        assert count == 0

    def test_already_suspended_not_double_counted(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)

        # Suspend once
        count1 = store.suspend_recently_active()
        assert count1 == 1

        # Create a new session (the old one got reset on next access)
        entry2 = store.get_or_create_session(source)

        # Suspend again — the new session is recent but not yet suspended
        count2 = store.suspend_recently_active()
        assert count2 == 1

    def test_excluded_session_keys_are_not_suspended(self, tmp_path):
        store = _make_store(tmp_path)
        keep_source = _make_source(chat_id="keep")
        suspend_source = _make_source(chat_id="suspend")

        keep_entry = store.get_or_create_session(keep_source)
        suspend_entry = store.get_or_create_session(suspend_source)

        count = store.suspend_recently_active(
            exclude_session_keys={keep_entry.session_key}
        )
        assert count == 1

        with store._lock:
            store._ensure_loaded_locked()
            assert store._entries[keep_entry.session_key].suspended is False
            assert store._entries[suspend_entry.session_key].suspended is True


# ---------------------------------------------------------------------------
# Clean shutdown marker integration
# ---------------------------------------------------------------------------

class TestCleanShutdownMarker:
    """Test that the marker file controls session suspension on startup."""

    def test_marker_written_on_graceful_stop(self, tmp_path, monkeypatch):
        """stop() should write .clean_shutdown marker."""
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        marker = tmp_path / ".clean_shutdown"
        assert not marker.exists()

        # Create a minimal runner and call the shutdown logic directly
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner._restart_requested = False
        runner._restart_detached = False
        runner._restart_via_service = False
        runner._restart_task_started = False
        runner._running = True
        runner._draining = False
        runner._stop_task = None
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._pending_approvals = {}
        runner._background_tasks = set()
        runner._shutdown_event = MagicMock()
        runner._restart_drain_timeout = 5
        runner._exit_code = None
        runner._exit_reason = None
        runner.adapters = {}
        runner.config = GatewayConfig()

        # Mock heavy dependencies
        with patch("gateway.run.GatewayRunner._drain_active_agents", new_callable=AsyncMock, return_value=([], False)), \
             patch("gateway.run.GatewayRunner._finalize_shutdown_agents"), \
             patch("gateway.run.GatewayRunner._update_runtime_status"), \
             patch("gateway.status.remove_pid_file"), \
             patch("tools.process_registry.process_registry") as mock_proc_reg, \
             patch("tools.terminal_tool.cleanup_all_environments"), \
             patch("tools.browser_tool.cleanup_all_browsers"):
            mock_proc_reg.kill_all = MagicMock()

            import asyncio
            asyncio.get_event_loop().run_until_complete(runner.stop())

        assert marker.exists(), ".clean_shutdown marker should exist after graceful stop"

    def test_marker_skips_suspension_on_startup(self, tmp_path, monkeypatch):
        """If .clean_shutdown exists, suspend_recently_active should NOT be called."""
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        # Create the marker
        marker = tmp_path / ".clean_shutdown"
        marker.touch()

        # Create a store with a recently active session
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        assert not entry.suspended

        # Simulate what start() does:
        if marker.exists():
            marker.unlink()
            # Should NOT call suspend_recently_active
        else:
            store.suspend_recently_active()

        # Session should NOT be suspended
        with store._lock:
            store._ensure_loaded_locked()
            for e in store._entries.values():
                assert not e.suspended, "Session should NOT be suspended after clean shutdown"

        assert not marker.exists(), "Marker should be cleaned up"

    def test_no_marker_triggers_suspension(self, tmp_path, monkeypatch):
        """Without .clean_shutdown marker (crash), suspension should fire."""
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        marker = tmp_path / ".clean_shutdown"
        assert not marker.exists()

        # Create a store with a recently active session
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        assert not entry.suspended

        # Simulate what start() does:
        if marker.exists():
            marker.unlink()
        else:
            store.suspend_recently_active()

        # Session SHOULD be suspended (crash recovery)
        with store._lock:
            store._ensure_loaded_locked()
            suspended_count = sum(1 for e in store._entries.values() if e.suspended)
        assert suspended_count == 1, "Session should be suspended after crash (no marker)"

    def test_marker_written_on_restart_stop(self, tmp_path, monkeypatch):
        """stop(restart=True) should also write the marker."""
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        marker = tmp_path / ".clean_shutdown"

        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner._restart_requested = False
        runner._restart_detached = False
        runner._restart_via_service = False
        runner._restart_task_started = False
        runner._running = True
        runner._draining = False
        runner._stop_task = None
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._pending_approvals = {}
        runner._background_tasks = set()
        runner._shutdown_event = MagicMock()
        runner._restart_drain_timeout = 5
        runner._exit_code = None
        runner._exit_reason = None
        runner.adapters = {}
        runner.config = GatewayConfig()

        with patch("gateway.run.GatewayRunner._drain_active_agents", new_callable=AsyncMock, return_value=([], False)), \
             patch("gateway.run.GatewayRunner._finalize_shutdown_agents"), \
             patch("gateway.run.GatewayRunner._update_runtime_status"), \
             patch("gateway.status.remove_pid_file"), \
             patch("tools.process_registry.process_registry") as mock_proc_reg, \
             patch("tools.terminal_tool.cleanup_all_environments"), \
             patch("tools.browser_tool.cleanup_all_browsers"):
            mock_proc_reg.kill_all = MagicMock()

            import asyncio
            asyncio.get_event_loop().run_until_complete(runner.stop(restart=True))

        assert marker.exists(), ".clean_shutdown marker should exist after restart-stop too"

    def test_timed_out_restart_writes_planned_restart_marker(self, tmp_path, monkeypatch):
        """A planned restart that times out should preserve exact session keys."""
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner._restart_requested = True
        runner._restart_detached = False
        runner._restart_via_service = False
        runner._restart_task_started = False
        runner._running = True
        runner._draining = False
        runner._stop_task = None
        runner._running_agents = {"session:keep": MagicMock()}
        runner._pending_messages = {}
        runner._pending_approvals = {}
        runner._background_tasks = set()
        runner._shutdown_event = MagicMock()
        runner._restart_drain_timeout = 5
        runner._exit_code = None
        runner._exit_reason = None
        runner.adapters = {}
        runner.config = GatewayConfig()

        with patch("gateway.run.GatewayRunner._notify_active_sessions_of_shutdown", new_callable=AsyncMock), \
             patch("gateway.run.GatewayRunner._drain_active_agents", new_callable=AsyncMock, return_value=({"session:keep": MagicMock()}, True)), \
             patch("gateway.run.GatewayRunner._interrupt_running_agents"), \
             patch("gateway.run.GatewayRunner._finalize_shutdown_agents"), \
             patch("gateway.run.GatewayRunner._update_runtime_status"), \
             patch("gateway.status.remove_pid_file"), \
             patch("tools.process_registry.process_registry") as mock_proc_reg, \
             patch("tools.terminal_tool.cleanup_all_environments"), \
             patch("tools.browser_tool.cleanup_all_browsers"):
            mock_proc_reg.kill_all = MagicMock()

            import asyncio
            asyncio.get_event_loop().run_until_complete(runner.stop(restart=True))

        planned = tmp_path / ".planned_restart_sessions.json"
        assert planned.exists(), "planned restart marker should be written"
        assert 'session:keep' in planned.read_text(encoding='utf-8')
        assert not (tmp_path / ".restart_failure_counts").exists()

    def test_planned_restart_marker_excludes_preserved_sessions_from_suspension(self, tmp_path, monkeypatch):
        """Startup should preserve planned-restart sessions while still suspending crash leftovers."""
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        planned = tmp_path / ".planned_restart_sessions.json"
        planned.write_text('{"session_keys": ["session:keep"]}', encoding='utf-8')

        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner.session_store = MagicMock()
        runner.session_store.suspend_recently_active.return_value = 1
        runner._suspend_stuck_loop_sessions = MagicMock(return_value=0)

        runner._recover_previous_run_sessions()

        runner.session_store.suspend_recently_active.assert_called_once_with(
            exclude_session_keys={"session:keep"}
        )
        assert not planned.exists(), "planned restart marker should be consumed on startup"
