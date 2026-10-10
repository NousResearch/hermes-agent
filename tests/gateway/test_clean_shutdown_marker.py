"""Tests for the clean shutdown marker that prevents unwanted session auto-resets.

When the gateway shuts down gracefully (hermes update, gateway restart, /restart),
it writes a .clean_shutdown marker.  On the next startup, if the marker exists,
crash-turn recovery is skipped and orphan turn markers are discarded.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(platform=Platform.TELEGRAM, chat_id="123", user_id="u1"):
    return SessionSource(platform=platform, chat_id=chat_id, user_id=user_id)


def _make_store(tmp_path):
    config = GatewayConfig()
    return SessionStore(sessions_dir=tmp_path, config=config)


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
             patch("tools.browser_tool_lifecycle.cleanup_all_browsers"):
            mock_proc_reg.kill_all = MagicMock()

            import asyncio
            asyncio.get_event_loop().run_until_complete(runner.stop())

        assert marker.exists(), ".clean_shutdown marker should exist after graceful stop"


# ---------------------------------------------------------------------------
# resume_pending freshness gate (#46934)
# ---------------------------------------------------------------------------
