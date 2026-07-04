"""Tests for the clean shutdown marker that prevents unwanted session auto-resets.

When the gateway shuts down gracefully (hermes update, gateway restart, /restart),
it writes a .clean_shutdown marker.  On the next startup, if the marker exists,
suspend_recently_active() is skipped so users don't lose their sessions.

After a crash (no marker), suspension still fires as a safety net for stuck sessions.
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

        # Re-fetch — should be resume_pending (preserved, not wiped)
        refreshed = store.get_or_create_session(source)
        assert refreshed.resume_pending
        assert refreshed.session_id == entry.session_id  # same session preserved

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

    def test_already_resume_pending_not_double_counted(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)

        # Mark resume_pending once
        count1 = store.suspend_recently_active()
        assert count1 == 1

        # Re-fetch returns the SAME session (preserved, not reset)
        entry2 = store.get_or_create_session(source)
        assert entry2.session_id == entry.session_id

        # Second call skips already-resume_pending entries
        count2 = store.suspend_recently_active()
        assert count2 == 0


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

        # Session SHOULD be resume_pending (crash recovery preserves history)
        with store._lock:
            store._ensure_loaded_locked()
            resume_count = sum(1 for e in store._entries.values() if e.resume_pending)
        assert resume_count == 1, "Session should be resume_pending after crash (no marker)"

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


    def test_shutdown_cleanup_does_not_end_gateway_session_rows(self, tmp_path, monkeypatch):
        """Gateway process restart/stop must not mark live chats ended in state.db."""
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        agent = MagicMock()
        agent._end_session_on_close = True

        async def _run():
            await GatewayRunner._cleanup_agent_resources_off_loop(
                runner, agent, context="shutdown idle-cache"
            )

        import asyncio
        asyncio.get_event_loop().run_until_complete(_run())

        assert agent._end_session_on_close is False
        agent.close.assert_called_once()

    def test_session_expiry_cleanup_preserves_lazy_reset_boundary(self, tmp_path, monkeypatch):
        """Session expiry cleanup must not turn an expired chat into an agent_close row.

        The expiry watcher only tears down cached resources. The next inbound
        message owns the reset boundary, creating a fresh session with the
        normal auto-reset notice. If cleanup lets ``agent.close()`` end the
        SQLite row as ``agent_close``, stale-route recovery treats it as
        recoverable and resurrects the expired session instead.
        """
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        agent = MagicMock()
        agent._end_session_on_close = True

        async def _run():
            await GatewayRunner._cleanup_agent_resources_off_loop(
                runner, agent, context="session expiry"
            )

        import asyncio
        asyncio.get_event_loop().run_until_complete(_run())

        assert agent._end_session_on_close is False
        agent.close.assert_called_once()


# ---------------------------------------------------------------------------
# resume_pending freshness gate (#46934)
# ---------------------------------------------------------------------------

class TestResumePendingFreshnessGate:
    """A resume_pending session is only returned while it is still fresh.

    ``get_or_create_session`` returns a ``resume_pending`` session so its
    transcript reloads intact after a restart.  But the idle/daily reset
    policy keys on ``updated_at``, which is bumped to ``now`` on every
    message — so a zombie session that keeps receiving messages never trips
    it and would resume stale context forever.  The freshness gate keys on
    ``last_resume_marked_at`` (set once at resume-mark, never bumped) so it
    catches that case.
    """

    def _mark_resume_pending(self, store, source):
        """Put the session into resume_pending and return the entry."""
        store.get_or_create_session(source)
        count = store.suspend_recently_active()
        assert count == 1
        with store._lock:
            entry = store._entries[store._generate_session_key(source)]
        assert entry.resume_pending
        assert entry.last_resume_marked_at is not None
        return entry

    def test_fresh_resume_pending_returns_same_session(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = self._mark_resume_pending(store, source)

        # Within the freshness window (marked just now) → same session back.
        refreshed = store.get_or_create_session(source)
        assert refreshed.session_id == entry.session_id
        assert refreshed.resume_pending

    def test_stale_resume_pending_falls_through_to_reset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "3600")
        store = _make_store(tmp_path)
        source = _make_source()
        entry = self._mark_resume_pending(store, source)

        # Backdate the resume mark past the freshness window. Keep updated_at
        # fresh (as a per-message zombie would have) so the idle/daily policy
        # would NOT fire — only the freshness gate should catch this.
        with store._lock:
            entry.last_resume_marked_at = datetime.now() - timedelta(seconds=7200)
            entry.updated_at = datetime.now()
            store._save()

        fresh = store.get_or_create_session(source)
        # Zombie detected → brand-new session, not the stale transcript.
        assert fresh.session_id != entry.session_id
        assert not fresh.resume_pending

    def test_freshness_gate_disabled_returns_stale_session(self, tmp_path, monkeypatch):
        # Opt-out: window <= 0 restores the pre-fix "always fresh" behaviour.
        monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "0")
        store = _make_store(tmp_path)
        source = _make_source()
        entry = self._mark_resume_pending(store, source)

        with store._lock:
            entry.last_resume_marked_at = datetime.now() - timedelta(seconds=999999)
            entry.updated_at = datetime.now()
            store._save()

        refreshed = store.get_or_create_session(source)
        assert refreshed.session_id == entry.session_id
        assert refreshed.resume_pending
