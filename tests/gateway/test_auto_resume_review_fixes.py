from unittest.mock import AsyncMock, MagicMock, patch

import gateway.run as gateway_run
from gateway.config import GatewayConfig
from tests.gateway.test_clean_shutdown_marker import _make_source, _make_store


class TestShutdownTimeoutSnapshot:
    def test_timeout_marks_returned_timed_out_snapshot_not_mutated_running_agents(self, tmp_path, monkeypatch):
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        store = _make_store(tmp_path)
        timed_out_source = _make_source(chat_id="timed-out")
        timed_out_entry = store.get_or_create_session(timed_out_source)
        other_source = _make_source(chat_id="other")
        other_entry = store.get_or_create_session(other_source)

        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._restart_requested = True
        runner._restart_detached = False
        runner._restart_via_service = False
        runner._restart_task_started = False
        runner._running = True
        runner._draining = False
        runner._stop_task = None
        runner._running_agents = {other_entry.session_key: MagicMock()}
        runner._running_agents_ts = {}
        runner._pending_messages = {}
        runner._pending_approvals = {}
        runner._background_tasks = set()
        runner._shutdown_event = MagicMock()
        runner._restart_drain_timeout = 5
        runner._exit_code = None
        runner._exit_reason = None
        runner.adapters = {}
        runner.config = GatewayConfig()
        runner.session_store = store

        timed_out_snapshot = {timed_out_entry.session_key: MagicMock()}

        with patch("gateway.run.GatewayRunner._drain_active_agents", new_callable=AsyncMock, return_value=(timed_out_snapshot, True)), \
             patch("gateway.run.GatewayRunner._notify_active_sessions_of_shutdown", new_callable=AsyncMock), \
             patch("gateway.run.GatewayRunner._finalize_shutdown_agents"), \
             patch("gateway.run.GatewayRunner._interrupt_running_agents"), \
             patch("gateway.run.GatewayRunner._update_runtime_status"), \
             patch("gateway.status.remove_pid_file"), \
             patch("tools.process_registry.process_registry") as mock_proc_reg, \
             patch("tools.terminal_tool.cleanup_all_environments"), \
             patch("tools.browser_tool.cleanup_all_browsers"):
            mock_proc_reg.kill_all = MagicMock()

            import asyncio
            asyncio.get_event_loop().run_until_complete(runner.stop(restart=True))

        timed_out_refreshed = store.get_or_create_session(timed_out_source)
        other_refreshed = store.get_or_create_session(other_source)
        assert timed_out_refreshed.resume_pending is True
        assert timed_out_refreshed.resume_reason == "restart_timeout"
        assert other_refreshed.resume_pending is False


class TestShutdownTimeoutCopy:
    def test_shutdown_timeout_note_mentions_shutdown_not_restart(self):
        result = gateway_run._prepend_restart_recovery_note(
            "continue",
            [],
            resume_pending=True,
            resume_reason="shutdown_timeout",
        )

        assert "gateway shutdown" in result
        assert "gateway restart" not in result

class TestResumePendingRecoveryWindow:
    def test_expired_resume_pending_falls_back_to_normal_reset_policy(self, tmp_path, monkeypatch):
        from datetime import datetime, timedelta

        import gateway.session as gateway_session
        from gateway.config import SessionResetPolicy

        base_time = datetime(2026, 4, 17, 12, 0, 0)
        monkeypatch.setattr(gateway_session, "_now", lambda: base_time)

        policy = SessionResetPolicy(mode="idle", idle_minutes=1)
        store = _make_store(tmp_path, policy=policy)
        source = _make_source(chat_id="resume-window")
        entry = store.get_or_create_session(source)
        store.mark_resume_pending(entry.session_key, reason="restart_timeout")

        monkeypatch.setattr(gateway_session, "_now", lambda: base_time + timedelta(minutes=10))
        resumed = store.get_or_create_session(source)

        assert resumed.session_id != entry.session_id
        assert resumed.was_auto_reset is True
        assert resumed.auto_reset_reason == "idle"


class TestResumePendingEscalation:
    def test_repeated_resume_marking_does_not_suspend_without_stuck_loop_counter(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source(chat_id="escalate")
        entry = store.get_or_create_session(source)

        assert store.mark_resume_pending(entry.session_key, reason="restart_timeout") is True
        assert store.mark_resume_pending(entry.session_key, reason="restart_timeout") is True
        assert store.mark_resume_pending(entry.session_key, reason="restart_timeout") is True

        refreshed = store.get_or_create_session(source)
        assert refreshed.session_id == entry.session_id
        assert refreshed.resume_pending is True
        assert refreshed.suspended is False
        assert not hasattr(refreshed, "resume_attempts")
