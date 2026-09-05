"""Regression tests for bounded cron post-run cleanup.

A cron worker must release its in-memory dispatch guard even when SQLite or an
agent resource finalizer stops returning after the model turn has ended.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from cron.scheduler import (
    _defer_cron_worker_teardown_if_running,
    _teardown_cron_agent,
    run_job,
)


_RUNTIME = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
}


class HangingSessionDB:
    def __init__(self, release: threading.Event):
        self.release = release
        self.entered = threading.Event()

    def get_compression_tip(self, _session_id):
        self.entered.set()
        self.release.wait()
        return None

    def end_session(self, *_args, **_kwargs):
        return None

    def close(self):
        return None


class HangingAgent:
    def __init__(self, release: threading.Event):
        self.release = release
        self.entered = threading.Event()

    def close(self):
        self.entered.set()
        self.release.wait()


def test_run_job_bounds_sessiondb_finalization(tmp_path):
    release = threading.Event()
    fake_db = HangingSessionDB(release)
    job = {"id": "cleanup-sessiondb-hang", "name": "test", "prompt": "hello"}

    try:
        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state_registry.acquire", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls, \
             patch("cron.scheduler._cron_cleanup_timeout_seconds", return_value=0.02):
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            started = time.monotonic()
            success, _output, final_response, error = run_job(job)
            elapsed = time.monotonic() - started

        assert fake_db.entered.wait(timeout=2.0)
        assert elapsed < 5.0
        assert success is True
        assert final_response == "ok"
        assert error is None
    finally:
        release.set()


def test_agent_teardown_is_bounded():
    release = threading.Event()
    agent = HangingAgent(release)

    try:
        started = time.monotonic()
        _teardown_cron_agent(agent, "cleanup-agent-hang", timeout_seconds=0.02)
        elapsed = time.monotonic() - started

        assert agent.entered.wait(timeout=2.0)
        assert elapsed < 5.0
    finally:
        release.set()


def test_detached_worker_teardown_waits_for_future():
    """A timed-out worker keeps its agent and SessionDB until completion."""
    future = Future()
    worker_state = {"future": future}
    fake_db = MagicMock()
    agent = MagicMock()

    with patch("cron.scheduler._teardown_detached_cron_worker") as teardown_worker:
        assert _defer_cron_worker_teardown_if_running(
            worker_state,
            fake_db,
            agent,
            "detached-worker",
            "detached worker",
            "cron_detached-worker",
        ) is True
        teardown_worker.assert_not_called()

        future.set_result({"final_response": "late"})

        teardown_worker.assert_called_once_with(
            fake_db,
            agent,
            "detached-worker",
            "detached worker",
            "cron_detached-worker",
        )


def test_dispatch_guard_releases_after_sessiondb_finalization_hang(tmp_path):
    """A second scheduler tick can fire the same job after cleanup times out."""
    import cron.scheduler as sched

    release = threading.Event()
    fake_db = HangingSessionDB(release)
    job = {
        "id": "cleanup-guard-hang",
        "name": "cleanup-guard-hang",
        "prompt": "hello",
        "schedule": "every 5m",
        "enabled": True,
        "next_run_at": "2020-01-01T00:00:00",
        "deliver": "local",
    }
    sched._parallel_pool = None
    sched._parallel_pool_max_workers = None
    sched._running_job_ids.clear()

    try:
        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state_registry.acquire", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls, \
             patch("cron.scheduler._cron_cleanup_timeout_seconds", return_value=0.02), \
             patch.object(sched, "get_due_jobs", return_value=[job]), \
             patch.object(sched, "advance_next_runs"), \
             patch.object(sched, "save_job_output", return_value="/tmp/out"), \
             patch.object(sched, "mark_job_run"), \
             patch.object(sched, "_deliver_result", return_value=None):
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            assert sched.tick(verbose=False) == 1
            assert "cleanup-guard-hang" not in sched.get_running_job_ids()
            assert sched.tick(verbose=False) == 1
    finally:
        release.set()
        sched._running_job_ids.discard("cleanup-guard-hang")
        sched._shutdown_parallel_pool()


def test_finalize_cron_session_disarms_end_session_on_close_only_on_success():
    """_end_session_on_close is disarmed ONLY when session_db.end_session succeeds."""
    from cron.scheduler import _finalize_cron_session

    # Case 1: end_session succeeds -> _end_session_on_close is disarmed (False)
    fake_db_success = MagicMock()
    fake_agent_success = MagicMock()
    fake_agent_success._end_session_on_close = True

    with patch("hermes_state_registry.release_or_close"):
        _finalize_cron_session(
            fake_db_success, fake_agent_success, "job-1", "job one", "cron_session_1"
        )
    assert fake_agent_success._end_session_on_close is False
    fake_db_success.end_session.assert_called_once()

    # Case 2: end_session raises -> _end_session_on_close remains armed (True)
    fake_db_fail = MagicMock()
    fake_db_fail.end_session.side_effect = RuntimeError("database locked")
    fake_agent_fail = MagicMock()
    fake_agent_fail._end_session_on_close = True

    with patch("hermes_state_registry.release_or_close"):
        _finalize_cron_session(
            fake_db_fail, fake_agent_fail, "job-2", "job two", "cron_session_2"
        )
    assert fake_agent_fail._end_session_on_close is True
    fake_db_fail.end_session.assert_called_once()

