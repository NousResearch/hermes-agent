"""Regression tests for bounded cron post-run cleanup.

A cron worker must release its in-memory dispatch guard even when SQLite or an
agent resource finalizer stops returning after the model turn has ended.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from cron.scheduler import run_job, _teardown_cron_agent


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
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls, \
             patch("cron.scheduler._cron_cleanup_timeout_seconds", return_value=0.02):
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            started = time.monotonic()
            success, _output, final_response, error = run_job(job)
            elapsed = time.monotonic() - started

        assert fake_db.entered.wait(timeout=0.5)
        assert elapsed < 0.5
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

        assert agent.entered.wait(timeout=0.5)
        assert elapsed < 0.5
    finally:
        release.set()


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
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
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
