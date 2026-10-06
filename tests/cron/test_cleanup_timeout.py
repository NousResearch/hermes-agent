"""Regression tests for bounded cron post-run cleanup.

A cron worker must release its in-memory dispatch guard even when SQLite or an
agent resource finalizer stops returning after the model turn has ended.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from cron.scheduler import _finalize_cron_session, _teardown_cron_agent, run_job
from cron.scheduler_detached_worker import defer_teardown_to_running_worker


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
    """A timed-out worker keeps its agent and SessionDB until its Future completes."""
    future = Future()
    fake_db = MagicMock()
    agent = MagicMock()

    with patch("cron.scheduler._finalize_cron_session") as finalize, \
         patch("cron.scheduler._teardown_cron_agent") as teardown_agent:
        assert defer_teardown_to_running_worker(
            future, fake_db, agent, "detached-worker", "detached worker", "cron_detached-worker") is True
        finalize.assert_not_called()
        teardown_agent.assert_not_called()

        future.set_result({"final_response": "late"})

        finalize.assert_called_once_with(fake_db, agent, "detached-worker", "detached worker", "cron_detached-worker")
        teardown_agent.assert_called_once_with(agent, "detached-worker")
    assert defer_teardown_to_running_worker(
        future, fake_db, agent, "detached-worker", "detached worker", "cron_detached-worker") is False


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
    sched._parallel_pools.clear()
    sched._parallel_pool_max_workers.clear()
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
        sched._running_job_ids.discard(sched._inflight_key("cleanup-guard-hang"))
        sched._shutdown_parallel_pool()


class _StallableSessionDB:
    """SessionDB double that can stall the first call of selected methods."""

    def __init__(self, stall_methods: set[str], stall_duration: float):
        self.stall_methods = stall_methods
        self.stall_duration = stall_duration
        self._stalled: set[str] = set()
        self.calls: list[tuple[str, tuple]] = []
        self.ended_reasons: dict[str, str] = {}

    def _maybe_stall(self, name: str) -> None:
        if name in self.stall_methods and name not in self._stalled:
            self._stalled.add(name)
            time.sleep(self.stall_duration)

    def get_compression_tip(self, session_id):
        self.calls.append(("get_compression_tip", (session_id,)))
        self._maybe_stall("get_compression_tip")
        return None

    def get_next_title_in_lineage(self, base_title):
        self.calls.append(("get_next_title_in_lineage", (base_title,)))
        return f"{base_title} #2"

    def session_lifecycle_statuses(self, session_ids):
        self.calls.append(("session_lifecycle_statuses", (tuple(session_ids),)))
        return {sid: "complete" for sid in session_ids}

    def set_session_title(self, session_id, title):
        self.calls.append(("set_session_title", (session_id, title)))
        self._maybe_stall("set_session_title")
        return title

    def end_session(self, session_id, end_reason):
        self.calls.append(("end_session", (session_id, end_reason)))
        self._maybe_stall("end_session")
        self.ended_reasons.setdefault(session_id, end_reason)

    def close(self):
        return None


def _finalize_with_stall(stall_methods):
    store = _StallableSessionDB(stall_methods, stall_duration=0.4)
    agent = MagicMock()
    agent._end_session_on_close = True
    with patch("cron.scheduler._cron_cleanup_timeout_seconds", return_value=0.02), \
         patch("hermes_state_registry.release_or_close"):
        _finalize_cron_session(store, agent, "job-76914", "job 76914", "sess-1")
    return store, agent


def test_finalize_retry_writes_ended_at_after_title_stall():
    store, agent = _finalize_with_stall({"set_session_title"})
    assert store.ended_reasons.get("sess-1") == "cron_complete"
    assert agent._end_session_on_close is False


def test_finalize_retry_writes_ended_at_after_end_session_stall():
    store, agent = _finalize_with_stall({"end_session"})
    assert store.ended_reasons.get("sess-1") == "cron_complete"
    assert agent._end_session_on_close is False


def test_finalize_retry_failure_keeps_agent_armed():
    store = _StallableSessionDB(set(), stall_duration=0.0)

    def _broken_end(_session_id, _end_reason):
        raise sqlite3.OperationalError("disk I/O error")

    store.end_session = _broken_end
    agent = MagicMock()
    agent._end_session_on_close = True
    with patch("cron.scheduler._cron_cleanup_timeout_seconds", return_value=0.02), \
         patch("hermes_state_registry.release_or_close"):
        _finalize_cron_session(store, agent, "job-76914", "job 76914", "sess-1")
    assert agent._end_session_on_close is True


def test_finalize_normal_path_ends_session_once():
    store, agent = _finalize_with_stall(set())
    end_calls = [call for call in store.calls if call[0] == "end_session"]
    assert len(end_calls) == 1
    assert store.ended_reasons.get("sess-1") == "cron_complete"
    assert agent._end_session_on_close is False
