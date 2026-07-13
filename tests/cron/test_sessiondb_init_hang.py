"""Regression test for a hung SessionDB() init permanently wedging a cron job.

Real-world incident: a cron job's ``SessionDB()`` construction inside
``run_job`` blocked forever (a wedged sqlite3.connect against state.db, no
other process holding a competing lock by the time it was diagnosed). Because
that call had no timeout of its own — unlike the agent's run_conversation,
which is already bounded by HERMES_CRON_TIMEOUT — the worker thread submitted
by ``_submit_with_guard`` never returned. Its ``finally`` block, which is the
only thing that discards the job ID from ``_running_job_ids``, never ran.
Every later tick logged "already running — skipping" and the job never fired
again until the whole gateway process was restarted days later.

These tests prove ``run_job`` now bounds the SessionDB init with its own
timeout (HERMES_CRON_SESSION_DB_TIMEOUT, default 10s) so a hang there can
never again wedge the job past that bound, and — end to end — that the
dispatch guard is released and the job becomes dispatchable again afterward.

Note: each test releases its ``never_set`` event in a ``finally`` before
returning. concurrent.futures.thread registers an atexit hook that joins
EVERY worker thread ever created by ANY ThreadPoolExecutor in the process
regardless of ``shutdown(wait=False)`` — an event left permanently unset
would hang the whole test process at interpreter exit, not just this test.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler import run_job


def _hanging_session_db(never_set: threading.Event):
    """Stand-in for hermes_state.SessionDB() that blocks until released —
    like the real incident's wedged sqlite3.connect, but bounded so the test
    process can still exit cleanly once the assertions are done."""
    never_set.wait(timeout=30)
    return MagicMock()


class TestSessionDbInitTimeout:
    def test_run_job_does_not_hang_when_sessiondb_init_wedges(self, tmp_path, monkeypatch):
        """run_job returns promptly even if SessionDB() never returns."""
        monkeypatch.setenv("HERMES_CRON_SESSION_DB_TIMEOUT", "0.2")
        never_set = threading.Event()
        job = {"id": "wedged-sessiondb", "name": "test", "prompt": "hello"}

        try:
            with patch("cron.scheduler._hermes_home", tmp_path), \
                 patch("cron.scheduler._resolve_origin", return_value=None), \
                 patch("hermes_cli.env_loader.load_hermes_dotenv"), \
                 patch("hermes_cli.env_loader.reset_secret_source_cache"), \
                 patch("hermes_state.SessionDB", side_effect=lambda: _hanging_session_db(never_set)), \
                 patch(
                     "hermes_cli.runtime_provider.resolve_runtime_provider",
                     return_value={
                         "api_key": "test-key",
                         "base_url": "https://example.invalid/v1",
                         "provider": "openrouter",
                         "api_mode": "chat_completions",
                     },
                 ), \
                 patch("run_agent.AIAgent") as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "ok"}
                mock_agent_cls.return_value = mock_agent

                start = time.monotonic()
                success, output, final_response, error = run_job(job)
                elapsed = time.monotonic() - start
        finally:
            never_set.set()

        # Bounded by the 0.2s timeout, not by the hang (which never resolves
        # on its own within the test).
        assert elapsed < 5.0
        # The run still completes successfully without a session store.
        assert success is True
        assert final_response == "ok"
        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["session_db"] is None

    def test_invalid_timeout_env_falls_back_to_default(self, tmp_path, monkeypatch, caplog):
        """A malformed HERMES_CRON_SESSION_DB_TIMEOUT logs a warning and still
        bounds the call (mirrors HERMES_CRON_TIMEOUT's own fallback)."""
        monkeypatch.setenv("HERMES_CRON_SESSION_DB_TIMEOUT", "not-a-number")
        fake_db = MagicMock()
        job = {"id": "bad-timeout-env", "name": "test", "prompt": "hello"}

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "test-key",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is True
        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["session_db"] is fake_db  # default 10s was plenty for a MagicMock


class TestDispatchGuardReleasedAfterHang:
    """End-to-end: the real bug symptom was every later tick silently
    skipping the job forever. Confirm the fix actually clears that path."""

    def test_guard_is_released_and_job_refires_after_sessiondb_hang(self, tmp_path, monkeypatch):
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_SESSION_DB_TIMEOUT", "0.2")
        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._running_job_ids.clear()

        never_set = threading.Event()
        job = {
            "id": "guard-sessiondb-hang",
            "name": "guard-sessiondb-hang",
            "prompt": "hello",
            "schedule": "every 5m",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00",
            "deliver": "local",
        }

        try:
            with patch("cron.scheduler._hermes_home", tmp_path), \
                 patch("cron.scheduler._resolve_origin", return_value=None), \
                 patch("hermes_cli.env_loader.load_hermes_dotenv"), \
                 patch("hermes_cli.env_loader.reset_secret_source_cache"), \
                 patch("hermes_state.SessionDB", side_effect=lambda: _hanging_session_db(never_set)), \
                 patch(
                     "hermes_cli.runtime_provider.resolve_runtime_provider",
                     return_value={
                         "api_key": "test-key",
                         "base_url": "https://example.invalid/v1",
                         "provider": "openrouter",
                         "api_mode": "chat_completions",
                     },
                 ), \
                 patch("run_agent.AIAgent") as mock_agent_cls, \
                 patch.object(sched, "get_due_jobs", return_value=[job]), \
                 patch.object(sched, "advance_next_run"), \
                 patch.object(sched, "save_job_output", return_value="/tmp/out"), \
                 patch.object(sched, "mark_job_run"), \
                 patch.object(sched, "_deliver_result", return_value=None):
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "ok"}
                mock_agent_cls.return_value = mock_agent

                n = sched.tick(verbose=False)  # sync=True by default: waits for the job
                assert n == 1

                # Without the fix this would still contain the job ID forever.
                assert "guard-sessiondb-hang" not in sched.get_running_job_ids()

                # A second tick can dispatch the same job again — before the
                # fix this would log "already running — skipping" and
                # return 0.
                n2 = sched.tick(verbose=False)
                assert n2 == 1
        finally:
            never_set.set()
            sched._running_job_ids.discard("guard-sessiondb-hang")
            sched._shutdown_parallel_pool()
