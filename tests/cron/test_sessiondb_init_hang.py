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

Assertions capture the timeout passed to ``Future.result(timeout=...)`` (and
optionally force an immediate ``TimeoutError``) — no wall-clock waits, so the
suite stays free of timing flakes under parallel load.
"""

import concurrent.futures
from unittest.mock import MagicMock, patch

from cron.scheduler import run_job

# Hold the real class: patching cron.scheduler.concurrent.futures.ThreadPoolExecutor
# also replaces concurrent.futures.ThreadPoolExecutor (same module object).
_REAL_TPE = concurrent.futures.ThreadPoolExecutor

_RUNTIME = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
}


def _session_db_executor(timeouts: list, *, instant_timeout: bool = True):
    """Wrap ``ThreadPoolExecutor`` so SessionDB's ``result(timeout=...)`` is observable.

    ``run_job`` is the only caller that passes a timeout to ``Future.result`` on
    this path (``submit(SessionDB).result(timeout=...)``). Other pools used by
    ``tick`` / the agent inactivity watchdog call ``result()`` with no timeout
    and are left alone. When ``instant_timeout`` is True, the timed wait raises
    immediately instead of sleeping — the production hang path without a clock.
    """

    def factory(max_workers=1, *args, **kwargs):
        real = _REAL_TPE(max_workers=max_workers)
        orig_submit = real.submit

        def submit(fn, *a, **k):
            fut = orig_submit(fn, *a, **k)
            orig_result = fut.result

            def result(*ra, **rk):
                timeout = ra[0] if ra else rk.get("timeout")
                if timeout is not None:
                    timeouts.append(timeout)
                    if instant_timeout:
                        raise concurrent.futures.TimeoutError()
                return orig_result(*ra, **rk)

            fut.result = result
            return fut

        real.submit = submit
        return real

    return factory


class TestSessionDbInitTimeout:
    def test_run_job_does_not_hang_when_sessiondb_init_wedges(self, tmp_path, monkeypatch):
        """run_job proceeds without a session store when SessionDB init times out."""
        monkeypatch.setenv("HERMES_CRON_SESSION_DB_TIMEOUT", "0.2")
        job = {"id": "wedged-sessiondb", "name": "test", "prompt": "hello"}
        timeouts: list = []

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB"), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value=_RUNTIME,
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls, \
             patch(
                 "cron.scheduler.concurrent.futures.ThreadPoolExecutor",
                 side_effect=_session_db_executor(timeouts),
             ):
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        # Env-resolved bound was passed to Future.result — not the 10s default,
        # and not an unbounded call.
        assert timeouts == [0.2]
        assert success is True
        assert final_response == "ok"
        assert mock_agent_cls.call_args.kwargs["session_db"] is None

    def test_invalid_timeout_env_falls_back_to_default(self, tmp_path, monkeypatch, caplog):
        """A malformed HERMES_CRON_SESSION_DB_TIMEOUT logs a warning and still
        bounds the call (mirrors HERMES_CRON_TIMEOUT's own fallback)."""
        monkeypatch.setenv("HERMES_CRON_SESSION_DB_TIMEOUT", "not-a-number")
        fake_db = MagicMock()
        job = {"id": "bad-timeout-env", "name": "test", "prompt": "hello"}
        timeouts: list = []

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value=_RUNTIME,
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls, \
             patch(
                 "cron.scheduler.concurrent.futures.ThreadPoolExecutor",
                 side_effect=_session_db_executor(timeouts, instant_timeout=False),
             ):
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            with caplog.at_level("WARNING"):
                success, output, final_response, error = run_job(job)

        # Invalid env → fall back to default 10s bound (still passed to result).
        assert timeouts == [10.0]
        assert success is True
        assert mock_agent_cls.call_args.kwargs["session_db"] is fake_db
        assert any(
            "HERMES_CRON_SESSION_DB_TIMEOUT" in rec.message
            for rec in caplog.records
        ), f"Expected warning about invalid timeout env var; got: {[r.message for r in caplog.records]}"

    def test_timeout_resolved_from_config_yaml(self, tmp_path, monkeypatch):
        """cron.session_db_timeout_seconds in config.yaml is respected when
        the env var is not set — the canonical config-first resolution path."""
        import yaml

        monkeypatch.delenv("HERMES_CRON_SESSION_DB_TIMEOUT", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"cron": {"session_db_timeout_seconds": 0.2}})
        )
        job = {"id": "config-timeout", "name": "test", "prompt": "hello"}
        timeouts: list = []

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB"), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value=_RUNTIME,
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls, \
             patch(
                 "cron.scheduler.concurrent.futures.ThreadPoolExecutor",
                 side_effect=_session_db_executor(timeouts),
             ):
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        # Config value was passed through — not the 10s default.
        assert timeouts == [0.2]
        assert success is True
        assert mock_agent_cls.call_args.kwargs["session_db"] is None


class TestDispatchGuardReleasedAfterHang:
    """End-to-end: the real bug symptom was every later tick silently
    skipping the job forever. Confirm the fix actually clears that path."""

    def test_guard_is_released_and_job_refires_after_sessiondb_hang(self, tmp_path, monkeypatch):
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_SESSION_DB_TIMEOUT", "0.2")
        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._running_job_ids.clear()

        job = {
            "id": "guard-sessiondb-hang",
            "name": "guard-sessiondb-hang",
            "prompt": "hello",
            "schedule": "every 5m",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00",
            "deliver": "local",
        }
        timeouts: list = []

        try:
            with patch("cron.scheduler._hermes_home", tmp_path), \
                 patch("cron.scheduler._resolve_origin", return_value=None), \
                 patch("hermes_cli.env_loader.load_hermes_dotenv"), \
                 patch("hermes_cli.env_loader.reset_secret_source_cache"), \
                 patch("hermes_state.SessionDB"), \
                 patch(
                     "hermes_cli.runtime_provider.resolve_runtime_provider",
                     return_value=_RUNTIME,
                 ), \
                 patch("run_agent.AIAgent") as mock_agent_cls, \
                 patch(
                     "cron.scheduler.concurrent.futures.ThreadPoolExecutor",
                     side_effect=_session_db_executor(timeouts),
                 ), \
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
                assert timeouts == [0.2]

                # Without the fix this would still contain the job ID forever.
                assert "guard-sessiondb-hang" not in sched.get_running_job_ids()

                # A second tick can dispatch the same job again — before the
                # fix this would log "already running — skipping" and
                # return 0.
                n2 = sched.tick(verbose=False)
                assert n2 == 1
        finally:
            sched._running_job_ids.discard("guard-sessiondb-hang")
            sched._shutdown_parallel_pool()
