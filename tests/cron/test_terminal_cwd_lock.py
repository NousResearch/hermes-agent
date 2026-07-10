"""Tests for per-job workdir ContextVar isolation in cron/scheduler.py.

Workdir cron jobs pin their cwd via the ``_SESSION_CWD`` ContextVar (set by
``set_session_cwd``) for their whole agent run.  Because ContextVars are
per-context (not process-global like ``os.environ``), multiple workdir jobs
can run in parallel without colliding.  These tests assert that contract:
the ContextVar is set during the run and reset (not leaked) afterwards,
even when the run raises an exception.
"""

import threading


def test_session_cwd_is_per_context(tmp_path):
    """Two concurrent contexts with different _SESSION_CWD values don't collide."""
    from agent.runtime_cwd import set_session_cwd, resolve_agent_cwd
    import contextvars

    dir_a = tmp_path / "A"
    dir_b = tmp_path / "B"
    dir_a.mkdir()
    dir_b.mkdir()

    observed: dict = {}

    def worker(cwd: str, label: str):
        ctx = contextvars.copy_context()
        def _run():
            set_session_cwd(cwd)
            import time
            time.sleep(0.05)  # overlap with the other thread
            observed[label] = str(resolve_agent_cwd())
        ctx.run(_run)

    t1 = threading.Thread(target=worker, args=(str(dir_a), "A"))
    t2 = threading.Thread(target=worker, args=(str(dir_b), "B"))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    # Each thread saw its OWN cwd, not the other's.
    assert observed.get("A") == str(dir_a)
    assert observed.get("B") == str(dir_b)


def test_run_job_resets_session_cwd_when_body_raises(tmp_path):
    """A workdir job whose run_job body raises must still RESET _SESSION_CWD.

    Regression for the leak that would leave _SESSION_CWD pointing at the
    job's workdir after an exception, corrupting subsequent jobs in the same
    context.  This asserts the ContextVar is reset (to "") after a raising run.
    """
    from unittest.mock import MagicMock, patch
    import cron.scheduler as sched
    from agent.runtime_cwd import _SESSION_CWD

    workdir = tmp_path / "proj"
    workdir.mkdir()
    job = {"id": "boom-job", "name": "boom", "prompt": "hi", "workdir": str(workdir)}

    # Force a raise right after set_session_cwd is called — the exact window
    # where a leaked ContextVar would corrupt subsequent jobs.
    real_info = sched.logger.info

    def _raise_on_workdir_log(msg, *args, **kwargs):
        if isinstance(msg, str) and "using workdir" in msg:
            raise RuntimeError("boom")
        return real_info(msg, *args, **kwargs)

    _SESSION_CWD.set("")

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch.object(sched.logger, "info", side_effect=_raise_on_workdir_log), \
         patch("hermes_state.SessionDB", return_value=MagicMock()):
        # run_job catches its own body exceptions and returns (False, ...);
        # it must not propagate, and it must reset _SESSION_CWD either way.
        success, _out, _final, _err = sched.run_job(job)

    assert success is False

    # If _SESSION_CWD leaked, it would still be the workdir path.
    # With the fix, the finally block resets it to "" (or the prior value).
    assert _SESSION_CWD.get() == "", \
        "_SESSION_CWD was leaked by run_job on exception"
