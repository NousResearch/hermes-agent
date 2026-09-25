"""Regression for #86721 — a one-shot `hermes cron run` invocation's
dispatched runner thread dies with the exiting process, leaving a stale
'claimed'/'running' row in cron/executions.db that blocks every subsequent
manual run of the same job. recover_interrupted_executions() already
existed and is correctly implemented, but was only ever called at the
long-lived scheduler ticker's own startup -- a one-shot CLI invocation has
no equivalent moment of its own, so the self-heal never ran for it.
"""

from __future__ import annotations

import sys








def test_one_shot_cli_path_reaps_dead_owner_row_before_returning_sync(tmp_path, monkeypatch):
    """The real one-shot `hermes cron run` shape (``_SESSION_ASYNC_DELIVERY`` scoped to False,
    as hermes_cli/cron.py::_job_action does) must still reap a 'running' row whose owner pid
    is provably dead before falling back to the sync run — the early return used to skip it."""
    import subprocess as sp
    import sqlite3

    from cron import executions
    from gateway.session_context import _SESSION_ASYNC_DELIVERY
    from tools import cronjob_tools

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    row = executions.create_execution("dead-owner-job", source="direct")
    proc = sp.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    dead_pid = proc.pid  # exited: provably dead
    with sqlite3.connect(tmp_path / "executions.db") as conn:
        conn.execute(
            "UPDATE executions SET status='running', process_id='other-process', pid=?, "
            "process_started_at=1 WHERE id=?", (dead_pid, row["id"]))

    token = _SESSION_ASYNC_DELIVERY.set(False)
    try:
        assert cronjob_tools._try_dispatch_background_run(
            {"id": "dead-owner-job", "name": "dead owner job", "deliver": "local"}) is None
    finally:
        _SESSION_ASYNC_DELIVERY.reset(token)

    assert executions.get_execution(row["id"])["status"] == "unknown"
