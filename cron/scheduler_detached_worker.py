"""Cron: teardown of a worker that outlived its ``run_job``.

``ThreadPoolExecutor.shutdown(wait=False)`` after an inactivity timeout does not stop a
worker already inside ``run_conversation``. Finalizing its SessionDB from ``run_job``'s
``finally`` would close a handle the worker is still writing to — the checkpoint/WAL-unlink
overlap behind #102827. The worker's Future owns the teardown instead.
"""

from __future__ import annotations

import concurrent.futures
import subprocess
import threading
from typing import Optional


def defer_teardown_to_running_worker(
    future: Optional[concurrent.futures.Future], session_db, agent, job_id: str, job_name: str,
    cron_session_id: str,
) -> bool:
    """Return True when the worker is still running and its Future will finalize the session
    and tear the agent down on completion; False when the caller must do it now."""
    if future is None or future.done():
        return False
    from cron.scheduler import _finalize_cron_session, _teardown_cron_agent

    def _finish(_future) -> None:
        try:
            if session_db:
                _finalize_cron_session(session_db, agent, job_id, job_name, cron_session_id)
        finally:
            _teardown_cron_agent(agent, job_id)

    # Runs inline if the worker finished between done() and here — still exactly once.
    future.add_done_callback(_finish)
    return True


def reap_terminal_worker_in_background(process: subprocess.Popen) -> None:
    """Keep the reap contract when the waiter returns before the worker exits.

    The ledger turning terminal lets ``_wait_for_external_cron_worker_body``
    return while the worker is still in final teardown. The gateway remains the
    worker's parent, so if nobody calls ``wait()`` afterwards the worker lingers
    as a zombie (STAT=Z) under the gateway until it is restarted (#114509). A
    short-lived daemon thread holds that single responsibility and ends with
    the process exit it waits for.
    """
    threading.Thread(
        target=process.wait,
        name=f"cron-worker-reap-{getattr(process, 'pid', '?')}",
        daemon=True,
    ).start()
