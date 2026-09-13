"""Cron: teardown of a worker that outlived its ``run_job``.

``ThreadPoolExecutor.shutdown(wait=False)`` after an inactivity timeout does not stop a
worker already inside ``run_conversation``. Finalizing its SessionDB from ``run_job``'s
``finally`` would close a handle the worker is still writing to — the checkpoint/WAL-unlink
overlap behind #102827. The worker's Future owns the teardown instead.
"""

from __future__ import annotations

import concurrent.futures
from typing import Optional

from agent.runtime_policy import is_authoritative


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
            from cron.scheduler import logger
            policy = getattr(agent, "runtime_policy", None)
            final_session_id = getattr(agent, "session_id", None) or cron_session_id
            if is_authoritative(policy):
                from cron.scheduler_settlement import settle_run
                agent._cron_settlement_failed = True
                try:
                    settle_run(agent, agent._cron_job, agent._cron_execution_id,
                               final_session_id, {"failed": True})
                except Exception:
                    logger.warning("Job '%s': detached policy finalization failed", job_id, exc_info=True)
            else:
                from hermes_cli.lifecycle import finalize_session
                try:
                    finalize_session(
                        session_id=final_session_id, runtime_run_id=agent.runtime_task_id,
                        runtime_policy=policy, platform="cron", cron_job_id=job_id,
                        cron_job_name=job_name, cron_max_turns=agent.cron_max_turns,
                        completed=False, failed=True, terminal_outcome=None,
                    )
                except Exception:
                    logger.warning("Job '%s': detached session finalization failed", job_id, exc_info=True)
            if session_db:
                _finalize_cron_session(session_db, agent, job_id, job_name, cron_session_id)
        finally:
            _teardown_cron_agent(agent, job_id)

    # Runs inline if the worker finished between done() and here — still exactly once.
    future.add_done_callback(_finish)
    return True
