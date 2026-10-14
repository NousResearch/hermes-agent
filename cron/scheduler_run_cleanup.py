"""Caller exit, delivery deferral and actual cleanup completion for a cron run."""
from __future__ import annotations

from contextvars import copy_context
from functools import partial
import logging

logger = logging.getLogger("cron.scheduler")


def close_cron_agent_resources(agent, job_id, *, on_finish=None):
    try:
        try:
            if agent is not None:
                agent.close()
        except (Exception, KeyboardInterrupt) as exc:
            logger.warning("Job '%s': failed to close agent resources: %s", job_id, exc, exc_info=True)
        try:
            from agent.auxiliary_client import cleanup_stale_async_clients
            cleanup_stale_async_clients()
        except Exception as exc:
            logger.warning("Job '%s': failed to reap stale auxiliary clients: %s", job_id, exc, exc_info=True)
    finally:
        # A bounded caller may have returned while this cleanup thread was still running.
        if on_finish is not None:
            on_finish()


def _finish_inline(scope, session_db, agent, job_id, job_name, session_id, deferred_agents):
    from cron.scheduler import _finalize_cron_session, _teardown_cron_agent

    try:
        if session_db:
            _finalize_cron_session(session_db, agent, job_id, job_name, session_id,
                                   workdir=scope.workdir)
    finally:
        queued = False
        try:
            if deferred_agents is not None and agent is not None:
                # Delivery keeps the live agent; its profile-bound cleanup also retains the owner.
                context = copy_context()
                deferred_agents.append(partial(context.run, _teardown_cron_agent,
                    agent, job_id, on_finish=scope.release))
                queued = True
        finally:
            if not queued:
                _teardown_cron_agent(agent, job_id, on_finish=scope.release)


def finish_run(scope, future, session_db, agent, job_id, job_name, session_id, deferred_agents):
    from cron.scheduler_detached_worker import defer_teardown_to_running_worker

    deferred = False
    try:
        deferred = defer_teardown_to_running_worker(
            future, session_db, agent, job_id, job_name, session_id,
            workdir=scope.workdir, on_finish=scope.release)
    finally:
        try:
            # ContextVar tokens belong to this caller, never the Future callback's context.
            scope.exit()
        finally:
            if not deferred:
                _finish_inline(scope, session_db, agent, job_id, job_name, session_id, deferred_agents)
