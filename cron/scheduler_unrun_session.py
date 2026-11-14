"""Cron: end the owner's session row for a fire whose agent never ran.

``gateway/session_cron.py`` commits a fire's session row before ``run_job``. Once the agent's
SessionDB is open, ``_finalize_cron_session`` (or a detached worker's teardown) ends that row; a
fire that returns before then (monitor output unchanged, ``wakeAgent=false``, blocked setup, a
failure before the agent) would otherwise leave it open until a restart or the stale-open sweep.
"""

import logging

logger = logging.getLogger(__name__)

# Distinct from ``cron_complete``/``cron_incomplete_no_output``: nothing answered in this session.
CRON_AGENT_SKIPPED = "cron_agent_skipped"


def end_unrun_owner_session(owner_execution, job_id: str) -> None:
    """End the row of the owner fire ``owner_execution`` (``current_execution()``); no-op without one.

    ``end_session`` only stamps an open row, so a row something else already ended keeps its reason."""
    if owner_execution is None:
        return
    from cron.scheduler import _BoundedCronSessionDB
    try:
        _BoundedCronSessionDB(owner_execution[0].db, job_id).end_session(owner_execution[1], CRON_AGENT_SKIPPED)
    except (Exception, KeyboardInterrupt) as e:
        logger.debug("Job '%s': failed to end unrun cron session: %s", job_id, e, exc_info=True)
