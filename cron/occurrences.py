"""Exact scheduled identities, independent of mutable jobs.json dispatch stamps."""
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


def scheduled_instant(value):
    """Canonicalize aware instants; legacy/ambiguous values carry no exact identity."""
    if not isinstance(value, str):
        return None
    try:
        instant = datetime.fromisoformat(value)
        if instant.tzinfo is None:
            return None
        return instant.astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def completed_occurrence(job, instant):
    """Unknown/failed/pruned attempts cannot prove completion: keep them eligible."""
    from cron.executions import _transaction

    instant = scheduled_instant(instant)
    if instant is None:
        return False
    try:
        with _transaction() as conn:
            return conn.execute(
                "SELECT 1 FROM executions WHERE job_id=? AND scheduled_instant=? "
                "AND status='completed' LIMIT 1", (str(job['id']), instant)
            ).fetchone() is not None
    except Exception:
        logger.warning("Cannot check completed occurrence for job %s", job['id'], exc_info=True)
        return False
