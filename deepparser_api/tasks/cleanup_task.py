from __future__ import annotations

import asyncio
import logging

from .. import db

logger = logging.getLogger(__name__)

_EXPIRED_STATUSES = ("READY", "PARSE_FAILED", "TIMEOUT", "PARSING")


async def cleanup_expired_jobs() -> None:
    """Delete jobs older than 24h regardless of status (PARSING covers orphans)."""
    async with db.connect() as conn:
        cursor = await conn.execute(
            """
            DELETE FROM parse_jobs
            WHERE status IN ({}) AND created_at < datetime('now', '-24 hours')
            """.format(",".join("?" * len(_EXPIRED_STATUSES))),
            _EXPIRED_STATUSES,
        )
        deleted = cursor.rowcount
        await conn.commit()
    if deleted:
        logger.info("cleanup deleted %d expired jobs", deleted)


async def run_cleanup_loop() -> None:
    """Run cleanup every hour forever. Survives individual DB errors."""
    while True:
        await asyncio.sleep(3600)
        try:
            await cleanup_expired_jobs()
        except Exception:
            logger.exception("cleanup loop error (non-fatal)")
