"""Retry only pre-execution SQLite contention within the handoff observation budget."""

import sqlite3
import time


def adopt_with_retry(execution_id: str, deadline: float):
    """Keep the existing CAS as the only authority; never replay a job or a refused CAS.

    The first attempt retains legacy cold-start behavior even after the observation window.
    Subsequent attempts need room for the ledger's five-second busy timeout. This bounds
    retry scheduling, not filesystem I/O or interpreter scheduling latency.
    """
    from cron.executions import adopt_claimed_execution

    while True:
        try:
            return adopt_claimed_execution(execution_id)
        except sqlite3.OperationalError as exc:
            code = getattr(exc, "sqlite_errorcode", 0)
            if code & 0xFF not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise
            # Do not start a fresh five-second lock wait at the very end of the parent's
            # observation window. The original exception remains the startup diagnostic.
            if deadline - time.monotonic() <= 5.25:
                raise
            time.sleep(0.25)
            if deadline - time.monotonic() <= 5.0:
                raise
