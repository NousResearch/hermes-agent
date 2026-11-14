"""Retry only pre-execution SQLite contention within the handoff observation budget."""

import sqlite3
import time


def adopt_with_retry(execution_id: str, deadline: float):
    """Keep the existing CAS as the only authority; never replay a job or a refused CAS.

    The first attempt retains legacy cold-start behavior even after the observation window.
    Subsequent attempts need room for the ledger's five-second busy timeout. This bounds
    retry scheduling, not filesystem I/O or interpreter scheduling latency.
    """
    # Resolve the ledger at call time, following cron's late-import convention.
    from cron.executions import adopt_claimed_execution

    while True:
        try:
            return adopt_claimed_execution(execution_id)
        except sqlite3.OperationalError as exc:
            code = getattr(exc, "sqlite_errorcode", 0)
            if code & 0xFF not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise
            # executions._connect uses sqlite_util.open_db's 5000 ms busy timeout:
            # reserve 5.0s for it plus 0.25s backoff before sleeping (5.25s total).
            # The original exception remains the startup diagnostic.
            if deadline - time.monotonic() <= 5.25:
                raise
            time.sleep(0.25)
            if deadline - time.monotonic() <= 5.0:
                raise
