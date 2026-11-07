"""Main's ``gateway/session_viewer_close.py`` bodies (as of the pinned main), the legacy side of the backend parity checks."""

import logging
from hermes_state_mutation_retirement import VIEWER_CLOSE_PREFIX
from persistence.core.errors import AuthorityRefusal


logger = logging.getLogger(__name__)


CLOSE_PREFIX = VIEWER_CLOSE_PREFIX  # route retirement drops it too


_BUSY_SQL = ("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status IN ('queued','started','unknown') "
             "UNION ALL SELECT 1 FROM worker_executions WHERE session_id IN (?,?) "
             "AND status IN ('registered','running','unknown') LIMIT 1")


def _busy(authority, session_id, target):
    with authority.db._read_ctx() as conn:
        return conn.execute(_BUSY_SQL, (session_id, session_id, target)).fetchone() is not None


def _persist_close(authority, session_id, record):
    from hermes_state_runtime import _epoch

    def write(conn):
        _epoch(conn, authority.epoch)
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (CLOSE_PREFIX + session_id, record))
    authority.db._execute_write(write)


def forget_persisted_close(authority, session_id, record):
    """Blocking: drop *session_id*'s stored close only if it is still exactly *record* (that close
    completed or was cancelled), and only while this authority owns the store. Returns whether a
    row went."""
    from hermes_state_runtime import _epoch

    def write(conn):
        _epoch(conn, authority.epoch)
        return conn.execute('DELETE FROM state_meta WHERE key=? AND value=?',
                            (CLOSE_PREFIX + session_id, record)).rowcount
    try:
        return bool(authority.db._execute_write(write))
    except AuthorityRefusal as exc:
        logger.info('Stored close for %s left in place: %s', session_id, exc)
        return False
