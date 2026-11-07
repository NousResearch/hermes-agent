"""Main's ``gateway/session_settlement_recovery.py`` bodies (as of the pinned main), the legacy side of the backend parity checks."""

from hermes_state_runtime import _admission, _epoch, settle_steer_transfers
from persistence.core.errors import AuthorityRefusal


def _mark_unknown(authority, row):
    transfers = _transfers(authority, authority.sessions.get(row['target_session_id']), row)

    def write(conn):
        _epoch(conn, authority.epoch)
        current = _admission(conn, row['admission_id'])
        if current['owner_epoch'] != authority.epoch or current['generation'] != row['generation']:
            raise AuthorityRefusal('stale_generation')
        if current['status'] in {'terminal', 'unknown'}:
            return current['status']
        if current['status'] != 'started':
            raise AuthorityRefusal('stale_generation')
        conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (row['admission_id'],))
        # A steer it carried is never replayed: interrupted if delivered, else queued again.
        settle_steer_transfers(conn, epoch=authority.epoch, generation=row['generation'],
                               transfers=transfers, outcome='unknown')
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (row['target_session_id'],))
        return 'unknown'
    return authority.db._execute_write(write)


def _transfers(authority, live, row):
    """The steers handed to this claim (``prompt.steer_queued``), read once its execution returned."""
    if live is None:
        return None
    from gateway.session_queued_controls import steer_transfers
    with live.event_stream.lock:
        return steer_transfers(authority, row['target_session_id'], row['generation'])
