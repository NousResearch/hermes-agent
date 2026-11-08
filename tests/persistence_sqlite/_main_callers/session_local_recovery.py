"""Main's ``gateway/session_local_recovery.py`` bodies (as of the pinned main), the legacy side of the backend parity checks."""

import logging


def unused_session(conn, session_id):
    """*session_id* holds nothing: no transcript, ledger rows or children (what the SessionStore's
    stale-route heal, #54878, mints as a route's replacement before any turn runs on it)."""
    return conn.execute(
        'SELECT 1 FROM messages WHERE session_id=? UNION ALL SELECT 1 FROM session_admissions '
        'WHERE target_session_id=? UNION ALL SELECT 1 FROM worker_executions WHERE session_id=? '
        'UNION ALL SELECT 1 FROM sessions WHERE parent_session_id=? LIMIT 1',
        (session_id, session_id, session_id, session_id)).fetchone() is None


def release_handed_off_receipts(db, *, epoch):
    """Boot heal for #143286: an older build's ``/handoff`` (or a platform ``/resume``) moved a local
    session onto a messaging lane but kept its local receipt, so every open, pin or rename refused it
    (``stored identity mismatch``) and its title never resolved. Retire each such receipt, so the row
    restores from its route like any other platform session, and move the rest of its conversation
    there with it (``hand_off_conversations``). Only receipts whose CURRENT target a messaging lane
    now holds: any other refused receipt stays as it is, logged when it is touched."""
    from gateway.session_route_recovery import _messaging_platform
    from hermes_state_local import hand_off_conversations, moved_receipts
    from hermes_state_runtime import _epoch

    def write(conn):
        _epoch(conn, epoch)
        moved = {owner: (target, source) for owner, (target, source) in moved_receipts(conn).items()
                 if _messaging_platform(source) is not None}
        return hand_off_conversations(conn, moved)
    for owner, (target, source) in db._execute_write(write).items():
        logging.getLogger(__name__).info(
            'Local session %s was handed off to %s (row %s); released its local receipt', owner, source, target)
