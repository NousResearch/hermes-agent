"""Main's ``gateway/session_api_turn.py`` bodies (as of the pinned main), the legacy side of the backend parity checks."""

from hermes_state_runtime import _json
from persistence.core.errors import AuthorityRefusal


_SETTINGS_PREFIX = 'gateway.api.settings.v1.'


def _admit_target(session_id, request_id, settings, conn, *, client_key=True):
    """Inside the admission's own off-loop transaction: one target per request id, and the
    session's settings for its later payload-less (internal) turns, committed with the admitted
    turn rather than by the drain on the owner loop (N50)."""
    _one_target(session_id, request_id, conn, client_key=client_key)
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                 (_SETTINGS_PREFIX + session_id, _json(settings)))


def _one_target(session_id, request_id, conn, *, client_key=True):
    """One API request id (an Idempotency-Key) names one admission. The ledger keys identity by
    target too, so the same key aimed at another session would otherwise run the work again.
    A deleted session's admission is gone from the ledger but its tombstone still spends the key
    (a target that already holds the key replays before this guard runs). Only a client key can
    name a tombstone: a server-minted id (key-less turn, run id) is fresh, so it skips that scan."""
    if conn.execute("SELECT 1 FROM session_admissions WHERE principal_id='api' AND request_id=? "
                    'AND target_session_id!=? LIMIT 1', (request_id, session_id)).fetchone():
        raise AuthorityRefusal('admission_conflict')
    if not client_key:
        return
    from hermes_state_terminal import terminal_api_admissions
    if any(row['target_session_id'] != session_id for row in terminal_api_admissions(conn, request_id)):
        raise AuthorityRefusal('admission_conflict')
