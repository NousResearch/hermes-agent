"""The store-wide epoch, the pending viewer closes and the two ledger-gated ends, with the epoch, close and ledger
reads the gateway, TUI and dashboard make."""

import time
from pathlib import Path
from typing import Optional

from persistence.core.errors import AuthorityRefusal

from ._runtime import _epoch, _text


def _profile_home_of(holder) -> Optional[Path]:
    """The profile home that owns *holder*'s store (the state.db's directory), or None.

    None for a store with no directory of its own (an in-memory or URI database): there is no
    profile home, so there is no profile lock to be fenced out of."""
    raw = str(getattr(holder, "db_path", "") or "")
    if not raw or raw == ':memory:' or raw.startswith('file:'):
        return None
    return Path(raw).parent


def _assert_profile_unowned(holder) -> None:
    """Refuse an epoch bump while another process holds the profile's ``gateway.lock``.

    The bump is a takeover, and the epoch is the fence every write is checked against: the
    owner captures it ONCE at boot, so a bump by a process that does not own the profile
    leaves the live owner permanently behind the fence (#142330).

    A reservation this process holds right now is ours: the gateway, a served secondary or a
    maintenance reservation here (a flock probe from this process would contend with its own
    lock). The lock record never grants it: it outlives a release, and a recorded pid cannot be
    checked across PID namespaces (#123081). Otherwise the flock decides, which also answers
    before a holder publishes its record. No lock file means no owner: the bump is the takeover."""
    from gateway.runtime_ownership import canonical_home, held_by_this_process, runtime_lock_held_elsewhere
    from gateway.status import _live_pid_from_record, _read_gateway_lock_record

    home = _profile_home_of(holder)
    if home is None:
        return
    home = canonical_home(home)
    lock_path = home / 'gateway.lock'
    if held_by_this_process(home) or not lock_path.exists() or not runtime_lock_held_elsewhere(home):
        return
    record = _read_gateway_lock_record(lock_path)
    owner = _live_pid_from_record(record) if record else None
    raise AuthorityRefusal(
        'ownership_conflict',
        field='runtime_epoch',
        detail=f'profile {home} is owned by {f"pid {owner}" if owner else "another process"}',
    )


def begin_runtime_epoch(holder, *, instance_id: str) -> int:
    """Caller MUST hold the profile ownership lock, not merely an endpoint ticket.

    The bump is the takeover of the store, and the owner captures the epoch once at boot:
    a bump from a process that does not own the profile leaves the live owner permanently
    behind the fence, and every write on that store then fails with ``stale_epoch`` until
    the owner restarts (#142330). The ownership is therefore verified here, not assumed."""
    _text(instance_id)
    _assert_profile_unowned(holder)
    def write(conn):
        conn.execute('''INSERT INTO runtime_epoch(singleton,epoch,instance_id) VALUES(1,1,?)
            ON CONFLICT(singleton) DO UPDATE SET epoch=runtime_epoch.epoch+1, instance_id=excluded.instance_id''', (instance_id,))
        return conn.execute('SELECT epoch FROM runtime_epoch WHERE singleton=1').fetchone()[0]
    return holder._execute_write(write)


def current_epoch(holder):
    """The stored epoch, None before the first ``begin_runtime_epoch``."""
    with holder._read_ctx() as conn:
        row = conn.execute('SELECT epoch FROM runtime_epoch WHERE singleton=1').fetchone()
    return row[0] if row else None


# Store *record* as *session_id*'s pending viewer close, only while this authority owns the store.
def _persist_close(holder, *, epoch, session_id, record):
    from ..conversation._store.retirement import VIEWER_CLOSE_PREFIX as CLOSE_PREFIX

    def write(conn):
        _epoch(conn, epoch)
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (CLOSE_PREFIX + session_id, record))
    holder._execute_write(write)


def forget_persisted_close(holder, *, epoch, session_id, record):
    """Blocking: drop *session_id*'s stored close only if it is still exactly *record* (that close
    completed or was cancelled), and only while this authority owns the store. Returns whether a
    row went."""
    from ..conversation._store.retirement import VIEWER_CLOSE_PREFIX as CLOSE_PREFIX

    def write(conn):
        _epoch(conn, epoch)
        return conn.execute('DELETE FROM state_meta WHERE key=? AND value=?',
                            (CLOSE_PREFIX + session_id, record)).rowcount
    return bool(holder._execute_write(write))


def stored_close(holder, session_id):
    """The stored close record, read as ``get_meta`` reads (on the writer connection under its lock)."""
    from ..conversation._store.retirement import VIEWER_CLOSE_PREFIX as CLOSE_PREFIX
    return holder.get_meta(CLOSE_PREFIX + session_id)


def stored_closes(holder):
    """``(session_id, record)`` of every stored close, in store order."""
    from ..conversation._store.retirement import VIEWER_CLOSE_PREFIX as CLOSE_PREFIX
    with holder._read_ctx() as conn:
        stored = conn.execute('SELECT key, value FROM state_meta WHERE key LIKE ?', (CLOSE_PREFIX + '%',)).fetchall()
    return [(key[len(CLOSE_PREFIX):], record) for key, record in stored]


# The status of the first unsettled admission or worker on one session id.
_CANONICAL_PENDING_SQL = """SELECT status FROM session_admissions WHERE target_session_id=? AND status!='terminal'
    UNION ALL SELECT status FROM worker_executions WHERE session_id=? AND status!='terminal' LIMIT 1"""


def pending_ledger_status(holder, key):
    """The status of the gateway ledger's first non-terminal admission or worker on *key*, else None."""
    with holder._read_ctx() as conn:
        row = next(iter(conn.execute(_CANONICAL_PENDING_SQL, (key, key))), None)
    return None if row is None else row[0]


def _end_unless_ledger_busy(holder, session_id: str, session_key, end_reason: str) -> bool:
    """End *session_id*'s row unless the gateway's durable ledger still holds work for it; True when
    the row may end (ended now, or already ended). The check and the stamp share one write txn.

    A viewer process (a dashboard, a second Desktop window) runs this reaper for a session the
    owning gateway is still executing; stamping it ended mid-turn made the owner treat its own live
    route as stale (#54878) and replace the conversation (#140642). The ledger names a local session
    by its logical id (the row's ``chat_id``), so the physical id, the key and that id are checked: a
    ``local-`` id, or the lineage owner its receipt names (an adopted conversation keeps its legacy id)."""
    from ..conversation._store.local import local_lineage_owner
    from ..conversation._store.sessions import _end_and_bump

    def write(conn):
        row = conn.execute("SELECT chat_id FROM sessions WHERE id=?", (session_id,)).fetchone()
        ids = {session_id, str(session_key or "")}
        if row is not None and str(row[0] or "").startswith("local-"):
            ids.add(str(row[0]))
        ids.add(local_lineage_owner(conn, session_id))
        for sid in filter(None, ids):
            if conn.execute(_CANONICAL_PENDING_SQL, (sid, sid)).fetchone() is not None:
                return False
        _end_and_bump(holder, conn, "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL",
                      (time.time(), end_reason, session_id), session_id, end_reason)
        return True

    return holder._execute_write(write)


def finalize_webhook(holder, *, epoch, session_id, admission_id, generation, admission_epoch, reason):
    """End only the settled admitted generation, atomically with the epoch fence."""
    from ..conversation._store.sessions import _end_and_bump
    sid = session_id
    def write(conn):
        return _end_and_bump(holder, conn, """
            UPDATE sessions SET ended_at=?, end_reason=?
            WHERE id=? AND ended_at IS NULL AND runtime_generation=?
              AND EXISTS (SELECT 1 FROM runtime_epoch WHERE singleton=1 AND epoch=?)
              AND EXISTS (SELECT 1 FROM session_admissions WHERE admission_id=?
                AND target_session_id=? AND status='terminal' AND generation=? AND owner_epoch=?)
              AND NOT EXISTS (SELECT 1 FROM session_admissions WHERE target_session_id=? AND status!='terminal')
            """, (time.time(), reason, sid, generation,
                  epoch, admission_id, sid, generation,
                  admission_epoch, sid), sid, reason)
    return bool(holder._execute_write(write))
