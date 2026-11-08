"""Finite exact-conversation admission fencing; never an execution reservation."""
import contextlib
from contextvars import ContextVar
import json
from pathlib import Path
import time

from hermes_cli.conversation_closeout import CloseoutStore, _PREFIX
from hermes_state_runtime import RuntimeStoreError

_REVIEW = ContextVar('hermes_closeout_review_admission', default=None)


def _record(db, conn, session_id):
    from hermes_state_local import local_lineage_owner
    owner = local_lineage_owner(conn, session_id)
    if owner == session_id:
        lineage = db.get_compression_lineage(session_id)
        owner = lineage[0] if lineage else session_id
    root = conn.execute('SELECT value FROM state_meta WHERE key=?', (_PREFIX + 'root:' + owner,)).fetchone()
    if root is None:
        return None
    binding = json.loads(root[0])
    row = conn.execute('SELECT value FROM state_meta WHERE key=?',
                       (CloseoutStore.operation_key(binding['closeout_id']),)).fetchone()
    return json.loads(row[0]) if row else None


@contextlib.contextmanager
def review_admission(db, operation):
    token = _REVIEW.set((str(Path(db.db_path).resolve()), operation))
    try:
        yield
    finally:
        _REVIEW.reset(token)


def require_admission(db, conn, session_id, *, now=None, request_id=None):
    try:
        record = _record(db, conn, session_id)
        if not record or record.get('_fence_released'):
            return
        if (_REVIEW.get() == (str(Path(db.db_path).resolve()), record['closeout_id'])
                and request_id in (None, 'closeout-review-' + record['closeout_id'])):
            return
        row = conn.execute('SELECT archived FROM sessions WHERE id=?', (session_id,)).fetchone()
        archived = bool(record['archived'] and row and row[0])
        if archived or record.get('_fence_until', 0) > (time.time() if now is None else now):
            raise RuntimeStoreError('conversation_closing', detail='This conversation is closing. Start a fresh chat or explicitly unarchive it.')
    except RuntimeStoreError:
        raise
    except (ValueError, TypeError, KeyError) as exc:
        raise RuntimeStoreError('storage_unavailable') from exc


def legacy_admission_refusal(session):
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    home = Path(session.get('profile_home') or get_hermes_home())
    path = home / 'state.db'
    if not path.is_file():
        return None
    try:
        with SessionDB(db_path=path) as db, db._read_ctx() as conn:
            require_admission(db, conn, str(session.get('session_key') or ''))
    except RuntimeStoreError as exc:
        return str(exc)
    return None


def release_fence(db, session_id, *, conn=None):
    """The existing explicit unarchive action releases a closeout, including its pending job."""
    def write(conn):
        record = _record(db, conn, session_id)
        if record is None:
            return
        record['_fence_released'] = True
        record['_fence_until'] = 0
        record['status'] = 'preserved'
        record['archived'] = False
        record['error'] = {'code': 'closeout_reopened', 'message': 'Closeout was released by explicit unarchive.'}
        conn.execute('UPDATE state_meta SET value=? WHERE key=?',
            (json.dumps(record, sort_keys=True, separators=(',', ':')),
             CloseoutStore.operation_key(record['closeout_id'])))
    if conn is not None:
        write(conn)
    else:
        db._execute_write(write)
