"""Closed input-custody SQL operations on the canonical admission connection."""
from dataclasses import dataclass
import hashlib
import hmac
import secrets
import time
import uuid

from hermes_state_runtime import RuntimeStoreError, _epoch
from hermes_state_terminal import terminal_admission

READY_KEY = 'gateway.input-reclamation.v3'
IDENTITY_FIELDS = ('principal_id', 'target_session_id', 'request_id', 'payload_digest', 'intent')


@dataclass(frozen=True)
class PreparedInputHandle:
    preparation_id: str
    token: str
    generation: int


@dataclass(frozen=True)
class AcceptedInputHandle:
    admission_id: str
    payload: dict


def create_schema(conn):
    # No FK to session_admissions: exact references outlive its retirement.
    statements = (
        '''CREATE TABLE IF NOT EXISTS input_custody_copies(
            copy_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, name TEXT NOT NULL,
            digest TEXT NOT NULL, size INTEGER NOT NULL, generation INTEGER NOT NULL,
            state TEXT NOT NULL, device TEXT, inode TEXT,
            UNIQUE(namespace,digest,name))''',
        '''CREATE TABLE IF NOT EXISTS input_custody_preparations(
            preparation_id TEXT PRIMARY KEY, token_digest TEXT NOT NULL, generation INTEGER NOT NULL,
            owner_epoch INTEGER NOT NULL, principal_id TEXT NOT NULL, target_session_id TEXT NOT NULL,
            request_id TEXT NOT NULL, intent TEXT NOT NULL, expires_at REAL NOT NULL,
            state TEXT NOT NULL, payload_digest TEXT, admission_id TEXT)''',
        '''CREATE TABLE IF NOT EXISTS input_custody_items(
            preparation_id TEXT NOT NULL, ordinal INTEGER NOT NULL, copy_id TEXT NOT NULL,
            generation INTEGER NOT NULL, PRIMARY KEY(preparation_id,ordinal))''',
        '''CREATE TABLE IF NOT EXISTS input_custody_refs(
            admission_id TEXT NOT NULL, ordinal INTEGER NOT NULL, copy_id TEXT NOT NULL,
            generation INTEGER NOT NULL, principal_id TEXT NOT NULL, target_session_id TEXT NOT NULL,
            request_id TEXT NOT NULL, payload_digest TEXT NOT NULL, intent TEXT NOT NULL,
            PRIMARY KEY(admission_id,ordinal))''',
        '''CREATE TABLE IF NOT EXISTS input_custody_legacy_ids(
            admission_id TEXT PRIMARY KEY, principal_id TEXT NOT NULL, target_session_id TEXT NOT NULL,
            request_id TEXT NOT NULL, payload_digest TEXT NOT NULL, intent TEXT NOT NULL)''',
        '''CREATE TABLE IF NOT EXISTS input_custody_branch_refs(
            branch_id TEXT NOT NULL, copy_id TEXT NOT NULL, generation INTEGER NOT NULL,
            PRIMARY KEY(branch_id,copy_id,generation))''',
        'CREATE TABLE IF NOT EXISTS input_custody_legacy_branches(branch_id TEXT PRIMARY KEY)',
        'CREATE INDEX IF NOT EXISTS input_custody_items_copy ON input_custody_items(copy_id,generation)',
        'CREATE INDEX IF NOT EXISTS input_custody_refs_copy ON input_custody_refs(copy_id,generation)',
    )
    for statement in statements:
        conn.execute(statement)


def token_digest(token):
    if not isinstance(token, str):
        raise RuntimeStoreError('invalid_params')
    return hashlib.sha256(token.encode()).hexdigest()


def begin_preparation(conn, *, epoch, principal_id, session_id, request_id, ttl=300):
    _epoch(conn, epoch)
    if type(ttl) not in (int, float) or not 0 < ttl <= 3600:
        raise RuntimeStoreError('invalid_params')
    handle = PreparedInputHandle(uuid.uuid4().hex, secrets.token_hex(32), 1)
    conn.execute('''INSERT INTO input_custody_preparations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
        (handle.preparation_id, token_digest(handle.token), handle.generation, epoch,
         principal_id, session_id, request_id, 'queue', time.time() + ttl, 'preparing', None, None))
    return handle


def preparation(conn, *, epoch, handle, states=('preparing', 'ready')):
    _epoch(conn, epoch)
    if not isinstance(handle, PreparedInputHandle):
        raise RuntimeStoreError('invalid_params')
    row = conn.execute('SELECT * FROM input_custody_preparations WHERE preparation_id=?',
                       (handle.preparation_id,)).fetchone()
    if (row is None or not hmac.compare_digest(row['token_digest'], token_digest(handle.token))
            or row['generation'] != handle.generation or row['owner_epoch'] != epoch
            or row['state'] not in states or row['expires_at'] <= time.time()):
        raise RuntimeStoreError('input_preparation_expired')
    return row


def add_copy(conn, *, handle, ordinal, name, digest, size):
    row = conn.execute('''SELECT * FROM input_custody_copies
        WHERE namespace='v3' AND digest=? AND name=?''', (digest, name)).fetchone()
    if row is None:
        copy_id = uuid.uuid4().hex
        conn.execute('INSERT INTO input_custody_copies VALUES(?,?,?,?,?,?,?,?,?)',
            (copy_id, 'v3', name, digest, size, 1, 'preparing', None, None))
    else:
        copy_id = row['copy_id']
        if row['size'] != size or row['state'] == 'sealed':
            raise RuntimeStoreError('input_preparation_busy')
        if row['state'] == 'removed':
            conn.execute("""UPDATE input_custody_copies SET generation=generation+1,
                state='preparing',device=NULL,inode=NULL WHERE copy_id=?""", (copy_id,))
    row = conn.execute('SELECT * FROM input_custody_copies WHERE copy_id=?', (copy_id,)).fetchone()
    conn.execute('INSERT INTO input_custody_items VALUES(?,?,?,?)',
                 (handle.preparation_id, ordinal, copy_id, row['generation']))
    return dict(row)


def preparation_copies(conn, handle):
    return [dict(row) for row in conn.execute('''SELECT c.*,i.ordinal,i.generation AS item_generation
        FROM input_custody_items i JOIN input_custody_copies c USING(copy_id)
        WHERE i.preparation_id=? ORDER BY i.ordinal''', (handle.preparation_id,))]


def finish_preparation(conn, *, epoch, handle, payload_digest):
    preparation(conn, epoch=epoch, handle=handle, states=('preparing',))
    copies = preparation_copies(conn, handle)
    if not copies or any(c['state'] != 'ready' or c['generation'] != c['item_generation'] for c in copies):
        raise RuntimeStoreError('storage_unavailable')
    conn.execute("UPDATE input_custody_preparations SET state='ready',payload_digest=? WHERE preparation_id=?",
                 (payload_digest, handle.preparation_id))


def _same_identity(saved, admission):
    return all(saved[field] == admission[field] for field in IDENTITY_FIELDS)


def admission_input_refs(conn, admission):
    refs = conn.execute('SELECT * FROM input_custody_refs WHERE admission_id=? ORDER BY ordinal',
                        (admission['admission_id'],)).fetchall()
    if not refs:
        return None
    if any(not _same_identity(row, admission) for row in refs):
        raise RuntimeStoreError('storage_unavailable')
    result = []
    for ref in refs:
        copy = conn.execute('SELECT * FROM input_custody_copies WHERE copy_id=?', (ref['copy_id'],)).fetchone()
        if copy is None or copy['generation'] != ref['generation'] or copy['state'] != 'ready':
            raise RuntimeStoreError('storage_unavailable')
        result.append(dict(copy))
    return result


def retry_payload(conn, *, handle, principal_id, session_id, request_id):
    from gateway.session_admission import admission_fingerprint
    if not isinstance(handle, AcceptedInputHandle):
        raise RuntimeStoreError('invalid_params')
    row = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (handle.admission_id,)).fetchone()
    row = row if row is not None else terminal_admission(conn, handle.admission_id)
    digest = admission_fingerprint(canonical_target=session_id, payload={'input': handle.payload, 'intent': 'queue'})
    if row is None or tuple(row[key] for key in ('principal_id', 'target_session_id', 'request_id', 'payload_digest')) != (
            principal_id, session_id, request_id, digest):
        raise RuntimeStoreError('admission_conflict')
    return handle.payload


def accept_prepared_input(conn, *, epoch, admission, handle):
    """Called before the admission writer commits, never after its receipt."""
    _epoch(conn, epoch)
    existing = admission_input_refs(conn, admission)
    if existing is not None:
        return  # Exact accepted retries never require a fresh lease.
    row = preparation(conn, epoch=epoch, handle=handle, states=('ready',))
    if not _same_identity(row, admission):
        raise RuntimeStoreError('admission_conflict')
    copies = preparation_copies(conn, handle)
    if not copies or any(c['state'] != 'ready' or c['generation'] != c['item_generation'] for c in copies):
        raise RuntimeStoreError('input_preparation_expired')
    for copy in copies:
        conn.execute('INSERT INTO input_custody_refs VALUES(?,?,?,?,?,?,?,?,?)',
            (admission['admission_id'], copy['ordinal'], copy['copy_id'], copy['generation'],
             *(admission[field] for field in IDENTITY_FIELDS)))
    conn.execute("UPDATE input_custody_preparations SET state='consumed',admission_id=? WHERE preparation_id=?",
                 (admission['admission_id'], handle.preparation_id))


def positively_retired(conn, saved):
    if conn.execute('SELECT 1 FROM session_admissions WHERE admission_id=?', (saved['admission_id'],)).fetchone():
        return False
    try:
        row = terminal_admission(conn, saved['admission_id'])
        return (isinstance(row, dict) and row.get('admission_id') == saved['admission_id']
                and row.get('status') == 'terminal' and _same_identity(row, saved))
    except (ValueError, TypeError, KeyError):
        return False


def copy_is_held(conn, copy, now):
    refs = conn.execute('SELECT * FROM input_custody_refs WHERE copy_id=? AND generation=?',
                        (copy['copy_id'], copy['generation'])).fetchall()
    if any(not positively_retired(conn, row) for row in refs):
        return True
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    branches = conn.execute('SELECT branch_id FROM input_custody_branch_refs WHERE copy_id=? AND generation=?',
                            (copy['copy_id'], copy['generation'])).fetchall()
    if copy['namespace'] != 'v3':
        branches += conn.execute('SELECT branch_id FROM input_custody_legacy_branches').fetchall()
    if any(not conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + row[0],)).fetchone()
           or conn.execute('SELECT 1 FROM sessions WHERE id=?', (row[0],)).fetchone() for row in branches):
        return True
    if conn.execute('''SELECT 1 FROM input_custody_items i JOIN input_custody_preparations p USING(preparation_id)
            WHERE i.copy_id=? AND i.generation=? AND p.state IN ('preparing','ready')
            AND p.expires_at>? LIMIT 1''', (copy['copy_id'], copy['generation'], now)).fetchone():
        return True
    if copy['namespace'] != 'v3':
        return any(not positively_retired(conn, row) for row in conn.execute('SELECT * FROM input_custody_legacy_ids'))
    return False


def copy_branch_input_refs(conn, source_session, child_session, *, physical_session=None):
    """Branch copies content ownership in its own transaction, without parsing text."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='input_custody_refs'").fetchone():
        return
    for source in {source_session, physical_session or source_session}:
        conn.execute('''INSERT OR IGNORE INTO input_custody_branch_refs
            SELECT ?,copy_id,generation FROM input_custody_refs WHERE target_session_id=?''',
            (child_session, source))
        conn.execute('''INSERT OR IGNORE INTO input_custody_branch_refs
            SELECT ?,copy_id,generation FROM input_custody_branch_refs WHERE branch_id=?''',
            (child_session, source))
        if (conn.execute('SELECT 1 FROM input_custody_legacy_ids WHERE target_session_id=?', (source,)).fetchone()
                or conn.execute('SELECT 1 FROM input_custody_legacy_branches WHERE branch_id=?', (source,)).fetchone()):
            conn.execute('INSERT OR IGNORE INTO input_custody_legacy_branches VALUES(?)', (child_session,))


def seal_copy(conn, copy, now):
    if copy_is_held(conn, copy, now):
        return False
    conn.execute("""UPDATE input_custody_preparations SET state='expired'
        WHERE preparation_id IN (SELECT preparation_id FROM input_custody_items WHERE copy_id=? AND generation=?)
        AND state IN ('preparing','ready') AND expires_at<=?""", (copy['copy_id'], copy['generation'], now))
    return conn.execute("UPDATE input_custody_copies SET state='sealed' WHERE copy_id=? AND generation=? AND state IN ('preparing','ready')",
                        (copy['copy_id'], copy['generation'])).rowcount == 1
