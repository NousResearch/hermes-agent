"""Canonical metadata-only logical attempts and explicit bounded preparation.

The projection outlives payload retirement, just like the canonical duplicate
receipt. It is not an HTTP receipt and cannot recover policy, grants or content.
Unattributable old terminal JSON cannot be recovered from one-way identity hashes.
"""
import hashlib
import hmac
import json
import sqlite3

from hermes_state_errors import StateDbReplacedError
from hermes_state_runtime import RuntimeStoreError, _text, admission_fingerprint

VERSION = 1
MAX_BATCH = 256

# No readiness marker is installed by DDL. Only explicit preparation can certify
# inventory coverage. Immutable projections cannot be pruned with HTTP receipts.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS logical_attempts (
    admission_id TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    intent TEXT NOT NULL,
    owner_scope TEXT,
    task_id TEXT,
    execution_generation INTEGER,
    CHECK ((task_id IS NULL AND execution_generation IS NULL) OR
           (owner_scope IS NOT NULL AND task_id IS NOT NULL AND execution_generation > 0))
);
-- Private generation anchor: recreated tables cannot inherit old coverage, even
-- when SQLite reuses a root page. Empty identities are forbidden to admissions.
INSERT OR IGNORE INTO logical_attempts(admission_id,principal_id,session_id,request_id,payload_digest,intent)
VALUES('','','','',lower(hex(randomblob(32))),'schema');
CREATE INDEX IF NOT EXISTS logical_attempt_exact
    ON logical_attempts(principal_id,session_id,owner_scope,task_id,execution_generation);
CREATE TABLE IF NOT EXISTS logical_attempt_coverage (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    version INTEGER NOT NULL,
    schema_cookie TEXT NOT NULL,
    phase TEXT NOT NULL,
    live_cursor INTEGER NOT NULL DEFAULT 0,
    terminal_cursor TEXT NOT NULL DEFAULT '',
    complete INTEGER NOT NULL DEFAULT 0 CHECK(complete IN (0,1))
);
CREATE TABLE IF NOT EXISTS logical_attempt_dirty (
    source TEXT NOT NULL,
    admission_id TEXT NOT NULL,
    session_id TEXT,
    PRIMARY KEY(source,admission_id)
);
CREATE INDEX IF NOT EXISTS logical_attempt_dirty_scope ON logical_attempt_dirty(session_id);
CREATE TRIGGER IF NOT EXISTS logical_attempt_no_replace BEFORE INSERT ON logical_attempts
WHEN NEW.admission_id!='' AND EXISTS(SELECT 1 FROM logical_attempts WHERE admission_id=NEW.admission_id)
BEGIN SELECT RAISE(ABORT,'logical attempt evidence is immutable'); END;
CREATE TRIGGER IF NOT EXISTS logical_attempt_no_update BEFORE UPDATE ON logical_attempts
BEGIN SELECT RAISE(ABORT,'logical attempt evidence is immutable'); END;
CREATE TRIGGER IF NOT EXISTS logical_attempt_no_delete BEFORE DELETE ON logical_attempts
BEGIN SELECT RAISE(ABORT,'logical attempt evidence has canonical lifetime'); END;
CREATE TRIGGER IF NOT EXISTS logical_attempt_live_insert AFTER INSERT ON session_admissions
BEGIN INSERT OR REPLACE INTO logical_attempt_dirty VALUES('live',NEW.admission_id,NEW.target_session_id); END;
CREATE TRIGGER IF NOT EXISTS logical_attempt_live_update
AFTER UPDATE OF admission_id,principal_id,target_session_id,request_id,payload_json,payload_digest,intent ON session_admissions
BEGIN INSERT OR REPLACE INTO logical_attempt_dirty VALUES('live',NEW.admission_id,NEW.target_session_id); END;
CREATE TRIGGER IF NOT EXISTS logical_attempt_live_delete AFTER DELETE ON session_admissions
BEGIN INSERT OR REPLACE INTO logical_attempt_dirty VALUES('live',OLD.admission_id,OLD.target_session_id); END;
CREATE TRIGGER IF NOT EXISTS logical_attempt_terminal_insert AFTER INSERT ON state_meta
WHEN NEW.key GLOB 'gateway.terminal_admission.v1.*'
BEGIN INSERT OR REPLACE INTO logical_attempt_dirty
SELECT 'terminal',substr(NEW.key,31),session_id FROM logical_attempts WHERE admission_id=substr(NEW.key,31);
INSERT OR IGNORE INTO logical_attempt_dirty VALUES('terminal',substr(NEW.key,31),NULL); END;
CREATE TRIGGER IF NOT EXISTS logical_attempt_terminal_update AFTER UPDATE OF value ON state_meta
WHEN NEW.key GLOB 'gateway.terminal_admission.v1.*'
BEGIN INSERT OR REPLACE INTO logical_attempt_dirty
SELECT 'terminal',substr(NEW.key,31),session_id FROM logical_attempts WHERE admission_id=substr(NEW.key,31);
INSERT OR IGNORE INTO logical_attempt_dirty VALUES('terminal',substr(NEW.key,31),NULL); END;
CREATE TRIGGER IF NOT EXISTS logical_attempt_terminal_delete AFTER DELETE ON state_meta
WHEN OLD.key GLOB 'gateway.terminal_admission.v1.*'
BEGIN INSERT OR REPLACE INTO logical_attempt_dirty
SELECT 'terminal',substr(OLD.key,31),session_id FROM logical_attempts WHERE admission_id=substr(OLD.key,31);
INSERT OR IGNORE INTO logical_attempt_dirty VALUES('terminal',substr(OLD.key,31),NULL); END;
"""

_FIELDS = ('admission_id', 'principal_id', 'session_id', 'request_id', 'payload_digest',
           'intent', 'owner_scope', 'task_id', 'execution_generation')


def _expected_ddl():
    # Parse our constant DDL, not database/user text. complete_statement handles
    # trigger bodies without executing SQL or opening a second database.
    statements = {}
    pending = ''
    for line in SCHEMA_SQL.splitlines():
        if line.lstrip().startswith('--'):
            continue
        pending += line + '\n'
        if sqlite3.complete_statement(pending):
            normalized = ' '.join(pending.strip().removesuffix(';').replace('IF NOT EXISTS ', '').split())
            if normalized.startswith('CREATE '):
                statements[normalized.split()[2]] = normalized
            pending = ''
    return statements


_EXPECTED_DDL = _expected_ddl()


def _schema_identity(conn):
    # Other owners legitimately install DDL after bootstrap. Validate only our
    # fixed schema objects, plus the non-reusable table generation anchor.
    names = tuple(_EXPECTED_DDL)
    rows = conn.execute(f'SELECT name,sql FROM sqlite_master WHERE name IN ({",".join("?" for _ in names)}) ORDER BY name', names).fetchall()
    if len(rows) != len(names) or any(' '.join(row['sql'].split()) != _EXPECTED_DDL[row['name']] for row in rows):
        raise RuntimeStoreError('storage_unavailable')
    anchor = conn.execute("SELECT payload_digest FROM logical_attempts WHERE admission_id='' AND principal_id='' AND session_id='' AND request_id='' AND intent='schema'").fetchone()
    if anchor is None or len(anchor[0]) != 64:
        raise RuntimeStoreError('storage_unavailable')
    return hashlib.sha256((repr([tuple(row) for row in rows]) + anchor[0]).encode()).hexdigest()


def _identity(row):
    values = dict(admission_id=row['admission_id'], principal_id=row['principal_id'],
        session_id=row['target_session_id'], request_id=row['request_id'],
        payload_digest=row['payload_digest'], intent=row['intent'])
    for value in values.values():
        _text(value)
    digest = values['payload_digest']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise RuntimeStoreError('storage_unavailable')
    if values['intent'] not in ('queue', 'steer', 'redirect'):
        raise RuntimeStoreError('storage_unavailable')
    return values


def _projection(row, *, erased=False):
    values = _identity(row)
    scope = task = generation = None
    if not erased:
        payload = json.loads(row['payload_json'])
        if not isinstance(payload, dict):
            raise RuntimeStoreError('storage_unavailable')
        digest = admission_fingerprint(canonical_target=values['session_id'],
            payload={'input': payload, 'intent': values['intent']})
        if not hmac.compare_digest(digest, values['payload_digest']):
            raise RuntimeStoreError('storage_unavailable')
        turn = payload.get('api_turn_v1')
        if isinstance(turn, dict):
            scope = turn.get('run_owner_scope')
            if scope is not None:
                _text(scope)
            settings = turn.get('settings')
            dispatch = settings.get('room_dispatch') if isinstance(settings, dict) else None
            if dispatch is not None:
                if not isinstance(dispatch, dict):
                    raise RuntimeStoreError('storage_unavailable')
                task, generation = dispatch.get('task_id'), dispatch.get('execution_generation')
                _text(task)
                if scope is None or type(generation) is not int or generation < 1:
                    raise RuntimeStoreError('storage_unavailable')
    return values | dict(owner_scope=scope, task_id=task, execution_generation=generation)


def project_admission(conn, row, *, migrating=False, erased=False):
    """Called only in the canonical admission/retirement/preparation transaction."""
    values = _projection(row, erased=erased)
    old = conn.execute('SELECT * FROM logical_attempts WHERE admission_id=?',
                       (values['admission_id'],)).fetchone()
    if old is not None:
        # Erasure must never replace exact pre-erasure identity with ambiguity.
        fields = _FIELDS[:6] if erased else _FIELDS
        if any(old[k] != values[k] for k in fields):
            raise RuntimeStoreError('storage_unavailable')
    else:
        if not migrating and values['task_id'] is not None:
            match = conn.execute('SELECT 1 FROM logical_attempts WHERE principal_id=? AND session_id=? '
                'AND owner_scope=? AND task_id=? AND execution_generation=? LIMIT 1',
                tuple(values[k] for k in ('principal_id', 'session_id', 'owner_scope', 'task_id', 'execution_generation'))).fetchone()
            if match:
                raise RuntimeStoreError('admission_conflict')
        conn.execute(f'INSERT INTO logical_attempts({",".join(_FIELDS)}) VALUES({",".join("?" for _ in _FIELDS)})',
                     tuple(values[k] for k in _FIELDS))
    conn.execute("DELETE FROM logical_attempt_dirty WHERE source='live' AND admission_id=?", (values['admission_id'],))


def finish_retirement(conn, admission_ids):
    for aid in admission_ids:
        conn.execute('DELETE FROM logical_attempt_dirty WHERE admission_id=?', (aid,))


def _terminal(conn, key, encoded):
    from hermes_state_terminal import ADMISSION_PREFIX, identity_key
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    aid = key[len(ADMISSION_PREFIX):]
    try:
        row = json.loads(encoded)
        if not isinstance(row, dict):
            raise ValueError('invalid terminal identity')
        # Establish independent physical attribution BEFORE checking corruptible
        # nonidentity fields. A bad status/digest/intent is not global ambiguity
        # when the retained identity hash and retirement marker prove this scope.
        principal, sid, request = (_text(row[k]) for k in
                                   ('principal_id', 'target_session_id', 'request_id'))
        identity = conn.execute('SELECT value FROM state_meta WHERE key=?',
            (identity_key(principal, sid, request),)).fetchone()
        retired = conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + sid,)).fetchone()
        if identity is None or json.loads(identity[0]) != aid or retired is None:
            raise ValueError('terminal identity is not witnessed')
        # Scope now has independent canonical identity evidence, even if payload
        # integrity fails. Never turn that failure into unrelated-profile poison.
        conn.execute('INSERT OR REPLACE INTO logical_attempt_dirty VALUES(?,?,?)',
                     ('terminal', aid, sid))
        if row.get('admission_id') != aid or row.get('status') != 'terminal':
            raise ValueError('invalid terminal admission')
        project_admission(conn, row, migrating=True, erased=row['payload_json'] == '{}')
        conn.execute("DELETE FROM logical_attempt_dirty WHERE source='terminal' AND admission_id=?", (aid,))
        return None
    except (ValueError, TypeError, KeyError):
        known = conn.execute('SELECT session_id FROM logical_attempts WHERE admission_id=?', (aid,)).fetchone()
        conn.execute('INSERT OR IGNORE INTO logical_attempt_dirty VALUES(?,?,?)',
                     ('terminal', aid, known[0] if known else None))
        pending = conn.execute("SELECT session_id FROM logical_attempt_dirty WHERE source='terminal' AND admission_id=?", (aid,)).fetchone()
        return 'scoped_terminal_corruption' if pending[0] is not None else 'unclassifiable_terminal'


def prepare_logical_attempt_index(db, *, batch_size=128):
    """One bounded batch, outside requests; cursor survives restart.

    Bootstrap calls once. Operators may call again on an owned database until
    complete. No background thread, implicit request backfill, or live migration
    is scheduled here. Errors name unavailable inventory, never invent identity.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH:
        raise RuntimeStoreError('invalid_params')
    from hermes_state_terminal import ADMISSION_PREFIX
    def write(conn):
        cookie = _schema_identity(conn)
        saved = conn.execute('SELECT * FROM logical_attempt_coverage WHERE singleton=1').fetchone()
        if saved is None or saved['version'] != VERSION or saved['schema_cookie'] != cookie:
            conn.execute("INSERT OR REPLACE INTO logical_attempt_coverage VALUES(1,?,?,'live',0,'',0)", (VERSION, cookie))
        state = dict(conn.execute('SELECT * FROM logical_attempt_coverage WHERE singleton=1').fetchone())
        processed = 0
        error = error_key = None
        if state['phase'] == 'live':
            rows = conn.execute('SELECT * FROM session_admissions WHERE seq>? ORDER BY seq LIMIT ?',
                                (state['live_cursor'], batch_size)).fetchall()
            for row in rows:
                try:
                    project_admission(conn, row, migrating=True)
                except (ValueError, KeyError, TypeError):
                    conn.execute('INSERT OR REPLACE INTO logical_attempt_dirty VALUES(?,?,?)',
                                 ('live', row['admission_id'], row['target_session_id']))
                state['live_cursor'] = row['seq']
            processed += len(rows)
            if len(rows) < batch_size:
                state['phase'] = 'terminal'
        if state['phase'] == 'terminal' and processed < batch_size:
            # Binary prefix range uses state_meta's primary-key B-tree, not LIKE.
            rows = conn.execute('SELECT key,value FROM state_meta WHERE key>? AND key>=? AND key<? ORDER BY key LIMIT ?',
                (state['terminal_cursor'], ADMISSION_PREFIX, ADMISSION_PREFIX[:-1] + '/', batch_size - processed)).fetchall()
            remaining = batch_size - processed
            for key, encoded in rows:
                failure = _terminal(conn, key, encoded)
                if failure:
                    error, error_key = failure, key
                state['terminal_cursor'] = key
            processed += len(rows)
            if len(rows) < remaining:
                state['phase'] = 'covered'
        if state['phase'] == 'covered' and processed < batch_size:
            rows = conn.execute('SELECT * FROM logical_attempt_dirty ORDER BY source,admission_id LIMIT ?',
                                (batch_size - processed,)).fetchall()
            for pending in rows:
                aid = pending['admission_id']
                row = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (aid,)).fetchone()
                if row is not None:
                    try:
                        project_admission(conn, row, migrating=True)
                    except (ValueError, TypeError, KeyError):
                        error, error_key = 'scoped_live_corruption', aid
                else:
                    key = ADMISSION_PREFIX + aid
                    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
                    if saved is not None:
                        failure = _terminal(conn, key, saved[0])
                        if failure:
                            error, error_key = failure, key
                    else:
                        error, error_key = 'missing_canonical_evidence', aid
                processed += 1
                if error:
                    break
        unknown = conn.execute('SELECT 1 FROM logical_attempt_dirty WHERE session_id IS NULL LIMIT 1').fetchone()
        complete = state['phase'] == 'covered' and unknown is None
        conn.execute('UPDATE logical_attempt_coverage SET phase=?,live_cursor=?,terminal_cursor=?,complete=? WHERE singleton=1',
                     (state['phase'], state['live_cursor'], state['terminal_cursor'], int(complete)))
        pending = conn.execute('SELECT 1 FROM logical_attempt_dirty LIMIT 1').fetchone() is not None
        return dict(processed=processed, complete=complete and not pending, coverage_complete=complete,
                    pending=pending, error=error, error_key=error_key,
                    live_cursor=state['live_cursor'], terminal_cursor=state['terminal_cursor'])
    return db._execute_write(write)


def lookup_logical_attempt(db, *, principal_id, session_id, owner_scope, task_id, execution_generation):
    """Authenticated read: None is certified absence, never an empty-cache guess.

    At most two exact records are read: the second is a contradictory canonical
    duplicate, not a LIMIT used to infer absence. No JSON payload is decoded.
    """
    for value in (principal_id, session_id, owner_scope, task_id):
        _text(value)
    if type(execution_generation) is not int or execution_generation < 1:
        raise RuntimeStoreError('invalid_params')
    try:
        db._halt_if_db_generation_changed()
        with db._read_ctx() as conn:
            conn.execute('BEGIN')
            try:
                cookie = _schema_identity(conn)
                state = conn.execute('SELECT version,schema_cookie,complete FROM logical_attempt_coverage WHERE singleton=1').fetchone()
                if state is None or tuple(state) != (VERSION, cookie, 1):
                    raise RuntimeStoreError('storage_unavailable')
                for sid in (None, session_id):
                    if conn.execute('SELECT 1 FROM logical_attempt_dirty WHERE session_id IS ? LIMIT 1', (sid,)).fetchone():
                        raise RuntimeStoreError('storage_unavailable')
                for scope in (None, owner_scope):
                    if conn.execute('SELECT 1 FROM logical_attempts WHERE principal_id=? AND session_id=? '
                            'AND owner_scope IS ? AND task_id IS NULL LIMIT 1', (principal_id, session_id, scope)).fetchone():
                        raise RuntimeStoreError('storage_unavailable')
                rows = conn.execute('SELECT * FROM logical_attempts WHERE principal_id=? AND session_id=? '
                    'AND owner_scope=? AND task_id=? AND execution_generation=? LIMIT 2',
                    (principal_id, session_id, owner_scope, task_id, execution_generation)).fetchall()
                if len(rows) > 1:
                    raise RuntimeStoreError('storage_unavailable')
                return dict(rows[0]) if rows else None
            finally:
                conn.execute('ROLLBACK')
    except (sqlite3.Error, StateDbReplacedError) as exc:
        raise RuntimeStoreError('storage_unavailable') from exc
