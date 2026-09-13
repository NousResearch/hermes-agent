"""Prospective, authority-owned Kanban journal. No publication or recovery.

The Python entry points are the supported local enrollment/reader interface.
Schema and records belong to the board database, not a plugin or profile.
"""
import json
import sqlite3
import uuid


class AuthorityHistoryError(ValueError):
    """History cannot establish the requested coverage or identity."""


_TABLES = {
    'authority_history_meta': '''CREATE TABLE authority_history_meta (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1),
        version INTEGER NOT NULL CHECK(version=1),
        incarnation TEXT NOT NULL, coverage_start INTEGER NOT NULL CHECK(coverage_start=1)
    )''',
    'authority_task_bindings': '''CREATE TABLE authority_task_bindings (
        task_id TEXT PRIMARY KEY, version INTEGER NOT NULL CHECK(version=1),
        repo_id TEXT NOT NULL, project_id TEXT NOT NULL
    )''',
    'authority_owner_bindings': '''CREATE TABLE authority_owner_bindings (
        task_id TEXT NOT NULL, token_hash TEXT NOT NULL,
        version INTEGER NOT NULL CHECK(version=1), consumer_id TEXT NOT NULL,
        runtime_id TEXT NOT NULL, owner_ref TEXT NOT NULL,
        PRIMARY KEY(task_id, token_hash)
    )''',
    'authority_run_bindings': '''CREATE TABLE authority_run_bindings (
        run_id INTEGER PRIMARY KEY, task_id TEXT NOT NULL, token_hash TEXT NOT NULL
    )''',
    'authority_history_removal': '''CREATE TABLE authority_history_removal (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1)
    )''',
    'authority_source_bindings': '''CREATE TABLE authority_source_bindings (
        source_event_id INTEGER PRIMARY KEY CHECK(source_event_id>0), task_id TEXT NOT NULL,
        run_id INTEGER, kind TEXT NOT NULL, observed_at INTEGER NOT NULL
    )''',
    'authority_history': '''CREATE TABLE authority_history (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        version INTEGER NOT NULL CHECK(version=1),
        record TEXT NOT NULL
    )''',
}


# NOT part of _TABLES, deliberately. Every _TABLES member receives the immutability
# and owned-insert guards; applying those to the lease would make it unwritable and
# so unusable as a capability. It holds at most one row, only inside a protected
# transaction, and is excluded from the ownership snapshot and audit records.
_LEASE_TABLE = ('CREATE TABLE IF NOT EXISTS authority_writer_lease ('
                ' singleton INTEGER PRIMARY KEY CHECK(singleton=1))')


def _ensure_lease_table(conn):
    """Idempotent, and safe on an already-enrolled database being upgraded."""
    started = not conn.in_transaction
    conn.execute(_LEASE_TABLE)
    # A lease can only be left behind by a process killed mid-transaction, which
    # rollback already discards. Clearing on open is belt-and-braces for a crash
    # that somehow committed one: a stale lease would be fail-OPEN, the one
    # direction this mechanism must never fail in.
    conn.execute('DELETE FROM authority_writer_lease')
    # That DELETE opens an implicit transaction in autocommit mode. Leaving it
    # open made connect() return mid-transaction and the caller's next write_txn
    # fail with 'cannot start a transaction within a transaction'.
    if started and conn.in_transaction:
        conn.execute('COMMIT')


def _guards():
    guards = {f'{table}_no_{op.lower()}':
            f"CREATE TRIGGER {table}_no_{op.lower()} BEFORE {op} ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'authority history is immutable'); END"
            for table in _TABLES for op in ('UPDATE', 'DELETE')}
    for table, key in (('tasks', 'id'), ('task_runs', 'task_id')):
        for op in ('INSERT', 'UPDATE', 'DELETE'):
            refs = ('NEW',) if op == 'INSERT' else ('OLD',) if op == 'DELETE' else ('OLD', 'NEW')
            enrolled = ' OR '.join(
                f'EXISTS(SELECT 1 FROM authority_task_bindings WHERE task_id={ref}.{key})'
                for ref in refs)
            name = f'authority_writer_{table}_{op.lower()}'
            guards[name] = (f'CREATE TRIGGER {name} BEFORE {op} ON {table} '
                            f'WHEN {enrolled} BEGIN SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM authority_writer_lease) '
                            "THEN RAISE(ABORT, 'authority mutation requires owned writer') END; END")
    keys = {'authority_history_meta': ('singleton',), 'authority_task_bindings': ('task_id',),
            'authority_owner_bindings': ('task_id', 'token_hash'),
            'authority_run_bindings': ('run_id',), 'authority_history_removal': ('singleton',),
            'authority_source_bindings': ('source_event_id',),
            'authority_history': ('sequence',)}
    for table, columns in keys.items():
        name = f'{table}_no_replace'
        predicate = ' AND '.join(f'{key}=NEW.{key}' for key in columns)
        guards[name] = (f'CREATE TRIGGER {name} BEFORE INSERT ON {table} '
                        f'WHEN EXISTS(SELECT 1 FROM {table} WHERE {predicate}) '
                        "BEGIN SELECT RAISE(ABORT, 'authority history is immutable'); END")
    for table in _TABLES:
        name = f'{table}_owned_insert'
        guards[name] = (f'CREATE TRIGGER {name} BEFORE INSERT ON {table} '
                        'BEGIN SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM authority_writer_lease) '
                        "THEN RAISE(ABORT, 'authority insert requires owned writer') END; END")
    return guards


from contextlib import contextmanager


@contextmanager
def owned_writer(conn):
    """Transaction-scoped write capability, readable by the database itself.

    WHY NOT A PER-CONNECTION UDF (the G3 defect this replaces). The guards live in
    the schema, so every connection prepares them; a UDF lives in one connection,
    so every other connection failed at statement PREPARE with "no such function"
    -- even for a non-bound task, even for a statement matching zero rows, because
    SQLite resolves a trigger body's function before evaluating its WHEN clause.
    The opt-in predicate therefore protected nothing and the gate was
    database-wide. A row any connection can read restores the intended scope and
    returns the failure to the guard's own IntegrityError instead of
    OperationalError, which a caller could not previously distinguish.

    WHY THE ROW MUST BE WRITTEN INSIDE THE CALLER'S TRANSACTION. This context
    manager is entered by ``kanban_db.write_txn`` BEFORE ``BEGIN IMMEDIATE``. A
    lease inserted at that point would be written in autocommit, survive the
    transaction, and permanently disable the guard for every later connection --
    fail-closed silently converted to fail-open, with the owned path still green.
    So the lease is taken lazily, only once a transaction is actually open, and is
    released before that transaction ends. Rollback or an abrupt exit discards it
    with the transaction, which is the fail-closed direction.
    """
    if not isinstance(conn, sqlite3.Connection):
        # Boundary-only test doubles never contain SQLite authority state.
        yield
        return
    taken = _take_lease(conn)
    try:
        yield
    finally:
        if taken:
            _release_lease(conn)


@contextmanager
def migration_writer(conn):
    """Bounded capability for connect-time migration (N2).

    The optional-column pass contains a real backfill, so on an enrolled board it
    is a guarded write. It runs before any normal write transaction and sometimes
    in autocommit, where owned_writer alone would silently no-op and the board
    could not be opened at all. This opens a transaction when one is not already
    held, takes the lease, and always releases it, so the authority is explicit,
    bounded to the migration, and leaves no permanent lease behind.
    """
    if not isinstance(conn, sqlite3.Connection) or not _lease_available(conn):
        yield
        return
    own_txn = not conn.in_transaction
    if own_txn:
        conn.execute('BEGIN IMMEDIATE')
    try:
        conn.execute('INSERT OR IGNORE INTO authority_writer_lease (singleton) VALUES (1)')
        yield
        conn.execute('DELETE FROM authority_writer_lease')
        if own_txn:
            conn.execute('COMMIT')
    except Exception:
        if own_txn:
            try:
                conn.execute('ROLLBACK')
            except sqlite3.Error:
                pass
        else:
            try:
                conn.execute('DELETE FROM authority_writer_lease')
            except sqlite3.Error:
                pass
        raise


def _lease_available(conn):
    try:
        conn.execute('SELECT 1 FROM authority_writer_lease LIMIT 1')
    except sqlite3.Error:
        return False
    return True


def _take_lease(conn):
    """Claim the transaction-scoped capability. No-op outside a transaction.

    Reentrant by returning False when a lease is already held. write_txn can now
    nest via savepoints, and the inner block runs inside the OUTER transaction.
    If the inner call both took and released, the outer would lose its capability
    for every remaining statement and its own guarded writes would abort. Only the
    outermost taker releases; rollback still discards the row with the
    transaction, so the fail-closed direction is unchanged.
    """
    if not conn.in_transaction or not _lease_available(conn):
        return False
    if conn.execute("SELECT 1 FROM authority_writer_lease").fetchone():
        return False
    conn.execute("INSERT INTO authority_writer_lease (singleton) VALUES (1)")
    return True


def _release_lease(conn):
    if conn.in_transaction:
        try:
            conn.execute('DELETE FROM authority_writer_lease')
        except sqlite3.Error:
            pass


def migrate(conn):
    """Owned additive migration; do not repair a partially missing journal."""
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    present = set(_TABLES) & names
    if present:
        if present != set(_TABLES):
            raise AuthorityHistoryError('incomplete authority history schema')
        # Upgrade path: an board enrolled before the lease existed still needs it,
        # or every guarded write on it would fail closed for want of a capability.
        _ensure_lease_table(conn)
        _validate_guards(conn)
        capability(conn)
        return
    conn.execute('BEGIN IMMEDIATE')
    try:
        # Recheck after serializing against another first-open migration.
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='authority_history_meta'").fetchone():
            conn.execute('COMMIT')
            return migrate(conn)
        conn.execute(_LEASE_TABLE)
        for sql in (*_TABLES.values(), *_guards().values()):
            conn.execute(sql)
        conn.execute('COMMIT')
    except Exception:
        conn.execute('ROLLBACK')
        raise


def _validate_guards(conn):
    tables = dict(conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'"))
    for name, sql in _TABLES.items():
        if tables.get(name) != sql:
            raise AuthorityHistoryError('missing or modified authority history schema: ' + name)
    actual = dict(conn.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger'"))
    for name, sql in _guards().items():
        if actual.get(name) != sql:
            raise AuthorityHistoryError('missing or modified authority history guard: ' + name)


def capability(conn):
    _validate_guards(conn)
    rows = conn.execute('SELECT * FROM authority_history_meta').fetchall()
    if not rows:
        if any(conn.execute(f'SELECT 1 FROM {table} LIMIT 1').fetchone() for table in (
                'authority_history', 'authority_task_bindings', 'authority_owner_bindings', 'authority_run_bindings', 'authority_source_bindings')):
            raise AuthorityHistoryError('history or bindings without enrollment')
        return None
    if len(rows) != 1:
        raise AuthorityHistoryError('ambiguous authority enrollment')
    row = dict(rows[0])
    if row['singleton'] != 1 or row['version'] != 1 or row['coverage_start'] != 1:
        raise AuthorityHistoryError('unknown authority capability')
    try:
        if str(uuid.UUID(row['incarnation'])) != row['incarnation']:
            raise ValueError
    except (ValueError, TypeError, AttributeError) as exc:
        raise AuthorityHistoryError('invalid authority incarnation') from exc
    tasks = set()
    for binding in conn.execute('SELECT * FROM authority_task_bindings'):
        if type(binding['version']) is not int or binding['version'] != 1:
            raise AuthorityHistoryError('unknown task binding version')
        for key in ('task_id', 'repo_id', 'project_id'):
            _identifier(binding[key])
        tasks.add(binding['task_id'])
    owners = set()
    for owner in conn.execute('SELECT * FROM authority_owner_bindings'):
        digest = owner['token_hash']
        if (owner['task_id'] not in tasks or type(owner['version']) is not int or owner['version'] != 1
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise AuthorityHistoryError('malformed retained owner binding')
        for key in ('consumer_id', 'runtime_id', 'owner_ref'):
            _identifier(owner[key])
        owners.add((owner['task_id'], digest))
    for run in conn.execute('SELECT * FROM authority_run_bindings'):
        if (type(run['run_id']) is not int or run['run_id'] <= 0
                or (run['task_id'], run['token_hash']) not in owners):
            raise AuthorityHistoryError('malformed retained run binding')
    return {k: row[k] for k in ('version', 'incarnation', 'coverage_start')}


def is_standard_kanban_database(relative_path):
    """Classify a normalized, destination-relative path, not just its basename.

    Standard locations are root kanban.db, kanban/**/kanban.db, and the
    same locations within recursively nested profiles/<name> directories.
    Callers must normalize/contain archive destinations before classification.
    """
    from pathlib import Path
    import os
    path = Path(relative_path)
    parts = tuple(os.path.normcase(p) for p in path.parts)
    if path.is_absolute() or '..' in parts:
        return False
    while len(parts) >= 3 and parts[0] == 'profiles':
        parts = parts[2:]
    return parts == ('kanban.db',) or (
        len(parts) >= 2 and parts[0] == 'kanban' and parts[-1] == 'kanban.db'
    )


def refuse_authority_copy(root):
    """Refuse profile copy activation for standard Kanban DB locations.

    Inspect staged copies only, never migrate or repair the source database.
    External/custom DB paths and arbitrary filesystem copies are unsupported.
    """
    from pathlib import Path
    root = Path(root)
    profiles = root / 'profiles'
    if profiles.is_dir():
        for profile in profiles.iterdir():
            if profile.is_dir():
                refuse_authority_copy(profile)
    candidates = [root / 'kanban.db']
    board_root = root / 'kanban'
    if board_root.exists():
        candidates.extend(board_root.rglob('kanban.db'))
    for path in candidates:
        if not is_standard_kanban_database(path.relative_to(root)):
            continue
        if not path.is_file():
            continue
        try:
            conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
            try:
                conn.row_factory = sqlite3.Row
                names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'authority_%' AND name != 'authority_writer_lease'")}
                if names and capability(conn) is not None:
                    raise AuthorityHistoryError('copied authority identity cannot be activated')
            finally:
                conn.close()
        except sqlite3.DatabaseError as exc:
            raise AuthorityHistoryError('cannot establish copied authority identity') from exc


def reserve_removal(conn):
    """Fence prospective enrollment before a filesystem removal, in the same DB.

    A failed filesystem removal deliberately leaves this irreversible marker:
    ordinary board use remains possible, but it cannot become a new authority.
    Old open connections also see the marker, closing the check/rename race.
    """
    from hermes_cli.kanban_db import write_txn
    with write_txn(conn):
        if capability(conn) is not None:
            raise AuthorityHistoryError('cannot remove an enrolled authority board')
        if conn.execute('SELECT 1 FROM authority_history_removal').fetchone():
            return
        conn.execute('INSERT INTO authority_history_removal VALUES (1)')


def enroll(conn):
    from hermes_cli.kanban_db import write_txn
    with write_txn(conn):
        if conn.execute('SELECT 1 FROM authority_history_removal').fetchone():
            raise AuthorityHistoryError('authority enrollment refused: board removal reserved')
        existing = capability(conn)
        if existing is not None:
            return existing
        conn.execute('INSERT INTO authority_history_meta VALUES (1, 1, ?, 1)', (str(uuid.uuid4()),))
        return capability(conn)


def _identifier(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise AuthorityHistoryError('invalid non-secret binding identifier')
    return value


def _token_hash(token):
    import hashlib
    if not isinstance(token, str) or not token:
        raise AuthorityHistoryError('missing owner token')
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def bind_task(conn, task_id, *, repo_id, project_id):
    from hermes_cli.kanban_db import write_txn, _append_event
    values = (task_id, 1, _identifier(repo_id), _identifier(project_id))
    with write_txn(conn):
        if capability(conn) is None:
            raise AuthorityHistoryError('authority not enrolled')
        old = conn.execute('SELECT * FROM authority_task_bindings WHERE task_id=?', (task_id,)).fetchone()
        if old:
            if tuple(old) != values:
                raise AuthorityHistoryError('immutable task binding conflict')
            return dict(old)
        task = conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
        if (not task or task['claim_lock'] is not None or task['current_run_id'] is not None
                or task['status'] not in {'todo', 'ready', 'triage', 'blocked', 'scheduled'}
                or conn.execute('SELECT 1 FROM task_runs WHERE task_id=?', (task_id,)).fetchone()):
            raise AuthorityHistoryError('cannot backfill task with prior or active runs')
        conn.execute('INSERT INTO authority_task_bindings VALUES (?,?,?,?)', values)
        _append_event(conn, task_id, 'history_bound')
        return dict(zip(('task_id', 'version', 'repo_id', 'project_id'), values))


def mint_claim_capability(host=None):
    """An unpredictable bearer value that KEEPS the hostname-prefix contract.

    N1. The previous default was `_claimer_id()`, i.e. host:pid: public,
    derivable from the task row, and therefore useless as the secret the owner
    binding's token hash assumes.

    C2, from the Mac review of 1cbddef5. The first correction minted a bare
    `cap_...` with no host segment, and every host-locality gate in kanban_db
    classifies with `lock.startswith(host_prefix)`. A live local worker whose TTL
    lapsed therefore failed the "is this mine" test and was reclaimed out from
    under itself, on enrolled boards only. The shape below keeps the prefix those
    gates parse while the part after the colon stays unpredictable.

    The value is a BEARER. It is returned to the claimant on the Task and must
    never be written to an event payload, CLI output, dashboard JSON or any other
    published record; use public_claim_label() for those.
    """
    import secrets
    if host is None:
        import socket
        try:
            host = socket.gethostname() or 'unknown'
        except Exception:
            host = 'unknown'
    return f"{host}:{secrets.token_urlsafe(24)}"


UNATTRIBUTED_OWNER = '(unattributed)'


def public_claim_label(lock):
    """A non-secret stand-in for a claim lock, safe to publish.

    C1. Returns only the host segment, which is already public in the task row
    and cannot be replayed as a capability. Never the bearer and never its hash:
    a hash is still a verifier for anyone who can guess the input, and the whole
    point is that published records carry no credential material at all.
    """
    if not lock:
        return None
    text = str(lock)
    host, sep, rest = text.partition(':')
    if not sep or not host or not rest:
        # FAIL CLOSED. Without a separator there is no part of this value we
        # can PROVE is the public host segment, and split(':', 1)[0] returns the
        # WHOLE capability, silently turning the redaction into a no-op. The
        # repo's own suites pass colon-free explicit claimers such as
        # 'private-bearer-token', and bind_owner treats an explicit claimer as a
        # secret, so falling through to the input made the two halves of the
        # code disagree about whether it may be published. A hash is not an
        # option either: it is still a verifier for anyone who can guess it.
        return UNATTRIBUTED_OWNER
    return host


def bind_owner(conn, task_id, *, claimer, consumer_id, runtime_id, owner_ref):
    """Public entry point: opens its own transaction."""
    from hermes_cli.kanban_db import write_txn
    with write_txn(conn):
        bind_owner_locked(conn, task_id, claimer=claimer, consumer_id=consumer_id,
                          runtime_id=runtime_id, owner_ref=owner_ref)


def bind_owner_locked(conn, task_id, *, claimer, consumer_id, runtime_id, owner_ref):
    """Same contract, but assumes the caller already holds the write transaction.

    The claim path must bind the owner in the SAME transaction as the claim CAS,
    so a failed binding rolls the claim back with it. write_txn issues
    BEGIN IMMEDIATE and cannot nest, hence this variant.
    """
    public = tuple(_identifier(v) for v in (consumer_id, runtime_id, owner_ref))
    if any(claimer in v for v in public):
        raise AuthorityHistoryError('bearer token is not a public owner identifier')
    values = (task_id, _token_hash(claimer), 1, *public)
    if capability(conn) is None or not conn.execute(
            'SELECT 1 FROM authority_task_bindings WHERE task_id=?', (task_id,)).fetchone():
        raise AuthorityHistoryError('task not enrolled')
    old = conn.execute('SELECT * FROM authority_owner_bindings WHERE task_id=? AND token_hash=?',
                       values[:2]).fetchone()
    if old:
        if tuple(old) != values:
            raise AuthorityHistoryError('immutable owner binding conflict')
        return
    conn.execute('INSERT INTO authority_owner_bindings VALUES (?,?,?,?,?,?)', values)


KINDS = frozenset('''archived assigned attached attachment_removed block_loop_detected
blocked claim_extended claim_rejected claimed commented completed
completion_blocked_hallucination created decomposed dependency_wait edited gave_up
heartbeat linked promoted promoted_manual reclaim_deferred reclaimed respawn_guarded
scheduled spawned specified stale suspected_hallucinated_references timed_out
tip_scratch_workspace unblocked unlinked crashed rate_limited protocol_violation
spawn_failed status reprioritized history_bound claim_renewed deleted
changes_requested descendant_invalidated imported pr_acceptance reconciled
review_reopened review_requested'''.split())
TASK_STATES = frozenset('triage todo ready running review blocked done archived deleted scheduled'.split())
# ``todo``: invalidate_descendants_for_parent_reopen closes a descendant's run
# with status='todo'. Unusual for a RUN, but the validator records what the
# system does; refusing it would fail an enrolled board closed on a first-party
# write. Derived by enumerating every literal written to task_runs.status, not
# by listing the ones that came to mind.
RUN_STATES = frozenset(
    "running done blocked crashed timed_out failed released reclaimed completed spawn_failed gave_up stale rate_limited scheduled todo".split()
)


def capture(conn, task_id, kind, source_event_id, run_id=None):
    """Append a versioned, token-free fact in the source event's transaction.

    Deliberately excludes arbitrary event payload, prose, error and metadata.
    Each record describes the observed task/run state at that emitted event.
    It does not assert process termination or physical execution exclusion.
    """
    cap = capability(conn)
    binding = conn.execute('SELECT * FROM authority_task_bindings WHERE task_id=?', (task_id,)).fetchone()
    if not binding:
        return
    if cap is None or not conn.in_transaction:
        raise AuthorityHistoryError('enrolled capture requires authority transaction')
    if kind not in KINDS:
        raise AuthorityHistoryError('unknown history event kind: ' + str(kind))
    task = conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
    if task is None and kind != 'deleted':
        raise AuthorityHistoryError('missing enrolled task')
    status = 'deleted' if kind == 'deleted' else task['status']
    if status not in TASK_STATES:
        raise AuthorityHistoryError('unknown task state')
    state = {'status': status, 'current_run_id': None if kind == 'deleted' else task['current_run_id'],
             'claim_expires': None if kind == 'deleted' else task['claim_expires']}
    if run_id is None and task is not None:
        run_id = task['current_run_id']
    run = conn.execute('SELECT * FROM task_runs WHERE id=?', (run_id,)).fetchone() if run_id else None
    owner = None
    run_state = None
    if run is not None:
        if run['task_id'] != task_id or run['status'] not in RUN_STATES:
            raise AuthorityHistoryError('unknown or mismatched run state')
        if kind == 'claimed':
            token_hash = _token_hash(run['claim_lock'])
            if not conn.execute('SELECT 1 FROM authority_owner_bindings WHERE task_id=? AND token_hash=?', (task_id, token_hash)).fetchone():
                raise AuthorityHistoryError('unknown owner binding')
            conn.execute('INSERT INTO authority_run_bindings VALUES (?,?,?)', (run_id, task_id, token_hash))
        binding_row = conn.execute('SELECT * FROM authority_run_bindings WHERE run_id=?', (run_id,)).fetchone()
        if run['claim_lock'] is not None and (binding_row is None
                or binding_row['task_id'] != task_id
                or binding_row['token_hash'] != _token_hash(run['claim_lock'])):
            raise AuthorityHistoryError('mismatched immutable run owner credential')
        owner_row = conn.execute('''SELECT o.version, o.consumer_id, o.runtime_id, o.owner_ref
            FROM authority_run_bindings r JOIN authority_owner_bindings o
            ON r.task_id=o.task_id AND r.token_hash=o.token_hash
            WHERE r.run_id=? AND r.task_id=?''', (run_id, task_id)).fetchone()
        if owner_row:
            owner = dict(owner_row)
        elif run['claim_lock'] is not None or run['status'] == 'running':
            raise AuthorityHistoryError('missing immutable run owner')
        run_state = {k: run[k] for k in ('id', 'status', 'claim_expires', 'started_at', 'ended_at')}
    observed = conn.execute('SELECT created_at FROM task_events WHERE id=? AND task_id=? AND kind=?',
                            (source_event_id, task_id, kind)).fetchone()
    if observed is None:
        raise AuthorityHistoryError('missing source event provenance')
    record = {'incarnation': cap['incarnation'], 'binding': dict(binding), 'kind': kind,
              'source_event_id': source_event_id, 'observed_at': observed[0], 'task_state': state,
              'run_state': run_state, 'owner': owner}
    conn.execute('INSERT INTO authority_source_bindings VALUES(?,?,?,?,?)',
                 (source_event_id, task_id, run_state['id'] if run_state else None, kind, observed[0]))
    _validate_record(record, cap, conn)
    conn.execute('INSERT INTO authority_history(version,record) VALUES (1,?)',
                 (json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=False),))


def validate_existing(conn):
    """Validate before legacy repair or any authority write; fresh DBs may lack schema."""
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'authority_%' AND name != 'authority_writer_lease'")}
    if not names:
        return
    if not set(_TABLES) <= names:
        raise AuthorityHistoryError('incomplete authority history schema')
    cap = capability(conn)
    if cap is None:
        return
    _validated_stream(conn, cap)
    for task in conn.execute('SELECT t.* FROM tasks t JOIN authority_task_bindings b ON b.task_id=t.id'):
        if task['status'] not in TASK_STATES - {'deleted'}:
            raise AuthorityHistoryError('unknown enrolled task state')
        run = conn.execute('SELECT * FROM task_runs WHERE id=?', (task['current_run_id'],)).fetchone()
        if task['status'] == 'running':
            if (run is None or run['task_id'] != task['id'] or run['status'] != 'running'
                    or run['ended_at'] is not None or not task['claim_lock']
                    or task['claim_lock'] != run['claim_lock']):
                raise AuthorityHistoryError('enrolled run invariant requires repair')
        elif task['current_run_id'] is not None or task['claim_lock'] is not None:
            raise AuthorityHistoryError('enrolled task invariant requires repair')
    for run in conn.execute('''SELECT r.* FROM task_runs r
            JOIN authority_task_bindings b ON b.task_id=r.task_id'''):
        binding = conn.execute('SELECT * FROM authority_run_bindings WHERE run_id=?', (run['id'],)).fetchone()
        if binding is not None:
            if binding['task_id'] != run['task_id'] or not conn.execute(
                    'SELECT 1 FROM authority_owner_bindings WHERE task_id=? AND token_hash=?',
                    (run['task_id'], binding['token_hash'])).fetchone():
                raise AuthorityHistoryError('mismatched immutable run owner')
        if run['ended_at'] is None:
            task = conn.execute('SELECT * FROM tasks WHERE id=?', (run['task_id'],)).fetchone()
            if (run['status'] != 'running' or task is None or task['status'] != 'running'
                    or task['current_run_id'] != run['id'] or binding is None
                    or not run['claim_lock'] or task['claim_lock'] != run['claim_lock']
                    or binding['token_hash'] != _token_hash(run['claim_lock'])):
                raise AuthorityHistoryError('enrolled run owner invariant requires repair')
        elif run['status'] == 'running' or run['claim_lock'] is not None:
            raise AuthorityHistoryError('terminal run invariant requires repair')
    return True


def _ownership_snapshot(conn):
    tasks = {r['id']: dict(r) for r in conn.execute('''SELECT t.id, t.status,
        t.claim_lock, t.claim_expires, t.current_run_id FROM tasks t
        JOIN authority_task_bindings b ON b.task_id=t.id''')}
    runs = {r['id']: dict(r) for r in conn.execute('''SELECT r.id, r.task_id,
        r.status, r.claim_lock, r.claim_expires, r.started_at, r.ended_at
        FROM task_runs r JOIN authority_task_bindings b ON b.task_id=r.task_id''')}
    return tasks, runs


def audit_start(conn):
    """Capture the enrolled ownership fields, not task prose or failure counters."""
    if not validate_existing(conn):
        return None
    return (*_ownership_snapshot(conn), conn.execute('SELECT COALESCE(MAX(sequence),0) FROM authority_history').fetchone()[0])


def audit_finish(conn, before):
    """Refuse un-emitted net authority transitions before COMMIT.

    Intermediate SQL statements are not independent committed transitions.
    The connection-local owned-writer gate covers unsupported SQL callers.
    """
    validate_existing(conn)
    if before is None:
        return
    old_tasks, old_runs, highwater = before
    tasks, runs = _ownership_snapshot(conn)
    records = [json.loads(r[0]) for r in conn.execute('SELECT record FROM authority_history WHERE sequence>? ORDER BY sequence', (highwater,))]
    for task_id in old_tasks.keys() | tasks.keys():
        matching = [r for r in records if r['binding']['task_id'] == task_id]
        if old_tasks.get(task_id) == tasks.get(task_id) and not matching:
            continue
        if not matching:
            raise AuthorityHistoryError('unjournaled task transition')
        state = matching[-1]['task_state']
        expected = ({k: tasks[task_id][k] for k in ('status', 'current_run_id', 'claim_expires')}
                    if task_id in tasks else {'status': 'deleted', 'current_run_id': None, 'claim_expires': None})
        if state != expected:
            raise AuthorityHistoryError('unjournaled final task state')
    for run_id in old_runs.keys() | runs.keys():
        task_id = (runs.get(run_id) or old_runs[run_id])['task_id']
        matching = [r for r in records if r['binding']['task_id'] == task_id]
        observations = [r['run_state'] for r in matching
                        if r['run_state'] is not None and r['run_state']['id'] == run_id]
        if old_runs.get(run_id) == runs.get(run_id) and not observations:
            continue
        if task_id not in tasks and matching and matching[-1]['kind'] == 'deleted':
            continue
        expected = ({k: runs[run_id][k] for k in ('id', 'status', 'claim_expires', 'started_at', 'ended_at')}
                    if run_id in runs else None)
        observations = [r['run_state'] for r in matching
                        if r['run_state'] is not None and r['run_state']['id'] == run_id]
        if expected is None or not observations or observations[-1] != expected:
            raise AuthorityHistoryError('unjournaled run transition')


def _validate_record(record, cap, conn):
    required = {'incarnation', 'binding', 'kind', 'source_event_id', 'observed_at', 'task_state', 'run_state', 'owner'}
    if (not isinstance(record, dict) or set(record) != required
            or record['incarnation'] != cap['incarnation'] or record['kind'] not in KINDS
            or type(record['source_event_id']) is not int or record['source_event_id'] <= 0):
        raise AuthorityHistoryError('malformed or unknown history record')
    binding = record['binding']
    if not isinstance(binding, dict) or set(binding) != {'task_id', 'version', 'repo_id', 'project_id'} or type(binding['version']) is not int or binding['version'] != 1:
        raise AuthorityHistoryError('unknown binding version or shape')
    for key in ('task_id', 'repo_id', 'project_id'):
        _identifier(binding[key])
    state = record['task_state']
    if not isinstance(state, dict) or set(state) != {'status', 'current_run_id', 'claim_expires'} or state['status'] not in TASK_STATES:
        raise AuthorityHistoryError('unknown task state')
    for key in ('current_run_id', 'claim_expires'):
        if state[key] is not None and type(state[key]) is not int:
            raise AuthorityHistoryError('invalid task state field')
    run = record['run_state']
    if run is not None:
        if not isinstance(run, dict) or set(run) != {'id', 'status', 'claim_expires', 'started_at', 'ended_at'} or run['status'] not in RUN_STATES:
            raise AuthorityHistoryError('unknown run state')
        for key in ('id', 'claim_expires', 'started_at', 'ended_at'):
            if run[key] is not None and type(run[key]) is not int:
                raise AuthorityHistoryError('invalid run state field')
    owner = record['owner']
    if owner is not None:
        if not isinstance(owner, dict) or set(owner) != {'version', 'consumer_id', 'runtime_id', 'owner_ref'} or type(owner['version']) is not int or owner['version'] != 1:
            raise AuthorityHistoryError('unknown owner binding')
        for key in ('consumer_id', 'runtime_id', 'owner_ref'):
            _identifier(owner[key])
    if record['kind'] == 'claimed' and (run is None or owner is None):
        raise AuthorityHistoryError('grant missing run owner')
    if type(record['observed_at']) is not int:
        raise AuthorityHistoryError('invalid observed source timestamp')
    source = conn.execute('SELECT * FROM authority_source_bindings WHERE source_event_id=?',
                          (record['source_event_id'],)).fetchone()
    expected_source = {'source_event_id': record['source_event_id'], 'task_id': binding['task_id'],
                       'run_id': run['id'] if run else None, 'kind': record['kind'],
                       'observed_at': record['observed_at']}
    if source is None or dict(source) != expected_source:
        raise AuthorityHistoryError('record contradicts retained source provenance')
    stored = conn.execute('SELECT * FROM authority_task_bindings WHERE task_id=?', (binding['task_id'],)).fetchone()
    if stored is None or dict(stored) != binding:
        raise AuthorityHistoryError('mismatched retained task binding')
    if state['current_run_id'] is not None and state['current_run_id'] <= 0:
        raise AuthorityHistoryError('invalid current run identity')
    if run is not None:
        if type(run['id']) is not int or run['id'] <= 0 or type(run['started_at']) is not int:
            raise AuthorityHistoryError('invalid run identity or start')
        if (run['status'] == 'running') != (run['ended_at'] is None):
            raise AuthorityHistoryError('invalid terminal run relation')
        retained = conn.execute('''SELECT r.task_id, o.version, o.consumer_id, o.runtime_id, o.owner_ref
            FROM authority_run_bindings r LEFT JOIN authority_owner_bindings o
            ON r.task_id=o.task_id AND r.token_hash=o.token_hash WHERE r.run_id=?''', (run['id'],)).fetchone()
        if retained is not None:
            expected = dict(retained)
            if expected.pop('task_id') != binding['task_id'] or owner != expected:
                raise AuthorityHistoryError('mismatched retained run owner')
        elif owner is not None or run['status'] == 'running':
            raise AuthorityHistoryError('missing retained run binding')
    elif owner is not None:
        raise AuthorityHistoryError('owner without run')
    if state['status'] == 'running':
        if (run is None or run['status'] != 'running' or owner is None
                or state['current_run_id'] != run['id']):
            raise AuthorityHistoryError('invalid running task relation')
    elif state['current_run_id'] is not None or state['claim_expires'] is not None:
        raise AuthorityHistoryError('invalid inactive task relation')


def _decode_record(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise AuthorityHistoryError('duplicate history JSON key')
            result[key] = value
        return result
    def constant(value):
        raise AuthorityHistoryError('non-finite history JSON value')
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError) as exc:
        raise AuthorityHistoryError('malformed history record') from exc


def _validated_stream(conn, cap):
    records = []
    sources = set()
    for expected, row in enumerate(conn.execute('SELECT * FROM authority_history ORDER BY sequence'), 1):
        if type(row['sequence']) is not int or row['sequence'] != expected:
            raise AuthorityHistoryError('unknown sequence or incomplete coverage')
        if type(row['version']) is not int or row['version'] != 1:
            raise AuthorityHistoryError('unknown record version')
        record = _decode_record(row['record'])
        _validate_record(record, cap, conn)
        if record['source_event_id'] in sources:
            raise AuthorityHistoryError('reused source event provenance')
        sources.add(record['source_event_id'])
        records.append({'sequence': row['sequence'], 'version': 1, **record})
    return records


def read(conn, *, incarnation, after=0, limit=1000):
    """Read a consistent global sequence snapshot, never infer pre-enrollment facts."""
    if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 10000:
        raise AuthorityHistoryError('invalid history cursor or limit')
    if conn.in_transaction:
        raise AuthorityHistoryError('reader requires its own snapshot')
    conn.execute('BEGIN')
    try:
        cap = capability(conn)
        if cap is None or cap['incarnation'] != incarnation:
            raise AuthorityHistoryError('unknown or stale authority incarnation')
        stream = _validated_stream(conn, cap)
        highwater = len(stream)
        if after > highwater:
            raise AuthorityHistoryError('unknown sequence or incomplete coverage')
        records = stream[after:after + limit]
        return {**cap, 'highwater': highwater, 'after': after, 'records': records,
                'next_sequence': records[-1]['sequence'] if records else after}
    finally:
        conn.execute('ROLLBACK')
