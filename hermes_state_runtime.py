"""Canonical runtime transactions; callers authorize and resolve session identity.

No execution, transport, or authority objects live here. Every mutation uses the
SessionDB transaction owner, including its inode guard and SQLite retry policy.
"""
import json
import uuid
from typing import NamedTuple, Optional

from agent.conversation_compression_archive import ABSORBED_ROW_IDS
from agent.message_metadata import (
    CANONICAL_ROW, DATA_USE, DB_ROW_SNAPSHOT, MESSAGE_UID, PERSISTENCE_ONLY_MESSAGE_FIELDS, TOOL_CALL_UID, TOOL_CALL_UIDS,
)
from hermes_state_keys import admission_fingerprint


class SessionGoneEvent(NamedTuple):
    """The ``on_session_delete`` event a committed runtime mutation owes its plugins."""
    session_ids: tuple
    reason: str


class RuntimeStoreError(ValueError):
    """A refusal by the state authority. ``reason`` is the bounded wire code (callers compare it as a
    string); ``field`` and ``detail`` name the check that fired and why, so a client and the gateway
    log can tell the reasons' many checks apart instead of reporting the bare code."""

    def __init__(self, reason: str, *, field: str | None = None, detail: str | None = None):
        self.reason = reason
        self.field = _one_line(field, 80)
        self.detail = _one_line(detail, 160)
        super().__init__(reason)


def _one_line(value, limit):
    """A refusal's ``field``/``detail`` as the log line and the client frame carry it: either may hold
    client text, so whitespace collapses to single spaces, other non-printables drop (no forged log
    lines), and the length is bounded below the client's frame limit."""
    if value is None:
        return None
    text = ''.join(ch for ch in ' '.join(str(value).split()) if ch.isprintable())
    return text[:limit] or None


def _text(value):
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise RuntimeStoreError('invalid_params')
    return value


def _json(value):
    if not isinstance(value, dict):
        raise RuntimeStoreError('invalid_params')
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeStoreError('invalid_params') from exc


def _epoch(conn, epoch):
    row = conn.execute('SELECT epoch FROM runtime_epoch WHERE singleton=1').fetchone()
    if type(epoch) is not int or row is None or row[0] != epoch:
        raise RuntimeStoreError('stale_epoch')


def _session(conn, session_id):
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError('not_found')
    return row


def _admission(conn, admission_id):
    row = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (admission_id,)).fetchone()
    if row is None:
        from hermes_state_terminal import terminal_admission
        row = terminal_admission(conn, admission_id)
    if row is None:
        raise RuntimeStoreError('not_found')
    return row


def _retire_admission_workers(conn, row, owner_epoch):
    """Retire the worker rows this admission's generation owns on BOTH the logical
    FIFO owner and its current physical transcript (reset/compression move the private
    worker assignment to the child while target_session_id stays the logical ID)."""
    lineage = json.loads(row['lineage_json'])
    targets = list({row['target_session_id'], lineage[-1] if lineage else row['target_session_id']})
    scope = f"session_id IN ({','.join('?' * len(targets))}) AND generation=? AND owner_epoch=?"
    # Settlement closes an execution that never wrote execution.finish: its read copies go too.
    from hermes_state_runtime_workers import compact_terminal_receipts
    for (execution_id,) in conn.execute(f"SELECT execution_id FROM worker_executions WHERE {scope} "
                                        "AND status!='terminal'", (*targets, row['generation'], owner_epoch)).fetchall():
        compact_terminal_receipts(conn, execution_id)
    conn.execute(f"UPDATE worker_executions SET status='terminal' WHERE {scope}",
                 (*targets, row['generation'], owner_epoch))


def _row(row):
    if row is None:
        return None
    result = dict(row)
    result['payload'] = json.loads(result.pop('payload_json'))
    result.pop('payload_digest')
    result.pop('lineage_json')
    result.pop('retired_at', None)  # tombstone retention stamp, not admission state
    return result


def begin_runtime_epoch(db, *, instance_id: str) -> int:
    """Caller MUST hold the profile ownership lock, not merely an endpoint ticket."""
    _text(instance_id)
    def write(conn):
        conn.execute('''INSERT INTO runtime_epoch(singleton,epoch,instance_id) VALUES(1,1,?)
            ON CONFLICT(singleton) DO UPDATE SET epoch=runtime_epoch.epoch+1, instance_id=excluded.instance_id''', (instance_id,))
        return conn.execute('SELECT epoch FROM runtime_epoch WHERE singleton=1').fetchone()[0]
    return db._execute_write(write)


# Main's busy-queue cap (GatewayRunner._BUSY_QUEUE_MAX_PENDING): queued inputs one session holds.
ADMISSION_QUEUE_MAX = 32


def admit_session_input(db, *, epoch: int, principal_id: str, session_id: str,
                        request_id: str, payload: dict, intent: str = 'queue',
                        _authorize_write=None, _replay_guard=None) -> dict:
    """Admit input; the trusted private guard raises to refuse a NEW write.

    The guard receives the owning transaction connection, must not commit it or
    perform external effects, and may run again on SQLite retry. Exact existing
    and terminal replays bypass it: they cannot create or change accepted work.
    ``_replay_guard(conn)`` runs on the exact live replay instead, inside the same
    transaction (a replacement's binding to the admission it retired).
    """
    for value in (principal_id, session_id, request_id):
        _text(value)
    if intent not in ('queue', 'steer', 'redirect'):
        raise RuntimeStoreError('invalid_params')
    encoded = _json(payload)
    from hermes_state_terminal import retry_terminal_admission
    retired = retry_terminal_admission(db, epoch=epoch, principal_id=principal_id, session_id=session_id,
        request_id=request_id, payload=payload, intent=intent)
    if retired is not None:
        return retired
    digest = admission_fingerprint(canonical_target=session_id, payload={'input': json.loads(encoded), 'intent': intent})
    def write(conn):
        _epoch(conn, epoch)
        _session(conn, session_id)
        old = conn.execute('''SELECT * FROM session_admissions
            WHERE principal_id=? AND target_session_id=? AND request_id=?''', (principal_id, session_id, request_id)).fetchone()
        if old is not None:
            if old['payload_digest'] != digest:
                raise RuntimeStoreError('admission_conflict')
            if _replay_guard is not None:
                _replay_guard(conn)
            return _row(old)
        queued = conn.execute('''SELECT COUNT(*) FROM session_admissions WHERE target_session_id=?
            AND status='queued' ''', (session_id,)).fetchone()[0]
        # Main's cap holds in both states: a FIFO paused behind an unknown turn runs every accepted
        # input after Discard, so it must not grow without bound while nobody resolves it.
        if queued >= ADMISSION_QUEUE_MAX:
            raise RuntimeStoreError('queue_full')
        if _authorize_write is not None:
            _authorize_write(conn)
        admission_id = uuid.uuid4().hex
        conn.execute('''INSERT INTO session_admissions(admission_id,request_id,principal_id,
            target_session_id,lineage_json,payload_json,payload_digest,intent,status,owner_epoch)
            VALUES(?,?,?,?,?,?,?,?,'queued',?)''',
            (admission_id, request_id, principal_id, session_id, json.dumps([session_id]), encoded, digest, intent, epoch))
        return _row(_admission(conn, admission_id))
    return db._execute_write(write)


def get_session_admission(db, *, admission_id: str) -> dict | None:
    with db._read_ctx() as conn:
        from hermes_state_terminal import terminal_admission
        row = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (admission_id,)).fetchone()
        return _row(row if row is not None else terminal_admission(conn, admission_id))


def find_session_admission(db, *, principal_id: str, session_id: str, request_id: str) -> dict | None:
    """The admission at its exact durable identity: the UNIQUE (principal, target, request) key."""
    with db._read_ctx() as conn:
        return _row(conn.execute('SELECT * FROM session_admissions WHERE principal_id=? AND target_session_id=? '
                                 'AND request_id=?', (principal_id, session_id, request_id)).fetchone())


def list_session_admissions(db, *, session_id: str, pending_only: bool = True) -> list[dict]:
    with db._read_ctx() as conn:
        # Pending is an explicit index-range IN, not ``!='terminal'``: a negated status cannot
        # bound the (target, status, seq) index, so every probe scanned the session's history.
        sql = ("SELECT * FROM session_admissions WHERE target_session_id=? "
               + ("AND status IN ('queued','started','unknown') " if pending_only else "") + "ORDER BY seq")
        return [_row(row) for row in conn.execute(sql, (session_id,))]


_ANY_HEAD = object()


def claim_session_input(db, *, epoch: int, session_id: str, _guard=None, head=_ANY_HEAD) -> dict | bool | None:
    """Claim the FIFO head. ``_guard(conn)`` (owner admission gate) raises to refuse inside the
    write transaction, so a drain that began while this waited on the writer still wins.
    ``head`` is the admission id the caller's preclaim checks validated (None: it saw nothing
    queued); if the head is another row by the time this write runs, nothing is claimed and False
    tells the caller to re-read and validate the row that is now first."""
    def write(conn):
        _epoch(conn, epoch)
        if _guard is not None:
            _guard(conn)
        session = _session(conn, session_id)
        blocked = conn.execute("SELECT status FROM session_admissions WHERE target_session_id=? AND status IN ('started','unknown')", (session_id,)).fetchall()
        if any(row[0] == 'unknown' for row in blocked):
            raise RuntimeStoreError('unknown_execution')
        from hermes_state_runtime_workers import worker_states
        workers = worker_states(conn, session_id)
        if 'unknown' in workers:
            raise RuntimeStoreError('unknown_execution')
        if blocked or workers:
            return None
        row = conn.execute("SELECT * FROM session_admissions WHERE target_session_id=? AND status='queued' ORDER BY seq LIMIT 1", (session_id,)).fetchone()
        if head is not _ANY_HEAD and (row['admission_id'] if row is not None else None) != head:
            return False
        if row is None:
            return None
        generation = session['runtime_generation'] + 1
        conn.execute('UPDATE sessions SET runtime_generation=? WHERE id=?', (generation, session_id))
        changed = conn.execute("UPDATE session_admissions SET status='started',owner_epoch=?,generation=? WHERE admission_id=? AND status='queued'", (epoch, generation, row['admission_id']))
        if changed.rowcount != 1:
            raise RuntimeStoreError('stale_generation')
        claimed = _row(_admission(conn, row['admission_id']))
        # The transcript high-water mark at the claim (``messages.id`` is AUTOINCREMENT across every
        # session): a row above it on this conversation was written by this turn, so settlement can
        # tell this turn's own failed-turn boundary from an earlier turn's (#134889 GW-21).
        claimed['transcript_mark'] = conn.execute('SELECT COALESCE(MAX(id), 0) FROM messages').fetchone()[0]
        return claimed
    return db._execute_write(write)


def settle_session_input(db, *, epoch: int, admission_id: str, generation: int, outcome: str,
                         result: dict | None = None, _transfers=None, _terminal_write=None) -> dict:
    """``_terminal_write(conn, row)`` is the owner's same-transaction effect (a transcript notice,
    ``gateway.session_results.finish_result``); like ``resolve_unknown_session_input``'s it must
    not commit, may run again on SQLite retry, and raising rolls the settlement back."""
    if outcome not in ('completed', 'interrupted', 'rejected', 'failed'):
        raise RuntimeStoreError('invalid_params')
    from hermes_state_terminal import compact_result, encode_result
    def write(conn):
        _epoch(conn, epoch)
        row = _admission(conn, admission_id)
        session = _session(conn, row['target_session_id'])
        if (row['status'] != 'started' or row['owner_epoch'] != epoch
                or type(generation) is not int or row['generation'] != generation
                or session['runtime_generation'] != generation):
            raise RuntimeStoreError('stale_generation')
        _retire_admission_workers(conn, row, epoch)
        if _terminal_write is not None:
            _terminal_write(conn, row)
        if result is not None:
            from hermes_state_terminal import RESULT_PREFIX
            encoded = _json(encode_result(compact_result(result, user_message=json.loads(row['payload_json']).get('text'))))
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                         'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                         (RESULT_PREFIX + admission_id, encoded))
        conn.execute("UPDATE session_admissions SET status='terminal',outcome=? WHERE admission_id=?", (outcome, admission_id))
        settle_steer_transfers(conn, epoch=epoch, generation=generation, transfers=_transfers, outcome=outcome)
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (row['target_session_id'],))
        return _row(_admission(conn, admission_id))
    return db._execute_write(write)


_CANCEL_QUEUED = "UPDATE session_admissions SET status='terminal',outcome='cancelled' WHERE admission_id=?"


def cancel_session_input(db, *, epoch: int, admission_id: str) -> dict:
    def write(conn):
        _epoch(conn, epoch)
        row = _admission(conn, admission_id)
        if row['status'] == 'unknown':
            raise RuntimeStoreError('unknown_execution')
        # A steer in transfer reads as its running claim and settles with it: as started, not cancellable.
        if row['status'] == 'started' or row['outcome'] == STEERING:
            raise RuntimeStoreError('stale_generation')
        if row['status'] == 'queued':
            conn.execute(_CANCEL_QUEUED, (admission_id,))
        return _row(_admission(conn, admission_id))
    return db._execute_write(write)


# A queued admission handed to its FIFO's running claim: terminal (never claimable, outside the queue
# cap) but not yet an outcome. It is bound to that claim's ``(owner_epoch, generation)`` and settles
# with it: ``steered`` once the claim completes with the text delivered, ``interrupted`` otherwise, or
# back to ``queued`` (its own FIFO place) when the text never reached the claim's agent.
STEERING = 'steering'


def steer_replayed(row) -> bool:
    """``row`` is a committed transfer (still in flight, or settled ``steered``): an exact retry of
    ``prompt.steer_queued`` replays it."""
    return row['status'] == 'terminal' and row['outcome'] in (STEERING, 'steered')


def _requeue_transfer(conn, *, admission_id: str, epoch: int, generation: int) -> bool:
    """A transfer still bound to the claim ``(epoch, generation)`` goes back to its FIFO place (``seq``)."""
    return conn.execute("UPDATE session_admissions SET status='queued',outcome=NULL,generation=NULL "
                        "WHERE admission_id=? AND status='terminal' AND outcome=? AND owner_epoch=? "
                        "AND generation=?", (admission_id, STEERING, epoch, generation)).rowcount == 1


def steer_session_input(db, *, epoch: int, admission_id: str, generation: int) -> tuple[dict, bool]:
    """Hand a still-queued admission to its FIFO's running claim: queued -> terminal ``steering``
    bound to ``generation`` in one write, only while ``generation`` is the started claim this owner
    holds, so the drain can never also claim it. ``(row, True)`` when this call committed the
    transfer, ``(row, False)`` for an exact retry of one already committed (never delivered twice)."""
    def write(conn):
        _epoch(conn, epoch)
        row = _admission(conn, admission_id)
        if steer_replayed(row):
            return _row(row), False
        if row['status'] != 'queued':
            raise RuntimeStoreError('admission_started')
        if type(generation) is not int or conn.execute(
                "SELECT 1 FROM session_admissions WHERE target_session_id=? AND status='started' "
                "AND owner_epoch=? AND generation=?", (row['target_session_id'], epoch, generation)).fetchone() is None:
            raise RuntimeStoreError('session_idle')
        conn.execute("UPDATE session_admissions SET status='terminal',outcome=?,owner_epoch=?,generation=? "
                     "WHERE admission_id=?", (STEERING, epoch, generation, admission_id))
        return _row(_admission(conn, admission_id)), True
    return db._execute_write(write)


def settle_steer_transfers(conn, *, epoch: int, generation: int, transfers, outcome: str) -> None:
    """In the receiving claim's terminal write: each admission bound to it settles ``steered`` only if
    its text reached the claim's agent (``transfers[id]``) and the claim completed; a delivered text
    under any other ending is ``interrupted`` (never replayed); an undelivered one is queued again."""
    for admission_id, delivered in (transfers or {}).items():
        if not delivered:
            _requeue_transfer(conn, admission_id=admission_id, epoch=epoch, generation=generation)
            continue
        conn.execute("UPDATE session_admissions SET outcome=? WHERE admission_id=? AND status='terminal' "
                     "AND outcome=? AND owner_epoch=? AND generation=?",
                     ('steered' if outcome == 'completed' else 'interrupted', admission_id, STEERING, epoch, generation))


def revert_steer_input(db, *, epoch: int, admission_id: str, generation: int) -> bool:
    """A transfer whose text never reached the claim's agent: the admission is queued again, in its
    original FIFO place (its ``seq``). False when the claim's settlement already decided it."""
    def write(conn):
        _epoch(conn, epoch)
        return _requeue_transfer(conn, admission_id=admission_id, epoch=epoch, generation=generation)
    return db._execute_write(write)


REPLACEMENT_PREFIX = 'gateway.queued_replacement.v1.'


def replacement_binding(principal_id: str, session_id: str, request_id: str) -> str:
    """The durable key binding a replacement's identity to the admission it retired."""
    return REPLACEMENT_PREFIX + admission_fingerprint(canonical_target=session_id,
                                                      payload={'principal': principal_id, 'request': request_id})


def retire_replaced_input(conn, admission_id: str, binding: str) -> None:
    """Inside the replacement's own admission write: the replaced row, still queued, is cancelled
    in the same transaction that admits its successor at the FIFO tail, which is bound to it."""
    row = _admission(conn, admission_id)
    if row['status'] != 'queued':
        raise RuntimeStoreError('admission_started')
    conn.execute(_CANCEL_QUEUED, (admission_id,))
    conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)', (binding, json.dumps(admission_id)))


def check_replaced_input(conn, admission_id: str, binding: str) -> None:
    """The exact replay of a replacement answers only the request that retired ``admission_id``."""
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (binding,)).fetchone()
    if saved is None or json.loads(saved[0]) != admission_id:
        raise RuntimeStoreError('admission_conflict')


def rehome_queued_inputs(db, *, epoch: int, old_session_id: str, new_session_id: str,
                         principal_prefix: str) -> list[str]:
    """Re-target the old owner's still-queued ``principal_prefix*`` admissions onto
    ``new_session_id`` in one transaction (the compression-exhaustion auto-reset). Each row keeps
    its admission id, request id and seq, so FIFO order and retry dedupe survive; the digest is
    re-keyed to the new target. Nothing moves while the old FIFO holds an ``unknown`` row (the
    operator's to resume). Returns the moved admission ids in FIFO order."""
    for value in (old_session_id, new_session_id, principal_prefix):
        _text(value)
    def write(conn):
        _epoch(conn, epoch)
        _session(conn, new_session_id)
        if conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status='unknown'",
                        (old_session_id,)).fetchone():
            return []
        rows = conn.execute("SELECT * FROM session_admissions WHERE target_session_id=? AND status='queued' "
                            "AND substr(principal_id,1,?)=? ORDER BY seq",
                            (old_session_id, len(principal_prefix), principal_prefix)).fetchall()
        for row in rows:
            digest = admission_fingerprint(canonical_target=new_session_id, payload={
                'input': json.loads(row['payload_json']), 'intent': row['intent']})
            conn.execute("UPDATE session_admissions SET target_session_id=?,lineage_json=?,payload_digest=? "
                         "WHERE admission_id=? AND status='queued'",
                         (new_session_id, json.dumps([new_session_id]), digest, row['admission_id']))
        if rows:
            for sid in (old_session_id, new_session_id):
                conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (sid,))
        return [row['admission_id'] for row in rows]
    return db._execute_write(write)


def retarget_local_automation(db, *, epoch: int, session_id: str, principal_id: str,
                               old_targets, new_target: str) -> list[str]:
    """Point a LOCAL owner's still-queued ``principal_id`` automation rows (a Bot Chat's queued
    ``hermes peer dm`` turns, plugin and completion notices) that name a pre-reset transcript in
    ``old_targets`` at ``new_target``, in one transaction (the compression-exhaustion auto-reset).
    A LOCAL FIFO's owner is its receipt id and survives the reset; only the descriptor's ``target``
    goes stale, and the drain refuses a stale one (``admission_conflict``), pausing the FIFO for
    good. Admission id, request id and seq are kept; the digest is re-keyed to the new payload.
    Nothing moves behind an ``unknown`` head. Returns the retargeted admission ids in FIFO order."""
    for value in (session_id, principal_id, new_target, *old_targets):
        _text(value)
    old_targets = set(old_targets)

    def write(conn):
        _epoch(conn, epoch)
        _session(conn, new_target)
        if conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status='unknown'",
                        (session_id,)).fetchone():
            return []
        moved = []
        for row in conn.execute("SELECT * FROM session_admissions WHERE target_session_id=? AND status='queued' "
                                "AND principal_id=? ORDER BY seq", (session_id, principal_id)).fetchall():
            payload = json.loads(row['payload_json'])
            descriptor = payload.get('local_automation_v1') if isinstance(payload, dict) else None
            if not isinstance(descriptor, dict) or descriptor.get('target') not in old_targets:
                continue
            descriptor['target'] = new_target
            digest = admission_fingerprint(canonical_target=session_id, payload={'input': payload, 'intent': row['intent']})
            conn.execute("UPDATE session_admissions SET payload_json=?,payload_digest=? WHERE admission_id=? AND status='queued'",
                         (_json(payload), digest, row['admission_id']))
            moved.append(row['admission_id'])
        if moved:
            conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (session_id,))
        return moved
    return db._execute_write(write)


def recover_session_inputs(db, *, epoch: int) -> int:
    """Never replay started work. Live worker adoption is a separate explicit operation."""
    def write(conn):
        _epoch(conn, epoch)
        from hermes_state_runtime_workers import retire_stale_children
        retire_stale_children(conn, epoch)
        conn.execute("UPDATE worker_executions SET status='unknown' WHERE status IN ('registered','running') AND owner_epoch!=?", (epoch,))
        # A steer handed to a lost claim may or may not have reached its agent: never replayed.
        conn.execute("UPDATE session_admissions SET outcome='interrupted' WHERE status='terminal' AND outcome=? "
                     "AND owner_epoch!=?", (STEERING, epoch))
        return conn.execute("UPDATE session_admissions SET status='unknown' WHERE status='started' AND owner_epoch!=?", (epoch,)).rowcount
    return db._execute_write(write)


def mutation_receipt_key(principal_id, session_id, request_id, *, operation, payload, expected_revision,
                         expected_generation=None):
    """``(state_meta key, digest)`` of a mutation request's durable exact-retry receipt."""
    key = 'gateway.mutation.v1.' + admission_fingerprint(
        canonical_target=session_id, payload={'principal': principal_id, 'request': request_id})
    fingerprint = {'operation': operation, 'payload': payload, 'expected_revision': expected_revision}
    if expected_generation is not None:
        fingerprint['expected_generation'] = expected_generation
    return key, admission_fingerprint(canonical_target=session_id, payload=fingerprint)


def replay_mutation_receipt(conn, key, digest):
    """The stored result of an exact retry, ``None`` when unseen; a different request refuses."""
    old = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    if old is None:
        return None
    receipt = json.loads(old[0])
    if receipt['digest'] != digest:
        raise RuntimeStoreError('admission_conflict')
    return receipt['result']


def mutate_runtime_session(db, *, epoch: int, principal_id: str, session_id: str,
                           request_id: str, expected_revision: int,
                           operation: str, payload: dict, expected_generation: int | None = None,
                           _live_guard=None, _authorize_write=None, _prepare_only=False, _prepared=None) -> dict:
    """Commit a closed metadata edit and its retry receipt in the same transaction.

    Caller authorizes the principal and resolves the canonical session. These
    metadata edits do not stop execution. Existing direct writers must migrate
    before this seam can provide universal revision fencing.
    """
    for value in (principal_id, session_id, request_id):
        _text(value)
    if type(expected_revision) is not int or expected_revision < 0:
        raise RuntimeStoreError('invalid_params')
    from hermes_state_mutations import validate_action, apply_action
    validate_action(operation, payload)
    if expected_generation is not None and (type(expected_generation) is not int or expected_generation < 0):
        raise RuntimeStoreError('invalid_params')
    if operation in {'delete', 'rewind', 'reset', 'branch', 'model', 'compress'} and expected_generation is None:
        raise RuntimeStoreError('invalid_params')
    # Snapshot caller data before waiting for the writer lock.
    payload = json.loads(_json(payload))
    key, digest = mutation_receipt_key(principal_id, session_id, request_id, operation=operation, payload=payload,
                                       expected_revision=expected_revision, expected_generation=expected_generation)

    def write(conn) -> tuple[dict, Optional[SessionGoneEvent]]:
        """The receipt result plus the on_session_delete event this commit owes (None on replay)."""
        _epoch(conn, epoch)
        replay = replay_mutation_receipt(conn, key, digest)
        if replay is not None:
            return replay, None
        session = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
        if session is None and operation != 'import':
            raise RuntimeStoreError('not_found')
        revision = session['runtime_revision'] if session else 0
        if revision != expected_revision:
            raise RuntimeStoreError('revision_conflict')
        if expected_generation is not None and (session is None or session['runtime_generation'] != expected_generation):
            raise RuntimeStoreError('stale_generation')
        if _live_guard is not None:
            from hermes_state_mutation_guards import delete_targets
            targets = delete_targets(conn, session_id) if operation == 'delete' else [session_id]
            _live_guard(targets)
        if _authorize_write is not None:
            _authorize_write(conn)
        if _prepare_only:
            from hermes_state_mutation_prepared import local_snapshot
            snapshot = local_snapshot(db, conn, session_id)
            if operation == 'branch':
                # The branch commit's own idle rule (queued input too), before the owner starts servers.
                from hermes_state_mutation_guards import require_idle
                require_idle(db, conn, list({session_id, snapshot['target']}))
            return {'snapshot': snapshot}, None
        affected, projection = apply_action(db, conn, session_id, operation, payload, prepared=_prepared)
        event = None
        if operation == 'delete':
            event = SessionGoneEvent(tuple(projection.get('deleted_ids') or (session_id,)), 'delete')
        elif payload.get('archived') is True and session is not None and not session['archived']:
            event = SessionGoneEvent(tuple(affected) or (session_id,), 'archive')
        conn.executemany('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?',
                         [(target,) for target in affected])
        updated = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
        result = {'session_id': session_id, 'revision': updated['runtime_revision'] if updated else expected_revision + 1,
                  'execution_generation': updated['runtime_generation'] if updated else expected_generation,
                  'operation': operation, **projection}
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (key, _json({'digest': digest, 'result': result})))
        return result, event
    result, event = db._execute_write(write)
    if event is not None:
        from hermes_state_common import emit_session_deleted
        emit_session_deleted(db, event.session_ids, reason=event.reason, primary=session_id)
    return result


_IMPORT_KEY = 'gateway.prompt_admissions_import.v1'


def _canonical_chain(conn, session_id):
    # Same selector as SessionDB.get_compression_chain, on OUR transaction connection.
    from hermes_state_compression import _CHAIN_STEP_SQL
    _session(conn, session_id)
    chain = [session_id]
    for _ in range(100):
        child = conn.execute(_CHAIN_STEP_SQL, (chain[-1],)).fetchone()
        if child is None:
            return chain
        if child['id'] in chain:
            raise RuntimeStoreError('admission_conflict')
        chain.append(child['id'])
    raise RuntimeStoreError('admission_conflict')


def _import_legacy_row(conn, row, epoch, principal_id):
    for field in ('admission_id', 'target_session_id', 'root'):
        _text(row[field])
    lineage = json.loads(row['lineage'])
    if not isinstance(lineage, list) or row['target_session_id'] not in lineage:
        raise RuntimeStoreError('invalid_params')
    payload = json.loads(row['payload'])
    encoded = _json(payload)
    intent = payload.get('intent', 'queue')
    intent = 'redirect' if intent == 'interrupt' else intent
    if intent not in ('queue', 'steer', 'redirect') or row['status'] not in ('queued', 'started', 'unknown', 'terminal'):
        raise RuntimeStoreError('invalid_params')
    chain = _canonical_chain(conn, row['target_session_id'])
    target = chain[-1]
    status = 'unknown' if row['status'] == 'started' else row['status']
    generation = row['generation']
    if generation is None and status == 'unknown':
        generation = _session(conn, target)['runtime_generation'] + 1
    if generation is not None and (type(generation) is not int or generation < 0):
        raise RuntimeStoreError('invalid_params')
    digest = admission_fingerprint(canonical_target=target, payload={'input': payload, 'intent': intent})
    conn.execute("""INSERT INTO session_admissions(admission_id,request_id,principal_id,
        target_session_id,lineage_json,payload_json,payload_digest,intent,status,outcome,owner_epoch,generation)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (row['admission_id'], row['admission_id'], principal_id,
        target, json.dumps(chain), encoded, digest, intent, status, row['outcome'], epoch, generation))
    if generation is not None:
        conn.execute('UPDATE sessions SET runtime_generation=MAX(runtime_generation,?) WHERE id=?', (generation, target))


def import_legacy_session_admissions(db, *, epoch: int, source_path, principal_id: str,
                                     writers_drained: bool) -> int:
    """Import a frozen legacy snapshot; original file remains rollback evidence.

    The caller proves the old writers are drained before invoking this function.
    Fingerprinting uses SQL snapshot data, never a raw open/close on a live inode.
    """
    from contextlib import closing
    from pathlib import Path
    import hashlib
    import sqlite3
    from hermes_cli.sqlite_safe_read import connect_tracked
    if writers_drained is not True:
        raise RuntimeStoreError('invalid_params')
    _text(principal_id)
    source_path = Path(source_path).resolve(strict=True)
    with closing(connect_tracked(source_path.as_uri() + '?mode=ro', uri=True)) as source:
        source.row_factory = sqlite3.Row
        source.execute('PRAGMA query_only=ON')
        source.execute('BEGIN')
        rows = [dict(r) for r in source.execute('SELECT * FROM admissions ORDER BY seq')]
        executions = []
        if source.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='executions'").fetchone():
            executions = [dict(r) for r in source.execute('SELECT * FROM executions ORDER BY root')]
        fingerprint = hashlib.sha256(_json({'rows': rows, 'executions': executions}).encode('utf-8')).hexdigest()
        marker = _json({'source': str(source_path), 'fingerprint': fingerprint, 'imported_id_count': len(rows), 'principal_id': principal_id})
        def write(conn):
            _epoch(conn, epoch)
            old = conn.execute('SELECT value FROM state_meta WHERE key=?', (_IMPORT_KEY,)).fetchone()
            if old is not None:
                if old[0] != marker:
                    raise RuntimeStoreError('admission_conflict')
                return 0
            for execution in executions:
                generation = execution['generation']
                if type(generation) is not int or generation < 0:
                    raise RuntimeStoreError('invalid_params')
                target = _canonical_chain(conn, _text(execution['root']))[-1]
                conn.execute('UPDATE sessions SET runtime_generation=MAX(runtime_generation,?) WHERE id=?', (generation, target))
            for row in rows:
                _import_legacy_row(conn, row, epoch, principal_id)
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (_IMPORT_KEY, marker))
            return len(rows)
        return db._execute_write(write)


def resolve_unknown_session_input(db, *, epoch: int, admission_id: str, generation: int,
                                  _terminal_write=None, _captured=None) -> dict:
    """Explicit operator acknowledgement; resolves uncertainty, never requeues it.

    ``_terminal_write(conn, row)`` is the owner's same-transaction effect (the discarded turn's
    transcript boundary): it must not commit, may run again on SQLite retry, and raising rolls
    the whole resolution back, so the FIFO can never advance past an unclosed lost turn.
    ``_captured=(outcome, stored_result)`` is the exact result this owner captured for the turn
    before its settlement write failed; it commits with the resolution (only for an ``unknown``
    row stamped by this same owner epoch), so the finished answer is not thrown away."""
    from hermes_state_terminal import RESULT_PREFIX
    def write(conn):
        _epoch(conn, epoch)
        row = _admission(conn, admission_id)
        if (type(generation) is int and row['generation'] == generation and row['status'] == 'terminal'
                and (row['outcome'] == 'interrupted' or conn.execute(
                    'SELECT 1 FROM state_meta WHERE key=?', (RESULT_PREFIX + admission_id,)).fetchone())):
            return _row(row)
        if type(generation) is not int or row['status'] != 'unknown' or row['generation'] != generation:
            raise RuntimeStoreError('stale_generation')
        outcome = 'interrupted'
        if _captured is not None and row['owner_epoch'] == epoch:
            outcome, stored = _captured
            if outcome not in ('completed', 'interrupted', 'failed'):
                raise RuntimeStoreError('invalid_params')
            from hermes_state_terminal import compact_result, encode_result
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                         'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                         (RESULT_PREFIX + admission_id, _json(encode_result(compact_result(
                             stored, user_message=json.loads(row['payload_json']).get('text'))))))
        if _terminal_write is not None:
            _terminal_write(conn, row)
        _retire_admission_workers(conn, row, row['owner_epoch'])
        conn.execute("UPDATE session_admissions SET status='terminal',outcome=? WHERE admission_id=?",
                     (outcome, admission_id))
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (row['target_session_id'],))
        return _row(_admission(conn, admission_id))
    return db._execute_write(write)


def _worker_public(row):
    result = dict(row)
    result.pop('adoption_digest')
    return result


def _worker_assignment(conn, execution_id, session_id, generation):
    row = conn.execute('SELECT * FROM worker_executions WHERE execution_id=?', (execution_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError('not_found')
    if row['session_id'] != session_id:
        raise RuntimeStoreError('permission_denied')
    if (type(generation) is not int or row['generation'] != generation
            or _session(conn, session_id)['runtime_generation'] != generation):
        raise RuntimeStoreError('stale_generation')
    return row


def _linked_worker_admission(conn, session_id, generation, owner_epoch, statuses):
    placeholders = ','.join('?' for _ in statuses)
    rows = conn.execute(f"""SELECT * FROM session_admissions
        WHERE generation=? AND owner_epoch=? AND status IN ({placeholders})
          AND (target_session_id=? OR EXISTS (
              SELECT 1 FROM json_each(session_admissions.lineage_json) WHERE value=?
          )) ORDER BY seq""",
        (generation, owner_epoch, *statuses, session_id, session_id)).fetchall()
    if len(rows) > 1:
        raise RuntimeStoreError('admission_conflict')
    return rows[0] if rows else None


def _secret_digest(secret):
    import hashlib
    _text(secret)
    return hashlib.sha256(secret.encode('utf-8')).hexdigest()


def register_worker_execution(db, *, epoch: int, execution_id: str, session_id: str,
                              generation: int, kind: str, adoption_secret: str, require_idle: bool = False) -> dict:
    for value in (execution_id, session_id):
        _text(value)
    if kind not in ('cron', 'child', 'compute', 'kanban'):
        raise RuntimeStoreError('invalid_params')
    digest = _secret_digest(adoption_secret)
    def write(conn):
        _epoch(conn, epoch)
        session = _session(conn, session_id)
        if type(generation) is not int or session['runtime_generation'] != generation:
            raise RuntimeStoreError('stale_generation')
        old = conn.execute('SELECT * FROM worker_executions WHERE execution_id=?', (execution_id,)).fetchone()
        if old is not None:
            if (old['session_id'], old['generation'], old['kind'], old['owner_epoch'], old['adoption_digest']) != (session_id, generation, kind, epoch, digest):
                raise RuntimeStoreError('admission_conflict')
            return _worker_public(old)
        from hermes_state_runtime_workers import runtime_lineage, worker_states
        targets = runtime_lineage(conn, session_id)
        if require_idle and any(conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status IN ('queued','started','unknown')", (sid,)).fetchone() for sid in targets):
            raise RuntimeStoreError('stale_generation')
        if worker_states(conn, session_id):
            raise RuntimeStoreError('stale_generation')
        if any(conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status='unknown'", (sid,)).fetchone() for sid in targets):
            raise RuntimeStoreError('unknown_execution')
        conn.execute("""INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,status,adoption_digest)
            VALUES(?,?,?,?,?,'registered',?)""", (execution_id, session_id, kind, epoch, generation, digest))
        return _worker_public(_worker_assignment(conn, execution_id, session_id, generation))
    return db._execute_write(write)


def adopt_worker_execution(db, *, epoch: int, execution_id: str, session_id: str,
                           generation: int, adoption_secret: str) -> dict:
    """Authority first verifies the original producer claim and live worker proof."""
    import hmac
    digest = _secret_digest(adoption_secret)
    def write(conn):
        _epoch(conn, epoch)
        row = _worker_assignment(conn, execution_id, session_id, generation)
        if row['status'] == 'terminal':
            raise RuntimeStoreError('stale_generation')
        if not hmac.compare_digest(row['adoption_digest'], digest):
            raise RuntimeStoreError('permission_denied')
        # An admission-reserved worker reports to the reserving owner over that owner's pipes and
        # dies with them, so only its same-epoch handshake adopts it. A late handshake reaching a
        # restarted owner would re-arm the recovered `unknown` admission as `started` with nobody
        # left to settle it (and Discard refuses `started`); refuse it so recovery's verdict stands.
        if execution_id.startswith('admission-worker:') and row['owner_epoch'] != epoch:
            raise RuntimeStoreError('stale_epoch')
        linked = _linked_worker_admission(
            conn, session_id, generation, row['owner_epoch'], ('started', 'unknown'))
        if linked is not None:
            changed = conn.execute("""UPDATE session_admissions SET owner_epoch=?,status='started'
                WHERE admission_id=? AND generation=? AND owner_epoch=?
                AND status IN ('started','unknown')""",
                (epoch, linked['admission_id'], generation, row['owner_epoch']))
            if changed.rowcount != 1:
                raise RuntimeStoreError('stale_generation')
        changed = conn.execute(
            "UPDATE worker_executions SET owner_epoch=?,status='running' WHERE execution_id=?",
            (epoch, execution_id))
        if changed.rowcount != 1:
            raise RuntimeStoreError('stale_generation')
        return _worker_public(_worker_assignment(conn, execution_id, session_id, generation))
    return db._execute_write(write)


_MESSAGE_FIELDS = frozenset({
    'role', 'content', 'tool_name', 'tool_calls', 'tool_call_id', 'token_count',
    'finish_reason', 'reasoning', 'reasoning_content', 'reasoning_details',
    'codex_reasoning_items', 'codex_message_items', 'platform_message_id', 'message_id',
    'observed', 'effect_disposition', '_compressed_summary', 'timestamp', 'api_content',
    'display_kind', 'display_metadata', '_row_id', '_canonical_content', DB_ROW_SNAPSHOT, CANONICAL_ROW, DATA_USE,
    # Main's durable message identity (message_uid / merge witness / tool-call uids) and the alternation
    # repair's row counts ride every persisted dict; the owner's insert and coverage read them.
}) | PERSISTENCE_ONLY_MESSAGE_FIELDS | {ABSORBED_ROW_IDS}
# Compaction handoffs (archive/publish) also carry the summary's user-turn flag. This is the closed
# boundary the owner validates AND the projection the worker facade applies to live dicts first.
HANDOFF_MESSAGE_FIELDS = _MESSAGE_FIELDS | {'_compressed_summary_has_user_turn'}
# Row state the owner's insert and transcript repair stamp on each message (main mutates the caller's
# dict in place; a worker gets it back as an annotation). None = absent, so a stale adoption is cleared.
# The owner-minted identity rides too: a worker dict without it re-mints a new uid on every later
# archive/rotation copy, so the logical message changes identity across generations.
_ROW_ANNOTATION_KEYS = ('_row_id', 'timestamp', DB_ROW_SNAPSHOT, CANONICAL_ROW,
                        MESSAGE_UID, TOOL_CALL_UIDS, TOOL_CALL_UID)


def row_annotations(messages):
    return [{key: msg.get(key) for key in _ROW_ANNOTATION_KEYS} for msg in messages]


def _worker_append(db, conn, session_id, payload):
    if set(payload) - {'messages', 'turn_lease_holder'} or 'messages' not in payload:
        raise RuntimeStoreError('invalid_params')
    messages = payload['messages']
    if not isinstance(messages, list) or len(messages) > 1000:
        raise RuntimeStoreError('invalid_params')
    for msg in messages:
        if (not isinstance(msg, dict) or set(msg) - _MESSAGE_FIELDS
                or msg.get('role') not in ('user', 'assistant', 'system', 'tool')):
            raise RuntimeStoreError('invalid_params')
        # Existing row annotations can adopt only rows in THIS transcript.
        if '_row_id' in msg and not conn.execute(
                'SELECT 1 FROM messages WHERE id=? AND session_id=?',
                (msg['_row_id'], session_id)).fetchone():
            raise RuntimeStoreError('permission_denied')
    holder = payload.get('turn_lease_holder')
    if holder is not None:
        _text(holder)
    count = db._append_messages_in_transaction(conn, session_id, messages, turn_lease_holder=holder)
    return {'count': count, 'annotations': row_annotations(messages)}


def _worker_turn(db, conn, session_id, payload, operation):
    import time
    from hermes_state_compression import _claim_lease_row
    from hermes_state import _compression_lock_holder_process_is_dead
    allowed = {'holder'} if operation == 'turn.release' else {'holder', 'ttl_seconds'}
    if set(payload) != allowed:
        raise RuntimeStoreError('invalid_params')
    holder = _text(payload['holder'])
    ttl = payload.get('ttl_seconds', 300)
    if type(ttl) not in (float, int) or not 0.1 <= ttl <= 3600:
        raise RuntimeStoreError('invalid_params')
    key = db._session_turn_lease_key_on_conn(conn, session_id)
    now = time.time()
    if operation == 'turn.acquire':
        value = _claim_lease_row(conn, 'session_turn_leases', 'conversation_id', key,
            holder, now, now + ttl,
            lambda h, e: float(e) <= now or _compression_lock_holder_process_is_dead(h))[0]
    elif operation == 'turn.renew':
        value = conn.execute('UPDATE session_turn_leases SET expires_at=? WHERE conversation_id=? AND holder=?',
                             (now + ttl, key, holder)).rowcount > 0
    else:
        conn.execute('DELETE FROM session_turn_leases WHERE conversation_id=? AND holder=?', (key, holder))
        value = None
    return {'value': value}


def _worker_usage(db, conn, session_id, payload, *, auxiliary=False):
    from hermes_state_usage import _MODEL_USAGE_FIELDS, _TOKEN_COUNTERS
    source = None
    if not auxiliary:
        # ``source`` is the worker platform (turn_usage._agent_session_source). A Desktop
        # managed worker passes ``gui``; created_surface maps that to ``desktop``. The row
        # is usually stamped at admission. A compression child can still be NULL (or an
        # auxiliary-usage ``unknown``), and this worker never create_session's it, so fill
        # only an unset column, as the creator upsert does.
        raw = payload.get('source')
        source = raw if isinstance(raw, str) else None
        payload = {k: v for k, v in payload.items() if k != 'source'}
    allowed = (_MODEL_USAGE_FIELDS - {'billing_mode', 'actual_cost_usd', 'cost_status', 'cost_source'} | {'task'}) if auxiliary else (_MODEL_USAGE_FIELDS | {'pricing_version', 'absolute', 'task'})
    if set(payload) - allowed:
        raise RuntimeStoreError('invalid_params')
    for key, value in payload.items():
        if key in (*_TOKEN_COUNTERS, 'api_call_count'):
            valid = type(value) is int and 0 <= value <= 2**53
        elif key in ('estimated_cost_usd', 'actual_cost_usd'):
            valid = value is None or (type(value) in (int, float) and 0 <= value <= 1e12)
        elif key == 'absolute':
            valid = type(value) is bool
        else:
            valid = value is None or (isinstance(value, str) and len(value) <= 4096)
        if not valid:
            raise RuntimeStoreError('invalid_params')
    if auxiliary:
        if not payload.get('task'):
            raise RuntimeStoreError('invalid_params')
        db._record_model_usage(conn, session_id, **({'api_call_count': 1} | payload))
    else:
        db._update_token_counts_in_transaction(conn, session_id, **payload)
        # The managed worker's platform is the policy label (gui), not the in-process
        # agent surface (desktop). Stamp the same created_surface the insert uses.
        from gateway.session_policy import created_surface
        from hermes_state_compression import _UNSET_CREATED_SQL, stamp_compression_ancestors
        surfaced = created_surface(source)
        if surfaced:
            conn.execute(
                "UPDATE sessions SET created_source=? WHERE id=? AND "
                + _UNSET_CREATED_SQL.format(col="created_source"),
                (surfaced, session_id))
            stamp_compression_ancestors(conn, session_id)
    return {'value': None}


def _worker_finish(db, conn, session_id, payload):
    if payload:
        raise RuntimeStoreError('invalid_params')
    return {'status': 'terminal'}


def mutate_worker_execution(db, *, epoch, execution_id, session_id, generation,
                            sequence, operation, payload):
    """One closed durable mutation and receipt; never call a self-committing API here."""
    from hermes_state_worker_context import (worker_context, worker_prompt, worker_row_role, worker_sidecars,
                                             worker_tail_role, worker_tool_names)
    from hermes_state_worker_compression import WORKER_COMPRESSION_HANDLERS
    from hermes_state_worker_lifecycle import WORKER_LIFECYCLE_HANDLERS
    handlers = {
        **WORKER_LIFECYCLE_HANDLERS,
        **WORKER_COMPRESSION_HANDLERS,
        'session.context': worker_context,
        'session.prompt': worker_prompt,
        'session.sidecars': worker_sidecars,
        'session.tools': worker_tool_names,
        'session.tail_role': worker_tail_role,
        'session.row_role': worker_row_role,
        'transcript.append': _worker_append,
        'execution.finish': _worker_finish,
        'usage.main': _worker_usage,
        'usage.auxiliary': lambda db, conn, sid, p: _worker_usage(db, conn, sid, p, auxiliary=True),
        **{name: (lambda db, conn, sid, p, op=name: _worker_turn(db, conn, sid, p, op))
           for name in ('turn.acquire', 'turn.renew', 'turn.release')},
    }
    return apply_worker_receipt(db, epoch=epoch, execution_id=execution_id, session_id=session_id,
                                generation=generation, sequence=sequence, operation=operation,
                                payload=payload, handlers=handlers)


def apply_worker_receipt(db, *, epoch, execution_id, session_id, generation, sequence,
                         operation, payload, handlers):
    """The one worker receipt transaction every ``worker.persist`` family shares.

    An exact (execution, sequence, digest) retry returns the stored result; otherwise the
    epoch, assignment, live status and next sequence are fenced, ``handlers[operation]`` runs on
    the transaction connection (it must not commit), and its result is stored as the receipt.
    ``execution.finish`` is the only operation that makes the execution terminal.
    """
    from hermes_state_worker_compression import worker_receipt_assignment
    if type(sequence) is not int or sequence < 1 or not isinstance(operation, str) or operation not in handlers:
        raise RuntimeStoreError('invalid_params')
    encoded = _json(payload)
    if len(encoded.encode('utf-8', errors='surrogatepass')) > 4 * 1024 * 1024:
        raise RuntimeStoreError('invalid_params')
    digest = admission_fingerprint(canonical_target=session_id,
                                  payload={'operation': operation, 'payload': json.loads(encoded)})
    def write(conn):
        _epoch(conn, epoch)
        row = worker_receipt_assignment(conn, execution_id, session_id, generation, sequence, digest)
        if row['owner_epoch'] != epoch:
            raise RuntimeStoreError('stale_epoch')
        old = conn.execute('SELECT * FROM worker_receipts WHERE execution_id=? AND sequence=?',
                           (execution_id, sequence)).fetchone()
        if old is not None:
            if old['payload_digest'] != digest:
                raise RuntimeStoreError('admission_conflict')
            from hermes_state_runtime_workers import retired_receipt
            if retired_receipt(old['result_json']):
                raise RuntimeStoreError('stale_generation')  # a terminal execution's read copy
            return json.loads(old['result_json'])
        if row['status'] not in ('registered', 'running'):
            raise RuntimeStoreError('stale_generation')
        if sequence != row['last_sequence'] + 1:
            raise RuntimeStoreError('invalid_params')
        # Each SQLite retry gets fresh rows; rolled-back annotations must not escape.
        changes = conn.total_changes
        result = handlers[operation](db, conn, session_id, json.loads(encoded))
        # The CAS fence moves only when the handler changed durable state (or the execution
        # closes): a read-only receipt (context, history, lineage, lifecycle) must not make a
        # client's mutation built on the pre-read revision fail revision_conflict.
        changed = conn.total_changes != changes or operation == 'execution.finish'
        conn.execute('INSERT INTO worker_receipts(execution_id,sequence,payload_digest,result_json) VALUES(?,?,?,?)',
                     (execution_id, sequence, digest, json.dumps(result, ensure_ascii=True, allow_nan=False)))
        conn.execute("UPDATE worker_executions SET last_sequence=?,status=? WHERE execution_id=?",
                     (sequence, 'terminal' if operation == 'execution.finish' else 'running', execution_id))
        if changed:
            conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (session_id,))
        if operation == 'execution.finish':
            from hermes_state_runtime_workers import compact_terminal_receipts
            compact_terminal_receipts(conn, execution_id)
        return result
    return db._execute_write(write, patience_s=db._TRANSCRIPT_WRITE_PATIENCE_S)

# Preservation-drain source API. This remains in the state authority so the caller never reads
# SessionDB tables directly. It returns only bounded control/recovery metadata, not payload/result bodies.
from contextlib import contextmanager as _snapshot_contextmanager
import math as _snapshot_math
import sqlite3 as _snapshot_sqlite3
import time as _snapshot_time


_RUNTIME_SNAPSHOT_SCHEMA = 'runtime-admission-snapshot-v1'
_RUNTIME_SNAPSHOT_MAX_RECORDS = 2048
_RUNTIME_SNAPSHOT_MAX_BYTES = 16 * 1024 * 1024
_RUNTIME_SNAPSHOT_PAGE_SIZE = 64
_RUNTIME_SNAPSHOT_MAX_TEXT = 4096
_RUNTIME_SNAPSHOT_MAX_TEXT_BYTES = 16 * 1024
_RUNTIME_SNAPSHOT_MAX_LINEAGE = 256
_RUNTIME_SNAPSHOT_MAX_RECEIPT_BYTES = 4 * 1024 * 1024
_RUNTIME_ADMISSION_STATUSES = ('queued', 'started', 'unknown')
_RUNTIME_WORKER_STATUSES = ('registered', 'running', 'unknown')
_RUNTIME_SNAPSHOT_TEXT_SQL_LIMIT = _RUNTIME_SNAPSHOT_MAX_TEXT + 1


def _snapshot_invalid(field):
    raise RuntimeStoreError('snapshot_schema', field=field)


def _snapshot_bound(value, field, ceiling):
    if type(value) is not int or not 1 <= value <= ceiling:
        raise RuntimeStoreError('invalid_params', field=field)
    return value


def _snapshot_deadline(value):
    if type(value) not in (int, float) or not _snapshot_math.isfinite(value):
        raise RuntimeStoreError('invalid_params', field='deadline')
    deadline = float(value)
    if deadline <= _snapshot_time.monotonic():
        raise RuntimeStoreError('snapshot_deadline', field='deadline')
    return deadline


def _snapshot_check_deadline(deadline):
    if _snapshot_time.monotonic() >= deadline:
        raise RuntimeStoreError('snapshot_deadline', field='deadline')


def _snapshot_text(value, field, *, allow_none=False):
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value or len(value) > _RUNTIME_SNAPSHOT_MAX_TEXT:
        _snapshot_invalid(field)
    return value


def _snapshot_number(value, field, *, minimum=0, allow_none=False):
    if value is None and allow_none:
        return None
    if type(value) is not int or value < minimum:
        _snapshot_invalid(field)
    return value


def _snapshot_sql_error(exc, deadline):
    if _snapshot_time.monotonic() >= deadline:
        raise RuntimeStoreError('snapshot_deadline', field='deadline') from exc
    message = str(exc).lower()
    if any(marker in message for marker in (
        'no such table', 'no such column', 'malformed', 'not a database',
        'database disk image is malformed', 'datatype mismatch',
    )):
        raise RuntimeStoreError('snapshot_schema', detail='runtime schema read failed') from exc
    raise RuntimeStoreError('snapshot_unavailable', detail='runtime read did not complete') from exc


def _snapshot_generation_guard(db):
    checker = getattr(db, '_raise_if_db_replaced', None)
    if not callable(checker):
        raise RuntimeStoreError('snapshot_generation', detail='owner generation guard unavailable')
    try:
        checker()
    except RuntimeStoreError:
        raise
    except (OSError, RuntimeError, _snapshot_sqlite3.Error, ValueError) as exc:
        raise RuntimeStoreError('snapshot_generation', detail='runtime generation changed') from exc


def _snapshot_closed(db):
    return getattr(db, '_conn', None) is None or bool(getattr(db, '_read_conns_closed', False))


def _snapshot_require_runtime_tables(conn, deadline):
    names = ('runtime_epoch', 'session_admissions', 'worker_executions', 'worker_receipts')
    cursor, _ = _snapshot_execute(
        conn,
        '''SELECT 'main' AS schema_name, name, type FROM main.sqlite_master
           WHERE name COLLATE NOCASE IN (?, ?, ?, ?)
           UNION ALL
           SELECT 'temp' AS schema_name, name, type FROM temp.sqlite_master
           WHERE name COLLATE NOCASE IN (?, ?, ?, ?)
           LIMIT 9''',
        names + names,
        deadline,
    )
    found = set()
    for row in _snapshot_fetchmany(cursor, conn, 9, deadline):
        name = row['name'].lower()
        if row['schema_name'] != 'main' or row['type'] != 'table' or name in found:
            raise RuntimeStoreError(
                'snapshot_schema', field=name, detail='runtime census requires an unshadowed main table')
        found.add(name)
    for name in names:
        if name not in found:
            raise RuntimeStoreError('snapshot_schema', field=name, detail='runtime census table missing')


@_snapshot_contextmanager
def _snapshot_owner_read(db, deadline):
    """Use the existing SessionDB owner connection without reopen or an unbounded lock wait."""
    _snapshot_check_deadline(deadline)
    if _snapshot_closed(db):
        raise RuntimeStoreError('snapshot_closed', detail='state owner is closed')
    lock = getattr(db, '_lock', None)
    if lock is None or not hasattr(lock, 'acquire'):
        raise RuntimeStoreError('snapshot_unavailable', detail='state owner lock unavailable')
    remaining = deadline - _snapshot_time.monotonic()
    if remaining <= 0 or not lock.acquire(timeout=remaining):
        raise RuntimeStoreError('snapshot_deadline', field='deadline')
    conn = None
    previous_busy_timeout = None
    transaction = False
    try:
        if _snapshot_closed(db):
            raise RuntimeStoreError('snapshot_closed', detail='state owner closed while acquiring')
        _snapshot_check_deadline(deadline)
        _snapshot_generation_guard(db)
        _snapshot_check_deadline(deadline)
        conn = getattr(db, '_conn', None)
        try:
            owner_busy = conn is None or conn.in_transaction
        except _snapshot_sqlite3.ProgrammingError as exc:
            raise RuntimeStoreError('snapshot_closed', detail='state owner connection is closed') from exc
        if owner_busy:
            raise RuntimeStoreError('snapshot_unavailable', detail='state owner is busy')
        try:
            previous_busy_timeout = conn.execute('PRAGMA busy_timeout').fetchone()[0]
            remaining_ms = max(1, int((deadline - _snapshot_time.monotonic()) * 1000))
            conn.execute(f'PRAGMA busy_timeout={remaining_ms}')
            conn.execute('BEGIN')
            transaction = True
        except _snapshot_sqlite3.Error as exc:
            _snapshot_sql_error(exc, deadline)
        _snapshot_require_runtime_tables(conn, deadline)
        yield conn
        _snapshot_check_deadline(deadline)
        try:
            conn.commit()
        except _snapshot_sqlite3.Error as exc:
            _snapshot_sql_error(exc, deadline)
        transaction = False
        _snapshot_generation_guard(db)
    except BaseException:
        if transaction and conn is not None:
            try:
                conn.rollback()
            except _snapshot_sqlite3.Error:
                pass
        raise
    finally:
        restoration_error = None
        if conn is not None and previous_busy_timeout is not None:
            try:
                conn.execute(f'PRAGMA busy_timeout={int(previous_busy_timeout)}')
            except _snapshot_sqlite3.Error:
                if not _snapshot_closed(db):
                    restoration_error = RuntimeStoreError(
                        'snapshot_unavailable', detail='could not restore owner timeout')
        try:
            lock.release()
        finally:
            if restoration_error is not None:
                raise restoration_error


def _snapshot_capped_text(row, field, *, allow_none=False, error_field=None):
    error_field = error_field or field
    value = row[field]
    actual_bytes = row[f'{field}_bytes']
    selected_limit = row['_snapshot_text_limit'] if '_snapshot_text_limit' in row.keys() else None
    if selected_limit is not None:
        if type(selected_limit) is not int or selected_limit < 0:
            _snapshot_invalid('snapshot_text_limit')
        if actual_bytes is not None and actual_bytes > selected_limit:
            raise RuntimeStoreError('snapshot_limit', field=error_field)
    if actual_bytes is None:
        if value is None and allow_none:
            return None
        _snapshot_invalid(error_field)
    if type(actual_bytes) is not int or actual_bytes < 0 or actual_bytes > _RUNTIME_SNAPSHOT_MAX_TEXT_BYTES:
        raise RuntimeStoreError('snapshot_limit', field=error_field)
    if value is None:
        if allow_none:
            return None
        _snapshot_invalid(error_field)
    if not isinstance(value, str) or not value or len(value) > _RUNTIME_SNAPSHOT_MAX_TEXT:
        _snapshot_invalid(error_field)
    return value


def _snapshot_encode(value, field):
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode('utf-8', errors='surrogatepass')
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeStoreError('snapshot_schema', field=field) from exc
    return encoded


def _snapshot_charge(budget, value, field, *, extra_bytes=0):
    encoded = _snapshot_encode(value, field)
    if type(extra_bytes) is not int or extra_bytes < 0:
        _snapshot_invalid(field)
    if (budget['records'] >= budget['max_records']
            or budget['bytes'] + len(encoded) + extra_bytes > budget['max_bytes']):
        raise RuntimeStoreError('snapshot_limit', field='budget')
    budget['records'] += 1
    budget['bytes'] += len(encoded) + extra_bytes


def _snapshot_execute(conn, sql, params, deadline):
    _snapshot_check_deadline(deadline)
    interrupted = False

    def progress():
        nonlocal interrupted
        if _snapshot_time.monotonic() >= deadline:
            interrupted = True
            return 1
        return 0

    conn.set_progress_handler(progress, 1000)
    try:
        return conn.execute(sql, params), interrupted
    except _snapshot_sqlite3.Error as exc:
        if interrupted or _snapshot_time.monotonic() >= deadline:
            raise RuntimeStoreError('snapshot_deadline', field='deadline') from exc
        _snapshot_sql_error(exc, deadline)
    finally:
        conn.set_progress_handler(None, 0)


def _snapshot_fetchmany(cursor, conn, size, deadline):
    _snapshot_check_deadline(deadline)
    interrupted = False

    def progress():
        nonlocal interrupted
        if _snapshot_time.monotonic() >= deadline:
            interrupted = True
            return 1
        return 0

    conn.set_progress_handler(progress, 1000)
    try:
        return cursor.fetchmany(size)
    except _snapshot_sqlite3.Error as exc:
        if interrupted or _snapshot_time.monotonic() >= deadline:
            raise RuntimeStoreError('snapshot_deadline', field='deadline') from exc
        _snapshot_sql_error(exc, deadline)
    finally:
        conn.set_progress_handler(None, 0)


def _snapshot_budget_text_limit(budget, text_fields):
    if type(text_fields) is not int or text_fields < 1:
        _snapshot_invalid('snapshot_text_fields')
    remaining = budget['max_bytes'] - budget['bytes']
    if remaining <= 0:
        return 0
    return min(_RUNTIME_SNAPSHOT_TEXT_SQL_LIMIT, remaining // text_fields)


def _snapshot_fetch_records(conn, sql, params, *, deadline, budget, convert,
                            text_limit, text_fields):
    remaining = max(0, budget['max_records'] - budget['records'])
    cursor, _ = _snapshot_execute(conn, sql, (*params, remaining + 1), deadline)
    records = []
    while True:
        _snapshot_check_deadline(deadline)
        row_text_bytes = max(1, text_limit) * text_fields
        available_bytes = max(1, budget['max_bytes'] - budget['bytes'])
        budget_page = max(1, available_bytes // row_text_bytes)
        page_size = min(_RUNTIME_SNAPSHOT_PAGE_SIZE, budget_page, remaining + 1 - len(records))
        if page_size <= 0:
            raise RuntimeStoreError('snapshot_limit', field='records')
        page = _snapshot_fetchmany(cursor, conn, page_size, deadline)
        if not page:
            break
        for row in page:
            if budget['records'] >= budget['max_records']:
                raise RuntimeStoreError('snapshot_limit', field='records')
            record, extra_bytes = convert(row)
            _snapshot_charge(budget, record, 'snapshot.record', extra_bytes=extra_bytes)
            records.append(record)
        if len(page) < page_size:
            break
    if len(records) > remaining:
        raise RuntimeStoreError('snapshot_limit', field='records')
    return records


def _snapshot_epoch_row(conn, deadline):
    text = _RUNTIME_SNAPSHOT_TEXT_SQL_LIMIT
    cursor, _ = _snapshot_execute(
        conn,
        f'''SELECT singleton, epoch,
            substr(instance_id,1,{text}) AS instance_id,
            length(CAST(instance_id AS BLOB)) AS instance_id_bytes,
            {text} AS _snapshot_text_limit
            FROM runtime_epoch WHERE singleton=1 LIMIT 2''',
        (),
        deadline,
    )
    rows = _snapshot_fetchmany(cursor, conn, 2, deadline)
    if len(rows) != 1:
        _snapshot_invalid('runtime_epoch')
    value = rows[0]
    if value['singleton'] != 1:
        _snapshot_invalid('runtime_epoch.singleton')
    epoch = _snapshot_number(value['epoch'], 'runtime_epoch.epoch', minimum=1)
    instance_id = _snapshot_capped_text(value, 'instance_id', error_field='runtime_epoch.instance_id')
    return {'epoch': epoch, 'instance_id': instance_id}


def _snapshot_admission_row(row):
    status = _snapshot_capped_text(row, 'status', error_field='admission.status')
    if status not in _RUNTIME_ADMISSION_STATUSES:
        _snapshot_invalid('admission.status')
    owner_epoch = _snapshot_number(row['owner_epoch'], 'admission.owner_epoch', minimum=1, allow_none=True)
    generation = _snapshot_number(row['generation'], 'admission.generation', allow_none=True)
    if status in ('started', 'unknown') and (owner_epoch is None or generation is None):
        _snapshot_invalid('admission.execution_fence')
    lineage_json = _snapshot_capped_text(row, 'lineage_json', error_field='admission.lineage_json')
    try:
        lineage = json.loads(lineage_json)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeStoreError('snapshot_schema', field='admission.lineage_json') from exc
    if (not isinstance(lineage, list) or not lineage or len(lineage) > _RUNTIME_SNAPSHOT_MAX_LINEAGE
            or any(not isinstance(item, str) or not item or len(item) > _RUNTIME_SNAPSHOT_MAX_TEXT for item in lineage)
            or len(set(lineage)) != len(lineage)):
        _snapshot_invalid('admission.lineage_json')
    target_session_id = _snapshot_capped_text(row, 'target_session_id', error_field='admission.target_session_id')
    if target_session_id not in lineage:
        _snapshot_invalid('admission.lineage.target_session_id')
    return {
        'seq': _snapshot_number(row['seq'], 'admission.seq', minimum=1),
        'admission_id': _snapshot_capped_text(row, 'admission_id', error_field='admission.admission_id'),
        'request_id': _snapshot_capped_text(row, 'request_id', error_field='admission.request_id'),
        'principal_id': _snapshot_capped_text(row, 'principal_id', error_field='admission.principal_id'),
        'target_session_id': target_session_id,
        'lineage_json': lineage_json,
        'lineage': lineage,
        'payload_digest': _snapshot_capped_text(row, 'payload_digest', error_field='admission.payload_digest'),
        'intent': _snapshot_capped_text(row, 'intent', error_field='admission.intent'),
        'status': status,
        'outcome': _snapshot_capped_text(row, 'outcome', allow_none=True, error_field='admission.outcome'),
        'owner_epoch': owner_epoch,
        'generation': generation,
    }, 0


def _snapshot_worker_row(row):
    status = _snapshot_capped_text(row, 'status', error_field='worker.status')
    if status not in _RUNTIME_WORKER_STATUSES:
        _snapshot_invalid('worker.status')
    kind = _snapshot_capped_text(row, 'kind', error_field='worker.kind')
    if kind not in ('cron', 'child', 'compute', 'kanban'):
        _snapshot_invalid('worker.kind')
    return {
        'execution_id': _snapshot_capped_text(row, 'execution_id', error_field='worker.execution_id'),
        'session_id': _snapshot_capped_text(row, 'session_id', error_field='worker.session_id'),
        'kind': kind,
        'owner_epoch': _snapshot_number(row['owner_epoch'], 'worker.owner_epoch', minimum=1),
        'generation': _snapshot_number(row['generation'], 'worker.generation'),
        'status': status,
        'last_sequence': _snapshot_number(row['last_sequence'], 'worker.last_sequence'),
    }, 0


def _snapshot_receipt_row(row, workers):
    execution_id = _snapshot_capped_text(row, 'execution_id', error_field='worker_receipt.execution_id')
    worker = workers.get(execution_id)
    if worker is None:
        _snapshot_invalid('worker_receipt.execution_id')
    result_bytes = _snapshot_number(row['result_bytes'], 'worker_receipt.result_bytes')
    if result_bytes > _RUNTIME_SNAPSHOT_MAX_RECEIPT_BYTES:
        _snapshot_invalid('worker_receipt.result_json')
    sequence = _snapshot_number(row['sequence'], 'worker_receipt.sequence', minimum=1)
    if sequence > worker['last_sequence']:
        _snapshot_invalid('worker_receipt.sequence.relation')
    return {
        'execution_id': execution_id,
        'sequence': sequence,
        'payload_digest': _snapshot_capped_text(row, 'payload_digest', error_field='worker_receipt.payload_digest'),
        'result_bytes': result_bytes,
    }, result_bytes


def _snapshot_validate_receipt_body(conn, record, deadline):
    cursor, _ = _snapshot_execute(
        conn,
        '''SELECT json_valid(result_json) AS result_valid,
            CASE WHEN json_valid(result_json) = 1 THEN json_type(result_json) END AS result_type
            FROM worker_receipts WHERE execution_id=? AND sequence=? LIMIT 2''',
        (record['execution_id'], record['sequence']),
        deadline,
    )
    rows = _snapshot_fetchmany(cursor, conn, 2, deadline)
    if len(rows) != 1 or rows[0]['result_valid'] != 1:
        _snapshot_invalid('worker_receipt.result_json')
    if rows[0]['result_type'] != 'object':
        _snapshot_invalid('worker_receipt.result_json.type')


def _snapshot_read_epoch(db, deadline):
    _snapshot_check_deadline(deadline)
    with _snapshot_owner_read(db, deadline) as conn:
        return _snapshot_epoch_row(conn, deadline)


def snapshot_runtime_admission_workers(
    db,
    *,
    epoch: int,
    deadline: float,
    max_records: int = 512,
    max_bytes: int = 4 * 1024 * 1024,
) -> dict:
    """Read one bounded, owner-consistent runtime admission/worker snapshot.

    The query is deliberately global and does not join sessions or any in-memory authority
    registry: queued, started, and unknown durable admissions remain visible even when an owning
    authority session has disappeared, and prior-epoch unknown rows are retained. The owner loop
    supplies one absolute monotonic deadline. Records are paged under one shared count/encoded-byte
    budget; receipt bodies are validated and charged by their producer-encoded byte length but are
    never returned. A caller still needs an outer profile owner/drain lease before using this census.
    """
    if type(epoch) is not int or epoch < 1:
        raise RuntimeStoreError('invalid_params', field='epoch')
    deadline = _snapshot_deadline(deadline)
    max_records = _snapshot_bound(max_records, 'max_records', _RUNTIME_SNAPSHOT_MAX_RECORDS)
    max_bytes = _snapshot_bound(max_bytes, 'max_bytes', _RUNTIME_SNAPSHOT_MAX_BYTES)
    budget = {'max_records': max_records, 'max_bytes': max_bytes, 'records': 0, 'bytes': 0}
    with _snapshot_owner_read(db, deadline) as conn:
        opened = _snapshot_epoch_row(conn, deadline)
        if opened['epoch'] != epoch:
            raise RuntimeStoreError('stale_epoch')
        admission_text = _snapshot_budget_text_limit(budget, 9)
        admission_sql = f'''SELECT seq,
            substr(admission_id,1,{admission_text}) AS admission_id,
            length(CAST(admission_id AS BLOB)) AS admission_id_bytes,
            substr(request_id,1,{admission_text}) AS request_id,
            length(CAST(request_id AS BLOB)) AS request_id_bytes,
            substr(principal_id,1,{admission_text}) AS principal_id,
            length(CAST(principal_id AS BLOB)) AS principal_id_bytes,
            substr(target_session_id,1,{admission_text}) AS target_session_id,
            length(CAST(target_session_id AS BLOB)) AS target_session_id_bytes,
            substr(lineage_json,1,{admission_text}) AS lineage_json,
            length(CAST(lineage_json AS BLOB)) AS lineage_json_bytes,
            substr(payload_digest,1,{admission_text}) AS payload_digest,
            length(CAST(payload_digest AS BLOB)) AS payload_digest_bytes,
            substr(intent,1,{admission_text}) AS intent,
            length(CAST(intent AS BLOB)) AS intent_bytes,
            substr(status,1,{admission_text}) AS status,
            length(CAST(status AS BLOB)) AS status_bytes,
            substr(outcome,1,{admission_text}) AS outcome,
            length(CAST(outcome AS BLOB)) AS outcome_bytes,
            owner_epoch, generation,
            {admission_text} AS _snapshot_text_limit
            FROM session_admissions
            WHERE status IS NULL OR status != 'terminal'
            ORDER BY seq LIMIT ?'''
        admission_records = _snapshot_fetch_records(
            conn, admission_sql, (), deadline=deadline, budget=budget,
            convert=_snapshot_admission_row, text_limit=admission_text, text_fields=9)
        worker_text = _snapshot_budget_text_limit(budget, 4)
        worker_sql = f'''SELECT
            substr(execution_id,1,{worker_text}) AS execution_id,
            length(CAST(execution_id AS BLOB)) AS execution_id_bytes,
            substr(session_id,1,{worker_text}) AS session_id,
            length(CAST(session_id AS BLOB)) AS session_id_bytes,
            substr(kind,1,{worker_text}) AS kind,
            length(CAST(kind AS BLOB)) AS kind_bytes,
            owner_epoch, generation,
            substr(status,1,{worker_text}) AS status,
            length(CAST(status AS BLOB)) AS status_bytes,
            last_sequence,
            {worker_text} AS _snapshot_text_limit
            FROM worker_executions
            WHERE status IS NULL OR status != 'terminal'
            ORDER BY execution_id LIMIT ?'''
        worker_records = _snapshot_fetch_records(
            conn, worker_sql, (), deadline=deadline, budget=budget,
            convert=_snapshot_worker_row, text_limit=worker_text, text_fields=4)
        workers_by_id = {record['execution_id']: record for record in worker_records}
        receipt_records = []
        worker_ids = list(workers_by_id)
        if worker_ids:
            receipt_text = _snapshot_budget_text_limit(budget, 2)
            placeholders = ','.join('?' for _ in worker_ids)
            receipt_sql = f'''SELECT
                substr(r.execution_id,1,{receipt_text}) AS execution_id,
                length(CAST(r.execution_id AS BLOB)) AS execution_id_bytes,
                r.sequence,
                substr(r.payload_digest,1,{receipt_text}) AS payload_digest,
                length(CAST(r.payload_digest AS BLOB)) AS payload_digest_bytes,
                length(CAST(r.result_json AS BLOB)) AS result_bytes,
                {receipt_text} AS _snapshot_text_limit
                FROM worker_receipts AS r
                WHERE r.execution_id IN ({placeholders})
                ORDER BY r.execution_id, r.sequence LIMIT ?'''
            receipt_records = _snapshot_fetch_records(
                conn, receipt_sql, tuple(worker_ids), deadline=deadline, budget=budget,
                convert=lambda row: _snapshot_receipt_row(row, workers_by_id),
                text_limit=receipt_text, text_fields=2)
            for receipt in receipt_records:
                _snapshot_validate_receipt_body(conn, receipt, deadline)
        closed = _snapshot_epoch_row(conn, deadline)
        if closed != opened:
            raise RuntimeStoreError('stale_epoch')
    current = _snapshot_read_epoch(db, deadline)
    if current != opened:
        raise RuntimeStoreError('stale_epoch')
    admission_status_counts = {status: 0 for status in _RUNTIME_ADMISSION_STATUSES}
    for record in admission_records:
        admission_status_counts[record['status']] += 1
    worker_status_counts = {status: 0 for status in _RUNTIME_WORKER_STATUSES}
    for record in worker_records:
        worker_status_counts[record['status']] += 1
    _snapshot_check_deadline(deadline)
    return {
        'schema': _RUNTIME_SNAPSHOT_SCHEMA,
        'epoch': opened['epoch'],
        'instance_id': opened['instance_id'],
        'admissions': admission_records,
        'workers': worker_records,
        'worker_receipts': receipt_records,
        'counts': {
            'admissions': len(admission_records),
            'workers': len(worker_records),
            'worker_receipts': len(receipt_records),
            'admissions_by_status': admission_status_counts,
            'workers_by_status': worker_status_counts,
            'records': budget['records'],
            'encoded_bytes': budget['bytes'],
        },
    }
