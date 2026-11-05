"""The authority's shared transaction steps: the input checks, the epoch fence, the admission and session row reads,
the row projection, the steering transfer, the queued-replacement binding, the mutation receipt replay, the worker
assignment checks and the terminal tombstone reads.

These steps raise the contract's ``AuthorityRefusal`` with main's reason codes.
Repositories run every verb inside :func:`refused`, which turns the fence codes into the contract's typed refusals, and
the conversation bodies' refusals and the store's SQLite failures into the contract's error family.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from typing import Iterator, Optional

from persistence.authority.vocabulary import ADMITTED_REQUEST_CONFLICT, STEERING, admission_fingerprint
from persistence.core.canonical import canonical_json
from persistence.core.errors import AuthorityRefusal, FenceStale, ProofRejected, StaleFenceKind, UnknownExecution
from session_rules.results import compact_result

from .._holder import StoreHolder
from ..conversation._errors import refusals
from ..conversation._store.terminal import (
    ADMISSION_PREFIX, IDENTITY_PREFIX, RESULT_PREFIX, WORKER_PREFIX, encode_result, identity_key, load_result,
)

__all__ = [
    "ADMISSION_PREFIX",
    "IDENTITY_PREFIX",
    "REPLACEMENT_PREFIX",
    "RESULT_PREFIX",
    "STEERING",
    "WORKER_PREFIX",
    "check_replaced_input",
    "compact_result",
    "encode_result",
    "identity_key",
    "load_result",
    "refused",
    "replacement_binding",
    "replay_mutation_receipt",
    "retire_replaced_input",
    "retry_terminal_admission",
    "settle_steer_transfers",
    "terminal_admission",
    "terminal_api_admissions",
]


_FENCE_KINDS = {kind.value: kind for kind in StaleFenceKind}


def _typed(exc: AuthorityRefusal) -> AuthorityRefusal:
    """The contract subclass for a fence, unknown-execution or proof refusal, ``exc`` itself for any other code."""
    if exc.reason in _FENCE_KINDS:
        typed: AuthorityRefusal = FenceStale(_FENCE_KINDS[exc.reason], "authority", "")
    elif exc.reason == "unknown_execution":
        typed = UnknownExecution("authority", "")
    elif exc.reason == "permission_denied":
        typed = ProofRejected("authority", "")
    else:
        return exc
    typed.field, typed.detail = exc.field, exc.detail
    return typed


@contextmanager
def refused(holder: Optional[StoreHolder] = None, session_id: Optional[str] = None) -> Iterator[None]:
    """Raise these steps' refusals as the contract's typed refusals, and the conversation bodies' refusals and SQLite
    failures as the contract's errors (``holder`` and ``session_id`` as the conversation family's ``refusals``)."""
    try:
        with refusals(holder, session_id):
            yield
    except AuthorityRefusal as exc:
        if type(exc) is not AuthorityRefusal:
            raise
        typed = _typed(exc)
        if typed is exc:
            raise
        raise typed from exc


def _text(value):
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise AuthorityRefusal('invalid_params')
    return value


# The contract package keeps the one copy.
_json = canonical_json


def _epoch(conn, epoch):
    row = conn.execute('SELECT epoch FROM runtime_epoch WHERE singleton=1').fetchone()
    if type(epoch) is not int or row is None or row[0] != epoch:
        raise AuthorityRefusal('stale_epoch')


def _session(conn, session_id):
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
    if row is None:
        raise AuthorityRefusal('not_found')
    return row


def _admission(conn, admission_id):
    row = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (admission_id,)).fetchone()
    if row is None:
        row = terminal_admission(conn, admission_id)
    if row is None:
        raise AuthorityRefusal('not_found')
    return row


def _retire_admission_workers(conn, row, owner_epoch):
    """Retire the worker rows this admission's generation owns on BOTH the logical
    FIFO owner and its current physical transcript (reset/compression move the private
    worker assignment to the child while target_session_id stays the logical ID)."""
    lineage = json.loads(row['lineage_json'])
    targets = list({row['target_session_id'], lineage[-1] if lineage else row['target_session_id']})
    scope = f"session_id IN ({','.join('?' * len(targets))}) AND generation=? AND owner_epoch=?"
    # Settlement closes an execution that never wrote execution.finish: its read copies go too.
    from ..workers._runtime_workers import compact_terminal_receipts
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


_CANCEL_QUEUED = "UPDATE session_admissions SET status='terminal',outcome='cancelled' WHERE admission_id=?"


def _requeue_transfer(conn, *, admission_id: str, epoch: int, generation: int) -> bool:
    """A transfer still bound to the claim ``(epoch, generation)`` goes back to its FIFO place (``seq``)."""
    return conn.execute("UPDATE session_admissions SET status='queued',outcome=NULL,generation=NULL "
                        "WHERE admission_id=? AND status='terminal' AND outcome=? AND owner_epoch=? "
                        "AND generation=?", (admission_id, STEERING, epoch, generation)).rowcount == 1


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
        raise AuthorityRefusal('admission_started')
    conn.execute(_CANCEL_QUEUED, (admission_id,))
    conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)', (binding, json.dumps(admission_id)))


def check_replaced_input(conn, admission_id: str, binding: str) -> None:
    """The exact replay of a replacement answers only the request that retired ``admission_id``."""
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (binding,)).fetchone()
    if saved is None or json.loads(saved[0]) != admission_id:
        raise AuthorityRefusal('admission_conflict')


def replay_mutation_receipt(conn, key, digest):
    """The stored result of an exact retry, ``None`` when unseen; a different request refuses."""
    old = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    if old is None:
        return None
    receipt = json.loads(old[0])
    if receipt['digest'] != digest:
        raise AuthorityRefusal('admission_conflict')
    return receipt['result']


def _worker_public(row):
    result = dict(row)
    result.pop('adoption_digest')
    return result


def _worker_assignment(conn, execution_id, session_id, generation):
    row = conn.execute('SELECT * FROM worker_executions WHERE execution_id=?', (execution_id,)).fetchone()
    if row is None:
        raise AuthorityRefusal('not_found')
    if row['session_id'] != session_id:
        raise AuthorityRefusal('permission_denied')
    if (type(generation) is not int or row['generation'] != generation
            or _session(conn, session_id)['runtime_generation'] != generation):
        raise AuthorityRefusal('stale_generation')
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
        raise AuthorityRefusal('admission_conflict')
    return rows[0] if rows else None


def _secret_digest(secret):
    _text(secret)
    return hashlib.sha256(secret.encode('utf-8')).hexdigest()


def terminal_api_admissions(conn, request_id):
    """Retired API admissions that carried *request_id*: a deleted session's tombstones, kept for
    the exact-retry horizon, still spend its Idempotency-Key for every other target."""
    return [json.loads(row[0]) for row in conn.execute(
        "SELECT value FROM state_meta WHERE key GLOB ? "
        "AND json_extract(value, '$.principal_id')='api' AND json_extract(value, '$.request_id')=?",
        (ADMISSION_PREFIX + '*', request_id))]


def terminal_admission(conn, admission_id):
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                         (ADMISSION_PREFIX + admission_id,)).fetchone()
    return json.loads(saved[0]) if saved else None


def retry_terminal_admission(holder, *, epoch, principal_id, session_id, request_id, payload, intent='queue'):
    """None means a live target; a retired miss/conflict is a definitive refusal.

    This is a read-only identity lookup, not authorization to attach, restore a
    route, or write. HTTP authentication/grant checks must precede this query.
    """
    for value in (principal_id, session_id, request_id):
        _text(value)
    digest = admission_fingerprint(canonical_target=session_id,
        payload={'input': json.loads(_json(payload)), 'intent': intent})
    with holder._read_ctx() as conn:
        _epoch(conn, epoch)
        from ..conversation._store.retirement import RETIRED_PREFIX
        if not conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + session_id,)).fetchone():
            return None
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (identity_key(principal_id, session_id, request_id),)).fetchone()
        if saved is None:
            raise AuthorityRefusal('not_found')
        row = terminal_admission(conn, json.loads(saved[0]))
        if row['payload_digest'] != digest:
            raise AuthorityRefusal('admission_conflict', **ADMITTED_REQUEST_CONFLICT)
        return _row(row)
