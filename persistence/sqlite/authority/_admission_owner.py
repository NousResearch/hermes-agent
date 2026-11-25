"""The gateway's admission SQL on the store's own transactions: the typed extras the admission write runs before its
insert (the queued replacement, the automation supersede, the API target claim, the ``/retry`` transcript rewrite,
the hosted member check), the discarded-turn closer settle and resolution run, the unknown marking of a failed
settlement, and the gateway's admission reads.
"""

from __future__ import annotations

import json
import sqlite3

from persistence.authority.admissions import ActiveReplace, CarrierRewind, RetryRewriteRefused
from persistence.core.errors import AuthorityRefusal
from session_rules.api_payloads import compacted_api_payload
from session_rules.automation import folded_payload

from ._runtime import (
    ADMISSION_PREFIX,
    RESULT_PREFIX,
    _admission,
    _epoch,
    _json,
    _row,
    load_result,
    replacement_binding,
    retire_replaced_input,
    settle_steer_transfers,
    terminal_api_admissions,
)

__all__ = [
    "_busy",
    "_mark_unknown",
    "admission_result",
    "api_compaction_candidates",
    "authorize_admission",
    "close_discarded_turn",
    "compact_api_payloads",
    "finish_result_precheck",
    "has_pending",
    "list_unknown",
    "match_admissions",
    "pending_targets",
    "terminal_ids",
    "unfinalized_webhook_admissions",
    "webhook_ingress_payloads",
]


def authorize_admission(holder, conn, *, epoch, principal_id, session_id, request_id, replaces=None, supersedes=None,
                        api_target=None, rewrite=None, hosted=None, fold=None) -> None:
    """The typed extras of one admission, inside its write after the queue cap and gate and before the insert."""
    if replaces is not None:
        retire_replaced_input(conn, replaces.admission_id, replacement_binding(principal_id, session_id, request_id))
    if supersedes is not None:
        _replace_queued(conn, supersedes, session_id)
    if api_target is not None:
        _admit_target(session_id, request_id, dict(api_target.settings), conn, client_key=api_target.client_key)
    if rewrite is not None:
        _retry_rewrite(holder, conn, rewrite)
    if hosted is not None:
        _hosted_authorize_write(conn, epoch, hosted)
    if fold is not None:
        _fold_queued(conn, fold.admission_id, session_id, fold.skip_ids)


def _replace_queued(conn, admission_id, session_id):
    """Write guard: cancel the replaced queued tick in the replacement's own transaction; refuse the
    replacement when that tick was claimed or settled first (its output already ran)."""
    changed = conn.execute("UPDATE session_admissions SET status='terminal',outcome='cancelled' "
                           "WHERE admission_id=? AND target_session_id=? AND status='queued'",
                           (admission_id, session_id)).rowcount
    if changed != 1:
        raise AuthorityRefusal('stale_generation')


def _fold_queued(conn, admission_id, session_id, skip):
    """``_replace_queued`` for a completion fold, which must still be the FIFO tail when it commits
    (a prompt admitted since the tail was read keeps the earlier result ahead of it; ``skip`` names
    the finishing process's ticks, withdrawn right after). The replaced row keeps only its own
    identity: its text and earlier identities ride the batch, so no fold stores them twice."""
    row = conn.execute("SELECT seq, payload_json FROM session_admissions WHERE admission_id=? "
                       "AND target_session_id=? AND status='queued'", (admission_id, session_id)).fetchone()
    later = row and conn.execute("SELECT admission_id FROM session_admissions WHERE target_session_id=? "
                                 "AND status='queued' AND seq>?", (session_id, row[0])).fetchall()
    if row is None or any(other[0] not in skip for other in later):
        raise AuthorityRefusal('stale_generation')
    _replace_queued(conn, admission_id, session_id)
    payload = folded_payload(json.loads(row[1]))
    conn.execute('UPDATE session_admissions SET payload_json=? WHERE admission_id=?',
                 (json.dumps(payload, separators=(',', ':'), ensure_ascii=False), admission_id))


# The state_meta key prefix of an API session's stored turn settings.
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
    if any(row['target_session_id'] != session_id for row in terminal_api_admissions(conn, request_id)):
        raise AuthorityRefusal('admission_conflict')


def _retry_rewrite(holder, conn, rewrite):
    """The ``/retry`` rewrite inside the re-send's admission write: the carrier-aware rewind or the active-transcript
    replacement, each refusing when the transcript changed since the plan was read."""
    try:
        if isinstance(rewrite, CarrierRewind):
            from ..conversation._store.mutation_transcript import rewind_in_transaction
            rewind_in_transaction(holder, conn, rewrite.session_id, rewrite.target_row_id,
                                  preserve_compaction_handoff=True,
                                  expected_active_ids=list(rewrite.expected_active_ids),
                                  expected_target_content=rewrite.expected_target_content)
        elif isinstance(rewrite, ActiveReplace):
            from ..conversation._store.messages import replace_messages_in_transaction
            replace_messages_in_transaction(holder, conn, rewrite.session_id, [dict(m) for m in rewrite.messages],
                                            active_only=True, reject_active_turn_lease=True,
                                            expected_active_ids=list(rewrite.expected_active_ids))
    except (AuthorityRefusal, sqlite3.OperationalError):
        raise  # gate refusals and writer contention keep their own handling
    except Exception as exc:
        raise RetryRewriteRefused(rewrite.session_id, str(exc)) from exc


# The hosted room's member check: the epoch must be current and the member still allowed to submit.
def _hosted_authorize_write(conn, epoch, guard):
    # Admission checks must share the FIFO writer's snapshot. Opening
    # another transaction here would reintroduce the revocation race.
    from ..gateway.hosted_rooms import _member_authorized_locked
    _epoch(conn, epoch)
    # Raise to refuse rather than return False: a guard that only returns False
    # is ignored wherever the admission hook signals refusal by raising.
    if not _member_authorized_locked(conn, guard, operation='submit'):
        raise AuthorityRefusal('permission_denied')


def close_discarded_turn(holder, conn, row, notice=None, *, replace_after=None, display_metadata=None):
    """Close a discarded ``unknown`` turn the way a failed turn is closed, on the resolution's
    own transaction.

    The lost input stays in the transcript for the user to resend, but an open ``user`` tail would be
    merged into the follower's provider request by consecutive-user repair, re-sending the discarded
    turn as context. A Hermes-authored boundary (``display_kind=failed_turn``, stripped of its type
    before the wire) ends it. Its side effects are unknown, so the hedged copy. The boundary lands on
    the CURRENT physical transcript (local reset/compression lineage tip), resolved on this
    connection. Idempotent on the durable tail, like the gateway and core failed-turn closers.

    A turn lost while parked on a tool (an approval or clarify prompt) ends on its assistant
    ``tool_calls`` row with no result. That tail is open too: repair prunes the unanswered call,
    drops the now-empty row and merges the discarded input into the follower's user turn, so it
    gets the same boundary (merged onto the call row, whose calls are then pruned).

    ``notice`` (an unattended automation turn settled by the owner, GW-10) replaces the hedged copy
    and is written even on a closed tail: nobody resolved it, so the transcript must say why.

    ``replace_after`` (the claim's transcript mark, an owner-stop notice): when the tail is a
    failed-turn boundary written after that mark, it is THIS turn's own (the agent's core closer
    wrote "Send it again" for the interrupted turn), so the notice replaces its copy instead of
    stacking a second boundary (#134889 GW-21). An earlier turn's boundary stays.

    ``display_metadata`` (``{error, error_surface}``) rides the boundary so a reopened chat redraws
    the turn's error card from history (``agent.conversation_loop._failed_turn_display_metadata``)."""
    import time
    from session_rules.turns import FAILED_TURN_DISPLAY_KIND, PARTIAL_FAILED_TURN_NOTICE
    from ..conversation._store.messages import _append_messages_in_transaction, _encode_display_metadata
    from ._admission_store import _canonical_chain
    from ._local import local_physical_target
    target = _canonical_chain(conn, local_physical_target(conn, row['target_session_id']))[-1]
    tail = conn.execute("SELECT role, tool_calls, id, display_kind FROM messages WHERE session_id=? AND active=1 "
                        "AND role NOT IN ('session_meta','system') ORDER BY id DESC LIMIT 1", (target,)).fetchone()
    if (notice is not None and type(replace_after) is int and tail is not None and tail[0] == 'assistant'
            and tail[3] == FAILED_TURN_DISPLAY_KIND and tail[2] > replace_after):
        conn.execute("UPDATE messages SET content=?, display_metadata=? WHERE id=?",
                     (notice, _encode_display_metadata(display_metadata), tail[2]))
        return True
    if notice is None and (tail is None or not (
            tail[0] == 'user' or (tail[0] == 'assistant' and tail[1] not in (None, '', '[]')))):
        return False
    boundary = {'role': 'assistant', 'content': notice or PARTIAL_FAILED_TURN_NOTICE, 'timestamp': time.time(),
                'display_kind': FAILED_TURN_DISPLAY_KIND}
    if display_metadata:
        boundary['display_metadata'] = display_metadata
    _append_messages_in_transaction(holder, conn, target, [boundary])
    return True


# A failed settlement: this owner's started claim becomes ``unknown``; a terminal or unknown row keeps its status.
def _mark_unknown(holder, *, epoch, admission_id, generation, transfers=None):
    def write(conn):
        _epoch(conn, epoch)
        current = _admission(conn, admission_id)
        if current['owner_epoch'] != epoch or current['generation'] != generation:
            raise AuthorityRefusal('stale_generation')
        if current['status'] in {'terminal', 'unknown'}:
            return current['status']
        if current['status'] != 'started':
            raise AuthorityRefusal('stale_generation')
        conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (admission_id,))
        # A steer it carried is never replayed: interrupted if delivered, else queued again.
        settle_steer_transfers(conn, epoch=epoch, generation=generation,
                               transfers=transfers, outcome='unknown')
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (current['target_session_id'],))
        return 'unknown'
    return holder._execute_write(write)


def finish_result_precheck(holder, *, epoch, admission_id, generation):
    """``(row, stored result)`` of this owner's claim: the result ``None`` while the row is not terminal."""
    with holder._read_ctx() as conn:
        _epoch(conn, epoch)
        current = _admission(conn, admission_id)
        if current['owner_epoch'] != epoch or current['generation'] != generation:
            raise AuthorityRefusal('stale_generation')
        if current['status'] == 'terminal':
            saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                                 (RESULT_PREFIX + admission_id,)).fetchone()
            if saved is None:
                raise AuthorityRefusal('storage_unavailable')
            return _row(current), load_result(saved[0])
        return _row(current), None


def admission_result(holder, admission_id):
    """``(row, stored result)`` of a terminal admission with a stored result, else ``None``."""
    with holder._read_ctx() as conn:
        row = _admission(conn, admission_id)
        if row['status'] != 'terminal':
            return None
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (RESULT_PREFIX + admission_id,)).fetchone()
        return (_row(row), load_result(saved[0])) if saved else None


def match_admissions(holder, *, request_id=None, admission_id=None, principal_id=None, session_id=None,
                     include_retired=False):
    """Live rows matching every given filter in ``seq`` order, then (``include_retired`` with a request id) the
    retired API tombstones carrying that request id that match the other filters: the partial-key lookups of the
    gateway's webhook ingress, automation, bot and API server paths."""
    given = [(column, value) for column, value in (('principal_id', principal_id), ('target_session_id', session_id),
                                                   ('request_id', request_id), ('admission_id', admission_id))
             if value is not None]
    where = ' AND '.join(f'{column}=?' for column, _ in given)
    with holder._read_ctx() as conn:
        rows = [_row(row) for row in conn.execute(
            'SELECT * FROM session_admissions' + (' WHERE ' + where if where else '') + ' ORDER BY seq',
            [value for _, value in given])]
        if include_retired and request_id is not None:
            rows.extend(_row(row) for row in terminal_api_admissions(conn, request_id)
                        if all(row[column] == value for column, value in given))
    return rows


def pending_targets(holder, *, statuses, principal_id=None):
    """The distinct targets of admissions in ``statuses`` (of ``principal_id`` when given), as the gateway runtime
    and the API turn path read them."""
    statuses = sorted(statuses)
    sql = ("SELECT DISTINCT target_session_id FROM session_admissions WHERE "
           + ("principal_id=? AND " if principal_id is not None else "")
           + f"status IN ({','.join('?' * len(statuses))})")
    params = ((principal_id,) if principal_id is not None else ()) + tuple(statuses)
    return frozenset(row[0] for row in holder._read_all(sql, params))


# Every admission left ``unknown``, in ``seq`` order.
def list_unknown(holder):
    with holder._read_ctx() as conn:
        return [_row(r) for r in conn.execute(
            "SELECT * FROM session_admissions WHERE status='unknown' ORDER BY seq").fetchall()]


def has_pending(holder, *, session_ids=(), statuses=None):
    """Whether a row targets one of ``session_ids`` (any target when empty) with a status in ``statuses`` (any when
    ``None``). The gateway's idle exit asks for any queued or started row, its session authorities for any row on a
    session or its lineage owner."""
    clauses, params = [], []
    if session_ids:
        clauses.append(f"target_session_id IN ({','.join('?' * len(session_ids))})")
        params.extend(session_ids)
    if statuses is not None:
        statuses = sorted(statuses)
        clauses.append(f"status IN ({','.join('?' * len(statuses))})")
        params.extend(statuses)
    sql = "SELECT 1 FROM session_admissions" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " LIMIT 1"
    return bool(holder._read_all(sql, params))


# Open work on a session: an unsettled admission, or a live worker on the session or its physical target.
_BUSY_SQL = ("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status IN ('queued','started','unknown') "
             "UNION ALL SELECT 1 FROM worker_executions WHERE session_id IN (?,?) "
             "AND status IN ('registered','running','unknown') LIMIT 1")


def _busy(holder, session_id, target):
    with holder._read_ctx() as conn:
        return conn.execute(_BUSY_SQL, (session_id, session_id, target)).fetchone() is not None


def webhook_ingress_payloads(holder, chat_id):
    # Only the signed ingress admission carries the destination; later automation rows
    # (process/watch completions) on the same chat share its source but never compete.
    query = """SELECT payload_json FROM session_admissions
        WHERE json_extract(payload_json, '$.native_text_v1.source.chat_id')=?
          AND json_extract(payload_json, '$.native_text_v1.source.platform')='webhook'
          AND json_extract(payload_json, '$.native_text_v1.automation') IS NULL"""
    return [json.loads(row['payload_json']) for row in holder._read_all(query, (chat_id,))]


# Settled webhook ingress admissions whose session is still open.
def unfinalized_webhook_admissions(holder):
    rows = holder._read_all("""
        SELECT a.admission_id FROM sessions AS s
        JOIN session_admissions AS a ON a.target_session_id=s.id
        WHERE s.source='webhook' AND s.ended_at IS NULL
          AND a.status='terminal' AND a.generation IS NOT NULL
          AND json_extract(a.payload_json, '$.native_text_v1.source.platform')='webhook'
          AND json_extract(a.payload_json, '$.native_text_v1.automation') IS NULL
        """)
    return [row['admission_id'] for row in rows]


# Which of the given admissions are settled: a terminal ledger row or a retired admission's record, 500 ids a read.
def terminal_ids(holder, admission_ids):
    terminal = []
    ids = [*admission_ids]
    with holder._read_ctx() as canon:
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ','.join('?' * len(chunk))
            terminal += [row[0] for row in canon.execute(
                f"SELECT admission_id FROM session_admissions WHERE status='terminal' "
                f"AND admission_id IN ({marks})", chunk)]
            terminal += [row[0][len(ADMISSION_PREFIX):] for row in canon.execute(
                f'SELECT key FROM state_meta WHERE key IN ({marks})',
                [ADMISSION_PREFIX + admission_id for admission_id in chunk])]
    return frozenset(terminal)


# Settled API admissions still carrying inline image data, or a stored history outside the Responses API.
# LIKE, not instr(): SQLite LIKE is ASCII case-insensitive, as ``_data_url_bytes`` is.
_CANDIDATES = ("SELECT admission_id FROM session_admissions WHERE status='terminal' AND principal_id='api' "
               "AND ((json_type(payload_json, '$.api_turn_v1.media') IS NOT NULL "
               "AND payload_json LIKE '%data:image/%') OR (json_type(payload_json, '$.api_turn_v1.history')='array' "
               "AND request_id NOT LIKE 'responses:%'))")


def api_compaction_candidates(holder):
    candidates = _CANDIDATES
    # The scan runs on a read snapshot; a write transaction opens only when there is work.
    with holder._read_ctx() as conn:
        return [row[0] for row in conn.execute(candidates)]


def compact_api_payloads(holder, ids):
    """Each given row still matching the candidate predicate, re-read and rewritten by
    ``compacted_api_payload`` in one write; returns the rows written."""
    candidates = _CANDIDATES

    def write(conn):
        compacted = 0
        for row_id in ids:
            row = conn.execute(candidates.replace('SELECT admission_id', 'SELECT payload_json, request_id')
                               + ' AND admission_id=?', (row_id,)).fetchone()
            if row is None:
                continue
            payload = compacted_api_payload(json.loads(row[0]), row[1])
            if payload is not None:
                conn.execute('UPDATE session_admissions SET payload_json=? WHERE admission_id=?',
                             (_json(payload), row_id))
                compacted += 1
        return compacted
    return holder._execute_write(write) if ids else 0
