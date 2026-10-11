"""Durable deletion fences outside prunable transcript rows."""
import json
from hermes_state_runtime import RuntimeStoreError, admission_fingerprint, _json, _text

RETIRED_PREFIX = 'gateway.retired_session.v1.'


def retired_session(db, session_id):
    with db._read_ctx() as conn:
        return conn.execute('SELECT 1 FROM state_meta WHERE key=?',
                            (RETIRED_PREFIX + session_id,)).fetchone() is not None


def has_mutation_receipt(db, principal_id, session_id, request_id):
    for value in (principal_id, session_id, request_id):
        _text(value)
    key = 'gateway.mutation.v1.' + admission_fingerprint(
        canonical_target=session_id, payload={'principal': principal_id, 'request': request_id})
    with db._read_ctx() as conn:
        return conn.execute('SELECT 1 FROM state_meta WHERE key=?', (key,)).fetchone() is not None


_TRANSCRIPT_FIELDS = frozenset({'messages', 'last_reasoning', 'tools'})


def _compact_result(conn, admission_id):
    """A retired result keeps the exact-retry outcome (final response, flags, usage/accounting)
    but not the transcript copies a turn result carries (cumulative ``messages``, reasoning)."""
    from hermes_state_terminal import RESULT_PREFIX
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (RESULT_PREFIX + admission_id,)).fetchone()
    if saved is None:
        return
    value = json.loads(saved[0])
    result = value.get('result')
    if isinstance(result, dict):
        value['result'] = {**{k: v for k, v in result.items() if k not in _TRANSCRIPT_FIELDS}, 'messages': []}
    conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(value), RESULT_PREFIX + admission_id))


def retire_terminal_receipts(conn, session_ids):
    from hermes_state_terminal import ADMISSION_PREFIX, WORKER_PREFIX, identity_key
    for sid in session_ids:
        admissions = conn.execute('SELECT * FROM session_admissions WHERE target_session_id=?', (sid,)).fetchall()
        workers = conn.execute('SELECT * FROM worker_executions WHERE session_id=?', (sid,)).fetchall()
        states = {r['status'] for r in [*admissions, *workers]} - {'terminal'}
        if states:
            raise RuntimeStoreError('unknown_execution' if 'unknown' in states else 'session_busy')
        for raw in admissions:
            row = dict(raw)
            from hermes_state_media import retire_media
            retire_media(conn, json.loads(row['payload_json']), row['request_id'])
            # Keep the digest for exact retries, not another copy of user input/history.
            row['payload_json'] = '{}'
            row['lineage_json'] = '[]'
            _compact_result(conn, row['admission_id'])
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                         (ADMISSION_PREFIX + row['admission_id'], _json(row)))
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                         (identity_key(row['principal_id'], sid, row['request_id']), json.dumps(row['admission_id'])))
        for raw in workers:
            row = dict(raw)
            receipts = [dict(r) for r in conn.execute(
                'SELECT sequence,payload_digest,result_json FROM worker_receipts WHERE execution_id=? ORDER BY sequence',
                (row['execution_id'],))]
            # A terminal worker can only replay an explicit ``execution.finish`` receipt. Every
            # other result (history/context reads) is user data that must not outlive the delete,
            # even when it is the last one: settlement terminalizes a failed worker without a
            # finish receipt. Digests stay so a late duplicate is still recognised as a conflict.
            closing = admission_fingerprint(canonical_target=sid,
                payload={'operation': 'execution.finish', 'payload': {}})
            for receipt in receipts:
                if receipt['payload_digest'] != closing:
                    receipt['result_json'] = None
            row['receipts'] = receipts
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                         (WORKER_PREFIX + row['execution_id'], _json(row)))
            conn.execute('DELETE FROM worker_receipts WHERE execution_id=?', (row['execution_id'],))
        conn.execute('DELETE FROM worker_executions WHERE session_id=?', (sid,))
        conn.execute('DELETE FROM session_admissions WHERE target_session_id=?', (sid,))


_LIVE_LEDGER_SQL = """SELECT 1 FROM session_admissions WHERE target_session_id=? AND status IN ('queued','started','unknown')
    UNION ALL SELECT 1 FROM worker_executions WHERE session_id=? AND status IN ('registered','running','unknown') LIMIT 1"""


def retire_sessions(conn, session_ids):
    """The complete deletion fence for ids that are being deleted in this transaction.

    Terminal receipts alone are not enough: without the ``RETIRED_PREFIX`` marker an exact
    retry of a settled request reports ``not_found`` instead of its terminal receipt, a late
    accounting/constructor backfill recreates the deleted row, and the same request is then
    admitted a second time against it. Every delete path (canonical mutate, legacy
    ``delete_session*``, prunes and sweeps) must publish the whole fence, so it lives here once."""
    retire_terminal_receipts(conn, session_ids)
    retire_mutation_receipts(conn, session_ids)
    retire_routes(conn, session_ids)
    from hermes_state_local import retire_local_receipts
    retire_local_receipts(conn, session_ids)


# Mutation-receipt fields that copy user content: the rewound message, a compaction summary and
# a user-set title (rename/sidebar).
_MUTATION_TRANSCRIPT_FIELDS = ('target_message', 'summary', 'title')
# Every session a receipt result can name. A local owner's receipt is keyed by its logical id, but
# its rewind/compress content comes from the PHYSICAL target (a reset child): deleting only that
# child must strip it too, or the surviving owner's exact retry replays the deleted text.
_RECEIPT_SESSION_PATHS = ('$.result.session_id', '$.result.target_message.session_id',
                          '$.result.target_session_id', '$.result.previous_target_session_id')


def retire_mutation_receipts(conn, session_ids):
    """Keep each exact-retry mutation receipt (digest, ids, revision) but drop the user content
    a rewind/compress/rename result carries, so a retry after the delete cannot read it back."""
    if not session_ids:
        return
    named = ' OR '.join(f"json_extract(value,'{path}') IN (SELECT value FROM json_each(?1))"
                        for path in _RECEIPT_SESSION_PATHS)
    rows = conn.execute(
        "SELECT key,value FROM state_meta WHERE key GLOB 'gateway.mutation.v1.*' AND "
        f"CASE WHEN json_valid(value) THEN ({named}) END", (json.dumps(list(session_ids)),)).fetchall()
    for key, raw in rows:
        receipt = json.loads(raw)
        result = receipt.get('result')
        if isinstance(result, dict) and any(f in result for f in _MUTATION_TRANSCRIPT_FIELDS):
            receipt['result'] = {k: (None if k in _MUTATION_TRANSCRIPT_FIELDS else v) for k, v in result.items()}
            conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(receipt), key))


def _local_conversations(conn):
    """Member id -> every id of its local conversation: the creation id (policy, FIFO, generation),
    each receipt lineage segment and the current target. One prefix scan, not a lookup per swept
    id. Malformed receipts name nothing."""
    from hermes_state_local import POLICY_PREFIX
    members = {}
    for (raw,) in conn.execute('SELECT value FROM state_meta WHERE key GLOB ?', (POLICY_PREFIX + '*',)):
        try:
            receipt = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(receipt, dict):
            continue
        entry, lineage = receipt.get('entry'), receipt.get('lineage')
        ids = tuple(x for x in [receipt.get('session_id'), *(lineage if isinstance(lineage, list) else []),
                                entry.get('session_id') if isinstance(entry, dict) else None] if isinstance(x, str))
        members.update(dict.fromkeys(ids, ids))
    return members


def retire_prunable(conn, session_ids):
    """Sweep variant of :func:`retire_sessions`: fence the idle sessions and return only those ids.
    A session with live or unknown work is skipped, so one busy row cannot abort a whole
    prune/empty-session sweep (explicit deletes still refuse with ``session_busy``). A local
    reset/compression conversation is swept only whole: while any of its ids is kept or has live
    work (the owner's queued admission continues on the current segment), every one is skipped."""
    swept, conversations = set(session_ids), _local_conversations(conn)

    def removable(sid):
        whole = conversations.get(sid, (sid,))
        return swept.issuperset(whole) and not any(conn.execute(_LIVE_LEDGER_SQL, (x, x)).fetchone() for x in whole)
    quiet = [sid for sid in session_ids if removable(sid)]
    retire_sessions(conn, quiet)
    return quiet


def delete_in_transaction(db, conn, session_id, payload):
    from hermes_state_mutation_guards import require_idle, delete_targets
    targets = delete_targets(conn, session_id)
    require_idle(db, conn, targets)
    retire_sessions(conn, targets)
    for sid in targets:
        db._bump_conversation_generation(conn, sid, 'session_reset')
        conn.execute('UPDATE sessions SET parent_session_id=NULL, runtime_revision=runtime_revision+1 WHERE parent_session_id=?', (sid,))
        conn.execute('DELETE FROM messages WHERE session_id=?', (sid,))
    conn.executemany('DELETE FROM sessions WHERE id=?', [(sid,) for sid in targets])
    db._delete_unreferenced_system_prompts(conn)
    return set(), {'deleted_ids': targets}


def entry_session_id(raw):
    """The ``session_id`` a stored routing/receipt JSON object names, else None. Unrelated rows
    are not validated here: a malformed one (an array, invalid JSON) names no target, so it can
    neither be retired nor abort the delete of a different session; the routing loader skips it too."""
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value.get('session_id') if isinstance(value, dict) else None


def retire_routes(conn, session_ids):
    # OR IGNORE: the marker is a fact, not a receipt; re-retiring an id that was recreated
    # beside its tombstone must not abort the delete that removes it again.
    for sid in session_ids:
        conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)',
                     (RETIRED_PREFIX + sid, '{}'))
        conn.execute('DELETE FROM state_meta WHERE key=?', ('gateway.api.settings.v1.' + sid,))
    targets = set(session_ids)
    for row in conn.execute('SELECT scope,session_key,entry_json FROM gateway_routing').fetchall():
        if entry_session_id(row['entry_json']) in targets:
            conn.execute('DELETE FROM gateway_routing WHERE scope=? AND session_key=?',
                         (row['scope'], row['session_key']))
