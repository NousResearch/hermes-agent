"""Private terminal-only tombstone queries; never reconstruct executable history.

The authenticated edge supplies the original principal/target/request and frozen
payload. A retired target permits only that exact digest, never fresh admission.
Worker receipt reads additionally require the original private adoption proof.
"""
import hmac
import json
import time

from hermes_state_runtime import ADMITTED_REQUEST_CONFLICT, RuntimeStoreError, _epoch, _json, _row, _secret_digest, _text, admission_fingerprint

RESULT_PREFIX = 'gateway.admission.result.v1.'
ADMISSION_PREFIX = 'gateway.terminal_admission.v1.'
IDENTITY_PREFIX = 'gateway.terminal_identity.v1.'
WORKER_PREFIX = 'gateway.terminal_worker.v1.'


def compact_result(result, *, user_message=None):
    """Retain replay/accounting and current-turn API output, never cumulative history."""
    result = dict(result)
    value = result.get('result')
    if not isinstance(value, dict):
        return result
    value = dict(value)
    # The tool schema list rides every turn result for in-process transcript logging; it is the
    # same session-wide copy each turn (tens of KB, more with MCP) and no stored-result reader uses it.
    value.pop('tools', None)
    messages = value.get('messages')
    if isinstance(messages, list):
        # Agent results include all previous turns. APIs only project this turn's
        # assistant/tool suffix; previous_response_id stores its own history. Compression
        # is the exception: its replacement snapshot is the only immutable record of the
        # rewritten prefix for this turn. A later live transcript may include a successor.
        if not value.get('_compressed') and not value.get('_messages_are_turn_suffix'):
            from agent.turn_context import reanchor_current_turn_user_idx
            anchor = value.get('current_turn_user_idx')
            if not (type(anchor) is int and 0 <= anchor < len(messages)
                    and isinstance(messages[anchor], dict) and messages[anchor].get('role') == 'user'):
                anchor = reanchor_current_turn_user_idx(messages, user_message)
            value['messages'] = messages[anchor + 1:]
            if anchor >= 0 and isinstance(messages[anchor], dict) and messages[anchor].get('api_content'):
                # The user row as the model saw it (memory prefetch, plugin context): a chained
                # Responses turn replays it instead of a row rebuilt from the raw request text.
                value['_turn_user_row'] = {k: messages[anchor][k] for k in ('role', 'content', 'api_content')}
            # Recovery/verification nudges are later user-role rows within this same turn.
            # Consumers must not infer a new boundary from them after the real anchor is gone.
            value['_messages_are_turn_suffix'] = True
            value.pop('current_turn_user_idx', None)
        value['messages'] = [_stored_message(message) for message in value['messages']]
    result['result'] = value
    return result


# Live-only replay sidecar the transcript never persists: a turn reloaded from state.db replays the
# persisted ``reasoning_details`` instead (``agent.bedrock_adapter._reasoning_details_blocks``), and its
# bytes repeat them (Bedrock's redacted reasoning is the same base64 in both). A chained turn built
# from this copy replays the prior turn the way a resumed session does.
_LIVE_ONLY_FIELDS = frozenset({'bedrock_content_blocks'})


def _stored_message(message):
    """The stored copy of one turn message: what its transcript row keeps, as a resumed session
    reads it. Live-only sidecars are dropped and inline media (a screenshot's data URL) becomes the
    transcript's text marker (``agent.session_persistence._durable_content``)."""
    if not isinstance(message, dict):
        return message
    out = {k: v for k, v in message.items() if k not in _LIVE_ONLY_FIELDS}
    if _has_inline_media(out.get('content')):
        from agent.session_persistence import _durable_content
        out['content'] = _durable_content(out['content'])
    return out


def _has_inline_media(content):
    from agent.session_persistence import _IMAGE_PART_TYPES
    from agent.tool_dispatch_helpers import _is_multimodal_tool_result
    return _is_multimodal_tool_result(content) or (isinstance(content, list) and any(
        isinstance(part, dict) and part.get('type') in _IMAGE_PART_TYPES for part in content))


# Top-level row key only ``encode_result`` writes. The row's top level is built by the gateway
# (``result``/``usage``), never by the agent or a message, so message fields can't collide with it.
_STORED = '_stored'


# Message fields a stored result keeps by reference when the turn's transcript row already holds
# them byte-exact (tool output, reasoning blobs, tool-call arguments: the bulk of a busy store's rows).
_REF_FIELDS = ('content', 'tool_calls', 'reasoning', 'reasoning_content', 'reasoning_details',
               'codex_reasoning_items', 'codex_message_items')
_JSON_COLUMNS = frozenset({'tool_calls', 'reasoning_details', 'codex_reasoning_items', 'codex_message_items'})
_REF_MIN_CHARS = 64  # a short field costs more as a reference than it saves
# Top-level key ``reference_transcript_rows`` hands to ``encode_result``, which moves it into the
# envelope. Like ``_STORED``, the row's top level is gateway-built, so nothing else writes it.
_ROW_REFS = '_row_refs'


def _transcript_rows(conn, row_ids):
    rows = {}
    ids = sorted(set(row_ids))
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        rows.update((row[0], row) for row in conn.execute(
            f"SELECT id, {', '.join(_REF_FIELDS)} FROM messages WHERE id IN ({','.join('?' * len(chunk))})",
            chunk).fetchall())
    return rows


def _row_field(row, field):
    """*field* of a transcript row as ``get_messages_as_conversation`` decodes it."""
    raw = row[1 + _REF_FIELDS.index(field)]
    if raw is None:
        return None
    if field == 'content':
        from hermes_state import SessionDB
        return SessionDB._decode_content(raw)
    if field in _JSON_COLUMNS:
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw
    return raw


def reference_transcript_rows(conn, result):
    """Store the turn's heavy message fields by reference to the transcript rows that hold them.

    The agent flushes every message of the turn to ``messages`` before it settles; each flushed
    dict carries its row (``_row_id`` with ``_db_persisted``). A field is referenced only when the
    row's stored value decodes to exactly the dict's value, so a read rebuilds the same message.
    Its slot keeps ``None`` (key order is unchanged) and ``_row_refs`` names the row and fields.
    Call it on the compacted, redacted copy settlement stores, in the settlement's transaction;
    a reader redacts what it reloads too (``gateway.session_results.admission_result``). A row
    rewritten later in place (a tool result the turn budget spilled) reloads as the transcript
    now holds it; a row that is gone (its session deleted: retirement drops ``messages`` first)
    reloads as ``None``. A turn whose rows were never flushed is stored by value, as before."""
    value = result.get('result') if isinstance(result, dict) else None
    if (not isinstance(value, dict) or not isinstance(messages := value.get('messages'), list)
            or _STORED in result):  # a producer's own ``_stored`` is stored whole (``encode_result``)
        return result
    from agent.persisted_turn_receipt import committed_row_id
    owned = {i: row_id for i, m in enumerate(messages) if (row_id := committed_row_id(m)) is not None}
    if not owned:
        return result
    rows, refs, out = _transcript_rows(conn, owned.values()), [], list(messages)
    for i, row_id in owned.items():
        row, message = rows.get(row_id), messages[i]
        if row is None:
            continue
        # The reply stays where ``encode_result`` keeps it (``final_response``): a read without the
        # transcript (delivery after settlement, retirement) still has it.
        fields = [f for f in _REF_FIELDS if message.get(f) is not None
                  and not (f == 'content' and message[f] == value.get('final_response'))
                  and len(json.dumps(message[f], ensure_ascii=False, default=str)) > _REF_MIN_CHARS
                  and _row_field(row, f) == message[f]]
        if fields:
            out[i] = {k: None if k in fields else v for k, v in message.items()}
            refs.append([i, row_id, fields])
    if not refs:
        return result
    return {**result, 'result': {**value, 'messages': out}, _ROW_REFS: refs}


def _valid_refs(refs, messages):
    return isinstance(refs, list) and refs and isinstance(messages, list) and all(
        isinstance(ref, list) and len(ref) == 3 and type(ref[0]) is int and 0 <= ref[0] < len(messages)
        and type(ref[1]) is int and isinstance(ref[2], list) and ref[2] and isinstance(messages[ref[0]], dict)
        and all(f in _REF_FIELDS and f in messages[ref[0]] and messages[ref[0]][f] is None for f in ref[2])
        for ref in refs)


def _restore_refs(conn, messages, refs, transform):
    rows = _transcript_rows(conn, [ref[1] for ref in refs])
    for index, row_id, fields in refs:
        row = rows.get(row_id)
        if row is None:
            continue  # the row is gone: its fields stay None (documented on reference_transcript_rows)
        messages[index] = {k: transform(_row_field(row, k)) if k in fields else v
                           for k, v in messages[index].items()}


def encode_result(result):
    """The stored row keeps the reply once. When the turn's closing assistant message carries
    exactly ``final_response``, its ``content`` is stored as null and the row gets
    ``_stored: {v: 1, reply_ref: <that message's index>}``. Retirement drops ``messages`` and
    keeps ``final_response``, so a deleted session's exact retry still has its reply. A result
    that already has ``_stored`` is stored whole with that value escaped as ``{v: 1, raw: ...}``.
    Fields :func:`reference_transcript_rows` referenced ride the envelope as ``row_refs``."""
    if not isinstance(result, dict):
        return result
    refs = result.get(_ROW_REFS)
    result = {k: v for k, v in result.items() if k != _ROW_REFS}
    if _STORED in result:
        return {**result, _STORED: {'v': 1, 'raw': result[_STORED]}}
    value = result.get('result')
    if not isinstance(value, dict) or not isinstance(value.get('messages'), list):
        return result
    reply, messages = value.get('final_response'), list(value['messages'])
    envelope = {'v': 1, 'row_refs': refs} if _valid_refs(refs, messages) else {'v': 1}
    last = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], dict)
                 and messages[i].get('role') == 'assistant'), None)
    if isinstance(reply, str) and reply and last is not None and messages[last].get('content') == reply:
        messages[last] = {**messages[last], 'content': None}
        envelope['reply_ref'] = last
    if len(envelope) == 1:
        return result
    return {**result, 'result': {**value, 'messages': messages}, _STORED: envelope}


def load_result(text, *, conn=None, transform=None):
    """Decode a stored result row. Only a row carrying ``encode_result``'s exact envelope is
    decoded (to the dict it encoded, key order included); every other row, including all rows
    written before it, is returned as stored. With *conn*, referenced message fields are reloaded
    from their transcript rows through *transform* (the storage redaction); without it they stay
    ``None`` (the reply and the result's own fields are always restored)."""
    # Decodes in place: safe only because each call parses its own copy. A cache in front of this
    # must hand out copies, never the decoded dict itself.
    result = json.loads(text)
    if not isinstance(result, dict) or not isinstance(envelope := result.get(_STORED), dict):
        return result
    if envelope.keys() == {'v', 'raw'} and type(envelope['v']) is int and envelope['v'] == 1:
        result[_STORED] = envelope['raw']
        return result
    value, ref, refs = result.get('result'), envelope.get('reply_ref', -1), envelope.get('row_refs')
    if not isinstance(value, dict) or not isinstance(messages := value.get('messages'), list):
        return result
    if not ('v' in envelope and envelope.keys() <= {'v', 'reply_ref', 'row_refs'} and len(envelope) > 1
            and type(envelope['v']) is int and envelope['v'] == 1
            and ('row_refs' not in envelope or _valid_refs(refs, messages))
            and ('reply_ref' not in envelope or (
                type(ref) is int and 0 <= ref < len(messages)
                and isinstance(value.get('final_response'), str) and isinstance(messages[ref], dict)
                and messages[ref].get('role') == 'assistant' and messages[ref].get('content', 0) is None))):
        return result
    if 'reply_ref' in envelope:
        messages[ref] = {k: value['final_response'] if k == 'content' else v for k, v in messages[ref].items()}
    if refs and conn is not None:
        _restore_refs(conn, messages, refs, transform or (lambda v: v))
    del result[_STORED]
    return result


def identity_key(principal_id, session_id, request_id):
    return IDENTITY_PREFIX + admission_fingerprint(canonical_target=session_id,
        payload={'principal': principal_id, 'request': request_id})


# A settled turn's stored result answers one exact re-send of that turn (an API Idempotency-Key
# retry, a bot/cron delivery, a recovery that re-reads its own settlement). It is a retry cache,
# not history: the newest this many per session survive, and nothing for a session that has been
# quiet past the retry horizon. Only-written-up rows stayed forever — 45.5 MB of one store.
ADMISSION_RESULT_KEEP = 16


def reclaim_admission_results(conn, *, older_than=None):
    """Bound the per-turn result cache the same way the tombstones are bounded.

    Three reclaimed classes: a result with no settled admission and no tombstone still inside the
    horizon (nothing will ever read it), one whose session has been inactive past ``older_than``
    (no retry is left to answer; default: the tombstone's own ``RETRY_HORIZON_S``), and every
    result past the newest ``ADMISSION_RESULT_KEEP`` of its session. The newest results are
    untouched, so an in-flight retry still resolves byte-exact. A retired session keeps its
    compacted copy here until ``prune_retired_tombstones`` closes the horizon — that is the delete
    path's exact-retry answer, not this cache.
    """
    from hermes_state_mutation_retirement import RETRY_HORIZON_S
    prefix, cut = RESULT_PREFIX, len(RESULT_PREFIX) + 1
    cutoff = time.time() - float(RETRY_HORIZON_S if older_than is None else older_than)

    conn.execute(
        "DELETE FROM state_meta WHERE key GLOB ? AND NOT EXISTS ("
        " SELECT 1 FROM session_admissions a WHERE a.status='terminal' AND a.admission_id=substr(state_meta.key,?))"
        " AND NOT EXISTS (SELECT 1 FROM state_meta t WHERE t.key=? || substr(state_meta.key,?)"
        "  AND CASE WHEN json_valid(t.value)"
        "           THEN COALESCE(json_extract(t.value,'$.retired_at') >= ?, 1) ELSE 1 END)",
        (prefix + '*', cut, ADMISSION_PREFIX, cut, cutoff))
    from hermes_state_common import _sql_session_last_active_by_id
    active = _sql_session_last_active_by_id('a.target_session_id')
    conn.execute(
        f"DELETE FROM state_meta WHERE key GLOB ? AND key IN ("
        f" SELECT m.key FROM state_meta m JOIN session_admissions a ON a.admission_id=substr(m.key,?)"
        f" WHERE a.status='terminal' AND COALESCE({active}, 0) < ?)", (prefix + '*', cut, cutoff))
    seen, victims = {}, []
    for key, session_id in conn.execute(
            "SELECT m.key, a.target_session_id FROM state_meta m JOIN session_admissions a"
            " ON a.admission_id=substr(m.key,?) WHERE m.key GLOB ? AND a.status='terminal'"
            " ORDER BY a.target_session_id, a.seq DESC", (cut, prefix + '*')).fetchall():
        seen[session_id] = seen.get(session_id, 0) + 1
        if seen[session_id] > ADMISSION_RESULT_KEEP:
            victims.append((key,))
    conn.executemany('DELETE FROM state_meta WHERE key=?', victims)


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


def retry_terminal_admission(db, *, epoch, principal_id, session_id, request_id, payload, intent='queue'):
    """None means a live target; a retired miss/conflict is a definitive refusal.

    This is a read-only identity lookup, not authorization to attach, restore a
    route, or write. HTTP authentication/grant checks must precede this query.
    """
    for value in (principal_id, session_id, request_id):
        _text(value)
    digest = admission_fingerprint(canonical_target=session_id,
        payload={'input': json.loads(_json(payload)), 'intent': intent})
    with db._read_ctx() as conn:
        _epoch(conn, epoch)
        from hermes_state_mutation_retirement import RETIRED_PREFIX
        if not conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + session_id,)).fetchone():
            return None
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (identity_key(principal_id, session_id, request_id),)).fetchone()
        if saved is None:
            raise RuntimeStoreError('not_found')
        row = terminal_admission(conn, json.loads(saved[0]))
        if row['payload_digest'] != digest:
            raise RuntimeStoreError('admission_conflict', **ADMITTED_REQUEST_CONFLICT)
        return _row(row)


def terminal_worker_receipt(db, *, execution_id, session_id, generation, sequence,
                            adoption_secret, payload_digest):
    """Exact retired producer evidence; possession never reactivates the worker."""
    proof = _secret_digest(adoption_secret)
    with db._read_ctx() as conn:
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (WORKER_PREFIX + execution_id,)).fetchone()
        if saved is None:
            raise RuntimeStoreError('not_found')
        worker = json.loads(saved[0])
        if worker['session_id'] != session_id or not hmac.compare_digest(worker['adoption_digest'], proof):
            raise RuntimeStoreError('permission_denied')
        if type(generation) is not int or generation != worker['generation']:
            raise RuntimeStoreError('stale_generation')
        receipt = next((r for r in worker['receipts'] if r['sequence'] == sequence), None)
        if receipt is None:
            raise RuntimeStoreError('stale_generation')
        if receipt['payload_digest'] != payload_digest:
            raise RuntimeStoreError('admission_conflict')
        if receipt['result_json'] is None:
            raise RuntimeStoreError('stale_generation')
        return json.loads(receipt['result_json'])
