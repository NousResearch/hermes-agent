"""The terminal-admission key prefixes and the stored-result codec."""


import json
import time

from persistence.sqlite._state.keys import admission_fingerprint


RESULT_PREFIX = 'gateway.admission.result.v1.'


ADMISSION_PREFIX = 'gateway.terminal_admission.v1.'


IDENTITY_PREFIX = 'gateway.terminal_identity.v1.'


WORKER_PREFIX = 'gateway.terminal_worker.v1.'


_STORED = '_stored'


#: Sibling key recording when a settlement's result was stored (unix seconds, as text). It lives
#: beside the result row rather than inside its payload, so the stored-result contract and its
#: exact-equality readers stay untouched; ``prune_settled_results`` ages a LIVE session's results by
#: it. Written once per admission (first settlement wins -- a rewrite never extends its life).
SETTLED_PREFIX = 'gateway.admission.settled_at.v1.'


def settled_key(admission_id):
    return SETTLED_PREFIX + admission_id


def stamp_settled(conn, admission_id, *, now=None):
    """Record a settlement's time. Idempotent: an existing stamp is kept."""
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO NOTHING',
                 (settled_key(admission_id), str(int(time.time() if now is None else now))))


def encode_result(result):
    """The stored row keeps the reply once. When the turn's closing assistant message carries
    exactly ``final_response``, its ``content`` is stored as null and the row gets
    ``_stored: {v: 1, reply_ref: <that message's index>}``. Retirement drops ``messages`` and
    keeps ``final_response``, so a deleted session's exact retry still has its reply. A result
    that already has ``_stored`` is stored whole with that value escaped as ``{v: 1, raw: ...}``."""
    if not isinstance(result, dict):
        return result
    if _STORED in result:
        return {**result, _STORED: {'v': 1, 'raw': result[_STORED]}}
    value = result.get('result')
    if not isinstance(value, dict) or not isinstance(value.get('messages'), list):
        return result
    reply, messages = value.get('final_response'), list(value['messages'])
    last = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], dict)
                 and messages[i].get('role') == 'assistant'), None)
    if not (isinstance(reply, str) and reply and last is not None and messages[last].get('content') == reply):
        return result
    messages[last] = {**messages[last], 'content': None}
    return {**result, 'result': {**value, 'messages': messages}, _STORED: {'v': 1, 'reply_ref': last}}


def load_result(text):
    """Decode a stored result row. Only a row carrying ``encode_result``'s exact envelope is
    decoded (to the dict it encoded, key order included); every other row, including all rows
    written before it, is returned as stored."""
    # Decodes in place: safe only because each call parses its own copy. A cache in front of this
    # must hand out copies, never the decoded dict itself.
    result = json.loads(text)
    if not isinstance(result, dict) or not isinstance(envelope := result.get(_STORED), dict):
        return result
    if envelope.keys() == {'v', 'raw'} and type(envelope['v']) is int and envelope['v'] == 1:
        result[_STORED] = envelope['raw']
        return result
    value, ref = result.get('result'), envelope.get('reply_ref')
    messages = value.get('messages') if isinstance(value, dict) else None
    if not (envelope.keys() == {'v', 'reply_ref'} and type(envelope['v']) is int and envelope['v'] == 1
            and type(ref) is int and isinstance(messages, list) and 0 <= ref < len(messages)
            and isinstance(value.get('final_response'), str) and isinstance(messages[ref], dict)
            and messages[ref].get('role') == 'assistant' and messages[ref].get('content', 0) is None):
        return result
    messages[ref] = {k: value['final_response'] if k == 'content' else v for k, v in messages[ref].items()}
    del result[_STORED]
    return result


def identity_key(principal_id, session_id, request_id):
    return IDENTITY_PREFIX + admission_fingerprint(canonical_target=session_id,
        payload={'principal': principal_id, 'request': request_id})
