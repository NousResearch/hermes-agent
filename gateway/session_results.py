"""Exact execution results commit with settlement, before delivery/publication."""
import json
from contextvars import ContextVar

from hermes_state_runtime import RuntimeStoreError, _admission, _epoch, settle_session_input

execution_result: ContextVar[dict | None] = ContextVar("execution_result", default=None)
from hermes_state_terminal import RESULT_PREFIX as _RESULT_PREFIX


def _redacted(value):
    """The stored result is a state.db copy of the model's answer and history: redacted at this
    storage boundary like every transcript row (``security.redact_secrets``). The live viewer
    prints the completion event, never this copy."""
    if isinstance(value, str):
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(value)
    if isinstance(value, dict):
        return {k: _redacted(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redacted(v) for v in value]
    return value


def retain_result(db, *, epoch, row, result):
    return settle_session_input(db, epoch=epoch, admission_id=row['admission_id'],
                                generation=row['generation'], outcome='completed', result=_redacted(result))


def finish_result(db, *, epoch, row, response, outcome, result=None):
    """Delivery failure cannot rewrite an already committed execution outcome.

    `result` is the exact structured result captured in-process; the managed
    worker path commits its own before this runs and is read back here.
    """
    with db._read_ctx() as conn:
        _epoch(conn, epoch)
        current = _admission(conn, row['admission_id'])
        if current['owner_epoch'] != epoch or current['generation'] != row['generation']:
            raise RuntimeStoreError('stale_generation')
        if current['status'] == 'terminal':
            saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                                 (_RESULT_PREFIX + row['admission_id'],)).fetchone()
            if saved is None:
                raise RuntimeStoreError('storage_unavailable')
            result = json.loads(saved[0])
            return dict(current), result['result'].get('final_response') or ''
    if result is None:
        result = {'result': {'final_response': response or '', 'messages': []}, 'usage': {}}
    value = result['result']
    if value.get('interrupted'):
        outcome = 'interrupted'
    elif value.get('failed') or value.get('error'):
        outcome = 'failed'
    if outcome in ('failed', 'interrupted'):
        value['failed' if outcome == 'failed' else 'interrupted'] = True
        value['completed'] = False
    settled = settle_session_input(db, epoch=epoch, admission_id=row['admission_id'],
        generation=row['generation'], outcome=outcome, result=_redacted(result))
    return settled, response


def admission_result(db, admission_id):
    with db._read_ctx() as conn:
        row = _admission(conn, admission_id)
        if row['status'] != 'terminal':
            return None
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (_RESULT_PREFIX + admission_id,)).fetchone()
        return json.loads(saved[0]) if saved else None


def close_discarded_turn(db, target_id):
    """Close a discarded ``unknown`` turn the way a failed turn is closed.

    The lost input stays in the transcript for the user to resend, but an open ``user`` tail would be
    merged into the follower's provider request by consecutive-user repair, re-sending the discarded
    turn as context. A Hermes-authored boundary (``display_kind=failed_turn``, stripped of its type
    before the wire) ends it. Its side effects are unknown, so the hedged copy. Idempotent on the
    durable tail, like the gateway and core failed-turn closers."""
    if db.latest_conversation_role(target_id) != 'user':
        return False
    import time
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND, PARTIAL_FAILED_TURN_NOTICE
    db.append_message(target_id, 'assistant', PARTIAL_FAILED_TURN_NOTICE, timestamp=time.time(),
                      display_kind=FAILED_TURN_DISPLAY_KIND)
    return True
