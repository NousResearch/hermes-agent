"""Exact execution results commit with settlement, before delivery/publication."""
import json
from contextvars import ContextVar

from hermes_state_runtime import RuntimeStoreError, _admission, _epoch, settle_session_input

execution_result: ContextVar[dict | None] = ContextVar("execution_result", default=None)
from hermes_state_terminal import RESULT_PREFIX as _RESULT_PREFIX


def record_unexecuted_failure(reply):
    """The admitted turn never ran because it failed (agent initialization raised, history
    unreadable): commit a failed result for the executing admission, so no surface settles it
    ``completed`` with the apology as the turn's output. Returns ``reply`` unchanged."""
    captured = execution_result.get()
    if captured is not None and 'result' not in captured:
        captured['result'] = {'final_response': '', 'messages': [], 'failed': True, 'completed': False,
                              'error': str(reply or 'The admitted turn failed.')}
    return reply


def record_terminal_failure(result):
    """Retain an owner-side failure even if an abandoned executor returns while delivery unwinds."""
    captured = execution_result.get()
    if captured is not None:
        captured['terminal_result'] = result
    return result


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


def close_discarded_turn(db, conn, row):
    """Close a discarded ``unknown`` turn the way a failed turn is closed, on the resolution's
    own transaction (``resolve_unknown_session_input(_terminal_write=...)``).

    The lost input stays in the transcript for the user to resend, but an open ``user`` tail would be
    merged into the follower's provider request by consecutive-user repair, re-sending the discarded
    turn as context. A Hermes-authored boundary (``display_kind=failed_turn``, stripped of its type
    before the wire) ends it. Its side effects are unknown, so the hedged copy. The boundary lands on
    the CURRENT physical transcript (local reset/compression lineage tip), resolved on this
    connection. Idempotent on the durable tail, like the gateway and core failed-turn closers."""
    import time
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND, PARTIAL_FAILED_TURN_NOTICE
    from hermes_state_local_lineage import local_physical_target
    from hermes_state_runtime import _canonical_chain
    target = _canonical_chain(conn, local_physical_target(conn, row['target_session_id']))[-1]
    tail = conn.execute("SELECT role FROM messages WHERE session_id=? AND active=1 "
                        "AND role NOT IN ('session_meta','system') ORDER BY id DESC LIMIT 1", (target,)).fetchone()
    if tail is None or tail[0] != 'user':
        return False
    db._append_messages_in_transaction(conn, target, [{
        'role': 'assistant', 'content': PARTIAL_FAILED_TURN_NOTICE, 'timestamp': time.time(),
        'display_kind': FAILED_TURN_DISPLAY_KIND}])
    return True
