"""Exact execution results commit with settlement, before delivery/publication."""
import json
import time
from contextvars import ContextVar
from functools import partial

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


def prepare_result(row, response, outcome, result=None):
    """``(outcome, stored_result)``: the outcome the result's own flags imply and the compact,
    redacted copy settlement stores. Pure CPU work, safe off-loop and outside the stream lock."""
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
    # Compact before redacting, so the walk covers this turn's output, not every earlier turn.
    from hermes_state_terminal import compact_result
    return outcome, _redacted(compact_result(result, user_message=row['payload'].get('text')))


def finish_result(db, *, epoch, row, response, outcome, result=None, prepared=None):
    """Delivery failure cannot rewrite an already committed execution outcome.

    `result` is the exact structured result captured in-process; the managed
    worker path commits its own before this runs and is read back here.
    `prepared` is ``prepare_result``'s output computed before the caller took a lock.
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
    outcome, stored = prepared or prepare_result(row, response, outcome, result)
    settled = settle_session_input(db, epoch=epoch, admission_id=row['admission_id'],
        generation=row['generation'], outcome=outcome, result=stored)
    return settled, response


def admission_result(db, admission_id):
    with db._read_ctx() as conn:
        row = _admission(conn, admission_id)
        if row['status'] != 'terminal':
            return None
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (_RESULT_PREFIX + admission_id,)).fetchone()
        return json.loads(saved[0]) if saved else None


def result_reclaimed(saved):
    """A completed turn's result past the retry horizon (``reclaim_settled_results``): its reply,
    outcome and accounting stand, its transcript copies are gone. ``prompt.receipt`` and the runs
    projection say so (``result_reclaimed``); only a byte-exact API replay needs what was dropped."""
    return isinstance(saved, dict) and 'reclaimed_at' in saved


_RECLAIM_INTERVAL_S = 3600
# Rows per write transaction and transactions per pass: a burst of week-old turns, or the first pass
# after an upgrade, is spread over the hourly passes instead of holding the writer.
_RECLAIM_BATCH = 200
_RECLAIM_BATCHES = 25
_last_reclaim: dict[str, float] = {}


def schedule_result_reclaim(authority):
    """After a settlement, at most hourly per store: the reclaim pass as tracked background work that
    the drain never awaits (retirement still joins it)."""
    now = time.time()
    if now - _last_reclaim.get(authority.db.db_path, 0) < _RECLAIM_INTERVAL_S:
        return None
    _last_reclaim[authority.db.db_path] = now
    from gateway.session_runtime_workers import track_mutation
    return track_mutation(authority, reclaim_expired_results(authority))


async def reclaim_expired_results(authority):
    """Stamp results settled before horizon stamps existed, then reduce those past the retry horizon,
    one bounded write transaction per batch off the owner loop. Stops when the owner drains."""
    import asyncio
    import hermes_state_terminal as terminal
    budget = _RECLAIM_BATCHES
    try:
        for step in (terminal.backfill_result_horizon, terminal.reclaim_settled_results):
            while budget and not (authority.retiring or authority.runner._draining):
                done = await asyncio.to_thread(authority.db._execute_write,
                                               partial(step, now=time.time(), limit=_RECLAIM_BATCH))
                budget -= bool(done)
                if done < _RECLAIM_BATCH:
                    break
    except Exception:
        import logging
        logging.getLogger(__name__).exception('Reclaiming settled results past the retry horizon failed')


def close_discarded_turn(db, conn, row):
    """Close a discarded ``unknown`` turn the way a failed turn is closed, on the resolution's
    own transaction (``resolve_unknown_session_input(_terminal_write=...)``).

    The lost input stays in the transcript for the user to resend, but an open ``user`` tail would be
    merged into the follower's provider request by consecutive-user repair, re-sending the discarded
    turn as context. A Hermes-authored boundary (``display_kind=failed_turn``, stripped of its type
    before the wire) ends it. Its side effects are unknown, so the hedged copy. The boundary lands on
    the CURRENT physical transcript (local reset/compression lineage tip), resolved on this
    connection. Idempotent on the durable tail, like the gateway and core failed-turn closers.

    A turn lost while parked on a tool (an approval or clarify prompt) ends on its assistant
    ``tool_calls`` row with no result. That tail is open too: repair prunes the unanswered call,
    drops the now-empty row and merges the discarded input into the follower's user turn, so it
    gets the same boundary (merged onto the call row, whose calls are then pruned)."""
    import time
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND, PARTIAL_FAILED_TURN_NOTICE
    from hermes_state_local_lineage import local_physical_target
    from hermes_state_runtime import _canonical_chain
    target = _canonical_chain(conn, local_physical_target(conn, row['target_session_id']))[-1]
    tail = conn.execute("SELECT role, tool_calls FROM messages WHERE session_id=? AND active=1 "
                        "AND role NOT IN ('session_meta','system') ORDER BY id DESC LIMIT 1", (target,)).fetchone()
    if tail is None or not (tail[0] == 'user' or (tail[0] == 'assistant' and tail[1] not in (None, '', '[]'))):
        return False
    db._append_messages_in_transaction(conn, target, [{
        'role': 'assistant', 'content': PARTIAL_FAILED_TURN_NOTICE, 'timestamp': time.time(),
        'display_kind': FAILED_TURN_DISPLAY_KIND}])
    return True
