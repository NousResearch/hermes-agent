"""Exact execution results commit with settlement, before delivery/publication."""
from contextvars import ContextVar

execution_result: ContextVar[dict | None] = ContextVar("execution_result", default=None)


def record_unexecuted_failure(reply, reason=None, error=None):
    """The admitted turn never ran because it failed (agent initialization raised, history
    unreadable): commit a failed result for the executing admission, so no surface settles it
    ``completed`` with the apology as the turn's output. A *reason* is stamped as the turn's
    non-retryable failure code, so clients render that code's card. *error* replaces the reply
    as the result's ``error`` (the raw cause a local client shows as its details line).
    Returns ``reply`` unchanged."""
    captured = execution_result.get()
    if captured is not None and 'result' not in captured:
        captured['result'] = {'final_response': '', 'messages': [], 'failed': True, 'completed': False,
                              'error': str(error or reply or 'The admitted turn failed.')}
        if reason:
            from agent.turn_failure_copy import stamp_failure
            stamp_failure(captured['result'], reason, False)
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
        return {k: _redacted_cron_result(v) if k == 'cron_result' else _redacted(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [_redacted(v) for v in value]
    return value


def _redacted_cron_result(value):
    """Redact a canonical run tuple while preserving only validated length framing."""
    if (not isinstance(value, list) or len(value) != 4
            or not isinstance(value[0], bool) or not isinstance(value[1], str)):
        return _redacted(value)
    document = value[1]
    from cron.scheduler_prompt import (
        _PROMPT_FRAME_RE, _PROMPT_HEADING, _PROMPT_SEPARATOR,
        _RESPONSE_FRAME_RE, _RESPONSE_HEADING, _RESPONSE_TERMINATOR,
    )
    prompt = _PROMPT_FRAME_RE.search(document)
    response = None
    if prompt and document.startswith(_PROMPT_HEADING, prompt.end() - len(_PROMPT_HEADING)):
        body_start = prompt.end()
        try:
            prompt_end = body_start + int(prompt.group(1))
        except ValueError:
            return _redacted(value)
        response_start = prompt_end + len(_PROMPT_SEPARATOR)
        frame = (_RESPONSE_FRAME_RE.match(document, response_start)
                 if document[prompt_end:response_start] == _PROMPT_SEPARATOR else None)
        if frame:
            tail = document[frame.end():]
            try:
                length = int(frame.group(1))
            except ValueError:
                return _redacted(value)
            if len(tail) == length + len(_RESPONSE_TERMINATOR) and tail.endswith(_RESPONSE_TERMINATOR):
                header = document[:prompt.start()]
                prompt_body = document[body_start:prompt_end]
                response_body = tail[:-len(_RESPONSE_TERMINATOR)]
                from agent.redact import redact_sensitive_text
                header = redact_sensitive_text(header)
                prompt_body = redact_sensitive_text(prompt_body)
                response_body = redact_sensitive_text(response_body)
                # Never weaken whole-document masking across component boundaries.
                # Prefer ordinary fail-closed output to a valid but leaky frame.
                old_counts = (header + document[prompt.start():body_start] + prompt_body
                              + document[prompt_end:frame.end()] + response_body
                              + _RESPONSE_TERMINATOR)
                if old_counts != redact_sensitive_text(document):
                    return _redacted(value)
                document = (header + f"**Prompt Characters:** {len(prompt_body)}\n"
                            + _PROMPT_HEADING + prompt_body + _PROMPT_SEPARATOR
                            + f"**Response Characters:** {len(response_body)}\n"
                            + _RESPONSE_HEADING + response_body + _RESPONSE_TERMINATOR)
                response = document
    if response is None:
        document = _redacted(document)
    return [value[0], document, *(_redacted(item) for item in value[2:])]


def retain_result(store, *, epoch, row, result):
    from gateway.session_authority import admission_row
    return admission_row(store.admissions.settle(epoch=epoch, admission_id=row['admission_id'],
                                                 generation=row['generation'], outcome='completed',
                                                 result=_redacted(result)))


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
    from session_rules.results import compact_result
    return outcome, _redacted(compact_result(result, user_message=row['payload'].get('text')))


def finish_result(store, *, epoch, row, response, outcome, result=None, prepared=None, transfers=None):
    """Delivery failure cannot rewrite an already committed execution outcome.

    `store` is the profile's storage interface handle.
    `result` is the exact structured result captured in-process; the managed
    worker path commits its own before this runs and is read back here.
    `prepared` is ``prepare_result``'s output computed before the caller took a lock.
    `transfers` are the queued steers handed to this claim; they settle in the same write.
    """
    from gateway.session_authority import admission_row
    from persistence.authority.admissions import TurnBoundary
    current = store.admissions.settled_result(admission_id=row['admission_id'], epoch=epoch,
                                              generation=row['generation'])
    if current.result is not None:
        return admission_row(current.record), current.result['result'].get('final_response') or ''
    outcome, stored = prepared or prepare_result(row, response, outcome, result)
    # A turn the owner's restart/stop interrupted: its reason commits as a Hermes transcript notice
    # in this same write, so a viewer that reconnects after the restart reads why (#134889 GW-21);
    # the live ``message.complete.warning`` carries the same copy. A retry that finds the row
    # terminal returns above, so the notice is written once.
    # A gateway-owned refusal (a message typed into a cron session) commits its explanation and
    # error card as the transcript's failed-turn boundary in this same write too: the admission
    # result is not history, so a reopened chat would otherwise show nothing about the refusal.
    from gateway.session_settlement_recovery import gateway_refusal_boundary, owner_stop_notice
    notice, display_metadata = owner_stop_notice(result, outcome, row), None
    if notice is None and (refusal := gateway_refusal_boundary(result, outcome)) is not None:
        notice, display_metadata = refusal
    boundary = None if notice is None else TurnBoundary(
        notice=notice, replace_after=row.get('transcript_mark'), display_metadata=display_metadata)
    settled = store.admissions.settle(epoch=epoch, admission_id=row['admission_id'], generation=row['generation'],
                                      outcome=outcome, result=stored, transfers=transfers, boundary=boundary)
    return admission_row(settled), response


def admission_result(store, admission_id, *, persistence=None):
    """The stored result of a terminal admission, else None. *store* is the profile's storage interface
    handle (*persistence*, when given, is used instead); a caller that passes only its ``state.db`` has
    no handle to read through: ``StoreUnavailable``."""
    store = persistence if persistence is not None else store
    if not hasattr(store, 'admissions'):
        from persistence.core.errors import StoreUnavailable
        raise StoreUnavailable('no storage interface handle for the admission result')
    saved = store.admissions.settled_result(admission_id=admission_id)
    return None if saved is None else saved.result
