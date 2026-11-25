"""The clarify tool's whole question batch for a classic-CLI viewer's turn, in one request.

Main's HermesCLI answered ``clarify_callback(questions)`` with its batch panel: every question shown
at once, one active, answers revisitable until all are locked, ``{answers: {qid: raw | None},
outcome}``. The messaging path asks one card per question. For a classic-CLI local viewer the owner
sends the batch as the existing ``clarify`` server→client request (``ClarifyRequestParams`` →
``ClarifyResult``) through the same request/response machinery as the sudo prompt, so main's own panel
answers it unchanged, and blocks until it answers, the turn is interrupted, or the clarify timeout
(the same config both sides read) passes.
"""
from __future__ import annotations

_QUESTION_KEYS = ("qid", "question", "choices", "multi_select")


def ask_clarify_batch(turn, questions) -> dict:
    from gateway.session_server_requests import ask_viewer_result
    from tools.clarify_gateway import get_clarify_timeout
    from gateway.session_cli_turn import cli_turn
    authority, session_id, generation = turn._approval_owner
    selected = cli_turn(getattr(turn._ctx, 'inbound_message_id', None)).get('clarify_projection') == 'canonical_batch_v1'
    live = authority.sessions.get(session_id)
    agent = (getattr(turn._ctx, "agent_holder", None) or [None])[0]
    interrupted = (lambda: bool(getattr(agent, "_interrupt_requested", False))) if agent is not None else (
        lambda: False)
    if live is None:
        return {"answers": {}, "outcome": "cancelled"}
    wire = [{key: entry.get(key) for key in _QUESTION_KEYS if entry.get(key) is not None} for entry in questions]
    timeout = get_clarify_timeout()
    # ``qids``: the viewer's batch panel locks each answer with ``clarify.lock`` (#140400).
    result = ask_viewer_result(live.event_stream, session_id, "clarify", {"questions": wire},
                               cancelled=interrupted, timeout=timeout if timeout > 0 else None,
                               qids=[entry["qid"] for entry in questions],
                               batch_owner=(authority, generation) if selected else None)
    if result is None:
        return {"answers": {}, "outcome": "cancelled" if interrupted() else "timed_out"}
    answers = result.get("answers")
    if not isinstance(answers, dict):
        return {"answers": {}, "outcome": "cancelled"}
    known = {entry["qid"] for entry in questions}
    clean = {qid: (value if isinstance(value, str) else None) for qid, value in answers.items() if qid in known}
    # ``session_server_requests`` settles a batch with its terminal outcome (an absent outcome on a frame
    # that carried answers is ``submitted``; a cancel-all or error is ``cancelled``).
    outcome = result.get("outcome") if result.get("outcome") in ("submitted", "timed_out", "cancelled") else "cancelled"
    return {"answers": clean, "outcome": outcome}


async def batch_capabilities(connection, ref, params):
    from persistence.core.errors import AuthorityRefusal
    from tui_gateway import server
    result = server.client_capabilities(None, params, connection.transport, canonical=True)
    if 'error' in result:
        raise AuthorityRefusal('invalid_params')
    return result['result']


def _batch_entitled(connection, ref):
    from gateway.session_server_requests import may_settle
    from persistence.core.errors import AuthorityRefusal
    if not getattr(connection.transport, 'canonical_clarify_projection', False):
        raise AuthorityRefusal('invalid_params')
    connection.authority.authorize(connection.actor, ref, 'session:respond')
    live = connection.authority.sessions.get(ref.session_id)
    if live is None or not may_settle(connection.authority, connection.actor, ref.session_id, live.event_stream):
        raise AuthorityRefusal('permission_denied')
    return lambda sid, stream: sid == ref.session_id and may_settle(connection.authority, connection.actor, sid, stream)


async def batch_lock(connection, ref, params):
    import json
    from gateway.session_server_requests import lock_answer
    from persistence.core.errors import AuthorityRefusal
    from tui_gateway.contracts.canonical import CanonicalBatchLockParams
    parsed = _parse(CanonicalBatchLockParams, params)
    entitled = _batch_entitled(connection, ref)
    answer = parsed.answer
    if isinstance(answer, list):
        answer = json.dumps(answer, ensure_ascii=False)
    try:
        remaining = lock_answer(parsed.request_id, parsed.question_id, answer, session_ok=entitled,
                                identity=parsed.identity.model_dump())
    except ValueError:
        raise AuthorityRefusal('invalid_params') from None
    return {'status': 'expired'} if remaining is None else {'status': 'ok', 'remaining': remaining}


async def batch_respond(connection, ref, params):
    from gateway.session_server_requests import resolve
    from tui_gateway.contracts.canonical import CanonicalBatchRespondParams
    parsed = _parse(CanonicalBatchRespondParams, params)
    entitled = _batch_entitled(connection, ref)
    # Preserve the native answered/presence side effect; settlement itself still arbitrates in one registry.
    from gateway.session_presence import seen
    accepted = resolve({'id': parsed.request_id, 'result': parsed.result.model_dump()}, session_ok=entitled,
                       identity=parsed.identity.model_dump(),
                       answered=lambda sid: seen(connection, sid))
    return {'status': 'ok' if accepted else 'expired'}


def batch_snapshot(connection, stream, session_id):
    from gateway.session_server_requests import canonical_batches, may_settle, open_requests
    selected = getattr(connection.transport, 'canonical_clarify_projection', False)
    with stream.lock:
        entitled = may_settle(connection.authority, connection.actor, session_id, stream)
        # The selected record is answer-bearing even in a legacy reconnect representation.
        # Do not extend session:read into session:respond merely by choosing an older encoding.
        result = {'open_requests': open_requests(stream, session_id, canonical=selected or not entitled)}
        if selected:
            # Resume may await metadata after attach captured its older cursor.
            # Fence queued openings with the stream watermark of THIS registry read.
            epoch, sequence = stream.watermark()
            result['clarify_batch_watermark'] = {'replay_epoch': epoch, 'last_sequence': sequence}
            result['clarify_batches'] = ([{**entry, 'replayed': True} for entry in canonical_batches(stream, session_id)]
                                         if entitled else [])
        return result


def _parse(model, params):
    from pydantic import ValidationError
    from persistence.core.errors import AuthorityRefusal
    try:
        return model.model_validate(params, strict=True)
    except ValidationError:
        raise AuthorityRefusal('invalid_params') from None
