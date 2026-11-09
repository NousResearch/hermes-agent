"""``clarify.lock`` on the session owner (#140400).

The owner sends a local viewer's clarify batch as one ``clarify`` server request
(``gateway/session_clarify_batch.py``). Ink's batch panel and Desktop's batch card answer it one
question at a time with ``clarify.lock {request_id, question_id, answer}`` (``ClarifyLockParams`` /
``ClarifyLockResult``), as they did against main's sidecar; the owner had no such verb, so every
answer was refused as an unknown method and the turn waited out ``clarify_timeout``. Only a viewer
attached to the request's session with ``session:respond`` may lock its answers. A lock may name no
profile: a Desktop window opened for a sibling profile takes that profile for its shared socket's own,
so its lock reaches the socket's launch connection. Like the closing response frame, it then settles
the request as the sibling connection on the same socket that attached the session.
"""
import json

from hermes_state_runtime import RuntimeStoreError


async def clarify_lock(connection, ref, params):
    from gateway.session_server_requests import lock_answer, may_settle
    # The dispatcher already refused unknown keys and mistyped fields against the declared
    # ``ClarifyLockParams`` (``tui_gateway/contracts/canonical.py``); a missing or empty id is ours to refuse.
    # ``profile`` naming another served profile was routed to its owner by ``dispatch``.
    request_id, question_id = params.get('request_id'), params.get('question_id')
    if not isinstance(request_id, str) or not request_id or not isinstance(question_id, str) or not question_id:
        raise RuntimeStoreError('invalid_params')
    answer = params.get('answer', '')
    if answer is not None and not isinstance(answer, str):
        answer = json.dumps(answer, ensure_ascii=False)

    def entitled(session_id, events):
        # The same rule the closing response frame for this request id goes through
        # (``AuthorityConnection._resolve_response``).
        return any(may_settle(answering.authority, answering.actor, session_id, events)
                   for answering in connection.answering_connections())
    try:
        remaining = lock_answer(request_id, question_id, answer, session_ok=entitled)
    except ValueError:
        raise RuntimeStoreError('invalid_params') from None
    if remaining is None:
        # The wait already ended (timeout / cancel) while the card was still visible: not an error.
        return {'status': 'expired'}
    return {'status': 'ok', 'remaining': remaining}
