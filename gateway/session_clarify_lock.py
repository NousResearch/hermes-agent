"""``clarify.lock`` on the session owner (#140400).

The owner sends a local viewer's clarify batch as one ``clarify`` server request
(``gateway/session_clarify_batch.py``). Ink's batch panel and Desktop's batch card answer it one
question at a time with ``clarify.lock {request_id, question_id, answer}`` (``ClarifyLockParams`` /
``ClarifyLockResult``), as they did against main's sidecar; the owner had no such verb, so every
answer was refused as an unknown method and the turn waited out ``clarify_timeout``. Only a viewer
attached to the request's session with ``session:respond`` may lock its answers.
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
    authority, actor = connection.authority, connection.actor

    def entitled(session_id, events):
        # The same rule the closing response frame for this request id goes through.
        return may_settle(authority, actor, session_id, events)
    try:
        remaining = lock_answer(request_id, question_id, answer, session_ok=entitled)
    except ValueError:
        raise RuntimeStoreError('invalid_params') from None
    if remaining is None:
        # The wait already ended (timeout / cancel) while the card was still visible: not an error.
        return {'status': 'expired'}
    return {'status': 'ok', 'remaining': remaining}
