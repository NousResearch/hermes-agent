"""Responses retries observe their original admission before resolving mutable conversation names."""
from functools import partial
import hashlib
import json
import uuid

from hermes_state_runtime import RuntimeStoreError


def request_binding(adapter, request, durable_key):
    from gateway.platforms.api_server import _make_request_fingerprint
    return _make_request_fingerprint({'request': durable_key[1],
        'declared_session_key': adapter._parse_session_key_header(request)[0]},
        keys=('request', 'declared_session_key'))


def _request_admission_rows(db, request_id):
    """A lost derivative pointer cannot erase accepted identity on retirement."""
    from hermes_state_terminal import ADMISSION_PREFIX
    with db._read_ctx() as conn:
        rows = [dict(row) for row in conn.execute(
            "SELECT * FROM session_admissions WHERE principal_id='api' AND request_id=?", (request_id,))]
        retired = conn.execute(
            "SELECT value FROM state_meta WHERE key GLOB ? "
            "AND json_extract(value, '$.principal_id')='api' AND json_extract(value, '$.request_id')=?",
            (ADMISSION_PREFIX + '*', request_id))
        rows.extend(json.loads(row[0]) for row in retired)
    return rows


def find_admission(adapter, durable_key, request_id):
    """Scope comes from authenticated profile/key identity, never a caller-supplied session id.

    The compact pointer survives response-cache eviction and transcript retirement. A process
    loss between the authority commit and pointer commit is repaired from that exact request id.
    """
    from gateway.session_authorities import active_authority
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import get_session_admission, _row
    authority = active_authority(adapter.gateway_runner)
    if authority is None or adapter._ensure_session_db() is not authority.db:
        raise RuntimeStoreError('profile_mismatch')
    store = adapter._current_response_store()
    admission_id = store.request_admission(durable_key[0])
    if admission_id:
        row = get_session_admission(authority.db, admission_id=admission_id)
        if row is None:
            raise RuntimeStoreError('storage_unavailable')
    else:
        rows = _request_admission_rows(authority.db, request_id)
        if len(rows) > 1:
            raise RuntimeStoreError('admission_conflict')
        if not rows:
            return None
        row = _row(rows[0])
    if row['principal_id'] != 'api' or row['request_id'] != request_id:
        raise RuntimeStoreError('permission_denied')
    if store.request_created_at(durable_key[0]) is None:
        # An older admission with no wire fingerprint cannot authorize a changed request just
        # because its caller reused the same key. Existing completed cache records still replay.
        raise RuntimeStoreError('admission_conflict')
    store.bind_request_admission(durable_key[0], row['admission_id'])
    ref = SessionRef(authority.profile_id, row['target_session_id'])
    if row['status'] in {'queued', 'started'}:
        from gateway.session_api import restore_api_session
        from gateway.session_api_turn import check_api_turn
        restore_api_session(authority, ref.session_id)
        check_api_turn(authority, ref, row['payload'])
    return authority, ref, row


def admitted_context(admitted, *, replay=False):
    from gateway.session_api_turn import observe_api_turn
    _, ref, row = admitted
    payload = row['payload']
    data = payload.get('api_turn_v1', {})
    from gateway.session_api_media import rehydrate_api_images
    return dict(session_id=ref.session_id, user_message=rehydrate_api_images(payload.get('text', ''), data.get('media')),
                conversation_history=data.get('history') or [],
                instructions=data.get('settings', {}).get('ephemeral_system_prompt'),
                run_kwargs={}, run_agent=partial(observe_api_turn, admitted),
                terminal_replay=replay or row['status'] == 'terminal')


def _admit_steps(adapter, durable_key, request_id, binding, run_kwargs, model):
    """The final lookup and the admission are one API admission drive (``run_steps_off_loop``
    serializes them per authority): simultaneous observers converge on one admission."""
    from gateway.session_api_turn import admit_api_steps
    store = adapter._current_response_store()
    if not store.bind_request_key(durable_key[0], binding, model=model):
        raise RuntimeStoreError('admission_conflict')
    admitted = find_admission(adapter, durable_key, request_id)
    if admitted is not None:
        return (*admitted, True)
    try:
        admitted = yield from admit_api_steps(adapter, request_id=request_id,
            **{'route_source': 'global', 'confirmed_runtime_lock': False,
               'session_history_delivery': '', **run_kwargs})
    except Exception:
        # A definite pre-admission refusal does not reserve a poisoned identity. If storage
        # cannot prove absence, leave the binding intact for later recovery.
        if find_admission(adapter, durable_key, request_id) is None:
            store.forget_unadmitted_request(durable_key[0])
        raise
    store.bind_request_admission(durable_key[0], admitted[2]['admission_id'])
    return (*admitted, False)


async def _admit(adapter, durable_key, request_id, binding, run_kwargs, model):
    from gateway.session_api_turn import admit_api_turn_async
    *admitted, replay = await admit_api_turn_async(
        adapter, _admit_steps(adapter, durable_key, request_id, binding, run_kwargs, model))
    return admitted_context(tuple(admitted), replay=replay)


def _stored_owner(adapter, session_id):
    """A stored response keeps the physical transcript tip its turn ended on, which is a
    compression child after an out-of-place compression. The canonical admission owner is
    the lineage root (the API binding, route and FIFO stay there), so a chained turn admits
    on that root; the legacy in-process path still continues the physical tip itself."""
    from gateway.session_authorities import active_authority
    authority = active_authority(adapter.gateway_runner) if session_id else None
    return authority.logical_owner(session_id) if authority is not None else session_id


async def prepare_context(adapter, request, body, gateway_session_key, durable_key, scope, key):
    """Validate first-time requests; exact retries use the accepted payload and result only."""
    import asyncio
    from gateway.platforms.api_server import (
        _auto_truncate_response_history, _content_has_visible_payload, _error_response)
    from gateway.platforms.api_server_openai_routes import _parse_responses_input, _parse_conversation_history
    request_id = f'responses:{scope}:{key}' if durable_key else None
    if durable_key:
        admitted = find_admission(adapter, durable_key, request_id)
        if admitted is not None:
            return admitted_context(admitted, replay=True), None
    previous = body.get('previous_response_id')
    if body.get('conversation'):
        previous = adapter._current_response_store().get_conversation(body['conversation'])
    messages, error = _parse_responses_input(body['input'])
    if error is not None:
        return None, error
    history = []
    if body.get('conversation_history'):
        history, error = _parse_conversation_history(body['conversation_history'])
        if error is not None:
            return None, error
    stored_session_id = None
    instructions = body.get('instructions')
    if not history and previous:
        stored = adapter._current_response_store().get(previous)
        if stored is None:
            return None, _error_response(f'Previous response not found: {previous}', 404)
        history = list(stored.get('conversation_history', []))
        stored_session_id = await asyncio.to_thread(_stored_owner, adapter, stored.get('session_id'))
        if instructions is None:
            instructions = stored.get('instructions')
    history.extend(messages[:-1])
    user_message = messages[-1].get('content', '') if messages else ''
    if not _content_has_visible_payload(user_message):
        return None, _error_response('No user message found in input', 400)
    if body.get('truncation') == 'auto':
        history = _auto_truncate_response_history(history)
    declared_selected = not stored_session_id and bool(gateway_session_key)
    session_id = stored_session_id or await asyncio.to_thread(
        adapter._declared_conversation_session, gateway_session_key) or str(uuid.uuid4())
    if durable_key and not stored_session_id and not gateway_session_key:
        session_id = 'response-' + hashlib.sha256(f'{scope}\0{key}'.encode()).hexdigest()
    route, overrides, error = adapter._select_request_route(body, session_id=session_id,
        gateway_session_key=gateway_session_key, model_alias=body.get('model'))
    if error is not None:
        return None, error
    from gateway.platforms.api_server import _request_relay_metadata
    run_kwargs = dict(user_message=user_message, conversation_history=history,
        ephemeral_system_prompt=instructions, session_id=session_id,
        gateway_session_key=gateway_session_key, bind_declared_conversation=declared_selected,
        **overrides, route=route, relay_metadata=_request_relay_metadata(body))
    if durable_key:
        return await _admit(adapter, durable_key, request_id, request_binding(adapter, request, durable_key),
                            run_kwargs, body.get('model', adapter._model_name)), None
    return dict(session_id=session_id, user_message=user_message, conversation_history=history,
                instructions=instructions, run_kwargs=run_kwargs, run_agent=adapter._run_agent), None
