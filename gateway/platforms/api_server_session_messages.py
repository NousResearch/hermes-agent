"""Read and append session transcripts through the authenticated API edge."""
from aiohttp import web


async def append_messages(adapter, request):
    from gateway.platforms.api_server import _error_response, _openai_error
    from gateway.platforms.api_server_authority_runs import _authority, store_error_response
    from gateway.session_api import bind_api_steps, run_steps_off_loop
    from gateway.session_append import append_messages as record_messages
    from gateway.session_contract import SessionActor, SessionRef
    from persistence.authority.external_messages import validate_messages
    from gateway.session_persistence import authority_persistence
    from persistence.core.errors import AuthorityRefusal, STORE_FAILURES
    body, error = await adapter._read_json_body(request)
    if error is not None:
        return error
    session_id = request.match_info['session_id']
    authority = _authority(adapter)
    if authority is None:
        return _error_response('Session authority unavailable', 503, code='runtime_unavailable')
    try:
        if set(body) != {'messages'}:
            raise AuthorityRefusal('invalid_params')
        messages = validate_messages(body['messages'])
        if await authority_persistence(authority).aio.sessions.get(session_id) is None:
            return _error_response('Session not found', 404, code='not_found')
        # Only the authenticated API edge may bind a newly created API session.
        # Binding never creates a model turn; it retains the same API ownership
        # checks as /v1/runs and refuses a local or foreign conversation.
        await run_steps_off_loop(authority, bind_api_steps(authority, session_id))
        actor = SessionActor('api', authority.profile_id, frozenset({'session:submit'}), 'api-append')
        result = await record_messages(authority, actor, SessionRef(authority.profile_id, session_id), messages)
        return web.json_response({'object': 'hermes.session.messages', **result})
    except (AuthorityRefusal, *STORE_FAILURES, OSError) as exc:
        return store_error_response(_openai_error, exc)


async def read_messages(self, request: web.Request) -> web.Response:
    from gateway.platforms.api_server import _coerce_request_bool, _error_response
    from persistence.conversation.messages import TranscriptRead, TranscriptView
    """GET /api/sessions/{session_id}/messages."""
    session_id = request.match_info["session_id"]
    _, err = await self._get_existing_session_or_404(session_id)
    if err:
        return err
    store = self._store_beside(await self._ensure_store_async())
    resolved_id = await store.aio.lineage.resolve_resume_target(session_id)
    raw_limit, raw_offset = request.query.get("limit"), request.query.get("offset", "0")
    order = request.query.get("order")
    if order not in (None, "oldest", "latest"):
        return _error_response("order must be one of: oldest, latest", 400, code="invalid_pagination")
    try:
        offset = int(raw_offset)
        requested_limit = None if raw_limit is None else int(raw_limit)
    except (TypeError, ValueError):
        offset = requested_limit = -1
    if offset < 0 or (requested_limit is not None and requested_limit < 0):
        return _error_response("limit and offset must be non-negative integers", 400, code="invalid_pagination")
    default_page = requested_limit is None
    latest_page = order == "latest" or (order is None and default_page)
    limit = 500 if default_page else min(requested_limit, 500)
    include_compacted = _coerce_request_bool(request.query.get("include_compacted"), default=False)
    # Compression lineage: return root→tip messages, matching the REST router (#51058).
    messages = await store.aio.messages.read_transcript(resolved_id, TranscriptRead(
        view=TranscriptView.DISPLAY if include_compacted else TranscriptView.ACTIVE, limit=limit, offset=offset,
        latest=latest_page, include_lineage=True))
    return web.json_response({
        "object": "list", "session_id": resolved_id,
        "data": [self._message_response(m.as_wire_row()) for m in messages],
        "pagination": {
            "limit": limit, "offset": offset,
            "order": order or ("latest" if default_page else "oldest"),
            "returned": len(messages)}})
