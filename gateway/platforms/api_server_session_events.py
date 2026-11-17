"""HTTP long-poll over the durable session event log (gateway/session_event_log.py).

``GET /api/sessions/{id}/events?after=&limit=&types=&turn_id=&wait_ms=`` answers at once when
anything newer than ``after`` is logged, else waits up to ``wait_ms`` for it. ``GET /v1/runs/{id}``
takes ``wait_ms`` (+ ``status=``) to wait for a run to reach a target state. Session writes
(``POST /v1/runs``, ``POST /api/sessions/{id}/chat``) report the log's ``seq`` after their own
events, and session reads take ``min_seq`` to wait (<= 2 s) for the log to catch up with a write
the client just made. All of it is plain request/response HTTP, for clients that cannot keep a
WebSocket open.
"""
import asyncio
import json
import time

try:
    from aiohttp import web
except ImportError:  # the api_server adapter reports the missing dependency itself
    web = None  # type: ignore[assignment]

from gateway.platforms.api_server_run_idempotency import TERMINAL_STATUSES

MAX_WAIT_MS = 30000
MAX_LIMIT = 500
MIN_SEQ_WAIT_S = 2.0
_POLL_S = 0.1


def _int_param(request, name, default, maximum):
    raw = request.query.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return None
    return None if value < 0 else min(value, maximum)


def _log(adapter):
    from gateway.session_authorities import active_authority
    authority = active_authority(adapter.gateway_runner) if adapter.gateway_runner is not None else None
    return getattr(getattr(authority, 'sessions', None), 'log', None)


def _disabled():
    return web.json_response({'error': {
        'message': 'The session event log is off for this profile (config.yaml sessions.event_log: true)',
        'code': 'event_log_disabled'}}, status=409)


async def _wait_for_head(log, session_id, past, deadline):
    """Wait until the log's committed head for *session_id* passes *past*, or *deadline*."""
    await asyncio.to_thread(log.head, session_id)  # a first read of the file stays off the loop
    while log.head(session_id) <= past and time.monotonic() < deadline:
        await asyncio.sleep(_POLL_S)
    return log.head(session_id)


async def handle_session_events(adapter, request):
    """GET /api/sessions/{session_id}/events — durable events after ``after``, long-polled."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    session_id = request.match_info['session_id']
    _, err = await adapter._get_existing_session_or_404(session_id)
    if err:
        return err
    log = _log(adapter)
    if log is None:
        return _disabled()
    after = _int_param(request, 'after', 0, 2 ** 62)
    limit = _int_param(request, 'limit', 100, MAX_LIMIT)
    wait_ms = _int_param(request, 'wait_ms', 0, MAX_WAIT_MS)
    if after is None or limit is None or wait_ms is None:
        return web.json_response({'error': {'message': 'after, limit and wait_ms must be non-negative integers',
                                            'code': 'invalid_params'}}, status=400)
    types = tuple(t.strip() for t in request.query.get('types', '').split(',') if t.strip())
    turn_id = request.query.get('turn_id') or None
    deadline = time.monotonic() + wait_ms / 1000
    seen = await asyncio.to_thread(log.head, session_id)
    while True:
        events, last_seq, first_seq = await asyncio.to_thread(
            log.read, session_id, after, limit=limit or 1, types=types, turn_id=turn_id)
        if events or time.monotonic() >= deadline:
            break
        # Filtered out or nothing yet: wait for the head to move past what was just read.
        seen = await _wait_for_head(log, session_id, max(seen, last_seq), deadline)
    events = events[:limit]
    return web.json_response({
        'object': 'hermes.session.events', 'session_id': session_id, 'events': events,
        'next_after': events[-1]['seq'] if events else after, 'last_seq': last_seq,
        # Older events were trimmed: a reader whose ``after`` is below this missed some.
        'first_seq': first_seq, 'truncated': bool(first_seq and after + 1 < first_seq)})


def _run_targets(request):
    raw = request.query.get('status')
    return frozenset(s.strip() for s in raw.split(',') if s.strip()) if raw else TERMINAL_STATUSES


async def wait_for_run(adapter, request, load):
    """``GET /v1/runs/{id}?wait_ms=``: *load()* -> ``(status, error)``, re-read until the status is in
    ``status=`` (default: any terminal state) or ``wait_ms`` (<= 30 s) passes."""
    status, err = load()
    wait_ms = _int_param(request, 'wait_ms', 0, MAX_WAIT_MS)
    if err is not None or not wait_ms:
        return status, err
    targets, deadline = _run_targets(request), time.monotonic() + wait_ms / 1000
    while status.get('status') not in targets and time.monotonic() < deadline:
        await asyncio.sleep(0.25)
        status, err = load()
        if err is not None:
            return status, err
    return status, None


def _write_session_id(adapter, request, body):
    session_id = request.match_info.get('session_id') or body.get('session_id')
    run_id = body.get('run_id')
    if not session_id and run_id:
        session_id = (adapter._run_statuses.get(run_id) or {}).get('session_id')
        if not session_id:
            from gateway.platforms.api_server_authority_runs import run_admission
            owned = run_admission(adapter, run_id)
            session_id = owned[1]['target_session_id'] if owned is not None else None
    return session_id


def event_seq_middleware(adapter):
    """Writes report the log's ``seq``; reads honour ``min_seq`` (read-your-write, <= 2 s)."""
    async def middleware(request, handler):
        log = _log(adapter)
        session_id = request.match_info.get('session_id')
        if log is not None and request.method == 'GET' and session_id and 'min_seq' in request.query:
            target = _int_param(request, 'min_seq', 0, 2 ** 62)
            if target is None:
                return web.json_response({'error': {'message': 'min_seq must be a non-negative integer',
                                                    'code': 'invalid_params'}}, status=400)
            await _wait_for_head(log, session_id, target - 1, time.monotonic() + MIN_SEQ_WAIT_S)
        response = await handler(request)
        if (log is None or request.method != 'POST' or not isinstance(response, web.Response)
                or response.status >= 300 or response.content_type != 'application/json'
                or request.match_info.route.resource is None
                or request.match_info.route.resource.canonical.rsplit('/', 1)[-1] not in ('chat', 'runs')):
            return response
        body = json.loads(response.text or 'null')
        session_id = _write_session_id(adapter, request, body) if isinstance(body, dict) else None
        if not session_id:
            return response
        await asyncio.to_thread(log.flush)
        body['seq'] = await asyncio.to_thread(log.head, session_id)
        response.headers['X-Hermes-Event-Seq'] = str(body['seq'])
        response.body = json.dumps(body).encode()
        return response
    return web.middleware(middleware)


def http_routes(adapter):
    async def handler(request):
        return await handle_session_events(adapter, request)
    return [('GET', '/api/sessions/{session_id}/events', handler)]
