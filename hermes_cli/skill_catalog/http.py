"""Optional HTTP delegation before legacy request-model validation.

The original route handler remains authoritative when no catalog is registered,
including validation errors and background hub action serialization. Provider
references never reach that handler while this profile has live catalogs.
"""
import asyncio
import contextvars
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse

from hermes_cli.skill_catalog.installed import installed_records
from hermes_cli.skill_catalog.validation import CatalogError
from hermes_cli.skill_catalog.worker import deadline_for
from hermes_cli.web_routers._common import destructive_profile


class CatalogRoute(APIRoute):
    def get_route_handler(self):
        stock = super().get_route_handler()

        async def handle(request):
            try:
                return await delegate(request, stock)
            except CatalogError as exc:
                return JSONResponse(exc.payload, status_code=exc.status)
        return handle


def _service(profile, *, deadline=None):
    from hermes_cli.skill_catalog.bootstrap import lookup
    return lookup(profile, deadline=deadline)


_lookup_lock = threading.Lock()
_lookup_slots = threading.BoundedSemaphore(4)
_lookup_executor = None


async def _lookup_service(profile, deadline):
    """Cold joins own bounded capacity, never the stock public-search executor."""
    global _lookup_executor
    from hermes_cli.skill_catalog.bootstrap import discovery_required
    if not discovery_required(profile):
        return None
    if not _lookup_slots.acquire(blocking=False):
        raise CatalogError('busy', 'Catalog discovery is busy.')
    try:
        with _lookup_lock:
            if _lookup_executor is None:
                _lookup_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='catalog-lookup')
        context = contextvars.copy_context()
        future = _lookup_executor.submit(context.run, _service, profile, deadline=deadline)
    except BaseException:
        _lookup_slots.release()
        raise
    # Cancellation does not free capacity while the actual waiter is still alive.
    future.add_done_callback(lambda _future: _lookup_slots.release())
    return await asyncio.wrap_future(future)


class _Body(dict):
    """Retain JSON member ambiguity until the optional-provider branch is known."""

    def __init__(self, pairs):
        super().__init__(pairs)
        self.duplicate = len(pairs) != len(self)
        self.profiles = [value for key, value in pairs if key == 'profile']


def _hint(raw):
    try:
        return json.loads(raw, object_pairs_hook=_Body) if raw else _Body([])
    except (ValueError, UnicodeError):
        return None


def _body(raw):
    body = _hint(raw)
    if not isinstance(body, _Body) or body.duplicate:
        raise CatalogError('invalid_request', 'Invalid catalog request.')
    return dict(body)


def _fields(body, required, optional=()):
    if not set(required) <= body.keys() or body.keys() - set(required) - set(optional):
        raise CatalogError('invalid_request', 'Invalid catalog request fields.')


def _profile(body, query):
    selected = body.get('profile')
    if 'profile' in body and type(selected) is not str:
        raise CatalogError('invalid_request', 'Invalid profile.')
    if selected is not None and query is not None and selected != query:
        raise CatalogError('invalid_request', 'Conflicting catalog profiles.')
    return selected or query


async def _selected_service(request, hint, deadline):
    queries = request.query_params.getlist('profile')
    profiles = hint.profiles if isinstance(hint, _Body) else []
    selectors = queries + profiles
    ambiguous = (len(queries) > 1 or len(profiles) > 1
                 or any(type(value) is not str for value in profiles)
                 or len({value for value in selectors if type(value) is str}) > 1)
    if ambiguous:
        # A last-value-wins parser must not hide a live catalog behind an unknown
        # profile. Inspect each proposed scope without granting either authority.
        candidates = dict.fromkeys(value for value in selectors if type(value) is str)
        for profile in candidates or (None,):
            try:
                service = await _lookup_service(profile, deadline)
            except HTTPException:
                continue
            if service is not None:
                raise CatalogError('invalid_request', 'Ambiguous catalog profiles.')
        return None
    profile = next(iter(profiles or queries), None)
    return await _lookup_service(profile, deadline)


def _invoke(service, method, *args, **kwargs):
    # Hydration belongs to this worker, never the ASGI/gateway event loop.
    from agent.runtime_scope import bind_profile_scope
    with bind_profile_scope(Path(service.home)):
        return getattr(service, method)(*args, **kwargs)


async def _provider_action(service, route, body, deadline):
    identifier = body['identifier']
    if route in ('preview', 'scan'):
        _fields(body, ('identifier',), ('profile', 'version'))
        data = await asyncio.to_thread(_invoke, service, 'read',
            'inspect' if route == 'preview' else 'scan', identifier, {'version': body.get('version')}, deadline=deadline)
        return JSONResponse(data)
    if route in ('install', 'update', 'uninstall'):
        _fields(body, ('identifier',), ('profile', 'review_token', 'acknowledgements', 'name'))
        # An absent/invalid source never becomes a hub action.
        await asyncio.to_thread(service.resolve, identifier)
        if 'review_token' not in body:
            raise CatalogError('review_required', 'Review this action before confirming.')
        result = await asyncio.to_thread(_invoke, service, 'confirm', identifier,
            route, body['review_token'], body.get('acknowledgements', []), deadline=deadline)
        return JSONResponse(result)
    raise CatalogError('invalid_request', 'Unsupported catalog action.')


async def delegate(request, stock):
    route = request.url.path.rsplit('/', 1)[-1]
    method = {'preview': 'inspect', 'update': 'check', 'updates': 'check',
              'install': 'confirm', 'uninstall': 'confirm', 'sources': 'availability',
              'official': 'availability'}.get(route, route)
    deadline = (time.monotonic() + 5 if route in ('official', 'sources') else
        deadline_for(method if method in ('inspect', 'check', 'confirm', 'review',
        'operation', 'scan', 'search', 'availability') else 'availability'))
    if (request.url.path.startswith('/api/skills/hub/') and request.method == 'GET'
            and len(request.query_params.getlist('profile')) > 1):
        # Resolve ambiguous read scopes before starting independent public search.
        # With no source, the stock handler still owns duplicate-query semantics.
        if await _selected_service(request, None, deadline) is None:
            return await stock(request)
    if request.url.path == '/api/skills/hub/search' and not request.query_params.get('source', '').startswith('plugin.'):
        public = asyncio.create_task(stock(request))
        async def completed(_request):
            return await public
        try:
            return await _delegate(request, completed, deadline)
        except CatalogError as exc:
            if exc.code != 'busy':
                raise
            # Optional discovery cannot discard an independent public result.
            response = await public
            if response.status_code != 200:
                return response
            data = json.loads(response.body)
            data['provider_errors'] = [dict(code=exc.code, message=exc.message)]
            return JSONResponse(data)
        finally:
            if not public.done():
                public.cancel()
                await asyncio.gather(public, return_exceptions=True)
    return await _delegate(request, stock, deadline)


async def _bounded_body(request):
    """Buffer at most 256 KiB for routing; replay received chunks on stock bypass.

    ASGI delivers whole transport-owned chunks. Keep references for replay, but
    never copy or decode an over-limit chunk, nor request the remaining tail.
    """
    from collections import deque
    from starlette.requests import ClientDisconnect
    receive = request.receive
    buffered = bytearray()
    while True:
        message = await receive()
        if message['type'] == 'http.disconnect':
            raise ClientDisconnect()
        chunk = message.get('body', b'')
        if len(buffered) + len(chunk) > 256 * 1024:
            messages = deque((dict(type='http.request', body=bytes(buffered), more_body=True), message))
            async def replay():
                return messages.popleft() if messages else await receive()
            request._receive = replay
            return None
        buffered.extend(chunk)
        if not message.get('more_body', False):
            request._body = bytes(buffered)
            return request._body


async def _delegate(request, stock, provider_deadline):
    route = request.url.path.rsplit('/', 1)[-1]
    if not request.url.path.startswith('/api/skills/hub/'):
        return await stock(request)
    query_profile = request.query_params.get('profile')
    raw = await _bounded_body(request) if request.method == 'POST' else b''
    if raw is None:
        return await stock(request)
    # Keep every selector until optional-provider admission has resolved it.
    # Providerless requests retain the stock last-value/body-profile behavior.
    service = await _selected_service(request, _hint(raw), provider_deadline)
    if service is None:
        return await stock(request)
    if request.method == 'POST':
        body = _body(raw) if raw else {}
        profile = _profile(body, query_profile)
        destructive_profile(profile, 'Skill catalog action')
    else:
        body = dict(request.query_params)
    if route in ('review', 'confirm', 'operation'):
        return await _extension(service, route, body, provider_deadline)
    if route == 'search':
        return await _search(request, stock, service, provider_deadline)
    identifier = body.get('identifier')
    if type(identifier) is str and identifier.startswith('plugin.'):
        return await _provider_action(service, route, body, provider_deadline)
    if route == 'uninstall':
        await asyncio.to_thread(_scoped_managed_name, service, body.get('name'))
    response = await stock(request)
    if response.status_code != 200:
        return response
    data = json.loads(response.body)
    if route in ('official', 'sources'):
        sources = await asyncio.to_thread(_invoke, service, 'sources', deadline=provider_deadline)
        if sources:
            if route == 'official':
                data['provider_sources'] = sources
            else:
                data['sources'].extend(sources)
                try:
                    data['installed'].update(await asyncio.to_thread(_installed_map, service, deadline=provider_deadline))
                except CatalogError as exc:
                    data['provider_errors'] = [dict(code=exc.code, message=exc.message)]
    elif route in ('updates', 'update'):
        updates, errors = await asyncio.to_thread(_updates, service, deadline=provider_deadline)
        if route == 'updates':
            data['updates'].extend(updates)
        else:
            data['provider_review_candidates'] = updates
        if errors:
            data['provider_errors'] = errors
    headers = {key: value for key, value in response.headers.items() if key.lower() != 'content-length'}
    return JSONResponse(data, status_code=response.status_code, headers=headers)


async def _search(request, stock, service, deadline):
    from hermes_cli.skill_catalog.search import start
    params = request.query_params
    source = params.get('source', 'all')
    if source != 'all' and not source.startswith('plugin.'):
        return await stock(request)
    query = params.get('q', '')
    try:
        limit = int(params.get('limit', '20'))
    except ValueError:
        raise CatalogError('invalid_request', 'Invalid catalog search limit.') from None
    provider = asyncio.create_task(start(service, query, source, limit, params.get('cursor'), deadline=deadline))
    if source.startswith('plugin.'):
        result = await provider
        result['installed'] = await asyncio.to_thread(_installed_map, service, deadline=deadline)
        return JSONResponse(result)
    # Public fan-out including its index-miss fallback executes unchanged and
    # concurrently. No provider waiter occupies a public worker or its deadline.
    public = await stock(request)
    try:
        extension = await provider
    except CatalogError as exc:
        extension = dict(results=[], source_counts={}, timed_out=[],
                         provider_errors=[dict(code=exc.code, message=exc.message)], next_cursor=None)
    if public.status_code != 200:
        return public
    data = json.loads(public.body)
    data['results'].extend(extension['results'])
    data['source_counts'].update(extension['source_counts'])
    data['timed_out'].extend(extension['timed_out'])
    if extension['provider_errors']:
        data['provider_errors'] = extension['provider_errors']
    if extension['next_cursor'] is not None or params.get('cursor'):
        data['next_cursor'] = extension['next_cursor']
    return JSONResponse(data)


async def _extension(service, route, body, deadline):
    if route == 'review':
        _fields(body, ('identifier', 'action'), ('version', 'profile'))
        result = await asyncio.to_thread(_invoke, service, 'review', body['identifier'],
            body['action'], body.get('version'), deadline=deadline)
    elif route == 'confirm':
        _fields(body, ('identifier', 'action', 'review_token', 'acknowledgements'), ('profile',))
        result = await asyncio.to_thread(_invoke, service, 'confirm', body['identifier'],
            body['action'], body['review_token'], body['acknowledgements'], deadline=deadline)
    else:
        _fields(body, ('identifier', 'operation_id'), ('profile',))
        result = await asyncio.to_thread(_invoke, service, 'operation', body['identifier'], body['operation_id'], deadline=deadline)
    return JSONResponse(result)


def _installed_map(service, *, deadline=None):
    rows = {}
    deadline = deadline if deadline is not None else time.monotonic() + 3
    for entry in service.entries():
        for record in installed_records(entry, deadline=deadline):
            rows[record['identifier']] = dict(name=record['name'], trust_level='community', scan_verdict='unknown')
    return rows


def _scoped_managed_name(service, name):
    from hermes_cli.plugins_loader import _plugin_home_scope
    with _plugin_home_scope(service.home):
        return _managed_name(service, name)


def _managed_name(service, name):
    from tools.skills_tool import _skill_catalog
    names = {row['load_name']: row['path'].parent.relative_to(service.home / 'skills').as_posix()
             for row in _skill_catalog(skip_disabled=True)
             if row['path'].parent.is_relative_to(service.home / 'skills')}
    for entry in service.entries():
        for record in installed_records(entry):
            if record['target_relative_path'] == names.get(name) and record['managed_state'] in ('installed', 'pending'):
                error = CatalogError('review_required', 'Review this action before confirming.')
                error.payload['error']['identifier'] = record['identifier']
                raise error


def _updates(service, *, deadline=None):
    from hermes_cli.skill_catalog.worker import call
    rows, errors = [], []
    deadline = deadline_for('check', deadline)
    for entry in service.entries():
        cursor, seen = None, set()
        try:
            for _ in range(100):
                page = call(entry, 'check', {'ids': [], 'cursor': cursor}, deadline=deadline)
                rows.extend(row for row in page['items'] if row['state'] == 'update_available')
                cursor = page['next_cursor']
                if cursor is None:
                    break
                if cursor in seen:
                    raise CatalogError('invalid_response', 'Catalog update pagination is incomplete.')
                seen.add(cursor)
            else:
                raise CatalogError('invalid_response', 'Catalog update pagination is incomplete.')
        except CatalogError as exc:
            if exc.code != 'unsupported':
                errors.append(dict(source=entry.source_id, code=exc.code, message=exc.message))
    return rows, errors


async def missing_extension(request: Request):
    """Preserve the authenticated app's pre-extension fallback bytes and headers."""
    if request.method == 'POST':
        raise HTTPException(status_code=405, detail='Method Not Allowed', headers={'Allow': 'GET'})
    return JSONResponse({'detail': f'No such API endpoint: {request.url.path}'}, status_code=404)
