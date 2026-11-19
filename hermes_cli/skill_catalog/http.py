"""Optional HTTP delegation before legacy request-model validation.

The original route handler remains authoritative when no catalog is registered,
including validation errors and background hub action serialization. Provider
references never reach that handler while this profile has live catalogs.
"""
import asyncio
import json
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse

from hermes_constants import get_hermes_home
from hermes_cli.skill_catalog.installed import installed_records
from hermes_cli.skill_catalog.service import Service
from hermes_cli.skill_catalog.validation import CatalogError
from hermes_cli.web_routers._common import destructive_profile
from hermes_cli.web_server_profiles import _config_profile_scope


class CatalogRoute(APIRoute):
    def get_route_handler(self):
        stock = super().get_route_handler()

        async def handle(request):
            try:
                return await delegate(request, stock)
            except CatalogError as exc:
                return JSONResponse(exc.payload, status_code=exc.status)
        return handle


def _service(profile):
    with _config_profile_scope(profile):
        service = Service(get_hermes_home())
        return service if service.entries() else None


def _body(raw):
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeError):
        raise CatalogError('invalid_request', 'Invalid catalog request.') from None
    if type(body) is not dict:
        raise CatalogError('invalid_request', 'Invalid catalog request.')
    return body


def _fields(body, required, optional=()):
    if not set(required) <= body.keys() or body.keys() - set(required) - set(optional):
        raise CatalogError('invalid_request', 'Invalid catalog request fields.')


def _profile(body, query):
    selected = body.get('profile')
    if selected is not None and type(selected) is not str:
        raise CatalogError('invalid_request', 'Invalid profile.')
    if selected is not None and query is not None and selected != query:
        raise CatalogError('invalid_request', 'Conflicting catalog profiles.')
    return selected or query


def _invoke(service, method, *args):
    # Hydration belongs to this worker, never the ASGI/gateway event loop.
    from hermes_cli.plugins_loader import _plugin_home_scope
    with _plugin_home_scope(Path(service.home)):
        return getattr(service, method)(*args)


async def _provider_action(service, route, body):
    identifier = body['identifier']
    if route in ('preview', 'scan'):
        _fields(body, ('identifier',), ('profile', 'version'))
        data = await asyncio.to_thread(_invoke, service, 'read',
            'inspect' if route == 'preview' else 'scan', identifier, {'version': body.get('version')})
        return JSONResponse(data)
    if route in ('install', 'update', 'uninstall'):
        _fields(body, ('identifier',), ('profile', 'review_token', 'acknowledgements', 'name'))
        # An absent/invalid source never becomes a hub action.
        await asyncio.to_thread(service.resolve, identifier)
        if 'review_token' not in body:
            raise CatalogError('review_required', 'Review this action before confirming.')
        result = await asyncio.to_thread(_invoke, service, 'confirm', identifier,
            route, body['review_token'], body.get('acknowledgements', []))
        return JSONResponse(result)
    raise CatalogError('invalid_request', 'Unsupported catalog action.')


async def delegate(request, stock):
    route = request.url.path.rsplit('/', 1)[-1]
    if not request.url.path.startswith('/api/skills/hub/'):
        return await stock(request)
    query_profile = request.query_params.get('profile')
    raw = await request.body() if request.method == 'POST' else b''
    # Read only enough body state to select the old body-profile winner. Invalid
    # bodies still get the untouched stock validator if there is no live source.
    try:
        hint = json.loads(raw) if raw else {}
    except (ValueError, UnicodeError):
        hint = {}
    body_profile = hint.get('profile') if type(hint) is dict else None
    if body_profile is not None and query_profile is not None and body_profile != query_profile:
        if await asyncio.to_thread(_service, query_profile) is not None:
            raise CatalogError('invalid_request', 'Conflicting catalog profiles.')
    lookup_profile = body_profile if type(body_profile) is str and body_profile else query_profile
    service = await asyncio.to_thread(_service, lookup_profile)
    if service is None:
        return await stock(request)
    if request.method == 'POST':
        body = _body(raw) if raw else {}
        profile = _profile(body, query_profile)
        destructive_profile(profile, 'Skill catalog action')
    else:
        body = dict(request.query_params)
    if route in ('review', 'confirm', 'operation'):
        return await _extension(service, route, body)
    if route == 'search':
        return await _search(request, stock, service)
    identifier = body.get('identifier')
    if type(identifier) is str and identifier.startswith('plugin.'):
        return await _provider_action(service, route, body)
    if route == 'uninstall':
        await asyncio.to_thread(_scoped_managed_name, service, body.get('name'))
    response = await stock(request)
    if response.status_code != 200:
        return response
    data = json.loads(response.body)
    if route in ('official', 'sources'):
        sources = await asyncio.to_thread(_invoke, service, 'sources')
        if sources:
            if route == 'official':
                data['provider_sources'] = sources
            else:
                data['sources'].extend(sources)
                data['installed'].update(await asyncio.to_thread(_installed_map, service))
    elif route in ('updates', 'update'):
        updates, errors = await asyncio.to_thread(_updates, service)
        if route == 'updates':
            data['updates'].extend(updates)
        else:
            data['provider_review_candidates'] = updates
        if errors:
            data['provider_errors'] = errors
    headers = {key: value for key, value in response.headers.items() if key.lower() != 'content-length'}
    return JSONResponse(data, status_code=response.status_code, headers=headers)


async def _search(request, stock, service):
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
    provider = asyncio.create_task(start(service, query, source, limit, params.get('cursor')))
    if source.startswith('plugin.'):
        result = await provider
        result['installed'] = await asyncio.to_thread(_installed_map, service)
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


async def _extension(service, route, body):
    if route == 'review':
        _fields(body, ('identifier', 'action'), ('version', 'profile'))
        result = await asyncio.to_thread(_invoke, service, 'review', body['identifier'],
            body['action'], body.get('version'))
    elif route == 'confirm':
        _fields(body, ('identifier', 'action', 'review_token', 'acknowledgements'), ('profile',))
        result = await asyncio.to_thread(_invoke, service, 'confirm', body['identifier'],
            body['action'], body['review_token'], body['acknowledgements'])
    else:
        _fields(body, ('identifier', 'operation_id'), ('profile',))
        result = await asyncio.to_thread(_invoke, service, 'operation', body['identifier'], body['operation_id'])
    return JSONResponse(result)


def _installed_map(service):
    rows = {}
    for entry in service.entries():
        for record in installed_records(entry):
            rows[record['identifier']] = record['name']
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


def _updates(service):
    import time
    from hermes_cli.skill_catalog.worker import call
    rows, errors = [], []
    deadline = time.monotonic() + 15
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
    """Unregistered extension fallback; disposition of stock 405 is tracked."""
    raise HTTPException(status_code=404, detail='Not Found')
