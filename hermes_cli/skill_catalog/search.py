"""Provider search has separate bounded orchestration and scoped opaque cursors."""
import asyncio
import base64
import hashlib
import hmac
import json
import threading
import time

from agent.memory_provider import spawn_context_thread
from hermes_cli.skill_catalog.validation import CatalogError, _pattern, _SOURCE
from hermes_cli.skill_catalog.worker import call

_lock = threading.Lock()
_active: set[str] = set()


def _scope(service, query, source, limit):
    from hermes_cli.web_server import _SESSION_TOKEN
    value = [str(service.home), query, source, limit,
             [(entry.source_id, entry.generation) for entry in service.entries()]]
    return hmac.new(_SESSION_TOKEN.encode(), json.dumps(value).encode(), hashlib.sha256).hexdigest()


def _cursor_signature(payload):
    from hermes_cli.web_server import _SESSION_TOKEN
    return hmac.new(_SESSION_TOKEN.encode(), json.dumps(payload, sort_keys=True,
        separators=(',', ':')).encode(), hashlib.sha256).hexdigest()


def _decode(cursor, scope):
    if cursor is None:
        return {}
    try:
        if type(cursor) is not str or len(cursor) > 4096:
            raise ValueError
        _pattern(cursor, r'[A-Za-z0-9_-]+')
        payload = json.loads(base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)))
        if type(payload) is not dict or set(payload) != {'scope', 'sources', 'signature'}:
            raise ValueError
        signature = payload.pop('signature')
        if (type(signature) is not str or not hmac.compare_digest(signature, _cursor_signature(payload))
                or payload['scope'] != scope):
            raise ValueError
        sources = payload['sources']
        if type(sources) is not dict or len(sources) > 16:
            raise ValueError
        for source, value in sources.items():
            _pattern(source, _SOURCE)
            _pattern(value, r'[A-Za-z0-9_-]{16,256}')
        return sources
    except (ValueError, TypeError, UnicodeError):
        raise CatalogError('invalid_request', 'Catalog cursor does not match this search.') from None


def _encode(sources, scope):
    if not sources:
        return None
    payload = dict(scope=scope, sources=sources)
    payload['signature'] = _cursor_signature(payload)
    result = base64.urlsafe_b64encode(json.dumps(payload,
        separators=(',', ':')).encode()).decode().rstrip('=')
    if len(result) > 4096:
        raise CatalogError('invalid_response', 'Catalog cursor is too large.')
    return result


def search(service, query, source, limit, cursor, deadline):
    from hermes_cli.skill_catalog.validation import validate_request
    validate_request('search', {'query': query, 'limit': limit, 'cursor': None})
    scope = _scope(service, query, source, limit)
    cursors = _decode(cursor, scope)
    entries = tuple(entry for entry in service.entries() if source in ('all', entry.source_id))
    if len(entries) > 16:
        raise CatalogError('busy', 'Select a catalog source before searching.')
    if source.startswith('plugin.') and not entries:
        raise CatalogError('not_found', 'Catalog source is not registered.')
    if cursor is not None:
        entries = tuple(entry for entry in entries if entry.source_id in cursors)
    rows, counts, errors, timed_out, next_cursors = [], {}, [], [], {}
    seen = set()
    for entry in entries:
        try:
            availability = call(entry, 'availability', {}, deadline=deadline)
            if availability['state'] == 'hidden':
                continue
            if availability['state'] != 'ready':
                code = 'sign_in_required' if availability['state'] == 'sign_in_required' else 'provider_unavailable'
                raise CatalogError(code, availability['message'])
            page = call(entry, 'search', {'query': query, 'limit': limit,
                        'cursor': cursors.get(entry.source_id)}, deadline=deadline)
            counts[entry.source_id] = len(page['items'])
            for row in page['items']:
                if row['identifier'] not in seen:
                    rows.append(row)
                    seen.add(row['identifier'])
            if page['next_cursor'] is not None:
                next_cursors[entry.source_id] = page['next_cursor']
        except CatalogError as exc:
            errors.append(dict(source=entry.source_id, code=exc.code, message=exc.message))
            if exc.code == 'timeout':
                timed_out.append(entry.source_id)
    return dict(results=rows, source_counts=counts, timed_out=timed_out,
                provider_errors=errors, next_cursor=_encode(next_cursors, scope))


async def start(service, query, source, limit, cursor, *, deadline=None):
    """One waiting orchestrator per profile, never a public-search executor slot."""
    end = min(deadline, time.monotonic() + 5) if deadline is not None else time.monotonic() + 5
    key = str(service.home)
    with _lock:
        if key in _active:
            raise CatalogError('busy', 'Catalog search is busy.')
        _active.add(key)
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    def finish(result, error):
        if not future.done():
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(result)
    def run():
        result, error = None, None
        try:
            result = search(service, query, source, limit, cursor, end)
        except CatalogError as exc:
            error = exc
        finally:
            with _lock:
                _active.discard(key)
            if not loop.is_closed():
                loop.call_soon_threadsafe(finish, result, error)
    thread = spawn_context_thread(run, name='skill-catalog-search')
    try:
        thread.start()
    except RuntimeError:
        with _lock:
            _active.discard(key)
        raise CatalogError('provider_error', 'Catalog search could not start.') from None
    return await future
