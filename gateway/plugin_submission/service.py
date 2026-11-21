"""Synchronous plugin submission and observation over the existing automation FIFO."""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from gateway.config import Platform
from gateway.delivery import prime_revived_egress
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_authorities import authority_for_home
from gateway.session_authority import admission_row
from gateway.session_automation import _owned_by, _route_owner
from gateway.session_identity import replace_source
from gateway.session_persistence import authority_persistence
from persistence.core.errors import AuthorityRefusal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Request:
    session_key: str
    expected_session_id: str
    request_id: str
    plugin_id: str
    plugin_home: Path
    content: str | None = None

    @property
    def identity(self):
        return json.dumps(['plugin', self.plugin_id, self.session_key, self.request_id], separators=(',', ':'))

    def refused(self, error, *, status='refused'):
        return dict(ok=False, status=status, error=error, request_id=self.request_id,
                    session_id=self.expected_session_id, admission_id=None, outcome=None, result=None)


def call(runner, *, content=None, session_key, expected_session_id, request_id, plugin_id,
         plugin_home, timeout=30):
    """Wait for admission, never execution. Must run outside the gateway event loop."""
    request = Request(session_key, expected_session_id, request_id, plugin_id, plugin_home, content)
    if (any(not isinstance(value, str) or not value.strip() for value in
            (session_key, expected_session_id, request_id, plugin_id))
            or len(request_id) > 256 or plugin_home is None
            or (content is not None and (not isinstance(content, str) or not content.strip()))):
        return request.refused('invalid_params')
    loop = getattr(runner, '_gateway_loop', None)
    if loop is None or loop.is_closed() or not loop.is_running() or not runner._plugin_injection_accepting():
        return request.refused('gateway_unavailable')
    try:
        current = asyncio.get_running_loop()
    except RuntimeError:
        current = None
    if current is loop:
        return request.refused('requires_worker_thread')
    context = getattr(runner, '_plugin_injection_context', None)
    context = contextvars.Context() if context is None else context.copy()
    operation = dispatch(runner, request)
    try:
        future = context.run(asyncio.run_coroutine_threadsafe, operation, loop)
    except RuntimeError:
        operation.close()
        return request.refused('gateway_unavailable')
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        # The ledger write may already have committed. Keep it running; retry the same identity.
        return request.refused('timeout', status='unknown')
    except concurrent.futures.CancelledError:
        return request.refused('gateway_unavailable', status='unknown')
    except Exception:
        logger.exception('Plugin submission observation failed')
        return request.refused('internal_error', status='unknown')


async def dispatch(runner, request):
    try:
        if not runner._plugin_injection_accepting():
            raise AuthorityRefusal('gateway_unavailable')
        authority = authority_for_home(runner, request.plugin_home)
        if authority is None:
            raise AuthorityRefusal('gateway_unavailable')
        async with runner._plugin_injection_scope(None, request.plugin_home):
            return await _dispatch_owned(runner, authority, request)
    except AuthorityRefusal as exc:
        return request.refused(exc.reason)


async def _dispatch_owned(runner, authority, request):
    arrived = authority.route_epoch(request.session_key)

    def check():
        if authority.route_epoch(request.session_key) != arrived:
            raise AuthorityRefusal('admission_conflict')
        if not runner._plugin_injection_accepting():
            raise AuthorityRefusal('gateway_unavailable')

    entry = await runner.async_session_store.lookup_by_session_key(request.session_key)
    if entry is None or entry.origin is None:
        raise AuthorityRefusal('not_found')
    source = replace_source(runner._restored_source(entry))
    if source.platform == Platform.LOCAL:
        live = runner._plugin_local_session(source, request.session_key,
                                            plugin_home=request.plugin_home, plugin_id=request.plugin_id)
        if live is None:
            raise AuthorityRefusal('permission_denied')
        from gateway.session_runtime_workers import track_mutation
        return await asyncio.shield(track_mutation(authority,
            _dispatch_local(runner, authority, request, live, check)))
    else:
        if not runner._plugin_messaging_session_owned(request.session_key,
                plugin_home=request.plugin_home, plugin_id=request.plugin_id):
            raise AuthorityRefusal('permission_denied')
        if not runner._plugin_injection_authorized(source, plugin_id=request.plugin_id,
                                                   target=f'session={request.session_key}'):
            raise AuthorityRefusal('permission_denied')
        adapter = runner._delivery_adapter_for(source)
    return await _dispatch_bound(runner, authority, request, entry, source, adapter, check)


async def _dispatch_local(runner, authority, request, live, check):
    # Reset publishes a new transcript under this gate; keep it until the receipt is committed.
    async with live.mutation_lock:
        entry = await runner.async_session_store.lookup_by_session_key(request.session_key)
        if entry is None:
            raise AuthorityRefusal('not_found')
        return await _dispatch_bound(runner, authority, request, entry, live.source,
                                     runner._adapter_for_source(live.source), check)


async def _dispatch_bound(runner, authority, request, entry, source, adapter, check):
    if adapter is None:
        raise AuthorityRefusal('gateway_unavailable')
    store = authority_persistence(authority)
    if not await asyncio.to_thread(_owned_by, store, request.expected_session_id, entry.session_id):
        raise AuthorityRefusal('admission_conflict')
    owner = _route_owner(authority, entry)
    row = await asyncio.to_thread(store.admissions.find, principal_id='automation:' + request.session_key,
                                  session_id=owner, request_id=request.identity)
    if row is not None:
        check()
        if request.content is not None and row.payload.get('text') != request.content:
            raise AuthorityRefusal('admission_conflict')
        return await _project(authority, request, row)
    if request.content is None:
        raise AuthorityRefusal('not_found')
    if not runner._plugin_injection_accepting():
        raise AuthorityRefusal('gateway_unavailable')
    event = MessageEvent(text=request.content, message_type=MessageType.TEXT, source=source,
                         internal=True, allow_gateway_control=False, message_id=request.identity,
                         metadata={'gateway_session_key': request.session_key,
                                   'gateway_session_id': request.expected_session_id})
    prime_revived_egress(adapter, event)
    try:
        receipt = await authority.admit_automation(adapter, event, request.identity, check=check)
    except AuthorityRefusal as exc:
        if exc.reason != 'admission_conflict':
            raise
        # Another identical request may have committed while the human provenance advanced.
        row = await asyncio.to_thread(store.admissions.find, principal_id='automation:' + request.session_key,
                                      session_id=owner, request_id=request.identity)
        if row is None or row.payload.get('text') != request.content:
            raise
        check()
    else:
        row = await asyncio.to_thread(store.admissions.get, admission_id=receipt.admission_id)
    return await _project(authority, request, row)


async def _project(authority, request, record):
    row = admission_row(record)
    live = authority.sessions.get(record.target_session_id)
    if live is not None:
        with live.event_stream.lock:
            # A committed result is not observable until its completion frame has been published.
            row = live.settling.get(record.admission_id, row)
    receipt = authority._receipt(row)
    result = None
    if receipt.status == 'terminal':
        stored = await asyncio.to_thread(authority_persistence(authority).admissions.settled_result,
                                         admission_id=receipt.admission_id)
        if stored is not None and stored.result is not None:
            text = (stored.result.get('result') or {}).get('final_response') or ''
            result = dict(final_response=text[:16000], truncated=len(text) > 16000)
    return dict(ok=True, status=receipt.status, request_id=request.request_id,
                admission_id=receipt.admission_id, session_id=receipt.ref.session_id,
                outcome=receipt.outcome, result=result)
