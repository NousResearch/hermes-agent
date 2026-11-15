"""Durable debounce of signed webhook events, handed to the existing producer ledger."""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, UTC

from agent.async_utils import consume_detached_task_result, drain_task, drain_thread
from gateway.platforms.webhook_coalesce import (
    DEFAULT_MAX_WAIT_SECONDS, DEFAULT_WINDOW_SECONDS, group_key, prompt_with_note,
)
from gateway.platforms.webhook_coalesce_binding import Binding, binding_from, ensure_store_identity, expected_binding
from gateway.platforms.webhook_coalesce_store import CoalesceStore
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
RETRY_SECONDS = 1.0
MAX_HANDOFFS = 8


class DurableWebhookCoalescer:
    def __init__(self, adapter):
        self.adapter = adapter
        self.store = CoalesceStore(get_hermes_home())
        self._changed = asyncio.Event()
        self._worker = None
        self._inflight = {}
        self.delivery_turns = {}
        self.intakes = set()
        self._closed = False

    async def binding(self, route_name, profile, delivery_id, payload):
        from gateway.platforms.event import MessageEvent, MessageType
        from gateway.platforms.webhook_delivery import delivery_info, route_digest, snapshot_destination
        from gateway.platforms.webhook_ingress import producer_scope

        self.adapter._reload_dynamic_routes()
        source = self.adapter._admission_source(route_name, profile, delivery_id)
        event = MessageEvent(text='', message_type=MessageType.TEXT, source=source, raw_message=payload,
                             message_id=delivery_id, timestamp=datetime.fromtimestamp(0, UTC))
        async with producer_scope(self.adapter, event) as authority:
            store_id = await drain_thread(ensure_store_identity, authority.db)
            digest = route_digest(self.adapter, source.chat_id)
            route = self.adapter._routes[route_name]
            destination = snapshot_destination(self.adapter, delivery_info(self.adapter, route, profile, payload, route_name))
            return binding_from(self.adapter, authority, store_id, digest, destination)

    async def start(self, *, recover=True):
        self._closed = False
        if recover:
            self._ensure_worker()

    def _ensure_worker(self):
        if not self._closed and (self._worker is None or self._worker.done()):
            self._worker = asyncio.create_task(self._recover())

    async def enqueue(self, *, route_name, coalesce, payload, event_type, prompt, delivery_id, profile, revision):
        if self._closed:
            return 'unavailable'
        key = group_key(route_name, coalesce['key'], payload, event_type, self.adapter._render_prompt)
        if key is None:
            return None
        from gateway.platforms.webhook import _WebhookDeliveryIdentity
        identity = _WebhookDeliveryIdentity.from_parts(profile, route_name, delivery_id)
        bound = await self.binding(route_name, profile, delivery_id, payload)
        if (self._closed or (bound.route, bound.connector,
                self.adapter._admission_source(route_name, profile, delivery_id).profile) != revision):
            return 'unavailable'
        work = dict(payload=payload, prompt=prompt, event_type=event_type, delivery_id=delivery_id,
                    profile=profile, route_name=route_name, binding=bound.to_wire())
        member = identity.session_chat_id
        group = json.dumps([bound.profile_id, identity.profile, key])
        self._ensure_worker()
        try:
            return await drain_thread(self.store.accept, member, group, work,
                                      float(coalesce.get('window_seconds', DEFAULT_WINDOW_SECONDS)),
                                      float(coalesce.get('max_wait_seconds', DEFAULT_MAX_WAIT_SECONDS)))
        finally:
            self._changed.set()

    async def _recover(self):
        while not self._closed:
            self._changed.clear()
            try:
                delay = await self._schedule()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('[webhook] Coalescing recovery unavailable; pending work retained')
                delay = RETRY_SECONDS
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=delay)
            except TimeoutError:
                continue

    async def _schedule(self):
        delay = None
        for row in await drain_thread(self.store.schedule):
            if row['id'] in self._inflight:
                continue
            wait = row['due'] - time.time()
            if wait <= 0 and len(self._inflight) < MAX_HANDOFFS:
                self._launch(row['id'])
            elif wait > 0:
                delay = wait if delay is None else min(delay, wait)
        return delay

    def _launch(self, batch_id, *, flush=False):
        task = asyncio.create_task(self._handoff(batch_id, flush=flush))
        self._inflight[batch_id] = task

        def settled(done):
            consume_detached_task_result(done)
            self._inflight.pop(batch_id, None)
            self._changed.set()

        task.add_done_callback(settled)
        return task

    async def _handoff(self, batch_id, *, flush=False):
        try:
            row = await drain_thread(self.store.claim, batch_id, flush)
            if row is None:
                return
            await self._admit(row)
        except asyncio.CancelledError:
            raise
        except Exception:
            if await drain_thread(self.store.defer, batch_id, RETRY_SECONDS, 'admission_unavailable'):
                logger.warning('[webhook] Coalesced batch held; admission unavailable', exc_info=True)

    async def _admit(self, row):
        work = json.loads(row['work'])
        bound = await self.binding(work['route_name'], work['profile'], work['delivery_id'], work['payload'])
        route = self.adapter._routes.get(work['route_name'])
        if (bound != Binding.from_wire(work['binding']) or not route or not route.get('enabled', True)
                or not isinstance(route.get('coalesce'), dict)):
            if await drain_thread(self.store.defer, row['id'], RETRY_SECONDS, 'binding_changed'):
                logger.warning('[webhook] Coalesced batch held; route/profile binding changed')
            return
        token = expected_binding.set(bound)
        try:
            outcome = await self.adapter._admit_agent_run(
                work['payload'], prompt_with_note(work['prompt'], row['count']), work['delivery_id'], time.time(),
                route_config=route, route_name=work['route_name'], profile=work['profile'], event_type=work['event_type'])
        finally:
            expected_binding.reset(token)
        if outcome is None:
            await drain_thread(self.store.defer, row['id'], RETRY_SECONDS, 'admission_unavailable')
        else:
            await drain_thread(self.store.finish, row['id'])

    async def flush(self):
        self._closed = True
        owner = asyncio.create_task(self._flush_closed())
        cancelled = None
        while True:
            try:
                await asyncio.shield(owner)
                break
            except asyncio.CancelledError as exc:
                if owner.cancelled():
                    raise
                if cancelled is None:
                    cancelled = exc
                    asyncio.get_running_loop().call_soon(owner.cancel)
            except Exception:
                if cancelled is not None:
                    raise cancelled
                raise
        if cancelled is not None:
            raise cancelled

    async def _flush_closed(self):
        try:
            if self.intakes:
                await drain_task(asyncio.gather(*self.intakes, return_exceptions=True))
            if self._worker is not None:
                self._worker.cancel()
                await asyncio.gather(self._worker, return_exceptions=True)
            if not self.store.path.exists():
                return
            try:
                rows = await drain_thread(self.store.schedule)
            except Exception:
                logger.exception('[webhook] Flush storage unavailable; pending work retained')
                return
            for row in rows:
                if row['id'] not in self._inflight and len(self._inflight) < MAX_HANDOFFS:
                    self._launch(row['id'], flush=True)
            tasks = list(self._inflight.values())
            if tasks:
                await asyncio.wait(tasks, timeout=2)
        finally:
            tasks = list(self._inflight.values())
            if self._worker is not None:
                tasks.append(self._worker)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


def ingress_revision(adapter, route_name, route, profile, delivery_id):
    from gateway.platforms.webhook_delivery import route_config_digest
    return (route_config_digest(route), hashlib.sha256(adapter.token.encode()).hexdigest(),
            adapter._admission_source(route_name, profile, delivery_id).profile)


async def prior_coalesced(coalescer, profile, route_name, delivery_id):
    if not coalescer.store.path.exists():
        return False
    from gateway.platforms.webhook import _WebhookDeliveryIdentity
    identity = _WebhookDeliveryIdentity.from_parts(profile, route_name, delivery_id)
    return await drain_thread(coalescer.store.seen, identity.session_chat_id)


async def accept_coalesced(coalescer, **kwargs):
    from aiohttp import web
    try:
        outcome = await coalescer.enqueue(**kwargs)
    except Exception:
        logger.exception('[webhook] Could not persist coalesced delivery')
        outcome = 'unavailable'
    if outcome is None:
        return None
    if outcome not in {'coalesced', 'duplicate'}:
        return web.json_response({'error': 'Coalescing unavailable; retry this delivery'}, status=503)
    return web.json_response({'status': outcome, 'route': kwargs['route_name'],
                              'event': kwargs['event_type'], 'delivery_id': kwargs['delivery_id']},
                             status=200 if outcome == 'duplicate' else 202)
