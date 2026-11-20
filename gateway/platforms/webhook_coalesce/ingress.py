"""One admission decision per delivery across live webhook route-mode changes."""
import asyncio
import logging
import time

from aiohttp import web

from agent.async_utils import drain_task
from gateway.platforms.webhook_coalesce.config import group_key
from gateway.platforms.webhook_coalesce.recovery import accept_coalesced, prior_coalesced, revision_current
from gateway.session_authority import keyed_turn

logger = logging.getLogger(__name__)


async def dispatch_received(adapter, *, request, route_config, route_name, profile, payload,
                            prompt, event_type, delivery_id, revision):
    from gateway.platforms.webhook import _WebhookDeliveryIdentity
    identity = _WebhookDeliveryIdentity.from_parts(profile, route_name, delivery_id)

    async def admit_once():
        if not revision_current(adapter, route_name, profile, delivery_id, revision):
            return web.json_response({'error': 'Route changed; retry this delivery'}, status=503)
        now = time.time()
        coalesce = route_config.get('coalesce')
        non_durable = bool(route_config.get('deliver_only') or route_config.get('cron_job'))
        key = (group_key(route_name, coalesce['key'], payload, event_type, adapter._render_prompt)
               if isinstance(coalesce, dict) else None)
        if isinstance(coalesce, dict) and key is None and not non_durable:
            logger.warning('[webhook] Coalesce key %r unresolved for route %r; choose a key resolvable for accepted events',
                           coalesce['key'], route_name)
        seen_at = adapter._seen_deliveries.get(identity)
        if ((non_durable or key is not None) and seen_at is not None
                and now - seen_at < adapter._idempotency_ttl):
            return duplicate(delivery_id)
        try:
            seen = await prior_coalesced(adapter._coalescer, profile, route_name, delivery_id)
        except Exception:
            logger.exception('[webhook] Coalesced receipt lookup unavailable')
            return web.json_response({'error': 'Receipt lookup unavailable; retry this delivery'}, status=503)
        if not revision_current(adapter, route_name, profile, delivery_id, revision):
            return web.json_response({'error': 'Route changed; retry this delivery'}, status=503)
        if seen:
            return duplicate(delivery_id)
        if non_durable:
            adapter._record_delivery_id(identity, now)
        if route_config.get('cron_job'):
            return adapter._handle_cron_trigger(prompt, route_config, route_name, event_type, delivery_id, profile)
        if route_config.get('deliver_only'):
            return await adapter._handle_deliver_only(prompt, payload, route_config, route_name,
                                                      event_type, delivery_id, profile)
        if key is not None:
            return await accept_coalesced(adapter._coalescer, route_name=route_name, coalesce=coalesce, key=key,
                payload=payload, event_type=event_type, prompt=prompt, delivery_id=delivery_id, profile=profile,
                revision=revision)
        response = await adapter._dispatch_agent_run(request, route_config, route_name, profile, payload, prompt,
                                                     event_type, delivery_id, now)
        if response.status in (200, 202):
            adapter._record_delivery_id(identity, time.time())
        return response

    async with keyed_turn(adapter._coalescer.delivery_turns, identity):
        if adapter._coalescer._closed:
            return web.json_response({'error': 'Listener closing; retry this delivery'}, status=503)
        task = asyncio.create_task(admit_once())
        adapter._coalescer.intakes.add(task)
        try:
            return await drain_task(task)
        finally:
            adapter._coalescer.intakes.discard(task)


def duplicate(delivery_id):
    return web.json_response({'status': 'duplicate', 'delivery_id': delivery_id}, status=200)
