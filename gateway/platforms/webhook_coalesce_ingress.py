"""One admission decision per delivery across live webhook route-mode changes."""
import asyncio
import logging
import time

from aiohttp import web

from agent.async_utils import drain_task
from gateway.platforms.webhook_coalesce_recovery import accept_coalesced, ingress_revision, prior_coalesced
from gateway.session_authority import keyed_turn

logger = logging.getLogger(__name__)


async def dispatch_received(adapter, *, request, route_config, route_name, profile, payload,
                            prompt, event_type, delivery_id, revision):
    from gateway.platforms.webhook import _WebhookDeliveryIdentity
    identity = _WebhookDeliveryIdentity.from_parts(profile, route_name, delivery_id)

    async def admit_once():
        adapter._reload_dynamic_routes()
        current = adapter._routes.get(route_name)
        if (adapter._coalescer._closed or not current or not current.get('enabled', True)
                or ingress_revision(adapter, route_name, current, profile, delivery_id) != revision):
            return web.json_response({'error': 'Route changed; retry this delivery'}, status=503)
        now = time.time()
        coalesce = route_config.get('coalesce')
        non_durable = bool(route_config.get('deliver_only') or route_config.get('cron_job'))
        # Ordinary same-mode retries still use the producer's immutable-envelope check.
        seen_at = adapter._seen_deliveries.get(identity)
        if ((non_durable or isinstance(coalesce, dict)) and seen_at is not None
                and now - seen_at < adapter._idempotency_ttl):
            return duplicate(delivery_id)
        try:
            if await prior_coalesced(adapter._coalescer, profile, route_name, delivery_id):
                return duplicate(delivery_id)
        except Exception:
            logger.exception('[webhook] Coalesced receipt lookup unavailable')
            return web.json_response({'error': 'Receipt lookup unavailable; retry this delivery'}, status=503)
        if non_durable:
            adapter._record_delivery_id(identity, now)
        if route_config.get('cron_job'):
            return adapter._handle_cron_trigger(prompt, route_config, route_name, event_type, delivery_id, profile)
        if route_config.get('deliver_only'):
            return await adapter._handle_deliver_only(prompt, payload, route_config, route_name,
                                                      event_type, delivery_id, profile)
        if isinstance(coalesce, dict):
            response = await accept_coalesced(adapter._coalescer, route_name=route_name, coalesce=coalesce,
                payload=payload, event_type=event_type, prompt=prompt, delivery_id=delivery_id, profile=profile,
                revision=revision)
            if response is not None:
                return response
        response = await adapter._dispatch_agent_run(request, route_config, route_name, profile, payload, prompt,
                                                     event_type, delivery_id, now)
        if response.status in (200, 202):
            adapter._record_delivery_id(identity, time.time())
        return response

    async with keyed_turn(adapter._coalescer.delivery_turns, identity):
        # A disconnected HTTP waiter cannot release this key ahead of its physical commit.
        return await drain_task(asyncio.create_task(admit_once()))


def duplicate(delivery_id):
    return web.json_response({'status': 'duplicate', 'delivery_id': delivery_id}, status=200)
