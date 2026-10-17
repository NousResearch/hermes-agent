"""MCP Events deliveries on routes that ``tools/mcp_tool_events.py`` created (the route carries an
``mcp_event`` block). The endpoint-verification echo and the ``terminated`` / ``gap`` control
envelopes are answered here without an agent run; an event occurrence continues through the normal
route pipeline, keyed for idempotency by its ``eventId``. Signatures (Standard Webhooks, ``whsec_``)
are already verified by the adapter before any of this runs."""

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MAX_CHALLENGE_CHARS = 512
_MAX_NOTICE_DETAIL_CHARS = 200


def _json(body: dict, status: int = 200):
    from aiohttp import web  # lazy: the adapter is only constructed when aiohttp is installed
    return web.json_response(body, status=status)


def _mcp_event(route_config: dict) -> Optional[dict]:
    event = route_config.get("mcp_event")
    return event if isinstance(event, dict) else None


def mcp_event_delivery_id(route_config: dict, payload: Any) -> Optional[str]:
    """``eventId`` is the sketch's dedup key (stable across retries and redeliveries), unlike the
    per-attempt ``webhook-id``."""
    if _mcp_event(route_config) is None or not isinstance(payload, dict):
        return None
    event_id = payload.get("eventId")
    return f"mcp-event:{event_id}" if isinstance(event_id, str) and event_id else None


def _update_event_block(route_name: str, **changes: Any) -> None:
    from hermes_cli.webhook import _mutate_subscriptions

    def mutate(subscriptions: dict) -> None:
        route = subscriptions.get(route_name)
        if isinstance(route, dict) and isinstance(route.get("mcp_event"), dict):
            subscriptions[route_name] = {**route, "mcp_event": {**route["mcp_event"], **changes}}
    _mutate_subscriptions(mutate)


def _remove_route(route_name: str) -> None:
    from hermes_cli.webhook import _mutate_subscriptions
    _mutate_subscriptions(lambda subscriptions: subscriptions.pop(route_name, None))


async def _verification(adapter, route_name: str, route_config: dict, payload: dict, profile) -> Any:
    challenge = payload.get("challenge")
    if not isinstance(challenge, str) or not challenge or len(challenge) > _MAX_CHALLENGE_CHARS:
        return _json({"error": "Invalid verification challenge"}, status=400)
    logger.info("[webhook] MCP event route %s answered the server's verification challenge", route_name)
    return _json({"challenge": challenge})


async def _terminated(adapter, route_name: str, route_config: dict, payload: dict, profile) -> Any:
    event = _mcp_event(route_config) or {}
    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    await asyncio.to_thread(_remove_route, route_name)
    logger.warning("[webhook] MCP server '%s' ended event subscription %s (%s): %s", event.get("server"),
                   route_name, event.get("name"), error)
    if route_config.get("deliver", "log") != "log":
        detail = str(error.get("message") or "no reason given")[:_MAX_NOTICE_DETAIL_CHARS]
        notice = (f"MCP server '{event.get('server')}' ended the subscription to '{event.get('name')}': "
                  f"{detail} (code {error.get('code')}). Subscribe again to keep receiving these events.")
        await adapter._handle_deliver_only(notice, payload, route_config, route_name, "terminated",
                                           f"mcp-terminated:{route_name}", profile)
    return _json({"status": "terminated"})


async def _gap(adapter, route_name: str, route_config: dict, payload: dict, profile) -> Any:
    cursor = payload.get("cursor")
    if isinstance(cursor, str) and cursor:
        await asyncio.to_thread(_update_event_block, route_name, cursor=cursor)
    logger.warning("[webhook] MCP event route %s: the server skipped events (gap); resuming from a fresh cursor",
                   route_name)
    return _json({"status": "gap"})


_CONTROL_HANDLERS = {"verification": _verification, "terminated": _terminated, "gap": _gap}


async def handle_mcp_event_control(adapter, route_name: str, route_config: dict, payload: Any,
                                   profile) -> Any:
    """The response for a control envelope on an MCP event route; None lets the delivery continue
    as an event occurrence (whose replay cursor, when the server sends one, is remembered first)."""
    if _mcp_event(route_config) is None or not isinstance(payload, dict):
        return None
    handler = _CONTROL_HANDLERS.get(payload.get("type"))
    if handler is not None:
        return await handler(adapter, route_name, route_config, payload, profile)
    cursor = payload.get("cursor")
    if isinstance(cursor, str) and cursor and cursor != route_config["mcp_event"].get("cursor"):
        await asyncio.to_thread(_update_event_block, route_name, cursor=cursor)
    return None
