"""MCP Events deliveries on routes that ``tools/mcp_tool_events.py`` created (the route carries an
``mcp_event`` block). The endpoint-verification echo and the ``terminated`` / ``gap`` control
envelopes are answered here without an agent run; an event occurrence continues through the normal
route pipeline, keyed for idempotency by its ``eventId``. Signatures (Standard Webhooks, ``whsec_``)
are already verified by the adapter before any of this runs."""

import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MAX_CHALLENGE_CHARS = 512
_MAX_NOTICE_DETAIL_CHARS = 200
_MAX_EVENT_DATA_CHARS = 32_000


def _json(body: dict, status: int = 200):
    from aiohttp import web  # lazy: the adapter is only constructed when aiohttp is installed
    return web.json_response(body, status=status)


def _mcp_event(route_config: dict) -> Optional[dict]:
    event = route_config.get("mcp_event")
    return event if isinstance(event, dict) else None


def render_mcp_event_prompt(adapter, route_config: dict, payload: Any, event_type: str,
                            route_name: str) -> Optional[str]:
    """The prompt for an occurrence on an MCP event route whose template uses ``{data}``; None otherwise.
    ``{data}`` takes the whole payload (up to ``_MAX_EVENT_DATA_CHARS``, with a note when cut) instead of
    the 2,000-character cap other placeholders get, and the JSON goes in after the rest of the template
    renders, so text inside the payload is never read as a placeholder."""
    if _mcp_event(route_config) is None or not isinstance(payload, dict):
        return None
    template = route_config.get("prompt")
    if not isinstance(template, str) or "{data}" not in template:
        return None
    data = json.dumps(payload.get("data"), indent=2, ensure_ascii=False)
    if len(data) > _MAX_EVENT_DATA_CHARS:
        data = (f"{data[:_MAX_EVENT_DATA_CHARS]}\n[truncated: the first {_MAX_EVENT_DATA_CHARS:,} of "
                f"{len(data):,} characters]")
    return data.join(adapter._render_prompt(part, payload, event_type, route_name) if part else ""
                     for part in template.split("{data}"))


def mcp_event_delivery_id(route_config: dict, payload: Any) -> Optional[str]:
    """``eventId`` is the sketch's dedup key (stable across retries and redeliveries), unlike the
    per-attempt ``webhook-id``."""
    if _mcp_event(route_config) is None or not isinstance(payload, dict):
        return None
    event_id = payload.get("eventId")
    return f"mcp-event:{event_id}" if isinstance(event_id, str) and event_id else None


def _store_cursor(route_name: str, cursor: str) -> None:
    # Subscription state lives beside the route, never in it: an admitted run's reply is checked against
    # a digest of the route config, so a cursor written there would strand every run still in flight.
    from tools.mcp_tool_events import store_cursor
    store_cursor(route_name, cursor)


def _remove_route(route_name: str) -> None:
    from hermes_cli.webhook import _mutate_subscriptions
    from tools.mcp_tool_events import drop_state
    _mutate_subscriptions(lambda subscriptions: subscriptions.pop(route_name, None))
    drop_state(route_name)


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
        await asyncio.to_thread(_store_cursor, route_name, cursor)
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
    kind = payload.get("type")
    handler = _CONTROL_HANDLERS.get(kind) if isinstance(kind, str) else None
    if handler is not None:
        return await handler(adapter, route_name, route_config, payload, profile)
    if "type" in payload:
        # The sketch: a body with a top-level ``type`` is a control envelope. Acknowledge one this client
        # does not know (a later revision's) so the server stops retrying it; it never starts a run.
        logger.info("[webhook] MCP event route %s: ignoring a control envelope of unknown type %r", route_name, kind)
        return _json({"status": "ignored", "type": kind if isinstance(kind, str) else None})
    cursor = payload.get("cursor")
    if isinstance(cursor, str) and cursor:
        await asyncio.to_thread(_store_cursor, route_name, cursor)
    return None
