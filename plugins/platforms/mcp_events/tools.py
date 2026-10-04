"""MCP Events client tools (the ``mcp_events`` toolset): let the agent discover an
emitter's event catalog and manage its own subscriptions. Deliveries arrive on
the platform adapter's webhook endpoint; these tools only manage the
subscription relationship. Config-gated: hidden until the receiver is configured."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Optional

from . import protocol, security

logger = logging.getLogger(__name__)

_TOOLSET = "mcp_events"


def _tools_available() -> bool:
    """The toolset appears once the receiver is configured (secret set or the
    ``mcp_events:`` config section present) — subscribing with nowhere for
    deliveries to land is a silent misconfiguration."""
    try:
        sec = security.MCPEventsSecurityContext.capture()
        return bool(sec.webhook_secret or sec.public_base_url)
    except Exception:
        return False


def _callback_base() -> str:
    sec = security.MCPEventsSecurityContext.capture()
    if sec.public_base_url:
        return sec.public_base_url
    return f"http://{sec.resolve_bind_host()}:{sec.port}"


def _guard_emitter(url: str) -> Optional[str]:
    """Returns an error string, or None when the emitter URL is acceptable."""
    if not security.is_safe_emitter_url(url):
        return (f"Refusing emitter URL {url!r}: not a public http(s) address. "
                "Subscribe to public emitters, or run the receiver in localhost mode for local development.")
    return None


def _guard_deliverable(url: str) -> Optional[str]:
    """Fail fast when a remote emitter could never deliver to us."""
    sec = security.MCPEventsSecurityContext.capture()
    if sec.localhost_only():
        host = __import__("urllib.parse", fromlist=["urlparse"]).urlparse(url).hostname or ""
        try:
            from agent.proxy_bypass import is_loopback_host
            remote = not is_loopback_host(host) and host != "localhost"
        except Exception:
            remote = host not in {"127.0.0.1", "localhost", "::1"}
        if remote:
            return ("The webhook receiver is localhost-only (no MCP_EVENTS_WEBHOOK_SECRET set), "
                    "so a remote emitter could never deliver. Configure the secret and "
                    "mcp_events.public_base_url first, then subscribe.")
    return None


def _fmt_events(events: list[dict]) -> str:
    if not events:
        return "The emitter advertises no events."
    lines = [f"- {e.get('name')}" + (f": {e.get('description')}" if e.get("description") else "")
             for e in events]
    return "Events advertised by the emitter:\n" + "\n".join(lines)


def mcp_events_list(emitter_url: str) -> str:
    """List the events an MCP event emitter advertises (``events/list``)."""
    if err := _guard_emitter(emitter_url):
        return err
    if not protocol.server_supports_events(emitter_url):
        logger.debug("MCP Events: %s did not advertise the events capability; trying events/list anyway", emitter_url)
    try:
        return _fmt_events(protocol.list_events(emitter_url))
    except Exception as e:
        return f"events/list failed: {e}"


def mcp_events_subscribe(emitter_url: str, event: str, filter_json: str = "") -> str:
    """Subscribe this Hermes agent to ``event`` on ``emitter_url``.

    ``filter_json`` (optional) is a JSON object of emitter-side filter arguments.
    Deliveries wake the agent in a per-subscription conversation.
    """
    if err := _guard_emitter(emitter_url):
        return err
    if err := _guard_deliverable(emitter_url):
        return err
    if not (event or "").strip():
        return "An event name is required."
    filter_args: dict = {}
    if filter_json.strip():
        try:
            filter_args = json.loads(filter_json)
        except Exception:
            return "filter_json is not valid JSON."
        if not isinstance(filter_args, dict):
            return "filter_json must be a JSON object."
    sec = security.MCPEventsSecurityContext.capture()
    store = protocol.SubscriptionStore()
    local_id = uuid.uuid4().hex[:16]
    callback_url = f"{_callback_base()}/mcp/events/webhook/{local_id}"
    try:
        record = protocol.subscribe(emitter_url, event.strip(), callback_url,
                                    sec.webhook_secret, filter_args=filter_args or None)
    except Exception as e:
        return f"events/subscribe failed: {e}"
    record["local_id"] = local_id
    store.add(record)
    security.audit("subscribe", emitter_url, record["id"], f"event {event!r}")
    return (f"Subscribed to {event!r} on {emitter_url}.\n"
            f"Subscription id: {record['id']}\n"
            f"Deliveries will wake the agent in conversation 'mcp-events:{record['id']}'." +
            (f"\nExpires: {record['expires_at']}" if record.get("expires_at") else ""))


def mcp_events_unsubscribe(subscription_id: str) -> str:
    """Cancel a subscription by id (see ``mcp_events_subscriptions``)."""
    store = protocol.SubscriptionStore()
    sub = store.resolve(subscription_id)
    if sub is None:
        return f"No subscription {subscription_id!r}."
    ok = protocol.unsubscribe(sub["emitter_url"], sub["id"])
    store.remove(sub["id"])
    security.audit("unsubscribe", sub["emitter_url"], sub["id"], "cancelled by agent")
    return (f"Unsubscribed {sub['id']!r} ({sub.get('event')!r} on {sub['emitter_url']})."
            + ("" if ok else " The emitter did not acknowledge; the local record is removed."))


def mcp_events_subscriptions() -> str:
    """List this agent's active MCP event subscriptions."""
    subs = protocol.SubscriptionStore().list()
    if not subs:
        return "No MCP event subscriptions."
    lines = []
    for s in subs:
        line = f"- {s['id']}: {s.get('event')!r} on {s.get('emitter_url')}"
        if s.get("expires_at"):
            line += f" (expires {s['expires_at']})"
        lines.append(line)
    return "MCP event subscriptions:\n" + "\n".join(lines)


_TOOLS: dict[str, tuple] = {
    "mcp_events_list": (
        mcp_events_list,
        "List the events an MCP event emitter advertises (events/list). Takes emitter_url (the emitter's MCP HTTP endpoint).",
        {"emitter_url": {"type": "string", "description": "The emitter's MCP HTTP endpoint URL."}},
        ["emitter_url"],
    ),
    "mcp_events_subscribe": (
        mcp_events_subscribe,
        "Subscribe this Hermes agent to an event on an MCP event emitter. Signed deliveries wake the agent in a per-subscription conversation.",
        {"emitter_url": {"type": "string", "description": "The emitter's MCP HTTP endpoint URL."},
         "event": {"type": "string", "description": "Event name from mcp_events_list."},
         "filter_json": {"type": "string", "description": "Optional JSON object of emitter-side filter arguments."}},
        ["emitter_url", "event"],
    ),
    "mcp_events_unsubscribe": (
        mcp_events_unsubscribe,
        "Cancel an MCP event subscription by id.",
        {"subscription_id": {"type": "string", "description": "Subscription id from mcp_events_subscriptions."}},
        ["subscription_id"],
    ),
    "mcp_events_subscriptions": (
        mcp_events_subscriptions,
        "List this agent's active MCP event subscriptions.",
        {},
        [],
    ),
}


def register_tools(ctx) -> None:
    """Register the client tools in the ``mcp_events`` toolset (config-gated)."""
    for name, (handler, description, properties, required) in _TOOLS.items():
        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        ctx.register_tool(name=name, toolset=_TOOLSET, handler=handler, description=description,
                          schema={"name": name, "description": description, "parameters": parameters},
                          emoji="\U0001f4e1", check_fn=_tools_available)  # satellite antenna
