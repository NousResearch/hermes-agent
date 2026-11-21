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
    # An emitter the operator lists in mcp_events.trusted_emitters is allowed even on
    # loopback: a webhook secret (which spec-following emitters require) otherwise
    # rules out an MCP server on the same machine.
    sec = security.MCPEventsSecurityContext.capture()
    host = (__import__("urllib.parse", fromlist=["urlparse"]).urlparse(url).hostname or "").lower()
    if host in sec.trusted_emitters or url in sec.trusted_emitters:
        return None
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


def _resolve_target(emitter: str, emitter_url: str) -> tuple[Optional[str], Optional[dict], Optional[str], Optional[str]]:
    """``emitter`` (a name configured under ``mcp_events.emitters:``) wins over a
    raw ``emitter_url``. Named emitters are operator-approved, so the URL guard
    and the loopback refusal do not apply to them — and their credentials never
    leave config/.env, so they never reach the model or the logs. Returns
    ``(url, headers, display_name, error)``."""
    name = (emitter or "").strip()
    if name:
        resolved = security.MCPEventsSecurityContext.capture().resolve_emitter(name)
        if resolved is None:
            known = ", ".join(security.MCPEventsSecurityContext.capture().emitter_names()) or "(none configured)"
            return None, None, None, f"Emitter {name!r} is not configured. Known emitters: {known}."
        url, headers = resolved
        return url, headers, name, None
    url = (emitter_url or "").strip()
    if not url:
        return None, None, None, ("Provide an emitter name (configured under mcp_events.emitters) "
                                  "or an emitter_url.")
    if err := _guard_emitter(url):
        return None, None, None, err
    return url, None, url, None


def _fmt_events(events: list[dict]) -> str:
    if not events:
        return "The emitter advertises no events."
    lines = [f"- {e.get('name')}" + (f": {e.get('description')}" if e.get("description") else "")
             for e in events]
    return "Events advertised by the emitter:\n" + "\n".join(lines)


def mcp_events_list(emitter: str = "", emitter_url: str = "") -> str:
    """List the events an MCP event emitter advertises (``events/list``)."""
    url, headers, _display, err = _resolve_target(emitter, emitter_url)
    if err:
        return err
    if not protocol.server_supports_events(url, headers=headers):
        logger.debug("MCP Events: %s did not advertise the events capability; trying events/list anyway", url)
    try:
        return _fmt_events(protocol.list_events(url, headers=headers))
    except Exception as e:
        return f"events/list failed: {e}"


def mcp_events_subscribe(emitter: str = "", event: str = "", filter_json: str = "",
                         emitter_url: str = "") -> str:
    """Subscribe this Hermes agent to ``event`` on an emitter.

    ``emitter`` is the name of an emitter configured under ``mcp_events.emitters:``
    (preferred — its URL and auth headers resolve from config/.env and never reach
    the model or the logs). ``emitter_url`` subscribes to an unconfigured emitter
    directly. ``filter_json`` (optional) is a JSON object of emitter-side filter
    arguments. Deliveries wake the agent in a per-subscription conversation.
    """
    url, headers, display, err = _resolve_target(emitter, emitter_url)
    if err:
        return err
    if err := _guard_deliverable(url):
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
        record = protocol.subscribe(url, event.strip(), callback_url,
                                    sec.webhook_secret, filter_args=filter_args or None,
                                    headers=headers)
    except Exception as e:
        return f"events/subscribe failed: {e}"
    record["local_id"] = local_id
    if display != url:
        record["emitter_name"] = display
    store.add(record)
    security.audit("subscribe", display, record["id"], f"event {event!r}")
    return (f"Subscribed to {event!r} on {display}.\n"
            f"Subscription id: {record['id']}\n"
            f"Deliveries will wake the agent in conversation 'mcp-events:{record['id']}'." +
            (f"\nExpires: {record['expires_at']}" if record.get("expires_at") else ""))


def mcp_events_unsubscribe(subscription_id: str) -> str:
    """Cancel a subscription by id (see ``mcp_events_subscriptions``)."""
    store = protocol.SubscriptionStore()
    sub = store.resolve(subscription_id)
    if sub is None:
        return f"No subscription {subscription_id!r}."
    # Re-resolve a named emitter's URL and auth headers so rotated credentials in
    # config/.env are picked up; the stored URL is the fallback.
    url, headers = sub["emitter_url"], None
    if sub.get("emitter_name"):
        resolved = security.MCPEventsSecurityContext.capture().resolve_emitter(sub["emitter_name"])
        if resolved is not None:
            url, headers = resolved
    ok = protocol.unsubscribe(url, sub, headers=headers)
    store.remove(sub["id"])
    security.audit("unsubscribe", _display(sub), sub["id"], "cancelled by agent")
    return (f"Unsubscribed {sub['id']!r} ({sub.get('event')!r} on {_display(sub)})."
            + ("" if ok else " The emitter did not acknowledge; the local record is removed."))


def _display(sub: dict) -> str:
    return str(sub.get("emitter_name") or sub.get("emitter_url") or "unknown")


def mcp_events_subscriptions() -> str:
    """List this agent's active MCP event subscriptions."""
    subs = protocol.SubscriptionStore().list()
    if not subs:
        return "No MCP event subscriptions."
    lines = []
    for s in subs:
        line = f"- {s['id']}: {s.get('event')!r} on {_display(s)}"
        if s.get("expires_at"):
            line += f" (expires {s['expires_at']})"
        lines.append(line)
    return "MCP event subscriptions:\n" + "\n".join(lines)


_TOOLS: dict[str, tuple] = {
    "mcp_events_list": (
        mcp_events_list,
        "List the events an MCP event emitter advertises (events/list). Takes emitter (a name configured under mcp_events.emitters) or emitter_url (the emitter's MCP HTTP endpoint).",
        {"emitter": {"type": "string", "description": "Configured emitter name from mcp_events.emitters (preferred)."},
         "emitter_url": {"type": "string", "description": "The emitter's MCP HTTP endpoint URL (for emitters not configured by name)."}},
        [],
    ),
    "mcp_events_subscribe": (
        mcp_events_subscribe,
        "Subscribe this Hermes agent to an event on an MCP event emitter. Signed deliveries wake the agent in a per-subscription conversation. Pass emitter (a configured name) or emitter_url.",
        {"emitter": {"type": "string", "description": "Configured emitter name from mcp_events.emitters (preferred)."},
         "event": {"type": "string", "description": "Event name from mcp_events_list."},
         "filter_json": {"type": "string", "description": "Optional JSON object of emitter-side filter arguments."},
         "emitter_url": {"type": "string", "description": "The emitter's MCP HTTP endpoint URL (for emitters not configured by name)."}},
        ["event"],
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
        # The registry calls handler(args_dict, **context); these handlers take named parameters.
        def dispatch(args, _handler=handler, **_context):
            return _handler(**(args or {}))
        ctx.register_tool(name=name, toolset=_TOOLSET, handler=dispatch, description=description,
                          schema={"name": name, "description": description, "parameters": parameters},
                          emoji="\U0001f4e1", check_fn=_tools_available)  # satellite antenna
