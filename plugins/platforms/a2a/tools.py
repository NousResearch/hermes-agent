"""
A2A client tools — let the Hermes agent talk to *other* agents as a peer.

Tools (registered in the ``a2a`` toolset):
  - a2a_discover(url)         -> fetch + summarize a peer's Agent Card
  - a2a_call(agent, message)  -> send a task to a peer, return its reply
  - a2a_list()                -> list configured peers + persisted conversations
  - a2a_orchestrate(...)      -> fan-out task to multiple peers by capability

Peers are resolved from config.yaml under ``a2a_agents``::

    a2a_agents:
      researcher:
        url: "http://localhost:9999"
        auth: { type: bearer, token: "sk-..." }
        timeout: 120
        capabilities: [web_search, research]

Transport is stdlib urllib (no a2a-sdk dependency). The wire format is the A2A
JSON-RPC ``message/send`` method, so any A2A-compliant peer works.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, TypedDict

from . import protocol, security

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120
_ORCHESTRATE_MAX_WORKERS = 6  # max parallel peers for fan-out


# --------------------------------------------------------------------------
# Peer resolution
# --------------------------------------------------------------------------

def _load_config() -> dict:
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        return {}


def _resolve_peer(agent: str) -> Optional[dict]:
    """Resolve a peer name to {url, auth, timeout, capabilities}, or treat ``agent`` as a URL."""
    if agent.startswith("http://") or agent.startswith("https://"):
        return {"url": agent, "auth": {}, "timeout": _DEFAULT_TIMEOUT, "capabilities": []}
    cfg = _load_config()
    peers = cfg.get("a2a_agents") or {}
    entry = peers.get(agent)
    if not entry:
        return None
    return {
        "url": entry.get("url", ""),
        "auth": entry.get("auth", {}) or {},
        "timeout": int(entry.get("timeout", _DEFAULT_TIMEOUT)),
        "capabilities": entry.get("capabilities", []) or [],
    }


def _auth_header(auth: dict) -> dict:
    if auth and auth.get("type") == "bearer" and auth.get("token"):
        return {"Authorization": f"Bearer {auth['token']}"}
    return {}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _http_get_json(url: str, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (configured peers)
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, body: dict, headers: dict, timeout: int) -> dict:
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **headers}
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (configured peers)
        return json.loads(resp.read().decode("utf-8"))


def _card_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/.well-known/agent.json"


def _rpc_url(base_url: str, card: Optional[dict]) -> str:
    # Prefer the URL the card advertises; fall back to the base.
    if card and isinstance(card.get("url"), str) and card["url"]:
        return card["url"]
    return base_url.rstrip("/")


# --------------------------------------------------------------------------
# Tool handlers
# --------------------------------------------------------------------------

def a2a_discover(args: dict, **_: Any) -> str:
    """Fetch and summarize the Agent Card at ``url``."""
    url = str(args.get("url") or "").strip()
    if not url:
        return "Error: 'url' is required (e.g. http://localhost:9999)."
    try:
        card = _http_get_json(_card_url(url), {}, _DEFAULT_TIMEOUT)
    except urllib.error.HTTPError as e:
        return f"Error: discovery failed — HTTP {e.code} from {url}."
    except Exception as e:
        return f"Error: could not reach {url} — {e}."

    name = card.get("name", "?")
    desc = card.get("description", "")
    caps = card.get("capabilities", {}) or {}
    skills = card.get("skills", []) or []
    auth = "yes" if card.get("security") else "no"
    lines = [
        f"Agent: {name}",
        f"Description: {desc}",
        f"URL: {card.get('url', url)}",
        f"Streaming: {bool(caps.get('streaming'))}  Push: {bool(caps.get('pushNotifications'))}  Auth required: {auth}",
        f"Skills ({len(skills)}):",
    ]
    for s in skills[:20]:
        lines.append(f"  - {s.get('name', s.get('id', '?'))}: {s.get('description', '')}")
    return "\n".join(lines)


def a2a_call(args: dict, **_: Any) -> str:
    """Send a task to a peer agent and return its reply.

    ``agent`` is a configured peer name (from ``a2a_agents``) or a direct URL.
    ``context_id`` continues a prior exchange (multi-turn) when provided.
    """
    # Accept common aliases models reach for (observed live: 'agent_name').
    agent = str(args.get("agent") or args.get("agent_name") or args.get("name") or "").strip()
    message = str(args.get("message") or args.get("text") or args.get("task") or "").strip()
    context_id = str(args.get("context_id") or args.get("contextId") or "").strip()
    if not agent or not message:
        return "Error: both 'agent' and 'message' are required."

    peer = _resolve_peer(agent)
    if not peer or not peer.get("url"):
        return (
            f"Error: unknown agent '{agent}'. Configure it under 'a2a_agents' in "
            f"config.yaml or pass a full http(s):// URL."
        )

    base_url = peer["url"]
    headers = _auth_header(peer["auth"])
    timeout = peer["timeout"]

    # Best-effort card fetch (to learn the rpc URL); non-fatal on failure.
    card = None
    try:
        card = _http_get_json(_card_url(base_url), headers, min(timeout, 30))
    except Exception:
        pass

    ctx = context_id or protocol.new_context_id()
    safe_message = security.redact_outbound(message)
    rpc_body = {
        "jsonrpc": "2.0",
        "id": protocol.new_task_id(),
        "method": "message/send",
        "params": {
            "message": protocol.text_message("user", safe_message),
            "contextId": ctx,  # Always send contextId so server persists under same id
        },
    }
    # Also set contextId inside message for legacy callers
    rpc_body["params"]["message"]["contextId"] = ctx

    security.audit("outbound", agent, rpc_body["id"], safe_message)
    protocol.persist_message(ctx, "user", safe_message, rpc_body["id"])
    protocol.metrics.outbound_total += 1

    try:
        resp = _http_post_json(_rpc_url(base_url, card), rpc_body, headers, timeout)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return f"Error: peer '{agent}' rejected auth (HTTP {e.code}). Check the configured token."
        if e.code == 429:
            return f"Error: peer '{agent}' rate limited us (HTTP 429). Retry later."
        return f"Error: call to '{agent}' failed — HTTP {e.code}."
    except Exception as e:
        return f"Error: call to '{agent}' failed — {e}."

    if "error" in resp:
        err = resp["error"]
        return f"Peer '{agent}' returned an error: {err.get('message', err)}"

    result = resp.get("result", {})
    reply = _reply_text_from_result(result)
    reply_ctx = result.get("contextId", ctx) if isinstance(result, dict) else ctx
    protocol.persist_message(reply_ctx, "agent", reply, rpc_body["id"])
    protocol.metrics.inbound_total += 1

    state = ""
    if isinstance(result, dict):
        state = (result.get("status") or {}).get("state", "")
    header = f"[{agent} · context {reply_ctx}"
    if state:
        header += f" · {state}"
    header += "]"
    return f"{header}\n{reply or '(no text reply)'}"


def _reply_text_from_result(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)
    # Artifacts first (final output), then status message (interim/clarify).
    for artifact in result.get("artifacts", []) or []:
        txt = protocol.extract_text(artifact)
        if txt:
            return txt
    status = result.get("status", {}) or {}
    msg = status.get("message")
    if msg:
        return protocol.extract_text(msg)
    # Bare message result (message/send may return a Message instead of a Task)
    return protocol.extract_text(result)


def a2a_list(args: dict | None = None, **_: Any) -> str:
    """List configured A2A peers and any persisted conversations."""
    cfg = _load_config()
    peers = cfg.get("a2a_agents") or {}
    lines = []
    if peers:
        lines.append(f"Configured peers ({len(peers)}):")
        for name, entry in peers.items():
            auth = (entry.get("auth") or {}).get("type", "none")
            caps = entry.get("capabilities", [])
            cap_str = f" caps: {', '.join(caps)}" if caps else ""
            lines.append(f"  - {name}: {entry.get('url', '?')} (auth: {auth}){cap_str}")
    else:
        lines.append("No peers configured. Add them under 'a2a_agents' in config.yaml.")

    convos = protocol.list_conversations()
    if convos:
        lines.append("")
        lines.append(f"Persisted conversations ({len(convos)}):")
        for c in convos[:25]:
            lines.append(f"  - {c}")

    # Show metrics snapshot
    m = protocol.metrics.snapshot()
    lines.append("")
    lines.append(f"Metrics: {m['inbound_total']} in / {m['outbound_total']} out, "
                 f"{m['tasks_completed']} completed, {m['tasks_failed']} failed, "
                 f"{m['streams_started']} streams, {m['push_sent']} push sent, "
                 f"{m['anti_loop_triggers']} anti-loop, {m['rate_limit_triggers']} rate-limited, "
                 f"avg {m['avg_latency_ms']}ms")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# a2a_orchestrate: capability-based routing with fan-out
# --------------------------------------------------------------------------

def _match_peers_by_capability(capability: str) -> list[tuple[str, dict]]:
    """Find configured peers that advertise the given capability."""
    cfg = _load_config()
    peers = cfg.get("a2a_agents") or {}
    matches = []
    for name, entry in peers.items():
        caps = entry.get("capabilities", []) or []
        if capability in caps or capability == "*":
            matches.append((name, entry))
    return matches


def _call_peer_sync(agent_name: str, peer_entry: dict, message: str, context_id: str = "") -> tuple[str, str]:
    """Call a single peer synchronously. Returns (agent_name, reply_text)."""
    try:
        base_url = peer_entry.get("url", "")
        headers = _auth_header(peer_entry.get("auth", {}))
        timeout = int(peer_entry.get("timeout", _DEFAULT_TIMEOUT))

        card = None
        try:
            card = _http_get_json(_card_url(base_url), headers, min(timeout, 30))
        except Exception:
            pass

        ctx = context_id or protocol.new_context_id()
        safe_message = security.redact_outbound(message)
        rpc_body = {
            "jsonrpc": "2.0",
            "id": protocol.new_task_id(),
            "method": "message/send",
            "params": {
                "message": protocol.text_message("user", safe_message),
                "contextId": ctx,  # Always send contextId so server persists under same id
            },
        }
        # Also set contextId inside message for legacy callers
        rpc_body["params"]["message"]["contextId"] = ctx

        security.audit("outbound", agent_name, rpc_body["id"], safe_message)
        protocol.persist_message(ctx, "user", safe_message, rpc_body["id"])
        protocol.metrics.outbound_total += 1

        resp = _http_post_json(_rpc_url(base_url, card), rpc_body, headers, timeout)
        if "error" in resp:
            err = resp["error"]
            return (agent_name, f"Error: {err.get('message', err)}")

        result = resp.get("result", {})
        reply = _reply_text_from_result(result)
        reply_ctx = result.get("contextId", ctx) if isinstance(result, dict) else ctx
        protocol.persist_message(reply_ctx, "agent", reply, rpc_body["id"])
        protocol.metrics.inbound_total += 1
        return (agent_name, reply or "(no reply)")
    except Exception as e:
        return (agent_name, f"Error: {e}")


def a2a_orchestrate(args: dict, **_: Any) -> str:
    """Fan-out a task to multiple peer agents by capability.

    Modes:
      - ``all``: send to all peers matching the capability, return all replies.
      - ``first``: send to all matching peers, return the first successful reply.
      - ``best``: send to all, return the longest/most detailed reply.

    Configured peers advertise capabilities in config.yaml::

      a2a_agents:
        researcher:
          url: "http://localhost:9991"
          capabilities: [web_search, research]
        coder:
          url: "http://localhost:9992"
          capabilities: [code, debug]
    """
    capability = str(args.get("capability") or "").strip()
    message = str(args.get("message") or args.get("task") or "").strip()
    mode = str(args.get("mode") or "all").strip().lower()
    context_id = str(args.get("context_id") or "").strip()

    if not message:
        return "Error: 'message' is required."
    if not capability:
        return "Error: 'capability' is required (or use '*' for all peers)."

    matches = _match_peers_by_capability(capability)
    if not matches:
        return f"Error: no configured peers advertise capability '{capability}'."

    if mode not in ("all", "first", "best"):
        mode = "all"

    # Fan-out
    results: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(len(matches), _ORCHESTRATE_MAX_WORKERS)) as pool:
        futures = {
            pool.submit(_call_peer_sync, name, entry, message, context_id): name
            for name, entry in matches
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results.append(fut.result())
                if mode == "first" and not results[-1][1].startswith("Error:"):
                    # Got a good reply, cancel remaining
                    for f in futures:
                        f.cancel()
                    break
            except Exception as e:
                results.append((name, f"Error: {e}"))

    # Sort results by peer name for deterministic output
    results.sort(key=lambda r: r[0])

    if mode == "best":
        # Pick the longest non-error reply
        best = max(results, key=lambda r: len(r[1]) if not r[1].startswith("Error:") else 0)
        return f"[best: {best[0]}]\n{best[1]}"
    elif mode == "first":
        # Return the first non-error reply
        for name, reply in results:
            if not reply.startswith("Error:"):
                return f"[first: {name}]\n{reply}"
        # All failed
        lines = ["All peers failed:"]
        for name, reply in results:
            lines.append(f"  {name}: {reply}")
        return "\n".join(lines)
    else:  # mode == "all"
        lines = [f"Orchestrated '{capability}' to {len(matches)} peer(s):"]
        for name, reply in results:
            lines.append(f"\n--- {name} ---")
            lines.append(reply)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Tool schemas + registration
# --------------------------------------------------------------------------

_FunctionSchema = TypedDict("_FunctionSchema", {"name": str, "description": str, "parameters": dict[str, Any]}, total=False)
_ToolSchema = TypedDict("_ToolSchema", {"type": str, "function": _FunctionSchema}, total=False)
_SCHEMAS: dict[str, _ToolSchema] = {
    "a2a_discover": {
        "type": "function",
        "function": {
            "name": "a2a_discover",
            "description": (
                "Fetch and summarize another agent's A2A Agent Card from a URL "
                "(its name, description, capabilities, and skills). Use this to "
                "find out what a remote agent can do before calling it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Base URL of the remote A2A agent, e.g. http://localhost:9999"},
                },
                "required": ["url"],
            },
        },
    },
    "a2a_call": {
        "type": "function",
        "function": {
            "name": "a2a_call",
            "description": (
                "Send a natural-language task to a remote A2A agent and return "
                "its reply. The agent is a peer (any A2A-compliant framework), "
                "not a sub-agent you control. Pass 'context_id' from a previous "
                "reply to continue a multi-turn exchange."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "description": "Configured peer name (from a2a_agents) or a full http(s):// URL."},
                    "message": {"type": "string", "description": "The task / message to send the peer, in natural language."},
                    "context_id": {"type": "string", "description": "Optional: context id from a prior reply, to continue the conversation."},
                },
                "required": ["agent", "message"],
            },
        },
    },
    "a2a_list": {
        "type": "function",
        "function": {
            "name": "a2a_list",
            "description": "List configured A2A peer agents, persisted A2A conversations, and metrics.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "a2a_orchestrate": {
        "type": "function",
        "function": {
            "name": "a2a_orchestrate",
            "description": (
                "Fan-out a task to multiple peer agents by capability. Peers are "
                "matched from config.yaml a2a_agents.*.capabilities. Modes: 'all' "
                "(return all replies), 'first' (first successful), 'best' (longest reply)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "capability": {"type": "string", "description": "Capability to match (e.g. 'research', 'code') or '*' for all peers."},
                    "message": {"type": "string", "description": "The task to send to all matching peers."},
                    "mode": {"type": "string", "enum": ["all", "first", "best"], "description": "How to aggregate results. Default: 'all'."},
                    "context_id": {"type": "string", "description": "Optional: shared context id for all peers."},
                },
                "required": ["capability", "message"],
            },
        },
    },
}

_HANDLERS = {
    "a2a_discover": a2a_discover,
    "a2a_call": a2a_call,
    "a2a_list": a2a_list,
    "a2a_orchestrate": a2a_orchestrate,
}


def register_tools(ctx) -> None:
    """Register the client tools in the ``a2a`` toolset."""
    for name, schema in _SCHEMAS.items():
        ctx.register_tool(
            name=name,
            toolset="a2a",
            schema=schema,
            handler=_HANDLERS[name],
            description=schema["function"]["description"],
            emoji="\U0001f9e9",  # puzzle piece
        )