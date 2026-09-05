"""A2A client tools (``a2a`` toolset): a2a_discover/call/list/history/orchestrate talk to *other*
agents. Peers come from config.yaml ``a2a_agents: {name: {url, auth: {type: bearer, token}, timeout,
capabilities}}``. Stdlib urllib; wire format is A2A v1.0 ``SendMessage`` (v0.3 replies still parse)."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from typing import Any, Optional, TypedDict

from gateway.platforms._shared import coerce_port as _coerce_int

from . import protocol, security

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120
_ORCHESTRATE_MAX_WORKERS = 6  # max parallel peers for fan-out


class _LocalWaitTimeout(TimeoutError):
    """A caller-owned wait budget expired before the detached task finished."""


_WAIT_DEADLINE: ContextVar[float | None] = ContextVar(
    "a2a_wait_deadline", default=None
)


def _request_timeout(default: float, deadline: float | None = None) -> float:
    """Return one request's remaining budget, or raise before an over-budget call."""
    if deadline is None:
        return max(0.1, float(default))
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _LocalWaitTimeout
    # Do not pad a short remaining budget: urllib must not receive the full
    # poll budget again after an Agent Card lookup or 404 fallback.
    return min(max(0.001, float(default)), remaining)


def _load_config() -> dict:
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        return {}


def _configured_peers() -> dict:
    return _load_config().get("a2a_agents") or {}


def _peer_from_entry(entry: dict, **extra: Any) -> dict:
    return {"url": entry.get("url", ""), "auth": entry.get("auth", {}) or {},
            "timeout": int(entry.get("timeout", _DEFAULT_TIMEOUT)), **extra}


def _resolve_peer(agent: str) -> Optional[dict]:
    """Peer name -> {url, auth, timeout, capabilities, tenant}, or treat ``agent`` as a URL."""
    if agent.startswith(("http://", "https://")):
        return {"url": agent, "auth": {}, "timeout": _DEFAULT_TIMEOUT, "capabilities": []}
    entry = _configured_peers().get(agent)
    return _peer_from_entry(entry, capabilities=entry.get("capabilities", []) or [], tenant=entry.get("tenant", "")) if entry else None


def _auth_header(auth: dict) -> dict:
    return {"Authorization": f"Bearer {auth['token']}"} if auth and auth.get("type") == "bearer" and auth.get("token") else {}


def _http_json(url: str, headers: dict, timeout: int, method: str, data: Optional[bytes] = None) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (configured peers)
        return json.loads(resp.read().decode("utf-8"))


def _http_get_json(url: str, headers: dict, timeout: int) -> dict:
    return _http_json(url, headers, timeout, "GET")


def _http_post_json(url: str, body: dict, headers: dict, timeout: int) -> dict:
    hdrs = {"Content-Type": "application/json", "A2A-Version": protocol.PROTOCOL_VERSION, **headers}
    return _http_json(url, hdrs, timeout, "POST", json.dumps(body).encode("utf-8"))


def _fetch_card(
    base_url: str,
    headers: dict,
    timeout: float,
    *,
    deadline: float | None = None,
) -> dict:
    """GET the v1.0 card, falling back to v0.2 without resetting *deadline*."""
    base = base_url.rstrip("/")

    def get(url: str) -> dict:
        return _http_get_json(url, headers, _request_timeout(timeout, deadline))

    try:
        return get(base + "/.well-known/agent-card.json")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    return get(base + "/.well-known/agent.json")


def _select_jsonrpc_interface(card: Optional[dict]) -> Optional[dict]:
    if isinstance(card, dict):
        for iface in card.get("supportedInterfaces", []) or []:
            if isinstance(iface, dict) and iface.get("protocolBinding") == "JSONRPC" and iface.get("url"):
                return iface
    return None


def _rpc_url(base_url: str, card: Optional[dict]) -> str:
    """Card's JSONRPC interface (v1.0 supportedInterfaces) > card's legacy top-level url > base."""
    if iface := _select_jsonrpc_interface(card):
        return str(iface["url"])
    if isinstance(card, dict) and isinstance(card.get("url"), str) and card["url"]:
        return card["url"]
    return base_url.rstrip("/")


def _interface_tenant(card: Optional[dict], peer: dict) -> str:
    iface = _select_jsonrpc_interface(card)
    if iface and iface.get("tenant"):
        return str(iface["tenant"])
    return str(peer.get("tenant") or "")


def _short_state(state: str) -> str:
    """TASK_STATE_COMPLETED -> completed (also passes through v0.3 states)."""
    return state.replace("TASK_STATE_", "").replace("_", "-").lower() if state else ""


class _TaskResult(TypedDict):
    reply: str
    context_id: str
    state: str
    task_id: str


def _send_task_result(
    agent_label: str,
    peer: dict,
    message: str,
    context_id: str,
    *,
    return_immediately: bool = False,
) -> _TaskResult:
    """Send one task and retain the protocol task id for later retrieval."""
    base_url = peer.get("url", "")
    headers = _auth_header(peer.get("auth", {}) or {})
    timeout = int(peer.get("timeout", _DEFAULT_TIMEOUT))

    # Best-effort card fetch (to learn the rpc URL); non-fatal on failure.
    card = None
    try:
        card = _fetch_card(base_url, headers, min(timeout, 30))
    except Exception:
        pass

    ctx = context_id or protocol.new_context_id()
    safe_message = security.redact_outbound(message)
    # v1.0: contextId lives inside the Message, not at the params top level.
    rpc_body = {
        "jsonrpc": "2.0",
        "id": protocol.new_task_id(),
        "method": "SendMessage",
        "params": {
            "message": protocol.text_message(protocol.ROLE_USER, safe_message, context_id=ctx),
        },
    }
    if return_immediately:
        rpc_body["params"]["configuration"] = {"returnImmediately": True}

    tenant = _interface_tenant(card, peer)
    if tenant:
        rpc_body["params"]["tenant"] = tenant

    security.audit("outbound", agent_label, rpc_body["id"], safe_message)
    protocol.persist_message(ctx, "user", safe_message, rpc_body["id"])
    protocol.metrics.outbound_total += 1

    resp = _http_post_json(_rpc_url(base_url, card), rpc_body, headers, timeout)
    if "error" in resp:
        err = resp["error"]
        raise ValueError(f"Peer '{agent_label}' returned an error: {err.get('message', err)}")

    result = resp.get("result", {})
    payload = protocol.unwrap_send_message_response(result)
    reply = _reply_text_from_result(payload)
    reply_ctx, state, task_id = ctx, "", ""
    if isinstance(payload, dict):
        reply_ctx = str(payload.get("contextId") or ctx)
        state = str((payload.get("status") or {}).get("state") or "")
        task_id = str(payload.get("id") or payload.get("taskId") or "")
    if reply:
        protocol.persist_message(reply_ctx, "agent", reply, task_id or rpc_body["id"])
    protocol.metrics.inbound_total += 1
    return {
        "reply": reply,
        "context_id": reply_ctx,
        "state": state,
        "task_id": task_id,
    }


def _get_task_result(
    agent: str,
    peer: dict,
    task_id: str,
    request_timeout: float | None = None,
) -> _TaskResult:
    """Issue one GetTask request and return a structured task snapshot."""
    base_url = peer.get("url", "")
    headers = _auth_header(peer.get("auth", {}) or {})
    timeout = float(peer.get("timeout", _DEFAULT_TIMEOUT))
    deadline = _WAIT_DEADLINE.get()
    if deadline is None and request_timeout is not None:
        deadline = time.monotonic() + max(0.0, request_timeout)
    card = None
    try:
        card = _fetch_card(
            base_url,
            headers,
            min(timeout, 30),
            deadline=deadline,
        )
    except _LocalWaitTimeout:
        raise
    except Exception:
        pass

    rpc_body = {
        "jsonrpc": "2.0",
        "id": protocol.new_task_id(),
        "method": "GetTask",
        "params": {"id": task_id},
    }
    tenant = _interface_tenant(card, peer)
    if tenant:
        rpc_body["params"]["tenant"] = tenant
    response = _http_post_json(
        _rpc_url(base_url, card),
        rpc_body,
        headers,
        _request_timeout(timeout, deadline),
    )
    if "error" in response:
        error = response["error"]
        raise ValueError(
            f"Peer '{agent}' returned an error: {error.get('message', error)}"
        )
    payload = protocol.unwrap_send_message_response(response.get("result", {}))
    if not isinstance(payload, dict):
        raise ValueError(f"Peer '{agent}' returned an invalid task response.")

    protocol.metrics.inbound_total += 1
    return {
        "reply": _reply_text_from_result(payload),
        "context_id": str(payload.get("contextId") or ""),
        "state": str((payload.get("status") or {}).get("state") or ""),
        "task_id": str(payload.get("id") or payload.get("taskId") or task_id),
    }


def _task_lookup_error(agent: str, error: Exception) -> str:
    """Format task lookup failures without exposing remote credentials."""
    if isinstance(error, urllib.error.HTTPError):
        if error.code in (401, 403):
            return f"Error: peer '{agent}' rejected auth (HTTP {error.code}). Check the configured token."
        if error.code == 429:
            return f"Error: peer '{agent}' rate limited us (HTTP 429). Retry later."
        return f"Error: task lookup on '{agent}' failed — HTTP {error.code}."
    if isinstance(error, ValueError):
        return str(error)
    return f"Error: task lookup on '{agent}' failed — {error}."


def _format_task_result(agent: str, task: _TaskResult) -> str:
    """Render one task snapshot consistently for get and wait tools."""
    header = f"[{agent} · task {task['task_id']}"
    if task["context_id"]:
        header += f" · context {task['context_id']}"
    if task["state"]:
        header += f" · {_short_state(task['state'])}"
    header += "]"
    return f"{header}\n{task['reply'] or '(task has no final text yet)'}"


def _format_local_wait_timeout(agent: str, task: _TaskResult) -> str:
    """Render a local timeout without implying cancellation of the remote task."""
    return (
        f"{_format_task_result(agent, task)}\n\n"
        "Local wait timed out; the remote task was not canceled. "
        "Call a2a_get_task or a2a_wait again with the same task id."
    )


def _persist_task_reply(task: _TaskResult) -> None:
    """Persist a reply once by its JSONL identity, not a bounded history tail."""
    if task["reply"]:
        protocol.persist_message_once(
            task["context_id"],
            "agent",
            task["reply"],
            task["task_id"],
        )


def _task_lookup_args(args: dict) -> tuple[str, str, dict | None, str | None]:
    """Resolve common task lookup arguments and return any validation error."""
    agent = str(args.get("agent") or args.get("agent_name") or args.get("name") or "").strip()
    task_id = str(args.get("task_id") or args.get("taskId") or args.get("id") or "").strip()
    if not agent or not task_id:
        return agent, task_id, None, "Error: both 'agent' and 'task_id' are required."
    peer = _resolve_peer(agent)
    if not peer or not peer.get("url"):
        error = (
            f"Error: unknown agent '{agent}'. Configure it under 'a2a_agents' in "
            f"config.yaml or pass a full http(s):// URL."
        )
        return agent, task_id, None, error
    return agent, task_id, peer, None


def a2a_get_task(args: dict, **_: Any) -> str:
    """Fetch the latest state and output of a previously submitted task."""
    agent, task_id, peer, error = _task_lookup_args(args)
    if error:
        return error
    assert peer is not None
    try:
        task = _get_task_result(agent, peer, task_id)
    except Exception as exc:
        return _task_lookup_error(agent, exc)
    _persist_task_reply(task)
    return _format_task_result(agent, task)


def a2a_wait(args: dict, **_: Any) -> str:
    """Poll a detached task until it reaches a caller-visible stopping state."""
    agent, task_id, peer, error = _task_lookup_args(args)
    if error:
        return error
    assert peer is not None
    try:
        timeout = max(0.0, min(float(args.get("timeout", 300)), 86400.0))
    except (TypeError, ValueError):
        timeout = 300.0
    try:
        poll_interval = max(
            0.1,
            min(float(args.get("poll_interval", 2)), 30.0),
        )
    except (TypeError, ValueError):
        poll_interval = 2.0

    deadline = time.monotonic() + timeout
    last_task: _TaskResult = {
        "reply": "",
        "context_id": "",
        "state": "",
        "task_id": task_id,
    }
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        token = _WAIT_DEADLINE.set(deadline)
        try:
            task = _get_task_result(
                agent,
                peer,
                task_id,
                request_timeout=remaining,
            )
        except _LocalWaitTimeout:
            return _format_local_wait_timeout(agent, last_task)
        except Exception as exc:
            if time.monotonic() >= deadline:
                return _format_local_wait_timeout(agent, last_task)
            return _task_lookup_error(agent, exc)
        finally:
            _WAIT_DEADLINE.reset(token)

        last_task = task
        if (
            task["state"] in protocol.TERMINAL_STATES
            or task["state"] == protocol.STATE_INPUT_REQUIRED
        ):
            _persist_task_reply(task)
            return _format_task_result(agent, task)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _format_local_wait_timeout(agent, task)
        time.sleep(min(poll_interval, remaining))


def _send_task(agent_label: str, peer: dict, message: str, context_id: str) -> tuple[str, str, str]:
    """Compatibility wrapper returning (reply_text, context_id, state)."""
    result = _send_task_result(agent_label, peer, message, context_id)
    return result["reply"], result["context_id"], result["state"]


def _reply_text_from_result(result: Any) -> str:
    result = protocol.unwrap_send_message_response(result)
    if not isinstance(result, dict):
        return str(result)
    # Artifacts first (final output), then status message (interim/clarify), else bare Message.
    for artifact in result.get("artifacts", []) or []:
        txt = protocol.extract_text(artifact)
        if txt:
            return txt
    return protocol.extract_text((result.get("status", {}) or {}).get("message") or result)


_AUTH_ERR = "Error: peer '{agent}' rejected auth (HTTP {code}). Check the configured token."
_HTTP_CALL_ERRORS = {401: _AUTH_ERR, 403: _AUTH_ERR, 429: "Error: peer '{agent}' rate limited us (HTTP 429). Retry later."}

def a2a_discover(args: dict, **_: Any) -> str:
    """Fetch and summarize the Agent Card at ``url``."""
    url = str(args.get("url") or "").strip()
    if not url:
        return "Error: 'url' is required (e.g. http://localhost:9999)."
    try:
        card = _fetch_card(url, {}, _DEFAULT_TIMEOUT)
    except urllib.error.HTTPError as e:
        return f"Error: discovery failed — HTTP {e.code} from {url}."
    except Exception as e:
        return f"Error: could not reach {url} — {e}."
    caps = card.get("capabilities", {}) or {}
    skills = card.get("skills", []) or []
    auth = "yes" if card.get("security") else "no"
    proto = ", ".join(
        f"{i.get('protocolBinding', '?')} v{i.get('protocolVersion', '?')}"
        for i in (card.get("supportedInterfaces", []) or []) if isinstance(i, dict)
    ) or f"v{card.get('protocolVersion', '?')} (pre-1.0 card)"
    lines = [f"Agent: {card.get('name', '?')}", f"Description: {card.get('description', '')}", f"URL: {_rpc_url(url, card)}",
             f"Protocol: {proto}",
             f"Streaming: {bool(caps.get('streaming'))}  Push: {bool(caps.get('pushNotifications'))}  Auth required: {auth}",
             f"Skills ({len(skills)}):"]
    lines.extend(f"  - {s.get('name', s.get('id', '?'))}: {s.get('description', '')}" for s in skills[:20])
    return "\n".join(lines)


def a2a_call(args: dict, **_: Any) -> str:
    """Send a task to a peer, optionally returning before it completes."""
    # Accept common aliases models reach for (observed live: 'agent_name').
    agent = str(args.get("agent") or args.get("agent_name") or args.get("name") or "").strip()
    message = str(args.get("message") or args.get("text") or args.get("task") or "").strip()
    context_id = str(args.get("context_id") or args.get("contextId") or "").strip()
    wait = args.get("wait") is not False
    if not agent or not message:
        return "Error: both 'agent' and 'message' are required."

    peer = _resolve_peer(agent)
    if not peer or not peer.get("url"):
        return (
            f"Error: unknown agent '{agent}'. Configure it under 'a2a_agents' in "
            f"config.yaml or pass a full http(s):// URL."
        )

    try:
        task = _send_task_result(
            agent,
            peer,
            message,
            context_id,
            return_immediately=not wait,
        )
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return f"Error: peer '{agent}' rejected auth (HTTP {e.code}). Check the configured token."
        if e.code == 429:
            return f"Error: peer '{agent}' rate limited us (HTTP 429). Retry later."
        return f"Error: call to '{agent}' failed — HTTP {e.code}."
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error: call to '{agent}' failed — {e}."

    reply = task["reply"]
    reply_ctx = task["context_id"]
    state = task["state"]
    task_id = task["task_id"]
    if not wait and not task_id:
        return f"Error: peer '{agent}' did not return a task id for the non-blocking call."

    header = f"[{agent}"
    if task_id:
        header += f" · task {task_id}"
    header += f" · context {reply_ctx}"
    if state:
        header += f" · {_short_state(state)}"
    header += "]"
    body = reply or "(no text reply)"
    if not wait and task_id and not reply:
        body = (
            "Task accepted. Retrieve it with a2a_get_task using "
            f"agent '{agent}' and task_id '{task_id}'."
        )
    if state == protocol.STATE_INPUT_REQUIRED:
        body += (
            "\n\n(The peer needs more input — answer by calling a2a_call again "
            f"with context_id '{reply_ctx}'.)"
        )
    return f"{header}\n{body}"


def a2a_list(args: dict | None = None, **_: Any) -> str:
    """List configured A2A peers, persisted conversations, and metrics."""
    peers = _configured_peers()
    lines = []
    if peers:
        lines.append(f"Configured peers ({len(peers)}):")
        for name, entry in peers.items():
            caps = entry.get("capabilities", [])
            lines.append(f"  - {name}: {entry.get('url', '?')} (auth: {(entry.get('auth') or {}).get('type', 'none')})"
                         + (f" caps: {', '.join(caps)}" if caps else ""))
    else:
        lines.append("No peers configured. Add them under 'a2a_agents' in config.yaml.")
    if convos := protocol.list_conversations():
        lines.append("")
        lines.append(f"Persisted conversations ({len(convos)}) — recall with a2a_history:")
        lines.extend(f"  - {c}" for c in convos[:25])
    m = protocol.metrics.snapshot()
    lines.append("")
    lines.append(f"Metrics: {m['inbound_total']} in / {m['outbound_total']} out, {m['tasks_completed']} completed, "
                 f"{m['tasks_failed']} failed, {m['streams_started']} streams, {m['push_sent']} push sent, "
                 f"{m['anti_loop_triggers']} anti-loop, {m['rate_limit_triggers']} rate-limited, avg {m['avg_latency_ms']}ms")
    return "\n".join(lines)


def a2a_history(args: dict, **_: Any) -> str:
    """Recall a persisted A2A conversation (survives compaction/restarts)."""
    context_id = str(args.get("context_id") or args.get("contextId") or "").strip()
    if not context_id:
        return "Error: 'context_id' is required (see a2a_list for known conversations)."
    limit = max(1, min(_coerce_int(args.get("limit") or 50, 50), 200))
    messages = protocol.load_conversation(context_id, limit=limit)
    if not messages:
        return f"No persisted conversation for context '{context_id}'."
    lines = [f"Conversation {context_id} (last {len(messages)} messages):"]
    for m in messages:
        text = (m.get("text") or "").strip()
        lines.append(f"[{m.get('role', '?')}] {text[:1000] + ' …[truncated]' if len(text) > 1000 else text}")
    return "\n".join(lines)


def _match_peers_by_capability(capability: str) -> list[tuple[str, dict]]:
    """Configured peers that advertise the capability ('*' matches all)."""
    return [(name, entry) for name, entry in _configured_peers().items()
            if capability in (entry.get("capabilities", []) or []) or capability == "*"]


def _call_peer_sync(agent_name: str, peer_entry: dict, message: str, context_id: str = "") -> tuple[str, str]:
    """Call a single peer synchronously -> (agent_name, reply_text)."""
    try:
        reply, _ctx, _state = _send_task(agent_name, _peer_from_entry(peer_entry), message, context_id)
        return (agent_name, reply or "(no reply)")
    except Exception as e:
        return (agent_name, f"Error: {e}")


def a2a_orchestrate(args: dict, **_: Any) -> str:
    """Fan-out a task to peers matching a capability. Modes: ``all``, ``first`` (first successful),
    ``best`` (longest successful — coarse; use ``all`` to judge yourself)."""
    capability = str(args.get("capability") or "").strip()
    message = str(args.get("message") or args.get("task") or "").strip()
    mode = str(args.get("mode") or "all").strip().lower()
    mode = mode if mode in ("all", "first", "best") else "all"
    context_id = str(args.get("context_id") or "").strip()
    if not message:
        return "Error: 'message' is required."
    if not capability:
        return "Error: 'capability' is required (or use '*' for all peers)."
    if not (matches := _match_peers_by_capability(capability)):
        return f"Error: no configured peers advertise capability '{capability}'."
    results: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(len(matches), _ORCHESTRATE_MAX_WORKERS)) as pool:
        futures = {pool.submit(_call_peer_sync, name, entry, message, context_id): name for name, entry in matches}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results.append(fut.result())
                if mode == "first" and not results[-1][1].startswith("Error:"):
                    for f in futures:  # good reply; cancel peers that haven't started
                        f.cancel()
                    break
            except Exception as e:
                results.append((name, f"Error: {e}"))
    results.sort(key=lambda r: r[0])  # deterministic output
    successes = [(name, reply) for name, reply in results if not reply.startswith("Error:")]
    if mode in ("best", "first"):
        if not successes:
            return "\n".join(["All peers failed:"] + [f"  {name}: {reply}" for name, reply in results])
        name, reply = max(successes, key=lambda r: len(r[1])) if mode == "best" else successes[0]
        return f"[{mode}: {name}]\n{reply}"
    return "\n".join([f"Orchestrated '{capability}' to {len(matches)} peer(s):"]
                     + [line for name, reply in results for line in (f"\n--- {name} ---", reply)])


def _str(description: str) -> dict:
    return {"type": "string", "description": description}


# name -> (handler, description, properties, required)
_TOOLS: dict[str, tuple[Any, str, dict, list[str]]] = {
    "a2a_discover": (a2a_discover,
                     "Fetch and summarize another agent's A2A Agent Card from a URL (its name, description, "
                     "capabilities, and skills). Use this to find out what a remote agent can do before calling it.",
                     {"url": _str("Base URL of the remote A2A agent, e.g. http://localhost:9999")}, ["url"]),
    "a2a_call": (a2a_call,
                 "Send a natural-language task to a remote A2A agent and return its reply. The agent is a peer "
                 "(any A2A-compliant framework), not a sub-agent you control. Pass 'context_id' from a previous "
                 "reply to continue a multi-turn exchange.",
                 {"agent": _str("Configured peer name (from a2a_agents) or a full http(s):// URL."),
                  "message": _str("The task / message to send the peer, in natural language."),
                  "context_id": _str("Optional: context id from a prior reply, to continue the conversation.")},
                 ["agent", "message"]),
    "a2a_list": (a2a_list, "List configured A2A peer agents, persisted A2A conversations, and metrics.", {}, []),
    "a2a_history": (a2a_history,
                    "Recall a persisted A2A conversation transcript by context_id (survives restarts and "
                    "context compaction). Use a2a_list to see known context ids.",
                    {"context_id": _str("Context id of the conversation to recall."),
                     "limit": {"type": "integer", "description": "Max messages to return (default 50, max 200)."}},
                    ["context_id"]),
    "a2a_orchestrate": (a2a_orchestrate,
                        "Fan-out a task to multiple peer agents by capability. Peers are matched from config.yaml "
                        "a2a_agents.*.capabilities. Modes: 'all' (return all replies), 'first' (first successful), "
                        "'best' (longest successful reply).",
                        {"capability": _str("Capability to match (e.g. 'research', 'code') or '*' for all peers."),
                         "message": _str("The task to send to all matching peers."),
                         "mode": {"type": "string", "enum": ["all", "first", "best"], "description": "How to aggregate results. Default: 'all'."},
                         "context_id": _str("Optional: shared context id for all peers.")},
                        ["capability", "message"]),
}


# Local detached-task extensions, on the upstream registration table.
_TOOLS["a2a_call"][2]["wait"] = {'type': 'boolean', 'description': 'Wait for completion (default true). Set false to return a task id immediately.'}
_TOOLS['a2a_get_task'] = (a2a_get_task, 'Fetch the current state and final output of a task returned by a non-blocking a2a_call.', {'agent': {'type': 'string', 'description': 'Configured peer name or a full http(s):// URL.'}, 'task_id': {'type': 'string', 'description': 'Task id returned by a2a_call(wait=false).'}}, ['agent', 'task_id'])
_TOOLS['a2a_wait'] = (a2a_wait, 'Wait for a detached A2A task to finish by polling GetTask. A local timeout never cancels the remote task.', {'agent': {'type': 'string', 'description': 'Configured peer name or a full http(s):// URL.'}, 'task_id': {'type': 'string', 'description': 'Task id returned by a2a_call(wait=false).'}, 'timeout': {'type': 'number', 'description': 'Maximum local wait in seconds (default 300, max 86400).'}, 'poll_interval': {'type': 'number', 'description': 'Seconds between GetTask polls (default 2).'}}, ['agent', 'task_id'])

def _a2a_tools_available() -> bool:
    """check_fn: serve the client tools ONLY when the operator opted into A2A (peers under
    ``a2a_agents``, inbound platform enabled, or A2A_PORT set). Fail closed.

    Maintainer-directed (#95681): these registered unconditionally, so every session on every install paid
    ~561 tok/call for tools whose only possible output without config is 'no peers configured'. A2A is
    unrelated to Bot Mode (bots talk over gateway RPCs) — for most installs this toolset is foreign-agent
    plumbing they never enabled. Config adds mid-session surface at the next compaction (#97073).
    """
    cfg = {}
    with contextlib.suppress(Exception):
        cfg = _load_config()
        if cfg.get("a2a_agents"):
            return True
    try:
        if os.getenv("A2A_PORT"):
            return True
        a2a_cfg = (cfg.get("platforms") or {}).get("a2a") or {}
        return bool(isinstance(a2a_cfg, dict) and a2a_cfg.get("enabled"))
    except Exception:  # noqa: BLE001
        return False


def register_tools(ctx) -> None:
    """Register the client tools in the ``a2a`` toolset (config-gated)."""
    for name, (handler, description, properties, required) in _TOOLS.items():
        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        ctx.register_tool(name=name, toolset="a2a", handler=handler, description=description,
                          schema={"name": name, "description": description, "parameters": parameters},
                          emoji="\U0001f9e9", check_fn=_a2a_tools_available)  # puzzle piece


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import TypedDict  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
