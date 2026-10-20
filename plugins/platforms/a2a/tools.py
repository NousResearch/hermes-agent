"""A2A client tools (``a2a`` toolset): a2a_discover/call/list/history/orchestrate talk to *other*
agents. Peers come from config.yaml ``a2a_agents: {name: {url, auth: {type: bearer, token}, timeout,
capabilities}}``. Stdlib urllib; wire format is A2A v1.0 ``SendMessage`` (v0.3 replies still parse)."""

from __future__ import annotations

import contextlib
import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from gateway.platforms._shared import coerce_port as _coerce_int, get_scoped_secret as _get_scoped_secret

from . import protocol, security

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120
_ORCHESTRATE_MAX_WORKERS = 6  # max parallel peers for fan-out
_RETRYABLE_POLL_HTTP_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

# Deterministic clock and sleep seams for testing
_sleep = time.sleep
_monotonic = time.monotonic


def _task_poll_error(agent: str, context_id: str, task_id: str, reason: str) -> str:
    context = f" · context {context_id}" if context_id else ""
    return (f"[{agent}{context} · task {task_id} · unknown]\n"
            f"Could not verify task status ({reason}); it may still be running. Do not resubmit it. "
            f"Resume with a2a_call using agent='{agent}' and task_id='{task_id}'.")


def _poll_task_until_terminal(agent: str, peer: dict, reply: str, reply_ctx: str, state: str,
                              task_id: str, start_time: float, total_timeout: int) -> tuple[str, str, str, str]:
    backoff_schedule = (1.0, 2.0, 3.0, 5.0, 8.0, 10.0)
    poll_idx = 0
    last_poll_error = ""
    while (_monotonic() - start_time) < total_timeout:
        delay = backoff_schedule[min(poll_idx, len(backoff_schedule) - 1)]
        remaining = total_timeout - (_monotonic() - start_time)
        if remaining <= 0:
            break
        _sleep(min(delay, remaining))
        poll_idx += 1
        try:
            reply, reply_ctx, state, _ = _send_task(agent, peer, "", reply_ctx, task_id=task_id)
            last_poll_error = ""
            if state not in (protocol.STATE_WORKING, protocol.STATE_SUBMITTED, "working", "submitted"):
                break
        except urllib.error.HTTPError as poll_err:
            if poll_err.code not in _RETRYABLE_POLL_HTTP_CODES:
                return reply, reply_ctx, state, _task_poll_error(agent, reply_ctx, task_id, f"HTTP {poll_err.code}")
            last_poll_error = f"HTTP {poll_err.code}"
            logger.debug("A2A: transient HTTP %s while polling task %s", poll_err.code, task_id)
        except ValueError:
            return reply, reply_ctx, state, _task_poll_error(agent, reply_ctx, task_id, "invalid peer response")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as poll_err:
            last_poll_error = f"transport error ({type(poll_err).__name__})"
            logger.debug("A2A: transient poll transport error for task %s (%s)", task_id, type(poll_err).__name__)
    error = _task_poll_error(agent, reply_ctx, task_id, last_poll_error) if last_poll_error else ""
    return reply, reply_ctx, state, error


def _load_config() -> dict:
    """Read-only view of config.yaml; peers are only read, never mutated (cache-safe)."""
    from hermes_cli.config import load_config_readonly
    return load_config_readonly() or {}


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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_json(url: str, headers: dict, timeout: int) -> dict:
    return _http_json(url, headers, timeout, "GET")


def _http_post_json(url: str, body: dict, headers: dict, timeout: int) -> dict:
    hdrs = {"Content-Type": "application/json", "A2A-Version": protocol.PROTOCOL_VERSION, **headers}
    return _http_json(url, hdrs, timeout, "POST", json.dumps(body).encode("utf-8"))


def _fetch_card(base_url: str, headers: dict, timeout: int) -> dict:
    """GET the v1.0 agent-card.json; on 404 fall back to the v0.2 agent.json alias."""
    base = base_url.rstrip("/")
    try:
        return _http_get_json(base + "/.well-known/agent-card.json", headers, timeout)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    return _http_get_json(base + "/.well-known/agent.json", headers, timeout)


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


def _send_task(agent_label: str, peer: dict, message: str, context_id: str,
               task_id: str = "") -> tuple[str, str, str, str]:
    """Send or poll an A2A task -> (reply_text, context_id, state, task_id).

    When ``task_id`` is supplied, queries GetTask directly. When omitted, sends
    SendMessage with ``configuration.returnImmediately: True`` so long tasks
    never block a client thread up to an arbitrary socket drop.
    """
    base_url = peer.get("url", "")
    headers = _auth_header(peer.get("auth", {}) or {})
    timeout = int(peer.get("timeout", _DEFAULT_TIMEOUT))
    try:
        card = _fetch_card(base_url, headers, min(timeout, 30))
    except Exception:
        card = None
    rpc_url = _rpc_url(base_url, card)
    iface = _select_jsonrpc_interface(card)
    tenant = str(iface["tenant"]) if iface and iface.get("tenant") else str(peer.get("tenant") or "")

    def _persist_client_receipt(ctx: str, rep: str, st: str, tid: str) -> None:
        if not rep or not tid:
            return
        st_norm = st.upper()
        is_terminal = (st_norm in protocol.TERMINAL_STATES
                       or st.lower() in ("completed", "failed", "canceled", "rejected")
                       or st == ""
                       or st_norm == protocol.STATE_INPUT_REQUIRED)
        if is_terminal:
            existing = protocol.load_conversation(ctx, limit=100)
            if not any(m.get("task_id") == tid and m.get("role") == "agent" for m in existing):
                protocol.persist_message(ctx, "agent", rep, tid)
                protocol.metrics.inbound_total += 1

    if task_id:
        poll_body: dict[str, Any] = {"jsonrpc": "2.0", "id": protocol.new_task_id(), "method": "GetTask",
                                     "params": {"id": task_id}}
        if tenant:
            poll_body["params"]["tenant"] = tenant
        resp = _http_post_json(rpc_url, poll_body, headers, min(timeout, 30))
        if "error" in resp:
            raise ValueError(f"Peer '{agent_label}' returned an error: {resp['error'].get('message', resp['error'])}")
        payload = protocol.unwrap_send_message_response(resp.get("result", {}))
        reply = _reply_text_from_result(payload)
        reply_ctx = payload.get("contextId", context_id or "") if isinstance(payload, dict) else (context_id or "")
        state = (payload.get("status") or {}).get("state", "") if isinstance(payload, dict) else ""
        _persist_client_receipt(reply_ctx, reply, state, task_id)
        return reply, reply_ctx, state, task_id

    ctx = context_id or protocol.new_context_id()
    safe_message = security.redact_outbound(message)
    rpc_body = {
        "jsonrpc": "2.0", "id": protocol.new_task_id(), "method": "SendMessage",
        "params": {
            "message": protocol.text_message(protocol.ROLE_USER, safe_message, context_id=ctx),
            "configuration": {"returnImmediately": True},
        },
    }
    if tenant:
        rpc_body["params"]["tenant"] = tenant
    security.audit("outbound", agent_label, rpc_body["id"], safe_message)
    protocol.persist_message(ctx, "user", safe_message, rpc_body["id"])
    protocol.metrics.outbound_total += 1
    resp = _http_post_json(rpc_url, rpc_body, headers, timeout)
    if "error" in resp:
        raise ValueError(f"Peer '{agent_label}' returned an error: {resp['error'].get('message', resp['error'])}")
    payload = protocol.unwrap_send_message_response(resp.get("result", {}))
    assigned_task_id = payload.get("id", "") if isinstance(payload, dict) else ""
    reply = _reply_text_from_result(payload)
    reply_ctx, state = ctx, ""
    if isinstance(payload, dict):
        reply_ctx = payload.get("contextId", ctx)
        state = (payload.get("status") or {}).get("state", "")
    _persist_client_receipt(reply_ctx, reply, state, assigned_task_id or rpc_body["id"])
    return reply, reply_ctx, state, assigned_task_id


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
    """Send a task to a peer (configured name or direct URL); ``context_id`` continues a prior exchange."""
    agent = str(args.get("agent") or args.get("agent_name") or args.get("name") or "").strip()
    message = str(args.get("message") or args.get("text") or args.get("task") or "").strip()
    context_id = str(args.get("context_id") or args.get("contextId") or "").strip()
    task_id = str(args.get("task_id") or args.get("taskId") or "").strip()
    if not agent:
        return "Error: 'agent' is required."
    if message and task_id:
        return "Error: cannot provide both 'message' and 'task_id'. Pass 'message' to start a task, or 'task_id' to poll an existing task."
    if not message and not task_id:
        return "Error: either 'message' or 'task_id' is required."
    peer = _resolve_peer(agent)
    if not peer or not peer.get("url"):
        return f"Error: unknown agent '{agent}'. Configure it under 'a2a_agents' in config.yaml or pass a full http(s):// URL."

    total_timeout = int(peer.get("timeout", _DEFAULT_TIMEOUT))
    start_time = _monotonic()
    effective_task_id = task_id
    error_context_id = context_id

    try:
        reply, reply_ctx, state, assigned_task_id = _send_task(agent, peer, message, context_id, task_id=task_id)
        effective_task_id = task_id or assigned_task_id
        error_context_id = reply_ctx

        # If a new task was submitted and immediately returned in progress, poll until completion or timeout.
        is_working_state = state in (protocol.STATE_WORKING, protocol.STATE_SUBMITTED, "working", "submitted")
        if not task_id and is_working_state and effective_task_id:
            reply, reply_ctx, state, poll_error = _poll_task_until_terminal(
                agent, peer, reply, reply_ctx, state, effective_task_id, start_time, total_timeout
            )
            if poll_error:
                return poll_error
    except urllib.error.HTTPError as e:
        if effective_task_id:
            return _task_poll_error(agent, error_context_id, effective_task_id, f"HTTP {e.code}")
        return _HTTP_CALL_ERRORS.get(e.code, "Error: call to '{agent}' failed — HTTP {code}.").format(agent=agent, code=e.code)
    except ValueError as e:
        if effective_task_id:
            return _task_poll_error(agent, error_context_id, effective_task_id, "invalid peer response")
        return str(e)
    except Exception as e:
        if effective_task_id:
            return _task_poll_error(agent, error_context_id, effective_task_id, "unexpected polling error")
        return f"Error: call to '{agent}' failed — {e}."

    short_state = state.replace("TASK_STATE_", "").replace("_", "-").lower()
    header_parts = [agent]
    if reply_ctx:
        header_parts.append(f"context {reply_ctx}")
    if effective_task_id:
        header_parts.append(f"task {effective_task_id}")
    if short_state:
        header_parts.append(short_state)
    header = "[" + " · ".join(header_parts) + "]"
    body = reply or "(no text reply)"
    if state == protocol.STATE_INPUT_REQUIRED:
        body += f"\n\n(The peer needs more input — answer by calling a2a_call again with context_id '{reply_ctx}'.)"
    elif state == protocol.STATE_WORKING:
        body = (f"The task is currently in progress on '{agent}'.\n"
                f"To inspect or continue waiting for its output, call a2a_call again with "
                f"agent='{agent}' and task_id='{effective_task_id}'.")
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
        reply = a2a_call({"agent": agent_name, "message": message, "context_id": context_id})
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
    def is_success(name: str, text: str) -> bool:
        if not text or text.startswith("Error:"):
            return False
        # Successful a2a_call format: [name · ... · completed]
        first_line = text.splitlines()[0].strip() if text else ""
        if first_line.startswith("[") and first_line.endswith("]"):
            state_token = first_line[1:-1].split("·")[-1].strip().lower()
            return state_token in ("completed", "input-required", "done")
        return False

    successes = [(name, reply) for name, reply in results if is_success(name, reply)]
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
                 "Send a natural-language task to a remote A2A agent or poll an existing task by task_id. "
                 "The agent is a peer (any A2A-compliant framework), not a sub-agent you control. Pass 'context_id' "
                 "from a previous reply to continue a multi-turn exchange, or 'task_id' to check on a long-running turn.",
                 {"agent": _str("Configured peer name (from a2a_agents) or a full http(s):// URL."),
                  "message": _str("Optional if task_id provided: the task / message to send the peer, in natural language."),
                  "context_id": _str("Optional: context id from a prior reply, to continue the conversation."),
                  "task_id": _str("Optional: task id from a prior working response, to inspect or wait for its completion.")},
                 ["agent"]),
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
        # Scoped like the platform gate: os.environ is the launch profile's under multiplexing (#122126).
        if _get_scoped_secret("A2A_PORT"):
            return True
        a2a_cfg = (cfg.get("platforms") or {}).get("a2a") or {}
        return bool(isinstance(a2a_cfg, dict) and a2a_cfg.get("enabled"))
    except Exception:
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
