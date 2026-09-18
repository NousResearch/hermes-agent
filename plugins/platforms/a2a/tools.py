"""
A2A client tools — let the Hermes agent talk to *other* agents as a peer.

Tools (registered in the ``a2a`` toolset):
  - a2a_discover(url)         -> fetch + summarize a peer's Agent Card
  - a2a_call(agent, message)  -> send a task to a peer, return its reply
  - a2a_list()                -> list configured peers + persisted conversations
  - a2a_history(context_id)   -> recall a persisted A2A conversation
  - a2a_orchestrate(...)      -> fan-out task to multiple peers by capability

Peers are resolved from config.yaml under ``a2a_agents``::

    a2a_agents:
      researcher:
        url: "http://localhost:9999"
        auth: { type: bearer, token: "sk-..." }
        timeout: 120
        capabilities: [web_search, research]

Transport is stdlib urllib (no a2a-sdk dependency). The wire format is the A2A
v1.0 JSON-RPC ``message/send`` method; replies from v0.3 peers still parse.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
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


def _a2a_tools_available() -> bool:
    """check_fn for the outbound client tools: serve them ONLY when the
    operator has opted into A2A somehow -- peers configured under
    ``a2a_agents`` in config.yaml, or the inbound platform enabled
    (a peer-reachable Hermes plausibly dials back).

    Maintainer-directed (#95681): these registered unconditionally, so
    every session on every install paid ~561 tok/call for tools whose
    only possible output without config is 'no peers configured'. A2A is
    unrelated to Bot Mode (bots talk over gateway RPCs) -- for most
    installs this toolset is foreign-agent plumbing they never enabled.
    Config adds mid-session surface at the next compaction (#97073).
    """
    cfg = {}
    try:
        cfg = _load_config()
        if cfg.get("a2a_agents"):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        import os as _os

        if _os.getenv("A2A_PORT"):
            return True
        platforms = cfg.get("platforms") or {}
        a2a_cfg = platforms.get("a2a") or {}
        if isinstance(a2a_cfg, dict) and a2a_cfg.get("enabled"):
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _resolve_peer(agent: str) -> Optional[dict]:
    """Resolve a peer name to {url, auth, timeout, capabilities, headers, allowed_rpc_origins}, or treat ``agent`` as a URL."""
    if agent.startswith("http://") or agent.startswith("https://"):
        return {"url": agent, "auth": {}, "timeout": _DEFAULT_TIMEOUT, "capabilities": [], "headers": {}, "allowed_rpc_origins": [], "tenant": ""}
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
        "tenant": entry.get("tenant", ""),
        "headers": entry.get("headers", {}) or {},
        "allowed_rpc_origins": entry.get("allowed_rpc_origins", []) or [],
    }


def _auth_header(auth: dict) -> dict:
    if auth and auth.get("type") == "bearer" and auth.get("token"):
        return {"Authorization": f"Bearer {auth['token']}"}
    return {}


# ---------------------------------------------------------------------------
# Origin + redirect security (ported from #86322)
# ---------------------------------------------------------------------------

def _url_origin(url: str) -> tuple[str, str]:
    """(scheme, host:port) of a URL, lowercased; port defaulted per scheme."""
    parsed = urllib.parse.urlsplit(url.strip())
    host = (parsed.hostname or "").lower()
    # parsed.port is None when absent; explicit :0 is a real (if unroutable)
    # port and must not be silently defaulted.
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    return parsed.scheme.lower(), f"{host}:{port}"


def _url_same_origin(candidate: str, configured: str) -> bool:
    """True when candidate and configured share scheme + host + port."""
    try:
        return _url_origin(candidate) == _url_origin(configured)
    except ValueError:
        return False


def _allowed_rpc_origins(peer: dict) -> list[str]:
    """Operator-pinned cross-origin RPC URLs exempt from the origin check.

    Entries are compared by ORIGIN (scheme + host + port), so an entry pins
    the whole service, not one exact path.
    """
    raw = peer.get("allowed_rpc_origins") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(u).rstrip("/") for u in raw if str(u).strip()]


def _origin_allowed(candidate: str, peer: dict) -> bool:
    """True when candidate's origin is the configured origin or a pinned
    allowed origin (origin-level match, not exact string)."""
    try:
        cand = _url_origin(candidate)
    except ValueError:
        return False
    try:
        if _url_same_origin(candidate, peer.get("url", "")):
            return True
    except ValueError:
        pass
    for entry in _allowed_rpc_origins(peer):
        try:
            if _url_origin(entry) == cand:
                return True
        except ValueError:
            continue
    return False


# Redirects are a credential-exfiltration vector: urllib's default opener
# follows 3xx responses and forwards the full header map (Authorization,
# proxy service tokens) to whatever host the redirect points at. Peers are
# semi-trusted (card-controlled), so every hop must stay inside the origin
# policy or the send dies.
class _NoCredentialRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Fail-closed redirect policy for credential-bearing requests."""

    def __init__(self, allowed_origins: tuple[str, ...] = ()):
        self.allowed_origins = allowed_origins
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        original = req.full_url
        if _url_same_origin(newurl, original) or any(
                _url_same_origin(newurl, o) for o in self.allowed_origins):
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        raise urllib.error.HTTPError(
            req.full_url, code,
            f"A2A redirect to cross-origin {newurl} refused (not same-origin, not in allowed_rpc_origins)",
            headers, fp)


def _open_url_no_redirect_leak(req: urllib.request.Request, timeout: int,
                               allowed_origins: tuple[str, ...] = ()) -> Any:
    """urlopen with fail-closed cross-origin redirect handling."""
    opener = urllib.request.build_opener(_NoCredentialRedirectHandler(allowed_origins))
    return opener.open(req, timeout=timeout)


class _A2aIndeterminateError(Exception):
    """A 524 (origin timeout) — the peer may have executed the task, but the
    response was lost.  Mutating sends are NEVER auto-retried: without a
    proven server-side idempotency contract a replay could execute the task
    twice.  The caller surfaces the indeterminate outcome; recovery composes
    with explicit task-identity polling instead."""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _http_get_json(url: str, headers: dict, timeout: int,
                   allowed_origins: tuple[str, ...] = ()) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Hermes-A2A/1.0", **headers}, method="GET")
    with _open_url_no_redirect_leak(req, timeout, allowed_origins) as resp:  # noqa: S310 (configured peers)
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, body: dict, headers: dict, timeout: int,
                    allowed_origins: tuple[str, ...] = ()) -> dict:
    data = json.dumps(body).encode("utf-8")
    # Custom peer headers are operator-controlled but Content-Type and
    # A2A-Version are protocol-owned and must not be clobbered; a config typo
    # would otherwise cause peer rejection or protocol-version mismatches.
    # User-Agent stays overridable (some proxies filter user agents).
    hdrs = {
        "User-Agent": "Hermes-A2A/1.0",
        **headers,
        "Content-Type": "application/json",
        "A2A-Version": protocol.PROTOCOL_VERSION,
    }
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")

    try:
        with _open_url_no_redirect_leak(req, timeout, allowed_origins) as resp:  # noqa: S310 (configured peers)
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 524:
            raise _A2aIndeterminateError(
                f"peer origin timed out behind its proxy (HTTP 524); the task "
                f"may have executed — outcome indeterminate, not retried"
            ) from e
        raise


def _card_url(base_url: str) -> str:
    # A2A v1.0 canonical discovery path. v0.2 used agent.json; servers may
    # still serve that as a legacy alias, but clients should prefer this.
    return base_url.rstrip("/") + "/.well-known/agent-card.json"


def _legacy_card_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/.well-known/agent.json"


def _fetch_card(base_url: str, headers: dict, timeout: int,
                allowed_origins: tuple[str, ...] = ()) -> dict:
    try:
        return _http_get_json(_card_url(base_url), headers, timeout, allowed_origins)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    return _http_get_json(_legacy_card_url(base_url), headers, timeout, allowed_origins)


def _select_jsonrpc_interface(card: Optional[dict]) -> Optional[dict]:
    if isinstance(card, dict):
        for iface in card.get("supportedInterfaces", []) or []:
            if isinstance(iface, dict) and iface.get("protocolBinding") == "JSONRPC" and iface.get("url"):
                return iface
    return None


def _rpc_url(base_url: str, card: Optional[dict]) -> str:
    """Prefer the card's JSONRPC interface (v1.0 supportedInterfaces), then the
    card's legacy top-level url, then the configured base."""
    iface = _select_jsonrpc_interface(card)
    if iface:
        return str(iface["url"])
    if isinstance(card, dict) and isinstance(card.get("url"), str) and card["url"]:
        return card["url"]
    return base_url.rstrip("/")


def _interface_tenant(card: Optional[dict], peer: dict) -> str:
    iface = _select_jsonrpc_interface(card)
    if iface and iface.get("tenant"):
        return str(iface["tenant"])
    return str(peer.get("tenant") or "")


# --------------------------------------------------------------------------
# Shared send path (used by a2a_call and a2a_orchestrate)
# --------------------------------------------------------------------------

def _short_state(state: str) -> str:
    """TASK_STATE_COMPLETED -> completed (also passes through v0.3 states)."""
    return state.replace("TASK_STATE_", "").replace("_", "-").lower() if state else ""


def _current_origin_session() -> dict:
    """Capture the local session that is making this A2A call, if any.

    The agent's tools run inside the gateway process, where the originating
    session's identity is bound via session ContextVars (``set_session_vars``
    + ``set_current_session_id``). Recording it per-context lets the inbound
    adapter WAKE that exact session when a later out-of-band push arrives on
    the context — the delegate_task mental model (fire-and-forget, get a turn
    when the peer is done), not just a conversation-store write nobody polls.

    Returns {} when no session is bound (pure CLI one-shot, or a process with
    no session context): there is no live session to wake later.
    """
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return {}
    platform = str(get_session_env("HERMES_SESSION_PLATFORM") or "").strip()
    if not platform:
        # CLI/TUI/desktop bind HERMES_SESSION_SOURCE and leave the platform
        # empty (see session_context.NON_MESSAGING_SESSION_SURFACES).
        platform = str(get_session_env("HERMES_SESSION_SOURCE") or "").strip()
    chat_id = str(get_session_env("HERMES_SESSION_CHAT_ID") or "").strip()
    session_id = str(get_session_env("HERMES_SESSION_ID") or "").strip()
    if not platform or (not chat_id and not session_id):
        return {}
    return {
        "platform": platform,
        "chat_id": chat_id,
        "chat_type": str(get_session_env("HERMES_SESSION_CHAT_TYPE") or "").strip(),
        "thread_id": str(get_session_env("HERMES_SESSION_THREAD_ID") or "").strip(),
        "user_id": str(get_session_env("HERMES_SESSION_USER_ID") or "").strip(),
        "profile": str(get_session_env("HERMES_SESSION_PROFILE") or "").strip(),
        "session_id": session_id,
    }


def _current_a2a_origin_target(platform_name: str) -> dict:
    """Delivery target for a confirmation emitted from an A2A session.

    When the CURRENT session is an A2A session (its chat_id is an A2A
    context) and that context has a recorded origin session on
    ``platform_name`` (e.g. the Discord thread where the exchange started),
    a confirmation must return to the origin's chat/thread — the session
    that initiated the call — not the platform home channel (the home
    fallback otherwise posts A2A confirmations to the platform-wide default
    channel instead of the originating thread). Returns
    ``{"chat_id", "thread_id", "chat_type"}`` or {} when not applicable.
    """
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return {}
    if str(get_session_env("HERMES_SESSION_PLATFORM") or "").strip().lower() != "a2a":
        return {}
    context_id = str(get_session_env("HERMES_SESSION_CHAT_ID") or "").strip()
    if not context_id:
        return {}
    try:
        from .adapter import A2AAdapter
        return A2AAdapter._origin_delivery_target(context_id, platform_name)
    except Exception:
        return {}


def _send_task(agent_label: str, peer: dict, message: str, context_id: str) -> tuple[str, str, str]:
    """Send one message/send to a peer. Returns (reply_text, context_id, state).

    Raises urllib errors / ValueError for the caller to format. Handles
    outbound redaction, audit, persistence, and metrics.
    """
    base_url = peer.get("url", "")
    headers = {**_auth_header(peer.get("auth", {}) or {}), **(peer.get("headers", {}) or {})}
    timeout = int(peer.get("timeout", _DEFAULT_TIMEOUT))
    auth = peer.get("auth", {}) or {}
    if auth and any(k.lower() == "authorization" for k in (peer.get("headers", {}) or {})):
        logger.warning(
            "A2A: peer '%s' custom headers override the derived Authorization "
            "header — deliberate proxy auth schemes only",
            agent_label)
    allowed = tuple(_allowed_rpc_origins(peer))

    # Best-effort card fetch (to learn the rpc URL); non-fatal on failure.
    # The card is fetched AT the configured origin with the full credential
    # map — same destination the operator pinned the credentials for (a
    # Cloudflare-Access-fronted peer otherwise 403s the card and streaming
    # discovery is lost). The egress bound is enforced on the RPC destination
    # after the fetch: a card-advertised cross-origin URL never receives them.
    card = None
    try:
        card = _fetch_card(base_url, headers, min(timeout, 30), allowed)
    except Exception:
        pass

    rpc_url = _rpc_url(base_url, card)
    if not _origin_allowed(rpc_url, peer):
        # The card advertised an RPC interface on a different origin than the
        # configured base URL. Sending there would forward operator secrets
        # (bearer tokens, proxy service tokens) to a card-controlled host.
        # Refuse: fall back to the configured origin, never follow the card.
        logger.warning(
            "A2A: peer '%s' card advertised cross-origin RPC URL %s; not in "
            "peer's allowed_rpc_origins — using configured origin %s instead",
            agent_label, rpc_url, base_url)
        rpc_url = base_url.rstrip("/")

    ctx = context_id or protocol.new_context_id()
    safe_message = security.redact_outbound(message)
    # Register this context→peer mapping on every live local A2A adapter so
    # an out-of-band completion push for this context can find its way back
    # to the peer. The context may have been born on ANY platform (discord,
    # telegram, CLI/ACP, api_server) — the local gateway only learns peers
    # from inbound A2A tasks, so without this an outbound-originated context
    # has no peer entry and _push_out_of_band drops the completion. Best-effort:
    # registration must
    # never fail the call.
    try:
        from .adapter import A2AAdapter
        A2AAdapter._register_context_peer(ctx, agent_label)
    except Exception:
        logger.debug("A2A: could not register context peer for %s", ctx, exc_info=True)
    # Record which LOCAL session created this context so an inbound push on
    # the same contextId can WAKE that session (self-post turn) instead of
    # only landing in the a2a conversation store. Same best-effort rule:
    # registration must never fail the call.
    try:
        origin = _current_origin_session()
        if origin:
            from .adapter import A2AAdapter
            A2AAdapter._register_context_session(ctx, origin)
    except Exception:
        logger.debug("A2A: could not register context origin session for %s", ctx, exc_info=True)
    # v1.0: contextId lives inside the Message, not at the params top level.
    sender: dict = {}
    try:
        from .adapter import A2AAdapter
        sender = A2AAdapter._own_sender()
    except Exception:
        logger.debug("A2A: could not derive own sender identity", exc_info=True)
    sender = dict(sender or {})
    # Advertise this client's read patience on the wire: the
    # receiving gateway computes patience = sender.timeout → its configured
    # a2a_agents[peer].timeout → 120, and pushes the reply out-of-band when
    # elapsed > patience + margin instead of writing into a dead socket.
    # Same trust domain as sender.url, which is already accepted.
    sender["timeout"] = timeout
    rpc_body = {
        "jsonrpc": "2.0",
        "id": protocol.new_task_id(),
        "method": "SendMessage",
        "params": {
            "message": protocol.text_message(protocol.ROLE_USER, safe_message, context_id=ctx, sender=sender),
        },
    }

    tenant = _interface_tenant(card, peer)
    if tenant:
        rpc_body["params"]["tenant"] = tenant

    security.audit("outbound", agent_label, rpc_body["id"], safe_message)
    protocol.persist_message(ctx, "user", safe_message, rpc_body["id"])
    protocol.metrics.outbound_total += 1

    resp = _http_post_json(rpc_url, rpc_body, headers, timeout, allowed_origins=allowed)
    if isinstance(resp, dict) and "error" in resp:
        # JSON-RPC peer operation failure — distinct from malformed result, propagate as jsonrpc category
        err = resp["error"]
        # Do not record inbound success metrics or persist reply for peer error
        raise ValueError(f"Peer '{agent_label}' returned JSON-RPC error {err.get('code', '')}: {err.get('message', err)}")
    result = resp.get("result") if isinstance(resp, dict) else None
    # Strict V1 parsing — peer must return a valid SendMessageResponse wrapper
    try:
        parsed = protocol.parse_send_message_result(result, "V1_WRAPPED")
    except protocol.A2AResultValidationError as ve:
        # Invalid/foreign result must never become empty successful reply — propagate structured invalid_response
        raise ValueError(f"Peer '{agent_label}' returned invalid A2A result ({ve.reason}): {ve.detail or result!r}") from ve
    payload = parsed.payload
    reply = parsed.text
    # Use validated context/state from parser, not raw payload fallback
    reply_ctx = parsed.context_id or ctx
    state = parsed.state
    # Persist and metrics only after successful validation
    protocol.persist_message(reply_ctx, "agent", reply, rpc_body["id"])
    protocol.metrics.inbound_total += 1
    return reply, reply_ctx, state


def _reply_text_from_result(result: Any) -> str:
    result = protocol.unwrap_send_message_response(result)
    if result is None:
        return ""
    if not isinstance(result, dict):
        # Invalid scalar results must not become pseudo-reply text; treat as empty
        # so callers can distinguish malformed from valid text.
        return ""
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


# --------------------------------------------------------------------------
# Tool handlers
# --------------------------------------------------------------------------

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

    name = card.get("name", "?")
    desc = card.get("description", "")
    caps = card.get("capabilities", {}) or {}
    skills = card.get("skills", []) or []
    auth = "yes" if card.get("security") else "no"
    ifaces = card.get("supportedInterfaces", []) or []
    proto = ", ".join(
        f"{i.get('protocolBinding', '?')} v{i.get('protocolVersion', '?')}"
        for i in ifaces if isinstance(i, dict)
    ) or f"v{card.get('protocolVersion', '?')} (pre-1.0 card)"
    lines = [
        f"Agent: {name}",
        f"Description: {desc}",
        f"URL: {_rpc_url(url, card)}",
        f"Protocol: {proto}",
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
    When ``context_id`` is a fan-out child context, only the owning peer may
    continue it — a different peer attempting to use it is rejected.
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

    # Conflict rejection: if context_id is a fan-out child, only the
    # owning peer may continue it. Prevents a different peer from
    # silently hijacking a child branch.
    if context_id:
        try:
            from .adapter import A2AAdapter
            claiming_peer = A2AAdapter._reject_child_reuse(context_id, agent)
            if claiming_peer:
                return (
                    f"Error: context '{context_id}' is owned by peer "
                    f"'{claiming_peer}', not '{agent}'. Use the owning peer "
                    f"to continue this fan-out child context."
                )
        except Exception:
            pass

    try:
        reply, reply_ctx, state = _send_task(agent, peer, message, context_id)
    except _A2aIndeterminateError as e:
        return (f"Error: call to '{agent}' is INDETERMINATE — {e}. "
                f"Do not blindly retry a mutating request; check with the peer "
                f"(task id {getattr(e, 'task_id', 'unknown')}) or retry only "
                f"if the operation is safe to repeat.")
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

    header = f"[{agent} · context {reply_ctx}"
    if state:
        header += f" · {_short_state(state)}"
    header += "]"
    body = reply or "(no text reply)"
    if state == protocol.STATE_INPUT_REQUIRED:
        body += (
            "\n\n(The peer needs more input — answer by calling a2a_call again "
            f"with context_id '{reply_ctx}'.)"
        )
    return f"{header}\n{body}"


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
        lines.append(f"Persisted conversations ({len(convos)}) — recall with a2a_history:")
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


def a2a_history(args: dict, **_: Any) -> str:
    """Recall a persisted A2A conversation by context_id.

    This is how prior A2A exchanges survive compaction/restarts: every turn is
    written to ~/.hermes/a2a_conversations/<context>.jsonl and can be reloaded
    here.
    """
    context_id = str(args.get("context_id") or args.get("contextId") or "").strip()
    if not context_id:
        return "Error: 'context_id' is required (see a2a_list for known conversations)."
    try:
        limit = max(1, min(int(args.get("limit") or 50), 200))
    except (ValueError, TypeError):
        limit = 50
    messages = protocol.load_conversation(context_id, limit=limit)
    if not messages:
        return f"No persisted conversation for context '{context_id}'."
    lines = [f"Conversation {context_id} (last {len(messages)} messages):"]
    for m in messages:
        role = m.get("role", "?")
        text = (m.get("text") or "").strip()
        if len(text) > 1000:
            text = text[:1000] + " …[truncated]"
        lines.append(f"[{role}] {text}")
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


def _call_peer_sync(agent_name: str, peer_entry: dict, message: str, context_id: str = "") -> tuple[str, str, str]:
    """Call a single peer synchronously. Returns (agent_name, reply_text, child_context_id)."""
    try:
        peer = {
            "url": peer_entry.get("url", ""),
            "auth": peer_entry.get("auth", {}) or {},
            "timeout": int(peer_entry.get("timeout", _DEFAULT_TIMEOUT)),
            "headers": peer_entry.get("headers", {}) or {},
            "allowed_rpc_origins": peer_entry.get("allowed_rpc_origins", []) or [],
            "tenant": peer_entry.get("tenant", "") or "",
            "capabilities": peer_entry.get("capabilities", []) or [],
        }
        reply, _ctx, _state = _send_task(agent_name, peer, message, context_id)
        return (agent_name, reply or "(no reply)", _ctx)
    except Exception as e:
        return (agent_name, f"Error: {e}", "")


def a2a_orchestrate(args: dict, **_: Any) -> str:
    """Fan-out a task to multiple peer agents by capability.

    Modes:
      - ``all``: send to all peers matching the capability, return all replies.
      - ``first``: send to all matching peers, return the first successful reply.
      - ``best``: send to all, return the longest successful reply (a coarse
        detail heuristic — use ``all`` when you want to judge yourself).

    Configured peers advertise capabilities in config.yaml::

      a2a_agents:
        researcher:
          url: "http://localhost:9991"
          capabilities: [web_search, research]
        coder:
          url: "http://localhost:9992"
          capabilities: [code, debug]

    Returns a machine-readable ``## Peer Mapping`` section with the
    parent_context_id and peer→child_context_id mapping so callers can
    resume a specific child branch with
    ``a2a_call(agent=peer, context_id=child_context_id)``.
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

    # Generate a parent context_id for this fan-out operation.
    parent_ctx = context_id or protocol.new_context_id()

    # Register the origin session for the parent context so child callbacks
    # can trace back to the originating Hermes session.
    origin: dict = {}
    try:
        origin = _current_origin_session()
        if origin:
            from .adapter import A2AAdapter
            A2AAdapter._register_context_session(parent_ctx, origin)
    except Exception:
        logger.debug("A2A: could not register fan-out origin session for %s", parent_ctx, exc_info=True)

    # Fan-out: each peer gets a distinct child context_id.
    # Track all submitted peer→child mappings at submission time so the
    # ownership map is complete regardless of result mode or early break.
    submitted_children: dict[str, str] = {}  # peer_name -> child_ctx (all submitted)
    results: list[tuple[str, str, str]] = []  # (name, reply, child_ctx)
    with ThreadPoolExecutor(max_workers=min(len(matches), _ORCHESTRATE_MAX_WORKERS)) as pool:
        futures = {}
        for name, entry in matches:
            child_ctx = protocol.new_context_id()
            submitted_children[name] = child_ctx
            # Register each child context with the origin session so a late
            # callback on the child wakes the original parent session.
            try:
                if origin:
                    from .adapter import A2AAdapter
                    A2AAdapter._register_context_session(child_ctx, origin)
            except Exception:
                logger.debug("A2A: could not register child context session for %s", child_ctx, exc_info=True)
            futures[pool.submit(_call_peer_sync, name, entry, message, child_ctx)] = name

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                raw = fut.result()
                # Normalize: production _call_peer_sync returns
                # (name, reply, child_context), but compatible callers and
                # test doubles may return (name, reply).  Pad to 3-tuple
                # so downstream code has a uniform shape.
                if len(raw) == 2:
                    result = (raw[0], raw[1], "")
                else:
                    result = raw
                results.append(result)
                # Update submitted mapping if _send_task returned a different
                # confirmed context for this peer.
                if result[2] and result[2] != submitted_children.get(name):
                    submitted_children[name] = result[2]
                if mode == "first" and not result[1].startswith("Error:"):
                    # Got a good reply; cancel peers that haven't started yet.
                    for f in futures:
                        f.cancel()
                    break
            except Exception as e:
                results.append((name, f"Error: {e}", ""))

    # Sort results by peer name for deterministic output
    results.sort(key=lambda r: r[0])

    # Build the peer→child-context ownership mapping from ALL submitted
    # peers (not just collected results).  Result mode controls which
    # replies are returned/selected, not which ownership mappings are
    # retained.
    peer_children: dict[str, str] = dict(submitted_children)

    # Register the complete fan-out children map for persistence and
    # callback routing — covers every peer submitted regardless of mode.
    if peer_children:
        try:
            from .adapter import A2AAdapter
            A2AAdapter._register_fanout_children(parent_ctx, peer_children)
        except Exception:
            logger.debug("A2A: could not register fan-out children for %s", parent_ctx, exc_info=True)

    successes = [(name, reply) for name, reply, _ in results if not reply.startswith("Error:")]

    def _all_failed() -> str:
        lines = ["All peers failed:"]
        for name, reply, _ in results:
            lines.append(f"  {name}: {reply}")
        return "\n".join(lines)

    # Build the machine-readable mapping section from all submitted peers.
    mapping_lines = [
        "",
        "## Peer Mapping",
        f"parent_context_id: {parent_ctx}",
        "peer_to_child:",
    ]
    for name, child_ctx in peer_children.items():
        mapping_lines.append(f"  {name}: {child_ctx}")
    mapping_section = "\n".join(mapping_lines)

    if mode == "best":
        if not successes:
            return _all_failed() + mapping_section
        best = max(successes, key=lambda r: len(r[1]))
        return f"[best: {best[0]}]\n{best[1]}" + mapping_section
    elif mode == "first":
        if not successes:
            return _all_failed() + mapping_section
        name, reply = successes[0]
        return f"[first: {name}]\n{reply}" + mapping_section
    else:  # mode == "all"
        lines = [f"Orchestrated '{capability}' to {len(matches)} peer(s):"]
        for name, reply, _ in results:
            lines.append(f"\n--- {name} ---")
            lines.append(reply)
        return "\n".join(lines) + mapping_section


# --------------------------------------------------------------------------
# Tool schemas + registration
# --------------------------------------------------------------------------

_FunctionSchema = TypedDict("_FunctionSchema", {"name": str, "description": str, "parameters": dict[str, Any]}, total=False)
_ToolSchema = TypedDict("_ToolSchema", {"type": str, "function": _FunctionSchema}, total=False)
_TOOL_DEFINITIONS: dict[str, _ToolSchema] = {
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
    "a2a_history": {
        "type": "function",
        "function": {
            "name": "a2a_history",
            "description": (
                "Recall a persisted A2A conversation transcript by context_id "
                "(survives restarts and context compaction). Use a2a_list to "
                "see known context ids."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "context_id": {"type": "string", "description": "Context id of the conversation to recall."},
                    "limit": {"type": "integer", "description": "Max messages to return (default 50, max 200)."},
                },
                "required": ["context_id"],
            },
        },
    },
    "a2a_orchestrate": {
        "type": "function",
        "function": {
            "name": "a2a_orchestrate",
            "description": (
                "Fan-out a task to multiple peer agents by capability. Peers are "
                "matched from config.yaml a2a_agents.*.capabilities. Modes: 'all' "
                "(return all replies), 'first' (first successful), 'best' (longest "
                "successful reply)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "capability": {"type": "string", "description": "Capability to match (e.g. 'research', 'code') or '*' for all peers."},
                    "message": {"type": "string", "description": "The task to send to all matching peers."},
                    "mode": {"type": "string", "enum": ["all", "first", "best"], "description": "How to aggregate results. Default: 'all'."},
                    "context_id": {"type": "string", "description": "Optional: parent/correlation context id for tracing (not reused as the peer conversation context — each peer gets a distinct child context)."},
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
    "a2a_history": a2a_history,
    "a2a_orchestrate": a2a_orchestrate,
}


def register_tools(ctx) -> None:
    """Register the client tools in the ``a2a`` toolset (config-gated)."""
    for name, definition in _TOOL_DEFINITIONS.items():
        fn_schema = definition.get("function") or {}
        if not fn_schema.get("name"):
            continue
        ctx.register_tool(
            name=name,
            toolset="a2a",
            schema=fn_schema,
            handler=_HANDLERS[name],
            description=fn_schema.get("description", ""),
            emoji="\U0001f9e9",  # puzzle piece
            check_fn=_a2a_tools_available,
        )
