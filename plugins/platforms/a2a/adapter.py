"""
A2A inbound platform adapter — exposes Hermes as an A2A-discoverable agent.

Design (the #11025 insight, done as a plugin with zero core edits):
  - Runs a stdlib http.server in a daemon thread (no a2a-sdk, no asyncio loop
    dependency at register() time — avoids the a2a_fleet "register outside a
    loop" bug class).
  - Serves the A2A v1.0 Agent Card at GET /.well-known/agent-card.json (and legacy agent.json).
  - JSON-RPC at POST /: message/send, message/stream (SSE), tasks/get,
    tasks/list, tasks/cancel, tasks/subscribe, tasks/pushNotificationConfig/create,
    tasks/pushNotificationConfig/get, tasks/pushNotificationConfig/list,
    tasks/pushNotificationConfig/delete.
  - Push notifications: config accepted inline in message/send
    (configuration.taskPushNotificationConfig) or via the create method;
    payloads are v1.0 StreamResponse objects, HMAC-signed.
  - Metrics at GET /metrics.
  - Each inbound task is filtered + framed (security.wrap_inbound) and routed
    into the agent's LIVE gateway session via the normal MessageEvent path, so
    the agent that replies is the same one talking to its user — full memory
    and context, not a throwaway clone.
  - The agent's reply comes back through ``adapter.send()``; we override that to
    fulfil a per-task Future the HTTP handler is blocked on, turning the
    async gateway into a synchronous request/response for the A2A caller.
    ``on_processing_complete`` resolves failures/cancellations promptly.
  - Every exchange is persisted to disk and audit-logged.

Bind safety: with no token configured, the server binds 127.0.0.1 only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import select
import socket
import sqlite3
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import weakref
from collections import deque
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.config import Platform

from . import protocol, security

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 9900
_ORPHAN_TIMEOUT = 300  # seconds before a pending task is considered orphaned
_WATCHDOG_INTERVAL = 60  # seconds between orphaned task watchdog runs
_MAX_BODY = 1_048_576  # 1MB max request body — prevents DoS via memory exhaustion
_SSE_KEEPALIVE = 5  # seconds between SSE keepalive comments
_MAX_CONTEXT_PEERS = 4096  # cap on the context→peer map (LRU-ish, insertion order)
# Seconds past the client's advertised read timeout (sender.timeout) before
# the server assumes the client gave up and marks the pending task
# out_of_band_only. The MSG_PEEK probe catches clients that CLOSE; this is
# the deterministic backstop for clients that stay connected but will
# discard the reply (the reply was consumed although the server saw
# "alive").
_PATIENCE_MARGIN = 30
# Short-window inbound dedupe: the same wire message
# (contextId, messageId) must not be dispatched twice.
_INBOUND_DEDUPE_WINDOW = 60.0
_INBOUND_DEDUPE_MAX = 1024

# Module-level registry of live A2A adapters (weak refs so a dead gateway
# never pins memory). The outbound client tools (plugins/platforms/a2a/tools.py)
# use this to register the *local* context→peer mapping whenever an agent
# makes an outbound A2A call from ANY platform origin (discord, telegram,
# CLI/ACP, api_server). Without this, _context_peers only ever learns peers
# from inbound A2A tasks, so a completion push for a context that was born on
# another platform finds no peer: no A2A inbound ever touched its contextId,
# so the push had nowhere to go).
_ADAPTERS: "dict[int, weakref.ReferenceType[A2AAdapter]]" = {}
_ADAPTERS_GUARD = threading.Lock()

# ── Context→peer persistence ──────────────────────────────────────────────
# _context_peers is otherwise in-memory only. A gateway restart wipes every
# registration, and nothing re-registers afterwards unless a fresh inbound
# A2A task or outbound a2a_call touches the same context — so out-of-band
# completion pushes silently drop until then: the notifier wakes the agent
# post-restart, adapter.send() has no peer, and the push never fires. We
# write-through on registration and reload on adapter start so the mapping
# survives restarts. Best-effort: persistence must never fail the call.
_CONTEXT_PEERS_FILE = "a2a_context_peers.json"

# ── Context→origin-session persistence ───────────────────────────────────
# Which LOCAL gateway session created each outbound A2A context (recorded by
# the client tools at a2a_call time). When an out-of-band push later arrives
# on that context, the adapter wakes the originating session via the same
# self-post mechanism the task watchers use — an explicit wake rather than a
# polled store read.
# Persisted alongside the peer map so the mapping survives a gateway restart
# (a context born before a restart must still wake its session afterwards).
_CONTEXT_SESSIONS_FILE = "a2a_context_sessions.json"


def _reset_worker_session_vars() -> None:
    """Reset session-context vars bound on an HTTP worker thread.

    ``_prepare_task`` binds the A2A session identity via ``set_session_vars``
    on the inbound HTTP worker thread so the asyncio Task created by
    ``run_coroutine_threadsafe`` snapshots it (Task construction copies the
    calling thread's context). After the dispatch is scheduled, the worker
    thread's OWN context must be restored to pristine ``_UNSET`` so the
    bindings don't linger on the threadpool thread for the next request.
    Best-effort: resetting must never fail the dispatch.
    """
    try:
        from gateway.session_context import reset_session_vars

        reset_session_vars()
    except Exception:
        pass


def _context_peers_path() -> Path:
    try:
        from hermes_constants import get_hermes_home
        base = Path(get_hermes_home())
    except Exception:
        base = Path(os.path.expanduser("~/.hermes"))
    return base / _CONTEXT_PEERS_FILE


def _persist_context_peers(peers: Dict[str, str]) -> None:
    """Best-effort write-through of the context→peer map to disk (atomic).

    Like context sessions, use the same restrictive 0o600 mode on the
    temp file before replacement so the final path never inherits a
    permissive umask.
    """
    try:
        path = _context_peers_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(peers, fh, ensure_ascii=False)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(path)
    except Exception:
        logger.debug("A2A: could not persist context peers", exc_info=True)


def _load_context_peers() -> Dict[str, str]:
    """Load the persisted context→peer map (empty dict on any failure)."""
    try:
        path = _context_peers_path()
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v}
    except Exception:
        logger.debug("A2A: could not load persisted context peers", exc_info=True)
    return {}


def _merge_context_peers(peers: Dict[str, str], extra: Dict[str, str]) -> Dict[str, str]:
    """Merge ``extra`` into ``peers``, bounded by _MAX_CONTEXT_PEERS."""
    out = dict(peers)
    for cid, peer in extra.items():
        if len(out) >= _MAX_CONTEXT_PEERS:
            break
        out.setdefault(cid, peer)
    return out


def _context_sessions_path() -> Path:
    try:
        from hermes_constants import get_hermes_home
        base = Path(get_hermes_home())
    except Exception:
        base = Path(os.path.expanduser("~/.hermes"))
    return base / _CONTEXT_SESSIONS_FILE


def _persist_context_sessions(sessions: Dict[str, dict]) -> None:
    """Best-effort write-through of the context→origin-session map (atomic).

    The map carries durable session ids + user/chat ids — same exposure
    class as the peers file, so write 0600 (never inherit a permissive
    umask). chmod the temp file BEFORE the atomic rename so the final
    path never exists with looser perms.
    """
    try:
        path = _context_sessions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(sessions, fh, ensure_ascii=False)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(path)
    except Exception:
        logger.debug("A2A: could not persist context sessions", exc_info=True)


def _load_context_sessions() -> Dict[str, dict]:
    """Load the persisted context→origin-session map (empty on any failure)."""
    try:
        path = _context_sessions_path()
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            out: Dict[str, dict] = {}
            for k, v in data.items():
                if isinstance(v, dict) and v.get("platform"):
                    out[str(k)] = v
            return out
    except Exception:
        logger.debug("A2A: could not load persisted context sessions", exc_info=True)
    return {}


def _merge_context_sessions(sessions: Dict[str, dict], extra: Dict[str, dict]) -> Dict[str, dict]:
    """Merge ``extra`` into ``sessions``, bounded by _MAX_CONTEXT_PEERS."""
    out = dict(sessions)
    for cid, origin in extra.items():
        if len(out) >= _MAX_CONTEXT_PEERS:
            break
        out.setdefault(cid, origin)
    return out


_LOOPBACK_ADDRS = {"127.0.0.1", "localhost", "::1"}


def _own_a2a_url(host: str, port: int) -> str:
    """Build this gateway's own A2A endpoint URL (the one peers push to)."""
    bind_host = host or "127.0.0.1"
    if bind_host in ("0.0.0.0", "::", ""):
        bind_host = "127.0.0.1"
    return f"http://{bind_host}:{int(port or _DEFAULT_PORT)}"


def _sender_url_acceptable(url: str, peers_cfg: dict) -> bool:
    """Whether a message ``sender.url`` may be trusted as a push target.

    Only http(s) URLs whose host is loopback (the shared-host case —
    every local gateway is ``127.0.0.1`` with a distinct port) or whose host
    already appears in a configured ``a2a_agents`` entry are accepted. This
    keeps a body-supplied URL from routing pushes to arbitrary external
    hosts: in localhost-only mode the connection is local anyway, and a
    remote/bearer-authenticated peer must appear in the operator's config
    before its URL is honored.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in _LOOPBACK_ADDRS or host.startswith("127."):
        return True
    for entry in peers_cfg.values():
        if not isinstance(entry, dict):
            continue
        try:
            eu = urllib.parse.urlparse(str(entry.get("url") or ""))
        except Exception:
            continue
        if eu.hostname and eu.hostname.lower() == host:
            return True
    return False


def _is_own_endpoint(url: str, host: str, port: int) -> bool:
    """Whether ``url`` points at this gateway's own A2A endpoint.

    A loopback-hosted URL whose port matches ours can only be this gateway
    when several gateways share one host (each binds a distinct port). Used by
    ``_push_out_of_band`` to catch a context→peer map that was refined to
    our own URL (an in-process loopback push stamps our own sender, and the
    inbound refinement accepts it) and deliver in-process instead of a
    synchronous HTTP self-call.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    hostname = parsed.hostname.lower()
    if hostname not in _LOOPBACK_ADDRS and not hostname.startswith("127."):
        return False
    try:
        peer_port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    except (ValueError, TypeError):
        return False
    return peer_port == int(port or _DEFAULT_PORT)


def _loopback_fallback_url(identity: str, host: str, port: int) -> str:
    """Return this gateway's own A2A URL when ``identity`` is a loopback ``ip:`` identity.

    Localhost-only mode (no bearer tokens configured) authenticates every
    inbound caller as ``ip:<addr>`` — the caller's listening port is not part
    of the identity, so the only port we can know is this gateway's own. The
    push re-enters the local gateway on the same contextId, which the owning
    session sees as the follow-up."""
    if not identity.startswith("ip:"):
        return ""
    addr = identity[3:].strip().lower()
    if addr not in _LOOPBACK_ADDRS and not addr.startswith("127."):
        return ""
    bind_host = host or "127.0.0.1"
    if bind_host in ("0.0.0.0", "::", ""):
        bind_host = "127.0.0.1"
    return f"http://{bind_host}:{int(port or _DEFAULT_PORT)}"


def _reply_timeout() -> float:
    """Seconds to wait for the agent to answer an inbound task."""
    try:
        return max(1.0, float(os.getenv("A2A_REPLY_TIMEOUT", "300")))
    except (ValueError, TypeError):
        return 300.0


def _default_agent_name() -> str:
    name = os.getenv("A2A_AGENT_NAME", "").strip()
    if name:
        return name
    try:
        import socket
        return f"hermes-{socket.gethostname()}"
    except Exception:
        return "hermes-agent"


def _clean_slug(value: str) -> str:
    """Return a URL-safe-ish single-segment slug for a served agent."""
    slug = str(value or "").strip().strip("/")
    return "" if slug in ("", "default", "root") else slug.split("/")[0]


def _join_url(base: str, prefix: str) -> str:
    base = (base or "").strip() or "/"
    if not base.endswith("/"):
        base += "/"
    prefix = (prefix or "").strip("/")
    if not prefix:
        return base
    return urllib.parse.urljoin(base, prefix + "/")


def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return os.getenv("HERMES_PROFILE", "default") or "default"


def _profile_home(profile: str) -> Optional[str]:
    try:
        from hermes_cli.profiles import get_profile_dir
        return str(get_profile_dir(profile))
    except Exception:
        if not profile or profile == "default":
            try:
                from hermes_cli.config import get_hermes_home
                return str(get_hermes_home())
            except Exception:
                return None
        return os.path.expanduser(f"~/.hermes/profiles/{profile}")

def _safe_context_slug(value: str, max_len: int = 96) -> str:
    """Sanitize attacker-provided context ids before using in session titles."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return (slug or "ctx")[:max_len]


def _method_info(method: str) -> tuple[str, bool]:
    """Return (canonical_operation, is_v1_method).

    Canonical operation names are lowercase internal labels. v1 methods use the
    PascalCase names from A2A v1.0 §5.3/§9.4; legacy aliases remain accepted.
    """
    mapping = {
        "SendMessage": ("send", True),
        "message/send": ("send", False),
        "SendStreamingMessage": ("stream", True),
        "message/stream": ("stream", False),
        "GetTask": ("get", True),
        "tasks/get": ("get", False),
        "ListTasks": ("list", True),
        "tasks/list": ("list", False),
        "CancelTask": ("cancel", True),
        "tasks/cancel": ("cancel", False),
        "SubscribeToTask": ("subscribe", True),
        "tasks/subscribe": ("subscribe", False),
        "CreateTaskPushNotificationConfig": ("push_create", True),
        "tasks/pushNotificationConfig/create": ("push_create", False),
        "tasks/pushNotificationConfig/set": ("push_create", False),
        "tasks/pushNotification/set": ("push_create", False),
        "GetTaskPushNotificationConfig": ("push_get", True),
        "tasks/pushNotificationConfig/get": ("push_get", False),
        "ListTaskPushNotificationConfigs": ("push_list", True),
        "tasks/pushNotificationConfig/list": ("push_list", False),
        "DeleteTaskPushNotificationConfig": ("push_delete", True),
        "tasks/pushNotificationConfig/delete": ("push_delete", False),
    }
    return mapping.get(method, ("", False))


class _A2AServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that carries a reference to its adapter."""

    daemon_threads = True

    def __init__(self, addr, handler_cls, adapter: "A2AAdapter"):
        super().__init__(addr, handler_cls)
        self.adapter = adapter


class A2ARequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for the A2A JSON-RPC surface.

    Module-level (not a closure) so request routing is unit-testable; all
    state lives on ``self.server.adapter``.
    """

    @property
    def adapter(self) -> "A2AAdapter":
        return self.server.adapter  # type: ignore[attr-defined]

    # Silence the default stderr access log.
    def log_message(self, format, *args):  # noqa: A002,N802
        logger.debug("A2A http: " + format, *args)

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        # Flush HERE so a dead socket surfaces OSError AT THE CALL SITE —
        # a buffered write into a half-closed socket "succeeds" silently via
        # TCP buffering (R2: the reply vanished because only the buffer, not
        # the client, received it) and the exception would otherwise only
        # fire later in the base handler's finish(), outside our catch.
        self.wfile.flush()

    def _request_public_url(self) -> str:
        """Derive the routable URL for this request.

        Priority: A2A_PUBLIC_URL env > X-Forwarded-Host / Host header (with
        scheme from X-Forwarded-Proto) > empty. Empty means "caller has no
        info, fall back to bind host". See gfdsa's k8s bind-host bug report
        (PR #41711).
        """
        explicit = os.getenv("A2A_PUBLIC_URL", "").strip()
        if explicit:
            return explicit
        host = self.headers.get("X-Forwarded-Host", "") or self.headers.get("Host", "")
        if not host:
            return ""
        host = host.split(",")[0].strip()
        scheme = (self.headers.get("X-Forwarded-Proto", "") or "http").split(",")[0].strip()
        return f"{scheme}://{host}/"

    def do_GET(self):  # noqa: N802
        route = self.adapter._route_for_path(self.path)
        agent = route["agent"]
        subpath = route["subpath"].rstrip("/") or "/"
        if subpath in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
            public_url = self._request_public_url() or None
            self._json(200, self.adapter._build_card(public_url, agent=agent))
            return
        if subpath in ("/", "/health"):
            payload = {
                "status": "ok",
                "agent": agent.get("name") or self.adapter.agent_name,
            }
            # Do not leak profile/tenant topology on remote unauthenticated GETs.
            # Agent Cards are intentionally public; health topology is not.
            if security.localhost_only() or security.authenticate(
                self.headers.get("Authorization"),
                self.client_address[0] if self.client_address else "",
            ) is not None:
                payload["served_agents"] = self.adapter._served_agent_summary(
                    public_url=self._request_public_url() or None)
            self._json(200, payload)
            return
        if subpath == "/metrics":
            self._json(200, protocol.metrics.snapshot())
            return
        self._json(404, {"error": "not found"})

    def _a2a_client_alive(self) -> bool:
        """Best-effort liveness probe for the client behind this request.

        Used while a blocking message/send waits for the agent's reply: the
        peer's ``a2a_call`` client times out (120s) and closes the connection
        long before the reply is ready when processing takes minutes. A stale
        pending task whose client is gone must be dropped so the reply takes
        the out-of-band push path instead of being written into a dead socket
        (otherwise a reply is consumed by a dead waiter and vanishes —
        the peer's session never wakes). Mirrors the SSE path's keepalive
        disconnect detection for the plain POST path.

        Non-destructive: ``select`` + ``MSG_PEEK``, never consumes data.
        Returns True when liveness cannot be determined (assume alive).
        """
        sock = getattr(self, "connection", None)
        if sock is None:
            return True
        try:
            readable, _, _ = select.select([sock], [], [], 0)
            if not readable:
                return True  # no EOF/data pending — assume alive
            # Data or EOF available. b"" (EOF) means the client closed.
            chunk = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            return bool(chunk)
        except (BlockingIOError, InterruptedError):
            return True
        except OSError:
            return False

    def _handle_send(self, req_id, params, identity, agent, is_v1):
        """Route a message/send with dead-client protection.

        The peer's ``a2a_call`` may time out (120s) while the agent is still
        working (processing routinely runs minutes). The reply then has no
        live connection to ride: ``_rpc_message_send`` resolves the stale
        pending task and the response write hits a closed socket. Catch that
        and push the completed reply out-of-band on the same contextId so the
        caller's session still receives it (and wakes).

        Spec layering for this protection:
        - the liveness probe pops the stale waiter while the reply is pending
          (fast path);
        - patience (sender.timeout + margin) is the deterministic backstop
          for clients that stay connected but will discard;
        - a broad ``OSError`` catch on the write closes the probe-race window
          for clients that RST the connection instead of reading.

        There is deliberately NO post-write liveness probe (deviation from
        the literal spec, verified against live traffic): a client that reads the
        response and closes cleanly (urllib sends ``Connection: close`` and
        closes right after reading) is indistinguishable from a client that
        closed without reading — both surface as EOF to MSG_PEEK — so a
        post-write probe double-delivers every clean exchange and sets up a
        self-perpetuating push ping-pong between two gateways (each rescue
        push's response triggers the peer's rescue push, until the anti-loop
        cuts the context).
        """
        result = self.adapter._rpc_message_send(
            req_id, params, identity, agent=agent, v1_response=is_v1,
            client_alive=self._a2a_client_alive,
        )
        if result is None:
            # out_of_band_only with a completed reply: already pushed
            # directly — skip the socket write entirely (the client is gone).
            self.close_connection = True
            return
        try:
            self._json(200, result)
        except OSError:
            self.adapter._push_reply_after_client_gone(req_id, result)

    def do_POST(self):  # noqa: N802
        adapter = self.adapter
        client_ip = self.client_address[0] if self.client_address else ""

        # Identity comes from the presented credential (or the socket in
        # localhost-only mode) — never from the request body.
        identity = security.authenticate(self.headers.get("Authorization"), client_ip)
        if identity is None:
            self._json(401, protocol.jsonrpc_error(None, protocol.ERR_UNAUTHORIZED, "unauthorized"))
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > _MAX_BODY:
                self._json(413, protocol.jsonrpc_error(None, protocol.ERR_PARSE, "payload too large"))
                return
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw.decode("utf-8"))
        except Exception:
            self._json(400, protocol.jsonrpc_error(None, protocol.ERR_PARSE, "parse error"))
            return

        if not isinstance(req, dict):
            self._json(400, protocol.jsonrpc_error(None, protocol.ERR_INVALID_PARAMS, "JSON-RPC request must be an object"))
            return

        req_id = req.get("id")
        method = str(req.get("method", ""))
        params = req.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            self._json(200, protocol.jsonrpc_error(req_id, protocol.ERR_INVALID_PARAMS, "params must be an object"))
            return

        version = (self.headers.get("A2A-Version") or "").strip()
        if version and version not in {"1.0", "1.0.0"}:
            self._json(200, protocol.jsonrpc_error(req_id, protocol.ERR_INVALID_PARAMS, f"unsupported A2A-Version: {version}"))
            return

        operation, is_v1 = _method_info(method)
        route = adapter._route_for_request(self.path, params)
        if route.get("error"):
            self._json(400, protocol.jsonrpc_error(req_id, protocol.ERR_INVALID_PARAMS, route["error"]))
            return
        agent = route["agent"]

        if not adapter._rate_limiter.allow(identity):
            protocol.metrics.rate_limit_triggers += 1
            self._json(429, protocol.jsonrpc_error(req_id, protocol.ERR_RATE_LIMITED, "rate limit exceeded"))
            return

        if not security.is_trusted_peer(identity):
            self._json(403, protocol.jsonrpc_error(
                req_id, protocol.ERR_UNTRUSTED_PEER, f"peer '{identity}' not trusted"))
            return

        if not operation:
            self._json(200, protocol.jsonrpc_error(
                req_id, protocol.ERR_METHOD_NOT_FOUND, f"method not found: {method}"))
            return

        if operation == "send":
            self._handle_send(req_id, params, identity, agent=agent, is_v1=is_v1)
            return
        if operation == "stream":
            adapter._rpc_message_stream(self, req_id, params, identity, agent=agent)
            return
        if operation == "get":
            self._json(200, adapter._rpc_tasks_get(req_id, params, agent=agent))
            return
        if operation == "list":
            self._json(200, adapter._rpc_tasks_list(req_id, params, agent=agent))
            return
        if operation == "cancel":
            self._json(200, adapter._rpc_tasks_cancel(req_id, params, agent=agent))
            return
        if operation == "subscribe":
            adapter._rpc_tasks_subscribe(self, req_id, params, agent=agent)
            return
        if operation == "push_create":
            self._json(200, adapter._rpc_push_config_create(req_id, params, agent=agent))
            return
        if operation == "push_get":
            self._json(200, adapter._rpc_push_config_get(req_id, params, agent=agent))
            return
        if operation == "push_list":
            self._json(200, adapter._rpc_push_config_list(req_id, params, agent=agent))
            return
        if operation == "push_delete":
            self._json(200, adapter._rpc_push_config_delete(req_id, params, agent=agent))
            return



class A2AAdapter(BasePlatformAdapter):
    """Inbound A2A server adapter."""

    def __init__(self, config, **kwargs):
        platform = Platform("a2a")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}
        self.port = int(os.getenv("A2A_PORT") or extra.get("port", _DEFAULT_PORT))
        self.host = security.resolve_bind_host()
        self.agent_name = _default_agent_name()
        self._advertised_toolsets = [
            t.strip() for t in (
                list(extra.get("advertised_toolsets") or [])
                or os.getenv("A2A_ADVERTISED_TOOLSETS", "").split(",")
            ) if str(t).strip()
        ]
        self._active_profile = _active_profile_name()
        self._agents = self._load_served_agents(extra)

        self._httpd: Optional[_A2AServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Per-adapter protocol state (not module-global): task store, anti-loop
        # turn tracking, and rate limiting.
        self.tasks = protocol.TaskStore()
        self._turns = protocol.TurnTracker()
        self._rate_limiter = protocol.RateLimiter()

        # Forwarded profile sessions: map (profile, agent_slug, context_id) -> session_id.
        self._profile_sessions: Dict[tuple[str, str, str], str] = {}
        self._profile_session_locks: Dict[tuple[str, str, str], threading.Lock] = {}
        self._profile_session_locks_guard = threading.Lock()

        # Pending reply futures, keyed by task_id. Each future resolves to a
        # (state, text) tuple. _pending_order keeps per-context FIFO order so
        # adapter.send() — which only knows the context — resolves the oldest
        # outstanding task for that context (no cross-talk between concurrent
        # requests sharing a context).
        self._pending: Dict[str, tuple[str, Future]] = {}
        self._pending_order: Dict[str, deque[str]] = {}
        self._pending_lock = threading.Lock()

        # Context → peer identity map. Recorded on every inbound task so an
        # out-of-band send (kanban notifier wake reply, late completion) with
        # no pending waiter can be pushed back to the peer that owns the
        # context — reusing the same contextId keeps the caller's session.
        self._context_peers: Dict[str, str] = {}
        self._context_peers_lock = threading.Lock()

        # Context → originating LOCAL session. Recorded by the client tools
        # at a2a_call time (which gateway session created this context), so
        # an inbound push on the context can WAKE that session via the kanban
        # watcher's self-post mechanism — agency (a fresh agent turn) rather
        # than a conversation-store write nobody polls.
        self._context_sessions: Dict[str, dict] = {}
        self._context_sessions_lock = threading.Lock()

        # Short-window inbound dedupe: (contextId, messageId) → first
        # arrival time. The same wire message must not be dispatched twice
        # (duplicate handoffs were observed in testing, and the push+retry
        # paths make double-delivery possible).
        self._inbound_seen: Dict[tuple[str, str], float] = {}
        self._inbound_seen_lock = threading.Lock()

        # Orphaned task watchdog
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None

        # Register this adapter so the outbound client tools can map local
        # contexts back to this gateway's peer table (see _register_context_peer).
        with _ADAPTERS_GUARD:
            _ADAPTERS[id(self)] = weakref.ref(self)

    # ── Cross-platform context peer registration ─────────────────────────

    @classmethod
    def _register_context_peer(cls, context_id: str, peer: str) -> None:
        """Record ``context_id`` → ``peer`` on every live local A2A adapter.

        Called by the outbound client tools before an A2A call so a context
        that was born on ANY platform (discord, telegram, CLI/ACP,
        api_server) is known to the local gateway. Without this, an
        out-of-band completion push for that context finds no peer in
        ``_context_peers`` and is dropped: no A2A inbound ever reached its
        gateway, so the push had nowhere to go.

        The peer identity is the *local* handle for the remote agent (the
        ``a2a_agents`` config key / URL), which is exactly what
        ``_push_out_of_band`` needs to resolve the callback target. Inbound
        A2A tasks keep recording the authenticated identity as before; this
        is an additive path for outbound-originated contexts.
        """
        if not context_id or not peer:
            return
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        union: Dict[str, str] = {}
        for ref in refs:
            adapter = ref()
            if adapter is None:
                continue
            with adapter._context_peers_lock:
                adapter._context_peers[context_id] = peer
                if len(adapter._context_peers) > _MAX_CONTEXT_PEERS:
                    adapter._context_peers.pop(next(iter(adapter._context_peers)), None)
                union.update(adapter._context_peers)
        # Write-through so the registration survives a gateway restart:
        # a restart wipes the in-memory map and no later inbound/outbound
        # task re-registered the context, so the completion push would be
        # dropped before any side effect. Merge with
        # the on-disk state so a registration made with no live adapters
        # (e.g. a CLI/ACP process) is still persisted for the next gateway
        # start, and never clobber the disk map with an empty union.
        disk = _load_context_peers()
        disk.update(union)
        _persist_context_peers(_merge_context_peers({}, disk))

    @classmethod
    def _register_context_session(cls, context_id: str, origin: dict) -> None:
        """Record ``context_id`` → the LOCAL session that created it.

        Called by the outbound client tools before an A2A call (the same
        moment they register the context→peer mapping), so a later out-of-band
        push on this context can WAKE that originating session — the session
        that called a2a_call gets a fresh turn when the peer pushes back,
        the same way a task-completion watcher wakes a task creator.

        ``origin`` carries the session identity captured from the session
        ContextVars at call time: platform, chat_id, chat_type, thread_id,
        user_id, profile, and the raw durable session_id (for the non-push
        adapter wake path). Best-effort: registration must never fail the
        call, and is bounded like the peer map (drop the oldest entry past
        _MAX_CONTEXT_PEERS).
        """
        if not context_id or not isinstance(origin, dict) or not origin.get("platform"):
            return
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        union: Dict[str, dict] = {}
        for ref in refs:
            adapter = ref()
            if adapter is None:
                continue
            with adapter._context_sessions_lock:
                adapter._context_sessions[context_id] = dict(origin)
                if len(adapter._context_sessions) > _MAX_CONTEXT_PEERS:
                    adapter._context_sessions.pop(next(iter(adapter._context_sessions)), None)
                union.update(adapter._context_sessions)
        # Write-through so the mapping survives a gateway restart — a context
        # born before a restart must still wake its session afterwards (the
        # same restart-wipe failure the peer map suffered).
        # setdefault the direct entry too: with no live adapter in this
        # process (CLI/one-shot) union is empty, and the registration must
        # still land on disk for the next gateway start.
        disk = _load_context_sessions()
        disk.update(union)
        disk.setdefault(context_id, dict(origin))
        _persist_context_sessions(_merge_context_sessions({}, disk))

    @classmethod
    def _own_sender(cls) -> dict:
        """Return this process's A2A AgentName identity for outbound messages.

        Reads the first live local adapter so the outbound client tools
        (which run in the same gateway process) can stamp ``sender`` on
        every message they send — the receiving gateway uses it to learn
        our real endpoint (host + port) for out-of-band pushes. When no
        adapter is live (CLI/helper/one-shot processes) the identity is
        derived from config/env (``_sender_from_config``), so the sender
        block is ALWAYS stamped — the receiver can refine the loopback
        identity to a routable peer instead of leaving ``ip:127.0.0.1``
        (helper-sent messages without a sender had their replies
        silently dropped).
        """
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        for ref in refs:
            adapter = ref()
            if adapter is not None:
                return adapter._sender_identity()
        return cls._sender_from_config()

    @classmethod
    def _sender_from_config(cls) -> dict:
        """Sender identity for processes with no live adapter (CLI/helpers).

        Derives ``{agentId, name, url}`` from A2A_AGENT_NAME / A2A_PUBLIC_URL
        / A2A_PORT (env first, then the HERMES_HOME config's
        ``platforms.a2a.port``), so every outbound wire message stamps a
        routable sender the receiving gateway can refine to.
        """
        name = os.getenv("A2A_AGENT_NAME", "").strip() or _default_agent_name()
        port = _DEFAULT_PORT
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            a2a_cfg = (cfg.get("platforms") or {}).get("a2a") or {}
            port = int(a2a_cfg.get("port") or _DEFAULT_PORT)
        except Exception:
            logger.debug("A2A: could not read a2a port from config for sender identity", exc_info=True)
        try:
            port = int(os.getenv("A2A_PORT") or port)
        except (ValueError, TypeError):
            pass
        public = os.getenv("A2A_PUBLIC_URL", "").strip()
        if public:
            url = public.rstrip("/")
        else:
            url = _own_a2a_url("127.0.0.1", port)
        return {"agentId": name, "name": name, "url": url}

    def _sender_identity(self) -> dict:
        """This adapter's A2A v1.0 AgentName (``agentId``/``name``/``url``)."""
        return {
            "agentId": self.agent_name,
            "name": self.agent_name,
            "url": _own_a2a_url(self.host, self.port),
        }

    def _refine_peer_identity(self, peer: str, params: dict, context_id: str) -> str:
        """Resolve a port-less ``ip:`` identity to a routable peer.

        Localhost-only mode authenticates every inbound caller as
        ``ip:<addr>`` — the caller's listening port is not part of the
        identity, so when several gateways share one host every peer (and
        this gateway itself) looks identical and an out-of-band push has
        nowhere real to go (the completion was pushed to the receiving
        gateway's OWN loopback endpoint instead of the calling gateway).

        The A2A v1.0 ``sender`` AgentName on the inbound message carries the
        peer's real endpoint. Prefer a configured ``a2a_agents`` key match
        (agentId/name), then a validated sender URL; otherwise return the
        authenticated identity unchanged (bearer-authenticated peers keep
        their token identity — they are already resolvable).
        """
        if not peer.startswith("ip:"):
            return peer
        msg = params.get("message") if isinstance(params, dict) else None
        sender = msg.get("sender") if isinstance(msg, dict) else None
        if not isinstance(sender, dict):
            return peer
        name = str(sender.get("agentId") or sender.get("name") or "").strip()
        url = str(sender.get("url") or "").strip()
        if not name and not url:
            return peer
        peers_cfg: dict = {}
        try:
            from . import tools as a2a_tools
            peers_cfg = (a2a_tools._load_config() or {}).get("a2a_agents") or {}
        except Exception:
            logger.debug("A2A: could not load a2a_agents for peer refinement", exc_info=True)
        if name and name in peers_cfg:
            logger.info(
                "A2A: refined ip: identity for context %s to configured peer %r (sender agentId)",
                context_id, name,
            )
            return name
        if url and _sender_url_acceptable(url, peers_cfg):
            logger.info(
                "A2A: refined ip: identity for context %s to sender url %s",
                context_id, url,
            )
            return url
        return peer

    @classmethod
    def _origin_delivery_target(cls, context_id: str, platform_name: str) -> dict:
        """Delivery target of the local session that started this A2A context.

        When an A2A context was born in a real gateway session (e.g. a
        Discord thread), confirmations the agent emits for that context must
        return to the origin session's chat/thread — not the platform home
        channel (deliveries return to whichever
        session started the A2A exchange; home only when no origin exists).
        Returns ``{"chat_id": ..., "thread_id": ..., "chat_type": ...}`` or
        ``{}`` when the origin is unknown, unrecorded, or on another platform.
        Checks the live in-memory map first, then the persisted write-through
        (a gateway restart wipes memory; the disk copy survives).
        """
        origin: dict = {}
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        for ref in refs:
            adapter = ref()
            if adapter is None:
                continue
            with adapter._context_sessions_lock:
                origin = adapter._context_sessions.get(context_id) or {}
            if origin:
                break
        if not origin:
            origin = (_load_context_sessions() or {}).get(context_id) or {}
        if not origin:
            return {}
        if str(origin.get("platform") or "").strip().lower() != str(platform_name or "").strip().lower():
            return {}
        chat_id = str(origin.get("chat_id") or "").strip()
        if not chat_id:
            return {}
        return {
            "chat_id": chat_id,
            "thread_id": str(origin.get("thread_id") or "").strip(),
            "chat_type": str(origin.get("chat_type") or "group").strip() or "group",
        }

    def _unregister_adapter(self) -> None:
        with _ADAPTERS_GUARD:
            _ADAPTERS.pop(id(self), None)

    @property
    def name(self) -> str:
        return "A2A"

    @property
    def authorization_is_upstream(self) -> bool:
        """A2A authenticates every inbound request via bearer token (or
        localhost-only binding) in ``do_POST`` before dispatch — the identity
        is already authorized upstream. Without this override, the gateway's
        per-platform user allow-list (``{PLATFORM}_ALLOWED_USERS``) rejects
        A2A peers because their identity is a token-derived name or pod IP,
        not a platform account the operator configures in an env allow-list.

        This is authorization delegated to the A2A bearer-token transport,
        not a fail-open: every request is 401'd if the credential is wrong.
        Reported by kuangmi-bit (PR #41711 comment, Jun 27).
        """
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self, **_kwargs) -> bool:
        # Gateway reconnection plumbing passes adapter-agnostic kwargs such as
        # ``is_reconnect``. A2A does not need them, but accepting them keeps the
        # plugin compatible with the BasePlatformAdapter lifecycle contract.
        # Capture the running gateway loop so the HTTP thread can marshal
        # events onto it via run_coroutine_threadsafe.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        try:
            self._httpd = _A2AServer((self.host, self.port), A2ARequestHandler, self)
        except OSError as e:
            logger.error("A2A: could not bind %s:%s — %s", self.host, self.port, e)
            self._set_fatal_error("bind_failed", f"A2A bind failed: {e}", retryable=True)
            return False

        self._server_thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="a2a-http",
            daemon=True,
        )
        self._server_thread.start()

        # Reset watchdog state for reconnection (disconnect sets the event)
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="a2a-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

        # Reload context→peer registrations persisted by a previous gateway
        # run. Without this, a restart wipes every registration and out-of-band
        # completion pushes drop until a fresh inbound task re-registers the
        # context: the gateway restarted between the original call and the
        # completion, and the push had no peer.
        with self._context_peers_lock:
            restored = _load_context_peers()
            merged = _merge_context_peers(self._context_peers, restored)
            self._context_peers.clear()
            self._context_peers.update(merged)
        if restored:
            logger.info(
                "A2A: restored %d context→peer registration(s) from %s",
                len(restored), _context_peers_path(),
            )

        # Restore context→origin-session registrations persisted by a
        # previous gateway run so pushes still wake their originating
        # sessions after a restart (a2a_call re-registers on the next call,
        # but a completion arriving right after a restart must not drop).
        restored_sessions = self._restore_persisted_context_sessions()
        if restored_sessions:
            logger.info(
                "A2A: restored %d context→origin-session registration(s) from %s",
                restored_sessions, _context_sessions_path(),
            )

        self._mark_connected()

        exposure = "localhost-only" if security.localhost_only() else "REMOTE (bearer auth)"
        logger.info(
            "A2A: serving Agent Card + JSON-RPC on http://%s:%s (%s) as %r; %d routed agent(s)",
            self.host, self.port, exposure, self.agent_name, len(self._agents),
        )
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        self._watchdog_stop.set()
        self._unregister_adapter()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        # Fail any in-flight replies so blocked HTTP threads don't hang.
        with self._pending_lock:
            for _ctx, fut in self._pending.values():
                if not fut.done():
                    fut.set_result((protocol.STATE_FAILED, "[agent shutting down]"))
            self._pending.clear()
            self._pending_order.clear()

    # ── Orphaned task watchdog ─────────────────────────────────────────────

    def _watchdog_loop(self) -> None:
        """Background thread that fails orphaned tasks (keeps them queryable)."""
        while not self._watchdog_stop.wait(_WATCHDOG_INTERVAL):
            try:
                for tid in self.tasks.fail_orphans(_ORPHAN_TIMEOUT):
                    logger.warning("A2A: orphaned task %s marked failed (timeout %ds)", tid, _ORPHAN_TIMEOUT)
                    protocol.metrics.tasks_failed += 1
            except Exception:
                logger.debug("A2A: watchdog error", exc_info=True)

    # ── Agent routing + Agent Cards ───────────────────────────────────────

    def _load_global_a2a_config(self) -> dict:
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}

    def _load_served_agents(self, extra: dict) -> dict[str, dict]:
        """Load served-agent routing config.

        Preferred config location is ``platforms.a2a.extra.agents``. A top-level
        ``a2a_served_agents`` fallback is accepted for scripts/tests. Root/default
        always maps to the live gateway session for backward compatibility.
        """
        raw = extra.get("agents") or extra.get("served_agents")
        if raw is None:
            cfg = self._load_global_a2a_config()
            raw = cfg.get("a2a_served_agents") or (cfg.get("a2a") or {}).get("served_agents")

        agents: dict[str, dict] = {}
        default_desc = os.getenv(
            "A2A_AGENT_DESCRIPTION",
            "Hermes Agent — a general-purpose agent reachable over A2A.",
        )
        agents[""] = {
            "slug": "",
            "path": "",
            "tenant": "",
            "profile": self._active_profile,
            "local": True,
            "name": self.agent_name,
            "description": default_desc,
            "advertised_toolsets": self._advertised_toolsets,
        }

        reserved = {"health", "metrics", ".well-known"}
        tenants: dict[str, str] = {}
        items = raw.items() if isinstance(raw, dict) else enumerate(raw or []) if isinstance(raw, list) else []
        for key, val in items:
            if not isinstance(val, dict):
                continue
            slug = _clean_slug(str(val.get("slug") or val.get("id") or key))
            if not slug:
                continue
            path_segment = _clean_slug(str(val.get("path") or slug))
            if not path_segment or path_segment in reserved:
                logger.warning("A2A: ignoring served agent %r with reserved/invalid path %r", slug, path_segment)
                continue
            profile = str(val.get("profile") or slug).strip()
            path = "/" + path_segment
            toolsets = val.get("advertised_toolsets") or val.get("toolsets") or val.get("capabilities") or []
            if isinstance(toolsets, str):
                toolsets = [t.strip() for t in toolsets.split(",") if t.strip()]
            local = bool(val.get("local")) or profile in ("", "default", self._active_profile)
            tenant = str(val.get("tenant") or slug).strip()
            if tenant:
                if tenant in tenants:
                    logger.warning(
                        "A2A: ignoring served agent %r with duplicate tenant %r already used by %r",
                        slug, tenant, tenants[tenant],
                    )
                    continue
                tenants[tenant] = slug
            agents[slug] = {
                "slug": slug,
                "path": path,
                "tenant": tenant,
                "profile": profile or slug,
                "local": local,
                "name": str(val.get("name") or f"Hermes {slug}"),
                "description": str(val.get("description") or f"Hermes profile '{profile or slug}' exposed over A2A."),
                "advertised_toolsets": list(toolsets or []),
                "timeout": int(val.get("timeout") or _reply_timeout()),
            }
        return agents

    def _served_agent_summary(self, public_url: Optional[str] = None) -> list[dict]:
        base = (public_url or "").strip() or f"http://{self.host}:{self.port}/"
        return [
            {
                "slug": a["slug"] or "default",
                "name": a.get("name"),
                "url": _join_url(base, a.get("path", "")),
                "tenant": a.get("tenant") or None,
                "profile": a.get("profile"),
                "local": bool(a.get("local")),
            }
            for a in self._agents.values()
        ]

    def _route_for_path(self, raw_path: str) -> dict:
        path = urllib.parse.urlsplit(raw_path or "/").path or "/"
        # Longest prefix wins. Default/root agent is the fallback.
        for agent in sorted(self._agents.values(), key=lambda a: len(a.get("path", "")), reverse=True):
            prefix = agent.get("path", "") or ""
            if prefix and (path == prefix or path.startswith(prefix + "/")):
                subpath = path[len(prefix):] or "/"
                if not subpath.startswith("/"):
                    subpath = "/" + subpath
                return {"agent": agent, "subpath": subpath}
        return {"agent": self._agents[""], "subpath": path}

    def _route_for_request(self, raw_path: str, params: dict) -> dict:
        route = self._route_for_path(raw_path)
        agent = route["agent"]
        tenant = str((params or {}).get("tenant") or "")
        # If no URL prefix chose a non-default agent, allow v1.0 tenant routing.
        if agent.get("slug") == "" and tenant:
            matches = [a for a in self._agents.values() if a.get("tenant") == tenant]
            if matches:
                route = {"agent": matches[0], "subpath": route["subpath"]}
                agent = matches[0]
        expected = str(agent.get("tenant") or "")
        if tenant and expected and tenant != expected:
            return {"error": f"tenant {tenant!r} does not match routed agent {agent.get('slug') or 'default'}"}
        return route

    def _build_card(self, public_url: Optional[str] = None, agent: Optional[dict] = None) -> dict:
        # Prefer per-request public URL (from X-Forwarded-Host / Host /
        # A2A_PUBLIC_URL) over bind host, so peers can call back when we're
        # behind a reverse proxy.
        agent = agent or self._agents[""]
        base = (public_url or "").strip() or f"http://{self.host}:{self.port}/"
        url = _join_url(base, agent.get("path", ""))
        return protocol.build_agent_card(
            name=agent.get("name") or self.agent_name,
            url=url,
            description=agent.get("description") or "Hermes Agent — a general-purpose agent reachable over A2A.",
            skills=self._advertised_skills(agent),
            streaming=bool(agent.get("local", True)),
            push_notifications=True,
            auth_required=not security.localhost_only(),
            tenant=str(agent.get("tenant") or ""),
        )

    def _advertised_skills(self, agent: Optional[dict] = None) -> list[dict]:
        """Dynamic Agent Card skills from the live tool registry.

        The card reflects what the agent can actually do right now. An
        explicit ``advertised_toolsets`` config (or A2A_ADVERTISED_TOOLSETS)
        restricts what we advertise; without a registry we fall back to that
        static list.
        """
        try:
            from tools.registry import registry as tool_registry
            names = tool_registry.get_registered_toolset_names()
            configured = (agent or {}).get("advertised_toolsets") if agent else self._advertised_toolsets
            allowed = set(configured or []) or None
            mapping = {
                n: tool_registry.get_tool_names_for_toolset(n)
                for n in names
                if allowed is None or n in allowed
            }
            if mapping:
                return protocol.skills_from_toolsets(mapping)
        except Exception:
            logger.debug("A2A: tool registry unavailable for Agent Card", exc_info=True)
        configured = (agent or {}).get("advertised_toolsets") if agent else self._advertised_toolsets
        return protocol.skills_from_toolsets(configured or [])

    # ── Pending reply plumbing ────────────────────────────────────────────

    def _add_pending(self, task_id: str, context_id: str) -> Future:
        fut: Future = Future()
        with self._pending_lock:
            self._pending[task_id] = (context_id, fut)
            self._pending_order.setdefault(context_id, deque()).append(task_id)
        return fut

    def _pop_pending(self, task_id: str) -> None:
        with self._pending_lock:
            entry = self._pending.pop(task_id, None)
            if entry:
                order = self._pending_order.get(entry[0])
                if order:
                    try:
                        order.remove(task_id)
                    except ValueError:
                        pass
                    if not order:
                        self._pending_order.pop(entry[0], None)

    def _resolve_task(self, task_id: str, state: str, text: str) -> bool:
        with self._pending_lock:
            entry = self._pending.get(task_id)
            if entry and not entry[1].done():
                entry[1].set_result((state, text))
                resolved = True
            else:
                resolved = False
        # Pop so resolved entries don't accumulate: HTTP paths pop via
        # _finalize_task, but in-process loopback pushes (and
        # on_processing_complete / cancel) resolve without a finalize call.
        if resolved:
            self._pop_pending(task_id)
        return resolved

    def _resolve_oldest_for_context(self, context_id: str, state: str, text: str) -> bool:
        with self._pending_lock:
            for task_id in list(self._pending_order.get(context_id, ())):
                entry = self._pending.get(task_id)
                if entry and not entry[1].done():
                    entry[1].set_result((state, text))
                    break
            else:
                return False
        self._pop_pending(task_id)
        return True

    def _scope_for_agent(self, agent: Optional[dict]) -> tuple[str, str]:
        agent = agent or self._agents[""]
        return str(agent.get("slug") or ""), str(agent.get("tenant") or "")

    def _forward_lock(self, key: tuple[str, str, str]) -> threading.Lock:
        with self._profile_session_locks_guard:
            lock = self._profile_session_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._profile_session_locks[key] = lock
            return lock

    # ── Inbound task handling ─────────────────────────────────────────────

    def _prepare_task(self, params: dict, peer: str, agent: Optional[dict] = None) -> tuple[Optional[dict], Optional[dict]]:
        """Validate, register, and dispatch an inbound message.

        Returns (terminal_task, None) when the task ends immediately
        (rejected / not ready), else (None, pending) where pending carries
        the future the caller must wait on. Runs on an HTTP worker thread.
        """
        agent = agent or self._agents[""]
        text = protocol.extract_text(params)
        context_id = protocol.extract_context_id(params) or protocol.new_context_id()
        task_id = protocol.new_task_id()
        # Localhost-only mode authenticates the caller as "ip:<addr>" with no
        # port — unresolvable as a push target when every gateway (including
        # this one) shares one host. Refine the
        # identity from the message's A2A v1.0 sender AgentName so the
        # context→peer registration below routes out-of-band pushes back to
        # the peer's REAL endpoint, port included.
        peer = self._refine_peer_identity(peer, params, context_id)

        # F: inbound dedupe — the same wire message (contextId + messageId)
        # must not be dispatched twice within a short window. Duplicate
        # handoffs were already observed in testing, and the push+retry
        # paths make double-delivery possible. Keyed by the peer-stamped
        # messageId so consecutive turns on one context never collide.
        msg_for_id = params.get("message") if isinstance(params, dict) else None
        message_id = str((msg_for_id.get("messageId") or "") if isinstance(msg_for_id, dict) else "").strip()
        if context_id and message_id and self._is_duplicate_inbound(context_id, message_id):
            rec = self.tasks.create(task_id, context_id, peer, *self._scope_for_agent(agent))
            self.tasks.complete(task_id, protocol.STATE_REJECTED, "")
            logger.warning(
                "A2A: duplicate inbound message %s on context %s within the dedupe window; rejecting",
                message_id, context_id,
            )
            return protocol.build_task(
                task_id, context_id, protocol.STATE_REJECTED,
                "Duplicate message.", created_at=rec["created_iso"],
            ), None

        # Anti-loop ping-pong protection
        turn = self._turns.track(context_id)
        if turn > protocol.max_pingpong_turns():
            protocol.metrics.anti_loop_triggers += 1
            logger.warning("A2A: anti-loop triggered for context %s (turn %d > %d)",
                           context_id, turn, protocol.max_pingpong_turns())
            rec = self.tasks.create(task_id, context_id, peer, *self._scope_for_agent(agent))
            self.tasks.complete(task_id, protocol.STATE_REJECTED, "")
            return protocol.build_task(
                task_id, context_id, protocol.STATE_REJECTED,
                f"Anti-loop protection: context {context_id} exceeded "
                f"{protocol.max_pingpong_turns()} turns. Start a new context or "
                f"increase A2A_MAX_PINGPONG_TURNS.",
                created_at=rec["created_iso"],
            ), None

        if not text:
            rec = self.tasks.create(task_id, context_id, peer, *self._scope_for_agent(agent))
            self.tasks.complete(task_id, protocol.STATE_REJECTED, "")
            return protocol.build_task(
                task_id, context_id, protocol.STATE_REJECTED,
                "Empty task — nothing to do.", created_at=rec["created_iso"],
            ), None

        framed = security.wrap_inbound(peer, text)
        security.audit("inbound", peer, task_id, text)
        protocol.persist_message(context_id, "user", text, task_id)
        protocol.metrics.inbound_total += 1

        rec = self.tasks.create(task_id, context_id, peer, *self._scope_for_agent(agent))
        # Bind the session identity for this A2A context so session-aware
        # tooling (task auto-subscription, notifier routing) can send
        # notifications back to the peer's context. ContextVars (NOT a
        # process-global os.environ write) are the mechanism the tool
        # process reads via get_session_env: the asyncio Task created by
        # run_coroutine_threadsafe below snapshots THIS thread's context, so
        # the bindings ride the whole dispatch chain and stay task-local.
        # os.environ would be last-writer-wins across concurrent A2A
        # contexts and leak into sibling sessions.
        _session_tokens: list = []
        try:
            from gateway.session_context import set_session_vars
            _session_tokens = set_session_vars(
                platform="a2a",
                chat_id=context_id,
                chat_type="dm",
                chat_name=f"a2a:{peer}",
                thread_id=task_id,
                user_id=peer,
                user_name=peer,
                async_delivery=True,
            )
        except Exception as exc:
            logger.warning("A2A: set_session_vars unavailable: %s", exc)
        # Remember which peer owns this context so an out-of-band send with
        # no pending waiter can be pushed back to the caller's session.
        # Bounded: drop the oldest entry past _MAX_CONTEXT_PEERS (dicts keep
        # insertion order) so a long-running gateway can't grow this forever.
        with self._context_peers_lock:
            self._context_peers[context_id] = peer
            if len(self._context_peers) > _MAX_CONTEXT_PEERS:
                self._context_peers.pop(next(iter(self._context_peers)), None)
            # Write-through on inbound registration too: a gateway restart
            # wipes the in-memory map, and the wake self-post path (the task
            # notifier) bypasses this handler — so the disk copy is the only
            # thing that survives to the next start.
            _persist_context_peers(_merge_context_peers(_load_context_peers(), {context_id: peer}))
        self._register_inline_push(task_id, params, agent=agent)

        if not agent.get("local", True):
            try:
                reply, state = self._forward_to_profile(agent, peer, context_id, framed, task_id)
                self.tasks.complete(task_id, state, reply)
                protocol.persist_message(context_id, "agent", reply, task_id)
                security.audit("outbound", peer, task_id, reply, context_id=context_id)
                if state == protocol.STATE_COMPLETED:
                    protocol.metrics.outbound_total += 1
                    protocol.metrics.tasks_completed += 1
                else:
                    protocol.metrics.tasks_failed += 1
                self._send_push_notification(task_id, context_id, reply, state)
                return protocol.build_task(task_id, context_id, state, reply, created_at=rec["created_iso"]), None
            finally:
                if _session_tokens:
                    _reset_worker_session_vars()

        if self._loop is None or self._message_handler is None:
            self.tasks.complete(task_id, protocol.STATE_FAILED, "")
            protocol.metrics.tasks_failed += 1
            return protocol.build_task(
                task_id, context_id, protocol.STATE_FAILED,
                "Agent gateway not ready to accept A2A tasks.",
                created_at=rec["created_iso"],
            ), None

        fut = self._add_pending(task_id, context_id)

        event = MessageEvent(
            text=framed,
            message_type=MessageType.TEXT,
            source=self.build_source(
                chat_id=context_id,
                chat_name=f"a2a:{peer}",
                chat_type="dm",
                user_id=peer,
                user_name=peer,
            ),
            message_id=task_id,
        )

        try:
            asyncio.run_coroutine_threadsafe(self.handle_message(event), self._loop)
        except Exception as e:
            self._pop_pending(task_id)
            msg = security.redact_outbound(f"Dispatch failed: {e}")
            self.tasks.complete(task_id, protocol.STATE_FAILED, msg)
            protocol.metrics.tasks_failed += 1
            return protocol.build_task(
                task_id, context_id, protocol.STATE_FAILED, msg,
                created_at=rec["created_iso"],
            ), None
        finally:
            # The asyncio Task already snapshotted this thread's context
            # (run_coroutine_threadsafe copies it at creation), so the
            # session vars bound above ride the dispatch. Reset the HTTP
            # worker thread's own context so the bindings don't linger on
            # the threadpool thread for the next request.
            if _session_tokens:
                try:
                    _reset_worker_session_vars()
                except Exception:
                    pass

        # Wake the originating local session (an explicit fresh turn, not a
        # polled store read): when this context was born in a real gateway
        # session via a2a_call, the inbound message must ALSO trigger a fresh
        # turn there — the same self-post pattern the task watchers use — so
        # the agent that made the call can
        # act on the push. Fire-and-forget: the wake must never block or
        # fail the inbound dispatch (best-effort, logged inside).
        try:
            if self._context_sessions.get(context_id):
                asyncio.run_coroutine_threadsafe(
                    self._wake_origin_session(context_id, framed), self._loop
                )
        except Exception as exc:
            logger.debug(
                "A2A: could not schedule origin-session wake for %s: %s",
                context_id, exc,
            )

        self.tasks.set_state(task_id, protocol.STATE_WORKING)
        return None, {
            "task_id": task_id,
            "context_id": context_id,
            "peer": peer,
            "future": fut,
            "created_iso": rec["created_iso"],
            "started": time.time(),
        }

    def _restore_persisted_context_sessions(self) -> int:
        """Merge persisted context→origin-session registrations into memory.

        Called from ``connect()`` (and directly in tests): a gateway restart
        wipes the in-memory map, and the disk copy is the only thing that
        lets a push arriving right after the restart still wake its
        originating session. Returns the number of restored entries.
        """
        with self._context_sessions_lock:
            restored = _load_context_sessions()
            merged = _merge_context_sessions(self._context_sessions, restored)
            self._context_sessions.clear()
            self._context_sessions.update(merged)
        return len(restored)

    async def _wake_origin_session(self, context_id: str, text: str) -> None:
        """Wake the local session that created this A2A context (if any).

        The inbound message has been dispatched into the a2a session (its
        normal path, protocol continuity preserved). When the context was
        born in a REAL gateway session — a discord session called
        a2a_call, and a task notifier / workflow engine pushed a completion
        back on the same contextId — that originating session must ALSO get
        a fresh agent turn so the agent can ACT on the push (deliver
        artifacts, dispatch follow-ups), the same self-post mechanism the
        task watcher uses to wake a task creator. Visibility without a
        turn is what this fixes: the push used to land only in the
        conversation store, invisible unless manually polled.

        Runs on the gateway loop (scheduled from the HTTP worker thread via
        run_coroutine_threadsafe). Best-effort by design: a failed wake must
        never fail or slow the inbound task itself.
        """
        with self._context_sessions_lock:
            origin = dict(self._context_sessions.get(context_id) or {})
        if not origin:
            return
        origin_platform = str(origin.get("platform") or "").strip()
        if not origin_platform or origin_platform == "a2a":
            # An a2a-originated context's session IS the session the inbound
            # dispatch above already woke — waking again would double-inject
            # the same message into the same session.
            return

        # Resolve the adapter that owns the originating platform (discord,
        # telegram, api_server, ...). Iterate by platform VALUE so unknown /
        # non-Platform values (cli, tui) simply find no adapter.
        gw = getattr(self, "gateway_runner", None)
        adapter = None
        if gw is not None:
            for _p, _a in (getattr(gw, "adapters", None) or {}).items():
                if str(getattr(_p, "value", _p)) == origin_platform:
                    adapter = _a
                    break
        if adapter is None:
            logger.debug(
                "A2A: no %r adapter to wake origin session for context %s; skipping",
                origin_platform, context_id,
            )
            return

        from gateway.wake import adapter_supports_push, deliver_wake

        if adapter_supports_push(adapter):
            chat_id = str(origin.get("chat_id") or "").strip()
            if not chat_id:
                logger.debug(
                    "A2A: origin session for context %s has no chat_id; cannot wake",
                    context_id,
                )
                return
            from gateway.session import SessionSource

            source = SessionSource(
                platform=adapter.platform,
                chat_id=chat_id,
                chat_type=str(origin.get("chat_type") or "group") or "group",
                thread_id=str(origin.get("thread_id") or "").strip() or None,
                user_id=str(origin.get("user_id") or "").strip() or None,
                profile=str(origin.get("profile") or "").strip() or None,
            )
        else:
            source = None
        session_id = str(origin.get("session_id") or "").strip()

        try:
            await deliver_wake(
                adapter,
                text=text,
                session_id=session_id,
                source=source,
            )
            logger.info(
                "A2A: woke origin %s session for context %s (inbound push)",
                origin_platform, context_id,
            )
        except Exception as exc:
            # Best-effort: the a2a session already processed the message; a
            # broken origin wake must not surface into the task dispatch.
            logger.warning(
                "A2A: wake of origin %s session for context %s failed: %s",
                origin_platform, context_id, exc,
            )

    def _profile_state_db(self, profile: str) -> Optional[str]:
        home = _profile_home(profile)
        if not home:
            return None
        return os.path.join(home, "state.db")

    def _lookup_forward_session(self, profile: str, title: str) -> str:
        db = self._profile_state_db(profile)
        if not db or not os.path.exists(db):
            return ""
        try:
            con = sqlite3.connect(db, timeout=5)
            row = con.execute(
                "SELECT id FROM sessions WHERE title = ? ORDER BY started_at DESC LIMIT 1",
                (title,),
            ).fetchone()
            con.close()
            return str(row[0]) if row else ""
        except Exception:
            logger.debug("A2A: could not lookup forwarded session", exc_info=True)
            return ""

    def _latest_a2a_session(self, profile: str, started_after: float) -> str:
        db = self._profile_state_db(profile)
        if not db or not os.path.exists(db):
            return ""
        try:
            con = sqlite3.connect(db, timeout=5)
            row = con.execute(
                "SELECT id FROM sessions WHERE source = 'a2a' AND started_at >= ? ORDER BY started_at DESC LIMIT 1",
                (started_after - 2.0,),
            ).fetchone()
            con.close()
            return str(row[0]) if row else ""
        except Exception:
            logger.debug("A2A: could not find latest forwarded session", exc_info=True)
            return ""

    def _title_forward_session(self, profile: str, session_id: str, title: str) -> None:
        db = self._profile_state_db(profile)
        if not db or not os.path.exists(db) or not session_id:
            return
        try:
            con = sqlite3.connect(db, timeout=5)
            con.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
            con.commit()
            con.close()
        except Exception:
            logger.debug("A2A: could not title forwarded session", exc_info=True)

    def _forward_to_profile(self, agent: dict, peer: str, context_id: str, framed_text: str, task_id: str) -> tuple[str, str]:
        """Forward a routed A2A task to another local Hermes profile.

        First contact creates a normal ``source=a2a`` CLI session, records its
        session id, and titles it deterministically. Later turns resume by the
        concrete session id, not by a non-existent name. The public CLI boundary
        is preserved while giving A2A contexts stable multi-turn continuity.
        """
        profile = str(agent.get("profile") or agent.get("slug") or "").strip()
        slug = str(agent.get("slug") or profile or "agent")
        safe_ctx = _safe_context_slug(context_id)
        session_title = f"a2a-{slug}-{safe_ctx}"
        key = (profile or "default", slug, safe_ctx)
        timeout = int(agent.get("timeout") or _reply_timeout())

        lock = self._forward_lock(key)
        with lock:
            session_id = self._profile_sessions.get(key) or self._lookup_forward_session(profile, session_title)
            cmd = ["hermes", "chat", "-q", framed_text, "-Q", "--source", "a2a"]
            if session_id:
                cmd.extend(["--resume", session_id])

            env = os.environ.copy()
            home = _profile_home(profile)
            if home:
                env["HERMES_HOME"] = home
            env["HERMES_A2A_PEER"] = peer
            # Carry the A2A session identity into the forwarded profile's
            # agent subprocess. A CLI process reads these via
            # get_session_env's os.environ fallback, so task
            # auto-subscription + notifier routing can push completions back to
            # this context. Set on the child env only — never on the
            # process-global os.environ (last-writer-wins across concurrent
            # A2A contexts).
            env["HERMES_SESSION_PLATFORM"] = "a2a"
            env["HERMES_SESSION_CHAT_ID"] = context_id
            env["HERMES_SESSION_THREAD_ID"] = task_id
            start = time.time()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout,
                    env=env, check=False, stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired:
                return "[profile did not reply in time]", protocol.STATE_FAILED
            except Exception as e:
                return security.redact_outbound(f"Profile dispatch failed: {e}"), protocol.STATE_FAILED
            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or f"profile exited {proc.returncode}").strip()
                return security.redact_outbound(msg[-2000:]), protocol.STATE_FAILED
            if not session_id:
                session_id = self._latest_a2a_session(profile, start)
                if session_id:
                    self._profile_sessions[key] = session_id
                    self._title_forward_session(profile, session_id, session_title)
            return security.redact_outbound((proc.stdout or "").strip()), protocol.STATE_COMPLETED

    def _finalize_task(self, pending: dict, state: str, reply: str) -> tuple[str, str]:
        """Record the outcome of a dispatched task. Returns (state, reply) after
        redaction and input-required detection."""
        task_id = pending["task_id"]
        context_id = pending["context_id"]
        peer = pending["peer"]
        self._pop_pending(task_id)

        reply = security.redact_outbound(reply or "")

        # The agent flags clarification requests with a leading marker; map
        # them to the A2A input-required state so the peer knows to answer.
        if state == protocol.STATE_COMPLETED:
            stripped = reply.lstrip()
            if stripped.upper().startswith(protocol.INPUT_REQUIRED_MARKER):
                state = protocol.STATE_INPUT_REQUIRED
                reply = stripped[len(protocol.INPUT_REQUIRED_MARKER):].strip()

        protocol.persist_message(context_id, "agent", reply, task_id)
        security.audit("outbound", peer, task_id, reply, context_id=context_id)

        if state in (protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED):
            protocol.metrics.outbound_total += 1
            protocol.metrics.tasks_completed += 1
            protocol.metrics.record_latency(time.time() - pending["started"])
        else:
            protocol.metrics.tasks_failed += 1

        self.tasks.complete(task_id, state, reply)
        self._send_push_notification(task_id, context_id, reply, state)
        return state, reply

    def _patience_for(self, params: dict, peer: str) -> float:
        """Client patience for a blocking message/send.

        Priority: the message's stamped ``sender.timeout`` (the client's
        own advertised read timeout) → the configured
        ``a2a_agents[peer].timeout`` → 120s default.

        A peer-supplied timeout is capped at ``_ORPHAN_TIMEOUT -
        _PATIENCE_MARGIN`` (270s) so the patience deadline never exceeds
        the orphan watchdog horizon.  Non-finite, zero, negative, and
        over-ceiling values are clamped or rejected consistently.
        """
        _TIMEOUT_CEILING = _ORPHAN_TIMEOUT - _PATIENCE_MARGIN  # 270s
        msg = params.get("message") if isinstance(params, dict) else None
        sender = msg.get("sender") if isinstance(msg, dict) else None
        if isinstance(sender, dict):
            try:
                t = float(sender.get("timeout") or 0)
                if math.isfinite(t) and t > 0:
                    return min(t, _TIMEOUT_CEILING)
            except (TypeError, ValueError):
                pass
        try:
            from . import tools as a2a_tools
            entry = a2a_tools._resolve_peer(peer)
            if entry:
                t = float(entry.get("timeout") or 0)
                if math.isfinite(t) and t > 0:
                    return min(t, _TIMEOUT_CEILING)
        except Exception:
            logger.debug("A2A: could not resolve peer timeout for patience", exc_info=True)
        return 120.0

    def _mark_out_of_band(self, pending: dict, reason: str, pop_waiter: bool) -> None:
        """Record that a pending task's client is gone.

        ``reason`` is the audit marker (``[client patience exceeded]`` or
        ``[client disconnected]``). ``pop_waiter=True`` (probe-death) removes
        the task from the per-context waiter queue so the late reply takes
        the no-waiter push path in ``send()`` and the HTTP thread is freed.
        ``pop_waiter=False`` (patience exceeded — the socket may still look
        alive) keeps the waiter so the reply resolves normally; the handler
        then pushes it directly and skips the socket write. First mark wins:
        a later probe-death must not change the strategy mid-wait.
        """
        with self._pending_lock:
            if pending.get("out_of_band_only"):
                return
            pending["out_of_band_only"] = True
            if pop_waiter:
                order = self._pending_order.get(pending["context_id"])
                if order:
                    try:
                        order.remove(pending["task_id"])
                    except ValueError:
                        pass
                    if not order:
                        self._pending_order.pop(pending["context_id"], None)
        logger.info(
            "A2A: %s for task %s (context %s); reply will take the out-of-band push path",
            reason, pending["task_id"], pending["context_id"],
        )
        security.audit(
            "outbound", pending["peer"], pending["task_id"], reason,
            context_id=pending["context_id"],
        )

    def _try_push_reply(self, pending: dict, state: str, reply: str) -> bool:
        """Push a completed reply out-of-band, dedupe-guarded.

        Returns True when the reply was pushed (or a concurrent path already
        pushed it). Never raises into the caller.
        """
        if state not in (protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED) or not reply:
            return False
        with self._pending_lock:
            if pending.get("pushed"):
                return True
            pending["pushed"] = True
        try:
            self._push_out_of_band(pending["context_id"], reply, want_reply=True)
        except Exception as exc:
            logger.warning(
                "A2A: out-of-band push for task %s failed: %s",
                pending.get("task_id"), exc,
            )
            return False
        return True

    def _is_duplicate_inbound(self, context_id: str, message_id: str) -> bool:
        """Windowed (contextId, messageId) dedupe.

        Bounded map; expired entries are pruned when the cap is hit. Returns
        True when the same wire message was seen within the window.
        """
        key = (context_id, message_id)
        now = time.time()
        with self._inbound_seen_lock:
            if len(self._inbound_seen) > _INBOUND_DEDUPE_MAX:
                for k, ts in list(self._inbound_seen.items()):
                    if now - ts > _INBOUND_DEDUPE_WINDOW:
                        del self._inbound_seen[k]
                while len(self._inbound_seen) > _INBOUND_DEDUPE_MAX:
                    self._inbound_seen.pop(next(iter(self._inbound_seen)), None)
            seen = self._inbound_seen.get(key)
            if seen is not None and now - seen <= _INBOUND_DEDUPE_WINDOW:
                return True
            self._inbound_seen[key] = now
            return False

    def _await_reply(self, pending: dict, keepalive=None, patience: Optional[float] = None) -> tuple[str, str, bool]:
        """Block until the task's future resolves (or times out).

        ``keepalive`` is an optional zero-arg callable invoked every
        _SSE_KEEPALIVE seconds while waiting (used by the SSE paths); if it
        raises, the client is gone (probe-death): the waiter is popped and
        the loop returns immediately with ``out_of_band_only=True`` so the
        late reply takes the push path.

        ``patience`` (POST message/send only) is the client's advertised
        read timeout. When elapsed > patience + _PATIENCE_MARGIN
        the client has given up — or will discard — even if the socket
        still looks alive (the alive-but-will-discard client is invisible
        to any probe; its reply is silently consumed). The task is marked
        out_of_band_only and the loop KEEPS waiting for the reply so it can
        be pushed directly instead of written into the dead socket.

        Returns (state, reply, out_of_band_only).
        """
        fut: Future = pending["future"]
        deadline = pending["started"] + _reply_timeout()
        patience_deadline = (
            pending["started"] + patience + _PATIENCE_MARGIN
            if patience is not None else deadline
        )
        while True:
            now = time.time()
            wait = max(0.0, deadline - now)
            if not pending.get("out_of_band_only"):
                wait = min(wait, max(0.0, patience_deadline - now))
            if keepalive:
                wait = min(wait, _SSE_KEEPALIVE)
            try:
                state, reply = fut.result(timeout=wait)
                return state, reply, pending.get("out_of_band_only", False)
            except FuturesTimeout:
                now = time.time()
                if now >= deadline:
                    return (
                        protocol.STATE_FAILED, "[agent did not reply in time]",
                        pending.get("out_of_band_only", False),
                    )
                if keepalive:
                    try:
                        keepalive()
                    except Exception:
                        self._mark_out_of_band(pending, "[client disconnected]", pop_waiter=True)
                        return (protocol.STATE_FAILED, "[client disconnected]", True)
                if now >= patience_deadline:
                    self._mark_out_of_band(pending, "[client patience exceeded]", pop_waiter=False)
                    # Keep waiting: when the reply resolves it must be pushed
                    # directly and the socket write skipped — the client is
                    # gone even though the socket may look alive.
            except Exception:
                return (
                    protocol.STATE_FAILED, "[agent did not reply in time]",
                    pending.get("out_of_band_only", False),
                )

    def _rpc_message_send(self, req_id: Any, params: dict, peer: str, agent: Optional[dict] = None, v1_response: bool = False, client_alive=None) -> Optional[dict]:
        """Handle one blocking message/send JSON-RPC request.

        ``client_alive`` (optional zero-arg callable returning bool) is the
        HTTP handler's liveness probe for the waiting client: while the reply
        is pending, the waiter is probed every keepalive tick, and a dead
        client pops the pending task so the eventual reply takes the
        out-of-band push path instead of vanishing into the closed socket.

        Patience is the deterministic backstop: when the client's
        advertised read timeout (+ margin) passes without a reply, the task
        is marked out_of_band_only and the reply — when it resolves — is
        pushed directly.

        Returns the JSON-RPC result dict to write, or None when the reply
        was already pushed out-of-band and the socket write must be skipped
        entirely (out_of_band_only with a completed reply).
        """
        terminal, pending = self._prepare_task(params, peer, agent=agent)
        if terminal is not None:
            result = protocol.send_message_response(terminal) if v1_response else terminal
            return protocol.jsonrpc_result(req_id, result)
        assert pending is not None  # _prepare_task returns (terminal, None) or (None, pending)
        patience = self._patience_for(params, pending["peer"])
        if client_alive is not None:
            def _probe() -> None:
                if not client_alive():
                    raise ConnectionResetError(
                        "A2A client disconnected while awaiting reply"
                    )
            state, reply, out_of_band_only = self._await_reply(
                pending, keepalive=_probe, patience=patience)
        else:
            state, reply, out_of_band_only = self._await_reply(
                pending, patience=patience)
        state, reply = self._finalize_task(pending, state, reply)
        if out_of_band_only:
            # The client was known gone when the reply resolved (patience
            # exceeded) or was already detected dead (probe). A completed
            # reply must NOT be written into the dead socket — push it
            # directly and skip the write. Probe-death markers
            # carry no reply text and return the FAILED result as before.
            if state in (protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED) and reply:
                if self._try_push_reply(pending, state, reply):
                    return None
        task = protocol.build_task(
            pending["task_id"], pending["context_id"], state, reply,
            created_at=pending["created_iso"],
        )
        result = protocol.send_message_response(task) if v1_response else task
        return protocol.jsonrpc_result(req_id, result)

    # ── Streaming (SSE) ───────────────────────────────────────────────────

    @staticmethod
    def _sse_headers(handler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        # v1.0: closing the stream signals the terminal state, so the socket
        # must actually close once we emit the done event.
        handler.close_connection = True

    @staticmethod
    def _sse_write(handler, chunk: str) -> None:
        handler.wfile.write(chunk.encode("utf-8"))
        handler.wfile.flush()

    def _emit_terminal(self, handler, task_id: str, context_id: str, state: str, reply: str,
                       req_id: Any = None) -> None:
        """Emit the final artifact/status events and close the stream (v1.0:
        closure signals terminal state, no ``final`` field).

        ``req_id`` is threaded into JSON-RPC-wrapped SSE frames per §9.4."""
        if reply and state == protocol.STATE_COMPLETED:
            self._sse_write(handler, protocol.sse_data(
                protocol.artifact_update(task_id, context_id, reply), req_id))
            self._sse_write(handler, protocol.sse_data(
                protocol.status_update(task_id, context_id, state), req_id))
        else:
            self._sse_write(handler, protocol.sse_data(
                protocol.status_update(task_id, context_id, state, reply), req_id))
        self._sse_write(handler, protocol.sse_done())

    def _rpc_message_stream(self, handler, req_id: Any, params: dict, peer: str, agent: Optional[dict] = None) -> None:
        """Handle message/stream as an SSE response of JSON-RPC-wrapped
        StreamResponse events (A2A v1.0 §9.4)."""
        protocol.metrics.streams_started += 1
        self._sse_headers(handler)

        try:
            terminal, pending = self._prepare_task(params, peer, agent=agent)
            if terminal is not None:
                self._emit_terminal(
                    handler, terminal["id"], terminal["contextId"],
                    terminal["status"]["state"],
                    protocol.extract_text(terminal.get("status", {}).get("message", {}) or {}),
                    req_id=req_id,
                )
                return

            assert pending is not None  # _prepare_task returns (terminal, None) or (None, pending)
            task_id, context_id = pending["task_id"], pending["context_id"]
            self._sse_write(handler, protocol.sse_data(protocol.stream_task(
                protocol.build_task(task_id, context_id, protocol.STATE_SUBMITTED, created_at=pending["created_iso"])),
                req_id))
            self._sse_write(handler, protocol.sse_data(
                protocol.status_update(task_id, context_id, protocol.STATE_WORKING), req_id))

            state, reply, _ = self._await_reply(
                pending, keepalive=lambda: self._sse_write(handler, ": keepalive\n\n"),
                # SSE clients stay connected for the whole stream; the client
                # patience applies only to the blocking POST path.
                patience=None,
            )
            state, reply = self._finalize_task(pending, state, reply)
            self._emit_terminal(handler, task_id, context_id, state, reply, req_id=req_id)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("A2A: stream client disconnected")

    def _rpc_tasks_subscribe(self, handler, req_id: Any, params: dict, agent: Optional[dict] = None) -> None:
        """Reconnect to an existing task's stream (v1.0 SubscribeToTask)."""
        task_id = str(params.get("taskId") or params.get("id") or "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent))
        if not rec:
            handler._json(200, protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}"))
            return

        self._sse_headers(handler)
        try:
            fut = self.tasks.watch(task_id, *self._scope_for_agent(agent))
            if fut is None:
                self._sse_write(handler, protocol.sse_done())
                return
            deadline = time.time() + _reply_timeout()
            while True:
                try:
                    state, reply = fut.result(timeout=_SSE_KEEPALIVE)
                    break
                except FuturesTimeout:
                    if time.time() >= deadline:
                        state, reply = rec["state"], rec.get("reply", "")
                        break
                    self._sse_write(handler, ": keepalive\n\n")
            self._emit_terminal(handler, task_id, rec["context_id"], state, reply, req_id=req_id)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("A2A: subscribe client disconnected")

    # ── Task queries ──────────────────────────────────────────────────────

    def _rpc_tasks_get(self, req_id: Any, params: dict, agent: Optional[dict] = None) -> dict:
        task_id = str(params.get("taskId") or params.get("id") or "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent))
        if not rec:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")
        history_len = params.get("historyLength")
        try:
            history_len = int(history_len) if history_len is not None else None
        except (TypeError, ValueError):
            history_len = None
        return protocol.jsonrpc_result(req_id, protocol.TaskStore.to_task(rec, history_length=history_len))

    def _rpc_tasks_list(self, req_id: Any, params: dict, agent: Optional[dict] = None) -> dict:
        try:
            offset = int(params.get("pageToken") or 0)
        except (ValueError, TypeError):
            offset = 0
        try:
            page_size = int(params.get("pageSize") or 50)
        except (ValueError, TypeError):
            page_size = 50
        recs, next_offset, total = self.tasks.list(
            context_id=str(params.get("contextId") or ""),
            state=str(params.get("status") or params.get("state") or ""),
            page_size=page_size,
            offset=max(0, offset),
            agent_slug=self._scope_for_agent(agent)[0],
            tenant=self._scope_for_agent(agent)[1],
            with_total=True,
        )
        include_artifacts = bool(params.get("includeArtifacts", False))
        history_len = params.get("historyLength")
        try:
            history_len = int(history_len) if history_len is not None else None
        except (TypeError, ValueError):
            history_len = None
        return protocol.jsonrpc_result(req_id, {
            "tasks": [protocol.TaskStore.to_task(r, history_length=history_len, include_artifacts=include_artifacts) for r in recs],
            "nextPageToken": str(next_offset) if next_offset else "",
            "pageSize": max(1, min(page_size, 100)),
            "totalSize": total,
        })

    def _rpc_tasks_cancel(self, req_id: Any, params: dict, agent: Optional[dict] = None) -> dict:
        task_id = str(params.get("taskId") or params.get("id") or "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent))
        if not rec:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")
        if rec["state"] in protocol.TERMINAL_STATES:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_CANCELABLE,
                f"task {task_id} already {rec['state']}")
        self.tasks.complete(task_id, protocol.STATE_CANCELED, "")
        self._turns.reset(rec["context_id"])
        self._resolve_task(task_id, protocol.STATE_CANCELED, "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent)) or rec
        return protocol.jsonrpc_result(req_id, protocol.TaskStore.to_task(rec))

    # ── Push notifications ────────────────────────────────────────────────

    def _register_inline_push(self, task_id: str, params: dict, agent: Optional[dict] = None) -> None:
        """v1.0: message/send can carry configuration.taskPushNotificationConfig."""
        cfg = (params.get("configuration") or {}).get("taskPushNotificationConfig") or {}
        if not isinstance(cfg, dict):
            return
        url = cfg.get("url") or (cfg.get("pushNotificationConfig") or {}).get("url") or ""
        if url:
            self.tasks.set_push_config(task_id, str(url), *self._scope_for_agent(agent))

    def _rpc_push_config_create(self, req_id: Any, params: dict, agent: Optional[dict] = None) -> dict:
        task_id = str(params.get("taskId") or "")
        cfg = params.get("pushNotificationConfig") or params.get("config") or {}
        url = str((cfg or {}).get("url") or "")
        if not task_id or not url:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS,
                "taskId and pushNotificationConfig.url required")
        stored = self.tasks.set_push_config(task_id, url, *self._scope_for_agent(agent))
        if stored is None:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")
        return protocol.jsonrpc_result(req_id, stored)

    def _rpc_push_config_get(self, req_id: Any, params: dict, agent: Optional[dict] = None) -> dict:
        """GetTaskPushNotificationConfig — retrieve a push config by task id."""
        task_id = str(params.get("taskId") or "")
        config_id = str(params.get("id") or params.get("configId") or "")
        if not task_id:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        cfg = self.tasks.get_push_config(task_id, config_id, *self._scope_for_agent(agent))
        if cfg is None:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND,
                f"push config not found for task: {task_id}")
        return protocol.jsonrpc_result(req_id, cfg)

    def _rpc_push_config_list(self, req_id: Any, params: dict, agent: Optional[dict] = None) -> dict:
        """ListTaskPushNotificationConfigs — list push configs for a task."""
        task_id = str(params.get("taskId") or "")
        if not task_id:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        configs = self.tasks.list_push_configs(task_id, *self._scope_for_agent(agent))
        return protocol.jsonrpc_result(req_id, {"configs": configs, "nextPageToken": ""})

    def _rpc_push_config_delete(self, req_id: Any, params: dict, agent: Optional[dict] = None) -> dict:
        """DeleteTaskPushNotificationConfig — remove a push config."""
        task_id = str(params.get("taskId") or "")
        config_id = str(params.get("id") or params.get("configId") or "")
        if not task_id:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        deleted = self.tasks.delete_push_config(task_id, config_id, *self._scope_for_agent(agent))
        if not deleted:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND,
                f"push config not found for task: {task_id}")
        return protocol.jsonrpc_result(req_id, {"deleted": True})

    def _send_push_notification(self, task_id: str, context_id: str, reply: str, state: str) -> None:
        """POST a v1.0 StreamResponse payload to the task's registered callback.

        Validates the callback URL to prevent SSRF — blocks internal/private
        addresses (169.254.x.x metadata, loopback, RFC1918 private ranges)
        unless we're in localhost-only mode (where internal access is expected).
        """
        callback_url = self.tasks.pop_push_url(task_id)
        if not callback_url:
            return

        if not security.is_safe_callback_url(callback_url):
            logger.warning("A2A: push notification for task %s blocked — unsafe callback URL: %s",
                           task_id, callback_url)
            protocol.metrics.push_failed += 1
            return

        # Push payload uses the StreamResponse format (same as streaming).
        payload = protocol.status_update(task_id, context_id, state, (reply or "")[:2000])

        signature = security.sign_push_payload(payload)
        headers = {"Content-Type": "application/json"}
        if signature:
            headers["X-A2A-Signature"] = signature

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(callback_url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                if 200 <= resp.status < 300:
                    protocol.metrics.push_sent += 1
                    logger.debug("A2A: push notification sent for task %s", task_id)
                else:
                    protocol.metrics.push_failed += 1
                    logger.warning("A2A: push notification for task %s got HTTP %d", task_id, resp.status)
        except Exception as e:
            protocol.metrics.push_failed += 1
            logger.warning("A2A: push notification for task %s failed: %s", task_id, e)

    # ── Sending (the agent's reply path) ──────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Fulfil the pending reply Future for this context.

        ``chat_id`` is the A2A context id we set as the source chat_id; the
        oldest outstanding task for that context receives the reply (the
        gateway session processes messages in order).

        The gateway marks final user-visible replies with ``metadata['notify']``
        (see ``_mark_notify_metadata`` in gateway.platforms.base — this is the
        base adapter's documented reply marker, not an incidental field).
        Progress, status, and editable preview sends intentionally lack the
        marker; those must not satisfy the JSON-RPC caller.
        """
        message_id = str(int(time.time() * 1000))
        if not (metadata or {}).get("notify"):
            logger.debug("A2A: ignoring non-final send for context %s", chat_id)
            return SendResult(success=True, message_id=message_id)
        if self._resolve_oldest_for_context(chat_id, protocol.STATE_COMPLETED, content or ""):
            return SendResult(success=True, message_id=message_id)
        # No waiter (e.g. a late chunk or out-of-band send) — push the message
        # back to the peer that owns this context as a NEW task, reusing the
        # same contextId so it lands in the caller's session (session
        # continuity). Without this, task-notifier wake replies and late
        # completions were silently dropped while reporting success.
        #
        # Loopback self-push guard: in localhost-only mode every inbound
        # caller authenticates as "ip:<addr>" with no port, so the only
        # resolvable target for a loopback identity is THIS gateway's own
        # endpoint. Self-pushing is correct for the notifier's
        # completion delivery — the sub's user_id is the loopback identity
        # and the message must re-enter the owning session (the watcher marks
        # that send with metadata["a2a_push"]=True). A session's own REPLY
        # must never be re-queued into the same session: that produced an
        # unbounded self-ping-pong once the loopback fallback became
        # resolvable — every reply was pushed back,
        # processed, and answered again forever. Unmarked sends to a
        # loopback peer are replies with no external destination — that is
        # a LOUD failure, not a silent success: a
        # helper-sent message refined to "ip:127.0.0.1" and its long reply
        # was dropped here with success=True and no
        # audit. The notifier/engine must rewind instead of advancing past a
        # lost event.
        if not (metadata or {}).get("a2a_push"):
            with self._context_peers_lock:
                _loop_peer = self._context_peers.get(chat_id, "")
            if _loop_peer and _loopback_fallback_url(_loop_peer, self.host, self.port):
                security.audit(
                    "push_dropped", _loop_peer, message_id,
                    "peer identity not resolvable", context_id=chat_id,
                )
                logger.warning(
                    "A2A: dropping out-of-band send for %s: loopback peer %r "
                    "is unresolvable and the message is an unmarked session "
                    "reply — no external destination (success=False)",
                    chat_id, _loop_peer,
                )
                return SendResult(
                    success=False, message_id=message_id,
                    error="peer identity not resolvable",
                )
        try:
            # want_reply=True for session-reply pushes (the peer answers
            # inside the push's HTTP response — the round-trip path); False
            # for notifier pushes (fire-and-forget, unchanged).
            await asyncio.to_thread(
                self._push_out_of_band, chat_id, content or "",
                not (metadata or {}).get("a2a_push"),
            )
        except Exception as exc:
            logger.warning("A2A: out-of-band push for context %s failed: %s", chat_id, exc)
            return SendResult(success=False, message_id=message_id, error=str(exc))
        return SendResult(success=True, message_id=message_id)

    def _push_out_of_band(self, context_id: str, text: str, want_reply: bool = False) -> None:
        """POST a new message/send to the peer that owns ``context_id``.

        Runs on a worker thread (blocking urllib). Resolves the peer from
        ``a2a_agents`` config by the identity recorded on the inbound task;
        unknown peers are a no-op (nothing to push to). The outbound message
        reuses the SAME contextId so the caller's Hermes routes it into the
        session that originally made the request.

        ``want_reply=True`` for session-reply pushes (``adapter.send()``
        without ``metadata["a2a_push"]``, and the dead-client rescues) —
        the round-trip path: the peer answers inside the push's HTTP response
        (a verdict arriving this way was previously
        discarded), so a non-empty reply is re-dispatched into the LOCAL
        gateway as an inbound message on the same contextId (loopback
        in-process path, origin-session wake). Never raises into the push
        caller. ``want_reply=False`` for notifier pushes —
        fire-and-forget, unchanged.

        Reply-path pushes never fall back to this gateway's own endpoint:
        an unresolvable reply peer is a LOUD failure (audit
        ``push_dropped`` + warning), not a self-ping-pong or a silent drop.
        The own-gateway loopback fallback is reserved for ``a2a_push``
        notifications.
        """
        with self._context_peers_lock:
            peer = self._context_peers.get(context_id, "")
        if not peer:
            logger.debug("A2A: out-of-band send for %s has no known peer; dropping", context_id)
            return
        from . import tools as a2a_tools

        entry = a2a_tools._resolve_peer(peer)
        if not entry or not entry.get("url"):
            # Localhost-only mode records inbound callers as "ip:<addr>" with
            # no port, and there is no a2a_agents key for the raw identity.
            # When the identity is a loopback address, the notifier path
            # falls back to this gateway's own A2A endpoint — the one local
            # endpoint guaranteed to route a same-contextId follow-up into
            # the session that owns the conversation (a registered
            # loopback "ip:" identity carries no port, so without the
            # fallback the push dropped).
            fallback = _loopback_fallback_url(peer, self.host, self.port)
            if fallback:
                if want_reply:
                    self._drop_unresolvable_reply(context_id, peer)
                    return
                logger.info(
                    "A2A: out-of-band send for %s: identity %r not in a2a_agents; "
                    "falling back to local endpoint %s",
                    context_id, peer, fallback,
                )
                # The fallback URL is THIS gateway's own endpoint (it is
                # built from self.host/self.port), so an HTTP round-trip
                # would be a synchronous self-call: the inbound handler only
                # answers after the agent session processes the message,
                # which routinely exceeds the client timeout — the audit
                # row + reply log never ran and the notifier logged a false
                # failure. Deliver in-process
                # instead: the exact same code path as an inbound
                # message/send, minus the connection and the wait.
                self._push_loopback_in_process(context_id, peer, text)
                return
            else:
                logger.debug("A2A: out-of-band send for %s: peer %r not configured; dropping", context_id, peer)
                return
        base_url = entry["url"]
        # Own-endpoint guard: if the resolved target is THIS gateway (the
        # context→peer map can be refined to our own URL — an in-process
        # loopback push stamps our own sender, and the inbound refinement
        # accepts it), deliver in-process instead of a synchronous HTTP
        # self-call. The inbound handler only answers after the session
        # processes the message, which routinely exceeds the client timeout.
        # Reply pushes (want_reply=True) refuse this fallback — an
        # unresolvable reply peer fails loudly.
        if _is_own_endpoint(base_url, self.host, self.port):
            if want_reply:
                self._drop_unresolvable_reply(context_id, peer)
                return
            logger.info(
                "A2A: out-of-band send for %s: resolved peer %r is this gateway "
                "(%s); delivering in-process",
                context_id, peer, base_url,
            )
            self._push_loopback_in_process(context_id, peer, text)
            return
        headers = a2a_tools._auth_header(entry.get("auth", {}) or {})
        timeout = int(entry.get("timeout", 120))
        card = None
        try:
            card = a2a_tools._fetch_card(base_url, headers, min(timeout, 30))
        except Exception:
            pass
        rpc_body = {
            "jsonrpc": "2.0",
            "id": protocol.new_task_id(),
            "method": "SendMessage",
            "params": {
                "message": protocol.text_message(
                    protocol.ROLE_USER, text, context_id=context_id, sender=self._sender_identity()
                ),
            },
        }
        tenant = a2a_tools._interface_tenant(card, entry)
        if tenant:
            rpc_body["params"]["tenant"] = tenant
        resp = None
        try:
            resp = a2a_tools._http_post_json(a2a_tools._rpc_url(base_url, card), rpc_body, headers, timeout)
        finally:
            # Bookkeeping runs even when the client times out: the receiving
            # gateway answers only after its agent session finishes, which
            # can exceed the client timeout even though the message was
            # delivered. The audit row + reply log are the
            # delivery records the notifier path relies on.
            protocol.persist_message(context_id, "agent", text)
            # The 'push' audit direction is documented in security.py but had no
            # caller — every out-of-band push was invisible in a2a_audit.jsonl,
            # which made diagnosing the push pipeline harder. Best-effort, like
            # every other audit call.
            security.audit("push", peer, rpc_body["id"], text, context_id=context_id)
            logger.info("A2A: pushed out-of-band reply for context %s to peer %s", context_id, peer)
        if want_reply and resp is not None:
            # Round-trip: the peer answered inside the push's HTTP
            # response — a verdict arriving this way was previously
            # discarded exactly here. Surface
            # a non-empty reply into the LOCAL gateway as an inbound message
            # on the same contextId, the exact path a remote push would take
            # (wrap_inbound framing + origin-session wake). Never raises into
            # the push caller.
            try:
                reply = a2a_tools._reply_text_from_result(resp.get("result"))
                if reply:
                    self._push_loopback_in_process(context_id, peer, reply)
            except Exception as exc:
                logger.warning(
                    "A2A: could not surface push reply for context %s: %s",
                    context_id, exc,
                )

    def _drop_unresolvable_reply(self, context_id: str, peer: str) -> None:
        """Loud failure for a reply push with no resolvable external target.

        An unmarked session reply whose peer is an unresolvable
        loopback identity must never be silently dropped
        and must never self-loop into this gateway's own session
        (unbounded self-ping-pong). Audit ``push_dropped`` + warn; the
        caller surfaces success=False.
        """
        security.audit(
            "push_dropped", peer, "", "peer identity not resolvable",
            context_id=context_id,
        )
        logger.warning(
            "A2A: out-of-band REPLY for context %s dropped: peer identity %r "
            "is not resolvable (no external destination)",
            context_id, peer,
        )

    def _push_reply_after_client_gone(self, req_id: Any, result: Optional[dict]) -> None:
        """Deliver a completed reply whose HTTP client disconnected first.

        Safety net for the blocking message/send path: the peer's
        ``a2a_call`` client can time out (120s) and close the connection
        while the agent is still working; the reply then resolves the stale
        pending task and the JSON-RPC response write hits the closed socket
        (a round-2 drop: a reply was consumed by a dead waiter and
        written into a dead connection — the caller's session never woke).
        The liveness probe in ``_rpc_message_send`` pops the stale waiter in
        the common case; this catches the probe-race window where the client
        dies between the last probe and the response write. Push the reply
        out-of-band on the same contextId (the no-waiter path), so the
        caller's session still receives it.

        Only COMPLETED / INPUT_REQUIRED replies with text are pushed —
        failed or empty tasks carry nothing worth delivering out-of-band.
        """
        try:
            inner = (result or {}).get("result")
            task = protocol.unwrap_send_message_response(inner)
            if not isinstance(task, dict):
                return
            context_id = str(task.get("contextId") or "").strip()
            state = str((task.get("status") or {}).get("state") or "").strip()
            if not context_id or state not in (
                protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED,
            ):
                logger.debug(
                    "A2A: not pushing reply after client disconnect for %s (state=%r)",
                    context_id, state,
                )
                return
            reply = protocol.extract_text((task.get("status") or {}).get("message", {}) or {})
            if not reply:
                return
            # Session-reply rescue: want_reply=True so the peer's
            # answer to this pushed reply re-enters our session from the
            # push's HTTP response instead of being discarded.
            self._push_out_of_band(context_id, reply, want_reply=True)
            logger.info(
                "A2A: client disconnected before response write; pushed reply "
                "for context %s out-of-band",
                context_id,
            )
        except Exception as exc:
            logger.warning(
                "A2A: could not push reply after client disconnect (req %s): %s",
                req_id, exc,
            )

    def _push_loopback_in_process(self, context_id: str, peer: str, text: str) -> None:
        """Deliver an out-of-band push to this gateway's own session in-process.

        Reuses the exact inbound path an HTTP message/send would take
        (``_prepare_task`` → event dispatch into the owning session) but
        without the synchronous wait for the agent's reply — the HTTP
        worker blocks on that wait, which is what made the loopback push
        time out on the client side (the handler answers only after the
        session processes the message). The pending task created here is
        still resolved by the session's normal reply path (``send()`` /
        ``on_processing_complete``), exactly as for an HTTP inbound task.
        """
        params = {
            "message": protocol.text_message(
                protocol.ROLE_USER, text, context_id=context_id, sender=self._sender_identity()
            ),
        }
        terminal, pending = self._prepare_task(params, peer)
        if terminal is not None:
            state = (terminal.get("status") or {}).get("state", "unknown")
            raise RuntimeError(
                f"in-process loopback push for context {context_id} rejected ({state})"
            )
        assert pending is not None  # _prepare_task returns (terminal, None) or (None, pending)
        protocol.persist_message(context_id, "agent", text)
        security.audit("push", peer, pending["task_id"], text, context_id=context_id)
        logger.info("A2A: pushed out-of-band reply for context %s to peer %s", context_id, peer)

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Resolve the task future when processing ends without a reply send.

        The success path resolves via send(); this hook catches failures,
        cancellations, and empty runs so the HTTP thread returns promptly
        instead of waiting out the reply timeout.
        """
        task_id = str(getattr(event, "message_id", "") or "")
        if not task_id:
            return
        if outcome == ProcessingOutcome.FAILURE:
            self._resolve_task(task_id, protocol.STATE_FAILED, "[agent processing failed]")
        elif outcome == ProcessingOutcome.CANCELLED:
            self._resolve_task(task_id, protocol.STATE_CANCELED, "")
        else:
            self._resolve_task(task_id, protocol.STATE_COMPLETED, "")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": f"a2a:{chat_id}", "type": "dm"}
