"""A2A inbound adapter: stdlib http.server (daemon thread) serving the Agent Card, /metrics and
JSON-RPC (message/send, message/stream SSE, tasks/*, push-config CRUD). Inbound tasks are framed
(security.wrap_inbound) and routed into the LIVE gateway session; ``send()`` fulfils the per-task
Future the HTTP handler blocks on. No token configured => binds 127.0.0.1 only."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, ProcessingOutcome, SendResult
from gateway.config import Platform
from gateway.platforms._shared import coerce_port as _to_int, profile_scoped as _profile_scoped

from . import protocol, security

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 9900
_ORPHAN_TIMEOUT, _WATCHDOG_INTERVAL = 300, 60  # seconds: pending task considered orphaned / watchdog period
_MAX_BODY = 1_048_576  # 1MB max request body — prevents DoS via memory exhaustion
_SSE_KEEPALIVE = 5  # seconds between SSE keepalive comments
_DEFAULT_DESCRIPTION = "Hermes Agent — a general-purpose agent reachable over A2A."

_ok = protocol.jsonrpc_result
_err = protocol.jsonrpc_error

# (adapter handler, v1.0 PascalCase method per §5.3/§9.4, *legacy slash aliases still accepted)
_METHOD_TABLE = (
    ("_rpc_message_send", "SendMessage", "message/send"),
    ("_rpc_message_stream", "SendStreamingMessage", "message/stream"),
    ("_rpc_tasks_get", "GetTask", "tasks/get"),
    ("_rpc_tasks_list", "ListTasks", "tasks/list"),
    ("_rpc_tasks_cancel", "CancelTask", "tasks/cancel"),
    ("_rpc_tasks_subscribe", "SubscribeToTask", "tasks/subscribe"),
    ("_rpc_push_config_create", "CreateTaskPushNotificationConfig", "tasks/pushNotificationConfig/create",
     "tasks/pushNotificationConfig/set", "tasks/pushNotification/set"),
    ("_rpc_push_config_get", "GetTaskPushNotificationConfig", "tasks/pushNotificationConfig/get"),
    ("_rpc_push_config_list", "ListTaskPushNotificationConfigs", "tasks/pushNotificationConfig/list"),
    ("_rpc_push_config_delete", "DeleteTaskPushNotificationConfig", "tasks/pushNotificationConfig/delete"),
)
# JSON-RPC method -> (adapter handler name, is_v1)
_METHODS: dict[str, tuple[str, bool]] = {
    m: (handler, i == 0) for handler, *methods in _METHOD_TABLE for i, m in enumerate(methods)
}


def _reply_timeout() -> float:
    """Seconds to wait for the agent to answer an inbound task."""
    try:
        return max(1.0, float(os.getenv("A2A_REPLY_TIMEOUT", "300")))
    except (ValueError, TypeError):
        return 300.0


def _default_agent_name() -> str:
    # Scope-aware: a secondary multiplex profile must not borrow the default profile's A2A_AGENT_NAME.
    name = "" if _profile_scoped() else os.getenv("A2A_AGENT_NAME", "").strip()
    if name:
        return name
    try:
        import socket
        return f"hermes-{socket.gethostname()}"
    except Exception:
        return "hermes-agent"


def _clean_slug(value: str) -> str:
    """URL-safe-ish single-segment slug for a served agent ("" for the root agent)."""
    slug = str(value or "").strip().strip("/")
    return "" if slug in ("", "default", "root") else slug.split("/")[0]


def _join_url(base: str, prefix: str) -> str:
    base = (base or "").strip() or "/"
    base = base if base.endswith("/") else base + "/"
    prefix = (prefix or "").strip("/")
    return urllib.parse.urljoin(base, prefix + "/") if prefix else base


def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return os.getenv("HERMES_PROFILE", "default") or "default"


def _profile_home(profile: str) -> Optional[str]:
    with contextlib.suppress(Exception):
        from hermes_cli.profiles import get_profile_dir
        return str(get_profile_dir(profile))
    if profile and profile != "default":
        return os.path.expanduser(f"~/.hermes/profiles/{profile}")
    with contextlib.suppress(Exception):
        from hermes_cli.config import get_hermes_home
        return str(get_hermes_home())
    return None


def _daemon_thread(target, name: str) -> threading.Thread:
    t = threading.Thread(target=target, name=name, daemon=True)
    t.start()
    return t


def _safe_context_slug(value: str, max_len: int = 96) -> str:
    """Sanitize attacker-provided context ids before using in session titles."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return (slug or "ctx")[:max_len]


def _state_db(profile: str, sql: str, params: tuple, log_msg: str, *, commit: bool = False) -> str:
    """Run one statement against a profile's state.db; first column of the first row or ""."""
    home = _profile_home(profile)
    db = os.path.join(home, "state.db") if home else ""
    if not db or not os.path.exists(db):
        return ""
    try:
        with contextlib.closing(sqlite3.connect(db, timeout=5)) as con:
            cur = con.execute(sql, params)
            row = None if commit else cur.fetchone()
            if commit:
                con.commit()
        return str(row[0]) if row else ""
    except Exception:
        logger.debug(log_msg, exc_info=True)
        return ""


class A2ARequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for the A2A JSON-RPC surface; all state lives on ``self.server.adapter``."""

    @property
    def adapter(self) -> "A2AAdapter":
        return self.server.adapter  # type: ignore[attr-defined]

    def log_message(self, format, *args):  # noqa: A002,N802
        logger.debug("A2A http: " + format, *args)  # silence the default stderr access log

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        for k, v in (("Content-Type", "application/json"), ("Content-Length", str(len(body)))):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, http_code: int, req_id: Any, code: int, message: str):
        self._json(http_code, _err(req_id, code, message))

    def _client_ip(self) -> str:
        return self.client_address[0] if self.client_address else ""

    def _request_public_url(self) -> str:
        """A2A_PUBLIC_URL > X-Forwarded-Host / Host (scheme from X-Forwarded-Proto) > "" (bind host).

        Empty means "caller has no info, fall back to bind host". See #41711.
        """
        explicit = os.getenv("A2A_PUBLIC_URL", "").strip()
        if explicit:
            return explicit
        host = (self.headers.get("X-Forwarded-Host", "") or self.headers.get("Host", "")).split(",")[0].strip()
        scheme = (self.headers.get("X-Forwarded-Proto", "") or "http").split(",")[0].strip()
        return f"{scheme}://{host}/" if host else ""

    def do_GET(self):  # noqa: N802
        adapter = self.adapter
        route = adapter._route_for_path(self.path)
        agent = route["agent"]
        subpath = route["subpath"].rstrip("/") or "/"
        public_url = self._request_public_url() or None
        if subpath in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
            return self._json(200, adapter._build_card(public_url, agent=agent))
        if subpath == "/metrics":
            return self._json(200, protocol.metrics.snapshot())
        if subpath not in ("/", "/health"):
            return self._json(404, {"error": "not found"})
        payload = {"status": "ok", "agent": agent.get("name") or adapter.agent_name}
        # Agent Cards are public; profile/tenant topology is not leaked on remote unauthenticated GETs.
        sec = adapter._security_context
        if sec.localhost_only() or sec.authenticate(self.headers.get("Authorization"), self._client_ip()) is not None:
            payload["served_agents"] = adapter._served_agent_summary(public_url=public_url)
        self._json(200, payload)

    def do_POST(self):  # noqa: N802
        adapter = self.adapter
        # Identity comes from the credential (or the socket in localhost-only mode) — never the body.
        identity = adapter._security_context.authenticate(self.headers.get("Authorization"), self._client_ip())
        if identity is None:
            return self._error(401, None, protocol.ERR_UNAUTHORIZED, "unauthorized")
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > _MAX_BODY:
                return self._error(413, None, protocol.ERR_PARSE, "payload too large")
            req = json.loads((self.rfile.read(length) if length else b"{}").decode("utf-8"))
        except Exception:
            return self._error(400, None, protocol.ERR_PARSE, "parse error")
        if not isinstance(req, dict):
            return self._error(400, None, protocol.ERR_INVALID_PARAMS, "JSON-RPC request must be an object")
        req_id, method = req.get("id"), str(req.get("method", ""))
        params = req["params"] if req.get("params") is not None else {}
        version = (self.headers.get("A2A-Version") or "").strip()
        route = adapter._route_for_request(self.path, params) if isinstance(params, dict) else {}
        handler_name, is_v1 = _METHODS.get(method, ("", False))
        # Ordered, lazily-evaluated checks -> (http status, error code, message); first failure wins
        # (the rate limiter must not be consulted for requests rejected before it).
        checks = (
            (lambda: not isinstance(params, dict), 200, protocol.ERR_INVALID_PARAMS, "params must be an object"),
            (lambda: version and version not in {"1.0", "1.0.0"}, 200, protocol.ERR_INVALID_PARAMS, f"unsupported A2A-Version: {version}"),
            (lambda: route.get("error"), 400, protocol.ERR_INVALID_PARAMS, route.get("error")),
            (lambda: not adapter._rate_limiter.allow(identity), 429, protocol.ERR_RATE_LIMITED, "rate limit exceeded"),
            (lambda: not adapter._security_context.is_trusted_peer(identity), 403, protocol.ERR_UNTRUSTED_PEER, f"peer '{identity}' not trusted"),
            (lambda: not handler_name, 200, protocol.ERR_METHOD_NOT_FOUND, f"method not found: {method}"),
        )
        for failed, http, code, msg in checks:
            if failed():
                if code == protocol.ERR_RATE_LIMITED:
                    protocol.metrics.rate_limit_triggers += 1
                return self._error(http, req_id, code, msg)
        agent = route["agent"]
        if handler_name == "_rpc_message_send":
            self._json(200, adapter._rpc_message_send(req_id, params, identity, agent=agent, v1_response=is_v1))
        elif handler_name == "_rpc_message_stream":
            adapter._rpc_message_stream(self, req_id, params, identity, agent=agent)
        elif handler_name == "_rpc_tasks_subscribe":
            adapter._rpc_tasks_subscribe(self, req_id, params, agent=agent)
        else:  # plain JSON task / push-config queries
            self._json(200, getattr(adapter, handler_name)(req_id, params, agent=agent))


class A2AAdapter(BasePlatformAdapter):
    """Inbound A2A server adapter."""

    def __init__(self, config, **kwargs):
        super().__init__(config=config, platform=Platform("a2a"))
        extra = getattr(config, "extra", {}) or {}
        # Scope-aware: a secondary multiplex profile must not borrow the default profile's bridged
        # A2A_PORT (falls closed to the module default). advertised_toolsets is deliberately unscoped.
        # (advertised_toolsets has the same env-leak shape but is left unscoped here — see the "Scope note"
        # in this fix's PR description: open PR #98937 is actively rewriting this field's None-vs-empty-list
        # semantics.)
        self._security_context = security.A2ASecurityContext.capture()
        _port_env = None if _profile_scoped() else os.getenv("A2A_PORT")
        self.port = int(_port_env or extra.get("port", _DEFAULT_PORT))
        self.host = self._security_context.resolve_bind_host()
        self.agent_name = _default_agent_name()
        configured_toolsets = list(extra.get("advertised_toolsets") or []) or os.getenv("A2A_ADVERTISED_TOOLSETS", "").split(",")
        self._advertised_toolsets = [t.strip() for t in configured_toolsets if str(t).strip()]
        self._active_profile = _active_profile_name()
        self._agents = self._load_served_agents(extra)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._server_thread = self._watchdog_thread = None  # type: Optional[threading.Thread]
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._watchdog_stop = threading.Event()
        # Per-adapter protocol state (not module-global).
        self.tasks, self._turns, self._rate_limiter = protocol.TaskStore(), protocol.TurnTracker(), protocol.RateLimiter()
        # Forwarded profile sessions: (profile, agent_slug, context_id) -> session_id.
        self._profile_sessions: Dict[tuple[str, str, str], str] = {}
        self._profile_session_locks: Dict[tuple[str, str, str], threading.Lock] = {}
        self._profile_session_locks_guard = threading.Lock()
        # Pending reply futures: task_id -> (context_id, Future). _pending_order keeps per-context
        # FIFO so adapter.send() — which only knows the context — resolves the oldest task.
        self._pending: Dict[str, tuple[str, Future]] = {}
        self._pending_order: Dict[str, deque[str]] = {}
        self._pending_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "A2A"

    @property
    def authorization_is_upstream(self) -> bool:
        """Requests are authenticated in ``do_POST``; the gateway's A2A_ALLOWED_USERS list would
        otherwise reject peers (identity is a token-derived name or IP). Wrong credentials still 401.

        This is authorization delegated to the A2A bearer-token transport, not a fail-open: every request is
        401'd if the credential is wrong. Reported by kuangmi-bit (PR #41711 comment, Jun 27).
        """
        return True

    async def connect(self, **_kwargs) -> bool:
        # Capture the gateway loop so the HTTP thread can marshal events via run_coroutine_threadsafe.
        self._loop = asyncio.get_running_loop()
        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), A2ARequestHandler)
        except OSError as e:
            logger.error("A2A: could not bind %s:%s — %s", self.host, self.port, e)
            self._set_fatal_error("bind_failed", f"A2A bind failed: {e}", retryable=True)
            return False
        self._httpd.daemon_threads = True
        self._httpd.adapter = self  # type: ignore[attr-defined]
        self._server_thread = _daemon_thread(self._httpd.serve_forever, "a2a-http")
        self._watchdog_stop.clear()  # disconnect sets it; reset for reconnection
        self._watchdog_thread = _daemon_thread(self._watchdog_loop, "a2a-watchdog")
        self._mark_connected()
        logger.info("A2A: serving Agent Card + JSON-RPC on http://%s:%s (%s) as %r; %d routed agent(s)", self.host, self.port,
                    "localhost-only" if self._security_context.localhost_only() else "REMOTE (bearer auth)", self.agent_name, len(self._agents))
        self._wire_plugin_handlers(None)  # plugin-registered native handlers
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        self._watchdog_stop.set()
        if self._httpd is not None:
            with contextlib.suppress(Exception):
                self._httpd.shutdown()
                self._httpd.server_close()
            self._httpd = None
        # Fail any in-flight replies so blocked HTTP threads don't hang.
        with self._pending_lock:
            for tid in list(self._pending):
                self._resolve_locked(tid, protocol.STATE_FAILED, "[agent shutting down]")
            self._pending.clear()
            self._pending_order.clear()

    def _watchdog_loop(self) -> None:
        """Background thread that fails orphaned tasks (keeps them queryable)."""
        while not self._watchdog_stop.wait(_WATCHDOG_INTERVAL):
            try:
                for tid in self.tasks.fail_orphans(_ORPHAN_TIMEOUT):
                    logger.warning("A2A: orphaned task %s marked failed (timeout %ds)", tid, _ORPHAN_TIMEOUT)
                    protocol.metrics.tasks_failed += 1
            except Exception:
                logger.debug("A2A: watchdog error", exc_info=True)

    def _load_served_agents(self, extra: dict) -> dict[str, dict]:
        """Served-agent routing from ``platforms.a2a.extra.agents`` (top-level ``a2a_served_agents``
        fallback for scripts/tests). Root/default always maps to the live gateway session."""
        raw = extra.get("agents") or extra.get("served_agents")
        if raw is None:
            try:
                from hermes_cli.config import load_config
                cfg = load_config() or {}
            except Exception:
                cfg = {}
            cfg = cfg if isinstance(cfg, dict) else {}
            raw = cfg.get("a2a_served_agents") or (cfg.get("a2a") or {}).get("served_agents")
        # Scope-aware like port: a secondary profile must not inherit A2A_AGENT_DESCRIPTION.
        default_desc = _DEFAULT_DESCRIPTION if _profile_scoped() else os.getenv("A2A_AGENT_DESCRIPTION", _DEFAULT_DESCRIPTION)
        agents: dict[str, dict] = {"": {
            "slug": "", "path": "", "tenant": "", "profile": self._active_profile, "local": True,
            "name": self.agent_name, "description": default_desc, "advertised_toolsets": self._advertised_toolsets,
        }}
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
            toolsets = val.get("advertised_toolsets") or val.get("toolsets") or val.get("capabilities") or []
            if isinstance(toolsets, str):
                toolsets = [t.strip() for t in toolsets.split(",") if t.strip()]
            tenant = str(val.get("tenant") or slug).strip()
            if tenant:
                if tenant in tenants:
                    logger.warning("A2A: ignoring served agent %r with duplicate tenant %r already used by %r",
                                   slug, tenant, tenants[tenant])
                    continue
                tenants[tenant] = slug
            agents[slug] = {
                "slug": slug, "path": "/" + path_segment, "tenant": tenant, "profile": profile or slug,
                "local": bool(val.get("local")) or profile in ("", "default", self._active_profile),
                "name": str(val.get("name") or f"Hermes {slug}"),
                "description": str(val.get("description") or f"Hermes profile '{profile or slug}' exposed over A2A."),
                "advertised_toolsets": list(toolsets or []),
                "timeout": int(val.get("timeout") or _reply_timeout()),
            }
        return agents

    def _base_url(self, public_url: Optional[str]) -> str:
        return (public_url or "").strip() or f"http://{self.host}:{self.port}/"

    def _served_agent_summary(self, public_url: Optional[str] = None) -> list[dict]:
        base = self._base_url(public_url)
        return [{"slug": a["slug"] or "default", "name": a.get("name"), "url": _join_url(base, a.get("path", "")),
                 "tenant": a.get("tenant") or None, "profile": a.get("profile"), "local": bool(a.get("local"))}
                for a in self._agents.values()]

    def _route_for_path(self, raw_path: str) -> dict:
        path = urllib.parse.urlsplit(raw_path or "/").path or "/"
        # Longest prefix wins. Default/root agent is the fallback.
        for agent in sorted(self._agents.values(), key=lambda a: len(a.get("path", "")), reverse=True):
            if (prefix := agent.get("path", "") or "") and (path == prefix or path.startswith(prefix + "/")):
                return {"agent": agent, "subpath": path[len(prefix):] or "/"}
        return {"agent": self._agents[""], "subpath": path}

    def _route_for_request(self, raw_path: str, params: dict) -> dict:
        route = self._route_for_path(raw_path)
        agent = route["agent"]
        tenant = str((params or {}).get("tenant") or "")
        # If no URL prefix chose a non-default agent, allow v1.0 tenant routing.
        if agent.get("slug") == "" and tenant and (matches := [a for a in self._agents.values() if a.get("tenant") == tenant]):
            agent = matches[0]
            route = {"agent": agent, "subpath": route["subpath"]}
        expected = str(agent.get("tenant") or "")
        if tenant and expected and tenant != expected:
            return {"error": f"tenant {tenant!r} does not match routed agent {agent.get('slug') or 'default'}"}
        return route

    def _build_card(self, public_url: Optional[str] = None, agent: Optional[dict] = None) -> dict:
        # Per-request public URL beats the bind host so peers behind a reverse proxy can call back.
        agent = agent or self._agents[""]
        return protocol.build_agent_card(
            name=agent.get("name") or self.agent_name, url=_join_url(self._base_url(public_url), agent.get("path", "")),
            description=agent.get("description") or _DEFAULT_DESCRIPTION, skills=self._advertised_skills(agent),
            streaming=bool(agent.get("local", True)), push_notifications=True,
            auth_required=not self._security_context.localhost_only(), tenant=str(agent.get("tenant") or ""),
        )

    def _advertised_skills(self, agent: Optional[dict] = None) -> list[dict]:
        """Agent Card skills from the live tool registry, restricted by ``advertised_toolsets``;
        static fallback without a registry."""
        configured = (agent or {}).get("advertised_toolsets") if agent else self._advertised_toolsets
        try:
            from tools.registry import registry as tool_registry
            allowed = set(configured or []) or None
            mapping = {n: tool_registry.get_tool_names_for_toolset(n)
                       for n in tool_registry.get_registered_toolset_names() if allowed is None or n in allowed}
            if mapping:
                return protocol.skills_from_toolsets(mapping)
        except Exception:
            logger.debug("A2A: tool registry unavailable for Agent Card", exc_info=True)
        return protocol.skills_from_toolsets(configured or [])

    def _add_pending(self, task_id: str, context_id: str) -> Future:
        fut: Future = Future()
        with self._pending_lock:
            self._pending[task_id] = (context_id, fut)
            self._pending_order.setdefault(context_id, deque()).append(task_id)
        return fut

    def _pop_pending(self, task_id: str) -> None:
        with self._pending_lock:
            entry = self._pending.pop(task_id, None)
            order = self._pending_order.get(entry[0]) if entry else None
            if order and task_id in order:
                order.remove(task_id)
            if order is not None and not order:
                self._pending_order.pop(entry[0], None)

    def _resolve_locked(self, task_id: str, state: str, text: str) -> bool:
        entry = self._pending.get(task_id)
        if not entry or entry[1].done():
            return False
        entry[1].set_result((state, text))
        return True

    def _resolve_task(self, task_id: str, state: str, text: str) -> bool:
        with self._pending_lock:
            return self._resolve_locked(task_id, state, text)

    def _resolve_oldest_for_context(self, context_id: str, state: str, text: str) -> bool:
        with self._pending_lock:
            return any(self._resolve_locked(tid, state, text) for tid in self._pending_order.get(context_id, ()))

    def _scope_for_agent(self, agent: Optional[dict]) -> tuple[str, str]:
        return tuple(str((agent or self._agents[""]).get(k) or "") for k in ("slug", "tenant"))

    def _forward_lock(self, key: tuple[str, str, str]) -> threading.Lock:
        with self._profile_session_locks_guard:
            return self._profile_session_locks.setdefault(key, threading.Lock())

    def _end_task(self, rec: dict, state: str, text: str, stored_reply: str = "") -> tuple[dict, None]:
        """Complete a task immediately (rejected / not ready) and build its terminal Task."""
        self.tasks.complete(rec["task_id"], state, stored_reply)
        protocol.metrics.tasks_failed += state == protocol.STATE_FAILED
        return protocol.build_task(rec["task_id"], rec["context_id"], state, text, created_at=rec["created_iso"]), None

    def _prepare_task(self, params: dict, peer: str, agent: Optional[dict] = None) -> tuple[Optional[dict], Optional[dict]]:
        """Validate, register, and dispatch an inbound message (HTTP worker thread). Returns
        (terminal_task, None) when it ends immediately, else (None, pending) with the future to wait on."""
        agent = agent or self._agents[""]
        text = protocol.extract_text(params)
        context_id = protocol.extract_context_id(params) or protocol.new_context_id()
        task_id = protocol.new_task_id()
        turn = self._turns.track(context_id)
        max_turns = protocol.max_pingpong_turns()
        rec = self.tasks.create(task_id, context_id, peer, *self._scope_for_agent(agent))
        if turn > max_turns:
            protocol.metrics.anti_loop_triggers += 1
            logger.warning("A2A: anti-loop triggered for context %s (turn %d > %d)", context_id, turn, max_turns)
            return self._end_task(rec, protocol.STATE_REJECTED, f"Anti-loop protection: context {context_id} exceeded "
                                  f"{max_turns} turns. Start a new context or increase A2A_MAX_PINGPONG_TURNS.")
        if not text:
            return self._end_task(rec, protocol.STATE_REJECTED, "Empty task — nothing to do.")
        framed = security.wrap_inbound(peer, text)
        security.audit("inbound", peer, task_id, text)
        protocol.persist_message(context_id, "user", text, task_id)
        protocol.metrics.inbound_total += 1
        self._register_inline_push(task_id, params, agent=agent)
        if not agent.get("local", True):
            reply, state = self._forward_to_profile(agent, peer, context_id, framed)
            self._record_outcome(task_id, context_id, peer, state, reply)
            return protocol.build_task(task_id, context_id, state, reply, created_at=rec["created_iso"]), None
        if self._loop is None or self._message_handler is None:
            return self._end_task(rec, protocol.STATE_FAILED, "Agent gateway not ready to accept A2A tasks.")
        fut = self._add_pending(task_id, context_id)
        event = MessageEvent(text=framed, message_type=MessageType.TEXT, message_id=task_id,
                             source=self.build_source(chat_id=context_id, chat_name=f"a2a:{peer}", chat_type="dm", user_id=peer, user_name=peer))
        try:
            asyncio.run_coroutine_threadsafe(self.handle_message(event), self._loop)
        except Exception as e:
            self._pop_pending(task_id)
            msg = security.redact_outbound(f"Dispatch failed: {e}")
            return self._end_task(rec, protocol.STATE_FAILED, msg, stored_reply=msg)
        self.tasks.set_state(task_id, protocol.STATE_WORKING)
        return None, {"task_id": task_id, "context_id": context_id, "peer": peer, "future": fut, "created_iso": rec["created_iso"], "started": time.time()}

    def _forward_to_profile(self, agent: dict, peer: str, context_id: str, framed_text: str) -> tuple[str, str]:
        """Forward a routed task to another local profile via ``hermes chat``. First contact creates a
        ``source=a2a`` session and titles it deterministically; later turns ``--resume`` that id."""
        profile = str(agent.get("profile") or agent.get("slug") or "").strip()
        slug = str(agent.get("slug") or profile or "agent")
        safe_ctx = _safe_context_slug(context_id)
        session_title = f"a2a-{slug}-{safe_ctx}"
        key = (profile or "default", slug, safe_ctx)
        timeout = int(agent.get("timeout") or _reply_timeout())
        with self._forward_lock(key):
            session_id = self._profile_sessions.get(key) or _state_db(
                profile, "SELECT id FROM sessions WHERE title = ? ORDER BY started_at DESC LIMIT 1",
                (session_title,), "A2A: could not lookup forwarded session")
            cmd = ["hermes", "chat", "-q", framed_text, "-Q", "--source", "a2a"] + (["--resume", session_id] if session_id else [])
            env = {**os.environ, "HERMES_A2A_PEER": peer}
            if home := _profile_home(profile):
                env["HERMES_HOME"] = home
            start = time.time()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                      timeout=timeout, env=env, check=False, stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                return "[profile did not reply in time]", protocol.STATE_FAILED
            except Exception as e:
                return security.redact_outbound(f"Profile dispatch failed: {e}"), protocol.STATE_FAILED
            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or f"profile exited {proc.returncode}").strip()
                return security.redact_outbound(msg[-2000:]), protocol.STATE_FAILED
            if not session_id and (session_id := _state_db(
                    profile, "SELECT id FROM sessions WHERE source = 'a2a' AND started_at >= ? ORDER BY started_at DESC LIMIT 1",
                    (start - 2.0,), "A2A: could not find latest forwarded session")):
                self._profile_sessions[key] = session_id
                _state_db(profile, "UPDATE sessions SET title = ? WHERE id = ?", (session_title, session_id),
                          "A2A: could not title forwarded session", commit=True)
            return security.redact_outbound((proc.stdout or "").strip()), protocol.STATE_COMPLETED

    def _record_outcome(self, task_id: str, context_id: str, peer: str, state: str, reply: str,
                        started: Optional[float] = None) -> None:
        """Persist + audit + count a finished task, mark it terminal, and fire its push callback."""
        protocol.persist_message(context_id, "agent", reply, task_id)
        security.audit("outbound", peer, task_id, reply)
        m = protocol.metrics
        if state in (protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED):
            m.outbound_total, m.tasks_completed = m.outbound_total + 1, m.tasks_completed + 1
            if started is not None:
                m.record_latency(time.time() - started)
        else:
            m.tasks_failed += 1
        self.tasks.complete(task_id, state, reply)
        self._send_push_notification(task_id, context_id, reply, state)

    def _finalize_task(self, pending: dict, state: str, reply: str) -> tuple[str, str]:
        """Record a dispatched task's outcome; returns (state, reply) after redaction and
        input-required detection (a leading marker flags a clarification request)."""
        task_id, context_id, peer = pending["task_id"], pending["context_id"], pending["peer"]
        self._pop_pending(task_id)
        reply = security.redact_outbound(reply or "")
        stripped = reply.lstrip()
        if state == protocol.STATE_COMPLETED and stripped.upper().startswith(protocol.INPUT_REQUIRED_MARKER):
            state, reply = protocol.STATE_INPUT_REQUIRED, stripped[len(protocol.INPUT_REQUIRED_MARKER):].strip()
        self._record_outcome(task_id, context_id, peer, state, reply, started=pending["started"])
        return state, reply

    @staticmethod
    def _await_future(fut: Future, deadline: float, keepalive, on_timeout: tuple[str, str]) -> tuple[str, str]:
        """Block until ``fut`` resolves or ``deadline`` passes (-> ``on_timeout``). ``keepalive`` runs
        every _SSE_KEEPALIVE seconds while waiting; if it raises, the client is gone and we stop."""
        while True:
            try:
                return fut.result(timeout=_SSE_KEEPALIVE if keepalive else max(0.0, deadline - time.time()))
            except FuturesTimeout:
                if time.time() >= deadline:
                    return on_timeout
                if keepalive:
                    try:
                        keepalive()
                    except Exception:
                        return (protocol.STATE_FAILED, "[client disconnected]")
            except Exception:
                return on_timeout

    def _await_reply(self, pending: dict, keepalive=None) -> tuple[str, str]:
        return self._await_future(pending["future"], pending["started"] + _reply_timeout(), keepalive,
                                  (protocol.STATE_FAILED, "[agent did not reply in time]"))

    def _rpc_message_send(self, req_id: Any, params: dict, peer: str, agent: Optional[dict] = None, v1_response: bool = False) -> dict:
        task, pending = self._prepare_task(params, peer, agent=agent)
        if task is None:
            state, reply = self._finalize_task(pending, *self._await_reply(pending))
            task = protocol.build_task(pending["task_id"], pending["context_id"], state, reply, created_at=pending["created_iso"])
        return _ok(req_id, protocol.send_message_response(task) if v1_response else task)

    @staticmethod
    def _sse_headers(handler) -> None:
        handler.send_response(200)
        for k, v in (("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache")):
            handler.send_header(k, v)
        handler.end_headers()
        handler.close_connection = True  # v1.0: stream closure signals the terminal state

    @staticmethod
    def _sse_write(handler, chunk: str) -> None:
        handler.wfile.write(chunk.encode("utf-8"))
        handler.wfile.flush()

    @classmethod
    def _keepalive(cls, handler):
        return lambda: cls._sse_write(handler, ": keepalive\n\n")

    def _emit_terminal(self, handler, task_id: str, context_id: str, state: str, reply: str, req_id: Any = None) -> None:
        """Emit the final artifact/status events and the closure marker. ``req_id`` threads into the
        JSON-RPC SSE envelope (§9.4)."""
        completed = bool(reply) and state == protocol.STATE_COMPLETED
        events = ([protocol.artifact_update(task_id, context_id, reply)] if completed else []) + [
            protocol.status_update(task_id, context_id, state, "" if completed else reply)]
        for ev in events:
            self._sse_write(handler, protocol.sse_data(ev, req_id))
        self._sse_write(handler, protocol.sse_done())

    def _rpc_message_stream(self, handler, req_id: Any, params: dict, peer: str, agent: Optional[dict] = None) -> None:
        """message/stream as an SSE response of JSON-RPC-wrapped StreamResponse events (§9.4)."""
        protocol.metrics.streams_started += 1
        self._sse_headers(handler)
        try:
            terminal, pending = self._prepare_task(params, peer, agent=agent)
            if terminal is not None:
                return self._emit_terminal(handler, terminal["id"], terminal["contextId"], terminal["status"]["state"],
                                           protocol.extract_text(terminal.get("status", {}).get("message", {}) or {}), req_id=req_id)
            task_id, context_id = pending["task_id"], pending["context_id"]
            submitted = protocol.build_task(task_id, context_id, protocol.STATE_SUBMITTED, created_at=pending["created_iso"])
            self._sse_write(handler, protocol.sse_data(protocol.stream_task(submitted), req_id))
            self._sse_write(handler, protocol.sse_data(protocol.status_update(task_id, context_id, protocol.STATE_WORKING), req_id))
            state, reply = self._finalize_task(pending, *self._await_reply(pending, keepalive=self._keepalive(handler)))
            self._emit_terminal(handler, task_id, context_id, state, reply, req_id=req_id)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("A2A: stream client disconnected")

    def _rpc_tasks_subscribe(self, handler, req_id: Any, params: dict, agent: Optional[dict] = None) -> None:
        """Reconnect to an existing task's stream (v1.0 SubscribeToTask)."""
        task_id, rec, error = self._find_task(req_id, params, agent)
        if error:
            return handler._json(200, error)
        self._sse_headers(handler)
        try:
            if (fut := self.tasks.watch(task_id, *self._scope_for_agent(agent))) is None:
                return self._sse_write(handler, protocol.sse_done())
            state, reply = self._await_future(fut, time.time() + _reply_timeout(), self._keepalive(handler),
                                              (rec["state"], rec.get("reply", "")))
            self._emit_terminal(handler, task_id, rec["context_id"], state, reply, req_id=req_id)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("A2A: subscribe client disconnected")

    def _find_task(self, req_id: Any, params: dict, agent: Optional[dict]) -> tuple[str, Optional[dict], Optional[dict]]:
        """(task_id, record, None) for a visible task, else (task_id, None, jsonrpc_error)."""
        task_id = str(params.get("taskId") or params.get("id") or "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent))
        return task_id, rec, None if rec else _err(req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")

    def _rpc_tasks_get(self, req_id: Any, params: dict, agent: Optional[dict] = None) -> dict:
        _task_id, rec, error = self._find_task(req_id, params, agent)
        return error or _ok(req_id, protocol.TaskStore.to_task(rec))

    def _rpc_tasks_list(self, req_id: Any, params: dict, agent: Optional[dict] = None) -> dict:
        offset = _to_int(params.get("pageToken") or 0, 0)
        page_size = _to_int(params.get("pageSize") or 50, 50)
        agent_slug, tenant = self._scope_for_agent(agent)
        recs, next_offset, total = self.tasks.list(
            context_id=str(params.get("contextId") or ""), state=str(params.get("status") or params.get("state") or ""),
            page_size=page_size, offset=max(0, offset), agent_slug=agent_slug, tenant=tenant, with_total=True)
        include_artifacts = bool(params.get("includeArtifacts", False))
        return _ok(req_id, {"tasks": [protocol.TaskStore.to_task(r, include_artifacts=include_artifacts) for r in recs],
                            "nextPageToken": str(next_offset) if next_offset else "",
                            "pageSize": max(1, min(page_size, 100)), "totalSize": total})

    def _rpc_tasks_cancel(self, req_id: Any, params: dict, agent: Optional[dict] = None) -> dict:
        task_id, rec, error = self._find_task(req_id, params, agent)
        if error:
            return error
        if rec["state"] in protocol.TERMINAL_STATES:
            return _err(req_id, protocol.ERR_TASK_NOT_CANCELABLE, f"task {task_id} already {rec['state']}")
        self.tasks.complete(task_id, protocol.STATE_CANCELED, "")
        self._turns.reset(rec["context_id"])
        self._resolve_task(task_id, protocol.STATE_CANCELED, "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent)) or rec
        return _ok(req_id, protocol.TaskStore.to_task(rec))

    def _register_inline_push(self, task_id: str, params: dict, agent: Optional[dict] = None) -> None:
        """v1.0: message/send can carry configuration.taskPushNotificationConfig."""
        cfg = (params.get("configuration") or {}).get("taskPushNotificationConfig") or {}
        url = (cfg.get("url") or (cfg.get("pushNotificationConfig") or {}).get("url") or "") if isinstance(cfg, dict) else ""
        if url:
            self.tasks.set_push_config(task_id, str(url), *self._scope_for_agent(agent))

    def _rpc_push_config_create(self, req_id: Any, params: dict, agent: Optional[dict] = None) -> dict:
        task_id = str(params.get("taskId") or "")
        url = str((params.get("pushNotificationConfig") or params.get("config") or {}).get("url") or "")
        if not task_id or not url:
            return _err(req_id, protocol.ERR_INVALID_PARAMS, "taskId and pushNotificationConfig.url required")
        stored = self.tasks.set_push_config(task_id, url, *self._scope_for_agent(agent))
        return _ok(req_id, stored) if stored is not None else _err(req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")

    def _push_config_op(self, req_id: Any, params: dict, agent: Optional[dict], op, render) -> dict:
        """Shared get/list/delete: ``op(task_id, config_id, slug, tenant)`` falsy => not found."""
        task_id = str(params.get("taskId") or "")
        if not task_id:
            return _err(req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        found = op(task_id, str(params.get("id") or params.get("configId") or ""), *self._scope_for_agent(agent))
        return _ok(req_id, render(found)) if found else _err(req_id, protocol.ERR_TASK_NOT_FOUND, f"push config not found for task: {task_id}")

    def _rpc_push_config_get(self, req_id: Any, params: dict, agent: Optional[dict] = None) -> dict:
        return self._push_config_op(req_id, params, agent, self.tasks.get_push_config, lambda cfg: cfg)

    def _rpc_push_config_list(self, req_id: Any, params: dict, agent: Optional[dict] = None) -> dict:
        # An empty list is a valid (non-error) result, hence the ``or [[]]`` sentinel through the shared op.
        return self._push_config_op(req_id, params, agent, lambda tid, _cid, *scope: self.tasks.list_push_configs(tid, *scope) or [[]],
                                    lambda found: {"configs": [c for c in found if c], "nextPageToken": ""})

    def _rpc_push_config_delete(self, req_id: Any, params: dict, agent: Optional[dict] = None) -> dict:
        return self._push_config_op(req_id, params, agent, self.tasks.delete_push_config, lambda _: {"deleted": True})

    def _send_push_notification(self, task_id: str, context_id: str, reply: str, state: str) -> None:
        """POST a v1.0 StreamResponse payload to the task's registered callback (SSRF-checked URL)."""
        def fail(msg: str, *args) -> None:
            protocol.metrics.push_failed += 1
            logger.warning("A2A: push notification for task %s " + msg, task_id, *args)

        callback_url = self.tasks.pop_push_url(task_id)
        if not callback_url:
            return
        if not security.is_safe_callback_url(callback_url, localhost_mode=self._security_context.localhost_only()):
            return fail("blocked — unsafe callback URL: %s", callback_url)
        payload = protocol.status_update(task_id, context_id, state, (reply or "")[:2000])
        headers = {"Content-Type": "application/json"}
        if signature := self._security_context.sign_push_payload(payload):
            headers["X-A2A-Signature"] = signature
        try:
            req = urllib.request.Request(callback_url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                status = resp.status
        except Exception as e:
            return fail("failed: %s", e)
        if not 200 <= status < 300:
            return fail("got HTTP %d", status)
        protocol.metrics.push_sent += 1
        logger.debug("A2A: push notification sent for task %s", task_id)

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Fulfil the oldest pending reply Future for this context (``chat_id`` = A2A context id).
        Only sends carrying ``metadata['notify']`` (the base adapter's final-reply marker) satisfy
        the caller; progress/status/preview sends must not."""
        if not (metadata or {}).get("notify"):
            logger.debug("A2A: ignoring non-final send for context %s", chat_id)
        elif not self._resolve_oldest_for_context(chat_id, protocol.STATE_COMPLETED, content or ""):
            logger.debug("A2A: send() for context %s had no pending waiter", chat_id)  # late chunk / out-of-band
        return SendResult(success=True, message_id=str(int(time.time() * 1000)))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": f"a2a:{chat_id}", "type": "dm"}

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Resolve the task future when processing ends without a reply send (failures,
        cancellations, empty runs) so the HTTP thread returns promptly."""
        task_id = str(getattr(event, "message_id", "") or "")
        if task_id:
            self._resolve_task(task_id, *{
                ProcessingOutcome.FAILURE: (protocol.STATE_FAILED, "[agent processing failed]"),
                ProcessingOutcome.CANCELLED: (protocol.STATE_CANCELED, ""),
            }.get(outcome, (protocol.STATE_COMPLETED, "")))
