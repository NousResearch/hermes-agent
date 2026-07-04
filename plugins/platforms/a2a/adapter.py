"""
A2A inbound platform adapter — exposes Hermes as an A2A-discoverable agent.

Design (the #11025 insight, done as a plugin with zero core edits):
  - Runs a stdlib http.server in a daemon thread (no a2a-sdk, no asyncio loop
    dependency at register() time — avoids the a2a_fleet "register outside a
    loop" bug class).
  - Serves the Agent Card at GET /.well-known/agent.json.
  - Accepts JSON-RPC ``message/send`` at POST /.
  - Streams via JSON-RPC ``message/stream`` at POST / → SSE response.
  - Push notifications via ``tasks/pushNotification/set`` + webhook callbacks.
  - Metrics at GET /metrics.
  - Each inbound task is filtered + framed (security.wrap_inbound) and routed
    into the agent's LIVE gateway session via the normal MessageEvent path, so
    the agent that replies is the same one talking to its user — full memory
    and context, not a throwaway clone.
  - The agent's reply comes back through ``adapter.send()``; we override that to
    fulfil a per-context Future the HTTP handler is blocked on, turning the
    async gateway into a synchronous request/response for the A2A caller.
  - Every exchange is persisted to disk and audit-logged.

Bind safety: with no A2A_BEARER_TOKEN, the server binds 127.0.0.1 only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import urllib.request
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.config import Platform

from . import protocol, security

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 9900
_REPLY_TIMEOUT = 300  # seconds to wait for the agent to answer an inbound task
_ORPHAN_TIMEOUT = 300  # seconds before a pending task is considered orphaned
_WATCHDOG_INTERVAL = 60  # seconds between orphaned task watchdog runs


def _default_agent_name() -> str:
    name = os.getenv("A2A_AGENT_NAME", "").strip()
    if name:
        return name
    try:
        import socket
        return f"hermes-{socket.gethostname()}"
    except Exception:
        return "hermes-agent"


class A2AAdapter(BasePlatformAdapter):
    """Inbound A2A server adapter."""

    def __init__(self, config, **kwargs):
        platform = Platform("a2a")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}
        self.port = int(os.getenv("A2A_PORT") or extra.get("port", _DEFAULT_PORT))
        self.host = security.resolve_bind_host()
        self.agent_name = _default_agent_name()

        self._httpd: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Per-context reply futures: an inbound HTTP request blocks on its
        # future until adapter.send() resolves it with the agent's reply.
        self._pending_replies: Dict[str, Future] = {}
        # Per-context streaming queues: for message/stream, the handler writes
        # SSE chunks and the send() method pushes intermediate results.
        self._streaming_queues: Dict[str, list] = {}
        self._pending_lock = threading.Lock()

        # Push notification callback URLs per task
        self._push_callbacks: Dict[str, str] = {}
        self._push_lock = threading.Lock()

        # Orphaned task watchdog
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()

    @property
    def name(self) -> str:
        return "A2A"

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

        adapter = self

        class _Handler(BaseHTTPRequestHandler):
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

            def _request_public_url(self) -> str:
                """Derive the routable URL for this request.

                Priority: A2A_PUBLIC_URL env > X-Forwarded-Host / Host
                header (with scheme from X-Forwarded-Proto) > empty.
                Empty means "caller has no info, fall back to bind host".
                See gfdsa's k8s bind-host bug report (PR #41711).
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
                if self.path.rstrip("/") in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
                    public_url = self._request_public_url() or None
                    self._json(200, adapter._build_card(public_url))
                    return
                if self.path.rstrip("/") in ("", "/health"):
                    self._json(200, {"status": "ok", "agent": adapter.agent_name})
                    return
                if self.path.rstrip("/") == "/metrics":
                    self._json(200, protocol.metrics.snapshot())
                    return
                self._json(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                # Auth (only meaningful when a token is configured; otherwise
                # we are localhost-only by construction).
                if not security.check_bearer(self.headers.get("Authorization")):
                    self._json(401, protocol.jsonrpc_error(None, -32001, "unauthorized"))
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length) if length else b"{}"
                    req = json.loads(raw.decode("utf-8"))
                except Exception:
                    self._json(400, protocol.jsonrpc_error(None, -32700, "parse error"))
                    return

                req_id = req.get("id")
                method = req.get("method", "")
                params = req.get("params", {}) or {}

                # Rate limit check
                peer_id = str(params.get("peer") or (params.get("message", {}) or {}).get("from") or "unknown")
                if not protocol.rate_limit_allow(peer_id):
                    protocol.metrics.rate_limit_triggers += 1
                    self._json(429, protocol.jsonrpc_error(req_id, -32002, "rate limit exceeded"))
                    return

                # Trusted peer check
                if not security.is_trusted_peer(peer_id):
                    self._json(403, protocol.jsonrpc_error(req_id, -32003, f"peer '{peer_id}' not trusted"))
                    return

                if method == "message/send":
                    result = adapter._handle_inbound_task(params, stream=False)
                    self._json(200, protocol.jsonrpc_result(req_id, result))
                    return
                if method == "message/stream":
                    adapter._handle_streaming(self, req_id, params)
                    return
                if method == "tasks/get":
                    task_id = params.get("taskId") or params.get("id", "")
                    info = protocol.pending_task_info(task_id)
                    if info:
                        self._json(200, protocol.jsonrpc_result(req_id, {
                            "id": task_id,
                            "contextId": info.get("context_id", ""),
                            "status": {"state": info.get("state", "working")},
                            "kind": "task",
                        }))
                    else:
                        self._json(200, protocol.jsonrpc_result(req_id, {"error": "task not found"}))
                    return
                if method == "tasks/cancel":
                    task_id = params.get("taskId") or params.get("id", "")
                    protocol.reset_turns(task_id)  # reset anti-loop on cancel
                    info = protocol.complete_pending_task(task_id, protocol.STATE_CANCELED)
                    self._json(200, protocol.jsonrpc_result(req_id, {
                        "id": task_id,
                        "status": {"state": protocol.STATE_CANCELED},
                        "kind": "task",
                    }))
                    return
                if method == "tasks/pushNotification/set":
                    task_id = params.get("taskId") or ""
                    callback_url = (params.get("pushNotificationConfig") or {}).get("url", "")
                    if task_id and callback_url:
                        with adapter._push_lock:
                            adapter._push_callbacks[task_id] = callback_url
                        self._json(200, protocol.jsonrpc_result(req_id, {"taskId": task_id, "registered": True}))
                    else:
                        self._json(200, protocol.jsonrpc_error(req_id, -32602, "taskId and pushNotificationConfig.url required"))
                    return

                self._json(200, protocol.jsonrpc_error(req_id, -32601, f"method not found: {method}"))

        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
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

        # Start orphaned task watchdog
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="a2a-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

        self._mark_connected()

        exposure = "localhost-only" if security.localhost_only() else "REMOTE (bearer auth)"
        logger.info(
            "A2A: serving Agent Card + JSON-RPC on http://%s:%s (%s) as %r",
            self.host, self.port, exposure, self.agent_name,
        )
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        self._watchdog_stop.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        # Fail any in-flight replies so blocked HTTP threads don't hang.
        with self._pending_lock:
            for fut in self._pending_replies.values():
                if not fut.done():
                    fut.set_result("[agent shutting down]")
            self._pending_replies.clear()
            self._streaming_queues.clear()

    # ── Orphaned task watchdog ─────────────────────────────────────────────

    def _watchdog_loop(self) -> None:
        """Background thread that cleans up orphaned tasks."""
        while not self._watchdog_stop.wait(_WATCHDOG_INTERVAL):
            try:
                cleared = protocol.clear_orphaned_tasks(_ORPHAN_TIMEOUT)
                for tid in cleared:
                    logger.warning("A2A: orphaned task %s cleaned up (timeout %ds)", tid, _ORPHAN_TIMEOUT)
                    protocol.metrics.tasks_failed += 1
            except Exception:
                logger.debug("A2A: watchdog error", exc_info=True)

    # ── Agent Card ────────────────────────────────────────────────────────

    def _build_card(self, public_url: Optional[str] = None) -> dict:
        # Dynamic Agent Cards: try to build from real toolset registry
        skills = []
        try:
            toolsets = []
            extra = getattr(self.config, "extra", {}) or {}
            toolsets = list(extra.get("advertised_toolsets") or [])

            # If we have a real toolset registry available, build dynamic skills
            registry = extra.get("_toolset_registry")
            if registry and isinstance(registry, dict):
                skills = protocol.skills_from_real_toolsets(registry)
            else:
                skills = protocol.skills_from_toolsets(toolsets)
        except Exception:
            skills = protocol.skills_from_toolsets([])

        # v4 fix: prefer per-request public URL (from X-Forwarded-Host
        # / Host / A2A_PUBLIC_URL) over bind host, so peers can call back
        # when we're behind a reverse proxy. See gfdsa's PR #41711 review.
        url = (public_url or "").strip() or f"http://{self.host}:{self.port}/"
        return protocol.build_agent_card(
            name=self.agent_name,
            url=url,
            description=os.getenv(
                "A2A_AGENT_DESCRIPTION",
                "Hermes Agent — a general-purpose agent reachable over A2A.",
            ),
            skills=skills,
            streaming=True,  # Phase 2: SSE streaming now supported
            push_notifications=True,  # Phase 2: push notifications now supported
            auth_required=not security.localhost_only(),
        )

    # ── Inbound task handling ─────────────────────────────────────────────

    def _handle_inbound_task(self, params: dict, stream: bool = False) -> dict:
        """Route an inbound A2A task into the live session and wait for reply.

        Runs on an HTTP worker thread. It marshals a MessageEvent onto the
        gateway loop and blocks (on a Future) until adapter.send() fulfils it.
        """
        text = protocol.extract_text(params)
        peer = str(params.get("peer") or (params.get("message", {}) or {}).get("from") or "remote-agent")
        # A2A spec: contextId lives at top level of params. The original code
        # only looked inside params.message (non-standard placement) so
        # every turn got a fresh contextId, breaking multi-turn memory.
        # Falls back to params.message.contextId for legacy callers.
        context_id = (
            params.get("contextId")                                   # A2A spec: top-level
            or (params.get("message", {}) or {}).get("contextId")    # legacy/non-standard
            or protocol.new_context_id()
        )
        task_id = protocol.new_task_id()

        # Anti-loop ping-pong protection
        turn = protocol.track_turn(context_id)
        if turn > protocol.max_pingpong_turns():
            protocol.metrics.anti_loop_triggers += 1
            logger.warning("A2A: anti-loop triggered for context %s (turn %d > %d)",
                          context_id, turn, protocol.max_pingpong_turns())
            return protocol.build_task(task_id, context_id, protocol.STATE_FAILED,
                f"Anti-loop protection: context {context_id} exceeded {protocol.max_pingpong_turns()} turns. "
                f"Start a new context or increase A2A_MAX_PINGPONG_TURNS.")

        if not text:
            return protocol.build_task(task_id, context_id, protocol.STATE_FAILED, "Empty task — nothing to do.")

        framed = security.wrap_inbound(peer, text)
        security.audit("inbound", peer, task_id, text)
        protocol.persist_message(context_id, "user", text, task_id)
        protocol.metrics.inbound_total += 1

        # Register as pending task for async tracking
        push_url = ""
        with self._push_lock:
            push_url = self._push_callbacks.get(task_id, "")
        protocol.register_pending_task(task_id, context_id, peer, push_url)

        if self._loop is None or self._message_handler is None:
            protocol.complete_pending_task(task_id, protocol.STATE_FAILED)
            return protocol.build_task(
                task_id, context_id, protocol.STATE_FAILED,
                "Agent gateway not ready to accept A2A tasks.",
            )

        fut: Future = Future()
        with self._pending_lock:
            self._pending_replies[context_id] = fut

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
            with self._pending_lock:
                self._pending_replies.pop(context_id, None)
            protocol.complete_pending_task(task_id, protocol.STATE_FAILED, f"Dispatch failed: {e}")
            return protocol.build_task(task_id, context_id, protocol.STATE_FAILED, f"Dispatch failed: {e}")

        try:
            reply = fut.result(timeout=_REPLY_TIMEOUT)
        except Exception:
            reply = "[agent did not reply in time]"
            protocol.metrics.tasks_failed += 1
        finally:
            with self._pending_lock:
                self._pending_replies.pop(context_id, None)

        reply = security.redact_outbound(reply or "")
        protocol.persist_message(context_id, "agent", reply, task_id)
        security.audit("outbound", peer, task_id, reply)
        protocol.metrics.outbound_total += 1
        protocol.metrics.tasks_completed += 1
        protocol.metrics.record_latency(0)  # Updated by send() for more accuracy

        # Complete pending task
        task_info = protocol.complete_pending_task(task_id, protocol.STATE_COMPLETED, reply)

        # Push notification if registered
        self._send_push_notification(task_id, context_id, reply, protocol.STATE_COMPLETED)

        return protocol.build_task(task_id, context_id, protocol.STATE_COMPLETED, reply)

    # ── Streaming handler ─────────────────────────────────────────────────

    def _handle_streaming(self, handler, req_id: Any, params: dict) -> None:
        """Handle message/stream as SSE response.

        Sends task state transitions as SSE events:
        1. submitted event
        2. working event
        3. (intermediate sends become artifact events — not yet wired to send())
        4. completed/failed event with final reply
        5. done event
        """
        protocol.metrics.streams_started += 1

        # Send SSE headers
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.end_headers()

        text = protocol.extract_text(params)
        peer = str(params.get("peer") or (params.get("message", {}) or {}).get("from") or "remote-agent")
        context_id = (
            params.get("contextId")
            or (params.get("message", {}) or {}).get("contextId")
            or protocol.new_context_id()
        )
        task_id = protocol.new_task_id()

        # Anti-loop check
        turn = protocol.track_turn(context_id)
        if turn > protocol.max_pingpong_turns():
            protocol.metrics.anti_loop_triggers += 1
            event = protocol.build_streaming_event("status", task_id, context_id, {
                "state": protocol.STATE_FAILED,
                "message": f"Anti-loop protection: exceeded {protocol.max_pingpong_turns()} turns.",
            })
            handler.wfile.write(event.encode("utf-8"))
            done = protocol.build_streaming_event("done", task_id, context_id)
            handler.wfile.write(done.encode("utf-8"))
            return

        # 1. Send submitted event
        event = protocol.build_streaming_event("task", task_id, context_id, {
            "status": {"state": protocol.STATE_SUBMITTED, "timestamp": protocol._now_iso()},
        })
        handler.wfile.write(event.encode("utf-8"))
        handler.wfile.flush()

        if not text:
            event = protocol.build_streaming_event("status", task_id, context_id, {
                "status": {"state": protocol.STATE_FAILED, "message": "Empty task — nothing to do."},
            })
            handler.wfile.write(event.encode("utf-8"))
            done = protocol.build_streaming_event("done", task_id, context_id)
            handler.wfile.write(done.encode("utf-8"))
            return

        framed = security.wrap_inbound(peer, text)
        security.audit("inbound", peer, task_id, text)
        protocol.persist_message(context_id, "user", text, task_id)
        protocol.metrics.inbound_total += 1

        # 2. Send working event
        event = protocol.build_streaming_event("task", task_id, context_id, {
            "status": {"state": protocol.STATE_WORKING, "timestamp": protocol._now_iso()},
        })
        handler.wfile.write(event.encode("utf-8"))
        handler.wfile.flush()

        # Route into agent and wait for reply
        if self._loop is None or self._message_handler is None:
            event = protocol.build_streaming_event("status", task_id, context_id, {
                "status": {"state": protocol.STATE_FAILED, "message": "Agent not ready."},
            })
            handler.wfile.write(event.encode("utf-8"))
            done = protocol.build_streaming_event("done", task_id, context_id)
            handler.wfile.write(done.encode("utf-8"))
            return

        fut: Future = Future()
        with self._pending_lock:
            self._pending_replies[context_id] = fut

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
            with self._pending_lock:
                self._pending_replies.pop(context_id, None)
            event = protocol.build_streaming_event("status", task_id, context_id, {
                "status": {"state": protocol.STATE_FAILED, "message": f"Dispatch failed: {e}"},
            })
            handler.wfile.write(event.encode("utf-8"))
            done = protocol.build_streaming_event("done", task_id, context_id)
            handler.wfile.write(done.encode("utf-8"))
            return

        # 3. Wait for reply (with keepalive pings)
        start = time.time()
        reply = None
        while True:
            try:
                reply = fut.result(timeout=5)
                break
            except TimeoutError:
                # Send keepalive comment
                handler.wfile.write(b": keepalive\n\n")
                handler.wfile.flush()
                if time.time() - start > _REPLY_TIMEOUT:
                    reply = "[agent did not reply in time]"
                    break
            except Exception:
                reply = "[agent did not reply in time]"
                break

        with self._pending_lock:
            self._pending_replies.pop(context_id, None)

        reply = security.redact_outbound(reply or "")
        protocol.persist_message(context_id, "agent", reply, task_id)
        security.audit("outbound", peer, task_id, reply)
        protocol.metrics.outbound_total += 1
        protocol.metrics.tasks_completed += 1

        # 4. Send completed event with reply
        event = protocol.build_streaming_event("task", task_id, context_id, {
            "status": {"state": protocol.STATE_COMPLETED, "timestamp": protocol._now_iso()},
            "artifacts": [{"parts": [{"kind": "text", "text": reply}]}],
        })
        handler.wfile.write(event.encode("utf-8"))
        handler.wfile.flush()

        # 5. Send done event
        done = protocol.build_streaming_event("done", task_id, context_id)
        handler.wfile.write(done.encode("utf-8"))
        handler.wfile.flush()

        # Push notification if registered
        self._send_push_notification(task_id, context_id, reply, protocol.STATE_COMPLETED)

    # ── Push notifications ────────────────────────────────────────────────

    def _send_push_notification(self, task_id: str, context_id: str, reply: str, state: str) -> None:
        """Send a push notification to the registered callback URL for this task."""
        with self._push_lock:
            callback_url = self._push_callbacks.pop(task_id, None)

        if not callback_url:
            return

        payload = {
            "taskId": task_id,
            "contextId": context_id,
            "state": state,
            "reply": reply[:2000],  # cap payload size
            "timestamp": protocol._now_iso(),
        }

        # HMAC sign the payload
        signature = security.sign_push_payload(payload)
        headers = {"Content-Type": "application/json"}
        if signature:
            headers["X-A2A-Signature"] = signature

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(callback_url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                if resp.status == 200:
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

        ``chat_id`` is the A2A context id we set as the source chat_id, so it
        keys straight back to the blocked HTTP request.

        The gateway marks final user-visible replies with ``metadata['notify']``.
        Progress, status, and editable preview sends intentionally lack that
        marker; those must not satisfy the JSON-RPC caller, or the caller sees
        a banner/status update instead of the agent's actual answer.
        """
        is_final_reply = bool((metadata or {}).get("notify"))
        with self._pending_lock:
            fut = self._pending_replies.get(chat_id)
            if fut is not None and not fut.done():
                if not is_final_reply:
                    logger.debug("A2A: ignoring non-final send for context %s", chat_id)
                    return SendResult(success=True, message_id=str(int(time.time() * 1000)))
                fut.set_result(content or "")
                return SendResult(success=True, message_id=str(int(time.time() * 1000)))
        # No waiter (e.g. a late streamed chunk or out-of-band send) — drop it.
        logger.debug("A2A: send() for context %s had no pending waiter", chat_id)
        return SendResult(success=True, message_id=str(int(time.time() * 1000)))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": f"a2a:{chat_id}", "type": "dm"}