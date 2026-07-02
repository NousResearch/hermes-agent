"""
A2A inbound platform adapter — exposes Hermes as an A2A-discoverable agent.

Design (the #11025 insight, done as a plugin with zero core edits):
  - Runs a stdlib http.server in a daemon thread (no a2a-sdk, no asyncio loop
    dependency at register() time — avoids the a2a_fleet "register outside a
    loop" bug class).
  - Serves the Agent Card at GET /.well-known/agent.json.
  - Accepts JSON-RPC ``message/send`` at POST /.
  - Each inbound task is filtered + framed (security.wrap_inbound) and routed
    into the agent's LIVE gateway session via the normal MessageEvent path, so
    the agent that replies is the same one talking to its user — full memory
    and context, not a throwaway clone.
  - The agent's reply comes back through ``adapter.send()``; we override that to
    fulfill a per-context Future the HTTP handler is blocked on, turning the
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
        self._pending_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "A2A"

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self) -> bool:
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

            def do_GET(self):  # noqa: N802
                if self.path.rstrip("/") in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
                    self._json(200, adapter._build_card())
                    return
                if self.path.rstrip("/") in ("", "/health"):
                    self._json(200, {"status": "ok", "agent": adapter.agent_name})
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

                if method in ("message/send", "message/stream"):
                    # We answer message/stream as a single (non-streamed) result.
                    result = adapter._handle_inbound_task(params)
                    self._json(200, protocol.jsonrpc_result(req_id, result))
                    return
                if method == "tasks/get":
                    self._json(200, protocol.jsonrpc_result(req_id, {"error": "task store not retained"}))
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
        self._mark_connected()

        exposure = "localhost-only" if security.localhost_only() else "REMOTE (bearer auth)"
        logger.info(
            "A2A: serving Agent Card + JSON-RPC on http://%s:%s (%s) as %r",
            self.host, self.port, exposure, self.agent_name,
        )
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
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

    # ── Agent Card ────────────────────────────────────────────────────────

    def _build_card(self) -> dict:
        toolsets = []
        try:
            extra = getattr(self.config, "extra", {}) or {}
            toolsets = list(extra.get("advertised_toolsets") or [])
        except Exception:
            pass
        return protocol.build_agent_card(
            name=self.agent_name,
            url=f"http://{self.host}:{self.port}/",
            description=os.getenv(
                "A2A_AGENT_DESCRIPTION",
                "Hermes Agent — a general-purpose agent reachable over A2A.",
            ),
            skills=protocol.skills_from_toolsets(toolsets),
            streaming=False,
            auth_required=not security.localhost_only(),
        )

    # ── Inbound task handling ─────────────────────────────────────────────

    def _handle_inbound_task(self, params: dict) -> dict:
        """Route an inbound A2A task into the live session and wait for reply.

        Runs on an HTTP worker thread. It marshals a MessageEvent onto the
        gateway loop and blocks (on a Future) until adapter.send() fulfils it.
        """
        text = protocol.extract_text(params)
        peer = str(params.get("peer") or (params.get("message", {}) or {}).get("from") or "remote-agent")
        context_id = (params.get("message", {}) or {}).get("contextId") or protocol.new_context_id()
        task_id = protocol.new_task_id()

        if not text:
            return protocol.build_task(task_id, context_id, protocol.STATE_FAILED, "Empty task — nothing to do.")

        framed = security.wrap_inbound(peer, text)
        security.audit("inbound", peer, task_id, text)
        protocol.persist_message(context_id, "user", text, task_id)

        if self._loop is None or self._message_handler is None:
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
            return protocol.build_task(task_id, context_id, protocol.STATE_FAILED, f"Dispatch failed: {e}")

        try:
            reply = fut.result(timeout=_REPLY_TIMEOUT)
        except Exception:
            reply = "[agent did not reply in time]"
        finally:
            with self._pending_lock:
                self._pending_replies.pop(context_id, None)

        reply = security.redact_outbound(reply or "")
        protocol.persist_message(context_id, "agent", reply, task_id)
        security.audit("outbound", peer, task_id, reply)
        return protocol.build_task(task_id, context_id, protocol.STATE_COMPLETED, reply)

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

        Concurrency note: the entire read-and-set sequence runs under
        _pending_lock. The original code released the lock before checking
        fut.done() and calling fut.set_result(), creating a TOCTOU race
        where disconnect() could call set_result() on the same Future
        between the lock release and the operation, raising InvalidStateError.
        """
        with self._pending_lock:
            fut = self._pending_replies.get(chat_id)
            if fut is not None and not fut.done():
                fut.set_result(content or "")
                return SendResult(success=True, message_id=str(int(time.time() * 1000)))
        # No waiter (e.g. a late streamed chunk or out-of-band send) — drop it.
        logger.debug("A2A: send() for context %s had no pending waiter", chat_id)
        return SendResult(success=True, message_id=str(int(time.time() * 1000)))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": f"a2a:{chat_id}", "type": "dm"}
