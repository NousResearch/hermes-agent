"""Inbound Streamable-HTTP MCP platform adapter.

A remote MCP client (Claude Code, Codex, any MCP host) connects to ``/mcp`` with a bearer
token and talks to *this* Hermes — the long-lived gateway agent with its memory, tools and
skills — as a chat peer. This is the inverse of ``hermes mcp serve``, which exposes
Hermes' *tools* to a host; here the host hands Hermes a task and waits for the reply.

Tools exposed to the client:
  whoami              authenticated caller name
  new_conversation    mint a fresh conversation id
  chat                start a Hermes turn (returns immediately)
  wait_reply          long-poll for the reply; shows live tool activity while working
  status              elapsed time + recent activity for a conversation
  cancel              interrupt a running turn
  history             recent exchanges (persisted; survives gateway restarts)

Inbound text is filtered and framed, then injected into the live gateway session. The
gateway routes tool-progress bubbles through ``send()``/``edit_message()`` WITHOUT
``metadata["notify"]`` and the final reply WITH ``notify=True``; the former feed the
``wait_reply`` activity line, the latter resolves the waiter.
"""

from __future__ import annotations

import asyncio
import collections
import contextvars
import json
import logging
import threading
import time
import uuid
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Optional

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)

from . import history, security
from .history import safe_conversation_id as _safe_conv

logger = logging.getLogger(__name__)

MAX_WAIT_S = 55.0    # common reverse proxies cut idle HTTP responses at 60-100s; stay under
_PROGRESS_KEEP = 12  # recent activity lines kept per conversation

_current_peer: contextvars.ContextVar[str] = contextvars.ContextVar("mcp_http_peer", default="")


def _fmt_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


async def _send_json(send, status: int, body: dict) -> None:
    raw = json.dumps(body).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode("ascii"))],
    })
    await send({"type": "http.response.body", "body": raw})


class _IdentityASGI:
    """Bearer auth on every request except ``/health``; the resolved identity is placed in a
    ContextVar so the MCP tool functions (which have no request object) can read it."""

    def __init__(self, app, settings: security.Settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if (scope.get("path") or "/").rstrip("/") == "/health":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in (scope.get("headers") or [])}
        client_ip = (scope.get("client") or ("", 0))[0]
        peer = security.authenticate(headers.get("authorization"), client_ip)
        if not peer:
            await _send_json(send, 401, {"error": "unauthorized"})
            return
        if not security.is_trusted_peer(peer, self.settings):
            await _send_json(send, 403, {"error": "forbidden", "peer": peer})
            return
        token = _current_peer.set(peer)
        try:
            await self.app(scope, receive, send)
        finally:
            _current_peer.reset(token)


@dataclass
class _Conv:
    """Per-conversation live state (memory only; the transcript itself is in ``history``)."""

    peer: str = ""
    future: Optional[Future] = None
    started_at: float = 0.0
    last_activity_at: float = 0.0
    last_message: str = ""
    progress: collections.deque = field(default_factory=lambda: collections.deque(maxlen=_PROGRESS_KEEP))
    progress_msgs: dict = field(default_factory=dict)  # message_id -> latest bubble text
    last_reply: str = ""
    turns: int = 0

    @property
    def busy(self) -> bool:
        return self.future is not None and not self.future.done()


class McpHttpAdapter(BasePlatformAdapter):
    # The caller only ever sees the FINAL reply (wait_reply returns whole text), so partial token
    # streaming buys nothing here. With editing "supported" the gateway streams the reply through
    # edit_message(finalize=True) and then SUPPRESSES its notify-send, leaving the waiter with an
    # empty reply. Declaring no-edit makes the gateway skip streaming for this platform and deliver
    # the reply via send(notify=True). Tool-progress bubbles still arrive and are captured.
    SUPPORTS_MESSAGE_EDITING = False

    # Gateway onboarding notices mean nothing to an MCP caller (there is no chat to /sethome).
    _NOISE_PREFIXES = ("📬 No home channel is set", "Type /sethome")

    def __init__(self, config, **kwargs):
        super().__init__(config=config, platform=Platform("mcp_http"))
        self.settings = security.Settings.from_extra(getattr(config, "extra", {}) or {})
        self.port = self.settings.port
        self.host = self.settings.bind_host()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._convs: dict[str, _Conv] = {}
        self._lock = threading.Lock()
        self._limiter = security.RateLimiter(self.settings.rate_limit)

    @property
    def authorization_is_upstream(self) -> bool:
        """Identity is established by the bearer token in ``_IdentityASGI``; the gateway's
        per-user allowlist does not apply to a peer that has already been authenticated."""
        return True

    # ------------------------------------------------------------------ state helpers

    def _conv(self, cid: str, peer: str = "") -> _Conv:
        with self._lock:
            c = self._convs.get(cid)
            if c is None:
                c = _Conv(peer=peer)
                self._convs[cid] = c
            elif peer and not c.peer:
                c.peer = peer
            return c

    @staticmethod
    def _require_peer() -> str:
        peer = _current_peer.get()
        if not peer:
            raise RuntimeError("MCP HTTP tool called with no authenticated identity")
        return peer

    @staticmethod
    def _own_conv(peer: str, cid: str) -> Optional[str]:
        """Conversation ids are namespaced by peer (``<peer>`` or ``<peer>-<suffix>``) so one
        client can never read or drive another client's thread."""
        if cid == _safe_conv(peer) or cid.startswith(_safe_conv(peer) + "-"):
            return None
        return f"conversation_id={cid} does not belong to {peer}"

    def _source_for(self, peer: str, cid: str):
        return self.build_source(chat_id=cid, chat_name=f"mcp:{peer}", chat_type="dm", user_id=peer, user_name=peer)

    # -------------------------------------------------------------------- turn flow

    def _resolve(self, cid: str, reply: str, *, failed: bool = False) -> bool:
        text = security.redact_outbound(("[failed] " + reply) if failed else (reply or ""))
        with self._lock:
            c = self._convs.get(cid)
            if c is None:
                return False
            fut = c.future
            resolved = fut is not None and not fut.done()
            if resolved:
                c.last_reply = text
                c.last_activity_at = time.time()
                fut.set_result(text)
            elif text.strip():
                # A second notify-send after the turn was already answered: keep it so the
                # caller still sees it through history / the next wait_reply.
                c.last_reply = (c.last_reply + "\n\n" + text).strip() if c.last_reply else text
            peer = c.peer
        if resolved:
            security.audit("outbound", peer, cid, text)
            history.append(cid, "hermes", text)
        return resolved

    def _start_chat(self, peer: str, cid: str, message: str) -> str:
        if not self._limiter.allow(peer):
            return "rate limited — wait a minute and try chat again"
        c = self._conv(cid, peer)
        if c.busy:
            return (
                f"already working on conversation_id={cid} ({_fmt_elapsed(time.time() - c.started_at)} so far, "
                f"last activity: {c.progress[-1] if c.progress else 'starting'}). "
                "Call wait_reply until it returns done, or cancel it."
            )
        if self._loop is None or self._message_handler is None:
            return "Hermes gateway is not attached to this MCP server yet — retry in a few seconds."
        framed = security.wrap_inbound(peer, message)
        security.audit("inbound", peer, cid, message)
        history.append(cid, peer, message)
        with self._lock:
            c.future = Future()
            c.started_at = c.last_activity_at = time.time()
            c.last_message = message
            c.progress.clear()
            c.progress_msgs.clear()
            c.turns += 1
        event = MessageEvent(
            text=framed, message_type=MessageType.TEXT, user_id=peer, user_name=peer,
            source=self._source_for(peer, cid), message_id=uuid.uuid4().hex, allow_gateway_control=False,
        )
        try:
            asyncio.run_coroutine_threadsafe(self.handle_message(event), self._loop)
        except RuntimeError as exc:  # loop closed (gateway shutting down)
            self._resolve(cid, str(exc), failed=True)
            return security.redact_outbound(f"Dispatch failed: {exc}")
        return (
            f"accepted conversation_id={cid}. Hermes is working. "
            f"Call wait_reply(conversation_id, timeout_s={int(MAX_WAIT_S)}) — it long-polls and returns "
            "'working … last activity: <what Hermes is doing>' until the reply is ready, then 'done' plus "
            "the text. Use status() for a summary or cancel() to stop. Do not start a second chat on "
            "the same id while it is working."
        )

    def _wait_reply(self, cid: str, timeout_s: float = MAX_WAIT_S) -> str:
        timeout_s = max(1.0, min(float(timeout_s or MAX_WAIT_S), MAX_WAIT_S))
        c = self._conv(cid)
        fut = c.future
        if fut is None:
            if c.last_reply:
                return f"done conversation_id={cid}\n\n{c.last_reply}"
            return f"no active chat for conversation_id={cid}. Call chat first."
        try:
            reply = fut.result(timeout=timeout_s)
        except FuturesTimeout:
            return self._working_line(cid, c)
        return f"done conversation_id={cid}\n\n{reply}"

    @staticmethod
    def _working_line(cid: str, c: _Conv) -> str:
        elapsed = _fmt_elapsed(time.time() - c.started_at)
        since = _fmt_elapsed(time.time() - c.last_activity_at)
        recent = list(c.progress)[-3:]
        activity = " | ".join(recent) if recent else "thinking (no tool calls yet)"
        return (
            f"working conversation_id={cid} — {elapsed} elapsed, last activity {since} ago: {activity}. "
            "Call wait_reply again with the same id."
        )

    def _status(self, cid: str) -> str:
        c = self._conv(cid)
        if c.started_at == 0 and not c.last_reply:
            return f"no conversation yet for conversation_id={cid}."
        state = "working" if c.busy else "idle"
        lines = [f"conversation_id={cid} state={state} turns={c.turns}"]
        if c.busy:
            lines.append(
                f"elapsed: {_fmt_elapsed(time.time() - c.started_at)}; "
                f"last activity {_fmt_elapsed(time.time() - c.last_activity_at)} ago"
            )
            lines.append(f"request: {c.last_message[:200]}")
        if c.progress:
            lines.append("recent activity:")
            lines.extend(f"  - {p}" for p in c.progress)
        if not c.busy and c.last_reply:
            lines.append(f"last reply ({len(c.last_reply)} chars): {c.last_reply[:300]}")
        return "\n".join(lines)

    def _cancel(self, peer: str, cid: str) -> str:
        c = self._conv(cid)
        if not c.busy:
            return f"nothing running on conversation_id={cid}."
        runner = self.gateway_runner
        if self._loop is None or runner is None:
            return "cancel unavailable (gateway loop not attached)."
        source = self._source_for(peer, cid)

        async def _do():
            session_key = runner._session_key_for_source(source)
            await runner._interrupt_and_clear_session(
                session_key, source,
                interrupt_reason=f"mcp_http cancel by {peer}", invalidation_reason="mcp_http_cancel",
            )
            await self.interrupt_session_activity(session_key, cid)

        try:
            asyncio.run_coroutine_threadsafe(_do(), self._loop).result(timeout=10)
        except Exception as exc:  # any interrupt failure is reported to the caller, not raised into MCP
            logger.warning("MCP HTTP: cancel failed for %s: %s", cid, exc)
            return security.redact_outbound(f"cancel requested but interrupt failed: {exc}")
        self._resolve(cid, "cancelled by caller", failed=True)
        return f"cancelled conversation_id={cid} after {_fmt_elapsed(time.time() - c.started_at)}."

    # ---------------------------------------------------------------------- MCP app

    def _build_mcp(self):
        from mcp.server.mcpserver import MCPServer
        from starlette.responses import JSONResponse

        mcp = MCPServer(
            "hermes-agent",
            instructions=(
                "This is a Hermes Agent instance on its operator's machine — a full agent with its own "
                "tools, memory and access. Talk to it like a colleague: give it the task and context.\n"
                "Loop: chat(message, conversation_id) returns immediately; then call "
                f"wait_reply(conversation_id, timeout_s={int(MAX_WAIT_S)}) repeatedly — while Hermes works it "
                "returns 'working … last activity: <tool activity>' so you can see progress; when finished it "
                "returns 'done' plus the reply. Turns commonly take 1–10 minutes for real investigations; keep "
                "polling rather than giving up. status() summarises a conversation; cancel() interrupts it; "
                "history() shows recent exchanges (useful after you restart). Reuse the same conversation_id to "
                "continue a thread with full context; new_conversation() starts a clean one. whoami() returns your "
                "authenticated identity."
            ),
        )
        adapter = self

        def _cid(peer: str, conversation_id: str) -> str:
            return _safe_conv(conversation_id) if conversation_id else _safe_conv(peer)

        @mcp.custom_route("/health", methods=["GET"])
        async def health(_request):
            with adapter._lock:
                busy = sum(1 for c in adapter._convs.values() if c.busy)
            return JSONResponse({
                "ok": True, "service": "hermes-mcp-http", "version": 2,
                "bind": f"{adapter.host}:{adapter.port}", "url": adapter.settings.display_url(adapter.host),
                "gateway_attached": adapter._loop is not None and adapter._message_handler is not None,
                "busy_conversations": busy,
            })

        @mcp.tool()
        def whoami() -> str:
            """Return the authenticated caller identity for this MCP connection."""
            return adapter._require_peer()

        @mcp.tool()
        def new_conversation() -> str:
            """Start a fresh Hermes conversation (clean context). Pass the returned id to chat()."""
            return f"{_safe_conv(adapter._require_peer())}-{uuid.uuid4().hex[:10]}"

        @mcp.tool()
        def chat(message: str, conversation_id: str = "") -> str:
            """Start a Hermes turn. Returns immediately; then poll wait_reply.

            Args:
                message: What you want Hermes to do or answer. Include the context it needs.
                conversation_id: Optional thread id. Omit for your stable per-identity
                    default thread. Reuse the same id to continue with full context.
            """
            peer = adapter._require_peer()
            if not (message or "").strip():
                return "message is required"
            cid = _cid(peer, conversation_id)
            return adapter._own_conv(peer, cid) or adapter._start_chat(peer, cid, message)

        @mcp.tool()
        def wait_reply(conversation_id: str = "", timeout_s: float = MAX_WAIT_S) -> str:
            """Long-poll for the Hermes reply. Returns 'working … last activity: …' with live
            tool activity until Hermes finishes, then 'done' plus the reply text.

            Args:
                conversation_id: Same id chat() returned or you passed in.
                timeout_s: Seconds to wait this call (max 55). Use the max; it returns early when done.
            """
            peer = adapter._require_peer()
            cid = _cid(peer, conversation_id)
            return adapter._own_conv(peer, cid) or adapter._wait_reply(cid, timeout_s)

        @mcp.tool()
        def status(conversation_id: str = "") -> str:
            """Summarise a conversation: working/idle, elapsed time, recent tool activity, last reply.

            Args:
                conversation_id: Thread id (omit for your default thread).
            """
            peer = adapter._require_peer()
            cid = _cid(peer, conversation_id)
            return adapter._own_conv(peer, cid) or adapter._status(cid)

        @mcp.tool()
        def cancel(conversation_id: str = "") -> str:
            """Interrupt the turn currently running on a conversation.

            Args:
                conversation_id: Thread id (omit for your default thread).
            """
            peer = adapter._require_peer()
            cid = _cid(peer, conversation_id)
            return adapter._own_conv(peer, cid) or adapter._cancel(peer, cid)

        @mcp.tool(name="history")  # `history` is the imported module; the tool keeps the public name
        def history_tool(conversation_id: str = "", limit: int = 10) -> str:
            """Recent exchanges on a conversation (persisted; survives Hermes restarts).

            Args:
                conversation_id: Thread id (omit for your default thread).
                limit: Number of exchanges (your message + Hermes reply pairs), max 20.
            """
            peer = adapter._require_peer()
            cid = _cid(peer, conversation_id)
            return adapter._own_conv(peer, cid) or history.render(cid, limit)

        return mcp

    def _run_uvicorn(self, app) -> None:
        import uvicorn

        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning", lifespan="on")
        self._server = uvicorn.Server(config)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._server.serve())
        finally:
            loop.close()

    async def connect(self, **_kwargs) -> bool:
        self._loop = asyncio.get_running_loop()
        try:
            from gateway.status import acquire_scoped_lock
            lock_key = f"{self.host}:{self.port}"
            if not acquire_scoped_lock("mcp_http", lock_key):
                logger.error("MCP HTTP: %s already served by another profile", lock_key)
                self._set_fatal_error("lock_conflict", "MCP HTTP port in use by another profile", retryable=False)
                return False
            self._lock_key = lock_key
        except ImportError:
            self._lock_key = None  # status module not available (tests)
        try:
            inner = self._build_mcp().streamable_http_app(
                streamable_http_path="/mcp", stateless_http=False,
                transport_security=security.transport_security(self.settings), host=self.host,
            )
            app = _IdentityASGI(inner, self.settings)
        except Exception as exc:  # SDK import/version drift — report, let the gateway retry
            logger.error("MCP HTTP: failed to build server: %s", exc)
            self._set_fatal_error("build_failed", f"MCP HTTP build failed: {exc}", retryable=True)
            return False

        self._thread = threading.Thread(target=self._run_uvicorn, args=(app,), name="mcp-http", daemon=True)
        self._thread.start()
        for _ in range(100):
            if self._server is not None and getattr(self._server, "started", False):
                break
            await asyncio.sleep(0.05)
        self._mark_connected()
        exposure = "localhost-only" if security.localhost_only() else "REMOTE (bearer auth)"
        logger.info("MCP HTTP: serving Streamable HTTP MCP on %s:%s (%s) url=%s",
                    self.host, self.port, exposure, self.settings.display_url(self.host))
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        lock_key = getattr(self, "_lock_key", None)
        if lock_key:
            from gateway.status import release_scoped_lock
            release_scoped_lock("mcp_http", lock_key)
        server = self._server
        if server is not None:
            server.should_exit = True
        with self._lock:
            for c in self._convs.values():
                if c.future is not None and not c.future.done():
                    c.future.set_result("[agent shutting down — call chat again in ~30s]")
        self._server = None

    # ------------------------------------------------------- gateway -> adapter callbacks

    def _record_progress(self, cid: str, message_id: str, content: str) -> None:
        """Tool-progress bubble (a send without ``notify``, or an edit of one)."""
        text = (content or "").strip()
        if not text or text.startswith(self._NOISE_PREFIXES):
            return
        with self._lock:
            c = self._convs.get(cid)
            if c is None:
                return
            c.last_activity_at = time.time()
            prev = c.progress_msgs.get(message_id, "")
            c.progress_msgs[message_id] = text
            # Bubbles accumulate lines via edit; only surface the NEW tail lines.
            new_lines = text[len(prev):].splitlines() if prev and text.startswith(prev) else text.splitlines()
            for line in new_lines:
                line = line.strip()
                if line.startswith(self._NOISE_PREFIXES):
                    continue
                if line and (not c.progress or c.progress[-1] != line):
                    c.progress.append(line[:200])

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[dict] = None) -> SendResult:
        message_id = uuid.uuid4().hex[:12]
        if (metadata or {}).get("notify"):
            if not self._resolve(chat_id, content or ""):
                logger.debug("MCP HTTP: send() for %s had no waiter", chat_id)
        else:
            self._record_progress(chat_id, message_id, content)
        return SendResult(success=True, message_id=message_id)

    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False,
                           metadata: Optional[dict] = None) -> SendResult:
        self._record_progress(chat_id, message_id, content)
        return SendResult(success=True, message_id=message_id)

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        cid = str(getattr(getattr(event, "source", None), "chat_id", "") or "")
        if not cid:
            return
        if outcome == ProcessingOutcome.FAILURE:
            self._resolve(cid, "agent processing failed", failed=True)
        elif outcome == ProcessingOutcome.CANCELLED:
            self._resolve(cid, "cancelled", failed=True)
        else:
            c = self._convs.get(cid)
            if c is not None and c.busy:
                # Finished without a notify-send: never hand back an empty string.
                tail = " | ".join(list(c.progress)[-5:])
                self._resolve(cid, "(Hermes finished this turn without a text reply."
                              + (f" Recent activity: {tail})" if tail else ")"))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        with self._lock:
            c = self._convs.get(chat_id)
            if c is not None:
                c.last_activity_at = time.time()

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"name": f"mcp:{chat_id}", "type": "dm"}
