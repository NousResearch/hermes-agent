
from __future__ import annotations

import asyncio
import logging
import math
import os
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
from .task_routing import TaskRPCHandler

from . import protocol, security
# HTTP/wire boundary extracted to http_transport.py (physical-line split).
from .http_transport import (  # noqa: F401 — re-exported for monkeypatch targets
    _A2AServer,
    _DATA_TRUNCATED_MARKER,
    _DETAIL_MAX_CODEPOINTS,
    _JSONRPC_CODE_MAX,
    _JSONRPC_CODE_MIN,
    _JSONRPC_KEY_MAX_CODEPOINTS,
    _JSONRPC_MAX_BYTES,
    _JSONRPC_MAX_DEPTH,
    _JSONRPC_MAX_WIDTH,
    _JSONRPC_STRING_MAX_CODEPOINTS,
    _MAX_BODY,
    _PORTABLE_NONBLOCK_RECV,
    _REDACTED_MARKER,
    _TRUNCATION_MARKER,
    _audit_safe,
    _bounded_redacted_detail,
    _failure_outcome,
    _method_info,
    _redacted_jsonrpc_detail,
    _redacted_reply_text,
    _sanitize_jsonrpc_value,
    _sanitize_string_for_jsonrpc,
    _send_result_from_outcome,
    _truncate_codepoints,
    A2ARequestHandler,
)

logger = logging.getLogger(__name__)

# _DEFAULT_PORT and _MAX_CONTEXT_PEERS imported from a2a_persistence
_ORPHAN_TIMEOUT = 300  # seconds before a pending task is considered orphaned
_WATCHDOG_INTERVAL = 60  # seconds between orphaned task watchdog runs
_SSE_KEEPALIVE = 5  # seconds between SSE keepalive comments
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

# Persistence utilities extracted to a2a_persistence.py
from .a2a_persistence import (
    _DEFAULT_PORT,
    _HAS_FCNTL,
    _HAS_MSVCRT,
    _LOOPBACK_ADDRS,
    _MAX_CONTEXT_PEERS,
    _MSVCRT_RETRIES,
    _MSVCRT_RETRY_DELAY,
    _THREAD_FALLBACK_LOCK,
    _active_profile_name,
    _bracket_ipv6,
    _clean_slug,
    _context_peers_path,
    _context_sessions_path,
    _default_agent_name,
    _fanout_children_path,
    _file_lock,
    _file_lock_fcntl,
    _file_lock_msvcrt,
    _file_lock_thread_fallback,
    _is_ipv6_literal,
    _is_own_endpoint,
    _join_url,
    _load_context_peers,
    _load_context_sessions,
    _load_fanout_children,
    _loopback_fallback_url,
    _merge_context_peers,
    _merge_context_sessions,
    _merge_fanout_children,
    _own_a2a_url,
    _persist_context_peers,
    _persist_context_sessions,
    _persist_fanout_children,
    _profile_home,
    _profile_scoped,
    _reply_timeout,
    _reset_worker_session_vars,
    _safe_context_slug,
    _sender_url_acceptable,
    _task_ledger_path,
    _try_persist_task_ledger,
)

class A2AAdapter(BasePlatformAdapter, TaskRPCHandler):

    def __init__(self, config, **kwargs):
        platform = Platform("a2a")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}
        # Scope-aware: a secondary multiplex profile must not borrow the
        # default profile's bridged A2A_PORT (mirrors the Buzz/SimpleX fix
        # for #98738) — an unconfigured profile falls closed to the module
        # default port instead. (advertised_toolsets has the same env-leak
        # shape but is left unscoped here — see the "Scope note" in this
        # fix's PR description: open PR #98937 is actively rewriting this
        # field's None-vs-empty-list semantics.)
        self._security_context = security.A2ASecurityContext.capture()
        _port_env = None if _profile_scoped() else os.getenv("A2A_PORT")
        self.port = int(_port_env or extra.get("port", _DEFAULT_PORT))
        self.host = self._security_context.resolve_bind_host()
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

        # Fan-out children: parent_context_id → {peer_name: child_context_id}
        # Recorded by a2a_orchestrate so callers can resume a specific
        # child branch with a2a_call(context_id=child_context_id), and
        # late callbacks trace back to the originating session.
        self._fanout_children: Dict[str, Dict[str, str]] = {}
        self._fanout_children_lock = threading.Lock()

        # Short-window inbound dedupe: (contextId, messageId) → first
        # arrival time. The same wire message must not be dispatched twice
        # (duplicate handoffs were observed in testing, and the push+retry
        # paths make double-delivery possible).
        self._inbound_seen: Dict[tuple[str, str], float] = {}
        self._inbound_seen_lock = threading.Lock()

        # Orphaned task watchdog
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None

        self._bounded_redacted_detail=_bounded_redacted_detail
        self._redacted_reply_text=_redacted_reply_text
        self._audit_safe=_audit_safe
        # Register this adapter so the outbound client tools can map local
        # contexts back to this gateway's peer table (see _register_context_peer).
        with _ADAPTERS_GUARD:
            _ADAPTERS[id(self)] = weakref.ref(self)

    # ── Cross-platform context peer registration ─────────────────────────

    @classmethod
    def _register_context_peer(cls, context_id: str, peer: str) -> None:
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
        # dropped before any side effect.  Always include the new
        # context_id→peer mapping directly (not only via union) so a
        # registration made with no live adapters (e.g. a CLI/ACP process)
        # is still persisted for the next gateway start.  Merge with
        # the on-disk state and never clobber existing entries.
        # Serialise the load→merge→write cycle with a file lock so
        # two concurrent registrations (e.g. two outbound a2a_call
        # threads) don't clobber each other's disk state.
        with _file_lock(_context_peers_path().with_suffix(".lock")):
            disk = _load_context_peers()
            disk[context_id] = peer
            disk.update(union)
            _persist_context_peers(_merge_context_peers({}, disk, _MAX_CONTEXT_PEERS))

    @classmethod
    def _register_context_session(cls, context_id: str, origin: dict) -> None:
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
        with _file_lock(_context_sessions_path().with_suffix(".lock")):
            disk = _load_context_sessions()
            disk.update(union)
            disk.setdefault(context_id, dict(origin))
            _persist_context_sessions(_merge_context_sessions({}, disk, _MAX_CONTEXT_PEERS))

    @classmethod
    def _own_sender(cls) -> dict:
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        for ref in refs:
            adapter = ref()
            if adapter is not None:
                return adapter._sender_identity()
        return cls._sender_from_config()

    @classmethod
    def _sender_from_config(cls) -> dict:
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
        return {
            "agentId": self.agent_name,
            "name": self.agent_name,
            "url": _own_a2a_url(self.host, self.port),
        }

    def _refine_peer_identity(self, peer: str, params: dict, context_id: str) -> str:
        if not peer.startswith("ip:"):
            return peer
        sender = protocol.extract_sender(params)
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
        # Do not promote from agentId/name alone — require URL/origin validation.
        if name and name in peers_cfg:
            cfg_entry = peers_cfg.get(name) if isinstance(peers_cfg.get(name), dict) else {}
            cfg_url_str = str((cfg_entry or {}).get("url") or "").strip()
            if url and cfg_url_str and _sender_url_acceptable(url, peers_cfg):
                try:
                    cfg_parsed = urllib.parse.urlparse(cfg_url_str)
                    sender_parsed = urllib.parse.urlparse(url)
                    if (cfg_parsed.hostname and sender_parsed.hostname
                            and cfg_parsed.hostname.lower() == sender_parsed.hostname.lower()
                            and cfg_parsed.port == sender_parsed.port):
                        logger.info(
                            "A2A: refined ip: identity for context %s to configured peer %r (sender agentId+url validated)",
                            context_id, name,
                        )
                        return name
                except Exception:
                    pass
            # No validated URL binding for this name — retain authenticated identity
            logger.info(
                "A2A: refusing to promote ip identity for context %s to %r without URL/origin validation; retaining %r",
                context_id, name, peer,
            )
            return peer
        if url and _sender_url_acceptable(url, peers_cfg):
            # Resolve back to the configured peer key when the sender URL
            # matches a configured peer's URL — returning the URL string
            # loses the bearer auth from a2a_agents config.
            for cfg_key, cfg_entry in peers_cfg.items():
                if not isinstance(cfg_entry, dict):
                    continue
                try:
                    cfg_url = urllib.parse.urlparse(str(cfg_entry.get("url") or ""))
                except Exception:
                    continue
                sender_parsed = urllib.parse.urlparse(url)
                if (cfg_url.hostname and sender_parsed.hostname
                        and cfg_url.hostname.lower() == sender_parsed.hostname.lower()
                        and cfg_url.port == sender_parsed.port):
                    logger.info(
                        "A2A: refined ip: identity for context %s to configured peer %r "
                        "(sender url %s matched config key %s — retaining bearer auth)",
                        context_id, cfg_key, url, cfg_key,
                    )
                    return cfg_key
            logger.info(
                "A2A: refined ip: identity for context %s to sender url %s",
                context_id, url,
            )
            return url
        return peer

    @classmethod
    def _origin_delivery_target(cls, context_id: str, platform_name: str) -> dict:
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
        return True

    # ── Fan-out children registration ────────────────────────────────────

    @classmethod
    def _register_fanout_children(
        cls, parent_context_id: str, peer_children: Dict[str, str],
        origin: Optional[dict] = None,
    ) -> None:
        if not parent_context_id or not peer_children:
            return
        new_entry = {parent_context_id: dict(peer_children)}
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        for ref in refs:
            adapter = ref()
            if adapter is None:
                continue
            with adapter._fanout_children_lock:
                adapter._fanout_children = _merge_fanout_children(
                    adapter._fanout_children, new_entry, _MAX_CONTEXT_PEERS,
                )
        # Persist to disk for restart recovery (bounded eviction).
        with _file_lock(_fanout_children_path().with_suffix(".lock")):
            disk = _load_fanout_children()
            merged = _merge_fanout_children(disk, new_entry, _MAX_CONTEXT_PEERS)
            _persist_fanout_children(merged)

    @classmethod
    def _get_fanout_children(cls, parent_context_id: str) -> dict:
        if not parent_context_id:
            return {}
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        for ref in refs:
            adapter = ref()
            if adapter is None:
                continue
            with adapter._fanout_children_lock:
                children = adapter._fanout_children.get(parent_context_id)
                if children:
                    return dict(children)
        disk = _load_fanout_children()
        return dict(disk.get(parent_context_id) or {})

    @classmethod
    def _reject_child_reuse(cls, child_context_id: str, requesting_peer: str) -> str:
        if not child_context_id:
            return ""
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        for ref in refs:
            adapter = ref()
            if adapter is None:
                continue
            with adapter._fanout_children_lock:
                for _parent, children in adapter._fanout_children.items():
                    if not isinstance(children, dict):
                        continue
                    for peer_name, cid in children.items():
                        if cid == child_context_id:
                            if peer_name != requesting_peer:
                                return peer_name
                            return ""
        # Disk fallback.
        disk = _load_fanout_children()
        for _parent, children in disk.items():
            if not isinstance(children, dict):
                continue
            for peer_name, cid in children.items():
                if cid == child_context_id:
                    if peer_name != requesting_peer:
                        return peer_name
                    return ""
        return ""

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
            merged = _merge_context_peers(self._context_peers, restored, _MAX_CONTEXT_PEERS)
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

        # Restore fan-out children map from disk so callers can still
        # resume child branches after a restart.
        restored_fanout = self._restore_persisted_fanout_children()
        if restored_fanout:
            logger.info(
                "A2A: restored %d fan-out parent→children registration(s) from %s",
                restored_fanout, _fanout_children_path(),
            )

        # Restore task ledger so GetTask/ListTasks/SubscribeToTask survive
        # gateway restarts.  Terminal task records (COMPLETED, FAILED,
        # CANCELED) and recent non-terminal tasks are persisted by
        # _persist_task_ledger on every task completion.
        restored_tasks = self.tasks.restore(_task_ledger_path())
        if restored_tasks:
            logger.info(
                "A2A: restored %d task record(s) from %s",
                restored_tasks, _task_ledger_path(),
            )

        self._mark_connected()

        exposure = (
            "localhost-only"
            if self._security_context.localhost_only()
            else "REMOTE (bearer auth)"
        )
        logger.info(
            "A2A: serving Agent Card + JSON-RPC on http://%s:%s (%s) as %r; %d routed agent(s)",
            self.host, self.port, exposure, self.agent_name, len(self._agents),
        )
        # Plugin-registered native handlers (ctx.register_platform_handler).
        self._wire_plugin_handlers(None)
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
        # Durable per-task shutdown: use the same coordinator as send/complete.
        # Do not resolve Futures or clear state before durable publish.
        pending_snapshot: list[tuple[str, str, Any]] = []
        with self._pending_lock:
            for tid, (ctx, fut) in list(self._pending.items()):
                pending_snapshot.append((tid, ctx, fut))
        # First, attempt durable FAILED publish per pending task; only on success resolve waiter
        for tid, ctx, fut in pending_snapshot:
            rec = self.tasks.get(tid)
            if not rec or rec.get("state") in protocol.TERMINAL_STATES:
                # Task already terminal or missing — still clean pending map but don't claim new terminal
                with self._pending_lock:
                    self._pending.pop(tid, None)
                    order = self._pending_order.get(ctx)
                    if order is not None:
                        try:
                            order.remove(tid)
                        except ValueError:
                            pass
                        if not order:
                            self._pending_order.pop(ctx, None)
                continue
            candidate = dict(rec)
            candidate["state"] = protocol.STATE_FAILED
            candidate["reply"] = "[agent shutting down]"
            candidate["completed_at"] = __import__("time").time()
            try:
                outcome = self.tasks.publish_durable(_task_ledger_path(), tid, candidate)
            except Exception as exc:
                logger.error("A2A: disconnect durable publish exception for task %s: %s", tid, exc, exc_info=True)
                outcome = protocol.DurablePublishOutcome(published=False, newly_published=False, record=rec, durable_state=rec.get("state", ""), error=str(exc))
            if outcome.published and outcome.newly_published:
                auth_rec = outcome.record if outcome.record is not None else rec; cb_task_id, cb_context_id, cb_state, cb_reply = str(auth_rec.get("task_id") or tid), str(auth_rec.get("context_id") or auth_rec.get("contextId") or ""), str(auth_rec.get("state") or protocol.STATE_FAILED), str(auth_rec.get("reply") if auth_rec.get("reply") is not None else "[agent shutting down]")
                # Commit succeeded — now resolve waiter with authoritative outcome and clean pending
                with self._pending_lock:
                    ent = self._pending.get(tid)
                    if ent is not None and ent[1] is fut and not fut.done():
                        try:
                            fut.set_result((cb_state, cb_reply))
                        except Exception:
                            pass
                    self._pending.pop(tid, None)
                    order = self._pending_order.get(ctx)
                    if order is not None:
                        try:
                            order.remove(tid)
                        except ValueError:
                            pass
                        if not order:
                            self._pending_order.pop(ctx, None)
                try:
                    self._send_push_notification(cb_task_id, cb_context_id, cb_reply, cb_state)
                except Exception:
                    pass
            else:
                logger.error("A2A: failed to durably publish FAILED on disconnect for task %s: %s — leaving WORKING", tid, outcome.error)
                # Leave pending and task at prior WORKING for restart recovery; do not resolve Future with terminal
                # Keep pending entry for potential restart handling, but transport is tearing down — waiter will be abandoned
                # We do NOT clear pending for this failed task; but to avoid leaking, we could keep it for restart.
                # For now, keep it so probe sees Future not done with terminal.
                continue
        # Also handle non-pending non-terminal tasks (orphan shutdown) via per-task durable publish
        try:
            # Snapshot of non-terminal tasks not already handled
            remaining_task_ids = []
            with self.tasks._lock:
                for tid, rec in list(self.tasks._tasks.items()):
                    if rec.get("state") not in protocol.TERMINAL_STATES:
                        # Skip those already attempted above
                        if not any(tid == ptid for ptid, _, _ in pending_snapshot):
                            remaining_task_ids.append(tid)
            for tid in remaining_task_ids:
                rec = self.tasks.get(tid)
                if not rec or rec.get("state") in protocol.TERMINAL_STATES:
                    continue
                cand = dict(rec)
                cand["state"] = protocol.STATE_FAILED
                cand["reply"] = "[agent shutting down]"
                cand["completed_at"] = __import__("time").time()
                try:
                    outcome = self.tasks.publish_durable(_task_ledger_path(), tid, cand)
                    if not outcome.published:
                        logger.error("A2A: disconnect orphan durable publish failed for %s: %s", tid, outcome.error)
                    elif outcome.newly_published:
                        auth2 = outcome.record if outcome.record is not None else cand; cb2_tid, cb2_ctx, cb2_state, cb2_reply = str(auth2.get("task_id") or tid), str(auth2.get("context_id") or auth2.get("contextId") or ""), str(auth2.get("state") or protocol.STATE_FAILED), str(auth2.get("reply") if auth2.get("reply") is not None else "[agent shutting down]")
                        try:
                            self._send_push_notification(cb2_tid, cb2_ctx, cb2_reply, cb2_state)
                        except Exception:
                            pass
                except Exception as exc:
                    logger.error("A2A: disconnect orphan publish exception for %s: %s", tid, exc, exc_info=True)
        except Exception:
            logger.error("A2A: failed to persist FAILED state on disconnect/shutdown", exc_info=True)

    # ── Orphaned task watchdog ─────────────────────────────────────────────

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(_WATCHDOG_INTERVAL):
            try:
                failed = self.tasks.fail_orphans(_ORPHAN_TIMEOUT)
                if failed:
                    for tid in failed:
                        logger.warning("A2A: orphaned task %s marked failed (timeout %ds)", tid, _ORPHAN_TIMEOUT)
                        protocol.metrics.tasks_failed += 1
                        try:
                            rec = self.tasks.get(tid)
                            if rec is not None:
                                self._send_push_notification(tid, rec.get("context_id", ""), rec.get("reply", ""), rec.get("state", ""))
                        except Exception:
                            pass
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
        raw = extra.get("agents") or extra.get("served_agents")
        if raw is None:
            cfg = self._load_global_a2a_config()
            raw = cfg.get("a2a_served_agents") or (cfg.get("a2a") or {}).get("served_agents")

        agents: dict[str, dict] = {}
        # Scope-aware for the same reason as port/toolsets above: a secondary
        # profile must not inherit the default profile's A2A_AGENT_DESCRIPTION.
        default_desc = (
            "Hermes Agent — a general-purpose agent reachable over A2A."
            if _profile_scoped()
            else os.getenv(
                "A2A_AGENT_DESCRIPTION",
                "Hermes Agent — a general-purpose agent reachable over A2A.",
            )
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
            auth_required=not self._security_context.localhost_only(),
            tenant=str(agent.get("tenant") or ""),
        )

    def _advertised_skills(self, agent: Optional[dict] = None) -> list[dict]:
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

    def _find_existing_nonterminal_task(self, context_id: str) -> Optional[dict]:
        recs, _ = self.tasks.list(context_id=context_id)
        for rec in recs:
            if rec["state"] not in protocol.TERMINAL_STATES:
                return rec
        return None

    def _prepare_task(self, params: dict, peer: str, agent: Optional[dict] = None) -> tuple[Optional[dict], Optional[dict]]:
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
            # Durable immediate rejection via disk-first primitive (section 5.7)
            agent_slug, tenant = self._scope_for_agent(agent)
            _now = time.time()
            _now_iso = protocol.now_iso()
            _candidate = {
                "task_id": task_id,
                "context_id": context_id,
                "peer": peer,
                "agent_slug": agent_slug,
                "tenant": tenant,
                "state": protocol.STATE_REJECTED,
                "reply": "",
                "created_at": _now,
                "created_iso": _now_iso,
                "push_url": "",
                "push_config_id": "",
            }
            _outcome = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate)
            if not _outcome.published:
                logger.error("A2A: failed to durably publish REJECTED dedupe for task %s: %s", task_id, _outcome.error)
                raise protocol.DurablePublishError(task_id, context_id, protocol.STATE_REJECTED, _outcome.durable_state, False)
            logger.warning(
                "A2A: duplicate inbound message %s on context %s within the dedupe window; rejecting",
                message_id, context_id,
            )
            rec = _outcome.record or _candidate
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
            agent_slug, tenant = self._scope_for_agent(agent)
            _now = time.time()
            _now_iso = protocol.now_iso()
            _candidate = {
                "task_id": task_id,
                "context_id": context_id,
                "peer": peer,
                "agent_slug": agent_slug,
                "tenant": tenant,
                "state": protocol.STATE_REJECTED,
                "reply": "",
                "created_at": _now,
                "created_iso": _now_iso,
                "push_url": "",
                "push_config_id": "",
            }
            _outcome = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate)
            if not _outcome.published:
                logger.error("A2A: failed to durably publish REJECTED anti-loop for task %s: %s", task_id, _outcome.error)
                raise protocol.DurablePublishError(task_id, context_id, protocol.STATE_REJECTED, _outcome.durable_state, False)
            rec = _outcome.record or _candidate
            return protocol.build_task(
                task_id, context_id, protocol.STATE_REJECTED,
                f"Anti-loop protection: context {context_id} exceeded "
                f"{protocol.max_pingpong_turns()} turns. Start a new context or "
                f"increase A2A_MAX_PINGPONG_TURNS.",
                created_at=rec["created_iso"],
            ), None

        if not text:
            agent_slug, tenant = self._scope_for_agent(agent)
            _now = time.time()
            _now_iso = protocol.now_iso()
            _candidate = {
                "task_id": task_id,
                "context_id": context_id,
                "peer": peer,
                "agent_slug": agent_slug,
                "tenant": tenant,
                "state": protocol.STATE_REJECTED,
                "reply": "",
                "created_at": _now,
                "created_iso": _now_iso,
                "push_url": "",
                "push_config_id": "",
            }
            _outcome = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate)
            if not _outcome.published:
                logger.error("A2A: failed to durably publish REJECTED empty for task %s: %s", task_id, _outcome.error)
                raise protocol.DurablePublishError(task_id, context_id, protocol.STATE_REJECTED, _outcome.durable_state, False)
            rec = _outcome.record or _candidate
            return protocol.build_task(
                task_id, context_id, protocol.STATE_REJECTED,
                "Empty task — nothing to do.", created_at=rec["created_iso"],
            ), None

        framed = security.wrap_inbound(peer, text)
        security.audit("inbound", peer, task_id, text)
        protocol.persist_message(context_id, "user", text, task_id)
        protocol.metrics.inbound_total += 1

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
            with _file_lock(_context_peers_path().with_suffix(".lock")):
                _persist_context_peers(_merge_context_peers(_load_context_peers(), {context_id: peer}, _MAX_CONTEXT_PEERS))
        # Inline push: pure validation before WORKING candidate construction.
        # The helper inspects only configuration.taskPushNotificationConfig,
        # accepts direct .url and nested .pushNotificationConfig.url, requires
        # dict containers / nonblank string / agreement / is_safe_callback_url.
        # Returns (push_url, push_config_id) with cfg- + 12 hex on success,
        # else ("", "").  No TaskStore mutation.
        try:
            _localhost_mode = self._security_context.localhost_only()
        except Exception:
            _localhost_mode = True
        try:
            _inline_url, _inline_cfg_id = self._inline_push_fields(task_id, params, localhost_mode=_localhost_mode)
        except Exception:
            _inline_url, _inline_cfg_id = "", ""

        # Write-ahead: durably create WORKING before any dispatch (disk-first, section 5.7).
        # The task ledger is the authority; memory is updated only after successful disk write.
        # If the write fails, the task remains ABSENT and dispatch is not invoked.
        _agent_slug, _tenant = self._scope_for_agent(agent)
        _now = time.time()
        _now_iso = protocol.now_iso()
        _candidate_working = {
            "task_id": task_id,
            "context_id": context_id,
            "peer": peer,
            "agent_slug": _agent_slug,
            "tenant": _tenant,
            "state": protocol.STATE_WORKING,
            "reply": "",
            "created_at": _now,
            "created_iso": _now_iso,
            "push_url": _inline_url,
            "push_config_id": _inline_cfg_id,
        }
        _outcome_working = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate_working)
        if not _outcome_working.published:
            logger.error("A2A: failed to durably publish WORKING for task %s: %s", task_id, _outcome_working.error)
            protocol.metrics.tasks_failed += 1
            if _session_tokens:
                try: _reset_worker_session_vars()
                except Exception: pass
            # Fail closed: no dispatch, structured persistence error. The caller (task_routing) will map this to -32603.
            raise protocol.DurablePublishError(task_id, context_id, protocol.STATE_WORKING, _outcome_working.durable_state, False)
        rec = _outcome_working.record or _candidate_working

        if not agent.get("local", True):
            try:
                reply, state = self._forward_to_profile(agent, peer, context_id, framed, task_id)
                # Durable publish for forwarded completion (central commit, section 5.4)
                _candidate_fwd = dict(rec)
                _candidate_fwd["state"] = state
                _candidate_fwd["reply"] = reply
                _candidate_fwd["completed_at"] = time.time()
                _outcome_fwd = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate_fwd)
                if not _outcome_fwd.published:
                    logger.error("A2A: failed to durably publish forwarded terminal %s for task %s: %s", state, task_id, _outcome_fwd.error)
                    protocol.metrics.tasks_failed += 1
                    raise protocol.DurablePublishError(task_id, context_id, state, _outcome_fwd.durable_state, True)
                # Post-commit side effects only after durable publish (section 5.4)
                if _outcome_fwd.newly_published:
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
            _candidate_gw = dict(rec)
            _candidate_gw["state"] = protocol.STATE_FAILED
            _candidate_gw["reply"] = ""
            _candidate_gw["completed_at"] = time.time()
            _outcome_gw = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate_gw)
            if not _outcome_gw.published:
                logger.error("A2A: failed to durably publish FAILED gateway-not-ready for task %s: %s", task_id, _outcome_gw.error)
                raise protocol.DurablePublishError(task_id, context_id, protocol.STATE_FAILED, _outcome_gw.durable_state, True)
            if _outcome_gw.newly_published:
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

        coro = self.handle_message(event)
        try:
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        except Exception as e:
            # Avoid un-awaited coroutine leak: run_coroutine_threadsafe rejects a
            # closed/stopping loop without consuming the coroutine, which would
            # otherwise emit RuntimeWarning: coroutine was never awaited.
            try:
                coro.close()
            except Exception:
                pass
            self._pop_pending(task_id)
            msg = security.redact_outbound(f"Dispatch failed: {e}")
            _candidate_disp = dict(rec)
            _candidate_disp["state"] = protocol.STATE_FAILED
            _candidate_disp["reply"] = msg
            _candidate_disp["completed_at"] = time.time()
            _outcome_disp = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate_disp)
            if not _outcome_disp.published:
                logger.error("A2A: failed to durably publish FAILED dispatch for task %s: %s", task_id, _outcome_disp.error)
                raise protocol.DurablePublishError(task_id, context_id, protocol.STATE_FAILED, _outcome_disp.durable_state, True)
            if _outcome_disp.newly_published:
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

        # Wake the originating local session after durable WORKING so the wake
        # is also ordered after persistence (origin dispatch is a second dispatch).
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

        return None, {
            "task_id": task_id,
            "context_id": context_id,
            "peer": peer,
            "future": fut,
            "created_iso": rec["created_iso"],
            "started": time.time(),
        }

    def _restore_persisted_context_sessions(self) -> int:
        with self._context_sessions_lock:
            restored = _load_context_sessions()
            merged = _merge_context_sessions(self._context_sessions, restored, _MAX_CONTEXT_PEERS)
            self._context_sessions.clear()
            self._context_sessions.update(merged)
        return len(restored)

    def _restore_persisted_fanout_children(self) -> int:
        disk = _load_fanout_children()
        if not disk:
            return 0
        with self._fanout_children_lock:
            merged = _merge_fanout_children(self._fanout_children, disk, _MAX_CONTEXT_PEERS)
            self._fanout_children.clear()
            self._fanout_children.update(merged)
        return len(disk)

    async def _wake_origin_session(self, context_id: str, text: str) -> None:
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


    def _patience_for(self, params: dict, peer: str) -> float:
        _TIMEOUT_CEILING = _ORPHAN_TIMEOUT - _PATIENCE_MARGIN  # 270s
        sender = protocol.extract_sender(params)
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

    def _try_push_reply(self, pending: dict, state: str, reply: str) -> protocol.PushOutcome:
        if state not in (protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED) or not reply:return _failure_outcome("routing","no reply to push",peer=str(pending.get("peer","")),task_id=str(pending.get("task_id","")),context_id=str(pending.get("context_id","")))
        with self._pending_lock:
            if pending.get("pushed"):return protocol.PushOutcome(success=True,category="transport",error="")
            pending["pushed"]=True
        try:
            outcome=self._push_out_of_band(pending["context_id"],reply,want_reply=True)
            if not outcome.success:logger.warning("A2A: out-of-band push for task %s returned failure %s: %s",_bounded_redacted_detail(pending.get("task_id"),128),_bounded_redacted_detail(outcome.category,64),_bounded_redacted_detail(outcome.error,_DETAIL_MAX_CODEPOINTS))
            return outcome
        except Exception as exc:
            b=_bounded_redacted_detail(exc,_DETAIL_MAX_CODEPOINTS)
            logger.warning("A2A: out-of-band push for task %s failed: %s",_bounded_redacted_detail(pending.get("task_id"),128),b)
            return _failure_outcome("transport",b,peer=str(pending.get("peer","")),task_id=str(pending.get("task_id","")),context_id=str(pending.get("context_id","")))

    def _is_duplicate_inbound(self, context_id: str, message_id: str) -> bool:
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

    def _await_reply(self, pending: dict, keepalive=None, patience: Optional[float] = None) -> tuple[str, str, bool, bool]:
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
                return state, reply, pending.get("out_of_band_only", False), False
            except FuturesTimeout:
                now = time.time()
                if now >= deadline:
                    return (
                        protocol.STATE_FAILED, "[agent did not reply in time]",
                        pending.get("out_of_band_only", False), False,
                    )
                if keepalive:
                    try:
                        keepalive()
                    except Exception:
                        self._mark_out_of_band(pending, "[client disconnected]", pop_waiter=True)
                        # Task authority: the client is gone but the agent may
                        # still complete this task.  Do NOT finalize the task as
                        # FAILED here — the late agent reply must finalize the
                        # original task record.  The caller skips
                        # _finalize_task and returns a transient error to the
                        # HTTP client.
                        return (protocol.STATE_FAILED, "[client disconnected]", True, True)
                if now >= patience_deadline:
                    self._mark_out_of_band(pending, "[client patience exceeded]", pop_waiter=False)
                    # Keep waiting: when the reply resolves it must be pushed
                    # directly and the socket write skipped — the client is
                    # gone even though the socket may look alive.
            except Exception:
                return (
                    protocol.STATE_FAILED, "[agent did not reply in time]",
                    pending.get("out_of_band_only", False), False,
                )
    # ── Streaming (SSE) ───────────────────────────────────────────────────
    # ── Task queries ──────────────────────────────────────────────────────
    # ── Push notifications ────────────────────────────────────────────────
    # ── Sending (the agent's reply path) ──────────────────────────────────

    def _durable_complete_pending(self, task_id: str, chat_id: str, content: str, message_id: str) -> tuple[bool, str]:
        # Stage candidate from current durable record — pending map/Future is NOT Task authority (Amendment D)
        rec = self.tasks.get(task_id)
        if rec is None:
            logger.warning("A2A: durable complete for unknown task %s — no authoritative TaskStore record (no fallback, Future unresolved)", task_id)
            return False, "task not found: no authoritative record"
        if rec.get("context_id") != chat_id:
            logger.warning("A2A: context mismatch for task %s: %r != %r", task_id, rec.get("context_id"), chat_id)
            return False, "context mismatch"
        if rec.get("state") in protocol.TERMINAL_STATES:
            # Already terminal — treat as not active for send authority
            return False, "task already terminal"
        candidate = dict(rec)
        candidate["state"] = protocol.STATE_COMPLETED
        candidate["reply"] = content or ""
        candidate["completed_at"] = __import__("time").time()
        try:
            outcome = self.tasks.publish_durable(_task_ledger_path(), task_id, candidate)
        except Exception as exc:
            logger.error("A2A: publish_durable exception for task %s: %s", task_id, exc, exc_info=True)
            return False, "A2A task state could not be durably published"
        if not outcome.published:
            logger.error("A2A: failed to durably publish COMPLETED for task %s: %s", task_id, outcome.error)
            return False, "A2A task state could not be durably published"
        # Publish succeeded — now resolve/remove Future atomically
        with self._pending_lock:
            ent = self._pending.get(task_id)
            if ent is not None and ent[0] == chat_id and not ent[1].done():
                try:
                    ent[1].set_result((protocol.STATE_COMPLETED, content or ""))
                except Exception:
                    pass
                order = self._pending_order.get(chat_id)
                if order is not None:
                    try:
                        order.remove(task_id)
                    except ValueError:
                        pass
                    if not order:
                        self._pending_order.pop(chat_id, None)
                self._pending.pop(task_id, None)
            elif ent is not None:
                # Pending exists but mismatched or already done — do not resolve again
                pass
        # Postcommit push notification (best-effort, only when newly published)
        if outcome.newly_published:
            try:
                self._send_push_notification(task_id, chat_id, content or "", protocol.STATE_COMPLETED)
            except Exception as exc:
                logger.debug("A2A: _durable_complete_pending push failed for %s: %s", task_id, exc)
        return True, ""

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        message_id = str(int(time.time() * 1000))
        # Task-authority: prefer the specific task via thread_id (ContextVar) to avoid cross-talk
        task_id_via_thread = ""
        try:
            from gateway.session_context import get_session_env
            task_id_via_thread = str(get_session_env("HERMES_SESSION_THREAD_ID") or "").strip()
        except Exception:
            task_id_via_thread = ""
        if task_id_via_thread:
            # Exact-thread path: pending or late TaskStore fallback, both disk-first
            with self._pending_lock:
                ent = self._pending.get(task_id_via_thread)
                has_pending = ent is not None and ent[0] == chat_id and not ent[1].done()
            if has_pending:
                ok, err = self._durable_complete_pending(task_id_via_thread, chat_id, content or "", message_id)
                if ok:
                    return SendResult(success=True, message_id=message_id)
                else:
                    return SendResult(success=False, message_id=message_id, error=err)
            # Late completion for disconnected task still WORKING — single durable commit
            try:
                rec_thr = self.tasks.get(task_id_via_thread)
                if rec_thr and rec_thr.get("context_id") == chat_id and rec_thr.get("state") not in protocol.TERMINAL_STATES:
                    logger.info("A2A: late completion for disconnected task %s (context %s) — finalizing original task record (thread_id path)", task_id_via_thread, chat_id)
                    # Use _finalize_task which is disk-first; it raises DurablePublishError on failure
                    try:
                        self._finalize_task({"task_id": task_id_via_thread, "context_id": chat_id, "peer": rec_thr.get("peer", ""), "started": rec_thr.get("created_at", time.time()), "created_iso": rec_thr.get("created_iso", "")}, protocol.STATE_COMPLETED, content or "", audit_direction="push")
                        return SendResult(success=True, message_id=message_id)
                    except protocol.DurablePublishError as dpe:
                        logger.error("A2A: late thread completion durability failed for %s: %s", task_id_via_thread, dpe)
                        return SendResult(success=False, message_id=message_id, error="A2A task state could not be durably published")
            except protocol.DurablePublishError:
                raise
            except Exception:
                pass
        if not task_id_via_thread and reply_to:
            cand = str(reply_to).strip()
            if cand:
                with self._pending_lock:
                    ent2 = self._pending.get(cand)
                    has_pending2 = ent2 is not None and ent2[0] == chat_id and not ent2[1].done()
                if has_pending2:
                    ok2, err2 = self._durable_complete_pending(cand, chat_id, content or "", message_id)
                    if ok2:
                        return SendResult(success=True, message_id=message_id)
                    else:
                        return SendResult(success=False, message_id=message_id, error=err2)
                # TaskStore fallback for reply_to — disk-first via _finalize_task
                try:
                    rec2 = self.tasks.get(cand)
                    if rec2 and rec2.get("context_id") == chat_id and rec2.get("state") not in protocol.TERMINAL_STATES:
                        logger.info("A2A: late completion for disconnected task %s (context %s) — finalizing via reply_to", cand, chat_id)
                        try:
                            self._finalize_task({"task_id": cand, "context_id": chat_id, "peer": rec2.get("peer", ""), "started": rec2.get("created_at", time.time()), "created_iso": rec2.get("created_iso", "")}, protocol.STATE_COMPLETED, content or "", audit_direction="push")
                            return SendResult(success=True, message_id=message_id)
                        except protocol.DurablePublishError as dpe:
                            logger.error("A2A: late reply_to completion durability failed for %s: %s", cand, dpe)
                            return SendResult(success=False, message_id=message_id, error="A2A task state could not be durably published")
                except protocol.DurablePublishError:
                    raise
                except Exception:
                    pass
        if not (metadata or {}).get("notify"):
            logger.debug("A2A: ignoring non-final send for context %s", chat_id)
            return SendResult(success=True, message_id=message_id)
        if task_id_via_thread:
            try:_r=self.tasks.get(task_id_via_thread)
            except:_r=None
            if _r and _r.get("context_id")!=chat_id:task_id_via_thread=""
            elif not _r:
                with self._pending_lock:
                    _e=self._pending.get(task_id_via_thread);_a=any(_c==chat_id and not _f.done() for _,(_c,_f) in self._pending.items()) or any(_rec.get("context_id")==chat_id and _rec.get("state") not in protocol.TERMINAL_STATES for _rec in self.tasks._tasks.values())
                    if (_e and _e[0]!=chat_id) or (not _e and not _a):task_id_via_thread=""
                    else:
                        logger.warning("A2A: thread_id %s not found/active for context %s — failing without fallback",_bounded_redacted_detail(task_id_via_thread,128),_bounded_redacted_detail(chat_id,128))
                        return SendResult(success=False,message_id=message_id,error="task not found for thread_id")
            else:logger.warning("A2A: thread_id %s not found/active for context %s — failing without fallback",_bounded_redacted_detail(task_id_via_thread,128),_bounded_redacted_detail(chat_id,128));return SendResult(success=False,message_id=message_id,error="task not found for thread_id")
        if reply_to and str(reply_to).strip():
            logger.warning("A2A: reply_to %s not found/active for context %s — failing without fallback", reply_to, chat_id)
            return SendResult(success=False, message_id=message_id, error="task not found for reply_to")
        # Context-only selection: count active tasks in this context
        _active_candidates = []
        with self._pending_lock:
            for _tid, (_ctx, _fut) in list(self._pending.items()):
                if _ctx == chat_id and not _fut.done():
                    _active_candidates.append(_tid)
        for _tid, _rec in list(self.tasks._tasks.items()):
            if _rec.get("context_id") == chat_id and _rec.get("state") not in protocol.TERMINAL_STATES and _rec.get("state") != protocol.STATE_SUBMITTED:
                if _tid not in _active_candidates:
                    _active_candidates.append(_tid)
        if len(_active_candidates) == 0:
            pass
        elif len(_active_candidates) == 1:
            _tid = _active_candidates[0]
            with self._pending_lock:
                _ent = self._pending.get(_tid)
                has_pen = _ent is not None and _ent[0] == chat_id and not _ent[1].done()
            if has_pen:
                ok3, err3 = self._durable_complete_pending(_tid, chat_id, content or "", message_id)
                if ok3:
                    return SendResult(success=True, message_id=message_id)
                else:
                    return SendResult(success=False, message_id=message_id, error=err3)
            # TaskStore fallback for the single active task — disk-first
            try:
                _rec = self.tasks.get(_tid)
                if _rec and _rec.get("context_id") == chat_id and _rec.get("state") not in protocol.TERMINAL_STATES:
                    logger.info("A2A: completing single active task %s for context %s via context-only fallback", _tid, chat_id)
                    try:
                        self._finalize_task(
                            {"task_id": _tid, "context_id": chat_id, "peer": _rec.get("peer", ""), "started": _rec.get("created_at", time.time()), "created_iso": _rec.get("created_iso", "")},
                            protocol.STATE_COMPLETED, content or "", audit_direction="push",
                        )
                        return SendResult(success=True, message_id=message_id)
                    except protocol.DurablePublishError as dpe:
                        logger.error("A2A: context fallback durability failed for %s: %s", _tid, dpe)
                        return SendResult(success=False, message_id=message_id, error="A2A task state could not be durably published")
            except protocol.DurablePublishError:
                raise
            except Exception:
                pass
        else:
            logger.warning("A2A: ambiguous task authority for context %s: %d active tasks", chat_id, len(_active_candidates))
            return SendResult(success=False, message_id=message_id, error="ambiguous task authority for context")
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
        #
        # Missing-peer guard: if no peer is registered for this context,
        # the push has no destination and MUST fail explicitly — reporting
        # success here would silently advance the Kanban cursor past a
        # lost event (the reviewer's finding: no-peer false success).
        if not (metadata or {}).get("a2a_push"):
            with self._context_peers_lock:_loop_peer=self._context_peers.get(chat_id,"")
            if _loop_peer and _loopback_fallback_url(_loop_peer,self.host,self.port):
                _o=_failure_outcome("routing","peer identity not resolvable",peer=_loop_peer,task_id=message_id,context_id=chat_id)
                logger.warning("A2A: dropping out-of-band send for %s: loopback peer %r is unresolvable",_bounded_redacted_detail(chat_id,128),_bounded_redacted_detail(_loop_peer,128))
                return _send_result_from_outcome(message_id,_o)
        with self._context_peers_lock:_push_peer=self._context_peers.get(chat_id,"")
        if not _push_peer:
            _o2=_failure_outcome("routing","no peer registered for context",peer="",task_id=message_id,context_id=chat_id)
            logger.warning("A2A: out-of-band send for %s has no registered peer",_bounded_redacted_detail(chat_id,128))
            return _send_result_from_outcome(message_id,_o2)
        try:
            outcome=await asyncio.to_thread(self._push_out_of_band,chat_id,content or "",not (metadata or {}).get("a2a_push"))
            if not outcome.success:return _send_result_from_outcome(message_id,outcome)
        except Exception as exc:
            b=_bounded_redacted_detail(exc,_DETAIL_MAX_CODEPOINTS)
            logger.warning("A2A: out-of-band push for context %s failed: %s",_bounded_redacted_detail(chat_id,128),b)
            try:
                with self._context_peers_lock:_audit_peer=self._context_peers.get(chat_id,"")
            except:_audit_peer=""
            _o3=_failure_outcome("transport",b,peer=_audit_peer,task_id=message_id,context_id=chat_id)
            return _send_result_from_outcome(message_id,_o3)
        return SendResult(success=True,message_id=message_id)
    def _push_out_of_band(self, context_id: str, text: str, want_reply: bool = False) -> protocol.PushOutcome:
        with self._context_peers_lock:peer=self._context_peers.get(context_id,"")
        if not peer:
            logger.debug("A2A: out-of-band send for %s has no known peer; dropping",_bounded_redacted_detail(context_id,128))
            return _failure_outcome("routing","no peer registered for context",peer="",task_id="",context_id=context_id)
        from . import tools as a2a_tools
        entry=a2a_tools._resolve_peer(peer)
        if not entry or not entry.get("url"):
            fallback=_loopback_fallback_url(peer,self.host,self.port)
            if fallback:
                if want_reply:return self._drop_unresolvable_reply(context_id,peer)
                logger.info("A2A: out-of-band send for %s: identity %r not in a2a_agents; falling back to local endpoint %s",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(peer,128),_bounded_redacted_detail(fallback,128))
                return self._push_loopback_in_process(context_id,peer,text,want_reply=False)
            else:return _failure_outcome("routing","registered peer not resolvable",peer=peer,task_id="",context_id=context_id)
        base_url=entry["url"]
        if _is_own_endpoint(base_url,self.host,self.port):
            if want_reply:return self._drop_unresolvable_reply(context_id,peer)
            logger.info("A2A: out-of-band send for %s: resolved peer %r is this gateway (%s); delivering in-process",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(peer,128),_bounded_redacted_detail(base_url,128))
            return self._push_loopback_in_process(context_id,peer,text,want_reply=False)
        headers={**a2a_tools._auth_header(entry.get("auth") or {}),**(entry.get("headers",{}) or {})}
        timeout=int(entry.get("timeout",120))
        allowed=tuple(a2a_tools._allowed_rpc_origins(entry))
        card=None
        try:card=a2a_tools._fetch_card(base_url,headers,min(timeout,30),allowed)
        except:pass
        rpc_url=a2a_tools._rpc_url(base_url,card)
        if not a2a_tools._origin_allowed(rpc_url,entry):
            logger.warning("A2A: peer '%s' card advertised cross-origin RPC URL %s; not in peer's allowed_rpc_origins — using configured origin %s instead",_bounded_redacted_detail(peer,128),_bounded_redacted_detail(rpc_url,300),_bounded_redacted_detail(base_url,300))
            rpc_url=base_url.rstrip("/")
        rpc_body={"jsonrpc":"2.0","id":protocol.new_task_id(),"method":"SendMessage","params":{"message":protocol.text_message(protocol.ROLE_USER,text,context_id=context_id,sender=self._sender_identity())}}
        tenant=a2a_tools._interface_tenant(card,entry)
        if tenant:rpc_body["params"]["tenant"]=tenant
        resp=None
        _push_outcome=protocol.PushOutcome(success=False,category="transport",error="unknown")
        try:
            resp=a2a_tools._http_post_json(rpc_url,rpc_body,headers,timeout,allowed_origins=allowed)
            if isinstance(resp,dict) and "error" in resp:
                _red,_pay=_redacted_jsonrpc_detail(resp["error"])
                logger.warning("A2A: out-of-band push for context %s got JSON-RPC error: %s",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(_red,_DETAIL_MAX_CODEPOINTS))
                _push_outcome=_failure_outcome("jsonrpc",_red,peer=peer,task_id="",context_id=context_id,payload=_pay)
            elif resp is None:
                b=_bounded_redacted_detail("no response",_DETAIL_MAX_CODEPOINTS)
                logger.warning("A2A: out-of-band push for context %s got no response",_bounded_redacted_detail(context_id,128))
                _push_outcome=_failure_outcome("transport",b,peer=peer,task_id="",context_id=context_id)
            elif not isinstance(resp,dict):
                b=_bounded_redacted_detail(resp,_DETAIL_MAX_CODEPOINTS)
                logger.warning("A2A: out-of-band push for context %s got invalid response: %s",_bounded_redacted_detail(context_id,128),b)
                _push_outcome=_failure_outcome("invalid_response",b,peer=peer,task_id="",context_id=context_id)
            else:
                try:
                    _parsed=protocol.parse_send_message_result(resp.get("result"),"V1_WRAPPED")
                    _raw_reply = _parsed.text if isinstance(getattr(_parsed,"text",None),str) else ""
                    _safe_reply = _redacted_reply_text(_raw_reply)
                    _audit_reply = _bounded_redacted_detail(_safe_reply,300)
                    _push_outcome=protocol.PushOutcome(success=True,category="transport",error="",payload=None)
                    _oob_safe_reply = _safe_reply
                    _oob_audit_reply = _audit_reply
                    _oob_has_reply = bool(_safe_reply)
                except protocol.A2AResultValidationError as ve:
                    b=_bounded_redacted_detail(f"{ve.reason}: {ve.detail}",_DETAIL_MAX_CODEPOINTS)
                    logger.warning("A2A: out-of-band push for context %s got malformed/invalid result: %s",_bounded_redacted_detail(context_id,128),b)
                    _push_outcome=_failure_outcome("invalid_response",b,peer=peer,task_id="",context_id=context_id)
        except Exception as exc:
            b=_bounded_redacted_detail(exc,_DETAIL_MAX_CODEPOINTS)
            logger.warning("A2A: out-of-band push for context %s failed: %s",_bounded_redacted_detail(context_id,128),b)
            if _push_outcome.success or _push_outcome.error=="unknown":_push_outcome=_failure_outcome("transport",b,peer=peer,task_id="",context_id=context_id)
        finally:
            if _push_outcome.success:
                try:
                    try:
                        safe_reply = locals().get("_oob_safe_reply", "")
                        audit_reply = locals().get("_oob_audit_reply", "")
                        has_reply = locals().get("_oob_has_reply", False)
                        if has_reply and safe_reply:
                            try:protocol.persist_message(context_id,"agent",safe_reply)
                            except Exception as e2:logger.warning("A2A: post-commit persist failed for %s: %s",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(e2,_DETAIL_MAX_CODEPOINTS))
                            try:_audit_safe("push",peer,"",audit_reply,context_id=context_id)
                            except Exception as e2:logger.warning("A2A: post-commit audit failed for %s: %s",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(e2,_DETAIL_MAX_CODEPOINTS))
                            try:
                                lr=self._push_loopback_in_process(context_id,peer,safe_reply,want_reply=True)
                                if not lr.success:logger.warning("A2A: surfaced loopback failed for %s: %s",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(lr.error,_DETAIL_MAX_CODEPOINTS))
                            except Exception as e2:logger.warning("A2A: surfaced loopback exception for %s: %s",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(e2,_DETAIL_MAX_CODEPOINTS))
                    except Exception as e2:logger.warning("A2A: post-commit extraction failed for %s: %s",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(e2,_DETAIL_MAX_CODEPOINTS))
                except:pass
        return _push_outcome
    def _drop_unresolvable_reply(self, context_id: str, peer: str) -> protocol.PushOutcome:
        _o=_failure_outcome("routing","peer identity not resolvable",peer=peer,task_id="",context_id=context_id)
        logger.warning("A2A: out-of-band REPLY for context %s dropped: peer identity %r is not resolvable",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(peer,128))
        return _o

    def _push_reply_after_client_gone(self, req_id: Any, result: Optional[dict], is_v1: bool = True) -> protocol.PushOutcome:
        try:
            inner=(result or {}).get("result")
            _mode="V1_WRAPPED" if is_v1 else "LEGACY_BARE"
            try:_parsed=protocol.parse_send_message_result(inner,_mode)
            except protocol.A2AResultValidationError as ve:
                b=_bounded_redacted_detail(f"{ve.reason}: {ve.detail}",_DETAIL_MAX_CODEPOINTS)
                logger.warning("A2A: rescue found invalid result for req %s: %s",_bounded_redacted_detail(req_id,128),b)
                return _failure_outcome("invalid_response",b,peer="",task_id=str(req_id),context_id="")
            if _parsed.kind=="task":context_id=_parsed.context_id;state=_parsed.state;reply=_parsed.text
            else:return _failure_outcome("routing","message result not pushable via rescue",peer="",task_id=str(req_id),context_id="")
            if not context_id or state not in (protocol.STATE_COMPLETED,protocol.STATE_INPUT_REQUIRED):return _failure_outcome("routing",f"state not pushable: {state!r}",peer="",task_id=str(req_id),context_id=context_id or "")
            if not reply:return _failure_outcome("routing","no reply to push",peer="",task_id=str(req_id),context_id=context_id)
            outcome=self._push_out_of_band(context_id,reply,want_reply=True)
            if not outcome.success:logger.warning("A2A: rescue push for context %s failed — reply not delivered (want_reply=True): %s",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(outcome.error,_DETAIL_MAX_CODEPOINTS));return outcome
            logger.info("A2A: client disconnected before response write; pushed reply for context %s out-of-band",_bounded_redacted_detail(context_id,128))
            return outcome
        except Exception as exc:
            b=_bounded_redacted_detail(exc,_DETAIL_MAX_CODEPOINTS)
            logger.warning("A2A: could not push reply after client disconnect (req %s): %s",_bounded_redacted_detail(req_id,128),b)
            return _failure_outcome("transport",b,peer="",task_id=str(req_id),context_id="")

    def _push_loopback_in_process(self, context_id: str, peer: str, text: str,want_reply: bool = False) -> protocol.PushOutcome:
        safe_text = _redacted_reply_text(text)
        audit_text = _bounded_redacted_detail(safe_text,300)
        params={"message":protocol.text_message(protocol.ROLE_USER,safe_text,context_id=context_id,sender=self._sender_identity())}
        try:terminal,pending=self._prepare_task(params,peer)
        except protocol.DurablePublishError as dpe:
            b=_bounded_redacted_detail(dpe,_DETAIL_MAX_CODEPOINTS)
            logger.error("A2A: loopback WORKING publish failed for context %s: %s",_bounded_redacted_detail(context_id,128),b)
            return _failure_outcome("durability",f"durability failure: {b}",peer=peer,task_id=getattr(dpe,"task_id","") or "",context_id=context_id)
        except Exception as exc:
            b=_bounded_redacted_detail(exc,_DETAIL_MAX_CODEPOINTS)
            logger.error("A2A: loopback _prepare_task exception for context %s: %s",_bounded_redacted_detail(context_id,128),b)
            return _failure_outcome("transport",b,peer=peer,task_id="",context_id=context_id)
        if terminal is not None:
            state=(terminal.get("status") or {}).get("state","unknown")
            logger.warning("A2A: loopback push for context %s rejected (%s)",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(state,128))
            return _failure_outcome("routing",f"rejected: {state}",peer=peer,task_id=str(terminal.get("id","")) if isinstance(terminal,dict) else "",context_id=context_id)
        assert pending is not None
        if want_reply:
            try:protocol.persist_message(context_id,"agent",safe_text)
            except Exception as exc:logger.warning("A2A: loopback want_reply persist failed for %s: %s",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(exc,_DETAIL_MAX_CODEPOINTS))
            try:_audit_safe("push",peer,pending["task_id"],audit_text,context_id=context_id)
            except Exception as exc:logger.warning("A2A: loopback want_reply audit failed for %s: %s",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(exc,_DETAIL_MAX_CODEPOINTS))
            logger.info("A2A: pushed out-of-band reply for context %s to peer %s (want_reply)",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(peer,128))
            return protocol.PushOutcome(success=True,category="transport",error="")
        else:
            try:
                self._finalize_task(pending,protocol.STATE_COMPLETED,safe_text,audit_direction="push")
                logger.info("A2A: delivered fire-and-forget loopback for context %s (task %s completed)",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(pending["task_id"],128))
                return protocol.PushOutcome(success=True,category="transport",error="")
            except protocol.DurablePublishError as dpe:
                b=_bounded_redacted_detail(dpe,_DETAIL_MAX_CODEPOINTS)
                logger.error("A2A: loopback COMPLETED publish failed for context %s task %s: %s",_bounded_redacted_detail(context_id,128),_bounded_redacted_detail(pending.get("task_id",""),128),b)
                return _failure_outcome("durability",f"durability failure: {b}",peer=peer,task_id=pending.get("task_id","") if isinstance(pending,dict) else "",context_id=context_id)
            except Exception as exc:
                b=_bounded_redacted_detail(exc,_DETAIL_MAX_CODEPOINTS)
                logger.error("A2A: loopback finalize exception for context %s: %s",_bounded_redacted_detail(context_id,128),b)
                return _failure_outcome("transport",b,peer=peer,task_id=pending.get("task_id","") if isinstance(pending,dict) else "",context_id=context_id)


    async def on_processing_complete(self, event, outcome):
        # Delegate to TaskRPCHandler's implementation (which handles deferred failure/cancel persistence)
        # This wrapper ensures TaskRPCHandler takes precedence over BasePlatformAdapter's no-op hook
        # despite MRO BasePlatformAdapter -> TaskRPCHandler.
        from .task_routing import TaskRPCHandler as _TRH
        return await _TRH.on_processing_complete(self, event, outcome)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": f"a2a:{chat_id}", "type": "dm"}
