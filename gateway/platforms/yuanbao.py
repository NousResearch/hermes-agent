"""Yuanbao platform adapter: WebSocket gateway client (AUTH_BIND, ping/pong heartbeat, reconnect),
inbound middleware pipeline (T05 push → MessageEvent) and outbound sender (T06 text/media).

Config under ``platforms.yuanbao.extra`` (or env): app_id/YUANBAO_APP_ID, app_secret/YUANBAO_APP_SECRET,
bot_id/YUANBAO_BOT_ID (optional, returned by sign-token), ws_url/YUANBAO_WS_URL, api_domain/YUANBAO_API_DOMAIN.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import collections
import contextlib
import dataclasses
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sys
import time
import urllib.parse
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Optional, Tuple

import httpx

try:
    import websockets
    import websockets.exceptions
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from agent.i18n import t
from gateway.platforms.base import (
    BasePlatformAdapter, SendResult,
    cache_document_from_bytes_async, cache_image_from_bytes_async, cache_video_from_bytes_async,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms import helpers as _mdchunk
from gateway.platforms._shared import get_scoped_secret as _yb_secret, profile_scoped as _profile_scoped
from gateway.platforms.helpers import MessageDeduplicator, cancel_task
from gateway.platforms.access_policy_mixin import OwnAccessPolicyMixin
from gateway.platforms.yuanbao_media import (
    download_url as media_download_url, get_cos_credentials, upload_to_cos,
    build_image_msg_body, build_file_msg_body, guess_mime_type, md5_hex,
)
from gateway.platforms.yuanbao_proto import (
    CMD_TYPE, WS_HEARTBEAT_RUNNING, WS_HEARTBEAT_FINISH, HERMES_INSTANCE_ID,
    _fields_to_dict, _get_string, _get_varint, _parse_fields,
    decode_conn_msg, decode_inbound_push, decode_forward_msg_data,
    decode_query_group_info_rsp, decode_get_group_member_list_rsp,
    encode_auth_bind, encode_ping, encode_push_ack, encode_send_c2c_message, encode_send_group_message,
    encode_send_private_heartbeat, encode_send_group_heartbeat, encode_query_group_info,
    encode_get_group_member_list, next_seq_no,
)
from gateway.session_transcript import TranscriptReadError

logger = logging.getLogger(__name__)

# AUTH_BIND / sign-token header values
from hermes_cli.version_info import get_version_info

_APP_VERSION = _BOT_VERSION = get_version_info().base_version
_YUANBAO_INSTANCE_ID = str(HERMES_INSTANCE_ID)
_OPERATION_SYSTEM = sys.platform

DEFAULT_WS_GATEWAY_URL = "wss://bot-wss.yuanbao.tencent.com/wss/connection"
DEFAULT_API_DOMAIN = "https://bot.yuanbao.tencent.com"
HEARTBEAT_INTERVAL_SECONDS = 30.0
CONNECT_TIMEOUT_SECONDS = 15.0
AUTH_TIMEOUT_SECONDS = 10.0
MAX_RECONNECT_ATTEMPTS = 100
DEFAULT_SEND_TIMEOUT = 30.0  # WS biz request timeout
# Caps the WS close handshake: websockets' own 5s close_timeout waits for a close echo an idle
# server never sends, stalling shutdown; a responsive server finishes well under 1s.
# Upper bound on the WS close handshake during teardown (#40383). The websockets connection's own
# close_timeout (5s) blocks until the server echoes the close frame; an idle/unresponsive server never
# replies, stalling gateway shutdown by the full timeout. Bounding the close await here keeps teardown fast
# — a responsive server completes the handshake in well under a second, so this only caps the pathological
# hang. Also bounds the reconnect / connect-failure cleanup paths that reuse _cleanup_ws(), where a graceful
# close is unnecessary anyway (the socket is being discarded to redial).
WS_CLOSE_TIMEOUT_S = 1.0
NO_RECONNECT_CLOSE_CODES = {4012, 4013, 4014, 4018, 4019, 4021}  # permanent errors — never reconnect
HEARTBEAT_TIMEOUT_THRESHOLD = 2  # consecutive missed pongs before reconnect
REPLY_HEARTBEAT_INTERVAL_S = 2.0   # RUNNING cadence
REPLY_HEARTBEAT_TIMEOUT_S = 30.0   # auto-FINISH after this much inactivity
SLOW_RESPONSE_TIMEOUT_S = 120.0  # push slow_response_message() when the agent is silent this long


def slow_response_message() -> str:
    """Patience notice pushed after SLOW_RESPONSE_TIMEOUT_S of agent silence (localized; the
    Chinese wording Yuanbao users historically saw lives in locales/zh.yaml)."""
    return t("platform.yuanbao.slow_response_notice")


# Cron wrapper markers, mirrored from cron/scheduler_delivery.py (``wrap_response``). The wrapper
# is not keyed yet; when it is, import the same key here so ``strip_cron_wrapper`` keeps matching.
CRON_WRAPPER_HEADER_PREFIX = "Cronjob Response: "
CRON_WRAPPER_DIVIDER = "\n-------------\n\n"
CRON_WRAPPER_FOOTER_PREFIX = '\n\nTo stop or manage this job, send me a new message (e.g. "stop reminder '

# Transcript anchors: [image|ybres:abc]  [file:report.pdf|ybres:xyz]  [voice|ybres:…]
_YB_RES_REF_RE = re.compile(r"\[(image|voice|video|file(?::[^|\]]*)?)\|ybres:([A-Za-z0-9_\-]+)\]")
# Anchors after local download: [image: /path]  [file: report.pdf → /path]  [video: /path]
_YB_LOCAL_MEDIA_RE = re.compile(r"\[(\w+):[^\]]*?(/[^\]]+?)\s*\]")
_RESOLVABLE_MEDIA_KINDS = frozenset({"image", "file", "video"})  # kinds injected into model context
_INDICATOR_RE = re.compile(r'\s*\(\d+/\d+\)$')  # "(1/3)" page indicators from BasePlatformAdapter
_TEXT_ELEM_TYPE = "TIMTextElem"

OBSERVED_MEDIA_BACKFILL_LOOKBACK = 50  # recent transcript messages scanned for observed media
OBSERVED_MEDIA_BACKFILL_MAX_RESOLVE_PER_TURN = 12
# platforms.yuanbao.extra.media_resolve_concurrency: 1 = sequential rollback knob;
# 6 = browser per-origin HTTP/1.1 ceiling; 12 = backfill cap.
_DEFAULT_RESOLVE_CONCURRENCY = 6
_MIN_RESOLVE_CONCURRENCY = 1
_MAX_RESOLVE_CONCURRENCY = 12


def _iter_ybres_refs(matches) -> Iterator[Tuple[str, str, str]]:
    """Turn ``_YB_RES_REF_RE`` matches into ``(rid, kind, filename)`` for resolvable kinds only."""
    for m in matches:
        kind, _, filename = m.group(1).partition(":")
        if kind.strip() in _RESOLVABLE_MEDIA_KINDS:
            yield m.group(2), kind.strip(), filename.strip()


def _text_elem(text: str) -> dict:
    """A TIMTextElem msg_body entry."""
    return {"msg_type": _TEXT_ELEM_TYPE, "msg_content": {"text": text}}


def _cancel_all(tasks: Dict[str, asyncio.Task]) -> None:
    """Cancel every unfinished task in *tasks* and clear the dict."""
    for task in list(tasks.values()):
        if not task.done():
            task.cancel()
    tasks.clear()


class MarkdownProcessor:
    """Yuanbao's fence/table-aware chunking policy over the shared chunker in gateway.platforms.helpers."""
    @classmethod
    def chunk_markdown_text(cls, text: str, max_chars: int = 4000, len_fn: Optional[Callable[[str], int]] = None) -> list[str]:
        """<= max_chars chunks at paragraph boundaries, never inside a fence or table (an oversized
        single block may exceed the limit)."""
        return _mdchunk.split_text_fence_aware(text, max_chars, len_fn, prefer_paragraphs=True, balance_fences=False)


class SignManager:
    """Sign-token acquisition, caching, signing and retry. All state is class-level so one
    shared client serves the whole process."""
    TOKEN_PATH = "/api/v5/robotLogic/sign-token"
    RETRYABLE_CODE = 10099
    MAX_RETRIES = 3
    RETRY_DELAY_S = 1.0
    CACHE_REFRESH_MARGIN_S = 60  # treat as expiring this many seconds early
    HTTP_TIMEOUT_S = 10.0
    _cache: dict[str, dict[str, Any]] = {}  # app_key → {"token", "bot_id", "expire_ts", ...}
    # Per-app_key refresh locks, created lazily from async context so they bind to the running
    # loop; disconnect() clears them to avoid stale locks across reconnects.
    _locks: dict[str, asyncio.Lock] = {}

    @classmethod
    def get_refresh_lock(cls, app_key: str) -> asyncio.Lock:
        """Per-app_key refresh lock (create on demand). Call only from a running event loop."""
        if app_key not in cls._locks:
            cls._locks[app_key] = asyncio.Lock()
        return cls._locks[app_key]

    @staticmethod
    def compute_signature(nonce: str, timestamp: str, app_key: str, app_secret: str) -> str:
        """HMAC-SHA256(key=app_secret, msg=nonce+timestamp+app_key+app_secret).hexdigest()."""
        plain = nonce + timestamp + app_key + app_secret
        return hmac.new(app_secret.encode(), plain.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def build_timestamp() -> str:
        """Beijing-time ISO-8601 timestamp without milliseconds (2006-01-02T15:04:05+08:00)."""
        return datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S+08:00")

    @classmethod
    def is_cache_valid(cls, entry: dict[str, Any]) -> bool:
        return entry["expire_ts"] - time.time() > cls.CACHE_REFRESH_MARGIN_S

    @classmethod
    def clear_locks(cls) -> None:
        cls._locks.clear()

    @classmethod
    def purge_expired(cls) -> int:
        """Drop expired token-cache entries; returns count purged."""
        now = time.time()
        expired_keys = [k for k, v in cls._cache.items() if now - v.get("expire_ts", 0) > 0]
        for k in expired_keys:
            cls._cache.pop(k, None)
        return len(expired_keys)

    @classmethod
    async def fetch(cls, app_key: str, app_secret: str, api_domain: str, route_env: str = "") -> dict[str, Any]:
        """POST sign-token, retrying RETRYABLE_CODE up to MAX_RETRIES times."""
        url = f"{api_domain.rstrip('/')}{cls.TOKEN_PATH}"
        async with httpx.AsyncClient(timeout=cls.HTTP_TIMEOUT_S) as client:
            for attempt in range(cls.MAX_RETRIES + 1):
                nonce = secrets.token_hex(16)
                timestamp = cls.build_timestamp()
                payload = {"app_key": app_key, "nonce": nonce,
                           "signature": cls.compute_signature(nonce, timestamp, app_key, app_secret), "timestamp": timestamp}
                headers = {"Content-Type": "application/json", "X-AppVersion": _APP_VERSION, "X-OperationSystem": _OPERATION_SYSTEM,
                           "X-Instance-Id": _YUANBAO_INSTANCE_ID, "X-Bot-Version": _BOT_VERSION}
                if route_env:
                    headers["X-Route-Env"] = route_env
                logger.info("Sign token request: url=%s%s", url, f" (retry {attempt}/{cls.MAX_RETRIES})" if attempt > 0 else "")
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    raise RuntimeError(f"Sign token API returned {response.status_code}: {response.text[:200]}")
                try:
                    result_data: dict[str, Any] = response.json()
                except Exception as exc:
                    raise ValueError(f"Sign token response parse error: {exc}") from exc
                code = result_data.get("code")
                if code == 0:
                    data = result_data.get("data")
                    if not isinstance(data, dict):
                        raise ValueError(f"Sign token response missing 'data' field: {result_data}")
                    logger.info("Sign token success: bot_id=%s", data.get("bot_id"))
                    return data
                if code != cls.RETRYABLE_CODE or attempt >= cls.MAX_RETRIES:
                    raise RuntimeError(f"Sign token error: code={code}, msg={result_data.get('msg', '')}")
                logger.warning("Sign token retryable: code=%s, retrying in %ss (attempt=%d/%d)",
                               code, cls.RETRY_DELAY_S, attempt + 1, cls.MAX_RETRIES)
                await asyncio.sleep(cls.RETRY_DELAY_S)
        raise RuntimeError("Sign token failed: max retries exceeded")

    @classmethod
    async def _fetch_into_cache(cls, app_key: str, app_secret: str, api_domain: str, route_env: str) -> None:
        data = await cls.fetch(app_key, app_secret, api_domain, route_env)
        duration: int = data.get("duration", 0)
        cls._cache[app_key] = {
            "token": data.get("token", ""), "bot_id": data.get("bot_id", ""), "duration": duration,
            "product": data.get("product", ""), "source": data.get("source", ""),
            "expire_ts": time.time() + (duration if duration > 0 else 3600),
        }

    @classmethod
    async def get_token(cls, app_key: str, app_secret: str, api_domain: str, route_env: str = "") -> dict[str, Any]:
        """WS auth token, served from cache while valid (with CACHE_REFRESH_MARGIN_S)."""
        cls.purge_expired()
        cached = cls._cache.get(app_key)
        if cached and cls.is_cache_valid(cached):
            logger.info("Using cached token (%ds remaining)", int(cached["expire_ts"] - time.time()))
            return dict(cached)
        async with cls.get_refresh_lock(app_key):
            cached = cls._cache.get(app_key)
            if cached and cls.is_cache_valid(cached):
                return dict(cached)
            await cls._fetch_into_cache(app_key, app_secret, api_domain, route_env)
        return dict(cls._cache[app_key])

    @classmethod
    async def force_refresh(cls, app_key: str, app_secret: str, api_domain: str, route_env: str = "") -> dict[str, Any]:
        """Clear the cached token and re-sign."""
        logger.warning("[force-refresh] Clearing cache and re-signing token: app_key=****%s", app_key[-4:])
        async with cls.get_refresh_lock(app_key):
            cls._cache.pop(app_key, None)
            await cls._fetch_into_cache(app_key, app_secret, api_domain, route_env)
        return dict(cls._cache[app_key])


from gateway.platforms.yuanbao_inbound import (
    AccessPolicy,
    DecodeMiddleware,
    InboundContext,
    InboundPipeline,
    InboundPipelineBuilder,
)



class ConnectionManager:
    """WebSocket lifecycle: open/close, AUTH_BIND, ping/pong heartbeat, receive loop, backoff reconnect."""
    _DEBOUNCE_WINDOW: float = 1.5  # seconds to wait for companion frames of a multi-part message
    _LOOPS = (("_heartbeat_task", "_heartbeat_loop", "heartbeat"), ("_recv_task", "_receive_loop", "recv"))

    def __init__(self, adapter: "YuanbaoAdapter") -> None:
        self._adapter = adapter
        self._ws = None  # websockets connection
        self._connect_id: Optional[str] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._pending_acks: Dict[str, asyncio.Future] = {}
        self._pending_pong: Optional[asyncio.Future] = None
        self._consecutive_hb_timeouts = self._reconnect_attempts = 0
        self._reconnecting: bool = False
        # Debounce buffer aggregating multi-part inbound messages: sender key -> frames / timer
        self._inbound_buffer: Dict[str, list] = {}
        self._inbound_timers: Dict[str, asyncio.TimerHandle] = {}

    @property
    def ws(self):
        return self._ws

    @property
    def is_connected(self) -> bool:
        """``ws.open`` may be a bool (websockets <14) or a method (>=14)."""
        if self._ws is None:
            return False
        open_attr = getattr(self._ws, "open", None)
        try:
            return open_attr is True or (callable(open_attr) and bool(open_attr()))
        except Exception:
            return False

    async def open(self) -> bool:
        """sign-token → WS connect → AUTH_BIND → start loops. Returns True on success."""
        adapter = self._adapter
        if not WEBSOCKETS_AVAILABLE:
            msg = "Yuanbao startup failed: 'websockets' package not installed"
            adapter._set_fatal_error("yuanbao_missing_dependency", msg, retryable=True)
            logger.warning("[%s] %s. Run: hermes pm repair", adapter.name, msg)
            return False
        if not adapter._app_key or not adapter._app_secret:
            msg = "Yuanbao startup failed: YUANBAO_APP_ID and YUANBAO_APP_SECRET are required"
            adapter._set_fatal_error("yuanbao_missing_credentials", msg, retryable=False)
            logger.error("[%s] %s", adapter.name, msg)
            return False
        if self.is_connected:
            logger.debug("[%s] Already connected, skipping connect()", adapter.name)
            return True
        if not adapter._acquire_platform_lock('yuanbao-app-key', adapter._app_key, 'Yuanbao app key'):
            return False
        try:
            logger.info("[%s] Fetching sign token from %s", adapter.name, adapter._api_domain)
            token_data = await adapter._get_cached_token()
            logger.info("[%s] Connecting to %s", adapter.name, adapter._ws_url)
            if not await self._dial(token_data):
                return False
            adapter._loop = asyncio.get_running_loop()
            self._connected(cancel_existing=False)
            logger.info("[%s] Connected. connectId=%s botId=%s", adapter.name, self._connect_id, adapter._bot_id)
            return True
        except asyncio.TimeoutError:
            logger.error("[%s] Connection timed out", adapter.name)
        except Exception as exc:
            logger.error("[%s] connect() failed: %s", adapter.name, exc, exc_info=True)
        await self._cleanup_ws()
        adapter._release_platform_lock()
        return False

    async def _dial(self, token_data: dict) -> bool:
        """Adopt the sign-token bot_id, open the WS (built-in ping/pong disabled) and run AUTH_BIND;
        cleans up on auth failure."""
        if token_data.get("bot_id"):
            self._adapter._bot_id = str(token_data["bot_id"])
        self._ws = await asyncio.wait_for(
            websockets.connect(  # type: ignore[attr-defined]
                self._adapter._ws_url, ping_interval=None, ping_timeout=None, close_timeout=5,
                happy_eyeballs_delay=0.25,  # race IPv6/IPv4 in loop.create_connection (#114265)
            ),
            timeout=CONNECT_TIMEOUT_SECONDS,
        )
        if not await self._authenticate(token_data):
            await self._cleanup_ws()
            return False
        return True

    def _connected(self, *, cancel_existing: bool) -> None:
        """Post-AUTH bookkeeping shared by open() and reconnect: mark connected, (re)start loops,
        register as the active adapter."""
        self._reconnect_attempts = 0
        self._adapter._mark_connected()
        self._start_loops(cancel_existing=cancel_existing)
        YuanbaoAdapter.set_active(self._adapter)

    def _start_loops(self, *, cancel_existing: bool) -> None:
        """(Re)start the heartbeat and receive loops for the current connect_id."""
        for attr, coro_name, tag in self._LOOPS:
            old = getattr(self, attr)
            if cancel_existing and old and not old.done():
                old.cancel()
            setattr(self, attr, asyncio.create_task(getattr(self, coro_name)(), name=f"yuanbao-{tag}-{self._connect_id}"))

    async def close(self) -> None:
        """Cancel background tasks, fail pending futures, and close the WebSocket."""
        for attr, _coro_name, _tag in self._LOOPS:
            task = getattr(self, attr)
            if task:
                await cancel_task(task)
                setattr(self, attr, None)
        disc_exc = RuntimeError("YuanbaoAdapter disconnected")
        for fut in self._pending_acks.values():
            if not fut.done():
                fut.set_exception(disc_exc)
        self._pending_acks.clear()
        SignManager.clear_locks()  # avoid stale locks bound to a previous event loop
        await self._cleanup_ws()

    async def _authenticate(self, token_data: dict) -> bool:
        """Send AUTH_BIND and read frames until BIND_ACK; False on failure/timeout."""
        adapter = self._adapter
        if self._ws is None:
            return False
        uid = adapter._bot_id or token_data.get("bot_id", "")
        msg_id = str(uuid.uuid4())
        await self._ws.send(encode_auth_bind(
            biz_id="ybBot", uid=uid, source=token_data.get("source") or "bot", token=token_data.get("token", ""),
            msg_id=msg_id, app_version=_APP_VERSION, operation_system=_OPERATION_SYSTEM, bot_version=_BOT_VERSION,
            route_env=adapter._route_env or token_data.get("route_env", "") or "",
        ))
        logger.debug("[%s] AUTH_BIND sent (msg_id=%s uid=%s)", adapter.name, msg_id, uid)
        try:
            deadline = asyncio.get_running_loop().time() + AUTH_TIMEOUT_SECONDS
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    logger.error("[%s] AUTH_BIND timeout waiting for BIND_ACK", adapter.name)
                    return False
                raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    msg = decode_conn_msg(bytes(raw))
                except Exception:
                    continue
                head = msg.get("head", {})
                if head.get("cmd_type", -1) != CMD_TYPE["Response"] or head.get("cmd", "") != "auth-bind":
                    continue
                self._connect_id = self._extract_connect_id(msg)
                if not self._connect_id:
                    logger.error("[%s] BIND_ACK missing connectId", adapter.name)
                    return False
                logger.info("[%s] BIND_ACK received: connectId=%s", adapter.name, self._connect_id)
                return True
        except asyncio.TimeoutError:
            logger.error("[%s] AUTH_BIND timeout", adapter.name)
        except Exception as exc:
            logger.error("[%s] AUTH_BIND error: %s", adapter.name, exc, exc_info=True)
        return False

    def _pop_pending(self, msg_id: str) -> Optional[asyncio.Future]:
        """Pop the not-yet-done future registered for *msg_id*, if any."""
        fut = self._pending_acks.pop(msg_id, None) if msg_id else None
        return fut if fut is not None and not fut.done() else None

    def _extract_connect_id(self, decoded_msg: dict) -> Optional[str]:
        """connectId from a decoded BIND_ACK, or None (logs AuthBindRsp errors)."""
        data: bytes = decoded_msg.get("data", b"")
        if not data:
            return None
        try:
            fdict = _fields_to_dict(_parse_fields(data))
            code = _get_varint(fdict, 1)
            if code != 0:
                logger.error("[%s] AuthBindRsp error: code=%d message=%r", self._adapter.name, code, _get_string(fdict, 2))
                return None
            return _get_string(fdict, 3) or None
        except Exception as exc:
            logger.warning("[%s] Failed to extract connectId: %s", self._adapter.name, exc)
            return None

    async def _heartbeat_loop(self) -> None:
        """Send PING every HEARTBEAT_INTERVAL_SECONDS; reconnect after HEARTBEAT_TIMEOUT_THRESHOLD misses."""
        adapter = self._adapter
        try:
            while adapter._running:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
                if self._ws is None:
                    continue
                try:
                    msg_id = str(uuid.uuid4())
                    self._pending_pong = pong_future = asyncio.get_running_loop().create_future()
                    self._pending_acks[msg_id] = pong_future
                    await self._ws.send(encode_ping(msg_id))
                    logger.debug("[%s] PING sent (msg_id=%s)", adapter.name, msg_id)
                    try:
                        await asyncio.wait_for(pong_future, timeout=10.0)
                        self._consecutive_hb_timeouts = 0
                    except asyncio.TimeoutError:
                        self._pending_acks.pop(msg_id, None)
                        self._consecutive_hb_timeouts += 1
                        logger.warning("[%s] PONG timeout (%d/%d)", adapter.name, self._consecutive_hb_timeouts, HEARTBEAT_TIMEOUT_THRESHOLD)
                        if self._consecutive_hb_timeouts >= HEARTBEAT_TIMEOUT_THRESHOLD:
                            logger.warning("[%s] Heartbeat threshold exceeded, triggering reconnect", adapter.name)
                            self.schedule_reconnect()
                            return
                    finally:
                        self._pending_acks.pop(msg_id, None)
                        self._pending_pong = None
                except Exception as exc:
                    logger.debug("[%s] Heartbeat send failed: %s", adapter.name, exc)
        except asyncio.CancelledError:
            pass

    async def _receive_loop(self) -> None:
        """Read WS frames and dispatch by cmd_type; schedule reconnect unless the close code is permanent."""
        adapter = self._adapter
        try:
            async for raw in self._ws:  # type: ignore[union-attr]
                if isinstance(raw, (bytes, bytearray)):
                    await self._handle_frame(bytes(raw))
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed as close_exc:  # type: ignore[union-attr]
            close_code = getattr(close_exc, 'code', None)
            logger.warning("[%s] WebSocket connection closed: code=%s reason=%s", adapter.name, close_code, getattr(close_exc, 'reason', ''))
            if close_code and close_code in NO_RECONNECT_CLOSE_CODES:
                logger.error("[%s] Close code %d is non-recoverable, NOT reconnecting", adapter.name, close_code)
                adapter._mark_disconnected()
            else:
                self.schedule_reconnect()
        except Exception as exc:
            logger.warning("[%s] receive_loop exited: %s", adapter.name, exc)
            self.schedule_reconnect()

    async def _handle_frame(self, raw: bytes) -> None:
        adapter = self._adapter
        try:
            msg = decode_conn_msg(raw)
        except Exception as exc:
            logger.debug("[%s] Failed to decode frame: %s", adapter.name, exc)
            return
        head = msg.get("head", {})
        cmd_type = head.get("cmd_type", -1)
        cmd = head.get("cmd", "")
        msg_id = head.get("msg_id", "")
        data: bytes = msg.get("data", b"")
        if cmd_type == CMD_TYPE["Response"]:
            if cmd == "ping":  # HEARTBEAT_ACK
                logger.debug("[%s] HEARTBEAT_ACK received (msg_id=%s)", adapter.name, msg_id)
                pong = self._pending_pong if self._pending_pong is not None and not self._pending_pong.done() else self._pop_pending(msg_id)
                if pong is not None:
                    pong.set_result(True)
            elif cmd in {"send_group_heartbeat", "send_private_heartbeat"}:
                # Fire-and-forget heartbeat ACKs: nobody awaits them; discard to avoid "Unmatched" noise.
                logger.debug("[%s] Heartbeat ACK received: cmd=%s msg_id=%s", adapter.name, cmd, msg_id)
            elif msg_id and msg_id in self._pending_acks:  # response to an outbound RPC
                fut = self._pop_pending(msg_id)
                if fut is not None:
                    result = {"head": head}
                    if data:
                        result["data"] = data
                    fut.set_result(result)
            else:
                logger.debug("[%s] Unmatched Response: cmd=%s msg_id=%s", adapter.name, cmd, msg_id)
            return
        if cmd_type == CMD_TYPE["Push"]:
            logger.info("[%s] Push received: cmd=%s msg_id=%s data_len=%d", adapter.name, cmd, msg_id, len(data))
            if head.get("need_ack", False) and self._ws is not None:
                try:
                    await self._ws.send(encode_push_ack(head))
                except Exception as ack_exc:
                    logger.debug("[%s] Failed to send PushAck: %s", adapter.name, ack_exc)
            if msg_id and msg_id in self._pending_acks:
                fut = self._pop_pending(msg_id)
                if fut is not None:
                    try:
                        fut.set_result(decode_inbound_push(data) if data else {"head": head})
                    except Exception as exc:
                        fut.set_exception(exc)
                return
            if data:  # genuine inbound message — dispatch to AI
                logger.info("[%s] WS received inbound push, decoding and dispatching: cmd=%s, data_len=%d", adapter.name, cmd, len(data))
                self._push_to_inbound(data)
            return
        logger.debug("[%s] Ignoring frame: cmd_type=%d cmd=%s msg_id=%s", adapter.name, cmd_type, cmd, msg_id)

    def _extract_sender_key(self, raw_data: bytes) -> str:
        """Debounce key 'from_account:group_code' (JSON or protobuf), else a unique fallback."""
        with contextlib.suppress(Exception):
            parsed = json.loads(raw_data.decode("utf-8"))
            if isinstance(parsed, dict):
                from_account, group_code = DecodeMiddleware.json_sender_fields(parsed)
                if from_account:
                    return f"{from_account}:{group_code}"
        with contextlib.suppress(Exception):
            push = decode_inbound_push(raw_data)
            if push:
                return f"{push.get('from_account', '')}:{push.get('group_code', '')}"
        return f"__unknown_{id(raw_data)}"

    def _push_to_inbound(self, raw_data: bytes) -> None:
        """Debounced dispatch: frames from one sender within _DEBOUNCE_WINDOW run as ONE pipeline
        execution, merging multi-part messages (e.g. image + text pushed separately)."""
        key = self._extract_sender_key(raw_data)
        existing_timer = self._inbound_timers.pop(key, None)
        if existing_timer:
            existing_timer.cancel()
        self._inbound_buffer.setdefault(key, []).append(raw_data)
        logger.debug("[%s] Debounce: buffered frame for key=%s, count=%d", self._adapter.name, key, len(self._inbound_buffer[key]))
        self._inbound_timers[key] = asyncio.get_running_loop().call_later(self._DEBOUNCE_WINDOW, self._flush_inbound_buffer, key)

    def _flush_inbound_buffer(self, key: str) -> None:
        """Run the pipeline over the buffered frames for *key*."""
        self._inbound_timers.pop(key, None)
        data_list = self._inbound_buffer.pop(key, [])
        if not data_list:
            return
        adapter = self._adapter
        logger.info("[%s] Debounce flush: key=%s, aggregated %d frames", adapter.name, key, len(data_list))
        adapter._track_task(asyncio.create_task(
            adapter._inbound_pipeline.execute(InboundContext(adapter=adapter, raw_frames=data_list)), name=f"yuanbao-pipeline-{key}"))

    async def send_biz_request(self, encoded_conn_msg: bytes, req_id: str, timeout: float = DEFAULT_SEND_TIMEOUT) -> dict:
        """Send a business request and await its response future (pending_acks[req_id]), cleaning up on exit."""
        if self._ws is None:
            raise RuntimeError("Not connected")
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_acks[req_id] = future
        try:
            await self._ws.send(encoded_conn_msg)
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        finally:
            self._pending_acks.pop(req_id, None)

    def schedule_reconnect(self) -> None:
        """Schedule a reconnect only if running and not already reconnecting."""
        if self._adapter._running and not self._reconnecting:
            asyncio.create_task(self._reconnect_with_backoff())

    async def _reconnect_with_backoff(self) -> bool:
        if self._reconnecting:
            logger.debug("[%s] Reconnect already in progress, skipping", self._adapter.name)
            return False
        self._reconnecting = True
        try:
            return await self._do_reconnect()
        finally:
            self._reconnecting = False

    async def _do_reconnect(self) -> bool:
        """Reconnect loop (under the _reconnecting guard) with exponential backoff 1s, 2s, 4s, … capped at 60s."""
        adapter = self._adapter
        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            self._reconnect_attempts = attempt + 1
            wait = min(2 ** attempt, 60)
            logger.info("[%s] Reconnect attempt %d/%d in %ds", adapter.name, attempt + 1, MAX_RECONNECT_ATTEMPTS, wait)
            await asyncio.sleep(wait)
            await self._cleanup_ws()
            try:
                token_data = await SignManager.force_refresh(
                    adapter._app_key, adapter._app_secret, adapter._api_domain, route_env=adapter._route_env,
                )
                if not await self._dial(token_data):
                    logger.warning("[%s] Re-auth failed on attempt %d", adapter.name, attempt + 1)
                    continue
                self._consecutive_hb_timeouts = 0
                self._connected(cancel_existing=True)
                logger.info("[%s] Reconnected on attempt %d. connectId=%s", adapter.name, attempt + 1, self._connect_id)
                return True
            except asyncio.TimeoutError:
                logger.warning("[%s] Reconnect attempt %d timed out", adapter.name, attempt + 1)
            except Exception as exc:
                logger.warning("[%s] Reconnect attempt %d failed: %s", adapter.name, attempt + 1, exc)
        logger.error("[%s] Giving up after %d reconnect attempts", adapter.name, MAX_RECONNECT_ATTEMPTS)
        adapter._mark_disconnected()
        return False

    async def _cleanup_ws(self) -> None:
        """Close and clear the WS, bounded by WS_CLOSE_TIMEOUT_S so an unresponsive server can't stall teardown."""
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await asyncio.wait_for(ws.close(), timeout=WS_CLOSE_TIMEOUT_S)
            except asyncio.TimeoutError:
                # No close-frame echo in time; websockets force-closes the transport on cancel.
                logger.debug("[%s] WS close handshake exceeded %.1fs — dropping connection", self._adapter.name, WS_CLOSE_TIMEOUT_S)
            except Exception:
                pass


def _read_local_file(adapter, label: str, path: str, default_name: str, default_mime: str,
                     filename: Optional[str] = None) -> Tuple[bytes, str, str]:
    """(bytes, filename, content_type) for a local file; ValueError when missing."""
    if not os.path.isfile(path):
        raise ValueError(f"File not found: {path}")
    logger.info("[%s] %s: reading %s", adapter.name, label, path)
    with open(path, "rb") as f:
        file_bytes = f.read()
    filename = filename or os.path.basename(path) or default_name
    return file_bytes, filename, guess_mime_type(filename) or default_mime


class MediaSendHandler(ABC):
    """Media send strategy: subclasses provide acquire_file() and build_msg_body(); handle() runs
    the shared flow (check ws → cancel notifier → validate → COS upload → lock → dispatch)."""
    def needs_cos_upload(self) -> bool:
        """Override to return False for non-COS media (sticker)."""
        return True

    @abstractmethod
    async def acquire_file(self, adapter: "YuanbaoAdapter", **kwargs: Any) -> Tuple[bytes, str, str]:
        """(file_bytes, filename, content_type); raise ValueError when unobtainable."""

    @abstractmethod
    def build_msg_body(self, upload_result: dict, **kwargs: Any) -> list:
        """Platform-specific MsgBody list from the COS upload result."""

    async def handle(self, adapter: "YuanbaoAdapter", chat_id: str, reply_to: Optional[str] = None,
                     caption: Optional[str] = None, **kwargs: Any) -> "SendResult":
        if adapter._connection.ws is None:
            return SendResult(success=False, error="Not connected", retryable=True)
        adapter._outbound.slow_notifier.cancel(chat_id)
        try:
            file_bytes, filename, content_type = await self.acquire_file(adapter, **kwargs)
            if self.needs_cos_upload():
                # Stickers (TIMFaceElem) carry no bytes — validating them would yield "Empty file".
                validation_err = MessageSender.validate_media(file_bytes, filename, adapter.MEDIA_MAX_SIZE_MB)
                if validation_err:
                    return SendResult(success=False, error=validation_err)
                token_data = await adapter._get_cached_token()
                credentials = await get_cos_credentials(
                    app_key=adapter._app_key, api_domain=adapter._api_domain, token=token_data.get("token", ""),
                    filename=filename, bot_id=token_data.get("bot_id", "") or adapter._bot_id or "", route_env=adapter._route_env,
                )
                upload_result = await upload_to_cos(
                    file_bytes=file_bytes, filename=filename, content_type=content_type, credentials=credentials,
                    bucket=credentials["bucketName"], region=credentials["region"],
                )
                # Explicit keys win over caller kwargs (avoids "multiple values" TypeError).
                msg_body = self.build_msg_body(upload_result, **{
                    **kwargs, "file_uuid": md5_hex(file_bytes), "filename": filename, "content_type": content_type})
            else:
                msg_body = self.build_msg_body({}, **kwargs)
            if caption:
                msg_body.append(_text_elem(caption))
            return await adapter._outbound.sender.dispatch_msg_body(chat_id, msg_body, reply_to, group_code=kwargs.get("group_code", ""))
        except ValueError as ve:
            return SendResult(success=False, error=str(ve))
        except Exception as exc:
            logger.error("[%s] %s.handle() failed: %s", adapter.name, type(self).__name__, exc, exc_info=True)
            return SendResult(success=False, error=str(exc) or type(exc).__name__)


class _ImageHandler(MediaSendHandler):
    """Shared TIMImageElem body builder for image handlers."""
    def build_msg_body(self, upload_result, **kwargs):
        return build_image_msg_body(
            url=upload_result["url"], uuid=kwargs["file_uuid"], filename=kwargs["filename"], size=upload_result["size"],
            width=upload_result.get("width", 0), height=upload_result.get("height", 0), mime_type=kwargs["content_type"],
        )


class ImageUrlHandler(_ImageHandler):
    """Image from a URL (download → COS → TIMImageElem)."""
    async def acquire_file(self, adapter, **kwargs):
        image_url: str = kwargs["image_url"]
        logger.info("[%s] ImageUrlHandler: downloading %s", adapter.name, image_url)
        file_bytes, content_type = await media_download_url(image_url, max_size_mb=adapter.MEDIA_MAX_SIZE_MB)
        path_part = image_url.split("?")[0]
        if not content_type or content_type == "application/octet-stream":
            content_type = guess_mime_type(path_part) or "image/jpeg"
        return file_bytes, os.path.basename(path_part) or "image.jpg", content_type


class ImageFileHandler(_ImageHandler):
    """Image from a local path (read → COS → TIMImageElem)."""
    async def acquire_file(self, adapter, **kwargs):
        return _read_local_file(adapter, "ImageFileHandler", kwargs["image_path"], "image.jpg", "image/jpeg")


class DocumentHandler(MediaSendHandler):
    """Local file/document (read → COS → TIMFileElem)."""
    async def acquire_file(self, adapter, **kwargs):
        return _read_local_file(adapter, "DocumentHandler", kwargs["file_path"], "document", "application/octet-stream", kwargs.get("filename"))

    def build_msg_body(self, upload_result, **kwargs):
        return build_file_msg_body(url=upload_result["url"], filename=kwargs["filename"], uuid=kwargs["file_uuid"], size=upload_result["size"])


class StickerHandler(MediaSendHandler):
    """Sticker/emoji (TIMFaceElem, no COS upload)."""
    def needs_cos_upload(self) -> bool:
        return False

    async def acquire_file(self, adapter, **kwargs):
        return b"", "sticker", "application/octet-stream"  # no file bytes needed

    def build_msg_body(self, upload_result, **kwargs):
        from gateway.platforms.yuanbao_sticker import (
            get_sticker_by_name, get_random_sticker, build_face_msg_body, build_sticker_msg_body,
        )
        sticker_name = kwargs.get("sticker_name")
        if sticker_name is not None:
            sticker = get_sticker_by_name(sticker_name)
            if sticker is None:
                raise ValueError(f"Sticker not found: {sticker_name!r}")
            return build_sticker_msg_body(sticker)
        if kwargs.get("face_index") is not None:
            return build_face_msg_body(face_index=kwargs["face_index"])
        return build_sticker_msg_body(get_random_sticker())


class HeartbeatManager:
    """Reply heartbeat lifecycle: RUNNING every 2s, auto-FINISH after 30s idle, explicit stop."""
    def __init__(self, adapter: "YuanbaoAdapter") -> None:
        self._adapter = adapter
        self._reply_heartbeat_tasks: Dict[str, asyncio.Task] = {}
        self._reply_hb_last_active: Dict[str, float] = {}

    def _ready(self) -> bool:
        return self._adapter._connection.ws is not None and bool(self._adapter._bot_id)

    async def send_heartbeat_once(self, chat_id: str, heartbeat_val: int) -> None:
        """Send a single heartbeat (RUNNING or FINISH), best effort."""
        adapter = self._adapter
        if not self._ready():
            return
        try:
            if chat_id.startswith("group:"):
                encoded = encode_send_group_heartbeat(from_account=adapter._bot_id, group_code=chat_id[len("group:"):], heartbeat=heartbeat_val)
            else:
                encoded = encode_send_private_heartbeat(from_account=adapter._bot_id, to_account=chat_id.removeprefix("direct:"), heartbeat=heartbeat_val)
            await adapter._connection.ws.send(encoded)
            logger.debug("[%s] Reply heartbeat %s sent: chat=%s", adapter.name,
                         "RUNNING" if heartbeat_val == WS_HEARTBEAT_RUNNING else "FINISH", chat_id)
        except Exception as exc:
            logger.debug("[%s] send_heartbeat_once failed: %s", adapter.name, exc)

    async def start(self, chat_id: str) -> None:
        """Start or renew the periodic RUNNING sender."""
        if not self._ready():
            return
        self._reply_hb_last_active[chat_id] = time.time()
        existing = self._reply_heartbeat_tasks.get(chat_id)
        if not existing or existing.done():
            self._reply_heartbeat_tasks[chat_id] = asyncio.create_task(self._worker(chat_id), name=f"yuanbao-reply-hb-{chat_id}")

    async def _worker(self, chat_id: str) -> None:
        """Send RUNNING every 2s; after 30s without renewal (or WS loss) send FINISH and exit.
        A cancelled worker sends no FINISH — stop() decides that."""
        cancelled = False
        try:
            await self.send_heartbeat_once(chat_id, WS_HEARTBEAT_RUNNING)
            while True:
                await asyncio.sleep(REPLY_HEARTBEAT_INTERVAL_S)
                if (time.time() - self._reply_hb_last_active.get(chat_id, 0) > REPLY_HEARTBEAT_TIMEOUT_S
                        or self._adapter._connection.ws is None):
                    break
                await self.send_heartbeat_once(chat_id, WS_HEARTBEAT_RUNNING)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            pass
        finally:
            if not cancelled:
                await self.send_heartbeat_once(chat_id, WS_HEARTBEAT_FINISH)
            self._reply_heartbeat_tasks.pop(chat_id, None)
            self._reply_hb_last_active.pop(chat_id, None)

    async def stop(self, chat_id: str, send_finish: bool = True) -> None:
        """Stop the RUNNING sender and optionally send FINISH."""
        task = self._reply_heartbeat_tasks.pop(chat_id, None)
        if task and not task.done():
            await cancel_task(task)
        if send_finish:
            await self.send_heartbeat_once(chat_id, WS_HEARTBEAT_FINISH)

    async def close(self) -> None:
        _cancel_all(self._reply_heartbeat_tasks)
        self._reply_hb_last_active.clear()


class SlowResponseNotifier:
    """Per-chat timer that sends a courtesy 'please wait' after SLOW_RESPONSE_TIMEOUT_S without a reply."""
    def __init__(self, adapter: "YuanbaoAdapter", sender: "MessageSender") -> None:
        self._adapter = adapter
        self._sender = sender
        self._tasks: Dict[str, asyncio.Task] = {}

    async def start(self, chat_id: str) -> None:
        self.cancel(chat_id)
        self._tasks[chat_id] = asyncio.create_task(self._notifier(chat_id), name=f"yuanbao-slow-resp-{chat_id}")

    async def _notifier(self, chat_id: str) -> None:
        try:
            await asyncio.sleep(SLOW_RESPONSE_TIMEOUT_S)
            logger.info("[%s] Agent response exceeded %ds for %s, sending wait notice", self._adapter.name, int(SLOW_RESPONSE_TIMEOUT_S), chat_id)
            await self._sender.send_text_chunk(chat_id, slow_response_message())
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("[%s] Slow-response notifier failed: %s", self._adapter.name, exc)

    def cancel(self, chat_id: str) -> None:
        task = self._tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()

    async def close(self) -> None:
        _cancel_all(self._tasks)


class MessageSender:
    """Outbound dispatcher: per-chat locks (serial ordering), chunked text with retry, C2C/group
    encoding, media handler strategies, and send_direct for the send_message tool."""
    IMAGE_EXTS: ClassVar[frozenset] = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})
    CHAT_DICT_MAX_SIZE: ClassVar[int] = 1000  # Max distinct chat IDs in _chat_locks
    # @nickname bounded by whitespace / line edges
    _AT_USER_RE = re.compile(r'(?:(?<=\s)|(?<=^))@(\S+?)(?=\s|$)', re.MULTILINE)

    def __init__(self, adapter: "YuanbaoAdapter") -> None:
        self._adapter = adapter
        self._chat_locks: collections.OrderedDict[str, asyncio.Lock] = collections.OrderedDict()
        self._media_handlers: Dict[str, MediaSendHandler] = {
            "image_url": ImageUrlHandler(), "image_file": ImageFileHandler(), "document": DocumentHandler(), "sticker": StickerHandler(),
        }

    def get_chat_lock(self, chat_id: str) -> asyncio.Lock:
        """Per-chat-id lock with LRU eviction (prefers evicting an unlocked entry)."""
        if chat_id in self._chat_locks:
            self._chat_locks.move_to_end(chat_id)
        else:
            if len(self._chat_locks) >= self.CHAT_DICT_MAX_SIZE:
                self._chat_locks.pop(next((k for k in self._chat_locks if not self._chat_locks[k].locked()), next(iter(self._chat_locks))))
            self._chat_locks[chat_id] = asyncio.Lock()
        return self._chat_locks[chat_id]

    async def send_text(self, chat_id: str, content: str, reply_to: Optional[str] = None, group_code: str = "") -> "SendResult":
        """Send text with auto-chunking and per-chat-id ordering guarantee."""
        adapter = self._adapter
        if adapter._connection.ws is None:
            return SendResult(success=False, error="Not connected", retryable=True)
        adapter._outbound.slow_notifier.cancel(chat_id)
        async with self.get_chat_lock(chat_id):
            content_to_send = self.strip_cron_wrapper(content)
            chunks = self.truncate_message(content_to_send, adapter.MAX_TEXT_CHUNK)
            logger.info("[%s] truncate_message: input=%d chars, max=%d, output=%d chunk(s) sizes=%s",
                        adapter.name, len(content_to_send), adapter.MAX_TEXT_CHUNK, len(chunks), [len(c) for c in chunks])
            for i, chunk in enumerate(chunks):
                result = await self.send_text_chunk(chat_id, chunk, reply_to if i == 0 else None, group_code=group_code)
                if not result.success:
                    return result
        with contextlib.suppress(Exception):  # delivery done → FINISH heartbeat (RUNNING… → message → FINISH)
            await adapter._outbound.heartbeat.send_heartbeat_once(chat_id, WS_HEARTBEAT_FINISH)
        return SendResult(success=True)

    async def send_media(self, chat_id: str, handler_name: str, reply_to: Optional[str] = None,
                         caption: Optional[str] = None, **kwargs: Any) -> "SendResult":
        handler = self._media_handlers.get(handler_name)
        if handler is None:
            return SendResult(success=False, error=f"Unknown media handler: {handler_name!r}")
        return await handler.handle(self._adapter, chat_id, reply_to=reply_to, caption=caption, **kwargs)

    async def send_direct(self, chat_id: str, message: str, media_files: Optional[List[Tuple[str, bool]]] = None) -> Dict[str, Any]:
        """send_message tool entry: text first, then each media file by extension, on the running adapter."""
        adapter = self._adapter
        last_result: Optional["SendResult"] = None
        if message.strip():
            last_result = await adapter.send(chat_id, message)
            if not last_result.success:
                return {"error": f"Yuanbao send failed: {last_result.error}"}
        for media_path, _is_voice in media_files or []:
            send = adapter.send_image_file if Path(media_path).suffix.lower() in self.IMAGE_EXTS else adapter.send_document
            last_result = await send(chat_id, media_path)
            if not last_result.success:
                return {"error": f"Yuanbao media send failed: {last_result.error}"}
        if last_result is None:
            return {"error": "No deliverable text or media remained after processing"}
        return {"success": True, "platform": "yuanbao", "chat_id": chat_id, "message_id": last_result.message_id}

    async def dispatch_msg_body(self, chat_id: str, msg_body: list, reply_to: Optional[str] = None, group_code: str = "") -> "SendResult":
        """Lock + dispatch an arbitrary MsgBody to C2C or group."""
        async with self.get_chat_lock(chat_id):
            result = await self._send_msg_body(chat_id, msg_body, reply_to, group_code)
        return self._to_send_result(result)

    @staticmethod
    def _to_send_result(raw: dict) -> "SendResult":
        if raw.get("success"):
            return SendResult(success=True, message_id=raw.get("msg_key"))
        return SendResult(success=False, error=raw.get("error", "Unknown error"))

    async def send_text_chunk(self, chat_id: str, text: str, reply_to: Optional[str] = None, retry: int = 3, group_code: str = "") -> "SendResult":
        """Send a single text chunk with retry (exponential backoff: 1s, 2s, 4s)."""
        adapter = self._adapter
        last_error: str = "Unknown error"
        for attempt in range(retry):
            try:
                if chat_id.startswith("group:"):
                    msg_body = self._build_msg_body_with_mentions(text, chat_id[len("group:"):])
                else:
                    msg_body = [_text_elem(text)]
                raw = await self._send_msg_body(chat_id, msg_body, reply_to, group_code)
                if raw.get("success"):
                    return self._to_send_result(raw)
                last_error = raw.get("error", "Unknown error")
                logger.warning("[%s] send_text_chunk attempt %d/%d failed: %s", adapter.name, attempt + 1, retry, last_error)
            except Exception as exc:
                last_error = str(exc)
                logger.warning("[%s] send_text_chunk attempt %d/%d exception: %s", adapter.name, attempt + 1, retry, last_error)
            if attempt < retry - 1:
                await asyncio.sleep(2 ** attempt)
        logger.error("[%s] send_text_chunk max retries (%d) exceeded. Last error: %s", adapter.name, retry, last_error)
        return SendResult(success=False, error=f"Max retries exceeded: {last_error}")

    async def _send_msg_body(self, chat_id: str, msg_body: list, reply_to: Optional[str], group_code: str) -> dict:
        """Route a MsgBody to group (``group:<code>``) or C2C (``direct:<account>`` / bare account)."""
        if chat_id.startswith("group:"):
            return await self.send_group_msg_body(chat_id[len("group:"):], msg_body, reply_to)
        return await self.send_c2c_msg_body(chat_id.removeprefix("direct:"), msg_body, group_code=group_code)

    def _build_msg_body_with_mentions(self, text: str, group_code: str) -> list:
        """Parse @nickname patterns against the (unexpired) member cache into mixed TIMTextElem +
        TIMCustomElem(elem_type 1002) msg_body; plain text when no members are cached."""
        cached = self._adapter._member_cache.get(group_code)
        if cached and time.time() - cached[0] >= self._adapter.MEMBER_CACHE_TTL_S:
            del self._adapter._member_cache[group_code]
            cached = None
        if not cached or not cached[1]:
            return [_text_elem(text)]
        nickname_to_uid = {}
        for m in cached[1]:
            nick = m.get("nickname") or m.get("nick_name") or ""
            uid = m.get("user_id") or ""
            if nick and uid:
                nickname_to_uid[nick.lower()] = (nick, uid)
        msg_body: list = []
        last_idx = 0
        for match in self._AT_USER_RE.finditer(text):
            seg = text[last_idx:match.start()].strip()
            if seg:
                msg_body.append(_text_elem(seg))
            nickname = match.group(1)
            entry = nickname_to_uid.get(nickname.lower())
            if entry:
                real_nick, uid = entry
                msg_body.append({"msg_type": "TIMCustomElem",
                                 "msg_content": {"data": json.dumps({"elem_type": 1002, "text": f"@{real_nick}", "user_id": uid})}})
            else:
                msg_body.append(_text_elem(f"@{nickname}"))
            last_idx = match.end()
        tail = text[last_idx:].strip()
        if tail:
            msg_body.append(_text_elem(tail))
        return msg_body or [_text_elem(text)]

    async def send_c2c_msg_body(self, to_account: str, msg_body: list, group_code: str = "") -> dict:
        req_id = f"c2c_{next_seq_no()}"
        return await self._dispatch_encoded(self._adapter, encode_send_c2c_message(
            to_account=to_account, msg_body=msg_body, from_account=self._adapter._bot_id or "", msg_id=req_id, group_code=group_code,
        ), req_id)

    async def send_group_msg_body(self, group_code: str, msg_body: list, reply_to: Optional[str] = None) -> dict:
        req_id = f"grp_{next_seq_no()}"
        return await self._dispatch_encoded(self._adapter, encode_send_group_message(
            group_code=group_code, msg_body=msg_body, from_account=self._adapter._bot_id or "", msg_id=req_id, ref_msg_id=reply_to or "",
        ), req_id)

    @staticmethod
    async def _dispatch_encoded(adapter: "YuanbaoAdapter", encoded: bytes, req_id: str) -> dict:
        """Send pre-encoded bytes via WS → ``{"success", "msg_key" | "error"}``."""
        try:
            response = await adapter._connection.send_biz_request(encoded, req_id=req_id)
            return {"success": True, "msg_key": response.get("msg_id", "")}
        except asyncio.TimeoutError:
            return {"success": False, "error": f"Request timeout after {DEFAULT_SEND_TIMEOUT}s"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @staticmethod
    def validate_media(file_bytes: Optional[bytes], filename: str, max_size_mb: int = 20) -> Optional[str]:
        """Pre-upload check; error description or None."""
        if not file_bytes:
            return f"Empty file: {filename}"
        if len(file_bytes) > max_size_mb * 1024 * 1024:
            return f"File too large: {filename} ({len(file_bytes) / 1024 / 1024:.1f}MB > {max_size_mb}MB)"
        return None

    @staticmethod
    def truncate_message(content: str, max_length: int = 4000, len_fn: Optional[Callable[[str], int]] = None) -> List[str]:
        """Table/fence-aware chunking via MarkdownProcessor, stripping ``(1/3)`` page indicators."""
        if (len_fn or len)(content) <= max_length:
            return [content]
        chunks = [_INDICATOR_RE.sub('', c) for c in MarkdownProcessor.chunk_markdown_text(content, max_length, len_fn=len_fn)]
        return chunks or [content]

    @staticmethod
    def strip_cron_wrapper(content: str) -> str:
        """Strip the scheduler's cron header/footer wrapper; unchanged when the shape doesn't match."""
        if not content.startswith(CRON_WRAPPER_HEADER_PREFIX):
            return content
        divider = CRON_WRAPPER_DIVIDER
        footer_prefix = CRON_WRAPPER_FOOTER_PREFIX
        divider_pos = content.find(divider)
        footer_pos = content.rfind(footer_prefix)
        if divider_pos < 0 or footer_pos < 0 or footer_pos <= divider_pos or "\n(job_id: " not in content[:divider_pos]:
            return content
        return content[divider_pos + len(divider):footer_pos].strip() or content

    async def close(self) -> None:
        self._chat_locks.clear()


class OutboundManager:
    """Composes MessageSender, HeartbeatManager and SlowResponseNotifier (sender cancels the notifier
    before a send and emits the FINISH heartbeat after)."""
    def __init__(self, adapter: "YuanbaoAdapter") -> None:
        self._adapter = adapter
        self.sender: MessageSender = MessageSender(adapter)
        self.heartbeat: HeartbeatManager = HeartbeatManager(adapter)
        self.slow_notifier: SlowResponseNotifier = SlowResponseNotifier(adapter, self.sender)

    async def close(self) -> None:
        await self.sender.close()
        await self.heartbeat.close()
        await self.slow_notifier.close()


class YuanbaoAdapter(BasePlatformAdapter):
    """Yuanbao AI Bot adapter backed by a persistent WebSocket connection."""
    PLATFORM = Platform.YUANBAO
    MAX_TEXT_CHUNK: int = 4000  # Yuanbao single message character limit
    splits_long_messages = True  # send() auto-chunks via truncate_message(MAX_TEXT_CHUNK)
    MEDIA_MAX_SIZE_MB: int = 50
    DM_MAX_CHARS = 10000
    _active_instance: ClassVar[Optional["YuanbaoAdapter"]] = None
    # Per Hermes home: a multiplexed gateway runs one Yuanbao adapter per profile, and the tools /
    # send_message read "the" adapter from inside a profile-scoped turn, so last-wins would route
    # profile B's sends through profile A's bot. Registration and lookup both key on the ambient
    # override (connect/reconnect tasks inherit the profile's Context); the slot above serves the
    # unscoped path.
    _active_instances: ClassVar[Dict[str, "YuanbaoAdapter"]] = {}

    @classmethod
    def get_active(cls) -> Optional["YuanbaoAdapter"]:
        from hermes_constants import get_hermes_home_override, hermes_home_key

        if get_hermes_home_override() is None:
            return cls._active_instance
        return cls._active_instances.get(hermes_home_key())

    @classmethod
    def set_active(cls, adapter: Optional["YuanbaoAdapter"]) -> None:
        from hermes_constants import get_hermes_home_override, hermes_home_key

        if get_hermes_home_override() is None:
            cls._active_instance = adapter
        elif adapter is None:
            cls._active_instances.pop(hermes_home_key(), None)
        else:
            cls._active_instances[hermes_home_key()] = adapter

    def __init__(self, config: PlatformConfig, **kwargs: Any) -> None:
        super().__init__(config, Platform.YUANBAO)
        _extra = config.extra or {}
        self._app_key: str = (_extra.get("app_id") or "").strip()
        self._app_secret: str = (_extra.get("app_secret") or "").strip()
        self._bot_id: Optional[str] = _extra.get("bot_id") or None
        self._ws_url: str = (_extra.get("ws_url") or DEFAULT_WS_GATEWAY_URL).strip()
        self._api_domain: str = (_extra.get("api_domain") or DEFAULT_API_DOMAIN).rstrip("/")
        self._route_env: str = (_extra.get("route_env") or "").strip()
        # Media resolve concurrency clamped to [min, max] so a bad config can't hammer the backend.
        try:
            _raw_concurrency = int(_extra.get("media_resolve_concurrency", _DEFAULT_RESOLVE_CONCURRENCY))
        except (TypeError, ValueError):
            _raw_concurrency = _DEFAULT_RESOLVE_CONCURRENCY
        self.media_resolve_concurrency: int = max(_MIN_RESOLVE_CONCURRENCY, min(_MAX_RESOLVE_CONCURRENCY, _raw_concurrency))
        self._connection: ConnectionManager = ConnectionManager(self)
        self._outbound: OutboundManager = OutboundManager(self)
        self._inbound_tasks: set[asyncio.Task] = set()  # cancelled by disconnect()
        self._background_tasks: set[asyncio.Task] = set()  # keeps fire-and-forget tasks alive
        # group_code -> (updated_ts, members); used by @mention resolution, stale after MEMBER_CACHE_TTL_S
        self._member_cache: Dict[str, Tuple[float, list]] = {}
        self.MEMBER_CACHE_TTL_S: float = 300.0
        self._dedup = MessageDeduplicator(ttl_seconds=300)  # WS reconnect / network jitter
        self._group_queues: Dict[str, asyncio.Queue] = {}  # session_key → sequential dispatch queue
        # Recall support: msg_id/text being processed per session_key (RecallGuardMiddleware), plus a
        # bounded msg_id → content cache for content-match redaction when rows lack a message_id.
        self._processing_msg_ids: Dict[str, str] = {}
        self._processing_msg_texts: Dict[str, str] = {}
        self._msg_content_cache: Dict[str, str] = {}

        def _policy(kind: str) -> tuple[str, list[str]]:
            policy = (_extra.get(f"{kind}_policy") or _yb_secret(f"YUANBAO_{kind.upper()}_POLICY") or "pairing").strip().lower()
            raw = _extra.get(f"{kind}_allow_from") or _yb_secret(f"YUANBAO_{kind.upper()}_ALLOW_FROM", "")
            return policy, [x.strip() for x in raw.split(",") if x.strip()]
        self._access_policy = AccessPolicy(*_policy("dm"), *_policy("group"))
        self._inbound_pipeline: InboundPipeline = InboundPipelineBuilder.build()
        # Auto-sethome stays open when no home is set or the home is a group (upgradable by first DM).
        _existing_home = _yb_secret("YUANBAO_HOME_CHANNEL", "") or (config.home_channel.chat_id if config.home_channel else "")
        self._auto_sethome_done: bool = bool(_existing_home) and not _existing_home.startswith("group:")

    def _track_task(self, task: asyncio.Task) -> asyncio.Task:
        """Register a fire-and-forget task so it won't be GC'd prematurely."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    @property
    def enforces_own_access_policy(self) -> bool:
        """Intake gating lives in ``AccessPolicy`` (composed, not inherited), so the flag stays here."""
        return True

    def _sender_may_designate_home(self, ctx: InboundContext) -> bool:
        """Sender may persist YUANBAO_HOME_CHANNEL: strict allowlist, open opt-in, or pairing-approved
        (intake-only pairing forwards are excluded)."""
        policy: AccessPolicy = self._access_policy
        sender = str(ctx.from_account or "").strip()
        if not sender:
            return False
        if ctx.chat_type == "dm":
            if policy.is_dm_allowed(sender):
                return True
            if policy.dm_policy == "pairing":
                from gateway.pairing import PairingStore
                return PairingStore().is_approved(Platform.YUANBAO.value, sender)
            return False
        group_code = str(ctx.group_code or "").strip()
        if ctx.chat_type != "group" or not group_code:
            return False
        if policy.group_policy == "allowlist":
            return policy.is_group_allowed(group_code)
        return policy.group_policy == "open" and policy._open_dm_opted_in()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        ok = await self._connection.open()
        if ok:
            self._wire_plugin_handlers(None)  # plugin-registered native handlers
        return ok

    async def disconnect(self) -> None:
        """Cancel background tasks and close the WebSocket connection."""
        if YuanbaoAdapter._active_instance is self:
            YuanbaoAdapter._active_instance = None
        for home_key, active in list(YuanbaoAdapter._active_instances.items()):
            if active is self:
                del YuanbaoAdapter._active_instances[home_key]
        self._running = False
        self._mark_disconnected()
        self._release_platform_lock()
        await self._connection.close()
        await self._outbound.close()
        for task in list(self._inbound_tasks):
            if not task.done():
                task.cancel()
        self._inbound_tasks.clear()
        self._group_queues.clear()
        logger.info("[%s] Disconnected", self.name)

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None, group_code: str = "") -> SendResult:
        return await self._outbound.sender.send_text(chat_id, content, reply_to, group_code=group_code)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "group" if chat_id.startswith("group:") else "dm"}

    async def send_typing(self, chat_id: str, metadata: Optional[dict] = None) -> None:
        """Start the RUNNING heartbeat (best effort)."""
        with contextlib.suppress(Exception):
            await self._outbound.heartbeat.start(chat_id)

    async def stop_typing(self, chat_id: str) -> None:
        """Stop RUNNING without FINISH — send() emits FINISH after delivery so ordering is
        RUNNING… → message → FINISH."""
        with contextlib.suppress(Exception):
            await self._outbound.heartbeat.stop(chat_id, send_finish=False)

    async def _process_message_background(self, event, session_key: str) -> None:
        """Wrap base class processing with a slow-response notifier."""
        chat_id = event.source.chat_id
        await self._outbound.slow_notifier.start(chat_id)
        try:
            await super()._process_message_background(event, session_key)
        finally:
            self._outbound.slow_notifier.cancel(chat_id)
            # Clear RecallGuard tracking only if our msg_id is still current: a concurrent message may have
            # overwritten it (the drain task then owns it); id-less events never wrote one and must not pop.
            msg_id = event.message_id
            if msg_id and self._processing_msg_ids.get(session_key) == msg_id:
                self._processing_msg_ids.pop(session_key, None)
                self._processing_msg_texts.pop(session_key, None)

    async def _ws_query(self, label: str, group_code: str, encoded: bytes, decode_rsp, empty: dict) -> Optional[dict]:
        """Send an encoded group query over WS; return decoded biz payload, *empty* when none, None on failure."""
        if self._connection.ws is None:
            return None
        try:
            response = await self._connection.send_biz_request(encoded, req_id=decode_conn_msg(encoded)["head"]["msg_id"])
            status = response.get("head", {}).get("status", 0)
            if status != 0:
                logger.warning("[%s] %s failed: status=%d", self.name, label, status)
                return None
            biz_data = response.get("data", b"") or response.get("body", b"")
            return decode_rsp(biz_data) if biz_data and isinstance(biz_data, bytes) else empty
        except asyncio.TimeoutError:
            logger.warning("[%s] %s timeout: group=%s", self.name, label, group_code)
        except Exception as exc:
            logger.warning("[%s] %s failed: %s", self.name, label, exc)
        return None

    async def query_group_info(self, group_code: str) -> Optional[dict]:
        """Group info (name, owner, member count…); None on failure."""
        return await self._ws_query("query_group_info", group_code, encode_query_group_info(group_code),
                                    decode_query_group_info_rsp, {"group_code": group_code})

    async def get_group_member_list(self, group_code: str, offset: int = 0, limit: int = 200) -> Optional[dict]:
        """Group member list; None on failure. Populates ``_member_cache`` for @mention resolution."""
        result = await self._ws_query(
            "get_group_member_list", group_code, encode_get_group_member_list(group_code, offset=offset, limit=limit),
            decode_get_group_member_list_rsp, {"members": [], "next_offset": 0, "is_complete": True},
        )
        if result and result.get("members"):
            self._member_cache[group_code] = (time.time(), result["members"])
        return result

    async def send_dm(self, user_id: str, text: str, group_code: str = "") -> SendResult:
        """Proactive C2C DM (text capped at DM_MAX_CHARS); group_code marks a group-originated DM."""
        if not self._access_policy.is_dm_allowed(user_id):
            return SendResult(success=False, error="DM access denied for this user")
        if len(text) > self.DM_MAX_CHARS:
            text = text[:self.DM_MAX_CHARS] + t("platform.shared.truncated_suffix")
        return await self.send(f"direct:{user_id}", text, group_code=group_code)

    # Media sends delegate to MessageSender.send_media via the named handler strategy.
    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[dict] = None, **kwargs: Any) -> SendResult:
        return await self._outbound.sender.send_media(chat_id, "image_url", reply_to=reply_to, caption=caption, image_url=image_url, **kwargs)

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None,
                              reply_to: Optional[str] = None, metadata: Optional[dict] = None, **kwargs: Any) -> SendResult:
        return await self._outbound.sender.send_media(chat_id, "image_file", reply_to=reply_to, caption=caption, image_path=image_path, **kwargs)

    async def send_sticker(self, chat_id: str, sticker_name: Optional[str] = None, face_index: Optional[int] = None,
                           reply_to: Optional[str] = None, **kwargs: Any) -> SendResult:
        return await self._outbound.sender.send_media(chat_id, "sticker", reply_to=reply_to, sticker_name=sticker_name, face_index=face_index, **kwargs)

    async def send_document(self, chat_id: str, file_path: str, filename: Optional[str] = None, caption: Optional[str] = None,
                            reply_to: Optional[str] = None, metadata: Optional[dict] = None, **kwargs: Any) -> SendResult:
        return await self._outbound.sender.send_media(
            chat_id, "document", reply_to=reply_to, caption=caption, file_path=file_path, filename=filename, **kwargs,
        )

    async def _get_cached_token(self) -> dict:
        """Current valid sign token (module-level cache)."""
        return await SignManager.get_token(self._app_key, self._app_secret, self._api_domain, route_env=self._route_env)
