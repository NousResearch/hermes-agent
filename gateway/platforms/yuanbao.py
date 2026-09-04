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
from gateway.platforms.base import (
    BasePlatformAdapter, MessageEvent, MessageType, SendResult,
    cache_document_from_bytes_async, cache_image_from_bytes_async, cache_video_from_bytes_async,
)
from gateway.platforms import helpers as _mdchunk
from gateway.platforms._shared import get_scoped_secret as _yb_secret
from gateway.platforms.helpers import MessageDeduplicator
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
from gateway.session import build_session_key
from gateway.session_transcript import TranscriptReadError

logger = logging.getLogger(__name__)

# AUTH_BIND / sign-token header values
try:
    from hermes_cli import __version__ as _HERMES_VERSION
except ImportError:
    _HERMES_VERSION = "0.0.0"
_APP_VERSION = _BOT_VERSION = _HERMES_VERSION
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
SLOW_RESPONSE_TIMEOUT_S = 120.0  # push SLOW_RESPONSE_MESSAGE when the agent is silent this long
SLOW_RESPONSE_MESSAGE = "任务有点复杂，正在努力处理中，请耐心等待..."

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


async def _cancel_task(task: asyncio.Task) -> None:
    """Cancel *task* and wait for it to unwind (swallowing the CancelledError)."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


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


@dataclass
class InboundContext:
    """Mutable context passed through every inbound middleware in registration order."""
    adapter: Any  # YuanbaoAdapter (forward-ref avoids circular import)
    raw_frames: list = dc_field(default_factory=list)  # debounce-aggregated raw frames
    push: Optional[dict] = None  # DecodeMiddleware
    decoded_via: str = ""  # "json" | "protobuf"
    from_account: str = ""  # ExtractFieldsMiddleware …
    group_code: str = ""
    group_name: str = ""
    sender_nickname: str = ""
    msg_body: list = dc_field(default_factory=list)
    msg_id: str = ""
    cloud_custom_data: str = ""
    chat_id: str = ""  # ChatRoutingMiddleware …
    chat_type: str = ""  # "dm" | "group"
    chat_name: str = ""
    raw_text: str = ""  # ExtractContentMiddleware …
    media_refs: list = dc_field(default_factory=list)
    forwarded_records: Optional[dict] = None  # parsed ForwardMsgData for elem_type 1009
    owner_command: Optional[str] = None  # OwnerCommandMiddleware
    source: Optional[Any] = None  # SessionSource, BuildSourceMiddleware
    msg_type: Optional[Any] = None  # MessageType | YuanbaoMessageType, ClassifyMessageTypeMiddleware
    reply_to_message_id: Optional[str] = None  # QuoteContextMiddleware …
    reply_to_text: Optional[str] = None
    quote_media_refs: list = dc_field(default_factory=list)  # (rid, kind, filename)
    # MediaResolveMiddleware: deduped local paths — own media, then quoted media, else group-observed media
    media_urls: list = dc_field(default_factory=list)
    media_types: list = dc_field(default_factory=list)
    channel_prompt: Optional[str] = None  # GroupAttributionMiddleware


class InboundMiddleware(ABC):
    """Set class-level ``name`` and implement ``handle(ctx, next_fn)``; ``await next_fn()``
    continues the pipeline, returning without it stops."""
    name: str = ""

    @abstractmethod
    async def handle(self, ctx: InboundContext, next_fn: Callable) -> None: ...

    async def __call__(self, ctx: InboundContext, next_fn: Callable) -> None:
        return await self.handle(ctx, next_fn)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


class InboundPipeline:
    """Onion-model middleware pipeline: named middlewares, ``when`` guards, use_before/use_after/
    remove. Accepts ``InboundMiddleware`` instances or plain ``async def(ctx, next_fn)`` callables."""
    def __init__(self) -> None:
        self._middlewares: list = []  # (name, handler, when_fn | None)

    @staticmethod
    def _normalize(name_or_mw, handler=None):
        if isinstance(name_or_mw, InboundMiddleware):
            return name_or_mw.name, name_or_mw
        return name_or_mw, handler

    def use(self, name_or_mw, handler=None, when=None) -> "InboundPipeline":
        """Append ``pipeline.use(SomeMiddleware())`` or ``pipeline.use("name", fn)``."""
        return self._insert_relative(None, 0, name_or_mw, handler, when)

    def _insert_relative(self, target: Optional[str], offset: int, name_or_mw, handler, when) -> "InboundPipeline":
        """Insert at index(target)+offset; appends when *target* is None or not registered."""
        name, h = self._normalize(name_or_mw, handler)
        idx = next((i for i, (n, _, _) in enumerate(self._middlewares) if n == target), None)
        self._middlewares.insert(len(self._middlewares) if idx is None else idx + offset, (name, h, when))
        return self

    def use_before(self, target: str, name_or_mw, handler=None, when=None) -> "InboundPipeline":
        return self._insert_relative(target, 0, name_or_mw, handler, when)

    def use_after(self, target: str, name_or_mw, handler=None, when=None) -> "InboundPipeline":
        return self._insert_relative(target, 1, name_or_mw, handler, when)

    def remove(self, name: str) -> "InboundPipeline":
        self._middlewares = [(n, h, w) for n, h, w in self._middlewares if n != name]
        return self

    @property
    def middleware_names(self) -> list:
        return [n for n, _, _ in self._middlewares]

    async def execute(self, ctx: InboundContext) -> None:
        """Run the chain; each middleware receives ``(ctx, next_fn)``."""
        chain = self._middlewares
        index = 0

        async def next_fn() -> None:
            nonlocal index
            while index < len(chain):
                name, handler, when_fn = chain[index]
                index += 1
                if when_fn is not None and not when_fn(ctx):
                    continue
                try:
                    await handler(ctx, next_fn)
                except Exception:
                    logger.error("[InboundPipeline] middleware [%s] error", name, exc_info=True)
                    raise
                return
        await next_fn()


class DecodeMiddleware(InboundMiddleware):
    """Decode raw inbound frames (JSON or protobuf via ``decode_inbound_push``) into ctx.push."""
    name = "decode"

    @staticmethod
    def convert_json_msg_body(raw_body: list) -> list:
        """Normalize JSON msg_body (PascalCase or snake_case keys) to [{"msg_type", "msg_content"}]."""
        result = []
        for item in raw_body or []:
            if not isinstance(item, dict):
                continue
            msg_type = item.get("msg_type") or item.get("MsgType", "")
            msg_content = item.get("msg_content") or item.get("MsgContent", {})
            if isinstance(msg_content, str):
                try:
                    msg_content = json.loads(msg_content)
                except Exception:
                    msg_content = {"text": msg_content}
            result.append({"msg_type": msg_type, "msg_content": msg_content or {}})
        return result

    @staticmethod
    def json_sender_fields(raw_json: dict) -> Tuple[str, str]:
        """(from_account, group_code) accepting both Tencent IM PascalCase and internal snake_case keys."""
        from_account = raw_json.get("from_account", "") or raw_json.get("From_Account", "")
        group_code = raw_json.get("group_code", "") or raw_json.get("GroupId", "") or raw_json.get("group_id", "")
        return from_account, group_code

    @staticmethod
    def parse_json_push(raw_json: dict) -> dict | None:
        """JSON push → dict shaped like ``decode_inbound_push`` output; accepts both the callback
        format (callback_command/from_account/msg_body) and legacy keys (GroupId/MsgSeq/MsgKey/MsgBody)."""
        if not raw_json:
            return None
        from_account, group_code = DecodeMiddleware.json_sender_fields(raw_json)
        msg_body = DecodeMiddleware.convert_json_msg_body(raw_json.get("msg_body", []) or raw_json.get("MsgBody", []))
        # Recall callbacks may have neither from_account nor msg_body.
        if not from_account and not msg_body and not raw_json.get("callback_command"):
            return None
        return {
            "callback_command": raw_json.get("callback_command", ""),
            "from_account": from_account,
            "to_account": raw_json.get("to_account", "") or raw_json.get("To_Account", ""),
            "sender_nickname": raw_json.get("sender_nickname", "") or raw_json.get("nick_name", ""),
            "group_code": group_code,
            "group_name": raw_json.get("group_name", ""),
            "msg_seq": raw_json.get("msg_seq", 0) or raw_json.get("MsgSeq", 0),
            "msg_id": raw_json.get("msg_id", "") or raw_json.get("msg_key", "") or raw_json.get("MsgKey", ""),
            "msg_body": msg_body,
            "cloud_custom_data": raw_json.get("cloud_custom_data", "") or raw_json.get("CloudCustomData", ""),
            "bot_owner_id": raw_json.get("bot_owner_id", "") or raw_json.get("botOwnerId", ""),
            "recall_msg_seq_list": raw_json.get("recall_msg_seq_list") or None,
            "trace_id": (raw_json.get("log_ext") or {}).get("trace_id", "") if isinstance(raw_json.get("log_ext"), dict) else "",
        }

    def _decode_single(self, adapter, data: bytes) -> tuple:
        """One raw frame → (push_dict, decoded_via) or (None, '')."""
        try:
            conn_json = json.loads(data.decode("utf-8"))
        except Exception:
            conn_json = None
        if isinstance(conn_json, dict):
            push = self.parse_json_push(conn_json)
            return (push, "json") if push else (None, "")
        try:
            push = decode_inbound_push(data)
        except Exception:
            push = None
        return (push, "protobuf") if push else (None, "")

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if not ctx.raw_frames:
            return  # Stop pipeline — nothing to decode
        merged: Optional[dict] = None
        for data in ctx.raw_frames:
            push, via = self._decode_single(ctx.adapter, data)
            if not push:
                logger.info("[%s] Push decoded but no valid message. raw hex(first64)=%s",
                            ctx.adapter.name, data.hex()[:128] if data else "(empty)")
            elif merged is None:
                merged, ctx.decoded_via = push, via
                logger.info("[%s] Frame decoded (via=%s): len=%d", ctx.adapter.name, via, len(data))
            elif push.get("msg_body", []):  # subsequent pushes: append msg_body, newline-separated
                merged["msg_body"] = merged.get("msg_body", []) + [_text_elem("\n")] + push["msg_body"]
                logger.info("[%s] Merged %d extra msg_body elements from aggregated push", ctx.adapter.name, len(push["msg_body"]))
        if not merged:
            return  # Stop pipeline
        ctx.push = merged
        logger.info(
            "[%s] Push decoded (via=%s): from=%s group=%s msg_id=%s msg_types=%s",
            ctx.adapter.name, ctx.decoded_via, merged.get("from_account", ""), merged.get("group_code", ""),
            merged.get("msg_id", ""), [e.get("msg_type", "") for e in merged.get("msg_body", [])],
        )
        logger.debug("[%s] Push payload: %s", ctx.adapter.name, ctx.push)
        await next_fn()


class ExtractFieldsMiddleware(InboundMiddleware):
    """Copy common push fields onto ctx."""
    name = "extract-fields"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        for f in ("from_account", "group_code", "group_name", "sender_nickname", "msg_id", "cloud_custom_data"):
            setattr(ctx, f, ctx.push.get(f, ""))
        ctx.msg_body = ctx.push.get("msg_body", [])
        await next_fn()


class DedupMiddleware(InboundMiddleware):
    name = "dedup"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if ctx.msg_id and ctx.adapter._dedup.is_duplicate(ctx.msg_id):
            logger.debug("[%s] Duplicate message ignored: msg_id=%s", ctx.adapter.name, ctx.msg_id)
            return  # Stop pipeline
        await next_fn()


def _session_store(adapter):
    """Adapter's SessionStore, or None before ``set_session_store`` ran."""
    return getattr(adapter, "_session_store", None)


class RecallGuardMiddleware(InboundMiddleware):
    """Recall callbacks (Group.CallbackAfterRecallMsg / C2C.CallbackAfterMsgWithDraw).
    A: in transcript → redact; B: not in transcript → system note; C: being processed → interrupt + delayed redact."""
    name = "recall_guard"
    _RECALL_COMMANDS = frozenset({"Group.CallbackAfterRecallMsg", "C2C.CallbackAfterMsgWithDraw"})
    _REDACTED = "[This message was recalled/withdrawn by the sender; original content removed]"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        cmd = (ctx.push or {}).get("callback_command", "")
        if cmd in self._RECALL_COMMANDS:
            self._handle_recall(ctx, cmd)  # terminal: recalls never dispatch
        else:
            await next_fn()

    @staticmethod
    def _build_source(adapter, group_code: str, from_account: str):
        return adapter.build_source(
            chat_id=(f"group:{group_code}" if group_code else f"direct:{from_account}"),
            chat_type="group" if group_code else "dm",
            user_id=from_account or None,
            thread_id="main" if group_code else None,
        )

    @classmethod
    def _resolve_sid(cls, store, adapter, group_code: str, from_account: str) -> str:
        return store.get_or_create_session(cls._build_source(adapter, group_code, from_account)).session_id

    @classmethod
    def _redact(cls, adapter, store, sid: str, transcript: list, entry: dict, ok_msg: str, fail_msg: str, *ok_args) -> None:
        """Blank *entry* in place and persist *transcript* (warns, never raises)."""
        entry["content"] = cls._REDACTED
        try:
            store.rewrite_transcript(sid, transcript, active_only=True)
            logger.info(ok_msg, adapter.name, *ok_args)
        except Exception as exc:
            logger.warning(fail_msg, adapter.name, exc)

    def _handle_recall(self, ctx: InboundContext, cmd: str) -> None:
        adapter = ctx.adapter
        push = ctx.push or {}
        if cmd == "Group.CallbackAfterRecallMsg":
            seq_list = push.get("recall_msg_seq_list") or []
        else:
            mid, seq = push.get("msg_id") or "", push.get("msg_seq")
            seq_list = [{"msg_id": mid, "msg_seq": seq}] if (mid or seq) else []
        if not seq_list:
            logger.debug("[%s] Recall callback with empty seq_list, skipping", adapter.name)
            return
        group_code = (push.get("group_code") or "").strip()
        from_account = (push.get("from_account") or "").strip()
        for seq_entry in seq_list:
            recalled_id = seq_entry.get("msg_id") or str(seq_entry.get("msg_seq") or "")
            if not recalled_id:
                continue
            matched_sk = self._find_processing_session(adapter, recalled_id)
            if matched_sk is not None:
                self._interrupt_for_recall(adapter, matched_sk, recalled_id, group_code, from_account)
            else:
                self._patch_transcript(adapter, recalled_id, group_code, from_account, adapter._msg_content_cache.get(recalled_id))

    # -- Branch C: interrupt currently-processing message ---------------

    @staticmethod
    def _find_processing_session(adapter, recalled_id: str) -> Optional[str]:
        return next((sk for sk, mid in adapter._processing_msg_ids.items()
                     if mid == recalled_id and sk in adapter._active_sessions), None)

    @classmethod
    def _interrupt_for_recall(cls, adapter, session_key: str, recalled_id: str, group_code: str, from_account: str) -> None:
        where = f"group {group_code}" if group_code else f"direct chat with {from_account}"
        recall_text = (
            f"[CRITICAL — MESSAGE RECALLED] The user message that triggered your current task "
            f"(message_id=\"{recalled_id}\") in {where} has been recalled/withdrawn by the sender. "
            "IGNORE any prior system note asking you to finish processing tool results — the original request is void. "
            "Do NOT continue the task, do NOT call more tools, do NOT reference the recalled content. "
            "Reply only with a brief acknowledgment such as \"The message has been recalled.\" in the "
            "language the user was using."
        )
        # Set pending + signal directly (bypass handle_message to avoid busy-ack).
        # May overwrite a user message pending in the same ~200ms window — acceptable.
        adapter._pending_messages[session_key] = MessageEvent(
            text=recall_text, message_type=MessageType.TEXT, source=cls._build_source(adapter, group_code, from_account), internal=True)
        active_event = adapter._active_sessions.get(session_key)
        if active_event is not None:
            active_event.set()
        logger.info("[%s] Recall interrupt: msg_id=%s session=%s", adapter.name, recalled_id, session_key[:30])
        # The interrupted turn persists the recalled content *after* our interrupt — redact later.
        recalled_text = adapter._processing_msg_texts.get(session_key, "")
        if recalled_text:
            cls._schedule_content_redact(adapter, session_key, recalled_text, group_code, from_account)

    @classmethod
    def _schedule_content_redact(cls, adapter, session_key: str, recalled_text: str, group_code: str, from_account: str) -> None:
        async def _redact() -> None:
            store = _session_store(adapter)
            if not store:
                return
            try:
                sid = cls._resolve_sid(store, adapter, group_code, from_account)
            except Exception:
                return
            # Poll until the recalled content appears — the interrupted turn hasn't finished writing yet.
            for _ in range(30):
                await asyncio.sleep(0.5)
                try:
                    transcript = store.load_transcript(sid)
                except TranscriptReadError as exc:
                    # No readable rows means nothing to redact; polling on
                    # would just re-log the same failure (#100788).
                    logger.warning(
                        "[%s] Recall redact: transcript unreadable for "
                        "session %s: %s", adapter.name, sid, exc,
                    )
                    return
                except Exception:
                    continue
                for entry in transcript:
                    if entry.get("role") == "user" and entry.get("content") == recalled_text:
                        cls._redact(adapter, store, sid, transcript, entry, "[%s] Recall redact: session %s",
                                    "[%s] Recall redact failed: %s", session_key[:30])
                        return
            logger.debug("[%s] Recall redact: content not found after polling, session %s", adapter.name, session_key[:30])
        adapter._track_task(asyncio.create_task(_redact()))

    # -- Branch A/B: patch transcript (session idle) --------------------

    @classmethod
    def _patch_transcript(cls, adapter, recalled_id: str, group_code: str,
                          from_account: str, recalled_content: Optional[str] = None) -> None:
        store = _session_store(adapter)
        if not store:
            return
        try:
            sid = cls._resolve_sid(store, adapter, group_code, from_account)
        except Exception as exc:
            logger.warning("[%s] Recall: failed to resolve session: %s", adapter.name, exc)
            return
        try:
            # Load transcript from canonical store (state.db). Since PR #29278 added a
            # ``platform_message_id`` column to the messages table and ``append_to_transcript`` wires the
            # incoming dict's ``message_id`` into it, ``load_transcript`` returns rows with ``message_id``
            # set for any message that was observed with one — Branch A1 (exact id match) is the canonical
            # path again.
            transcript = store.load_transcript(sid)
        except TranscriptReadError as exc:
            # Not an empty transcript — the rows are unreadable, so recall has
            # nothing to match against (#100788).
            logger.warning("[%s] Recall: transcript unreadable: %s", adapter.name, exc)
            return
        except Exception as exc:
            logger.warning("[%s] Recall: failed to load transcript: %s", adapter.name, exc)
            return
        # A1: exact platform message_id match; A2: content-match fallback for rows without a
        # platform id (agent-processed @bot messages — run.py doesn't carry msg_id — or pre-column rows).
        target = next((e for e in transcript if e.get("message_id") == recalled_id), None)
        branch_label = "branch A1: id match"
        if target is None and recalled_content:
            target = next((e for e in transcript if e.get("role") == "user" and e.get("content") == recalled_content), None)
            branch_label = "branch A2: content match"
        if target is not None:
            cls._redact(adapter, store, sid, transcript, target, "[%s] Recall: redacted msg_id=%s (%s)",
                        "[%s] Recall: rewrite_transcript failed: %s", recalled_id, branch_label)
            return
        # Branch B: not found in transcript → append system note
        store.append_to_transcript(sid, {
            "role": "system", "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "content": f'[recall] message_id="{recalled_id}" has been recalled; do not quote or reference it.',
        })
        logger.info("[%s] Recall: system note for msg_id=%s (branch B)", adapter.name, recalled_id)


class SkipSelfMiddleware(InboundMiddleware):
    """Drop the bot's own messages."""
    name = "skip-self"

    @staticmethod
    def _is_self_reference(from_account: str, bot_id: Optional[str]) -> bool:
        return bool(from_account) and from_account == bot_id

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if self._is_self_reference(ctx.from_account, ctx.adapter._bot_id):
            logger.debug("[%s] Ignoring self-sent message from %s", ctx.adapter.name, ctx.from_account)
            return  # Stop pipeline
        await next_fn()


class ChatRoutingMiddleware(InboundMiddleware):
    """Derive chat_id / chat_type / chat_name from push fields."""
    name = "chat-routing"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if ctx.group_code:
            ctx.chat_id, ctx.chat_type, ctx.chat_name = f"group:{ctx.group_code}", "group", ctx.group_name or ctx.group_code
        else:
            ctx.chat_id, ctx.chat_type, ctx.chat_name = f"direct:{ctx.from_account}", "dm", ctx.sender_nickname or ctx.from_account
        await next_fn()


class AccessPolicy:
    """DM / group access rules shared by inbound middleware and outbound ``send_dm``."""
    def __init__(self, dm_policy: str, dm_allow_from: list[str], group_policy: str, group_allow_from: list[str]) -> None:
        self._dm_policy = dm_policy
        self._dm_allow_from = dm_allow_from
        self._group_policy = group_policy
        self._group_allow_from = group_allow_from

    def _open_dm_opted_in(self) -> bool:
        return any((_yb_secret(k, "") or "").lower() in {"true", "1", "yes"}
                   for k in ("GATEWAY_ALLOW_ALL_USERS", "YUANBAO_ALLOW_ALL_USERS"))

    def _evaluate(self, policy: str, allow_from: list[str], principal: str, *, pairing: bool) -> bool:
        """Shared allow/deny rule; *pairing* is the verdict for the "pairing" policy."""
        if policy == "allowlist":
            return principal in allow_from
        if policy == "pairing":
            return pairing
        if policy == "open":
            return self._open_dm_opted_in()
        return False  # "disabled" or unknown

    def is_dm_allowed(self, sender_id: str) -> bool:
        """Strict DM authorization — pairing does not imply access."""
        return self._evaluate(self._dm_policy, self._dm_allow_from, sender_id.strip(), pairing=False)

    def is_dm_intake_allowed(self, sender_id: str) -> bool:
        """Whether a DM may reach gateway intake (pairing handshake path)."""
        principal = str(sender_id or "").strip()
        return bool(principal) and self._evaluate(self._dm_policy, self._dm_allow_from, principal, pairing=True)

    def is_group_allowed(self, group_code: str) -> bool:
        return self._evaluate(self._group_policy, self._group_allow_from, group_code.strip(), pairing=False)

    @property
    def dm_policy(self) -> str:
        return self._dm_policy

    @property
    def group_policy(self) -> str:
        return self._group_policy


class AccessGuardMiddleware(InboundMiddleware):
    """Platform-level DM/group access filter."""
    name = "access-guard"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        policy: AccessPolicy = ctx.adapter._access_policy
        if ctx.chat_type == "dm" and not policy.is_dm_intake_allowed(ctx.from_account):
            logger.debug("[%s] DM from %s blocked by dm_policy=%s", ctx.adapter.name, ctx.from_account, policy.dm_policy)
            return  # Stop pipeline
        if ctx.chat_type == "group" and not policy.is_group_allowed(ctx.group_code):
            logger.debug("[%s] Group %s blocked by group_policy=%s", ctx.adapter.name, ctx.group_code, policy.group_policy)
            return  # Stop pipeline
        await next_fn()


class AutoSetHomeMiddleware(InboundMiddleware):
    """Silently designate the first inbound conversation as home channel (config.yaml + env);
    a group home is upgraded by the first DM. Runs after GroupAtGuard so unaddressed group traffic
    never claims it; only strictly-authorized senders (allowlist / open opt-in / pairing-approved)
    may — intake-only pairing forwards must not."""
    name = "auto-sethome"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        adapter = ctx.adapter
        if not adapter._auto_sethome_done and adapter._sender_may_designate_home(ctx):
            _cur_home = os.getenv("YUANBAO_HOME_CHANNEL", "")
            _should_set = not _cur_home or (_cur_home.startswith("group:") and ctx.chat_type == "dm")
            if ctx.chat_type == "dm":
                adapter._auto_sethome_done = True  # DM seen — no further upgrades needed
            if _should_set:
                self._persist_home(adapter, ctx)
        await next_fn()

    @staticmethod
    def _persist_home(adapter, ctx: InboundContext) -> None:
        try:
            from hermes_constants import get_hermes_home
            from hermes_cli.config import atomic_config_write, read_user_config_raw
            config_path = get_hermes_home() / "config.yaml"
            # Raw read: merged defaults must not be persisted to the user's file.
            user_config: dict = read_user_config_raw(config_path)
            user_config["YUANBAO_HOME_CHANNEL"] = ctx.chat_id
            atomic_config_write(config_path, user_config)
            os.environ["YUANBAO_HOME_CHANNEL"] = str(ctx.chat_id)
            logger.info("[%s] Auto-sethome: designated %s (%s) as Yuanbao home channel", adapter.name, ctx.chat_id, ctx.chat_name)
        except Exception as e:
            logger.warning("[%s] Auto-sethome failed: %s", adapter.name, e)


def _iter_custom_elems(msg_body: list) -> Iterator[Tuple[Any, dict]]:
    """Yield ``(custom, content)`` for each TIMCustomElem whose ``data`` parses as JSON (any type)."""
    for elem in msg_body or []:
        if not isinstance(elem, dict) or elem.get("msg_type") != "TIMCustomElem":
            continue
        content = elem.get("msg_content", {}) or {}
        data_str = content.get("data", "") if isinstance(content, dict) else ""
        if data_str:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                yield json.loads(data_str), content


def _file_name(content: dict) -> str:
    """First non-empty of file_name / fileName / filename, stripped."""
    return (str(content.get("file_name") or "").strip() or str(content.get("fileName") or "").strip()
            or str(content.get("filename") or "").strip())


def _media_ref(kind: str, url: str, name: str = "") -> Dict[str, str]:
    """media_refs entry; ``name`` is only emitted when non-empty."""
    ref: Dict[str, str] = {"kind": kind, "url": url}
    if name:
        ref["name"] = name
    return ref


class ExtractContentMiddleware(InboundMiddleware):
    """Extract raw text, media refs and forwarded records from msg_body."""
    name = "extract-content"
    _CARD_CONTENT_MAX_LENGTH = 1000
    _UNSUPPORTED = "[unsupported message type]"

    @staticmethod
    def _format_shared_link(custom: dict) -> str:
        """elem_type 1010 (share card) → bracket-placeholder text."""
        title, link = custom.get("title", ""), custom.get("link", "")
        lines = [f"[share_card: {title} | {link}]" if link else f"[share_card: {title}]"]
        max_len = ExtractContentMiddleware._CARD_CONTENT_MAX_LENGTH
        preview = next((v for v in (custom.get("card_content"), custom.get("wechat_des")) if v and isinstance(v, str)), None)
        if preview:
            lines.append(f"Preview: {preview[:max_len] + '...(truncated)' if len(preview) > max_len else preview}")
        if link:
            lines.append("[visit link for full content]")
        return "\n".join(lines)

    @staticmethod
    def _format_link_understanding(custom: dict) -> Optional[str]:
        """elem_type 1007 (link understanding card) → bracket-placeholder text."""
        content = custom.get("content")
        if not content:
            return None
        try:
            parsed = json.loads(content)
            link = parsed.get("link") if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            link = None
        return f"[link: {link} | visit link for full content]" if link and isinstance(link, str) else None

    @staticmethod
    def _parse_resource_id(url: str) -> str:
        """resourceId from a Yuanbao resource URL's query string, or ""."""
        try:
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query) if url else {}
            ids = query.get("resourceId") or query.get("resourceid") or []
            return str(ids[0]).strip() if ids else ""
        except Exception:
            return ""

    @staticmethod
    def _pick_image_url(content: dict) -> str:
        """URL of the medium image (index 1), falling back to index 0, else ""."""
        arr = content.get("image_info_array")
        arr = arr if isinstance(arr, list) else []
        image_info = arr[1] if len(arr) > 1 and isinstance(arr[1], dict) else arr[0] if arr and isinstance(arr[0], dict) else None
        return str((image_info or {}).get("url") or "").strip()

    @classmethod
    def _extract_text(cls, msg_body: list) -> str:
        """Plain text from MsgBody: text elems verbatim; media as ``[kind|ybres:RID]`` / ``[kind]``
        (file: ``[file:{name}|ybres:RID]``); TIMFaceElem ``[emoji: name]``; custom elems by
        elem_type. Parts are space-joined."""
        parts: list[str] = []
        for elem in msg_body:
            elem_type: str = elem.get("msg_type", "")
            content: dict = elem.get("msg_content", {})
            if elem_type == _TEXT_ELEM_TYPE:
                if content.get("text", ""):
                    parts.append(content["text"])
            elif elem_type in ("TIMImageElem", "TIMSoundElem", "TIMVideoFileElem"):
                kind = {"TIMImageElem": "image", "TIMSoundElem": "voice", "TIMVideoFileElem": "video"}[elem_type]
                url = cls._pick_image_url(content) if kind == "image" else str(content.get("url") or "").strip()
                rid = cls._parse_resource_id(url)
                parts.append(f"[{kind}|ybres:{rid}]" if rid else f"[{kind}]")
            elif elem_type == "TIMFileElem":
                filename = content.get("file_name", content.get("fileName", content.get("filename", "")))
                rid = cls._parse_resource_id(str(content.get("url") or "").strip())
                if rid:
                    parts.append(f"[file:{filename}|ybres:{rid}]" if filename else f"[file|ybres:{rid}]")
                else:
                    parts.append(f"[file: {filename}]" if filename else "[file]")
            elif elem_type == "TIMCustomElem":
                parts.append(cls._custom_elem_text(content.get("data", "")))
            elif elem_type == "TIMFaceElem":
                face_name = ""
                if content.get("data", ""):
                    try:
                        face_name = (json.loads(content["data"]).get("name") or "").strip()
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass
                parts.append(f"[emoji: {face_name}]" if face_name else "[emoji]")
            elif elem_type:
                parts.append(f"[{elem_type}]")  # unknown element type — keep as placeholder
        return " ".join(parts)

    @classmethod
    def _custom_elem_text(cls, data_val: str) -> str:
        """Text for a TIMCustomElem by elem_type: 1002 mention, 1010 share card, 1007 link card,
        1009 forwarded chat-record summary; malformed JSON is passed through verbatim."""
        if not data_val:
            return cls._UNSUPPORTED
        try:
            custom = json.loads(data_val)
        except (json.JSONDecodeError, TypeError):
            return data_val
        if not isinstance(custom, dict):
            return cls._UNSUPPORTED
        ctype = custom.get("elem_type")
        if ctype == 1002:
            return custom.get("text", "[mention]")
        if ctype == 1009:
            return custom.get("text", "[chat record]")
        if ctype == 1010:
            return cls._format_shared_link(custom)
        return (cls._format_link_understanding(custom) if ctype == 1007 else None) or cls._UNSUPPORTED

    @staticmethod
    def _rewrite_slash_command(text: str) -> str:
        """Strip; convert a leading full-width slash (Chinese IME) to ASCII so commands match."""
        text = text.strip()
        return '/' + text[1:] if text.startswith('\uff0f') else text

    @staticmethod
    def _extract_inbound_media_refs(msg_body: list) -> List[Dict[str, str]]:
        """Inbound image/file refs: ``[{"kind": "image", "url": ...}, {"kind": "file", "url": ..., "name": ...}]``."""
        refs: List[Dict[str, str]] = []
        for elem in msg_body or []:
            if not isinstance(elem, dict):
                continue
            msg_type = elem.get("msg_type", "")
            content = elem.get("msg_content", {}) or {}
            if not isinstance(content, dict):
                continue
            if msg_type == "TIMImageElem":
                image_url = ExtractContentMiddleware._pick_image_url(content)
                if image_url:
                    refs.append(_media_ref("image", image_url))
            elif msg_type == "TIMFileElem":
                file_url = str(content.get("url") or "").strip()
                if file_url:
                    refs.append(_media_ref("file", file_url, _file_name(content)))
        return refs

    @staticmethod
    def _extract_forwarded_records(msg_body: list, user_id: str = "") -> Optional[dict]:
        """ForwardMsgData for elem_type 1009 (WeChat forward), or None. Payload lives in
        ``msg_content.ext_map`` (pb field 999) under ``wexin_forward_msg_[id]_[userid]`` keys as
        base64 protobuf (NOT JSON); the first entry decoding to ``sub_type == 1`` wins."""
        for custom, content in _iter_custom_elems(msg_body):
            if not (isinstance(custom, dict) and custom.get("elem_type") == 1009):
                continue
            ext_map = content.get("ext_map") or {}
            if not isinstance(ext_map, dict) or not ext_map:
                return None
            for key, value in ext_map.items():
                if not key.startswith("wexin_forward_msg_") or not isinstance(value, str) or not value:
                    continue
                with contextlib.suppress(binascii.Error, ValueError):
                    data = decode_forward_msg_data(base64.b64decode(value))
                    if isinstance(data, dict) and data.get("sub_type") == 1:
                        return data
        return None

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        ctx.raw_text = self._rewrite_slash_command(self._extract_text(ctx.msg_body))
        ctx.media_refs = self._extract_inbound_media_refs(ctx.msg_body)
        ctx.forwarded_records = self._extract_forwarded_records(ctx.msg_body, ctx.from_account)
        await next_fn()


class PlaceholderFilterMiddleware(InboundMiddleware):
    """Skip pure placeholder messages (e.g. '[image]' with no media)."""
    name = "placeholder-filter"
    SKIPPABLE_PLACEHOLDERS: frozenset = frozenset({"[image]", "[图片]", "[file]", "[文件]", "[video]", "[视频]", "[voice]", "[语音]"})

    @classmethod
    def is_skippable_placeholder(cls, text: str, media_count: int = 0) -> bool:
        return media_count <= 0 and text.strip() in cls.SKIPPABLE_PLACEHOLDERS

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if self.is_skippable_placeholder(ctx.raw_text, len(ctx.media_refs)):
            logger.debug("[%s] Skipping placeholder message: %r", ctx.adapter.name, ctx.raw_text)
            return  # Stop pipeline
        await next_fn()


class OwnerCommandMiddleware(InboundMiddleware):
    """Bot-owner slash commands in groups: allowlisted commands skip @Bot; non-owner attempts are rejected."""
    name = "owner-command"
    ALLOWLIST: frozenset = frozenset({"/new", "/reset", "/retry", "/undo", "/stop", "/approve", "/deny", "/bg", "/btw", "/queue", "/q"})
    _rewrite_slash_command = staticmethod(ExtractContentMiddleware._rewrite_slash_command)

    @classmethod
    def _detect_owner_command(cls, *, push: dict, msg_body: list, chat_type: str, from_account: str) -> Tuple[Optional[str], Optional[str], bool]:
        """→ (cmd, cmd_line, is_owner); (None, None, False) when not an allowlisted command."""
        if chat_type != "group" or not cls.ALLOWLIST:
            return None, None, False
        # Only recognise commands when there is exactly one text segment.
        text_elems = [e for e in (msg_body or []) if e.get("msg_type") == _TEXT_ELEM_TYPE]
        if len(text_elems) != 1:
            return None, None, False
        cmd_line = cls._rewrite_slash_command((text_elems[0].get("msg_content") or {}).get("text", ""))
        if not cmd_line.startswith("/"):
            return None, None, False
        cmd = cmd_line.split(maxsplit=1)[0].lower()
        if cmd not in cls.ALLOWLIST:
            return None, None, False
        # Owner ⇔ push.from_account == push.bot_owner_id; these commands are privileged
        # (/approve, /stop, /reset…) so a non-owner must never run them.
        owner_id = str((push or {}).get("bot_owner_id") or "").strip()
        return cmd, cmd_line, bool(owner_id) and owner_id == from_account

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        adapter = ctx.adapter
        matched_cmd, cmd_line, is_owner = self._detect_owner_command(
            push=ctx.push, msg_body=ctx.msg_body, chat_type=ctx.chat_type, from_account=ctx.from_account,
        )
        if matched_cmd and not is_owner:
            logger.info("[%s] Reject non-owner slash command: chat=%s from=%s cmd=%s", adapter.name, ctx.chat_id, ctx.from_account, matched_cmd)
            adapter._track_task(asyncio.create_task(
                adapter.send(ctx.chat_id, f"⚠️ {matched_cmd} is only available to the creator in private chat mode"),
                name=f"yuanbao-owner-cmd-denial-{matched_cmd}"))
            return  # Stop pipeline
        if matched_cmd and is_owner and cmd_line:
            logger.info("[%s] Bot owner slash command: chat=%s from=%s cmd=%s", adapter.name, ctx.chat_id, ctx.from_account, matched_cmd)
            ctx.owner_command = matched_cmd
            ctx.raw_text = cmd_line  # clean command text
        await next_fn()


class BuildSourceMiddleware(InboundMiddleware):
    name = "build-source"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        ctx.source = ctx.adapter.build_source(
            chat_id=ctx.chat_id, chat_type=ctx.chat_type, chat_name=ctx.chat_name,
            user_id=ctx.from_account or None, user_name=ctx.sender_nickname or ctx.from_account,
            thread_id="main" if ctx.chat_type == "group" else None,
        )
        await next_fn()


class GroupAtGuardMiddleware(InboundMiddleware):
    """Group chat: observe non-@bot messages into the transcript; only @Bot (or owner commands) proceed."""
    name = "group-at-guard"

    @staticmethod
    def _iter_bot_mentions(msg_body: list, bot_id: Optional[str]) -> Iterator[dict]:
        """Yield @bot elems: TIMCustomElem whose data JSON has elem_type 1002 and user_id == bot_id."""
        if not bot_id:
            return
        for custom, _content in _iter_custom_elems(msg_body):
            if custom.get("elem_type") == 1002 and custom.get("user_id") == bot_id:
                yield custom

    @classmethod
    def _is_at_bot(cls, msg_body: list, bot_id: Optional[str]) -> bool:
        return any(True for _ in cls._iter_bot_mentions(msg_body, bot_id))

    @classmethod
    def _extract_bot_mention_text(cls, msg_body: list, bot_id: Optional[str]) -> str:
        """Display text used to @-mention this bot (e.g. ``@yuanbao-bot``), or ""."""
        return next((t for t in (str(c.get("text") or "").strip() for c in cls._iter_bot_mentions(msg_body, bot_id)) if t), "")

    @staticmethod
    def _build_group_channel_prompt(msg_body: list, bot_id: Optional[str]) -> str:
        """Per-turn group-chat prompt that highlights which message to respond to."""
        bot_mention = GroupAtGuardMiddleware._extract_bot_mention_text(msg_body, bot_id) or "unknown"
        return (
            "You are handling a Yuanbao group chat message.\n"
            f"- Your identity: user_id={bot_id or 'unknown'}, @-mention name in this group={bot_mention}\n"
            "- Lines in history prefixed with `[nickname|user_id]` are observed group context "
            "and are not necessarily addressed to you.\n"
            "- Treat only the current new message as a request explicitly directed at you, "
            "and answer it directly."
        )

    @classmethod
    def _observe_group_message(cls, adapter, source, sender_display: str, text: str, *, ctx: InboundContext,
                               msg_id: Optional[str] = None, forwarded_records: Optional[dict] = None) -> None:
        """Record a group message as ``role: user`` ``[nickname|user_id]\\n<content>`` without
        invoking the agent, so later @bot turns see the full conversation."""
        store = _session_store(adapter)
        if not store:
            return
        try:
            session_entry = store.get_or_create_session(source)
            body_text = text
            if forwarded_records:
                summary = ForwardedRecordsParseMiddleware.build_forward_text(forwarded_records, ctx=ctx, is_dispatch=False)
                if summary:
                    body_text = f"{text}\n{summary}" if text else summary
            entry: dict = {
                "role": "user", "content": f"[{sender_display}|{source.user_id or 'unknown'}]\n{body_text}",
                "timestamp": datetime.now(tz=timezone.utc).isoformat(), "observed": True,
            }
            if msg_id:
                entry["message_id"] = msg_id
            store.append_to_transcript(session_entry.session_id, entry)
        except Exception as exc:
            logger.warning("[%s] Failed to observe group message: %s", adapter.name, exc)

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        adapter = ctx.adapter
        if ctx.chat_type == "group" and not ctx.owner_command and not self._is_at_bot(ctx.msg_body, adapter._bot_id):
            self._observe_group_message(
                adapter, ctx.source, ctx.sender_nickname or ctx.from_account, ctx.raw_text,
                msg_id=ctx.msg_id or None, forwarded_records=ctx.forwarded_records, ctx=ctx,
            )
            logger.info("[%s] Group message observed (no @bot): chat=%s from=%s", adapter.name, ctx.chat_id, ctx.from_account)
            return  # Stop pipeline — message observed but not dispatched
        await next_fn()


class GroupAttributionMiddleware(InboundMiddleware):
    """Group @bot turns: build channel_prompt, rewrite raw_text to ``[nickname|user_id]\\n<content>``
    (matches observed-history format) and clear ``source.user_name`` to suppress the runner's
    ``[user_name]`` prefix."""
    name = "group-attribution"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if ctx.chat_type == "group" and not ctx.owner_command:
            ctx.channel_prompt = GroupAtGuardMiddleware._build_group_channel_prompt(ctx.msg_body, ctx.adapter._bot_id)
            ctx.raw_text = f"[{ctx.sender_nickname or ctx.from_account or 'unknown'}|{ctx.from_account or 'unknown'}]\n{ctx.raw_text}"
            if ctx.source is not None:
                ctx.source = dataclasses.replace(ctx.source, user_name=None)
        await next_fn()


class YuanbaoMessageType(Enum):
    CHAT_RECORD = "chat_record"  # yuanbao-local subtype; coerced back to MessageType in DispatchMiddleware


_ELEM_MESSAGE_TYPES = {"TIMImageElem": MessageType.PHOTO, "TIMSoundElem": MessageType.VOICE,
                       "TIMVideoFileElem": MessageType.VIDEO, "TIMFileElem": MessageType.DOCUMENT}


class ClassifyMessageTypeMiddleware(InboundMiddleware):
    """MessageType (or yuanbao-local YuanbaoMessageType) from text and msg_body elements."""
    name = "classify-msg-type"

    @staticmethod
    def _classify(text: str, msg_body: list):
        if text.startswith("/"):
            return MessageType.COMMAND
        for elem in msg_body:
            etype = elem.get("msg_type", "")
            mapped = _ELEM_MESSAGE_TYPES.get(etype)
            if mapped is not None:
                return mapped
            if etype == "TIMCustomElem":
                try:
                    custom = json.loads((elem.get("msg_content") or {}).get("data", ""))
                except (json.JSONDecodeError, TypeError):
                    custom = None
                if isinstance(custom, dict) and custom.get("elem_type") == 1009:
                    return YuanbaoMessageType.CHAT_RECORD
        return MessageType.TEXT

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        ctx.msg_type = self._classify(ctx.raw_text, ctx.msg_body)
        await next_fn()


class QuoteContextMiddleware(InboundMiddleware):
    """Extract quote/reply context from cloud_custom_data."""
    name = "quote-context"

    def _extract_quote_context(self, cloud_custom_data: str) -> Tuple[Optional[str], Optional[str]]:
        """(quote_id, quote_text) from cloud_custom_data → MessageEvent.reply_to_*."""
        try:
            parsed = json.loads(cloud_custom_data) if cloud_custom_data else None
        except (json.JSONDecodeError, TypeError):
            parsed = None
        quote = parsed.get("quote") if isinstance(parsed, dict) else None
        if not isinstance(quote, dict):
            return None, None
        quote_id = str(quote.get("id") or "").strip() or None
        desc = str(quote.get("desc") or "").strip()
        sender = str(quote.get("sender_nickname") or quote.get("sender_id") or "").strip()
        return quote_id, (f"{sender}: {desc}" if sender else desc) if desc else None

    async def _extract_media_refs_from_transcript(self, ctx: InboundContext) -> List[Tuple[str, str, str]]:
        """``(rid, kind, filename)`` for ybres anchors in the quoted transcript message; [] when
        there is no reply_to id, no store/source, or no resolvable anchors."""
        if ctx.reply_to_message_id is None:
            return []
        adapter = ctx.adapter
        media_refs: List[Tuple[str, str, str]] = []
        try:
            store = _session_store(adapter)
            if not store or ctx.source is None:
                return []
            history = store.load_transcript(store.get_or_create_session(ctx.source).session_id)
            for msg in reversed(history or []):
                mid = msg.get("message_id", "")
                if not mid or mid != ctx.reply_to_message_id:
                    continue
                _content = msg.get("content", "")
                if isinstance(_content, str) and "|ybres:" in _content:
                    media_refs.extend(_iter_ybres_refs(_YB_RES_REF_RE.finditer(_content)))
                break
        except TranscriptReadError as exc:
            # Quote resolution degrades to "no refs" rather than pretending
            # the quoted message was never seen (#100788).
            logger.warning(
                "[%s] quote transcript lookup: transcript unreadable: %s",
                getattr(adapter, "name", "yuanbao"), exc,
            )
        except Exception as exc:
            logger.warning("[%s] quote transcript lookup failed: %s", getattr(adapter, "name", "yuanbao"), exc)
        return media_refs

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        ctx.reply_to_message_id, ctx.reply_to_text = self._extract_quote_context(ctx.cloud_custom_data)
        ctx.quote_media_refs = await self._extract_media_refs_from_transcript(ctx)
        await next_fn()


class ForwardedRecordsParseMiddleware(InboundMiddleware):
    """Deep-parse WeChat forwarded chat records (elem_type 1009) on ``ctx.forwarded_records``:
    render media as ``[kind|ybres:RID]``, append refs to ``ctx.media_refs`` and rewrite raw_text.
    No run-time fallback for earlier forwards — GroupAtGuard already rendered summaries at observe
    time. On any failure raw_text is left untouched."""
    name = "forwarded-records-parse"
    FORWARD_MSG_TEXT_MAX_CHARS = 1000  # per-record text cap; record count is NOT capped

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        try:
            if ctx.forwarded_records:
                await self._send_loading_heartbeat(ctx)
                ctx.raw_text = self.build_forward_text(ctx.forwarded_records, ctx=ctx, is_dispatch=True)
        except Exception as exc:
            logger.warning("[%s] forwarded-records deep parse failed: %s", getattr(ctx.adapter, "name", "yuanbao"), exc)
        await next_fn()

    @staticmethod
    async def _send_loading_heartbeat(ctx: InboundContext) -> None:
        """Best-effort RUNNING heartbeat so the user sees a loading bubble."""
        with contextlib.suppress(Exception):
            await ctx.adapter._outbound.heartbeat.send_heartbeat_once(ctx.chat_id, WS_HEARTBEAT_RUNNING)

    @classmethod
    def _media_marker(cls, media: dict, plain_text: str = "") -> Tuple[str, Optional[Dict[str, str]]]:
        """One ``multimedia`` entry → ``(marker, ref)``: ``[kind|ybres:RID]`` + media_refs dict when a
        RID/URL is usable, else a plain ``[kind] name`` marker and ``ref=None``."""
        media_type = (media.get("type", "") or media.get("doc_type", "")).strip().lower()
        url = str(media.get("url") or "").strip()
        file_name = str(media.get("file_name") or "").strip()
        # media_id is directly usable as a ybres RID; else parse resourceId from the URL.
        rid = str(media.get("media_id") or "").strip() or ExtractContentMiddleware._parse_resource_id(url)
        kind = {"image": "image", "file": "file", "document": "file", "code": "file", "video": "video"}.get(media_type)
        if kind and url and rid:
            return f"[{kind}|ybres:{rid}] {file_name}".rstrip(), _media_ref(kind, url, file_name if kind == "file" else "")
        if kind == "image":
            return f"[image] {file_name or plain_text}".rstrip(), None
        if kind == "file":
            return f"[file] {file_name}".rstrip(), None
        if kind == "video":
            return f"[video] {file_name or url}".rstrip(), None
        if media_type == "url":  # link share (e.g. WeChat article) — keep URL for the agent
            return f"[link] {file_name or str(media.get('title') or '')} {url}".rstrip(), None
        return f"[{media_type or 'media'}] {url or file_name}".rstrip(), None

    @classmethod
    def _walk_forward_msgs(cls, forward_data: dict) -> Iterator[Tuple[str, str, List[Dict[str, str]]]]:
        """Yield ``(sender, body, refs)`` per ``ForwardMsgData['msg']`` record; body capped at
        FORWARD_MSG_TEXT_MAX_CHARS. ``refs`` keeps textual order — PatchAnchorsMiddleware relies on it."""
        for msg in (forward_data.get("msg") if isinstance(forward_data, dict) else None) or []:
            if not isinstance(msg, dict):
                continue
            plain_text = msg.get("plainText", "")
            refs: List[Dict[str, str]] = []
            parts: List[str] = []
            for mc in msg.get("msgContent", []) or []:
                if not isinstance(mc, dict):
                    continue
                mc_type = mc.get("type", 0)  # EnumMsgContentType: 1 TEXT, 2 MULTIMEDIA, 3 nested FORWARD
                if mc_type == 1:
                    parts.append(mc.get("text", ""))
                elif mc_type == 2:
                    for media in mc.get("multimedia", []) or []:
                        if isinstance(media, dict):
                            marker, ref = cls._media_marker(media, plain_text)
                            parts.append(marker)
                            if ref is not None:
                                refs.append(ref)
                elif mc_type == 3:
                    parts.append("[嵌套聊天记录]")
                elif plain_text:
                    parts.append(plain_text)
            rendered = "  ".join(p for p in parts if p) or plain_text
            if len(rendered) > cls.FORWARD_MSG_TEXT_MAX_CHARS:
                rendered = rendered[: cls.FORWARD_MSG_TEXT_MAX_CHARS] + "…(已截断)"
            yield msg.get("sender", ""), rendered, refs

    @classmethod
    def build_forward_text(cls, forward_data: dict, *, ctx: InboundContext, is_dispatch: bool) -> str:
        """Render ``ForwardMsgData`` as ``发送人：正文`` lines with media markers. When ``is_dispatch``,
        refs go to ``ctx.media_refs`` and a ``用户附言：`` footer is added (observe-time callers skip both)."""
        lines = [f"当前用户的昵称为{ctx.sender_nickname or '用户'}", "以下为用户的聊天记录"]
        for sender, body, refs in cls._walk_forward_msgs(forward_data):
            lines.append(f"{sender}：{body}")
            if is_dispatch:
                ctx.media_refs.extend(refs)
        text = "\n".join(lines)
        if is_dispatch and ctx.raw_text.strip():
            text += f"\n\n用户附言：{ctx.raw_text.strip()}"
        return text


class MediaResolveMiddleware(InboundMiddleware):
    """Resolve inbound media references to local cached files. Yuanbao COS hostnames resolve to
    private IPs (tripping vision_tools' SSRF guard), so we download ourselves and hand the model
    local paths."""
    name = "media-resolve"
    # Resource download cache keyed by resourceId: rid -> (local_path, mime, ts)
    _resource_cache: ClassVar[Dict[str, Tuple[str, str, float]]] = {}
    _RESOURCE_CACHE_TTL_S: ClassVar[int] = 24 * 60 * 60
    _RESOURCE_CACHE_MAX_SIZE: ClassVar[int] = 256

    @classmethod
    def _get_cached_resource(cls, resource_id: str) -> Optional[Tuple[str, str]]:
        """Cached ``(local_path, mime)`` if unexpired and the file still exists (cache dir may be swept)."""
        entry = cls._resource_cache.get(resource_id) if resource_id else None
        if entry is None:
            return None
        local_path, mime, ts = entry
        if time.time() - ts > cls._RESOURCE_CACHE_TTL_S or not os.path.isfile(local_path):
            cls._resource_cache.pop(resource_id, None)
            return None
        return local_path, mime

    @classmethod
    def _put_cached_resource(cls, resource_id: str, local_path: str, mime: str) -> None:
        """Cache a download result; evicts the oldest 25% when at capacity."""
        if not resource_id:
            return
        if len(cls._resource_cache) >= cls._RESOURCE_CACHE_MAX_SIZE:
            sorted_keys = sorted(cls._resource_cache, key=lambda k: cls._resource_cache[k][2])
            for k in sorted_keys[: cls._RESOURCE_CACHE_MAX_SIZE // 4]:
                cls._resource_cache.pop(k, None)
        cls._resource_cache[resource_id] = (local_path, mime, time.time())

    @classmethod
    def _append_cached_resource(cls, adapter, resource_id: str, media_paths: List[str], mimes: List[str]) -> bool:
        """Append a cached resource to the output lists; False on miss."""
        hit = cls._get_cached_resource(resource_id)
        if hit is None:
            return False
        logger.debug("[%s] resource cache hit: rid=%s path=%s", adapter.name, resource_id, hit[0])
        media_paths.append(hit[0])
        mimes.append(hit[1])
        return True

    @staticmethod
    def _guess_image_ext_from_url(url: str) -> str:
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
        return ext if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".tiff"} else ".jpg"

    @staticmethod
    async def _fetch_resource_url(adapter, resource_id: str) -> str:
        """Exchange a ``resourceId`` for a direct download URL via ``/api/resource/v1/download``,
        with a single 401-retry after token force-refresh. Raises on failure."""
        resource_id = resource_id.strip()
        if not resource_id:
            raise RuntimeError("missing resource_id")
        def _auth_headers(token_data: dict, fallback_source: str) -> Optional[dict]:
            token = str(token_data.get("token") or "").strip()
            bot_id = str(token_data.get("bot_id") or adapter._bot_id or adapter._app_key).strip()
            if not token or not bot_id:
                return None
            source = str(token_data.get("source") or fallback_source).strip() or "web"
            return {"Content-Type": "application/json", "X-ID": bot_id, "X-Token": token, "X-Source": source}
        headers = _auth_headers(await adapter._get_cached_token(), "web")
        if headers is None:
            raise RuntimeError("missing token or bot_id for resource download")
        api_url = f"{adapter._api_domain}/api/resource/v1/download"
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            for attempt in range(2):
                resp = await client.get(api_url, params={"resourceId": resource_id}, headers=headers)
                if resp.status_code == 401 and attempt == 0:
                    token_data = await SignManager.force_refresh(adapter._app_key, adapter._app_secret, adapter._api_domain)
                    headers = _auth_headers(token_data, headers["X-Source"] or "web")
                    if headers is None:
                        break
                    continue
                resp.raise_for_status()
                payload = resp.json()
                code = payload.get("code")
                if code not in {None, 0}:
                    raise RuntimeError(f"resource/v1/download failed: code={code}, msg={payload.get('msg', '')}")
                data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
                real_url = str((data or {}).get("url") or (data or {}).get("realUrl") or "").strip()
                if real_url:
                    return real_url
                raise RuntimeError("resource/v1/download missing url/realUrl")
        raise RuntimeError("resource/v1/download did not return a URL")

    @staticmethod
    async def _resolve_download_url(adapter, url: str) -> str:
        """Resolve a Yuanbao resource placeholder URL (``…/api/resource/download?resourceId=…``,
        which 401s on direct GET) to a fetchable URL via the business API; passthrough otherwise."""
        resource_id = ExtractContentMiddleware._parse_resource_id(url)
        if not resource_id:
            return url
        try:
            return await MediaResolveMiddleware._fetch_resource_url(adapter, resource_id)
        except Exception:
            return url

    @classmethod
    async def _download_and_cache(
        cls, adapter, *, fetch_url: str, kind: str,
        file_name: Optional[str] = None, log_tag: str = "", resource_id: str = "",
    ) -> Optional[Tuple[str, str]]:
        """Download a Yuanbao resource into the local media cache → ``(local_path, mime)`` or None.
        A *resource_id* is checked against the in-memory cache first."""
        if resource_id:
            hit = cls._get_cached_resource(resource_id)
            if hit is not None:
                logger.debug("[%s] resource cache hit: rid=%s path=%s", adapter.name, resource_id, hit[0])
                return hit
        try:
            file_bytes, content_type = await media_download_url(fetch_url, max_size_mb=adapter.MEDIA_MAX_SIZE_MB)
        except Exception as exc:
            logger.warning("[%s] inbound media download failed: kind=%s %s err=%s", adapter.name, kind, log_tag, exc)
            return None
        if kind == "image":
            ext = cls._guess_image_ext_from_url(fetch_url)
            try:
                local_path = await cache_image_from_bytes_async(file_bytes, ext=ext)
            except ValueError as exc:
                logger.warning("[%s] inbound image cache rejected: %s err=%s", adapter.name, log_tag, exc)
                return None
            mime = guess_mime_type(f"image{ext}")
            if not mime.startswith("image/"):
                mime = content_type if content_type.startswith("image/") else "image/jpeg"
        elif kind == "video":
            # Yuanbao video resources carry no reliable extension; default to mp4.
            local_path = await cache_video_from_bytes_async(file_bytes)
            mime = guess_mime_type(local_path) or (content_type if content_type.startswith("video/") else "video/mp4")
        else:  # file
            file_name = file_name or os.path.basename(urllib.parse.urlparse(fetch_url).path) or "file"
            try:
                local_path = await cache_document_from_bytes_async(file_bytes, file_name)
            except Exception as exc:
                logger.warning("[%s] inbound file cache failed: %s err=%s", adapter.name, log_tag, exc)
                return None
            mime = guess_mime_type(file_name) or content_type or "application/octet-stream"
        cls._put_cached_resource(resource_id, local_path, mime)
        return local_path, mime

    @classmethod
    async def _resolve_media_urls(cls, adapter, media_refs: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
        """Resolve inbound media refs → (local_paths, mime_types); same bounded-concurrency,
        order-preserving, exception-isolated contract as :meth:`_resolve_ybres_refs`."""
        media_urls: List[str] = []
        media_types: List[str] = []
        active: List[Tuple[str, str, str, str]] = []  # (kind, filename, rid, url)
        for ref in media_refs:
            kind = str(ref.get("kind") or "").strip().lower()
            url = str(ref.get("url") or "").strip()
            if kind not in _RESOLVABLE_MEDIA_KINDS or not url:
                continue
            rid = ExtractContentMiddleware._parse_resource_id(url)
            if rid and cls._append_cached_resource(adapter, rid, media_urls, media_types):
                continue
            active.append((kind, str(ref.get("name") or "").strip(), rid or "", url))
        if active:
            await cls._gather_resolve(
                adapter, active, "media", media_urls, media_types,
                get_url=lambda url: cls._resolve_download_url(adapter, url),
                fail_fmt="[%s] inbound media resolve failed: kind=%s url=%s err=%s", fail_args=lambda kind, url: (kind, url),
                crash_fmt="[%s] inbound media resolve crashed: kind=%s url=%s err=%s", crash_args=lambda kind, url: (kind, url[:80]),
                log_tag=lambda url: f"placeholder_url={url[:80]}",
            )
        return media_urls, media_types

    @classmethod
    async def _gather_resolve(cls, adapter, active, scope, out_paths, out_mimes, *,
                              get_url, fail_fmt, fail_args, crash_fmt, crash_args, log_tag) -> None:
        """Resolve ``(kind, filename, rid, key)`` items under bounded concurrency — ``await get_url(key)``
        then download+cache — appending successes in input order. ``return_exceptions=True`` isolates
        per-item failures; the batch summary line keeps stable fields (concurrency vs elapsed_ms) for
        offline aggregation."""
        semaphore = asyncio.Semaphore(adapter.media_resolve_concurrency)

        async def _one(kind: str, filename: str, rid: str, key: str) -> Optional[Tuple[str, str]]:
            async with semaphore:
                try:
                    fetch_url = await get_url(key)
                except Exception as exc:
                    logger.warning(fail_fmt, adapter.name, *fail_args(kind, key), exc)
                    return None
                return await cls._download_and_cache(
                    adapter, fetch_url=fetch_url, kind=kind, file_name=filename or None, log_tag=log_tag(key), resource_id=rid,
                )
        _t0 = time.monotonic()
        results = await asyncio.gather(*(_one(*item) for item in active), return_exceptions=True)
        _elapsed_ms = int((time.monotonic() - _t0) * 1000)
        _failed = 0
        for (kind, _filename, _rid, key), result in zip(active, results):
            if isinstance(result, BaseException):
                logger.warning(crash_fmt, adapter.name, *crash_args(kind, key), result)
            if result is None or isinstance(result, BaseException):
                _failed += 1
            else:
                out_paths.append(result[0])
                out_mimes.append(result[1])
        logger.info(
            "[%s] media resolve batch: scope=%s concurrency=%d total=%d ok=%d failed=%d elapsed_ms=%d",
            adapter.name, scope, adapter.media_resolve_concurrency, len(active), len(out_paths), _failed, _elapsed_ms,
        )

    @classmethod
    async def _resolve_ybres_refs(cls, adapter, refs: List[Tuple[str, str, str]], *, log_prefix: str) -> Tuple[List[str], List[str]]:
        """Resolve ``(rid, kind, filename)`` ybres tuples to local paths (bounded concurrency,
        input order preserved, per-rid failures isolated). Cache hits are served without a fetch."""
        media_paths: List[str] = []
        mimes: List[str] = []
        active = [(kind, filename, rid, rid) for rid, kind, filename in refs
                  if kind in _RESOLVABLE_MEDIA_KINDS and not cls._append_cached_resource(adapter, rid, media_paths, mimes)]
        if active:
            await cls._gather_resolve(
                adapter, active, "ybres", media_paths, mimes,
                get_url=lambda rid: cls._fetch_resource_url(adapter, rid),
                fail_fmt="[%s] %s resolve failed: rid=%s kind=%s err=%s", fail_args=lambda kind, rid: (log_prefix, rid, kind),
                crash_fmt="[%s] %s resolve crashed: rid=%s kind=%s err=%s", crash_args=lambda kind, rid: (log_prefix, rid, kind),
                log_tag=lambda rid: f"{log_prefix} rid={rid}",
            )
        return media_paths, mimes

    @classmethod
    async def _collect_observed_media(cls, adapter, source) -> Tuple[List[str], List[str]]:
        """Resolve recent observed image/file anchors from the transcript into ``(local_paths, mimes)``."""
        store = _session_store(adapter)
        if not store:
            return [], []
        try:
            history = store.load_transcript(store.get_or_create_session(source).session_id)
        except TranscriptReadError as exc:
            # Hydrate nothing rather than silently acting as if the session had no observed media.
            logger.warning("[%s] Observed-media hydration: transcript unreadable: %s", adapter.name, exc)
            return [], []
        except Exception as exc:
            logger.warning("[%s] Observed-media hydration setup failed: %s", adapter.name, exc)
            return [], []
        # Walk newest→oldest (matches within a message too) so the per-turn cap keeps the
        # *latest* refs; ``order`` is reversed back to chronological before resolving.
        order: List[Tuple[str, str, str]] = []  # (rid, kind, filename)
        seen: set = set()
        for msg in reversed((history or [])[-OBSERVED_MEDIA_BACKFILL_LOOKBACK:]):
            content = msg.get("content")
            if not isinstance(content, str) or "|ybres:" not in content:
                continue
            for rid, kind, filename in _iter_ybres_refs(reversed(list(_YB_RES_REF_RE.finditer(content)))):
                if rid not in seen:
                    seen.add(rid)
                    order.append((rid, kind, filename))
                if len(order) >= OBSERVED_MEDIA_BACKFILL_MAX_RESOLVE_PER_TURN:
                    break
            if len(order) >= OBSERVED_MEDIA_BACKFILL_MAX_RESOLVE_PER_TURN:
                break
        if not order:
            return [], []
        return await cls._resolve_ybres_refs(adapter, order[::-1], log_prefix="observed-media")

    @classmethod
    async def _resolve_quote_media(cls, adapter, quote_media_refs: List[Tuple[str, str, str]]) -> Tuple[List[str], List[str]]:
        """Resolve ybres anchors of the quoted message (from QuoteContextMiddleware)."""
        return await cls._resolve_ybres_refs(adapter, quote_media_refs, log_prefix="quote")

    @staticmethod
    def _collect_quote_local_media(ctx: InboundContext) -> Tuple[List[str], List[str]]:
        """DM quote fallback: ``(local_paths, mimes)`` for media PatchAnchorsMiddleware already
        rewrote to ``[image: /path]`` / ``[file: name → /path]`` on the original turn. Unresolved
        anchors were that turn's failure — no re-download here."""
        paths: List[str] = []
        mimes: List[str] = []
        cache = getattr(ctx.adapter, "_msg_content_cache", None)
        text = cache.get(ctx.reply_to_message_id) if ctx.reply_to_message_id and cache else None
        for m in _YB_LOCAL_MEDIA_RE.finditer(text if isinstance(text, str) else ""):
            kind = (m.group(1) or "").strip().lower()
            path = (m.group(2) or "").strip()
            if not path or path in paths or not os.path.exists(path):
                continue
            paths.append(path)
            mimes.append(guess_mime_type(os.path.basename(path)) or ("image/jpeg" if kind == "image" else "application/octet-stream"))
        return paths, mimes

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        # In groups only @bot / owner-command turns reach here (GroupAtGuard short-circuits the
        # rest), so media download and observed-media hydration need no @bot re-check.
        adapter = ctx.adapter
        urls: List[str] = []
        types: List[str] = []

        def _add_unique_pairs(pair_lists: Tuple[List[str], List[str]]) -> None:
            for u, m in zip(*pair_lists):
                if u and u not in urls:
                    urls.append(u)
                    types.append(m)
        own_pairs = await self._resolve_media_urls(adapter, ctx.media_refs)  # 1) media carried by this message
        own_count = sum(1 for u in own_pairs[0] if u)
        _add_unique_pairs(own_pairs)
        # 2) Quoted media takes priority; else observed-media backfill in groups only (DM media
        #    was already resolved on its own turn).
        if ctx.reply_to_message_id is not None:
            if ctx.quote_media_refs:
                _add_unique_pairs(await self._resolve_quote_media(adapter, ctx.quote_media_refs))
            else:  # DM rows carry no platform message_id → recover already-local media from the msg cache.
                _add_unique_pairs(self._collect_quote_local_media(ctx))
        elif ctx.chat_type == "group":
            try:
                _add_unique_pairs(await self._collect_observed_media(adapter, ctx.source))
            except Exception as exc:
                logger.warning("[%s] observed-image hydration raised, continuing anyway: %s", adapter.name, exc)
        ctx.media_urls = urls
        ctx.media_types = types
        # Re-check placeholder using ``own_count``: placeholder text with only quote/observed
        # media (no fresh attachment of its own) is still skippable.
        if PlaceholderFilterMiddleware.is_skippable_placeholder(ctx.raw_text, own_count):
            logger.debug("[%s] Skip placeholder after media download: %r", adapter.name, ctx.raw_text)
            return  # Stop pipeline
        await next_fn()


class PatchAnchorsMiddleware(InboundMiddleware):
    """Replace ``[kind|ybres:RID]`` anchors in raw_text with the local paths MediaResolveMiddleware
    produced, so the transcript records usable paths. Only resolved media (paths starting with
    ``/``) are substituted; other anchors stay untouched."""
    name = "patch-anchors"

    @staticmethod
    def _patch(text: str, urls: List[str], types: List[str]) -> str:
        patched = text
        for u, m in zip(urls, types):
            if not u.startswith("/"):
                continue
            anchor_match = _YB_RES_REF_RE.search(patched)
            if not anchor_match:
                break
            kind, _, filename = anchor_match.group(1).partition(":")
            kind = kind.strip()
            if kind == "image" and m.startswith("image/"):
                replacement = f"[image: {u}]"
            elif kind == "file":
                replacement = f"[file: {filename.strip() or os.path.basename(u)} → {u}]"
            elif kind == "video":
                replacement = f"[video: {u}]"
            else:
                continue
            patched = patched[: anchor_match.start()] + replacement + patched[anchor_match.end():]
        return patched

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        ctx.raw_text = self._patch(ctx.raw_text, ctx.media_urls, ctx.media_types)
        await next_fn()


class DispatchMiddleware(InboundMiddleware):
    """Build the MessageEvent and dispatch it (groups: serialised per session via a queue)."""
    name = "dispatch"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        adapter = ctx.adapter
        _sk = build_session_key(
            ctx.source,
            group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
        )

        async def _dispatch_inbound_event() -> None:
            if any(mt.startswith(("application/", "text/")) for mt in ctx.media_types):
                # Classification: DOCUMENT wins over PHOTO/VIDEO/AUDIO for mixed attachments — run.py's
                # image handling keys off the per-path image/* mime types regardless of message_type, but
                # document-context injection gates strictly on MessageType.DOCUMENT (same precedence as
                # Email/Signal, PR #44695).
                msg_type = MessageType.DOCUMENT
            else:  # yuanbao-local subtypes (CHAT_RECORD) are deep-parsed into text → TEXT downstream
                msg_type = ctx.msg_type if isinstance(ctx.msg_type, MessageType) else MessageType.TEXT
            event = MessageEvent(
                text=ctx.raw_text, message_type=msg_type, source=ctx.source, message_id=ctx.msg_id or None,
                raw_message=ctx.push, media_urls=list(ctx.media_urls), media_types=list(ctx.media_types),
                reply_to_message_id=ctx.reply_to_message_id, reply_to_text=ctx.reply_to_text,
                channel_prompt=ctx.channel_prompt,
            )
            if _sk and ctx.msg_id:
                adapter._processing_msg_ids[_sk] = ctx.msg_id
                adapter._processing_msg_texts[_sk] = ctx.raw_text or ""
            if ctx.msg_id and ctx.raw_text:
                cache = adapter._msg_content_cache
                cache[ctx.msg_id] = ctx.raw_text
                for k in list(cache)[:max(0, len(cache) - 200)]:  # bounded: drop oldest
                    del cache[k]
            await adapter.handle_message(event)
        if ctx.chat_type == "group":
            is_new = _sk not in adapter._group_queues
            queue = adapter._group_queues.setdefault(_sk, asyncio.Queue())
            queue.put_nowait(_dispatch_inbound_event)
            logger.info("[%s] Group message enqueued (qsize=%d) for %s", adapter.name, queue.qsize(), (_sk or "")[:50])
            if is_new:
                self._track_inbound(adapter, self._consume_group_queue(adapter, _sk), f"yuanbao-group-consumer-{(_sk or '')[:30]}")
        else:
            self._track_inbound(adapter, _dispatch_inbound_event(), f"yuanbao-inbound-{ctx.msg_id or 'unknown'}")
        await next_fn()

    @staticmethod
    def _track_inbound(adapter, coro, name: str) -> None:
        task = asyncio.create_task(coro, name=name)
        adapter._inbound_tasks.add(task)
        task.add_done_callback(adapter._inbound_tasks.discard)

    @staticmethod
    async def _consume_group_queue(adapter: "YuanbaoAdapter", session_key: str) -> None:
        """Drain the group queue one dispatch at a time, waiting for each to finish; exits after 2s idle."""
        queue = adapter._group_queues.get(session_key)
        if not queue:
            return
        try:
            while True:
                try:
                    dispatch_fn = await asyncio.wait_for(queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    break
                logger.debug("[%s] Group queue: dispatching for %s (remaining=%d)", adapter.name, (session_key or "")[:50], queue.qsize())
                try:
                    await dispatch_fn()
                    while session_key in adapter._active_sessions:
                        await asyncio.sleep(0.1)
                except Exception:
                    logger.exception("[%s] Group queue consumer error", adapter.name)
        finally:
            adapter._group_queues.pop(session_key, None)


class InboundPipelineBuilder:
    """Assembles the default Yuanbao inbound pipeline (order matters)."""
    _DEFAULT_MIDDLEWARES: list[type] = [
        DecodeMiddleware, ExtractFieldsMiddleware, RecallGuardMiddleware, DedupMiddleware, SkipSelfMiddleware,
        ChatRoutingMiddleware, AccessGuardMiddleware, ExtractContentMiddleware, PlaceholderFilterMiddleware,
        OwnerCommandMiddleware, BuildSourceMiddleware, GroupAtGuardMiddleware, AutoSetHomeMiddleware,
        GroupAttributionMiddleware, ClassifyMessageTypeMiddleware, QuoteContextMiddleware,
        ForwardedRecordsParseMiddleware, MediaResolveMiddleware, PatchAnchorsMiddleware, DispatchMiddleware,
    ]

    @classmethod
    def build(cls) -> InboundPipeline:
        pipeline = InboundPipeline()
        for mw_cls in cls._DEFAULT_MIDDLEWARES:
            pipeline.use(mw_cls())
        return pipeline


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
            logger.warning("[%s] %s. Run: pip install websockets", adapter.name, msg)
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
                await _cancel_task(task)
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
            await _cancel_task(task)
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
            await self._sender.send_text_chunk(chat_id, SLOW_RESPONSE_MESSAGE)
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
        if not content.startswith("Cronjob Response: "):
            return content
        divider = "\n-------------\n\n"
        footer_prefix = '\n\nTo stop or manage this job, send me a new message (e.g. "stop reminder '
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

    @classmethod
    def get_active(cls) -> Optional["YuanbaoAdapter"]:
        return cls._active_instance

    @classmethod
    def set_active(cls, adapter: Optional["YuanbaoAdapter"]) -> None:
        cls._active_instance = adapter

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
        _existing_home = os.getenv("YUANBAO_HOME_CHANNEL") or (config.home_channel.chat_id if config.home_channel else "")
        self._auto_sethome_done: bool = bool(_existing_home) and not _existing_home.startswith("group:")

    def _track_task(self, task: asyncio.Task) -> asyncio.Task:
        """Register a fire-and-forget task so it won't be GC'd prematurely."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    @property
    def enforces_own_access_policy(self) -> bool:
        """Yuanbao gates DM/group access at intake via dm_policy/group_policy."""
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
            YuanbaoAdapter.set_active(None)
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
            text = text[:self.DM_MAX_CHARS] + "\n...(truncated)"
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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

AUTH_FAILED_CODES = {4001, 4002, 4003}      # permanent auth failure, re-sign token

AUTH_RETRYABLE_CODES = {4010, 4011, 4099}   # transient, can retry with same token

class FileUrlHandler(MediaSendHandler):
    """Strategy: send file from a URL (download → COS → TIMFileElem)."""

    async def acquire_file(self, adapter, **kwargs):
        file_url: str = kwargs["file_url"]
        logger.info("[%s] FileUrlHandler: downloading %s", adapter.name, file_url)
        file_bytes, content_type = await media_download_url(
            file_url, max_size_mb=adapter.MEDIA_MAX_SIZE_MB,
        )
        filename = kwargs.get("filename")
        if not filename:
            path_part = file_url.split("?")[0]
            filename = os.path.basename(path_part) or "file"
        if not content_type or content_type == "application/octet-stream":
            content_type = guess_mime_type(filename) or "application/octet-stream"
        return file_bytes, filename, content_type

    def build_msg_body(self, upload_result, **kwargs):
        return build_file_msg_body(
            url=upload_result["url"],
            filename=kwargs["filename"],
            uuid=kwargs["file_uuid"],
            size=upload_result["size"],
        )

class GroupQueryService:
    """Encapsulates all group query operations (both low-level WS calls and
    higher-level AI-tool-facing wrappers).

    Responsibilities:
      - Low-level WS encode/decode for group info and member list queries
      - Chat-id parsing, error wrapping and result filtering for AI tools
      - Member cache population on the adapter
    """

    def __init__(self, adapter: "YuanbaoAdapter") -> None:
        self._adapter = adapter

    # ------------------------------------------------------------------
    # Low-level WS query methods
    # ------------------------------------------------------------------

    async def query_group_info_raw(self, group_code: str) -> Optional[dict]:
        """Query group info via WS (group name, owner, member count, etc.).

        Returns:
            Decoded dict or None on failure.
        """
        adapter = self._adapter
        if adapter._connection.ws is None:
            return None
        encoded = encode_query_group_info(group_code)
        from gateway.platforms.yuanbao_proto import decode_conn_msg as _decode
        decoded = _decode(encoded)
        req_id = decoded["head"]["msg_id"]
        try:
            response = await adapter._connection.send_biz_request(encoded, req_id=req_id)
            head = response.get("head", {})
            status = head.get("status", 0)
            if status != 0:
                logger.warning("[%s] query_group_info failed: status=%d", adapter.name, status)
                return None
            biz_data = response.get("data", b"") or response.get("body", b"")
            if biz_data and isinstance(biz_data, bytes):
                return decode_query_group_info_rsp(biz_data)
            return {"group_code": group_code}
        except asyncio.TimeoutError:
            logger.warning("[%s] query_group_info timeout: group=%s", adapter.name, group_code)
            return None
        except Exception as exc:
            logger.warning("[%s] query_group_info failed: %s", adapter.name, exc)
            return None

    async def get_group_member_list_raw(
        self, group_code: str, offset: int = 0, limit: int = 200
    ) -> Optional[dict]:
        """Query group member list via WS.

        Returns:
            Decoded dict or None on failure.  Also populates adapter._member_cache.
        """
        adapter = self._adapter
        if adapter._connection.ws is None:
            return None
        encoded = encode_get_group_member_list(group_code, offset=offset, limit=limit)
        from gateway.platforms.yuanbao_proto import decode_conn_msg as _decode
        decoded = _decode(encoded)
        req_id = decoded["head"]["msg_id"]
        try:
            response = await adapter._connection.send_biz_request(encoded, req_id=req_id)
            head = response.get("head", {})
            status = head.get("status", 0)
            if status != 0:
                logger.warning("[%s] get_group_member_list failed: status=%d", adapter.name, status)
                return None
            biz_data = response.get("data", b"") or response.get("body", b"")
            if biz_data and isinstance(biz_data, bytes):
                result = decode_get_group_member_list_rsp(biz_data)
            else:
                result = {"members": [], "next_offset": 0, "is_complete": True}
            if result and result.get("members"):
                adapter._member_cache[group_code] = (time.time(), result["members"])
            return result
        except asyncio.TimeoutError:
            logger.warning("[%s] get_group_member_list timeout: group=%s", adapter.name, group_code)
            return None
        except Exception as exc:
            logger.warning("[%s] get_group_member_list failed: %s", adapter.name, exc)
            return None

    # ------------------------------------------------------------------
    # AI-tool-facing wrappers (chat_id parsing + filtering)
    # ------------------------------------------------------------------

    async def query_group_info(self, chat_id: str) -> dict:
        """AI tool: Query current group info.

        No parameters needed (group_code extracted from session context).
        Returns group name, owner, member count, etc.
        """
        if not chat_id.startswith("group:"):
            return {"error": "This command is only available in group chats"}
        group_code = chat_id[len("group:"):]
        result = await self.query_group_info_raw(group_code)
        if result is None:
            return {"error": "Failed to query group info"}
        return result

    async def query_session_members(
        self,
        chat_id: str,
        action: str = "list_all",
        name: Optional[str] = None,
    ) -> dict:
        """AI tool: Query group member list.

        Args:
            chat_id: Chat ID (extracted from session context)
            action: 'find' (search by name) | 'list_bots' (list bots) | 'list_all' (list all)
            name: Search keyword when action='find'

        Returns:
            {"members": [...], "total": int, "mentionHint": str}
        """
        if not chat_id.startswith("group:"):
            return {"error": "This command is only available in group chats"}
        group_code = chat_id[len("group:"):]
        result = await self.get_group_member_list_raw(group_code)
        if result is None:
            return {"error": "Failed to query group members"}

        members = result.get("members", [])

        if action == "find" and name:
            query = name.lower()
            members = [
                m for m in members
                if query in (m.get("nickname", "") or "").lower()
                or query in (m.get("name_card", "") or "").lower()
                or query in (m.get("user_id", "") or "").lower()
            ]
        elif action == "list_bots":
            members = [m for m in members if "bot" in (m.get("nickname", "") or "").lower()]

        # Construct mentionHint
        mention_hint = ""
        if members and len(members) <= 10:
            names = [m.get("name_card") or m.get("nickname") or m.get("user_id", "") for m in members]
            mention_hint = "Mention with @name: " + ", ".join(names)

        return {
            "members": members[:50],  # Limit return count
            "total": len(members),
            "mentionHint": mention_hint,
        }

REPLY_REF_TTL_S = 300.0            # Reference dedup TTL (5 minutes)

def get_active_adapter() -> Optional["YuanbaoAdapter"]:
    """Delegate to ``YuanbaoAdapter.get_active()``."""
    return YuanbaoAdapter.get_active()

async def send_yuanbao_direct(
    adapter: "YuanbaoAdapter",
    chat_id: str,
    message: str,
    media_files: Optional[List[Tuple[str, bool]]] = None,
) -> Dict[str, Any]:
    """Delegate to ``OutboundManager.send_direct``."""
    return await adapter._outbound.sender.send_direct(chat_id, message, media_files)
# ---- END PLUGIN-COMPAT ----
