"""WeCom (Enterprise WeChat) AI Bot adapter over the ``openws`` WebSocket gateway.
Streaming lives in ``streaming.py``, media in ``media.py``, per-chat send queue in ``send_queue.py``.
Config (``platforms.wecom.extra``): ``bot_id``/``secret`` (or WECOM_BOT_ID / WECOM_SECRET), ``websocket_url``,
``dm_policy``/``group_policy`` (open|allowlist|disabled|pairing), ``allow_from``, ``group_allow_from``,
``groups: {<group_id>: {allow_from: [...]}}``."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]
try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]
AIOHTTP_AVAILABLE = aiohttp is not None
HTTPX_AVAILABLE = httpx is not None

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import gateway_trust_env, BasePlatformAdapter, MessageEvent, MessageType, SendResult
from utils import env_float

from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret
from plugins.platforms.wecom.send_queue import ChatSendQueueMixin
from plugins.platforms.wecom.media import WeComMediaMixin, APP_CMD_SEND
from plugins.platforms.wecom.streaming import (
    WeComStreamMixin, ReplyQueue, StreamTurn, APP_CMD_RESPONSE,
    STREAM_NOT_SUBSCRIBED_ERRCODE, MAX_STREAM_CONTENT_LENGTH,
    STREAM_SAFE_DURATION_SECONDS, STREAM_KEEPALIVE_INTERVAL_SECONDS, STREAM_KEEPALIVE_ENABLED_DEFAULT,
)


logger = logging.getLogger(__name__)

DEFAULT_WS_URL = "wss://openws.work.weixin.qq.com"

APP_CMD_SUBSCRIBE = "aibot_subscribe"
APP_CMD_CALLBACK = "aibot_msg_callback"
APP_CMD_LEGACY_CALLBACK = "aibot_callback"
APP_CMD_EVENT_CALLBACK = "aibot_event_callback"
APP_CMD_PING = "ping"

CALLBACK_COMMANDS = {APP_CMD_CALLBACK, APP_CMD_LEGACY_CALLBACK}
NON_RESPONSE_COMMANDS = CALLBACK_COMMANDS | {APP_CMD_EVENT_CALLBACK}

MAX_MESSAGE_LENGTH = 4000
CONNECT_TIMEOUT_SECONDS = 20.0
REQUEST_TIMEOUT_SECONDS = 15.0
HEARTBEAT_INTERVAL_SECONDS = 30.0
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]

DEDUP_MAX_SIZE = 1000


def check_wecom_requirements() -> bool:
    return AIOHTTP_AVAILABLE and HTTPX_AVAILABLE


def _coerce_list(value: Any) -> List[str]:
    """Coerce config values (None | "a, b" | iterable | scalar) into a trimmed, non-empty string list."""
    if isinstance(value, str):
        value = value.split(",")
    elif not isinstance(value, (list, tuple, set)):
        value = [] if value is None else [value]
    return [item for item in (str(item).strip() for item in value) if item]


def _normalize_entry(raw: str) -> str:
    """Normalize allowlist entries such as ``wecom:user:foo``."""
    value = re.sub(r"^wecom:", "", str(raw).strip(), flags=re.IGNORECASE)
    return re.sub(r"^(user|group):", "", value, flags=re.IGNORECASE).strip()


def _entry_matches(entries: List[str], target: str) -> bool:
    """Case-insensitive allowlist match with ``*`` support."""
    normalized_target = str(target).strip().lower()
    return any(_normalize_entry(e).lower() in ("*", normalized_target) for e in entries)


def _dict_or_empty(container: Dict[str, Any], key: str) -> Dict[str, Any]:
    return container.get(key) if isinstance(container.get(key), dict) else {}


def _content_of(container: Dict[str, Any], key: str) -> str:
    return str(_dict_or_empty(container, key).get("content") or "").strip()


def _bounded_put(store: Dict[str, str], key: str, value: str) -> bool:
    """Insert into an insertion-ordered dict bounded at DEDUP_MAX_SIZE; False if key/value empty."""
    key = str(key or "").strip()
    value = str(value or "").strip()
    if not key or not value:
        return False
    store[key] = value
    while len(store) > DEDUP_MAX_SIZE:
        store.pop(next(iter(store)))
    return True


class WeComAdapter(WeComStreamMixin, WeComMediaMixin, ChatSendQueueMixin, BasePlatformAdapter):
    """WeCom AI Bot adapter backed by a persistent WebSocket connection."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    SUPPORTS_MESSAGE_EDITING = False
    SUPPORTS_NATIVE_STREAMING = True  # msgtype "stream" via aibot_respond_msg, not edit-based
    MAX_STREAM_CONTENT_LENGTH = MAX_STREAM_CONTENT_LENGTH
    _SPLIT_THRESHOLD = 3900  # chunks near the 4000-char client split are almost certainly continued

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WECOM)
        extra = config.extra or {}

        def _extra_float(key: str, default: float) -> float:
            try:
                return float(extra.get(key, default))
            except (TypeError, ValueError):
                return default

        def _setting(*keys: str, env: str = "", default: str = "") -> str:
            return str(next((extra[k] for k in keys if extra.get(k)), None) or (_get_scoped_secret(env, default) if env else "")).strip()

        self._bot_id = _setting("bot_id", env="WECOM_BOT_ID")
        self._secret = _setting("secret", env="WECOM_SECRET")
        self._ws_url = _setting("websocket_url", "websocketUrl", env="WECOM_WEBSOCKET_URL", default=DEFAULT_WS_URL) or DEFAULT_WS_URL
        self._dm_policy = _setting("dm_policy", env="WECOM_DM_POLICY", default="pairing").lower()
        # WECOM_ALLOWED_USERS fallback: env-only allowlist setups otherwise drop every DM at intake.
        self._allow_from = _coerce_list(extra.get("allow_from") or extra.get("allowFrom") or _get_scoped_secret("WECOM_ALLOWED_USERS", ""))
        self._group_policy = _setting("group_policy", env="WECOM_GROUP_POLICY", default="pairing").lower()
        self._group_allow_from = _coerce_list(extra.get("group_allow_from") or extra.get("groupAllowFrom"))
        self._groups = extra.get("groups") if isinstance(extra.get("groups"), dict) else {}
        self._session = self._ws = self._http_client = self._listen_task = self._heartbeat_task = None
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._reply_queues: Dict[str, ReplyQueue] = {}
        self._dedup, self._reply_req_ids = MessageDeduplicator(max_size=DEDUP_MAX_SIZE), {}
        # Text batching (clients split long messages ~4000 chars); attachment-only frames are held
        # for the merge window so the trailing text callback joins the same event (official: 800ms).
        self._text_batch_delay_seconds = env_float("HERMES_WECOM_TEXT_BATCH_DELAY_SECONDS", 0.6)
        self._text_batch_split_delay_seconds = env_float("HERMES_WECOM_TEXT_BATCH_SPLIT_DELAY_SECONDS", 2.0)
        self._attachment_text_merge_delay_seconds = _extra_float("attachment_text_merge_delay_seconds", 0.8)
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}
        # Stream keep-alive config (see streaming.py STREAM_* constants).
        self._stream_safe_duration_seconds = _extra_float("stream_safe_duration_seconds", STREAM_SAFE_DURATION_SECONDS)
        self._stream_keepalive_enabled = bool(extra.get("stream_keepalive_enabled", STREAM_KEEPALIVE_ENABLED_DEFAULT))
        self._stream_keepalive_interval_seconds = _extra_float("stream_keepalive_interval_seconds", STREAM_KEEPALIVE_INTERVAL_SECONDS)
        self._device_id = uuid.uuid4().hex
        self._last_chat_req_ids: Dict[str, str] = {}
        # Turns keyed f"{chat_id}:{req_id|turn_id}"; expired chats clear on the next inbound req_id.
        self._stream_turns: Dict[str, StreamTurn] = {}
        self._stream_expired_chats, self._group_chat_ids = set(), set()  # groups can't receive proactive APP_CMD_SEND
        # Per-chat FIFO send queues (normal + control lanes) + token buckets — see send_queue.py.
        self._chat_queues, self._chat_workers, self._control_queues, self._control_workers, self._chat_token_usage = {}, {}, {}, {}, {}

    def _startup_failure(self, code: str, message: str, log_msg: str, *args: Any) -> bool:
        self._set_fatal_error(code, message, retryable=True)
        logger.warning(log_msg, self.name, message, *args)
        return False

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        for available, dep in ((AIOHTTP_AVAILABLE, "aiohttp"), (HTTPX_AVAILABLE, "httpx")):
            if not available:
                return self._startup_failure("wecom_missing_dependency", f"WeCom startup failed: {dep} not installed", "[%s] %s. Run: pip install %s", dep)
        if not self._bot_id or not self._secret:
            return self._startup_failure("wecom_missing_credentials", "WeCom startup failed: WECOM_BOT_ID and WECOM_SECRET are required", "[%s] %s")
        try:
            # Tighter keepalive so idle CLOSE_WAIT drains promptly.
            # See #18451.
            from gateway.platforms._http_client_limits import platform_httpx_limits
            from gateway.platforms.base import _ssrf_redirect_guard
            from tools.url_safety import create_ssrf_safe_async_client
            self._http_client = create_ssrf_safe_async_client(timeout=30.0, follow_redirects=True, event_hooks={"response": [_ssrf_redirect_guard]}, limits=platform_httpx_limits())
            await self._open_connection()
            self._mark_connected()
            self._listen_task, self._heartbeat_task = asyncio.create_task(self._listen_loop()), asyncio.create_task(self._heartbeat_loop())
            logger.info("[%s] Connected to %s", self.name, self._ws_url)
            self._wire_plugin_handlers(None)  # ctx.register_platform_handler hooks
            return True
        except Exception as exc:
            self._set_fatal_error("wecom_connect_error", f"WeCom startup failed: {exc}", retryable=True)
            logger.error("[%s] Failed to connect: %s", self.name, exc, exc_info=True)
            await self._teardown()
            return False

    async def disconnect(self) -> None:
        self._running = False
        self._mark_disconnected()
        for task in list(self._chat_workers.values()) + list(self._control_workers.values()):
            task.cancel()
        for registry in (self._chat_workers, self._control_workers, self._chat_queues, self._control_queues):
            registry.clear()
        for attr in ("_listen_task", "_heartbeat_task"):
            task = getattr(self, attr)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            setattr(self, attr, None)
        self._fail_all(RuntimeError("WeCom adapter disconnected"))
        await self._teardown()
        self._dedup.clear()
        logger.info("[%s] Disconnected", self.name)

    def _fail_all(self, exc: Exception) -> None:
        self._fail_pending_responses(exc)
        self._fail_reply_queues(exc)

    async def _cleanup_ws(self) -> None:
        """Close the live websocket, then its session, if any."""
        for attr in ("_ws", "_session"):
            live = getattr(self, attr)
            if live and not live.closed:
                await live.close()
            setattr(self, attr, None)

    async def _teardown(self) -> None:
        """_cleanup_ws, then close the httpx client."""
        await self._cleanup_ws()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def _open_connection(self) -> None:
        await self._cleanup_ws()
        # certifi's CA bundle so aiohttp trusts the same roots as urllib/requests (macOS stale OpenSSL path).
        import ssl as _ssl
        try:
            import certifi
            cafile = certifi.where()
        except ImportError:
            cafile = None
        _ssl_ctx = _ssl.create_default_context(cafile=cafile)
        self._session = aiohttp.ClientSession(trust_env=gateway_trust_env(), connector=aiohttp.TCPConnector(ssl=_ssl_ctx))
        self._ws = await self._session.ws_connect(self._ws_url, heartbeat=HEARTBEAT_INTERVAL_SECONDS * 2, timeout=CONNECT_TIMEOUT_SECONDS)
        req_id = self._new_req_id("subscribe")
        await self._send_json({"cmd": APP_CMD_SUBSCRIBE, "headers": {"req_id": req_id}, "body": {"bot_id": self._bot_id, "secret": self._secret, "device_id": self._device_id}})
        auth_payload = await self._wait_for_handshake(req_id)
        errcode = auth_payload.get("errcode", 0)
        if errcode not in {0, None}:
            raise RuntimeError(f"{auth_payload.get('errmsg', 'authentication failed')} (errcode={errcode})")

    async def _wait_for_handshake(self, req_id: str) -> Dict[str, Any]:
        if not self._ws:
            raise RuntimeError("WebSocket not initialized")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + CONNECT_TIMEOUT_SECONDS
        while (remaining := deadline - loop.time()) > 0:
            msg = await asyncio.wait_for(self._ws.receive(), timeout=remaining)
            if msg.type == aiohttp.WSMsgType.TEXT:
                payload = self._parse_json(msg.data)
                if not payload or payload.get("cmd") == APP_CMD_PING:
                    continue
                if self._payload_req_id(payload) == req_id:
                    return payload
                logger.debug("[%s] Ignoring pre-auth payload: %s", self.name, payload.get("cmd"))
            elif msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR}:
                raise RuntimeError("WeCom websocket closed during authentication")
        raise TimeoutError("Timed out waiting for WeCom subscribe acknowledgement")

    async def _listen_loop(self) -> None:
        backoff_idx = 0
        while self._running:
            try:
                await self._read_events()
                backoff_idx = 0
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if not self._running:
                    return
                logger.warning("[%s] WebSocket error: %s", self.name, exc)
                self._fail_all(RuntimeError("WeCom connection interrupted"))
                await asyncio.sleep(RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)])
                backoff_idx += 1
                try:
                    await self._open_connection()
                    backoff_idx = 0
                    self._mark_connected()
                    logger.info("[%s] Reconnected", self.name)
                except Exception as reconnect_exc:
                    logger.warning("[%s] Reconnect failed: %s", self.name, reconnect_exc)

    async def _read_events(self) -> None:
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        while self._running and self._ws and not self._ws.closed:
            msg = await self._ws.receive()
            if msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                await self._handle_frame(msg.data, msg.type == aiohttp.WSMsgType.BINARY)
            elif msg.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSING}:
                raise RuntimeError("WeCom websocket closed")
            else:
                logger.info("[%s] Inbound frame ignored: WSMsgType=%s", self.name, msg.type)

    async def _handle_frame(self, data: Any, is_binary: bool) -> None:
        """Parse one TEXT/BINARY frame and dispatch it; every drop is logged at INFO."""
        data_len = len(data) if isinstance(data, (str, bytes, bytearray)) else -1
        if is_binary:  # WeCom should send TEXT; log a preview so an unhandled transport isn't silently dropped
            decoded = data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else "<undecodable>"
            logger.info("[%s] Inbound BINARY frame received (len=%d) head=%r — attempting JSON parse", self.name, data_len, decoded[:200])
        payload = self._parse_json(data)
        if payload:
            await self._dispatch_payload(payload)
        elif is_binary:
            logger.info("[%s] BINARY frame not parseable as JSON — dropped", self.name)
        else:  # _parse_json logged the detail; make the DROP itself visible at INFO
            logger.info("[%s] Inbound TEXT frame dropped (unparseable/non-dict) len=%d", self.name, data_len)

    async def _heartbeat_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
                try:
                    if self._ws and not self._ws.closed:
                        await self._send_json({"cmd": APP_CMD_PING, "headers": {"req_id": self._new_req_id("ping")}, "body": {}})
                except Exception as exc:
                    logger.debug("[%s] Heartbeat send failed: %s", self.name, exc)
        except asyncio.CancelledError:
            pass

    async def _dispatch_payload(self, payload: Dict[str, Any]) -> None:
        req_id = self._payload_req_id(payload)
        cmd = str(payload.get("cmd") or "")
        body_dict = payload.get("body") if isinstance(payload.get("body"), dict) else None
        if self._reply_queues and cmd != APP_CMD_PING:
            logger.debug("[%s] _dispatch_payload[ALL]: req_id=%s cmd=%r active_queues=%s", self.name, req_id or "(none)", cmd or "(empty)", list(self._reply_queues.keys()))
        if req_id and self._reply_queues.get(req_id):
            logger.debug(
                "[%s] _dispatch_payload: req_id=%s cmd=%r has_pending_ack=%s errcode=%s in_NON_RESPONSE=%s payload_keys=%s", self.name, req_id, cmd,
                self._reply_queues[req_id].pending_ack is not None, body_dict.get("errcode", "N/A") if body_dict is not None else "N/A", cmd in NON_RESPONSE_COMMANDS, list(payload.keys()),
            )
        # Reply-queue acks (inbound req_id, no/other cmd) MUST win over _pending_responses.
        if req_id and cmd not in NON_RESPONSE_COMMANDS:
            if self._resolve_reply_ack(req_id, payload):
                return
            if req_id in self._pending_responses:
                future = self._pending_responses[req_id]
                if future and not future.done():
                    future.set_result(payload)
                return
        if cmd in CALLBACK_COMMANDS:
            await self._on_message(payload)
        elif cmd == APP_CMD_EVENT_CALLBACK:
            # Kicked by server (another connection exists): suppress reconnect like the official SDK.
            if str((payload.get("body") or {}).get("event_type") or "") == "disconnected_event":
                logger.warning("[%s] Kicked by server (another WS connection established). Suppressing reconnect to avoid mutual kicking. Check for duplicate gateway instances.", self.name)
                self._running = False
        elif cmd != APP_CMD_PING:
            logger.info("[%s] Unrouted websocket payload dropped: cmd=%r req_id=%s body_keys=%s", self.name, cmd or "(empty)", req_id or "(none)", list(body_dict.keys()) if body_dict is not None else None)

    def _fail_pending_responses(self, exc: Exception) -> None:
        for req_id, future in list(self._pending_responses.items()):
            if not future.done():
                future.set_exception(exc)
            self._pending_responses.pop(req_id, None)

    def _require_ws(self) -> None:
        if not self._ws or self._ws.closed:
            raise RuntimeError("WeCom websocket is not connected")

    async def _send_json(self, payload: Dict[str, Any]) -> None:
        self._require_ws()
        await self._ws.send_json(payload)

    async def _request(self, cmd: str, req_id: str, body: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        future = self._pending_responses[req_id] = asyncio.get_running_loop().create_future()
        try:
            await self._send_json({"cmd": cmd, "headers": {"req_id": req_id}, "body": body})
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending_responses.pop(req_id, None)

    async def _send_request(self, cmd: str, body: Dict[str, Any], timeout: float = REQUEST_TIMEOUT_SECONDS) -> Dict[str, Any]:
        self._require_ws()
        return await self._request(cmd, self._new_req_id(cmd), body, timeout)

    async def _send_reply_request(self, reply_req_id: str, body: Dict[str, Any], cmd: str = APP_CMD_RESPONSE, timeout: float = REQUEST_TIMEOUT_SECONDS) -> Dict[str, Any]:
        """Send a reply frame correlated to an inbound callback req_id."""
        self._require_ws()
        return await self._request(cmd, self._require_reply_req_id(reply_req_id), body, timeout)

    @staticmethod
    def _require_reply_req_id(reply_req_id: str) -> str:
        normalized = str(reply_req_id or "").strip()
        if not normalized:
            raise ValueError("reply_req_id is required")
        return normalized

    @staticmethod
    def _new_req_id(prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"

    @staticmethod
    def _payload_req_id(payload: Dict[str, Any]) -> str:
        headers = payload.get("headers")
        return str(headers.get("req_id") or "") if isinstance(headers, dict) else ""

    @staticmethod
    def _parse_json(raw: Any) -> Optional[Dict[str, Any]]:
        raw_len = len(raw) if isinstance(raw, (str, bytes)) else -1
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # WeCom sometimes sends raw control chars inside JSON strings; strict=False accepts them.
            try:
                text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
                payload = json.JSONDecoder(strict=False).decode(text)
                logger.info("WeCom payload required strict=False fallback (len=%d)", raw_len)
            except Exception as exc2:
                tail = raw[-100:] if isinstance(raw, (str, bytes)) and len(raw) > 100 else raw
                logger.warning("Failed to parse WeCom payload (strict=False also failed): error=%s len=%d tail=%r", exc2, raw_len, tail)
                return None
        except Exception as exc:
            logger.warning("Failed to parse WeCom payload: error=%s len=%d", exc, raw_len)
            return None
        return payload if isinstance(payload, dict) else None

    async def _on_message(self, payload: Dict[str, Any]) -> None:
        body = payload.get("body")
        if not isinstance(body, dict):
            return
        req_id = self._payload_req_id(payload)
        msg_id = str(body.get("msgid") or req_id or uuid.uuid4().hex)
        sender = _dict_or_empty(body, "from")
        sender_id = str(sender.get("userid") or "").strip()
        if self._dedup.is_duplicate(msg_id):
            # INFO: a msgid redelivered after a processing exception is dropped for the TTL.
            logger.info("[%s] Duplicate message %s ignored (dedup drop) req_id=%s sender=%r chattype=%r", self.name, msg_id, req_id, sender.get("userid") if sender else None, body.get("chattype"))
            return
        _bounded_put(self._reply_req_ids, msg_id, req_id)
        chat_id = str(body.get("chatid") or sender_id).strip()
        logger.info("[%s] Inbound callback: chattype=%r chatid=%r sender=%r msgtype=%r has_chatid=%s", self.name, body.get("chattype"), body.get("chatid"), sender_id, body.get("msgtype"), bool(body.get("chatid")))
        if not chat_id:
            logger.info("[%s] Missing chat id, skipping message; body_keys=%s", self.name, list(body.keys()))
            return
        is_group = str(body.get("chattype") or "").lower() == "group"
        if not self._admit_inbound(is_group, chat_id, sender_id):
            return
        # Post-policy: cache req_id so sends can fall back to passive reply (required in groups).
        self._remember_chat_req_id(chat_id, req_id)
        text, reply_text = self._extract_text(body)
        if is_group and text:
            text = re.sub(r"^@\S+\s*", "", text).strip()  # "@Bot /approve" -> "/approve"
        media_urls, media_types = await self._extract_media(body)
        message_type = self._derive_message_type(body, text, media_types)
        has_reply_context = bool(reply_text and (text or media_urls))
        if reply_text and not has_reply_context:  # quote-only message: the quote becomes the text
            text = reply_text
        if not text and not media_urls:
            logger.info("[%s] Empty WeCom message skipped: is_group=%s chat=%s msgtype=%r", self.name, is_group, chat_id, body.get("msgtype"))
            return
        source = self.build_source(chat_id=chat_id, chat_type="group" if is_group else "dm", user_id=sender_id or None, user_name=sender_id or None)
        event = MessageEvent(
            text=text, message_type=message_type, source=source, raw_message=payload, message_id=msg_id, media_urls=media_urls, media_types=media_types,
            reply_to_message_id=f"quote:{msg_id}" if has_reply_context else None, reply_to_text=reply_text if has_reply_context else None, timestamp=datetime.now(tz=timezone.utc),
        )
        # Only plain text is batched, EXCEPT attachment-only messages, which are held so the
        # trailing text callback merges instead of "interrupting" a run the attachment spawned.
        has_pending_batch = self._text_batch_key(event) in self._pending_text_batches
        is_attachment_only = bool(media_urls) and not (text or "").strip()
        if (message_type == MessageType.TEXT and (self._text_batch_delay_seconds > 0 or has_pending_batch)) or (is_attachment_only and self._attachment_text_merge_delay_seconds > 0):
            self._enqueue_text_event(event)
        else:
            await self.handle_message(event)

    def _admit_inbound(self, is_group: bool, chat_id: str, sender_id: str) -> bool:
        """Apply group_policy / dm_policy at intake; logs and returns False when dropped."""
        if not is_group:
            allowed = self._is_dm_intake_allowed(sender_id)
            if not allowed:
                logger.info("[%s] DM sender %s blocked by policy", self.name, sender_id)
            return allowed
        self._group_chat_ids.add(chat_id)
        allowed = self._is_group_allowed(chat_id, sender_id)
        if not allowed:
            logger.info(
                "[%s] Group message DROPPED by policy: chat=%s sender=%s group_policy=%r (set group_policy to 'open' or add to group_allow_from to receive)",
                self.name, chat_id, sender_id, self._group_policy,
            )
        return allowed

    def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Buffer + reset the flush timer; real text joining a buffered attachment promotes it to TEXT and inherits the quote context."""
        existing = self._pending_text_batches.get(self._text_batch_key(event))
        super()._enqueue_text_event(event)  # merge text/media + restart the flush timer
        if existing is not None and event.text and event.text.strip():
            existing.message_type = MessageType.TEXT
            if event.reply_to_text and not existing.reply_to_text:
                existing.reply_to_text = event.reply_to_text
                existing.reply_to_message_id = event.reply_to_message_id

    async def _flush_text_batch(self, key: str) -> None:
        current_task = asyncio.current_task()
        try:
            pending = self._pending_text_batches.get(key)
            if pending and pending.media_urls and not (pending.text or "").strip():
                delay = self._attachment_text_merge_delay_seconds  # attachment-only: wait for text
            elif pending and getattr(pending, "_last_chunk_len", 0) >= self._SPLIT_THRESHOLD:
                delay = self._text_batch_split_delay_seconds  # continuation almost certain
            else:
                delay = self._text_batch_delay_seconds
            await asyncio.sleep(delay)
            # Cancel-delivery race: CancelledError lands at the NEXT await, so this identity check
            # must stay synchronous (no await between it and the pop).
            if self._pending_text_batch_tasks.get(key) is not current_task:
                return
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            logger.info("[WeCom] Flushing batch %s (%d chars, %d media)", key, len(event.text or ""), len(event.media_urls or []))
            await self.handle_message(event)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    @staticmethod
    def _extract_text(body: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        msgtype = str(body.get("msgtype") or "").lower()
        if msgtype == "mixed":
            items = _dict_or_empty(body, "mixed").get("msg_item")
            text_parts = [_content_of(item, "text") for item in (items if isinstance(items, list) else []) if isinstance(item, dict) and str(item.get("msgtype") or "").lower() == "text"]
        else:  # voice transcript / appmsg attachment title (filename) follow the text; empties drop below
            text_parts = [
                _content_of(body, "text"), _content_of(body, "voice") if msgtype == "voice" else "",
                str(_dict_or_empty(body, "appmsg").get("title") or "").strip() if msgtype == "appmsg" else "",
            ]
        quote = _dict_or_empty(body, "quote")
        quote_type = str(quote.get("msgtype") or "").lower()
        reply_text = _content_of(quote, quote_type) or None if quote_type in ("text", "voice") else None
        return "\n".join(part for part in text_parts if part).strip(), reply_text

    @staticmethod
    def _derive_message_type(body: Dict[str, Any], text: str, media_types: List[str]) -> MessageType:
        if any(mtype.startswith(("application/", "text/")) for mtype in media_types):
            return MessageType.DOCUMENT
        if any(mtype.startswith("image/") for mtype in media_types):
            return MessageType.TEXT if text else MessageType.PHOTO
        if str(body.get("msgtype") or "").lower() == "voice":
            return MessageType.VOICE
        return MessageType.TEXT

    @property
    def enforces_own_access_policy(self) -> bool:
        """WeCom gates DM/group access at intake via dm_policy/group_policy."""
        return True

    def _open_dm_opted_in(self) -> bool:
        # Scoped reads: default profile's allow-all flag must not leak into a multiplexed profile.
        return any((_get_scoped_secret(var, "") or "").lower() in {"true", "1", "yes"} for var in ("GATEWAY_ALLOW_ALL_USERS", "WECOM_ALLOW_ALL_USERS"))

    def _is_dm_allowed(self, sender_id: str) -> bool:
        if self._dm_policy == "allowlist":
            return _entry_matches(self._allow_from, sender_id)
        return self._dm_policy == "open" and self._open_dm_opted_in()

    def _is_dm_intake_allowed(self, sender_id: str) -> bool:
        principal = str(sender_id or "").strip()
        return bool(principal) and (self._dm_policy == "pairing" or self._is_dm_allowed(principal))

    def _is_group_allowed(self, chat_id: str, sender_id: str) -> bool:
        if self._group_policy in ("disabled", "pairing") or (self._group_policy == "allowlist" and not _entry_matches(self._group_allow_from, chat_id)):
            return False
        group_cfg = self._resolve_group_cfg(chat_id)
        sender_allow = _coerce_list(group_cfg.get("allow_from") or group_cfg.get("allowFrom"))
        return _entry_matches(sender_allow, sender_id) if sender_allow else True

    def _resolve_group_cfg(self, chat_id: str) -> Dict[str, Any]:
        """Exact key, then case-insensitive key, then ``"*"``; only dict values count."""
        if not isinstance(self._groups, dict):
            return {}
        lowered = chat_id.lower()
        candidates = (self._groups.get(chat_id), next((v for k, v in self._groups.items() if isinstance(k, str) and k.lower() == lowered and isinstance(v, dict)), None), self._groups.get("*"))
        return next((c for c in candidates if isinstance(c, dict)), {})

    def _remember_chat_req_id(self, chat_id: str, req_id: str) -> None:
        """Cache the chat's latest inbound req_id; a fresh one also resurrects its stream channel."""
        if _bounded_put(self._last_chat_req_ids, chat_id, req_id):
            self._stream_expired_chats.discard(str(chat_id).strip())

    def _reply_req_id_for_message(self, reply_to: Optional[str]) -> Optional[str]:
        normalized = str(reply_to or "").strip()
        return None if not normalized or normalized.startswith("quote:") else self._reply_req_ids.get(normalized)

    def _cached_reply_req_id(self, chat_id: str, reply_to: Optional[str]) -> Optional[str]:
        """Explicit reply_to mapping, else the chat's last inbound req_id."""
        return self._reply_req_id_for_message(reply_to) or self._last_chat_req_ids.get(chat_id)

    async def _force_reconnect_on_stale_subscription(self, errcode: int) -> None:
        """On 846609 (subscription lost) drop req_ids bound to the dead session. Do NOT close the
        WS: a second connection gets kicked and invalidates the first (infinite kick loop)."""
        if errcode != STREAM_NOT_SUBSCRIBED_ERRCODE:
            return
        logger.warning("[%s] Got errcode %d (subscription lost) — clearing stale state", self.name, errcode)
        self._last_chat_req_ids.clear()
        self._reply_req_ids.clear()

    @staticmethod
    def _response_error(response: Dict[str, Any]) -> Optional[str]:
        errcode = response.get("errcode", 0)
        return None if errcode in {0, None} else f"WeCom errcode {errcode}: {response.get('errmsg') or 'unknown error'}"

    @classmethod
    def _raise_for_wecom_error(cls, response: Dict[str, Any], operation: str) -> None:
        error = cls._response_error(response)
        if error:
            raise RuntimeError(f"{operation} failed: {error}")

    def _markdown_body(self, content: str) -> Dict[str, Any]:
        return {"msgtype": "markdown", "markdown": {"content": content[:self.MAX_MESSAGE_LENGTH]}}

    async def _send_reply_markdown(self, reply_req_id: str, content: str) -> Dict[str, Any]:
        response = await self._send_reply_request(reply_req_id, self._markdown_body(content))
        self._raise_for_wecom_error(response, "send reply markdown")
        return response

    async def _send_proactive_markdown(self, chat_id: str, content: str) -> Dict[str, Any]:
        return await self._send_request(APP_CMD_SEND, {"chatid": chat_id, **self._markdown_body(content)})

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send standalone markdown (never touches active streams); serialized per chat for the 30 msgs/min
        limit (846607). ``metadata["is_approval_prompt"]`` uses the control lane."""
        if not chat_id:
            return SendResult(success=False, error="chat_id is required")
        metadata = metadata or {}  # pops mutate the caller's dict on purpose (consumed flags)
        is_control = metadata.pop("is_approval_prompt", False)
        # Approval *confirmations* must not consume the req_id the stream consumer still needs.
        force_proactive = bool(metadata.pop("force_proactive_send", False))
        return await self._enqueue_chat_send(chat_id, lambda: self._send_inner(chat_id, content, reply_to, force_proactive=force_proactive), is_control=is_control)

    async def _send_inner(self, chat_id: str, content: str, reply_to: Optional[str] = None, *, force_proactive: bool = False) -> SendResult:
        """Send under the per-chat queue; force_proactive skips passive reply except in groups."""
        try:
            reply_req_id = None if force_proactive and chat_id not in self._group_chat_ids else self._cached_reply_req_id(chat_id, reply_to)
            if reply_req_id:
                try:
                    response = await self._send_reply_markdown(reply_req_id, content)
                except (asyncio.TimeoutError, RuntimeError) as passive_err:
                    # req_id may be stale after a reconnect — proactive send needs none.
                    logger.warning("[%s] Passive reply failed (%s), falling back to proactive send", self.name, passive_err)
                    response = await self._send_proactive_markdown(chat_id, content)
            elif chat_id in self._group_chat_ids:
                logger.warning("[%s] No cached req_id for group chat %s — cannot send (groups require passive reply via req_id)", self.name, chat_id)
                return SendResult(success=False, error="No req_id available for group chat (passive reply required)")
            else:
                response = await self._send_proactive_markdown(chat_id, content)
        except asyncio.TimeoutError:
            return SendResult(success=False, error="Timeout sending message to WeCom")
        except Exception as exc:
            logger.error("[%s] Send failed: %s", self.name, exc)
            return self._send_failure(str(exc), str(STREAM_NOT_SUBSCRIBED_ERRCODE) in str(exc))
        if error := self._response_error(response):
            return self._send_failure(error, response.get("errcode", 0) == STREAM_NOT_SUBSCRIBED_ERRCODE)
        return SendResult(success=True, message_id=self._payload_req_id(response) or uuid.uuid4().hex[:12], raw_response=response)

    def _send_failure(self, error: str, subscription_lost: bool) -> SendResult:
        """Failed SendResult; on 846609 schedule the stale-req_id purge so later sends recover."""
        if subscription_lost:
            asyncio.ensure_future(self._force_reconnect_on_stale_subscription(STREAM_NOT_SUBSCRIBED_ERRCODE))
        return SendResult(success=False, error=error)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "group" if chat_id and chat_id.lower().startswith("group") else "dm"}


_QR_GENERATE_URL = "https://work.weixin.qq.com/ai/qc/generate"
_QR_QUERY_URL = "https://work.weixin.qq.com/ai/qc/query_result"
_QR_CODE_PAGE = "https://work.weixin.qq.com/ai/qc/gen?source=hermes&scode="
_QR_POLL_INTERVAL, _QR_POLL_TIMEOUT = 3, 300  # seconds (poll every 3s, give up after 5 minutes)


def qr_scan_for_bot_info(*, timeout_seconds: int = _QR_POLL_TIMEOUT) -> Optional[Dict[str, str]]:
    """Fetch a WeCom QR code, render it, poll until scanned or timeout; ``{"bot_id", "secret"}`` or None.
    The ``ai/qc/*`` endpoints back the admin console, not the public API, and may change."""
    import urllib.request
    import urllib.parse

    def _get_json(url: str, timeout: int) -> Dict[str, Any]:
        req = urllib.request.Request(url, headers={"User-Agent": "HermesAgent/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _fail(log_msg: str, detail: Any, shown: Any) -> None:
        logger.error(log_msg, detail)
        print(f" failed: {shown}")

    print("  Connecting to WeCom...", end="", flush=True)
    try:
        raw = _get_json(f"{_QR_GENERATE_URL}?source=hermes", 15)
    except Exception as exc:
        return _fail("WeCom QR: failed to fetch QR code: %s", exc, exc)
    scode, auth_url = (str((raw.get("data") or {}).get(k) or "").strip() for k in ("scode", "auth_url"))
    if not scode or not auth_url:
        return _fail("WeCom QR: unexpected response format: %s", raw, "unexpected response format")
    print(" done.\n")
    page_url = f"{_QR_CODE_PAGE}{urllib.parse.quote(scode)}"
    try:
        import qrcode as _qrcode
        qr = _qrcode.QRCode()
        qr.add_data(auth_url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        print(f"\n  Scan the QR code above, or open this URL directly:\n  {page_url}")
    except Exception:
        print(f"  Open this URL in WeCom on your phone:\n\n  {page_url}\n")
        print("  Tip: pip install qrcode  to display a scannable QR code here next time")
    print("\n  Fetching configuration results...", end="", flush=True)
    deadline = time.monotonic() + timeout_seconds
    query_url = f"{_QR_QUERY_URL}?scode={urllib.parse.quote(scode)}"
    while time.monotonic() < deadline:
        try:
            result = _get_json(query_url, 10)
            print(".", end="", flush=True)  # progress dot on every poll
        except Exception as exc:
            logger.debug("WeCom QR poll error: %s", exc)
            result = {}
        result_data = result.get("data") or {}
        if str(result_data.get("status") or "").lower() != "success":
            time.sleep(_QR_POLL_INTERVAL)
            continue
        bot_info = result_data.get("bot_info") or {}
        bot_id, secret = str(bot_info.get("botid") or bot_info.get("bot_id") or "").strip(), str(bot_info.get("secret") or "").strip()
        if bot_id and secret:
            print()
            return {"bot_id": bot_id, "secret": secret}
        logger.warning("WeCom QR: scan reported success but bot_info missing or incomplete: %s", result_data)
        print("\n  QR scan reported success but no bot credentials were returned.\n  This usually means the bot was not actually created on the WeCom side.\n  Falling back to manual credential entry.")
        return None
    print(f"\n  QR scan timed out ({timeout_seconds // 60} minutes). Please try again.")
    return None


async def _send_via(adapter, chat_id, message, *, live: bool):
    try:
        result = await adapter.send(chat_id, message)
    except Exception as e:
        return {"error": f"WeCom live adapter send failed: {e}" if live else f"WeCom send failed: {e}"}
    if result.success:
        return {"success": True, "platform": "wecom", "chat_id": chat_id, "message_id": result.message_id}
    return {"error": f"WeCom send failed: {result.error}"}


async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    """Reuse the live gateway adapter in-process, else connect ephemerally (WeCom allows ONE
    WebSocket per bot — a second connection kicks the first)."""
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
        adapter = runner.adapters.get(Platform.WECOM) if runner is not None else None
    except Exception:
        adapter = None
    if adapter is not None:
        return await _send_via(adapter, chat_id, message, live=True)
    if not check_wecom_requirements():
        return {"error": "WeCom requirements not met. Need aiohttp + WECOM_BOT_ID/SECRET."}
    try:
        adapter = WeComAdapter(pconfig)
        if not await adapter.connect():
            return {"error": f"WeCom: failed to connect - {getattr(adapter, 'fatal_error_message', None) or 'unknown error'}"}
        try:
            return await _send_via(adapter, chat_id, message, live=False)
        finally:
            await adapter.disconnect()
    except Exception as e:
        return {"error": f"WeCom send failed: {e}"}


_MANUAL_SETUP_STEPS = (
    "1. Go to WeCom Application → Workspace → Smart Robot -> Create smart robots",
    "2. Select API Mode",
    "3. Copy the Bot ID and Secret from the bot's credentials info",
    "4. The bot connects via WebSocket — no public endpoint needed",
)
# (menu label, env saves, (print level, message)...) per unauthorized-user choice; index 3 = skip
_ACCESS_CHOICES = (
    ("Enable open access (anyone can message the bot)", (("WECOM_DM_POLICY", "open"), ("GATEWAY_ALLOW_ALL_USERS", "true")),
     (("warning", "Open access enabled — anyone can use your bot!"),)),
    ("Use DM pairing (unknown users request access, you approve with 'hermes pairing approve')", (("WECOM_DM_POLICY", "pairing"),),
     (("success", "DM pairing mode — users will receive a code to request access."), ("info", "Approve with: hermes pairing approve <platform> <code>"))),
    ("Disable direct messages", (("WECOM_DM_POLICY", "disabled"),), (("warning", "Direct messages disabled."),)),
    ("Skip for now (bot will deny all users until configured)", (), (("info", "Skipped — configure later with 'hermes gateway setup'"),)),
)


def interactive_setup() -> None:
    from hermes_cli.config import get_env_value, remove_env_value, save_env_value
    from hermes_cli.setup import prompt_choice
    from hermes_cli.cli_output import prompt, prompt_yes_no, print_header, print_info, print_success, print_warning
    print_header("WeCom (Enterprise WeChat)")
    if get_env_value("WECOM_BOT_ID") and get_env_value("WECOM_SECRET"):
        print_success("WeCom is already configured.")
        if not prompt_yes_no("Reconfigure WeCom?", False):
            return
    method_idx = prompt_choice("How would you like to set up WeCom?", ["Scan QR code to obtain Bot ID and Secret automatically (recommended)", "Enter existing Bot ID and Secret manually"], 0)
    bot_id = secret = None
    if method_idx == 0:
        try:
            credentials = qr_scan_for_bot_info() or {}
        except KeyboardInterrupt:
            print_warning("WeCom setup cancelled.")
            return
        except Exception as exc:
            print_warning(f"QR scan failed: {exc}")
            credentials = {}
        if credentials:
            bot_id, secret = credentials.get("bot_id", ""), credentials.get("secret", "")
            print_success("✔ QR scan successful! Bot ID and Secret obtained.")
        if not bot_id or not secret:
            print_info("QR scan did not complete. Continuing with manual input.")
            bot_id = secret = None
    if not bot_id or not secret:
        for line in _MANUAL_SETUP_STEPS:
            print_info(line)
        creds = []
        for label, password in (("Bot ID", False), ("Secret", True)):
            creds.append(prompt(label, password=password))
            if not creds[-1]:
                print_warning(f"Skipped — WeCom won't work without a {label}.")
                return
        bot_id, secret = creds
    save_env_value("WECOM_BOT_ID", bot_id)
    save_env_value("WECOM_SECRET", secret)
    print_info("The gateway DENIES all users by default for security.")
    print_info("Enter user IDs to create an allowlist, or leave empty.")
    allowed = prompt("Allowed user IDs (comma-separated, or empty)", password=False)
    if allowed:
        save_env_value("WECOM_ALLOWED_USERS", allowed.replace(" ", ""))
        print_success("Saved — only these users can interact with the bot.")
    else:
        access_idx = prompt_choice("How should unauthorized users be handled?", [label for label, _, _ in _ACCESS_CHOICES], 1)
        _, saves, messages = _ACCESS_CHOICES[access_idx if access_idx in (0, 1, 2) else 3]
        for key, value in saves:
            save_env_value(key, value)
        for level, message in messages:
            {"warning": print_warning, "success": print_success, "info": print_info}[level](message)
    if home := prompt("Home chat ID (optional, for cron/notifications)", password=False).strip():
        save_env_value("WECOM_HOME_CHANNEL", home)
        print_success(f"Home channel set to {home}")
    elif remove_env_value("WECOM_HOME_CHANNEL"):
        print_info("Home channel cleared.")
    print_success("💬 WeCom configured!")


def _is_connected(config) -> bool:
    return bool((getattr(config, "extra", {}) or {}).get("bot_id"))


def _callback_is_connected(config) -> bool:
    """Callback mode: corp_id or a multi-app `apps` block."""
    extra = getattr(config, "extra", {}) or {}
    return bool(extra.get("corp_id") or extra.get("apps"))


def _build_adapter(config):
    return WeComAdapter(config)


def _build_callback_adapter(config):
    from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter
    return WecomCallbackAdapter(config)


def register(ctx) -> None:
    common = dict(install_hint="Run `hermes setup` to install WeCom support.", emoji="💼", allow_update_command=True)
    ctx.register_platform(
        name="wecom", label="WeCom (Enterprise WeChat)", adapter_factory=_build_adapter, check_fn=check_wecom_requirements,
        is_connected=_is_connected, validate_config=_is_connected, required_env=["WECOM_BOT_ID", "WECOM_SECRET"],
        setup_fn=interactive_setup, allowed_users_env="WECOM_ALLOWED_USERS", allow_all_env="WECOM_ALLOW_ALL_USERS",
        cron_deliver_env_var="WECOM_HOME_CHANNEL", standalone_sender_fn=_standalone_send, max_message_length=4000, **common,
    )
    from plugins.platforms.wecom.callback_adapter import check_wecom_callback_requirements, ensure_wecom_callback_requirements
    ctx.register_platform(
        name="wecom_callback", label="WeCom Callback (self-built apps)", adapter_factory=_build_callback_adapter,
        check_fn=check_wecom_callback_requirements, ensure_deps_fn=ensure_wecom_callback_requirements,
        is_connected=_callback_is_connected, validate_config=_callback_is_connected,
        required_env=["WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET"],
        allowed_users_env="WECOM_CALLBACK_ALLOWED_USERS", allow_all_env="WECOM_CALLBACK_ALLOW_ALL_USERS", **common,
    )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import Path  # noqa: F401,E402
import base64  # noqa: F401,E402
from dataclasses import dataclass  # noqa: F401,E402
from collections import deque  # noqa: F401,E402
import hashlib  # noqa: F401,E402
import mimetypes  # noqa: F401,E402
import os  # noqa: F401,E402
from urllib.parse import unquote  # noqa: F401,E402
from urllib.parse import urlparse  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'ABSOLUTE_MAX_BYTES': ('plugins.platforms.wecom.media', 'ABSOLUTE_MAX_BYTES'),
    'APP_CMD_UPLOAD_MEDIA_CHUNK': ('plugins.platforms.wecom.media', 'APP_CMD_UPLOAD_MEDIA_CHUNK'),
    'APP_CMD_UPLOAD_MEDIA_FINISH': ('plugins.platforms.wecom.media', 'APP_CMD_UPLOAD_MEDIA_FINISH'),
    'APP_CMD_UPLOAD_MEDIA_INIT': ('plugins.platforms.wecom.media', 'APP_CMD_UPLOAD_MEDIA_INIT'),
    'FILE_MAX_BYTES': ('plugins.platforms.wecom.media', 'FILE_MAX_BYTES'),
    'IMAGE_MAX_BYTES': ('plugins.platforms.wecom.media', 'IMAGE_MAX_BYTES'),
    'MAX_INTERMEDIATE_FRAMES': ('plugins.platforms.wecom.streaming', 'MAX_INTERMEDIATE_FRAMES'),
    'MAX_UPLOAD_CHUNKS': ('plugins.platforms.wecom.media', 'MAX_UPLOAD_CHUNKS'),
    'ReplyFrame': ('plugins.platforms.wecom.streaming', 'ReplyFrame'),
    'STREAM_EXPIRED_ERRCODE': ('plugins.platforms.wecom.streaming', 'STREAM_EXPIRED_ERRCODE'),
    'STREAM_REQUEST_EXPIRED_ERRCODE': ('plugins.platforms.wecom.streaming', 'STREAM_REQUEST_EXPIRED_ERRCODE'),
    'STREAM_VERSION_CONFLICT_ERRCODE': ('plugins.platforms.wecom.streaming', 'STREAM_VERSION_CONFLICT_ERRCODE'),
    'UPLOAD_CHUNK_SIZE': ('plugins.platforms.wecom.media', 'UPLOAD_CHUNK_SIZE'),
    'VIDEO_MAX_BYTES': ('plugins.platforms.wecom.media', 'VIDEO_MAX_BYTES'),
    'VOICE_MAX_BYTES': ('plugins.platforms.wecom.media', 'VOICE_MAX_BYTES'),
    'VOICE_SUPPORTED_MIMES': ('plugins.platforms.wecom.media', 'VOICE_SUPPORTED_MIMES'),
    'WeComStreamExpiredError': ('plugins.platforms.wecom.streaming', 'WeComStreamExpiredError'),
    'cache_document_from_bytes_async': ('gateway.platforms.base', 'cache_document_from_bytes_async'),
    'cache_image_from_bytes_async': ('gateway.platforms.base', 'cache_image_from_bytes_async'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
