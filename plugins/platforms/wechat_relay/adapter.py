"""Independent WeChat relay platform adapter for Hermes.

Boundary notes:
- This implements the WeChat Relay / 微信中转 transport.
- It is not Hermes' built-in ``weixin`` adapter and does not patch it.
- Default port is 9797. Port 8787 is deliberately rejected because it belongs
  to the OpenClaw mobile-shell notification path, not WeChat relay.

Accepted inbound shapes are intentionally permissive because Android relay
payloads have drifted over time. Text notifications become Hermes
``MessageEvent`` objects; media/non-text placeholders are acknowledged but not
sent to the agent until a safe artifact pipeline exists.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9797
DEFAULT_SEND_PATH = "/_openclaw/wechat/send"
HEALTH_PATH = "/_openclaw/notify/healthz"
CONNECTIONS_PATH = "/_openclaw/notify/connections"
WS_PATH = "/_openclaw/notify/ws"
INBOUND_PATH = "/_openclaw/wechat/inbound"

_MEDIA_EVENT_TYPES = {
    "wechat.media_inbound",
    "media_inbound",
    "media",
    "image",
    "photo",
    "file",
    "document",
    "audio",
    "voice",
    "video",
    "sticker",
}
_PLACEHOLDER_RE = re.compile(
    r"^\s*\[(?:图片|照片|文件|语音|视频|动画表情|表情|链接|image|photo|file|voice|video|sticker)\]\s*$",
    re.IGNORECASE,
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y"}


def _split_csv(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _safe_key(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    return re.sub(r"[^A-Za-z0-9_.:@+-]+", "_", text)[:200] or fallback


def _first(payload: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _message_category(text: str) -> str:
    lowered = text.lower()
    if any(k in text for k in ("不启动", "不动", "报警", "故障", "异常", "缺", "不对", "对不上", "问题", "卡", "坏")):
        return "异常/问题"
    if any(k in text for k in ("完成", "好了", "已", "通过", "装好了", "上电", "调试", "空跑", "验收")):
        return "进度"
    if any(k in text for k in ("几点", "什么时候", "多久", "能不能", "需要", "吗", "？", "?")):
        return "询问"
    return "消息"


def _extract_device_hint(text: str) -> str:
    match = re.search(r"([A-Za-z]?\d+\s*(?:号机|#|号|台)|\d+\s*(?:号机|#|号|台)|机械手|控制器|相机|工位|线)", text)
    return match.group(1).replace(" ", "") if match else "未识别"


def _json_response(web: Any, data: dict[str, Any], *, status: int = 200) -> Any:
    return web.Response(
        text=json.dumps(data, ensure_ascii=False),
        status=status,
        content_type="application/json",
    )


def _env_enablement() -> Optional[dict[str, Any]]:
    if not _truthy(os.getenv("WECHAT_RELAY_ENABLED")):
        return None
    return {
        "host": os.getenv("WECHAT_RELAY_HOST", DEFAULT_HOST),
        "port": int(os.getenv("WECHAT_RELAY_PORT") or DEFAULT_PORT),
        "send_url": os.getenv("WECHAT_RELAY_SEND_URL") or "",
        "shared_secret": os.getenv("WECHAT_RELAY_SHARED_SECRET") or "",
        "allowed_logical_keys": _split_csv(os.getenv("WECHAT_RELAY_ALLOWED_LOGICAL_KEYS")),
        "enable_legacy_openclaw_paths": _truthy(os.getenv("WECHAT_RELAY_ENABLE_OPENCLAW_PATHS", "true")),
        "allow_insecure_local": _truthy(os.getenv("WECHAT_RELAY_ALLOW_INSECURE_LOCAL") or ""),
    }


def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> dict[str, Any]:
    section = yaml_cfg.get("wechat_relay")
    if not isinstance(section, dict):
        section = {}

    def seed_env(key: str, env_name: str) -> None:
        value = section.get(key)
        if value is not None and not os.getenv(env_name):
            os.environ[env_name] = str(value)

    seed_env("enabled", "WECHAT_RELAY_ENABLED")
    seed_env("host", "WECHAT_RELAY_HOST")
    seed_env("port", "WECHAT_RELAY_PORT")
    seed_env("send_url", "WECHAT_RELAY_SEND_URL")
    seed_env("shared_secret", "WECHAT_RELAY_SHARED_SECRET")
    seed_env("allow_all_users", "WECHAT_RELAY_ALLOW_ALL_USERS")
    seed_env("allow_insecure_local", "WECHAT_RELAY_ALLOW_INSECURE_LOCAL")

    extra = dict(platform_cfg.get("extra") or {})
    for key in (
        "host",
        "port",
        "send_url",
        "shared_secret",
        "allowed_logical_keys",
        "auto_reply",
        "require_mention_in_groups",
        "mention_names",
        "collect_group_messages",
        "collection_state_file",
        "collection_project",
        "enable_legacy_openclaw_paths",
        "allow_insecure_local",
    ):
        if key in section:
            extra[key] = section[key]
    return extra


def check_requirements() -> bool:
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        logger.warning("WeChat Relay requires aiohttp")
        return False


def validate_config(config: PlatformConfig) -> bool:
    if not config.enabled and not _truthy(os.getenv("WECHAT_RELAY_ENABLED")):
        return False
    try:
        port = int(config.extra.get("port") or os.getenv("WECHAT_RELAY_PORT") or DEFAULT_PORT)
    except (TypeError, ValueError):
        return False
    return port > 0 and port != 8787


def is_connected(config: PlatformConfig) -> bool:
    return validate_config(config)


@dataclass
class RelayConnection:
    ws: Any
    role: str
    logical_key: str
    title: str
    connected_at: float


class WeChatRelayAdapter(BasePlatformAdapter):
    """Dedicated Hermes gateway platform for Android WeChat relay."""

    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("wechat_relay"))
        extra = config.extra or {}
        self._host = str(extra.get("host") or os.getenv("WECHAT_RELAY_HOST") or DEFAULT_HOST)
        self._port = int(extra.get("port") or os.getenv("WECHAT_RELAY_PORT") or DEFAULT_PORT)
        if self._port == 8787:
            raise ValueError("wechat_relay refuses port 8787; use dedicated port 9797")
        raw_send_url = str(extra.get("send_url") or os.getenv("WECHAT_RELAY_SEND_URL") or "").strip()
        self._send_url = raw_send_url or f"http://127.0.0.1:{self._port}{DEFAULT_SEND_PATH}"
        self._shared_secret = str(extra.get("shared_secret") or os.getenv("WECHAT_RELAY_SHARED_SECRET") or "")
        self._auto_reply = _truthy(extra.get("auto_reply", True))
        self._require_mention_in_groups = _truthy(extra.get("require_mention_in_groups", False))
        self._mention_names = set(
            _split_csv(extra.get("mention_names") or os.getenv("WECHAT_RELAY_MENTION_NAMES") or "超进化")
        )
        self._collect_group_messages = _truthy(extra.get("collect_group_messages", False))
        self._collection_state_file = str(
            extra.get("collection_state_file")
            or os.getenv("WECHAT_RELAY_COLLECTION_STATE_FILE")
            or os.path.expanduser("~/.hermes/workspace/mpm_state.json")
        )
        self._collection_project = str(
            extra.get("collection_project") or os.getenv("WECHAT_RELAY_COLLECTION_PROJECT") or "班旗项目"
        )
        self._enable_legacy_paths = _truthy(extra.get("enable_legacy_openclaw_paths", True))
        self._allowed_logical_keys = set(_split_csv(extra.get("allowed_logical_keys")))
        self._allow_insecure_local = _truthy(
            extra.get("allow_insecure_local") or os.getenv("WECHAT_RELAY_ALLOW_INSECURE_LOCAL") or ""
        )
        self._connections: dict[int, RelayConnection] = {}
        self._runner: Any = None
        self._site: Any = None
        self._client_session: Any = None

    @property
    def enforces_own_access_policy(self) -> bool:
        return True

    def _authorized_request(self, request: Any) -> bool:
        if not self._shared_secret:
            # No secret configured: deny by default.  Set allow_insecure_local=true
            # (or WECHAT_RELAY_ALLOW_INSECURE_LOCAL=true) only for local dev/test
            # environments where secret management is intentionally skipped.
            return self._allow_insecure_local
        header_secret = request.headers.get("x-wechat-relay-secret") or ""
        auth = request.headers.get("authorization") or ""
        return header_secret == self._shared_secret or auth == f"Bearer {self._shared_secret}"

    def _authorized_logical_key(self, logical_key: str) -> bool:
        return not self._allowed_logical_keys or logical_key in self._allowed_logical_keys

    async def connect(self) -> bool:
        from aiohttp import web

        if self._port == 8787:
            self._set_fatal_error("bad_port", "wechat_relay refuses port 8787", retryable=False)
            return False

        app = web.Application()
        app.router.add_get(HEALTH_PATH, self._handle_health)
        app.router.add_get(CONNECTIONS_PATH, self._handle_connections)
        app.router.add_get(WS_PATH, self._handle_ws)
        app.router.add_post(INBOUND_PATH, self._handle_inbound_http)
        app.router.add_post(DEFAULT_SEND_PATH, self._handle_send_probe)
        if self._enable_legacy_paths:
            # Android/OpenClaw baseline path compatibility. Names remain
            # _openclaw because the Android app currently targets that path;
            # this adapter still stays isolated under the Hermes platform name
            # ``wechat_relay`` and does not touch OpenClaw/Hermes Weixin code.
            app.router.add_get("/healthz", self._handle_health)
            app.router.add_post("/wechat/inbound", self._handle_inbound_http)

        self._runner = web.AppRunner(app)
        try:
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
        except Exception as exc:
            self._set_fatal_error("listen_failed", f"failed to listen on {self._host}:{self._port}: {exc}", retryable=True)
            logger.error("WeChat Relay listen failed on %s:%s: %s", self._host, self._port, exc)
            return False

        self._mark_connected()
        logger.info("WeChat Relay listening on http://%s:%s", self._host, self._port)
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        for conn in list(self._connections.values()):
            try:
                await conn.ws.close()
            except Exception:
                pass
        self._connections.clear()
        if self._client_session is not None:
            await self._client_session.close()
            self._client_session = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None

    async def _handle_health(self, request: Any) -> Any:
        from aiohttp import web

        return _json_response(web, {
            "ok": True,
            "platform": "wechat_relay",
            "kind": "wechat-relay",
            "port": self._port,
            "connections": len(self._connections),
            "paths": {"ws": WS_PATH, "inbound": INBOUND_PATH, "send": DEFAULT_SEND_PATH},
        })

    async def _handle_connections(self, request: Any) -> Any:
        from aiohttp import web

        if not self._authorized_request(request):
            return _json_response(web, {"ok": False, "error": "unauthorized"}, status=401)
        connections = []
        for conn in self._connections.values():
            connections.append({
                "clientRole": conn.role,
                "logicalKey": conn.logical_key,
                "title": conn.title,
                "connectedAt": conn.connected_at,
            })
        return _json_response(web, {"ok": True, "connections": connections})

    async def _handle_send_probe(self, request: Any) -> Any:
        from aiohttp import web

        if not self._authorized_request(request):
            return _json_response(web, {"ok": False, "error": "unauthorized"}, status=401)
        # Compatibility endpoint matching OpenClaw's baseline:
        # POST /_openclaw/wechat/send forwards text to the connected Android
        # relay client over WebSocket. Tests can hit it directly; outbound
        # Hermes replies use the same forwarding code when no explicit
        # WECHAT_RELAY_SEND_URL is configured.
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        logical_key = str(payload.get("logicalKey") or payload.get("chatId") or "")
        text = str(payload.get("text") or payload.get("content") or "")
        result = await self._send_to_connected_android(logical_key, text, payload)
        return _json_response(web, result, status=200 if result.get("ok") else 503)

    async def _handle_inbound_http(self, request: Any) -> Any:
        from aiohttp import web

        if not self._authorized_request(request):
            return _json_response(web, {"ok": False, "error": "unauthorized"}, status=401)
        try:
            payload = await request.json()
        except Exception as exc:
            return _json_response(web, {"ok": False, "error": f"invalid_json: {exc}"}, status=400)
        result = await self._dispatch_payload(payload)
        return _json_response(web, result, status=200 if result.get("ok") else 400)

    async def _handle_ws(self, request: Any) -> Any:
        from aiohttp import WSMsgType, web

        if not self._authorized_request(request):
            return _json_response(web, {"ok": False, "error": "unauthorized"}, status=401)

        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        logical_key = _safe_key(request.query.get("logicalKey") or request.query.get("chatId"), "unknown")
        role = str(request.query.get("clientRole") or request.query.get("role") or "wechat-relay")
        title = str(request.query.get("title") or request.query.get("chatTitle") or logical_key)
        self._connections[id(ws)] = RelayConnection(ws, role, logical_key, title, time.time())
        await ws.send_json({"ok": True, "type": "wechat_relay.connected", "platform": "wechat_relay"})
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await ws.send_json({"ok": False, "error": "invalid_json"})
                        continue
                    if isinstance(payload.get("payload"), dict):
                        inner = dict(payload["payload"])
                        inner.setdefault("event", payload.get("type") or payload.get("event") or "wechat.message")
                        payload = inner
                    payload.setdefault("logicalKey", logical_key)
                    payload.setdefault("title", title)
                    result = await self._dispatch_payload(payload)
                    await ws.send_json(result)
                elif msg.type == WSMsgType.ERROR:
                    logger.debug("WeChat Relay websocket error: %s", ws.exception())
        finally:
            self._connections.pop(id(ws), None)
        return ws

    async def _dispatch_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_type = str(_first(payload, "event", "type", "eventType", default="wechat.message"))
        logical_raw = _first(payload, "logicalKey", "logical_key", "conversationId", "chatId", "from", default="unknown")
        logical_key = _safe_key(logical_raw, "unknown")
        if not self._authorized_logical_key(logical_key):
            return {"ok": False, "error": "logical_key_not_allowed", "logicalKey": logical_key}

        text = str(_first(payload, "text", "content", "message", "body", default="")).strip()
        is_media = event_type in _MEDIA_EVENT_TYPES or bool(payload.get("media") or payload.get("mediaType"))
        if is_media or _PLACEHOLDER_RE.match(text):
            logger.info("WeChat Relay media/non-text inbound reserved: logicalKey=%s event=%s", logical_key, event_type)
            return {
                "ok": True,
                "ignored": True,
                "event": "wechat.media_inbound",
                "reason": "media relay reserved; not dispatched as text",
                "logicalKey": logical_key,
            }

        if not text:
            return {"ok": False, "error": "empty_text", "logicalKey": logical_key}

        title = str(_first(payload, "title", "chatTitle", "senderName", "name", default=logical_key))
        sender_id = str(_first(payload, "senderId", "sender_id", "fromUser", "from", default=logical_key))
        sender_name = str(_first(payload, "senderName", "sender", "fromName", default=title))
        message_id = str(_first(payload, "messageId", "message_id", "id", default=f"wechat-relay-{int(time.time() * 1000)}"))
        chat_id = f"wechat:logical:{logical_key}"
        chat_type = "group" if self._looks_like_group_title(title, sender_name) else "dm"
        source = SessionSource(
            platform=Platform("wechat_relay"),
            chat_id=chat_id,
            chat_name=title,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_name,
            chat_id_alt=str(logical_raw),
            message_id=message_id,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=message_id,
        )
        if self._auto_reply:
            if self._should_suppress_group_reply(text, title, sender_name):
                collection = self._collect_group_message(
                    logical_key=logical_key,
                    chat_id=chat_id,
                    title=title,
                    sender_id=sender_id,
                    sender_name=sender_name,
                    text=text,
                    message_id=message_id,
                    payload=payload,
                )
                logger.info(
                    "WeChat Relay suppressed group message without mention: logicalKey=%s title=%s sender=%s collected=%s",
                    logical_key,
                    title,
                    sender_name,
                    collection.get("collected"),
                )
                return {
                    "ok": True,
                    "dispatched": False,
                    "ignored": True,
                    "collected": bool(collection.get("collected")),
                    "collectionError": collection.get("error"),
                    "reason": "group_mention_required",
                    "logicalKey": logical_key,
                    "chatId": chat_id,
                }
            await self.handle_message(event)
            return {"ok": True, "dispatched": True, "logicalKey": logical_key, "chatId": chat_id}
        return {"ok": True, "dispatched": False, "logicalKey": logical_key, "chatId": chat_id}

    def _looks_like_group_title(self, title: str, sender_name: str) -> bool:
        value = f"{title} {sender_name}"
        return any(marker in value for marker in ("群", "班旗", "交流", "项目", "技术"))

    def _should_suppress_group_reply(self, text: str, title: str, sender_name: str) -> bool:
        if not self._require_mention_in_groups:
            return False
        if not self._looks_like_group_title(title, sender_name):
            return False
        return not any(f"@{name}" in text or name in text for name in self._mention_names)

    def _collect_group_message(
        self,
        *,
        logical_key: str,
        chat_id: str,
        title: str,
        sender_id: str,
        sender_name: str,
        text: str,
        message_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._collect_group_messages:
            return {"collected": False, "reason": "collection_disabled"}
        try:
            path = Path(os.path.expanduser(self._collection_state_file))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                try:
                    state = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(state, dict):
                        state = {}
                except Exception:
                    backup = path.with_suffix(path.suffix + f".bad-{int(time.time())}")
                    path.rename(backup)
                    state = {}
            else:
                state = {}
            state.setdefault("schema", "mpm_state_v1")
            state.setdefault("projects", {})
            state.setdefault("devices", {})
            state.setdefault("issues", [])
            inbox = state.setdefault("message_inbox", [])
            if not isinstance(inbox, list):
                inbox = []
                state["message_inbox"] = inbox
            dedupe_key = f"wechat_relay:{chat_id}:{message_id}:{hash(text)}"
            if any(item.get("dedupe_key") == dedupe_key for item in inbox[-500:]):
                return {"collected": True, "deduped": True}
            item = {
                "time": _now_iso(),
                "platform": "wechat_relay",
                "project": self._collection_project,
                "chat_id": chat_id,
                "logical_key": logical_key,
                "chat_name": title,
                "chat_type": "group",
                "sender_id": sender_id,
                "sender_name": sender_name,
                "text": text,
                "message_id": message_id,
                "category": _message_category(text),
                "device_hint": _extract_device_hint(text),
                "requires_reply": any(f"@{name}" in text or name in text for name in self._mention_names),
                "dedupe_key": dedupe_key,
                "raw_event": {
                    k: payload.get(k)
                    for k in ("event", "type", "uuid", "logicalKey", "title", "senderName", "messageId")
                    if k in payload
                },
            }
            inbox.append(item)
            state["updated_at"] = _now_iso()
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(path)
            return {"collected": True, "category": item["category"], "device_hint": item["device_hint"]}
        except Exception as exc:
            logger.exception("WeChat Relay failed to collect suppressed group message: %s", exc)
            return {"collected": False, "error": str(exc)}

    async def _send_to_connected_android(
        self,
        logical_key: str,
        text: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Forward outbound text to a connected Android relay WebSocket.

        The OpenClaw baseline sends outbound replies to
        ``POST /_openclaw/wechat/send``.  In Hermes' low-intrusion version that
        endpoint lives on this dedicated ``wechat_relay`` adapter, then pushes a
        small ``outbound.sendText`` command over the Android relay WebSocket.
        This keeps the channel self-contained and avoids touching the existing
        ``weixin`` adapter.
        """
        logical_key = _safe_key(logical_key, "")
        text = str(text or "")
        if not text:
            return {"ok": False, "error": "empty_text", "logicalKey": logical_key}

        candidates = list(self._connections.values())
        if logical_key:
            exact = [c for c in candidates if c.logical_key == logical_key]
            if exact:
                candidates = exact
        if not candidates:
            return {
                "ok": False,
                "error": "android_relay_not_connected",
                "logicalKey": logical_key,
                "connected": len(self._connections),
            }

        command = {
            "type": "outbound.sendText",
            "event": "outbound.sendText",
            "logicalKey": logical_key or candidates[0].logical_key,
            "text": text,
            "payload": payload or {},
            "timestamp": int(time.time() * 1000),
        }
        last_error = None
        for conn in candidates:
            try:
                await conn.ws.send_json(command)
                message_id = f"wechat-relay-out-{command['timestamp']}"
                return {
                    "ok": True,
                    "messageId": message_id,
                    "logicalKey": command["logicalKey"],
                    "deliveredVia": "websocket",
                }
            except Exception as exc:  # noqa: BLE001 - connection may disappear mid-send
                last_error = str(exc)
                logger.debug("WeChat Relay websocket outbound failed: %s", exc)
        return {
            "ok": False,
            "error": last_error or "android_relay_send_failed",
            "logicalKey": logical_key,
            "connected": len(self._connections),
        }

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        import aiohttp

        logical_key = str(chat_id or "")
        if logical_key.startswith("wechat:logical:"):
            logical_key = logical_key[len("wechat:logical:"):]
        payload = {
            "logicalKey": logical_key,
            "chatId": chat_id,
            "text": content,
            "replyTo": reply_to,
            "metadata": metadata or {},
        }
        headers = {"content-type": "application/json"}
        if self._shared_secret:
            headers["x-wechat-relay-secret"] = self._shared_secret
        try:
            if self._send_url == f"http://127.0.0.1:{self._port}{DEFAULT_SEND_PATH}":
                data = await self._send_to_connected_android(logical_key, content, payload)
                if not data.get("ok"):
                    return SendResult(
                        success=False,
                        error=str(data.get("error") or data),
                        raw_response=data,
                        retryable=data.get("error") == "android_relay_not_connected",
                    )
                message_id = str(data.get("messageId") or data.get("id") or f"wechat-relay-out-{int(time.time() * 1000)}")
                return SendResult(success=True, message_id=message_id, raw_response=data)

            if self._client_session is None or self._client_session.closed:
                self._client_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
            async with self._client_session.post(self._send_url, json=payload, headers=headers) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    return SendResult(success=False, error=f"HTTP {resp.status}: {body}", raw_response=body, retryable=resp.status >= 500)
                try:
                    data = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    data = {"body": body}
                message_id = str(data.get("messageId") or data.get("id") or f"wechat-relay-out-{int(time.time() * 1000)}")
                return SendResult(success=True, message_id=message_id, raw_response=data)
        except Exception as exc:
            logger.warning("WeChat Relay outbound send failed: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {"id": chat_id, "type": "dm", "title": chat_id.replace("wechat:logical:", "")}


def register(ctx) -> None:
    ctx.register_platform(
        name="wechat_relay",
        label="WeChat Relay",
        adapter_factory=lambda cfg: WeChatRelayAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        env_enablement_fn=_env_enablement,
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="WECHAT_RELAY_ALLOWED_LOGICAL_KEYS",
        allow_all_env="WECHAT_RELAY_ALLOW_ALL_USERS",
        cron_deliver_env_var="WECHAT_RELAY_HOME_CHANNEL",
        max_message_length=1800,
        pii_safe=True,
        allow_update_command=False,
        emoji="💬",
        platform_hint=(
            "You are replying through an independent Android WeChat relay. "
            "This is the WeChat Relay / 微信中转 channel, not the built-in Weixin plugin. "
            "Use concise plain text. Non-text WeChat media events are reserved and should not be treated as normal text."
        ),
    )
