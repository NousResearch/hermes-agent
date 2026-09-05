"""WeCom callback-mode adapter (self-built apps): decrypt POSTed XML, queue for the agent, ack at once;
reply later via proactive ``message/send``. Multiple apps are scoped by ``corp_id:user_id``."""

from __future__ import annotations

import asyncio
import logging
import socket as _socket
import time
from typing import Any, Dict, List, Optional

# Untrusted pre-auth bodies are parsed with defusedxml (billion-laughs / XXE).
try:
    import defusedxml.ElementTree as ET
    DEFUSEDXML_AVAILABLE = True
except ImportError:
    ET = None  # type: ignore[assignment]
    DEFUSEDXML_AVAILABLE = False

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from plugins.platforms.wecom.wecom_crypto import WXBizMsgCrypt, WeComCryptoError

logger = logging.getLogger(__name__)

DEFAULT_HOST = None  # dual-stack bind ("0.0.0.0" broke IPv6-only); pin via extra.host
DEFAULT_PORT = 8645
DEFAULT_PATH = "/wecom/callback"
_MAX_BODY = 65_536  # pre-auth body cap: callbacks are small encrypted XML envelopes
ACCESS_TOKEN_TTL_SECONDS = 7200
MESSAGE_DEDUP_TTL_SECONDS = 300
_SEND_URL = "https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token="
_TOKEN_URL = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"


def check_wecom_callback_requirements() -> bool:
    """PASSIVE probe (registry ``check_fn``) — must never install anything."""
    return AIOHTTP_AVAILABLE and HTTPX_AVAILABLE and DEFUSEDXML_AVAILABLE


def ensure_wecom_callback_requirements() -> bool:
    """ACTIVE lazy-installer (``ensure_deps_fn``): installs ``defusedxml`` and rebinds globals.

    Registered as ``ensure_deps_fn``: the registry's ``create_adapter()`` runs it when the passive probe
    fails, right before the gateway connects the platform (#79812). Installs ``defusedxml`` (the only
    non-core dep; aiohttp/httpx ship with every messaging install) and rebinds the module globals. Before
    this hook existed, the passive ``check_fn`` returned False forever on installs without the ``wecom``
    extra and the ``platform.wecom_callback`` LAZY_DEPS entry was never exercised.
    """
    if check_wecom_callback_requirements():
        return True

    def _import() -> dict:
        import defusedxml.ElementTree as _ET
        return {"ET": _ET, "DEFUSEDXML_AVAILABLE": True}

    try:
        from tools.lazy_deps import ensure_and_bind
    except Exception:  # pragma: no cover — defensive
        return False
    return bool(ensure_and_bind("platform.wecom_callback", _import, globals(), prompt=False)) and check_wecom_callback_requirements()


def _ack():
    return web.Response(text="success", content_type="text/plain")


class WecomCallbackAdapter(BasePlatformAdapter):
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WECOM_CALLBACK)
        extra = config.extra or {}
        _raw_host = extra.get("host") or DEFAULT_HOST
        self._host = str(_raw_host) if _raw_host else None
        self._port = int(extra.get("port") or DEFAULT_PORT)
        self._path = str(extra.get("path") or DEFAULT_PATH)
        self._apps: List[Dict[str, Any]] = self._normalize_apps(extra)
        self._runner = self._site = self._app = self._http_client = self._poll_task = None
        self._message_queue: asyncio.Queue[MessageEvent] = asyncio.Queue()
        self._seen_messages: Dict[str, float] = {}
        self._user_app_map: Dict[str, str] = {}
        self._access_tokens: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _user_app_key(corp_id: str, user_id: str) -> str:
        return f"{corp_id}:{user_id}" if corp_id else user_id

    @staticmethod
    def _normalize_apps(extra: Dict[str, Any]) -> List[Dict[str, Any]]:
        apps = extra.get("apps")
        if isinstance(apps, list) and apps:
            return [dict(app) for app in apps if isinstance(app, dict)]
        if extra.get("corp_id"):
            return [{"name": extra.get("name") or "default", "corp_id": extra.get("corp_id", ""), "corp_secret": extra.get("corp_secret", ""),
                     "agent_id": str(extra.get("agent_id", "")), "token": extra.get("token", ""), "encoding_aes_key": extra.get("encoding_aes_key", "")}]
        return []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        del is_reconnect  # kwarg MUST exist (GatewayRunner passes it) even though unused
        if not self._apps:
            logger.warning("[WecomCallback] No callback apps configured")
            return False
        if not check_wecom_callback_requirements():
            logger.warning("[WecomCallback] aiohttp/httpx not installed")
            return False
        try:  # quick port-in-use check
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                sock.connect(("127.0.0.1", self._port))
            logger.error("[WecomCallback] Port %d already in use", self._port)
            return False
        except (ConnectionRefusedError, OSError):
            pass
        try:
            # Tighter keepalive so idle CLOSE_WAIT drains promptly (#18451).
            from gateway.platforms._http_client_limits import platform_httpx_limits
            self._http_client = httpx.AsyncClient(timeout=20.0, limits=platform_httpx_limits())
            # client_max_size → 413 before our handler / any signature work runs.
            self._app = web.Application(client_max_size=_MAX_BODY)
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_get(self._path, self._handle_verify)
            self._app.router.add_post(self._path, self._handle_callback)
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
            self._poll_task = asyncio.create_task(self._poll_loop())
            self._mark_connected()
            logger.info("[WecomCallback] HTTP server listening on %s:%s%s", self._host, self._port, self._path)
            for app in self._apps:
                try:
                    await self._refresh_access_token(app)
                except Exception as exc:
                    logger.warning("[WecomCallback] Initial token refresh failed for app '%s': %s", app.get("name", "default"), exc)
            return True
        except Exception:
            await self._cleanup()
            logger.exception("[WecomCallback] Failed to start")
            return False

    async def disconnect(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        await self._cleanup()
        self._mark_disconnected()
        logger.info("[WecomCallback] Disconnected")

    async def _cleanup(self) -> None:
        self._site = None
        if self._runner:
            await self._runner.cleanup()
        self._runner = self._app = None
        if self._http_client:
            await self._http_client.aclose()
        self._http_client = None

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        app = self._resolve_app_for_chat(chat_id)
        try:
            payload = {"touser": chat_id.split(":", 1)[-1], "msgtype": "text", "agentid": int(str(app.get("agent_id") or 0)), "text": {"content": content[:2048]}, "safe": 0}
            for _attempt in range(2):
                token = await self._get_access_token(app)
                resp = await self._http_client.post(f"{_SEND_URL}{token}", json=payload)
                data = resp.json()
                errcode = data.get("errcode")
                if errcode in {40001, 42001} and _attempt == 0:  # token rejected — evict so the retry fetches a fresh one
                    logger.warning("[WecomCallback] Token rejected for app '%s' (errcode=%s), refreshing", app.get("name", "default"), errcode)
                    self._access_tokens.pop(app["name"], None)
                    continue
                return SendResult(success=True, message_id=str(data.get("msgid", "")), raw_response=data) if errcode == 0 else SendResult(success=False, error=str(data))
            return SendResult(success=False, error="send failed after token refresh")
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    def _resolve_app_for_chat(self, chat_id: str) -> Dict[str, Any]:
        app_name = self._user_app_map.get(chat_id)
        if not app_name and ":" not in chat_id:  # legacy bare user_id — unique match only
            matching = [k for k in self._user_app_map if k.endswith(f":{chat_id}")]
            app_name = self._user_app_map.get(matching[0]) if len(matching) == 1 else app_name
        return self._get_app_by_name(app_name) or self._apps[0]

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "platform": "wecom_callback"})

    async def _handle_verify(self, request: web.Request) -> web.Response:
        """GET endpoint — WeCom URL verification handshake."""
        msg_signature, timestamp, nonce = self._signature_params(request)
        echostr = request.query.get("echostr", "")
        for app in self._apps:
            try:
                plain = self._crypt_for_app(app).verify_url(msg_signature, timestamp, nonce, echostr)
                return web.Response(text=plain, content_type="text/plain")
            except Exception:
                continue
        return web.Response(status=403, text="signature verification failed")

    async def _handle_callback(self, request: web.Request) -> web.Response:
        """POST endpoint — receive an encrypted message callback."""
        msg_signature, timestamp, nonce = self._signature_params(request)
        body_bytes = await request.read()  # explicit guard in addition to client_max_size
        if len(body_bytes) > _MAX_BODY:
            logger.warning("[WecomCallback] Payload too large (%d bytes) — rejected", len(body_bytes))
            return web.Response(status=413, text="payload too large")
        body = body_bytes.decode("utf-8", errors="replace")
        for app in self._apps:
            try:
                event = self._build_event(app, self._decrypt_request(app, body, msg_signature, timestamp, nonce))
                if event is not None:
                    # WeCom retries callbacks on timeout → duplicate inbound messages.
                    if event.message_id and self._is_duplicate(event.message_id):
                        logger.debug("[WecomCallback] Duplicate MsgId %s, skipping", event.message_id)
                        return _ack()
                    if event.source and event.source.user_id:
                        self._user_app_map[self._user_app_key(str(app.get("corp_id") or ""), event.source.user_id)] = app["name"]
                    await self._message_queue.put(event)
                return _ack()  # ack immediately — the reply arrives later via proactive message/send
            except WeComCryptoError:
                continue
            except Exception:
                logger.exception("[WecomCallback] Error handling message")
                break
        return web.Response(status=400, text="invalid callback payload")

    @staticmethod
    def _signature_params(request: web.Request):
        return tuple(request.query.get(k, "") for k in ("msg_signature", "timestamp", "nonce"))

    def _is_duplicate(self, message_id: str) -> bool:
        # Deduplicate: WeCom retries callbacks on timeout, producing duplicate inbound messages (#10305).
        now = time.time()
        if now - self._seen_messages.get(message_id, float("-inf")) < MESSAGE_DEDUP_TTL_SECONDS:
            return True
        self._seen_messages[message_id] = now
        if len(self._seen_messages) > 2000:  # prune expired entries
            cutoff = now - MESSAGE_DEDUP_TTL_SECONDS
            self._seen_messages = {k: v for k, v in self._seen_messages.items() if v > cutoff}
        return False

    async def _poll_loop(self) -> None:
        while True:
            event = await self._message_queue.get()
            try:
                task = asyncio.create_task(self.handle_message(event))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except Exception:
                logger.exception("[WecomCallback] Failed to enqueue event")

    def _decrypt_request(self, app: Dict[str, Any], body: str, msg_signature: str, timestamp: str, nonce: str) -> str:
        encrypt = ET.fromstring(body).findtext("Encrypt", default="")
        return self._crypt_for_app(app).decrypt(msg_signature, timestamp, nonce, encrypt).decode("utf-8")

    def _build_event(self, app: Dict[str, Any], xml_text: str) -> Optional[MessageEvent]:
        root = ET.fromstring(xml_text)
        msg_type = (root.findtext("MsgType") or "").lower()
        # Lifecycle events (enter_agent/subscribe) and non-text types are silently acknowledged.
        if msg_type not in {"text", "event"} or (msg_type == "event" and (root.findtext("Event") or "").lower() in {"enter_agent", "subscribe"}):
            return None
        user_id = root.findtext("FromUserName", default="")
        corp_id = root.findtext("ToUserName", default=app.get("corp_id", ""))
        content = root.findtext("Content", default="").strip() or ("/start" if msg_type == "event" else "")
        msg_id = root.findtext("MsgId") or f"{user_id}:{root.findtext('CreateTime', default='0')}"
        source = self.build_source(chat_id=self._user_app_key(corp_id, user_id), chat_name=user_id, chat_type="dm", user_id=user_id, user_name=user_id)
        return MessageEvent(text=content, message_type=MessageType.TEXT, source=source, raw_message=xml_text, message_id=msg_id)

    def _crypt_for_app(self, app: Dict[str, Any]) -> WXBizMsgCrypt:
        return WXBizMsgCrypt(token=str(app.get("token") or ""), encoding_aes_key=str(app.get("encoding_aes_key") or ""), receive_id=str(app.get("corp_id") or ""))

    def _get_app_by_name(self, name: Optional[str]) -> Optional[Dict[str, Any]]:
        return next((app for app in self._apps if app.get("name") == name), None) if name else None

    async def _get_access_token(self, app: Dict[str, Any]) -> str:
        cached = self._access_tokens.get(app["name"])
        return cached["token"] if cached and cached.get("expires_at", 0) > time.time() + 60 else await self._refresh_access_token(app)

    async def _refresh_access_token(self, app: Dict[str, Any]) -> str:
        resp = await self._http_client.get(_TOKEN_URL, params={"corpid": app.get("corp_id"), "corpsecret": app.get("corp_secret")})
        data = resp.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"WeCom token refresh failed: {data}")
        token = data["access_token"]
        expires_in = int(data.get("expires_in", ACCESS_TOKEN_TTL_SECONDS))
        self._access_tokens[app["name"]] = {"token": token, "expires_at": time.time() + expires_in}
        logger.info("[WecomCallback] Token refreshed for app '%s' (corp=%s), expires in %ss", app.get("name", "default"), app.get("corp_id", ""), expires_in)
        return token
