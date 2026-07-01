"""Nextcloud Talk/Spreed gateway platform adapter.

Native Hermes platform adapter for the local Nextcloud Talk bot integration.
It accepts signed Talk bot webhooks and sends replies through the Talk Bot OCS
API.  It intentionally stays dependency-light: stdlib HTTP server + urllib.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, Optional, Set, Tuple
from urllib.parse import quote, urlencode, urlparse, urlunparse

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import build_session_key

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 30000
DEFAULT_WEBHOOK_HOST = "127.0.0.1"
DEFAULT_WEBHOOK_PORT = 8767
DEFAULT_WEBHOOK_PATH = "/nextcloud-talk/webhook"
_DEFAULT_CHAT_HISTORY_LIMIT = 60
_INDICATOR_REFERENCE_LOOKUP_ATTEMPTS = 5
_INDICATOR_REFERENCE_LOOKUP_DELAY_SECONDS = 0.25


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _short_id(value: Any) -> str:
    """Return a log-safe identifier preview without dumping room/user tokens."""
    text = str(value or "")
    if len(text) <= 8:
        return text
    return f"{text[:4]}…{text[-4:]}"


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


def _session_safe_actor_id(value: Any) -> str:
    """Return a Nextcloud actor id that is safe for Hermes session keys."""
    actor_id = str(value or "").strip()
    if not actor_id:
        return ""
    return actor_id.replace("/", "_").replace("\\", "_").replace("..", "__")


def _floatish(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _basic_auth_header(username: str, password: str) -> str:
    import base64

    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _json_bytes(data: Dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _hmac_hex(secret: str, random_header: str, body: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), random_header.encode("utf-8") + body, hashlib.sha256).hexdigest()


def _verify_incoming_signature(headers: Dict[str, str], body: bytes, secret: str) -> bool:
    random_header = headers.get("x-nextcloud-talk-random", "")
    signature = headers.get("x-nextcloud-talk-signature", "")
    if not random_header or not signature or not secret:
        return False
    expected = _hmac_hex(secret, random_header, body)
    return hmac.compare_digest(expected, signature.lower())


def _extract_message(payload: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    actor_raw = payload.get("actor")
    obj_raw = payload.get("object")
    target_raw = payload.get("target")
    actor: Dict[str, Any] = actor_raw if isinstance(actor_raw, dict) else {}
    obj: Dict[str, Any] = obj_raw if isinstance(obj_raw, dict) else {}
    target: Dict[str, Any] = target_raw if isinstance(target_raw, dict) else {}
    reply_raw = obj.get("inReplyTo")
    in_reply_to: Dict[str, Any] = reply_raw if isinstance(reply_raw, dict) else {}

    meta = {
        "event_type": payload.get("type", ""),
        "actor_id": actor.get("id", ""),
        "actor_name": actor.get("name", ""),
        "actor_type": actor.get("type", ""),
        "message_id": obj.get("id", ""),
        "message_name": obj.get("name", ""),
        "conversation_token": target.get("id") or obj.get("id", ""),
        "conversation_name": target.get("name", ""),
        "reply_to_message_id": in_reply_to.get("id", ""),
    }

    if payload.get("type") != "Create" or obj.get("name") != "message":
        return None, meta

    content = obj.get("content", "")
    if isinstance(content, dict):
        return str(content.get("message", "")).strip(), meta
    if isinstance(content, str):
        try:
            rich = json.loads(content)
            if isinstance(rich, dict):
                return str(rich.get("message", "")).strip(), meta
        except json.JSONDecodeError:
            return content.strip(), meta
    return None, meta


@dataclass
class NextcloudTalkSettings:
    base_url: str
    bot_secret: str
    webhook_host: str = DEFAULT_WEBHOOK_HOST
    webhook_port: int = DEFAULT_WEBHOOK_PORT
    webhook_path: str = DEFAULT_WEBHOOK_PATH
    home_channel: str = ""
    request_timeout: int = 60
    processing_indicator_enabled: bool = False
    processing_indicator_delay_seconds: float = 0.6
    processing_indicator_text: str = "Tron verarbeitet …"
    control_user: str = ""
    control_password: str = ""
    native_typing_enabled: bool = True


def _websocket_url_from_signaling_server(server_url: str) -> str:
    """Return the Spreed HPB websocket URL for a Talk signaling server."""
    parsed = urlparse((server_url or "").rstrip("/"))
    if not parsed.scheme or not parsed.netloc:
        return ""
    scheme = "wss" if parsed.scheme == "https" else "ws" if parsed.scheme == "http" else parsed.scheme
    path = parsed.path.rstrip("/")
    if not path.endswith("/spreed"):
        path = f"{path}/spreed" if path else "/spreed"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


class _NextcloudTalkTypingClient:
    """HPB signaling client that makes the real Nextcloud user appear typing."""

    def __init__(self, adapter: "NextcloudTalkAdapter", token: str):
        self.adapter = adapter
        self.token = str(token)
        self._task: Optional[asyncio.Task] = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._participants: Set[str] = set()
        self._own_signaling_session_id = ""
        self._websocket: Any = None

    async def ensure_started(self) -> None:
        if self._task and not self._task.done():
            logger.info("Nextcloud Talk native typing already running for room=%s", _short_id(self.token))
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name=f"nextcloud-talk-typing:{self.token}")
        logger.info("Nextcloud Talk native typing start requested for room=%s", _short_id(self.token))

    async def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        task = self._task
        if task and not task.done():
            logger.info("Nextcloud Talk native typing stop requested for room=%s", _short_id(self.token))
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Nextcloud Talk native typing stop timed out; cancelling room=%s", _short_id(self.token))
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._task = None

    async def _run(self) -> None:
        try:
            await self._connect_join_and_relay_typing()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Nextcloud Talk native typing client failed for room=%s", _short_id(self.token), exc_info=True)

    async def _connect_join_and_relay_typing(self) -> None:
        import websockets

        room = await asyncio.to_thread(self.adapter._join_room_as_control_user, self.token)
        nextcloud_session_id = str(room.get("sessionId") or room.get("sessionid") or "")
        logger.info(
            "Nextcloud Talk native typing room join: room=%s session_id_present=%s",
            _short_id(self.token),
            bool(nextcloud_session_id),
        )
        settings = await asyncio.to_thread(self.adapter._get_signaling_settings, self.token)
        websocket_url = _websocket_url_from_signaling_server(str(settings.get("server") or ""))
        raw_hello_auth_params = settings.get("helloAuthParams")
        hello_auth_params: Dict[str, Any] = raw_hello_auth_params if isinstance(raw_hello_auth_params, dict) else {}
        logger.info(
            "Nextcloud Talk native typing signaling settings: room=%s mode=%s websocket_url_present=%s hello_versions=%s",
            _short_id(self.token),
            settings.get("signalingMode"),
            bool(websocket_url),
            sorted(hello_auth_params.keys()),
        )
        if not nextcloud_session_id or not websocket_url or not hello_auth_params:
            logger.warning(
                "Nextcloud Talk native typing prerequisites missing: room=%s session_id=%s websocket_url=%s hello_auth=%s",
                _short_id(self.token),
                bool(nextcloud_session_id),
                bool(websocket_url),
                bool(hello_auth_params),
            )
            return

        async with websockets.connect(websocket_url, open_timeout=10, close_timeout=2) as websocket:
            self._websocket = websocket
            logger.info("Nextcloud Talk native typing websocket connected for room=%s", _short_id(self.token))
            welcome = await asyncio.wait_for(websocket.recv(), timeout=10)
            welcome_payload = json.loads(welcome) if isinstance(welcome, str) else json.loads(welcome.decode("utf-8"))
            features = set(welcome_payload.get("welcome", {}).get("features", [])) if isinstance(welcome_payload, dict) else set()
            hello_version = "2.0" if "hello-v2" in features and hello_auth_params.get("2.0") else "1.0"
            auth_params = hello_auth_params.get(hello_version)
            if not auth_params:
                logger.warning(
                    "Nextcloud Talk native typing missing hello auth params: room=%s hello_version=%s",
                    _short_id(self.token),
                    hello_version,
                )
                return
            await websocket.send(json.dumps({
                "id": "1",
                "type": "hello",
                "hello": {
                    "version": hello_version,
                    "auth": {
                        "url": f"{self.adapter.settings.base_url}/ocs/v2.php/apps/spreed/api/v3/signaling/backend",
                        "params": auth_params,
                    },
                    "features": ["chat-relay"],
                },
            }, separators=(",", ":")))
            hello_response = await self._recv_until_type(websocket, "hello")
            self._own_signaling_session_id = str(hello_response.get("hello", {}).get("sessionid") or "")
            logger.info(
                "Nextcloud Talk native typing hello accepted: room=%s own_session_present=%s",
                _short_id(self.token),
                bool(self._own_signaling_session_id),
            )
            await websocket.send(json.dumps({
                "id": "2",
                "type": "room",
                "room": {"roomid": self.token, "sessionid": nextcloud_session_id},
            }, separators=(",", ":")))
            await self._recv_until_type(websocket, "room")
            logger.info(
                "Nextcloud Talk native typing room relay joined: room=%s participants=%d",
                _short_id(self.token),
                len(self._participants),
            )
            await self._broadcast_typing(True)
            await self._receive_events_until_stopped(websocket)
        self._websocket = None

    async def _recv_until_type(self, websocket: Any, expected_type: str) -> Dict[str, Any]:
        while True:
            raw = await asyncio.wait_for(websocket.recv(), timeout=10)
            payload = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict):
                self._handle_signaling_payload(payload)
                if payload.get("type") == expected_type:
                    return payload

    async def _receive_events_until_stopped(self, websocket: Any) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                payload = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
                if isinstance(payload, dict):
                    before = set(self._participants)
                    self._handle_signaling_payload(payload)
                    for session_id in sorted(self._participants - before):
                        logger.info(
                            "Nextcloud Talk native typing participant appeared: room=%s participant=%s",
                            _short_id(self.token),
                            _short_id(session_id),
                        )
                        await self._send_typing_to(session_id, True)
        finally:
            await self._broadcast_typing(False)
            with contextlib.suppress(Exception):
                await websocket.send(json.dumps({"type": "bye", "bye": {}}, separators=(",", ":")))

    def _handle_signaling_payload(self, payload: Dict[str, Any]) -> None:
        event = payload.get("event") if isinstance(payload.get("event"), dict) else None
        if event and event.get("target") == "room":
            event_type = event.get("type")
            if event_type == "join":
                self._add_participants(event.get("join") or [])
            elif event_type == "leave":
                self._remove_participants(event.get("leave") or [])
            return

        # Some Talk/HPB deployments do not emit a separate ``event: join`` for
        # users that were already present when the relay joins.  The initial
        # ``type: room`` response may still carry participant/session objects,
        # and ignoring those leaves native typing technically connected but
        # with zero recipients — a particularly elegant no-op, naturally.
        if payload.get("type") == "room":
            self._add_participants(self._extract_session_ids(payload))

    def _extract_session_ids(self, value: Any) -> Set[str]:
        session_ids: Set[str] = set()
        if isinstance(value, dict):
            session_id = str(value.get("sessionid") or value.get("sessionId") or "")
            if session_id:
                session_ids.add(session_id)
            for child in value.values():
                session_ids.update(self._extract_session_ids(child))
        elif isinstance(value, list):
            for item in value:
                session_ids.update(self._extract_session_ids(item))
        return session_ids

    def _add_participants(self, participants: Iterable[Any]) -> None:
        for participant in participants:
            if isinstance(participant, dict):
                session_id = str(participant.get("sessionid") or participant.get("sessionId") or "")
            else:
                session_id = str(participant or "")
            if session_id and session_id != self._own_signaling_session_id:
                self._participants.add(session_id)

    def _remove_participants(self, participants: Iterable[Any]) -> None:
        for participant in participants:
            if isinstance(participant, dict):
                session_id = str(participant.get("sessionid") or "")
            else:
                session_id = str(participant or "")
            if session_id:
                self._participants.discard(session_id)

    async def _broadcast_typing(self, typing: bool) -> None:
        if not self._websocket:
            logger.info(
                "Nextcloud Talk native typing broadcast skipped (no websocket): room=%s typing=%s",
                _short_id(self.token),
                typing,
            )
            return
        event_type = "startedTyping" if typing else "stoppedTyping"
        logger.info(
            "Nextcloud Talk native typing broadcast: room=%s typing=%s participants_known=%d recipient=room",
            _short_id(self.token),
            typing,
            len(self._participants),
        )
        # Broadcast to the room instead of unicasting to the locally observed
        # participants set. Existing participants usually do not emit a fresh
        # join event after this relay joins, so the set can legitimately be
        # empty while the user is already watching the room. Room-recipient
        # signaling is the protocol-supported way to reach those sessions.
        await self._websocket.send(json.dumps({
            "type": "message",
            "message": {
                "recipient": {"type": "room"},
                "data": {"type": event_type},
            },
        }, separators=(",", ":")))

    async def _send_typing_to(self, session_id: str, typing: bool) -> None:
        if not self._websocket or not session_id:
            return
        event_type = "startedTyping" if typing else "stoppedTyping"
        logger.info(
            "Nextcloud Talk native typing send event: room=%s participant=%s event=%s",
            _short_id(self.token),
            _short_id(session_id),
            event_type,
        )
        await self._websocket.send(json.dumps({
            "type": "message",
            "message": {
                "recipient": {"type": "session", "sessionid": session_id},
                "data": {"type": event_type, "to": session_id},
            },
        }, separators=(",", ":")))


class NextcloudTalkAdapter(BasePlatformAdapter):
    """Webhook-based Nextcloud Talk platform adapter."""

    supports_code_blocks = True
    supports_async_delivery = True
    splits_long_messages = True
    typed_command_prefix = "/"

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("nextcloud_talk"))
        extra = getattr(config, "extra", {}) or {}
        base_url = (
            _env("NEXTCLOUD_TALK_BASE_URL")
            or _env("NEXTCLOUD_BASE_URL")
            or str(extra.get("base_url") or extra.get("server_url") or "")
        ).rstrip("/")
        self.settings = NextcloudTalkSettings(
            base_url=base_url,
            bot_secret=_env("NEXTCLOUD_TALK_BOT_SECRET") or str(extra.get("bot_secret") or ""),
            webhook_host=_env("NEXTCLOUD_TALK_WEBHOOK_HOST") or str(extra.get("webhook_host") or DEFAULT_WEBHOOK_HOST),
            webhook_port=_env_int("NEXTCLOUD_TALK_WEBHOOK_PORT", int(extra.get("webhook_port") or DEFAULT_WEBHOOK_PORT)),
            webhook_path=_env("NEXTCLOUD_TALK_WEBHOOK_PATH") or str(extra.get("webhook_path") or DEFAULT_WEBHOOK_PATH),
            home_channel=_env("NEXTCLOUD_TALK_HOME_CHANNEL") or str(extra.get("home_channel") or ""),
            request_timeout=_env_int("NEXTCLOUD_TALK_REQUEST_TIMEOUT_SECONDS", int(extra.get("request_timeout") or 60)),
            processing_indicator_enabled=_boolish(
                extra.get("processing_indicator_enabled"),
                _boolish(_env("NEXTCLOUD_TALK_PROCESSING_INDICATOR_ENABLED") or None, False),
            ),
            processing_indicator_delay_seconds=max(
                0.0,
                _floatish(
                    extra.get("processing_indicator_delay_seconds")
                    or _env("NEXTCLOUD_TALK_PROCESSING_INDICATOR_DELAY_SECONDS"),
                    0.6,
                ),
            ),
            processing_indicator_text=str(
                extra.get("processing_indicator_text")
                or _env("NEXTCLOUD_TALK_PROCESSING_INDICATOR_TEXT")
                or "Tron verarbeitet …"
            ),
            control_user=_env("NEXTCLOUD_TALK_CONTROL_USER") or _env("NEXTCLOUD_WEBUI_USER"),
            control_password=_env("NEXTCLOUD_TALK_CONTROL_PASSWORD") or _env("NEXTCLOUD_WEBUI_PASSWORD"),
            native_typing_enabled=_boolish(
                extra.get("native_typing_enabled"),
                _boolish(_env("NEXTCLOUD_TALK_NATIVE_TYPING_ENABLED") or None, True),
            ),
        )
        self._typing_clients: Dict[str, _NextcloudTalkTypingClient] = {}
        self._typing_config_skip_logged = False
        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.settings.base_url or not self.settings.bot_secret:
            self._set_fatal_error("missing_config", "NEXTCLOUD_TALK_BASE_URL/NEXTCLOUD_BASE_URL and NEXTCLOUD_TALK_BOT_SECRET are required", retryable=False)
            return False
        self._loop = asyncio.get_running_loop()
        handler = self._make_handler()
        try:
            self._server = ThreadingHTTPServer((self.settings.webhook_host, self.settings.webhook_port), handler)
        except OSError as exc:
            self._set_fatal_error("bind_failed", f"Could not bind Nextcloud Talk webhook: {exc}", retryable=False)
            return False
        self._server_thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True)
        self._server_thread.start()
        self._mark_connected()
        logger.info(
            "[Nextcloud Talk] Webhook listening on http://%s:%s%s",
            self.settings.webhook_host,
            self.settings.webhook_port,
            self.settings.webhook_path,
        )
        logger.info(
            "[Nextcloud Talk] native typing config: enabled=%s control_user_present=%s control_password_present=%s fallback_indicator_enabled=%s fallback_delay=%.2fs",
            self.settings.native_typing_enabled,
            bool(self.settings.control_user),
            bool(self.settings.control_password),
            self.settings.processing_indicator_enabled,
            self.settings.processing_indicator_delay_seconds,
        )
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        await self._stop_all_native_typing_clients()
        if self._server:
            await asyncio.to_thread(self._server.shutdown)
            self._server.server_close()
        self._server = None
        self._server_thread = None

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        chunks = self.truncate_message(content or "", MAX_MESSAGE_LENGTH)
        last_id: Optional[str] = None
        continuations = []
        sent_count = 0
        try:
            for idx, chunk in enumerate(chunks or [""]):
                result = await asyncio.to_thread(self._send_one, chat_id, chunk, reply_to if idx == 0 else None)
                sent_count += 1
                last_id = result.get("message_id") or last_id
                if idx > 0 and last_id:
                    continuations.append(last_id)
            return SendResult(success=True, message_id=last_id, raw_response={"chunks": len(chunks)}, continuation_message_ids=tuple(continuations))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raw_response: Dict[str, Any] = {"detail": detail, "chunks_sent": sent_count, "total_chunks": len(chunks or [""])}
            if sent_count:
                raw_response["partial_delivery"] = True
            return SendResult(
                success=False,
                error=f"HTTP {exc.code}: {detail}",
                raw_response=raw_response,
                retryable=False,
            )
        except Exception as exc:
            raw_response = {"chunks_sent": sent_count, "total_chunks": len(chunks or [""])}
            if sent_count:
                raw_response["partial_delivery"] = True
            return SendResult(success=False, error=str(exc), raw_response=raw_response, retryable=not bool(sent_count))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"id": chat_id, "name": chat_id, "type": "group"}

    def _control_headers(self) -> Dict[str, str]:
        if not (self.settings.control_user and self.settings.control_password):
            raise RuntimeError("Nextcloud Talk control credentials are not configured")
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "OCS-APIRequest": "true",
            "Authorization": _basic_auth_header(self.settings.control_user, self.settings.control_password),
        }

    def _control_request_json(self, path: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = _json_bytes(payload) if payload is not None else None
        request = urllib.request.Request(
            f"{self.settings.base_url}{path}",
            data=body,
            headers=self._control_headers(),
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.settings.request_timeout) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _join_room_as_control_user(self, token: str) -> Dict[str, Any]:
        quoted_token = quote(str(token), safe="")
        data = self._control_request_json(
            f"/ocs/v2.php/apps/spreed/api/v4/room/{quoted_token}/participants/active?format=json",
            method="POST",
            payload={"force": False},
        )
        ocs = data.get("ocs", {}) if isinstance(data, dict) else {}
        room = ocs.get("data") if isinstance(ocs, dict) else None
        return room if isinstance(room, dict) else {}

    def _get_signaling_settings(self, token: str) -> Dict[str, Any]:
        quoted_token = quote(str(token), safe="")
        data = self._control_request_json(
            f"/ocs/v2.php/apps/spreed/api/v3/signaling/settings?format=json&token={quoted_token}",
            method="GET",
        )
        ocs = data.get("ocs", {}) if isinstance(data, dict) else {}
        settings = ocs.get("data") if isinstance(ocs, dict) else None
        return settings if isinstance(settings, dict) else {}

    def _can_native_type(self) -> bool:
        return bool(self.settings.native_typing_enabled and self.settings.control_user and self.settings.control_password)

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not chat_id:
            logger.info("Nextcloud Talk native typing skipped: missing chat_id")
            return
        if not self._can_native_type():
            if not self._typing_config_skip_logged:
                self._typing_config_skip_logged = True
                logger.warning(
                    "Nextcloud Talk native typing skipped: enabled=%s control_user_present=%s control_password_present=%s",
                    self.settings.native_typing_enabled,
                    bool(self.settings.control_user),
                    bool(self.settings.control_password),
                )
            return
        client = self._typing_clients.get(str(chat_id))
        if client is None:
            client = _NextcloudTalkTypingClient(self, str(chat_id))
            self._typing_clients[str(chat_id)] = client
            logger.info("Nextcloud Talk native typing client created for room=%s", _short_id(chat_id))
        else:
            logger.info("Nextcloud Talk native typing client reused for room=%s", _short_id(chat_id))
        await client.ensure_started()

    async def stop_typing(self, chat_id: str) -> None:
        client = self._typing_clients.pop(str(chat_id), None)
        if client is not None:
            logger.info("Nextcloud Talk native typing client stopping for room=%s", _short_id(chat_id))
            await client.stop()
        elif chat_id:
            logger.info("Nextcloud Talk native typing stop requested but no client exists for room=%s", _short_id(chat_id))

    async def _stop_all_native_typing_clients(self) -> None:
        clients = list(self._typing_clients.items())
        self._typing_clients.clear()
        for _chat_id, client in clients:
            with contextlib.suppress(Exception):
                await client.stop()

    def _send_one(
        self,
        token: str,
        message: str,
        reply_to: Optional[str] = None,
        _resolve_reference_id: bool = False,
    ) -> Dict[str, Any]:
        message = (message or "").strip()
        reference_id = secrets.token_hex(32)
        payload: Dict[str, Any] = {"message": message, "referenceId": reference_id}
        if reply_to:
            try:
                payload["replyTo"] = int(reply_to)
            except (TypeError, ValueError):
                pass
        body = _json_bytes(payload)
        random_header = secrets.token_hex(32)
        # Spreed/Talk 23 validates bot sends against random + plain message,
        # not random + JSON body.
        signature = _hmac_hex(self.settings.bot_secret, random_header, message.encode("utf-8"))
        request = urllib.request.Request(
            f"{self.settings.base_url}/ocs/v2.php/apps/spreed/api/v1/bot/{token}/message",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "OCS-APIRequest": "true",
                "X-Nextcloud-Talk-Bot-Random": random_header,
                "X-Nextcloud-Talk-Bot-Signature": signature,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.settings.request_timeout) as resp:
            raw = resp.read()
            status = getattr(resp, "status", 200)
        data = json.loads(raw.decode("utf-8")) if raw else {}
        message_id = None
        try:
            message_id = str(data.get("ocs", {}).get("data", {}).get("id") or "") or None
        except AttributeError:
            message_id = None
        # Bot responses can be data=null, so message id must be resolved from chat
        # history by reference ID using control credentials.
        if not message_id and self._can_cleanup_indicator_messages() and _resolve_reference_id:
            message_id = self._resolve_message_id_by_reference_id(token, reference_id)
        return {
            "status": status,
            "response": data,
            "message_id": message_id,
            "reference_id": reference_id,
        }

    def _can_cleanup_indicator_messages(self) -> bool:
        return bool(self.settings.control_user and self.settings.control_password)

    def _resolve_message_id_by_reference_id(
        self,
        token: str,
        reference_id: str,
        attempts: int = _INDICATOR_REFERENCE_LOOKUP_ATTEMPTS,
        retry_delay_seconds: float = _INDICATOR_REFERENCE_LOOKUP_DELAY_SECONDS,
    ) -> Optional[str]:
        if not self._can_cleanup_indicator_messages():
            return None
        if not reference_id:
            return None
        quoted_token = quote(str(token), safe="")
        limit = _DEFAULT_CHAT_HISTORY_LIMIT
        params = urlencode(
            {
                "lookIntoFuture": "0",
                "setReadMarker": "0",
                "limit": str(limit),
            },
        )
        headers = {
            "Accept": "application/json",
            "OCS-APIRequest": "true",
            "Authorization": _basic_auth_header(self.settings.control_user, self.settings.control_password),
        }
        max_attempts = max(1, attempts)
        for attempt in range(max_attempts):
            try:
                request = urllib.request.Request(
                    f"{self.settings.base_url}/ocs/v2.php/apps/spreed/api/v1/chat/{quoted_token}?{params}",
                    headers=headers,
                    method="GET",
                )
                with urllib.request.urlopen(request, timeout=self.settings.request_timeout) as resp:
                    raw = resp.read()
                if not raw:
                    # Empty body: transient; can happen with 304-like reverse-proxy
                    # behaviour.  Retry or exhaust attempts (implicit None return).
                    if attempt + 1 < max_attempts:
                        time.sleep(retry_delay_seconds)
                    continue
                data = json.loads(raw.decode("utf-8"))
                ocs_payload = data.get("ocs", {})
                messages = ocs_payload.get("data") if isinstance(ocs_payload, dict) else None
                if isinstance(messages, list):
                    for item in messages:
                        if not isinstance(item, dict):
                            continue
                        if str(item.get("referenceId") or "") == reference_id:
                            candidate = item.get("id")
                            if candidate is not None:
                                return str(candidate)
            except urllib.error.HTTPError as exc:
                status = exc.code
                if status in (401, 403, 404, 412):
                    return None
                if status == 304:
                    logger.debug("Nextcloud Talk chat message history lookup returned 304; retrying reference lookup.")
                else:
                    logger.debug("Nextcloud Talk reference lookup failed with status %s", status)
                if attempt + 1 >= max_attempts:
                    return None
            except (OSError, ValueError, KeyError, TypeError):
                logger.debug(
                    "Nextcloud Talk reference lookup failed for token=%s reference=%s",
                    quoted_token,
                    reference_id[:8],
                )
                if attempt + 1 >= max_attempts:
                    return None
            if attempt + 1 < max_attempts:
                time.sleep(retry_delay_seconds)
        return None

    def _delete_message(self, token: str, message_id: str) -> Dict[str, Any]:
        """Delete a Talk message via the normal chat API using control credentials.

        The bot endpoint can create messages but does not expose edit/delete in
        Talk 23.  If configured, we use a real Talk participant credential to
        remove our temporary processing indicator after the final Hermes reply
        lands.  Without those credentials the caller should simply skip the
        indicator rather than leave chat litter behind. A butler cleans up his
        tray, after all.
        """
        if not (self.settings.control_user and self.settings.control_password):
            raise RuntimeError("Nextcloud Talk control credentials are not configured")
        quoted_token = quote(str(token), safe="")
        quoted_message_id = quote(str(message_id), safe="")
        request = urllib.request.Request(
            f"{self.settings.base_url}/ocs/v2.php/apps/spreed/api/v1/chat/{quoted_token}/{quoted_message_id}",
            headers={
                "Accept": "application/json",
                "OCS-APIRequest": "true",
                "Authorization": _basic_auth_header(self.settings.control_user, self.settings.control_password),
            },
            method="DELETE",
        )
        with urllib.request.urlopen(request, timeout=self.settings.request_timeout) as resp:
            raw = resp.read()
            status = getattr(resp, "status", 200)
        data = json.loads(raw.decode("utf-8")) if raw else {}
        return {"status": status, "response": data}

    async def _await_message_processing_complete(self, event: MessageEvent) -> None:
        """Wait for the background task that BasePlatformAdapter spawned.

        ``handle_message()`` intentionally returns immediately after scheduling
        ``_process_message_background``.  The Talk status indicator, however,
        must wrap the actual agent turn rather than that dispatcher call.
        Follow ownership hand-offs so queued follow-up drain tasks do not leave
        the temporary indicator behind the final response.
        """
        session_store = getattr(self, "_session_store", None)
        if session_store is not None and hasattr(session_store, "_generate_session_key"):
            session_key = session_store._generate_session_key(event.source)
        else:
            session_key = build_session_key(
                event.source,
                group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
                thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
                profile=event.source.profile,
            )
        seen: set[asyncio.Task] = set()
        while True:
            task = self._session_tasks.get(session_key)
            if task is None or task in seen:
                return
            seen.add(task)
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                raise
            except Exception:
                # The background task logs/sends user-visible failures itself;
                # the indicator cleanup should still run.
                logger.debug("Nextcloud Talk background processing finished with error", exc_info=True)

    async def _handle_message_with_processing_indicator(self, event: MessageEvent) -> None:
        indicator_task: Optional[asyncio.Task] = None
        indicator_message_id: Optional[str] = None
        indicator_reference_id: Optional[str] = None

        async def _send_indicator_after_delay() -> Tuple[Optional[str], Optional[str]]:
            await asyncio.sleep(self.settings.processing_indicator_delay_seconds)
            result = await asyncio.to_thread(
                self._send_one,
                event.source.chat_id,
                self.settings.processing_indicator_text,
                event.message_id,
                True,
            )
            reference_id = result.get("reference_id")
            return result.get("message_id") or None, reference_id

        # The removable chat-message indicator is now opt-in only. Native Talk
        # typing is the preferred user-visible progress signal; the temporary
        # "Tron verarbeitet …" message remains available as a fallback for
        # deployments where native typing is unavailable or unreliable.
        can_cleanup = self._can_cleanup_indicator_messages()
        use_message_fallback = True
        if (
            use_message_fallback
            and self.settings.processing_indicator_enabled
            and can_cleanup
            and event.source.chat_id
            and self.settings.processing_indicator_text.strip()
        ):
            indicator_task = asyncio.create_task(_send_indicator_after_delay())

        try:
            await self.handle_message(event)
            await self._await_message_processing_complete(event)
        finally:
            if indicator_task:
                if indicator_task.done():
                    try:
                        indicator_message_id, indicator_reference_id = indicator_task.result()
                    except Exception:
                        logger.debug("Nextcloud Talk processing indicator send failed", exc_info=True)
                else:
                    indicator_task.cancel()
                    try:
                        await indicator_task
                    except asyncio.CancelledError:
                        pass

            if not indicator_message_id and indicator_reference_id:
                indicator_message_id = await asyncio.to_thread(
                    self._resolve_message_id_by_reference_id,
                    event.source.chat_id,
                    indicator_reference_id,
                    attempts=6,
                    retry_delay_seconds=_INDICATOR_REFERENCE_LOOKUP_DELAY_SECONDS * 0.5,
                )
            if indicator_message_id:
                try:
                    await asyncio.to_thread(self._delete_message, event.source.chat_id, indicator_message_id)
                except Exception:
                    logger.debug("Nextcloud Talk processing indicator cleanup failed", exc_info=True)

    def _make_handler(self):
        adapter = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
                raw = _json_bytes(payload)
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    self._send_json(200, {"ok": True, "service": "nextcloud-talk-platform"})
                    return
                self._send_json(404, {"ok": False, "error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                if self.path.rstrip("/") != adapter.settings.webhook_path.rstrip("/"):
                    self._send_json(404, {"ok": False, "error": "not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._send_json(411, {"ok": False, "error": "invalid content length"})
                    return
                body = self.rfile.read(length)
                headers = {key.lower(): value for key, value in self.headers.items()}
                status, payload = adapter._handle_webhook_sync(headers, body)
                self._send_json(status, payload)

            def log_message(self, format: str, *args: Any) -> None:
                logger.debug("Nextcloud Talk webhook: " + format, *args)

        return Handler

    def _handle_webhook_sync(self, headers: Dict[str, str], body: bytes) -> Tuple[int, Dict[str, Any]]:
        if not self.settings.bot_secret:
            return 503, {"ok": False, "error": "webhook not configured"}
        if not _verify_incoming_signature(headers, body, self.settings.bot_secret):
            return 401, {"ok": False, "error": "invalid signature"}
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return 400, {"ok": False, "error": "invalid json"}
        text, meta = _extract_message(payload)
        if not text:
            return 202, {"ok": True, "ignored": "no visible user message"}
        actor_id = str(meta.get("actor_id", ""))
        actor_type = str(meta.get("actor_type", ""))
        if actor_id.startswith("bots/") or actor_type == "Application":
            return 202, {"ok": True, "ignored": "bot/application message"}
        token = str(meta.get("conversation_token") or "")
        if not token:
            return 202, {"ok": True, "ignored": "missing conversation token"}
        if not self._loop or self._loop.is_closed():
            return 503, {"ok": False, "error": "adapter event loop not ready"}
        source = self.build_source(
            chat_id=token,
            chat_name=str(meta.get("conversation_name") or token),
            chat_type="group",
            user_id=_session_safe_actor_id(actor_id) or None,
            user_id_alt=actor_id or None,
            user_name=str(meta.get("actor_name") or ""),
            message_id=str(meta.get("message_id") or ""),
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=str(meta.get("message_id") or "") or None,
            reply_to_message_id=str(meta.get("reply_to_message_id") or "") or None,
        )
        def _schedule() -> None:
            task = asyncio.create_task(self._handle_message_with_processing_indicator(event))

            def _log_failure(done: asyncio.Task) -> None:
                try:
                    done.result()
                except Exception:
                    logger.exception("Nextcloud Talk message dispatch failed")

            task.add_done_callback(_log_failure)

        self._loop.call_soon_threadsafe(_schedule)
        return 202, {"ok": True, "accepted": True}


def check_requirements() -> bool:
    return bool((_env("NEXTCLOUD_TALK_BASE_URL") or _env("NEXTCLOUD_BASE_URL")) and _env("NEXTCLOUD_TALK_BOT_SECRET"))


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(
        (_env("NEXTCLOUD_TALK_BASE_URL") or _env("NEXTCLOUD_BASE_URL") or extra.get("base_url") or extra.get("server_url"))
        and (_env("NEXTCLOUD_TALK_BOT_SECRET") or extra.get("bot_secret"))
    )


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> Optional[dict]:
    base_url = _env("NEXTCLOUD_TALK_BASE_URL") or _env("NEXTCLOUD_BASE_URL")
    secret = _env("NEXTCLOUD_TALK_BOT_SECRET")
    if not (base_url and secret):
        return None
    home_channel = _env("NEXTCLOUD_TALK_HOME_CHANNEL")
    seed: Dict[str, Any] = {
        "base_url": base_url,
        "webhook_host": _env("NEXTCLOUD_TALK_WEBHOOK_HOST", DEFAULT_WEBHOOK_HOST),
        "webhook_port": _env("NEXTCLOUD_TALK_WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT)),
        "webhook_path": _env("NEXTCLOUD_TALK_WEBHOOK_PATH", DEFAULT_WEBHOOK_PATH),
        # Do not echo secret into config status surfaces.
    }
    if home_channel:
        seed["home_channel"] = {"chat_id": home_channel, "name": _env("NEXTCLOUD_TALK_HOME_CHANNEL_NAME", "Tron")}
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id=None,
    media_files=None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process delivery for cron/send_message.

    Matches the platform registry standalone_sender_fn contract:
    ``(pconfig, chat_id, message, *, thread_id=None, media_files=None,
    force_document=False)``.  Media attachments are currently unsupported by
    the Talk bot endpoint, so they are ignored rather than causing text
    delivery to fail.
    """
    cfg = pconfig if isinstance(pconfig, PlatformConfig) else PlatformConfig(enabled=True, extra={})
    adapter = NextcloudTalkAdapter(cfg)
    result = await adapter.send(chat_id, message, reply_to=thread_id)
    return {"success": result.success, "message_id": result.message_id, "error": result.error}


def _build_adapter(config: PlatformConfig):
    return NextcloudTalkAdapter(config)


def register(ctx) -> None:
    ctx.register_platform(
        name="nextcloud_talk",
        label="Nextcloud Talk",
        adapter_factory=_build_adapter,
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["NEXTCLOUD_TALK_BOT_SECRET", "NEXTCLOUD_TALK_BASE_URL"],
        install_hint="No extra Python dependencies required; configure NEXTCLOUD_TALK_BASE_URL or NEXTCLOUD_BASE_URL.",
        env_enablement_fn=_env_enablement,
        standalone_sender_fn=_standalone_send,
        cron_deliver_env_var="NEXTCLOUD_TALK_HOME_CHANNEL",
        allowed_users_env="NEXTCLOUD_TALK_ALLOWED_USERS",
        allow_all_env="NEXTCLOUD_TALK_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="💬",
        allow_update_command=True,
        pii_safe=True,
        platform_hint=(
            "You are communicating through Nextcloud Talk. Slash commands use plain text; "
            "Talk has no Telegram-style inline keyboard callbacks, so render choices as numbered lists."
        ),
    )
