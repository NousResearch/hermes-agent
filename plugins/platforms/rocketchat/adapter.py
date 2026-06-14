"""Rocket.Chat gateway adapter."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_media_bytes,
    proxy_kwargs_for_aiohttp,
    resolve_channel_prompt,
    resolve_proxy_url,
)
from gateway.platforms.helpers import MessageDeduplicator
from plugins.platforms.rocketchat.auth import (
    RocketChatBootstrapError,
    artifact_path,
    bootstrap_config_from_env,
    bootstrap_enabled,
    bootstrap_via_oauth,
    resolve_runtime_credentials,
)
from utils import is_truthy_value

logger = logging.getLogger(__name__)

# Register the dynamic plugin-backed platform enum member on import.
ROCKETCHAT_PLATFORM = Platform("rocketchat")

MAX_MESSAGE_LENGTH = 4096
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MAX_DELAY = 60.0
_RECONNECT_JITTER = 0.25
_DDP_VERSIONS = ["1", "pre2", "pre1"]

_ROOM_TYPE_TO_CHAT_TYPE = {
    "d": "dm",
    "c": "channel",
    "p": "group",
}


def _split_csv(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {part.strip() for part in str(value).split(",") if part.strip()}


def _coerce_bool(extra: dict[str, Any], key: str, env_var: str, default: bool) -> bool:
    if key in extra:
        return is_truthy_value(extra.get(key), default=default)
    raw = os.getenv(env_var)
    if raw is None or not str(raw).strip():
        return default
    return is_truthy_value(raw, default=default)


def _room_name(room: dict[str, Any]) -> str:
    return (
        str(room.get("fname") or "").strip()
        or str(room.get("name") or "").strip()
        or str(room.get("_id") or room.get("rid") or "Rocket.Chat")
    )


def _room_chat_type(room: dict[str, Any]) -> str:
    room_type = str(room.get("t") or "").lower()
    return _ROOM_TYPE_TO_CHAT_TYPE.get(room_type, "channel")


def _message_is_system(message: dict[str, Any]) -> bool:
    return bool(message.get("t"))


def _message_sender_id(message: dict[str, Any]) -> str:
    user = message.get("u") or {}
    return str(user.get("_id") or "").strip()


def _message_sender_name(message: dict[str, Any]) -> str:
    user = message.get("u") or {}
    return (
        str(user.get("username") or "").strip()
        or str(user.get("name") or "").strip()
        or _message_sender_id(message)
    )


class RocketChatAdapter(BasePlatformAdapter):
    """Gateway adapter for Rocket.Chat via REST + DDP."""

    supports_code_blocks = True
    typed_command_prefix = "!"

    def __init__(self, config: PlatformConfig):
        super().__init__(config, ROCKETCHAT_PLATFORM)

        extra = getattr(config, "extra", {}) or {}
        self._extra = extra
        self._base_url = (extra.get("url") or os.getenv("ROCKETCHAT_URL", "")).rstrip("/")
        self._command_prefix = str(
            extra.get("command_prefix")
            or os.getenv("ROCKETCHAT_COMMAND_PREFIX", "!")
            or "!"
        ).strip() or "!"
        self._require_mention = _coerce_bool(
            extra, "require_mention", "ROCKETCHAT_REQUIRE_MENTION", True
        )
        self._free_response_rooms = _split_csv(
            extra.get("free_response_rooms")
            if "free_response_rooms" in extra
            else os.getenv("ROCKETCHAT_FREE_RESPONSE_ROOMS", "")
        )
        self._allowed_rooms = _split_csv(
            extra.get("allowed_rooms")
            if "allowed_rooms" in extra
            else os.getenv("ROCKETCHAT_ALLOWED_ROOMS", "")
        )
        self._allowed_users = _split_csv(
            extra.get("allowed_users")
            if "allowed_users" in extra
            else os.getenv("ROCKETCHAT_ALLOWED_USERS", "")
        )
        self._allow_all_users = _coerce_bool(
            extra, "allow_all_users", "ROCKETCHAT_ALLOW_ALL_USERS", False
        )

        self._session: Any = None
        self._ws: Any = None
        self._ws_task: Optional[asyncio.Task] = None
        self._closing = False

        self._proxy = resolve_proxy_url(platform_env_var="ROCKETCHAT_PROXY")
        self._session_kwargs, self._request_kwargs = proxy_kwargs_for_aiohttp(self._proxy)

        self._runtime_creds = None
        self._bot_user_id = ""
        self._bot_username = ""
        self._bot_name = ""

        self._room_cache: dict[str, dict[str, Any]] = {}
        self._room_subscriptions: dict[str, str] = {}
        self._dedup = MessageDeduplicator()
        self._id_counter = 0
        self._method_waiters: dict[str, asyncio.Future] = {}

    async def connect(self) -> bool:
        """Validate auth, warm caches, and start the DDP listener."""
        try:
            self._runtime_creds = resolve_runtime_credentials(
                self._base_url,
                extra=self._extra,
            )
        except RocketChatBootstrapError as exc:
            logger.error("Rocket.Chat: %s", exc)
            return False

        if not self._base_url:
            logger.error("Rocket.Chat: ROCKETCHAT_URL is not configured")
            return False

        import aiohttp

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            **self._session_kwargs,
        )
        self._closing = False

        status, me = await self._api_get("me")
        if status != 200 or not isinstance(me, dict) or not me.get("_id"):
            await self._session.close()
            self._session = None
            logger.error("Rocket.Chat: %s", self._auth_failure_message(status, me))
            return False

        self._bot_user_id = str(me.get("_id") or "").strip()
        self._bot_username = str(me.get("username") or "").strip()
        self._bot_name = str(me.get("name") or "").strip() or self._bot_username

        await self._refresh_room_cache()

        self._ws_task = asyncio.create_task(self._ws_loop())
        self._mark_connected()
        logger.info(
            "Rocket.Chat: authenticated as @%s (%s) on %s",
            self._bot_username or self._bot_name,
            self._bot_user_id,
            self._base_url,
        )
        return True

    async def disconnect(self) -> None:
        """Stop websocket loop and close HTTP session."""
        self._closing = True

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
        self._ws_task = None

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        self._room_subscriptions.clear()
        self._method_waiters.clear()
        self._mark_disconnected()
        logger.info("Rocket.Chat: disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message to a room or DM."""
        if not content:
            return SendResult(success=True)

        formatted = self.format_message(content)
        thread_id = self._resolve_thread_id(metadata, reply_to)

        last_result = SendResult(success=True)
        for chunk in self.truncate_message(formatted, MAX_MESSAGE_LENGTH):
            payload = {"message": {"rid": chat_id, "msg": chunk}}
            if thread_id:
                payload["message"]["tmid"] = thread_id
                payload["message"]["tshow"] = False
            status, body = await self._api_post("chat.sendMessage", payload)
            if status != 200 or not isinstance(body, dict) or not body.get("success"):
                return self._send_error_result("Rocket.Chat send failed", status, body)
            message = body.get("message") or {}
            last_result = SendResult(
                success=True,
                message_id=str(message.get("_id") or "").strip() or None,
                raw_response=body,
            )
        return last_result

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """Edit a previously sent Rocket.Chat message."""
        del finalize
        payload = {
            "roomId": chat_id,
            "msgId": message_id,
            "text": self.format_message(content),
        }
        status, body = await self._api_post("chat.update", payload)
        if status != 200 or not isinstance(body, dict) or not body.get("success"):
            return self._send_error_result("Rocket.Chat edit failed", status, body)
        message = body.get("message") or {}
        return SendResult(
            success=True,
            message_id=str(message.get("_id") or message_id).strip() or message_id,
            raw_response=body,
        )

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._send_remote_media(
            chat_id=chat_id,
            media_url=image_url,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        del kwargs
        return await self._send_local_media(
            chat_id=chat_id,
            file_path=image_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        del kwargs
        return await self._send_local_media(
            chat_id=chat_id,
            file_path=file_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            file_name=file_name,
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return cached room metadata or a minimal fallback."""
        room = self._room_cache.get(str(chat_id), {})
        return {
            "name": _room_name(room) if room else str(chat_id),
            "type": _room_chat_type(room) if room else "channel",
        }

    def format_message(self, content: str) -> str:
        return super().format_message(content)

    async def _send_remote_media(
        self,
        *,
        chat_id: str,
        media_url: str,
        caption: Optional[str],
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> SendResult:
        import aiohttp

        try:
            async with self._session.get(
                media_url,
                timeout=aiohttp.ClientTimeout(total=60),
                **self._request_kwargs,
            ) as resp:
                if resp.status >= 400:
                    return SendResult(
                        success=False,
                        error=f"Rocket.Chat media download failed ({resp.status})",
                        retryable=resp.status >= 500 or resp.status == 429,
                    )
                data = await resp.read()
                filename = media_url.rsplit("/", 1)[-1].split("?")[0] or "attachment"
                content_type = resp.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        except Exception as exc:
            return SendResult(success=False, error=f"Rocket.Chat media download failed: {exc}")

        return await self._upload_and_confirm(
            chat_id=chat_id,
            file_bytes=data,
            filename=filename,
            content_type=content_type,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def _send_local_media(
        self,
        *,
        chat_id: str,
        file_path: str,
        caption: Optional[str],
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
        file_name: Optional[str] = None,
    ) -> SendResult:
        safe_path = self.validate_media_delivery_path(file_path)
        if not safe_path:
            return SendResult(success=False, error=f"Unsafe media path: {file_path}")

        path = Path(safe_path)
        if not path.exists():
            return SendResult(success=False, error=f"Media file does not exist: {safe_path}")

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return await self._upload_and_confirm(
            chat_id=chat_id,
            file_bytes=path.read_bytes(),
            filename=file_name or path.name,
            content_type=content_type,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def _upload_and_confirm(
        self,
        *,
        chat_id: str,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        caption: Optional[str],
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> SendResult:
        thread_id = self._resolve_thread_id(metadata, reply_to)
        upload_status, upload_body = await self._upload_room_media(
            chat_id=chat_id,
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
        )
        if upload_status != 200 or not isinstance(upload_body, dict) or not upload_body.get("success"):
            return self._send_error_result("Rocket.Chat media upload failed", upload_status, upload_body)

        file_id = str(((upload_body.get("file") or {}).get("_id")) or "").strip()
        if not file_id:
            return SendResult(success=False, error="Rocket.Chat media upload did not return a file ID")

        confirm_payload: dict[str, Any] = {}
        if caption:
            confirm_payload["msg"] = self.format_message(caption)
        if thread_id:
            confirm_payload["tmid"] = thread_id
        status, body = await self._api_post(
            f"rooms.mediaConfirm/{chat_id}/{file_id}",
            confirm_payload,
        )
        if status != 200 or not isinstance(body, dict) or not body.get("success"):
            return self._send_error_result("Rocket.Chat media confirm failed", status, body)
        message = body.get("message") or {}
        return SendResult(
            success=True,
            message_id=str(message.get("_id") or "").strip() or None,
            raw_response=body,
        )

    async def _upload_room_media(
        self,
        *,
        chat_id: str,
        file_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> tuple[int, Any]:
        import aiohttp

        form = aiohttp.FormData()
        form.add_field(
            "file",
            file_bytes,
            filename=filename,
            content_type=content_type or "application/octet-stream",
        )
        headers = {
            "X-Auth-Token": self._runtime_creds.auth_token,
            "X-User-Id": self._runtime_creds.user_id,
        }
        return await self._request(
            "POST",
            f"rooms.media/{chat_id}",
            headers=headers,
            data=form,
        )

    async def _refresh_room_cache(self) -> None:
        status, body = await self._api_get("rooms.get")
        if status != 200 or not isinstance(body, dict):
            logger.debug("Rocket.Chat: room cache refresh failed: %s", body)
            return

        for room in body.get("update") or []:
            if not isinstance(room, dict):
                continue
            room_id = str(room.get("_id") or room.get("rid") or "").strip()
            if room_id:
                self._room_cache[room_id] = room
        for room in body.get("remove") or []:
            if isinstance(room, dict):
                room_id = str(room.get("_id") or room.get("rid") or "").strip()
            else:
                room_id = str(room or "").strip()
            if room_id:
                self._room_cache.pop(room_id, None)

    async def _api_get(self, path: str) -> tuple[int, Any]:
        return await self._request("GET", path, headers=self._json_headers())

    async def _api_post(self, path: str, payload: dict[str, Any]) -> tuple[int, Any]:
        return await self._request("POST", path, headers=self._json_headers(), json=payload)

    async def _request(self, method: str, path: str, **kwargs: Any) -> tuple[int, Any]:
        if self._session is None:
            return 503, {"error": "Rocket.Chat session is not connected"}

        import aiohttp

        url = f"{self._base_url}/api/v1/{path.lstrip('/')}"
        try:
            async with self._session.request(
                method,
                url,
                timeout=aiohttp.ClientTimeout(total=60),
                **self._request_kwargs,
                **kwargs,
            ) as resp:
                body = await self._decode_response(resp)
                return resp.status, body
        except aiohttp.ClientError as exc:
            return 599, {"error": f"Rocket.Chat network error: {exc}"}

    async def _decode_response(self, resp) -> Any:
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "application/json" in content_type:
            try:
                return await resp.json()
            except Exception:
                return await resp.text()
        return await resp.text()

    def _json_headers(self) -> dict[str, str]:
        return {
            "X-Auth-Token": self._runtime_creds.auth_token,
            "X-User-Id": self._runtime_creds.user_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _send_error_result(self, prefix: str, status: int, body: Any) -> SendResult:
        message = self._api_error_message(status, body)
        retryable = status in {429, 500, 502, 503, 504, 599}
        return SendResult(
            success=False,
            error=f"{prefix}: {message}",
            raw_response=body,
            retryable=retryable,
        )

    def _api_error_message(self, status: int, body: Any) -> str:
        if isinstance(body, dict):
            detail = body.get("error") or body.get("message") or body.get("status")
            if detail:
                if status == 401 and self._runtime_creds and self._runtime_creds.auth_type != "pat":
                    return f"{detail}. {self._rebootstrap_hint()}"
                return str(detail)
        if isinstance(body, str) and body.strip():
            return body.strip()
        if status == 401 and self._runtime_creds and self._runtime_creds.auth_type != "pat":
            return self._rebootstrap_hint()
        return f"HTTP {status}"

    def _auth_failure_message(self, status: int, body: Any) -> str:
        detail = self._api_error_message(status, body)
        if status == 401 and self._runtime_creds and self._runtime_creds.auth_type != "pat":
            return f"Rocket.Chat auth token expired or invalid. {self._rebootstrap_hint()}"
        return f"Rocket.Chat authentication failed: {detail}"

    def _rebootstrap_hint(self) -> str:
        if bootstrap_enabled(self._extra):
            return (
                "Re-run the Rocket.Chat bootstrap flow to refresh the stored runtime credentials "
                f"({artifact_path(self._extra)})"
            )
        return "Check ROCKETCHAT_USER_ID / ROCKETCHAT_AUTH_TOKEN or switch to a PAT"

    def _resolve_thread_id(
        self,
        metadata: Optional[dict[str, Any]],
        reply_to: Optional[str],
    ) -> Optional[str]:
        metadata = metadata or {}
        thread_id = str(metadata.get("thread_id") or "").strip()
        if thread_id:
            return thread_id
        return None

    async def _ws_loop(self) -> None:
        delay = _RECONNECT_BASE_DELAY
        while not self._closing:
            try:
                await self._ws_connect_and_listen()
                delay = _RECONNECT_BASE_DELAY
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                err_text = str(exc).lower()
                if "401" in err_text or "403" in err_text or "not authorized" in err_text:
                    message = self._auth_failure_message(401, {"error": str(exc)})
                    logger.error("Rocket.Chat websocket auth failed: %s", message)
                    self._set_fatal_error("auth", message, retryable=False)
                    await self._notify_fatal_error()
                    return
                logger.warning(
                    "Rocket.Chat websocket error: %s — reconnecting in %.0fs",
                    exc,
                    delay,
                )

            if self._closing:
                return

            jitter = delay * _RECONNECT_JITTER * random.random()
            await asyncio.sleep(delay + jitter)
            delay = min(delay * 2, _RECONNECT_MAX_DELAY)

    async def _ws_connect_and_listen(self) -> None:
        import aiohttp

        ws_url = re.sub(r"^http", "ws", self._base_url) + "/websocket"
        self._ws = await self._session.ws_connect(
            ws_url,
            heartbeat=30.0,
            **self._request_kwargs,
        )
        self._room_subscriptions.clear()
        self._method_waiters.clear()

        await self._ws.send_json(
            {
                "msg": "connect",
                "version": _DDP_VERSIONS[0],
                "support": _DDP_VERSIONS,
            }
        )
        await self._await_ddp_connected()
        await self._ddp_login()
        await self._refresh_room_cache()
        await self._subscribe_all_rooms()

        async for raw_msg in self._ws:
            if self._closing:
                return

            if raw_msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    frame = json.loads(raw_msg.data)
                except (TypeError, json.JSONDecodeError):
                    continue
                await self._handle_ddp_frame(frame)
            elif raw_msg.type in {
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.ERROR,
            }:
                break

    async def _await_ddp_connected(self) -> None:
        while True:
            frame = await self._ws.receive_json()
            msg = frame.get("msg")
            if msg == "connected":
                return
            if msg == "ping":
                await self._ws.send_json({"msg": "pong"})
            elif msg == "failed":
                raise RuntimeError(f"DDP connect failed: {frame}")

    async def _ddp_login(self) -> None:
        login_id = self._next_id("login")
        future = self._create_waiter(login_id)
        await self._ws.send_json(
            {
                "msg": "method",
                "method": "login",
                "id": login_id,
                "params": [{"resume": self._runtime_creds.auth_token}],
            }
        )
        result = await future
        if result.get("error"):
            raise RuntimeError(str(result["error"]))
        data = result.get("result") or {}
        user_id = str(data.get("id") or "").strip()
        if user_id and not self._bot_user_id:
            self._bot_user_id = user_id

    async def _handle_ddp_frame(self, frame: dict[str, Any]) -> None:
        msg = frame.get("msg")
        if msg == "ping":
            await self._ws.send_json({"msg": "pong"})
            return
        if msg == "result":
            future = self._method_waiters.pop(str(frame.get("id") or ""), None)
            if future and not future.done():
                future.set_result(frame)
            return
        if msg == "updated":
            return
        if msg != "changed":
            return

        collection = frame.get("collection")
        fields = frame.get("fields") or {}
        if collection == "stream-room-messages":
            room_id = str(fields.get("eventName") or "").strip()
            args = fields.get("args") or []
            message = self._extract_stream_message(args)
            if room_id and isinstance(message, dict):
                await self._dispatch_incoming_message(room_id, message)
            return

        if collection != "stream-notify-user":
            return

        event_name = str(fields.get("eventName") or "")
        if "/" not in event_name:
            return
        _, suffix = event_name.split("/", 1)
        args = fields.get("args") or []
        if suffix == "subscriptions-changed":
            await self._handle_subscriptions_changed(args)
        elif suffix == "rooms-changed":
            await self._handle_rooms_changed(args)

    def _extract_stream_message(self, args: Any) -> Optional[dict[str, Any]]:
        if isinstance(args, list):
            if args and isinstance(args[0], dict):
                return args[0]
            if len(args) > 1 and isinstance(args[1], dict):
                return args[1]
        if isinstance(args, dict):
            return args
        return None

    async def _handle_subscriptions_changed(self, args: list[Any]) -> None:
        if len(args) < 2 or not isinstance(args[1], dict):
            await self._refresh_room_cache()
            await self._subscribe_all_rooms()
            return

        action = str(args[0] or "").lower()
        data = args[1]
        room_id = str(data.get("rid") or data.get("_id") or "").strip()
        if action in {"inserted", "updated"} and room_id:
            room = self._room_cache.get(room_id, {}).copy()
            room.update(data)
            room["_id"] = room_id
            self._room_cache[room_id] = room
            await self._subscribe_room(room_id)
            return

        await self._refresh_room_cache()
        await self._subscribe_all_rooms()

    async def _handle_rooms_changed(self, args: list[Any]) -> None:
        if len(args) < 2 or not isinstance(args[1], dict):
            await self._refresh_room_cache()
            await self._subscribe_all_rooms()
            return

        action = str(args[0] or "").lower()
        room = args[1]
        room_id = str(room.get("_id") or room.get("rid") or "").strip()
        if action in {"inserted", "updated"} and room_id:
            self._room_cache[room_id] = room
            await self._subscribe_room(room_id)
            return

        if action == "removed" and room_id:
            self._room_cache.pop(room_id, None)
            self._room_subscriptions.pop(room_id, None)
            return

        await self._refresh_room_cache()
        await self._subscribe_all_rooms()

    async def _subscribe_all_rooms(self) -> None:
        for room_id in list(self._room_cache):
            await self._subscribe_room(room_id)

    async def _subscribe_room(self, room_id: str) -> None:
        if not room_id or self._ws is None or room_id in self._room_subscriptions:
            return
        sub_id = self._next_id(f"room-{room_id}")
        self._room_subscriptions[room_id] = sub_id
        await self._ws.send_json(
            {
                "msg": "sub",
                "id": sub_id,
                "name": "stream-room-messages",
                "params": [room_id, False],
            }
        )

    async def _dispatch_incoming_message(self, room_id: str, message: dict[str, Any]) -> None:
        sender_id = _message_sender_id(message)
        if sender_id == self._bot_user_id:
            return
        if _message_is_system(message):
            return

        message_id = str(message.get("_id") or "").strip()
        if not message_id or self._dedup.is_duplicate(message_id):
            return

        room = self._room_cache.get(room_id, {"_id": room_id, "t": "c"})
        chat_type = _room_chat_type(room)

        if chat_type != "dm":
            if self._allowed_rooms and room_id not in self._allowed_rooms:
                return
        if not self._is_allowed_user(sender_id):
            return

        raw_text = str(message.get("msg") or "")
        normalized_text, is_command = self._normalize_command_text(raw_text)
        if chat_type != "dm":
            has_mention = self._message_mentions_bot(message, normalized_text)
            if self._require_mention and room_id not in self._free_response_rooms and not has_mention:
                return
            if has_mention:
                normalized_text = self._strip_bot_mentions(normalized_text).strip()
                normalized_text, is_command = self._normalize_command_text(normalized_text)

        media_urls, media_types, event_type = await self._download_incoming_attachments(message)

        source = self.build_source(
            chat_id=room_id,
            chat_name=_room_name(room),
            chat_type=chat_type,
            user_id=sender_id,
            user_name=_message_sender_name(message),
            thread_id=str(message.get("tmid") or "").strip() or None,
            chat_topic=str(room.get("topic") or "").strip() or None,
            parent_chat_id=room_id if message.get("tmid") else None,
            message_id=message_id,
        )

        event = MessageEvent(
            text=normalized_text,
            message_type=MessageType.COMMAND if is_command else event_type,
            source=source,
            raw_message=message,
            message_id=message_id,
            media_urls=media_urls,
            media_types=media_types,
            channel_prompt=resolve_channel_prompt(self.config.extra, room_id, None),
        )
        await self.handle_message(event)

    def _normalize_command_text(self, text: str) -> tuple[str, bool]:
        cleaned = str(text or "")
        if cleaned.startswith("/"):
            return cleaned, True
        prefix = self._command_prefix or "!"
        if cleaned.startswith(prefix) and len(cleaned) > len(prefix):
            candidate = cleaned[len(prefix):]
            if re.match(r"^[A-Za-z][A-Za-z0-9_-]*(?:\s|$)", candidate):
                return "/" + candidate, True
        return cleaned, False

    def _message_mentions_bot(self, message: dict[str, Any], text: str) -> bool:
        for mention in message.get("mentions") or []:
            if not isinstance(mention, dict):
                continue
            if str(mention.get("_id") or "").strip() == self._bot_user_id:
                return True
            if self._bot_username and str(mention.get("username") or "").strip() == self._bot_username:
                return True
        if self._bot_username and f"@{self._bot_username}".lower() in text.lower():
            return True
        return False

    def _strip_bot_mentions(self, text: str) -> str:
        if not self._bot_username:
            return text
        pattern = re.compile(rf"@{re.escape(self._bot_username)}\b", re.IGNORECASE)
        return pattern.sub("", text)

    async def _download_incoming_attachments(
        self,
        message: dict[str, Any],
    ) -> tuple[list[str], list[str], MessageType]:
        attachments = self._extract_message_files(message)
        if not attachments:
            return [], [], MessageType.TEXT

        media_urls: list[str] = []
        media_types: list[str] = []
        event_type = MessageType.TEXT

        for meta in attachments:
            try:
                cached = await self._download_one_attachment(meta)
            except Exception as exc:
                logger.warning("Rocket.Chat: attachment download failed: %s", exc)
                continue
            if not cached:
                logger.warning("Rocket.Chat: attachment download failed or returned unsupported media; continuing with text only")
                continue
            media_urls.append(cached.path)
            media_types.append(cached.media_type)
            if cached.kind == "image":
                event_type = MessageType.PHOTO
            elif cached.kind == "audio" and event_type == MessageType.TEXT:
                event_type = MessageType.VOICE
            elif cached.kind == "document" and event_type == MessageType.TEXT:
                event_type = MessageType.DOCUMENT
            elif cached.kind == "video" and event_type == MessageType.TEXT:
                event_type = MessageType.VIDEO

        return media_urls, media_types, event_type

    def _extract_message_files(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        files = []
        primary = message.get("file")
        if isinstance(primary, dict):
            files.append(primary)
        for file_meta in message.get("files") or []:
            if isinstance(file_meta, dict):
                files.append(file_meta)

        links: dict[str, dict[str, Any]] = {}
        for attachment in message.get("attachments") or []:
            if not isinstance(attachment, dict):
                continue
            link = str(attachment.get("title_link") or "").strip()
            if link:
                links[link] = attachment

        enriched = []
        for file_meta in files:
            file_copy = dict(file_meta)
            title_link = self._best_attachment_link(file_meta, links)
            if title_link:
                file_copy["url"] = title_link
            enriched.append(file_copy)
        return enriched

    def _best_attachment_link(self, file_meta: dict[str, Any], links: dict[str, dict[str, Any]]) -> str:
        existing = str(file_meta.get("url") or "").strip()
        if existing:
            return existing
        file_id = str(file_meta.get("_id") or "").strip()
        name = str(file_meta.get("name") or "").strip()
        if file_id and name:
            candidate = f"/file-upload/{file_id}/{name}"
            if candidate in links:
                return candidate
            return candidate
        if links:
            return next(iter(links))
        return ""

    async def _download_one_attachment(self, file_meta: dict[str, Any]):
        link = str(file_meta.get("url") or "").strip()
        if not link:
            return None
        full_url = urljoin(f"{self._base_url}/", link.lstrip("/"))
        headers = {
            "X-Auth-Token": self._runtime_creds.auth_token,
            "X-User-Id": self._runtime_creds.user_id,
        }
        import aiohttp

        async with self._session.get(
            full_url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
            **self._request_kwargs,
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} while downloading {link}")
            data = await resp.read()

        filename = str(file_meta.get("name") or "").strip() or "attachment"
        mime_type = str(file_meta.get("type") or "").strip() or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        default_kind = "image" if mime_type.startswith("image/") else None
        return cache_media_bytes(
            data,
            filename=filename,
            mime_type=mime_type,
            default_kind=default_kind,
        )

    def _is_allowed_user(self, user_id: str) -> bool:
        if self._allow_all_users:
            return True
        if not self._allowed_users:
            return False
        return user_id in self._allowed_users or "*" in self._allowed_users

    def _next_id(self, prefix: str = "rc") -> str:
        self._id_counter += 1
        return f"{prefix}-{self._id_counter}"

    def _create_waiter(self, waiter_id: str) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._method_waiters[waiter_id] = future
        return future


def check_rocketchat_requirements() -> bool:
    """Return True when Rocket.Chat is configured well enough to be usable."""
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        return False

    if not os.getenv("ROCKETCHAT_URL", "").strip():
        return False

    if bootstrap_enabled():
        return artifact_path().expanduser().exists()

    return bool(
        os.getenv("ROCKETCHAT_USER_ID", "").strip()
        and os.getenv("ROCKETCHAT_AUTH_TOKEN", "").strip()
    )


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    url = str(extra.get("url") or os.getenv("ROCKETCHAT_URL", "")).strip()
    if not url:
        return False
    if bootstrap_enabled(extra):
        return artifact_path(extra).expanduser().exists()
    user_id = str(extra.get("user_id") or os.getenv("ROCKETCHAT_USER_ID", "")).strip()
    auth_token = str(extra.get("auth_token") or os.getenv("ROCKETCHAT_AUTH_TOKEN", "")).strip()
    return bool(user_id and auth_token)


def _env_enablement() -> Optional[Dict[str, Any]]:
    """Seed PlatformConfig.extra from env-only Rocket.Chat setups."""
    url = os.getenv("ROCKETCHAT_URL", "").strip().rstrip("/")
    if not url:
        return None

    enabled_via_artifact = bootstrap_enabled() and artifact_path().expanduser().exists()
    enabled_via_env = bool(
        os.getenv("ROCKETCHAT_USER_ID", "").strip()
        and os.getenv("ROCKETCHAT_AUTH_TOKEN", "").strip()
    )
    if not enabled_via_artifact and not enabled_via_env:
        return None

    seed: dict[str, Any] = {
        "url": url,
    }
    if os.getenv("ROCKETCHAT_COMMAND_PREFIX", "").strip():
        seed["command_prefix"] = os.environ["ROCKETCHAT_COMMAND_PREFIX"].strip()
    if bootstrap_enabled():
        seed["bootstrap_enabled"] = True
        seed["bootstrap_artifact"] = str(artifact_path())
    home = os.getenv("ROCKETCHAT_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("ROCKETCHAT_HOME_CHANNEL_NAME", "Home"),
            "thread_id": os.getenv("ROCKETCHAT_HOME_CHANNEL_THREAD_ID", "").strip() or None,
        }
    return seed


def _apply_yaml_config(yaml_cfg: dict, rocketchat_cfg: dict) -> dict | None:
    """Translate config.yaml `rocketchat:` keys into runtime env vars."""
    del yaml_cfg

    mapping = {
        "require_mention": "ROCKETCHAT_REQUIRE_MENTION",
        "command_prefix": "ROCKETCHAT_COMMAND_PREFIX",
        "bootstrap_enabled": "ROCKETCHAT_BOOTSTRAP_ENABLED",
        "bootstrap_artifact": "ROCKETCHAT_BOOTSTRAP_ARTIFACT",
        "bootstrap_pat_name": "ROCKETCHAT_BOOTSTRAP_PAT_NAME",
    }
    for key, env_var in mapping.items():
        if key in rocketchat_cfg and not os.getenv(env_var):
            os.environ[env_var] = str(rocketchat_cfg[key]).lower() if isinstance(rocketchat_cfg[key], bool) else str(rocketchat_cfg[key])

    for key, env_var in (
        ("free_response_rooms", "ROCKETCHAT_FREE_RESPONSE_ROOMS"),
        ("allowed_rooms", "ROCKETCHAT_ALLOWED_ROOMS"),
    ):
        value = rocketchat_cfg.get(key)
        if value is not None and not os.getenv(env_var):
            if isinstance(value, list):
                value = ",".join(str(item) for item in value)
            os.environ[env_var] = str(value)

    return None


def _is_connected(config) -> bool:
    return validate_config(config)


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process cron sender for Rocket.Chat."""
    del force_document
    config = PlatformConfig(enabled=True, token=getattr(pconfig, "token", None), extra=dict(getattr(pconfig, "extra", {}) or {}))
    adapter = RocketChatAdapter(config)
    if not await adapter.connect():
        return {"error": "Rocket.Chat standalone send failed to connect"}
    try:
        result: SendResult
        if media_files:
            last: SendResult = SendResult(success=True)
            for media in media_files:
                file_path = media.get("path") if isinstance(media, dict) else str(media)
                is_voice = bool(media.get("is_voice")) if isinstance(media, dict) else False
                metadata = {"thread_id": thread_id} if thread_id else None
                ext = Path(file_path).suffix.lower()
                if not is_voice and ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
                    last = await adapter.send_image_file(chat_id, file_path, caption=message or None, metadata=metadata)
                else:
                    last = await adapter.send_document(chat_id, file_path, caption=message or None, metadata=metadata)
                if not last.success:
                    return {"error": last.error or "Rocket.Chat standalone media send failed"}
                message = ""
            return {"success": True, "message_id": last.message_id}

        result = await adapter.send(
            chat_id=chat_id,
            content=message,
            metadata={"thread_id": thread_id} if thread_id else None,
        )
        if result.success:
            return {"success": True, "message_id": result.message_id}
        return {"error": result.error or "Rocket.Chat standalone send failed"}
    finally:
        await adapter.disconnect()


def interactive_setup() -> None:
    """Interactive gateway setup for Rocket.Chat."""
    from hermes_cli.gateway import get_env_value, save_env_value
    from hermes_cli.setup_utils import print_header, print_info, print_success, prompt, prompt_yes_no

    print_header("Rocket.Chat")
    if get_env_value("ROCKETCHAT_URL") and (
        get_env_value("ROCKETCHAT_AUTH_TOKEN") or get_env_value("ROCKETCHAT_BOOTSTRAP_ENABLED")
    ):
        print_info("Rocket.Chat: already configured")
        if not prompt_yes_no("Reconfigure Rocket.Chat?", False):
            return

    print_info("Works with self-hosted or cloud Rocket.Chat workspaces.")
    url = prompt("Rocket.Chat server URL (e.g. https://chat.example.com)")
    if url:
        save_env_value("ROCKETCHAT_URL", url.rstrip("/"))

    use_bootstrap = prompt_yes_no("Use browser-assisted OAuth bootstrap instead of pasting a token?", False)
    if use_bootstrap:
        save_env_value("ROCKETCHAT_BOOTSTRAP_ENABLED", "true")
        artifact = prompt("Bootstrap artifact path (leave empty for ~/.hermes/rocketchat_auth.json)")
        if artifact:
            save_env_value("ROCKETCHAT_BOOTSTRAP_ARTIFACT", artifact)
        service_name = prompt("Rocket.Chat OAuth service name")
        authorize_url = prompt("OAuth authorize URL")
        token_url = prompt("OAuth token URL")
        client_id = prompt("OAuth client ID")
        client_secret = prompt("OAuth client secret", password=True)
        if service_name:
            save_env_value("ROCKETCHAT_OAUTH_SERVICE_NAME", service_name)
        if authorize_url:
            save_env_value("ROCKETCHAT_OAUTH_AUTHORIZE_URL", authorize_url)
        if token_url:
            save_env_value("ROCKETCHAT_OAUTH_TOKEN_URL", token_url)
        if client_id:
            save_env_value("ROCKETCHAT_OAUTH_CLIENT_ID", client_id)
        if client_secret:
            save_env_value("ROCKETCHAT_OAUTH_CLIENT_SECRET", client_secret)
        print_success("Rocket.Chat bootstrap configuration saved")
        if prompt_yes_no("Run bootstrap now?", True):
            try:
                runtime = asyncio.run(bootstrap_via_oauth(bootstrap_config_from_env(url)))
                print_success(
                    f"Rocket.Chat bootstrap complete for @{runtime.username or runtime.user_id}"
                )
            except Exception as exc:
                print_info(f"Bootstrap failed: {exc}")
        return

    user_id = prompt("Rocket.Chat user ID")
    auth_token = prompt("Rocket.Chat auth token or personal access token", password=True)
    if user_id:
        save_env_value("ROCKETCHAT_USER_ID", user_id)
    if auth_token:
        save_env_value("ROCKETCHAT_AUTH_TOKEN", auth_token)
    allowed_users = prompt("Allowed user IDs (comma-separated)")
    if allowed_users:
        save_env_value("ROCKETCHAT_ALLOWED_USERS", allowed_users.replace(" ", ""))
    home_room = prompt("Home room ID (optional)")
    if home_room:
        save_env_value("ROCKETCHAT_HOME_CHANNEL", home_room)
    print_success("Rocket.Chat configuration saved")


def register(ctx) -> None:
    """Plugin entry point for Rocket.Chat."""
    ctx.register_platform(
        name="rocketchat",
        label="Rocket.Chat",
        adapter_factory=lambda cfg: RocketChatAdapter(cfg),
        check_fn=check_rocketchat_requirements,
        validate_config=validate_config,
        is_connected=_is_connected,
        required_env=["ROCKETCHAT_URL"],
        install_hint="pip install aiohttp",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="ROCKETCHAT_ALLOWED_USERS",
        allow_all_env="ROCKETCHAT_ALLOW_ALL_USERS",
        cron_deliver_env_var="ROCKETCHAT_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🚀",
        allow_update_command=True,
        platform_hint=(
            "You are chatting through Rocket.Chat. DMs always work when the user is authorized. "
            "In shared rooms the safe typed command alias is `!`, which Hermes rewrites to slash commands "
            "before processing, so prefer `!new`, `!reset`, `!commands`, etc. over raw `/...` unless the "
            "workspace is known to deliver slash-prefixed text untouched. Rocket.Chat renders markdown reasonably "
            "well and supports message edits, so streaming replies may appear as in-place edits when the workspace "
            "permits editing. Existing Rocket.Chat thread IDs are preserved; Hermes does not auto-create new threads."
        ),
    )
