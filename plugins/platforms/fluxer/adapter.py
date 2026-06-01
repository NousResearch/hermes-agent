"""Fluxer platform plugin for Hermes Agent.

Text-first adapter:
- REST `POST /channels/:id/messages` for outbound messages.
- Fluxer Gateway websocket `MESSAGE_CREATE` events for inbound messages.

Fluxer self-hosting is still moving, so this adapter intentionally keeps the
surface conservative and easy to test. Media/rich embeds can layer on once the
API settles.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000
_DEFAULT_BASE_URL = "https://fluxer.app"
_GATEWAY_VERSION = 1


def _strip_slash(url: str) -> str:
    return (url or "").strip().rstrip("/")


def _api_base(base_url: str) -> str:
    """Normalize a user-provided Fluxer URL to the REST API base.

    Fluxer exposes routes under `/api` in the monolith, while some operators may
    choose to configure an already-scoped `/api` or `/api/v1` URL. Preserve those
    and append `/api` only for a plain origin.
    """
    base = _strip_slash(base_url)
    if not base:
        return ""
    if base.endswith("/api") or base.endswith("/api/v1") or "/api/" in base:
        return base
    return f"{base}/api"


def _build_identify_payload(bot_token: str) -> Dict[str, Any]:
    return {
        "op": 2,
        "d": {
            "token": bot_token,
            "properties": {
                "os": "linux",
                "browser": "hermes",
                "device": "hermes",
            },
        },
    }


def _headers(bot_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bot {bot_token}",
        "Content-Type": "application/json",
        "User-Agent": "Hermes-Fluxer/0.1",
    }


def _event_seq(payload: Dict[str, Any]) -> Optional[int]:
    seq = payload.get("s")
    try:
        return int(seq) if seq is not None else None
    except (TypeError, ValueError):
        return None


def _author_name(author: Dict[str, Any]) -> Optional[str]:
    return (
        author.get("global_name")
        or author.get("display_name")
        or author.get("username")
        or author.get("name")
    )


def _chat_type(raw: Any) -> str:
    if isinstance(raw, str):
        lowered = raw.lower()
        if lowered in {"dm", "direct", "private"}:
            return "dm"
        if lowered in {"group_dm", "group", "group-dm"}:
            return "group"
        if lowered in {"thread"}:
            return "thread"
        return "channel"
    # Discord-like channel types in several codebases: 1 = DM, 3 = group DM.
    if raw == 1:
        return "dm"
    if raw == 3:
        return "group"
    return "channel"


class FluxerAdapter(BasePlatformAdapter):
    """Fluxer adapter using bot REST + Gateway websocket APIs."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("fluxer"))
        extra = getattr(config, "extra", {}) or {}
        self.base_url = _strip_slash(
            os.getenv("FLUXER_BASE_URL") or extra.get("base_url") or _DEFAULT_BASE_URL
        )
        self.api_base_url = _api_base(self.base_url)
        self.bot_token = (
            os.getenv("FLUXER_BOT_TOKEN") or extra.get("bot_token") or ""
        ).strip()
        self.gateway_url = _strip_slash(
            os.getenv("FLUXER_GATEWAY_URL") or extra.get("gateway_url") or ""
        )
        self.bot_user_id: Optional[str] = str(extra.get("bot_user_id")) if extra.get("bot_user_id") else None
        self._ws = None
        self._listener_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_seq: Optional[int] = None
        self._seen_message_ids: set[str] = set()

    async def connect(self) -> bool:
        if not self.base_url or not self.bot_token:
            self._set_fatal_error(
                "missing_config",
                "Fluxer requires FLUXER_BASE_URL and FLUXER_BOT_TOKEN",
                retryable=False,
            )
            return False
        try:
            if not self.gateway_url:
                info = await self._request("GET", "/gateway/bot")
                self.gateway_url = _strip_slash(str(info.get("url") or ""))
            if not self.gateway_url:
                raise RuntimeError("Fluxer gateway URL missing from /gateway/bot")

            import websockets

            sep = "&" if "?" in self.gateway_url else "?"
            ws_url = f"{self.gateway_url}{sep}v={_GATEWAY_VERSION}&encoding=json"
            self._ws = await websockets.connect(ws_url, open_timeout=15, close_timeout=5, max_size=None)
            self._listener_task = asyncio.create_task(self._listen_loop(), name="fluxer-listen")
            self._mark_connected()
            return True
        except Exception as exc:
            logger.warning("Fluxer connect failed: %s", exc)
            self._set_fatal_error("connect_failed", f"Fluxer connect failed: {exc}", retryable=True)
            return False

    async def disconnect(self) -> None:
        self._running = False
        for task in (self._heartbeat_task, self._listener_task):
            if task and not task.done():
                task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._heartbeat_task = None
        self._listener_task = None
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        payload: Dict[str, Any] = {"content": content}
        if reply_to:
            payload["message_reference"] = {"message_id": str(reply_to)}
        if metadata:
            thread_id = metadata.get("thread_id")
            if thread_id and "message_reference" not in payload:
                # Fluxer thread semantics are still stabilizing; keep this as
                # metadata only when callers explicitly provide it.
                payload["message_reference"] = {"message_id": str(thread_id)}

        try:
            data = await self._request(
                "POST",
                f"/channels/{chat_id}/messages",
                json=payload,
            )
            return SendResult(success=True, message_id=str(data.get("id")) if data.get("id") else None, raw_response=data)
        except Exception as exc:
            logger.warning("Fluxer send failed: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        try:
            data = await self._request("GET", f"/channels/{chat_id}")
            return {
                "id": str(data.get("id") or chat_id),
                "name": data.get("name") or str(chat_id),
                "type": _chat_type(data.get("type")),
                "raw": data,
            }
        except Exception:
            return {"id": str(chat_id), "name": str(chat_id), "type": "channel"}

    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for Fluxer adapter") from exc

        url = urljoin(self.api_base_url + "/", path.lstrip("/"))
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.request(method, url, headers=_headers(self.bot_token), **kwargs)
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()

    async def _listen_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                payload = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
                await self._handle_gateway_dispatch(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._running:
                logger.warning("Fluxer listener stopped: %s", exc)
                self._set_fatal_error("listener_stopped", f"Fluxer listener stopped: {exc}", retryable=True)

    async def _heartbeat_loop(self, interval_ms: int) -> None:
        try:
            while self._running and self._ws is not None:
                await asyncio.sleep(max(interval_ms, 1000) / 1000)
                await self._ws.send(json.dumps({"op": 1, "d": self._last_seq}))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Fluxer heartbeat stopped: %s", exc)

    async def _handle_gateway_dispatch(self, payload: Dict[str, Any]) -> None:
        op = payload.get("op")
        self._last_seq = _event_seq(payload) or self._last_seq

        if op == 10:  # HELLO
            interval = int(((payload.get("d") or {}).get("heartbeat_interval") or 41250))
            if self._ws is not None:
                await self._ws.send(json.dumps(_build_identify_payload(self.bot_token)))
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(interval), name="fluxer-heartbeat")
            return
        if op != 0:  # not DISPATCH
            return

        event_name = payload.get("t")
        data = payload.get("d") or {}
        if event_name == "READY":
            user = data.get("user") or data.get("bot") or {}
            if user.get("id"):
                self.bot_user_id = str(user["id"])
            return
        if event_name != "MESSAGE_CREATE":
            return
        await self._handle_message_create(data, payload)

    async def _handle_message_create(self, data: Dict[str, Any], raw_payload: Dict[str, Any]) -> None:
        msg_id = str(data.get("id") or "")
        if msg_id:
            if msg_id in self._seen_message_ids:
                return
            self._seen_message_ids.add(msg_id)
            if len(self._seen_message_ids) > 2000:
                self._seen_message_ids = set(list(self._seen_message_ids)[-1000:])

        author = data.get("author") or data.get("user") or {}
        author_id = str(author.get("id") or data.get("author_id") or "")
        if author.get("bot") or (self.bot_user_id and author_id == str(self.bot_user_id)):
            return

        text = data.get("content") or ""
        if not text and not data.get("attachments"):
            return

        channel_id = str(data.get("channel_id") or data.get("channel", {}).get("id") or "")
        if not channel_id:
            return
        source = self.build_source(
            chat_id=channel_id,
            chat_name=(data.get("channel") or {}).get("name"),
            chat_type=_chat_type(data.get("channel_type") or (data.get("channel") or {}).get("type")),
            user_id=author_id or None,
            user_name=_author_name(author),
            guild_id=data.get("guild_id"),
            message_id=msg_id or None,
        )

        timestamp = datetime.now(tz=timezone.utc)
        ts_raw = data.get("timestamp") or data.get("created_at")
        if isinstance(ts_raw, str):
            try:
                timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                pass

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=raw_payload,
            message_id=msg_id or None,
            timestamp=timestamp,
        )
        await self.handle_message(event)


def check_requirements() -> bool:
    if not (os.getenv("FLUXER_BASE_URL") and os.getenv("FLUXER_BOT_TOKEN")):
        return False
    try:
        import httpx  # noqa: F401
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    base_url = os.getenv("FLUXER_BASE_URL") or extra.get("base_url", "")
    token = os.getenv("FLUXER_BOT_TOKEN") or extra.get("bot_token", "")
    return bool(str(base_url).strip() and str(token).strip())


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> dict | None:
    base_url = os.getenv("FLUXER_BASE_URL", "").strip()
    token = os.getenv("FLUXER_BOT_TOKEN", "").strip()
    if not (base_url and token):
        return None
    seed: dict = {"base_url": base_url, "bot_token": token}
    gateway_url = os.getenv("FLUXER_GATEWAY_URL", "").strip()
    if gateway_url:
        seed["gateway_url"] = gateway_url
    home = os.getenv("FLUXER_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("FLUXER_HOME_CHANNEL_NAME", "").strip() or home,
        }
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    adapter = FluxerAdapter(pconfig)
    metadata = {"thread_id": thread_id} if thread_id else None
    result = await adapter.send(chat_id, message, metadata=metadata)
    if result.success:
        return {"success": True, "platform": "fluxer", "chat_id": chat_id, "message_id": result.message_id}
    return {"error": result.error or "Fluxer send failed"}


def interactive_setup() -> None:
    print("Fluxer platform setup")
    print("Set FLUXER_BASE_URL and FLUXER_BOT_TOKEN in ~/.hermes/.env, then restart the gateway.")


def register(ctx) -> None:
    ctx.register_platform(
        name="fluxer",
        label="Fluxer",
        adapter_factory=lambda cfg: FluxerAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["FLUXER_BASE_URL", "FLUXER_BOT_TOKEN"],
        install_hint="pip install httpx websockets   # Fluxer adapter dependencies",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="FLUXER_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="FLUXER_ALLOWED_USERS",
        allow_all_env="FLUXER_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="⚡",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Fluxer, a Discord-like open-source chat "
            "platform. Fluxer supports rich Markdown, channels, DMs, files, "
            "and voice/video. Prefer normal Markdown for structured replies."
        ),
    )
