"""Official WhatsApp Agent Platform transport (personal agent chats).

Protocol: https://www.whatsapp.com/developer/WhatsApp-Agent-Platform-Developer-Manual.pdf
This is distinct from WhatsApp Web and the Business Cloud API.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
BASE_URL = "https://api.whatsapp.com/agent/v1"
MAX_LENGTH = 4096


def _token(config=None) -> str:
    return os.getenv("WHATSAPP_AGENT_TOKEN", "").strip() or str(
        (getattr(config, "extra", {}) or {}).get("token", "")
    ).strip()


def _env_enablement() -> Optional[dict]:
    return {} if _token() else None


def _state_path(token: str) -> Path:
    # The state is scoped by token, so replacing a key never reuses another
    # agent's cursor. Store no token or message content on disk.
    key_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
    return get_hermes_home() / "gateway" / f"whatsapp_agent_{key_hash}.json"


class WhatsAppAgentAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = MAX_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config=config, platform=Platform("whatsapp_agent"))
        self._token = _token(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._offset: Optional[int] = None
        self._last_recipient: Optional[str] = None
        self._lock_key: Optional[str] = None
        self._state_file = _state_path(self._token) if self._token else None
        self._last_send_at = 0.0
        self._last_poll_at = 0.0
        self._fresh_after = int(time.time())

    @property
    def authorization_is_upstream(self) -> bool:
        """The authenticated Agent Platform delivers only the creator's messages."""
        return True

    def _save_state(self) -> None:
        if not self._state_file:
            return
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        temp = self._state_file.with_suffix(".tmp")
        temp.write_text(json.dumps({"offset": self._offset, "recipient": self._last_recipient, "fresh_after": self._fresh_after}))
        os.chmod(temp, 0o600)
        temp.replace(self._state_file)

    def _load_state(self) -> None:
        if not self._state_file or not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text())
            self._fresh_after = int(data.get("fresh_after", self._fresh_after))
            offset = data.get("offset")
            self._offset = int(offset) if offset is not None else None
            recipient = data.get("recipient")
            if isinstance(recipient, str) and recipient.startswith("user:"):
                self._last_recipient = recipient
        except (OSError, ValueError, TypeError):
            logger.warning("WhatsApp Agent: invalid cursor state; starting at current head")

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self._token:
            self._set_fatal_error("config_missing", "WHATSAPP_AGENT_TOKEN is required", retryable=False)
            return False
        from gateway.status import acquire_scoped_lock
        lock_key = hashlib.sha256(self._token.encode()).hexdigest()
        if not acquire_scoped_lock("whatsapp_agent", lock_key):
            self._set_fatal_error("lock_conflict", "Agent token is already in use", retryable=False)
            return False
        self._lock_key = lock_key
        self._load_state()
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=httpx.Timeout(connect=10, read=35, write=10, pool=10),
        )
        self._mark_connected()
        self._poll_task = asyncio.create_task(self._poll())
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._lock_key:
            from gateway.status import release_scoped_lock
            release_scoped_lock("whatsapp_agent", self._lock_key)
            self._lock_key = None

    async def _poll(self) -> None:
        backoff = 2
        while self._running:
            try:
                loop = asyncio.get_running_loop()
                delay = 4.0 - (loop.time() - self._last_poll_at)
                if delay > 0:
                    await asyncio.sleep(delay)
                self._last_poll_at = loop.time()
                params = {"limit": 50, "timeout": 15, "offset": self._offset if self._offset is not None else 0}
                response = await self._client.get("/updates", params=params)
                if response.status_code in (401, 403):
                    self._set_fatal_error("auth_failed", "WhatsApp Agent token rejected", retryable=False)
                    self._mark_disconnected()
                    return
                if response.status_code == 204:
                    backoff = 2
                    continue
                if response.status_code == 429:
                    await asyncio.sleep(10)
                    continue
                response.raise_for_status()
                payload = response.json()
                for entry in payload.get("entry", []):
                    for change in entry.get("changes", []):
                        value = change.get("value", {})
                        for message in value.get("messages", []):
                            await self._receive(message)
                next_offset = payload.get("next_offset")
                if next_offset is not None:
                    self._offset = int(next_offset)
                    self._save_state()
                backoff = 2
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("WhatsApp Agent poll failed: %s", type(exc).__name__)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _receive(self, message: dict) -> None:
        sender = message.get("from", "")
        if not isinstance(sender, str) or not sender.startswith("user:"):
            return
        # On first setup, read from offset 0 for a stable cursor but skip
        # preexisting messages instead of automatically replying to old chats.
        try:
            message_time = int(message.get("timestamp", "0"))
        except (TypeError, ValueError):
            message_time = 0
        if message_time < self._fresh_after:
            return
        self._last_recipient = sender
        body = (message.get("text") or {}).get("body", "") if message.get("type") == "text" else ""
        if not body:
            return
        try:
            timestamp = datetime.fromtimestamp(int(message["timestamp"]), tz=timezone.utc)
        except (KeyError, TypeError, ValueError, OSError):
            timestamp = datetime.now(tz=timezone.utc)
        source = self.build_source(
            chat_id=sender, chat_name="WhatsApp Agent", chat_type="dm",
            user_id=sender, user_name="Owner",
        )
        await self.handle_message(MessageEvent(
            text=body, message_type=MessageType.TEXT, source=source,
            message_id=message.get("id"), raw_message=message, timestamp=timestamp,
        ))

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        recipient = chat_id or self._last_recipient
        if not recipient or not recipient.startswith("user:"):
            return SendResult(success=False, error="No valid creator user ID yet")
        if not self._client:
            return SendResult(success=False, error="Not connected")
        # The API accepts at most 4096 characters. Long responses are split.
        last_id = None
        for start in range(0, len(content), MAX_LENGTH):
            loop = asyncio.get_running_loop()
            delay = 5.0 - (loop.time() - self._last_send_at)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_send_at = loop.time()
            body = content[start:start + MAX_LENGTH]
            try:
                response = await self._client.post("/messages", json={
                    "messaging_product": "whatsapp", "to": recipient,
                    "type": "text", "text": {"body": body},
                })
                if response.status_code >= 300:
                    return SendResult(success=False, error=f"WhatsApp Agent HTTP {response.status_code}")
                messages = response.json().get("messages", [])
                last_id = messages[0].get("id") if messages else None
            except httpx.HTTPError as exc:
                return SendResult(success=False, error=f"WhatsApp Agent send failed: {type(exc).__name__}")
        return SendResult(success=True, message_id=last_id)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": "WhatsApp Agent", "type": "dm"}


def register(ctx) -> None:
    ctx.register_platform(
        name="whatsapp_agent", label="WhatsApp Agent",
        adapter_factory=lambda cfg: WhatsAppAgentAdapter(cfg),
        check_fn=lambda: bool(_token()), validate_config=lambda cfg: bool(_token(cfg)),
        is_connected=lambda cfg: bool(_token(cfg)),
        required_env=["WHATSAPP_AGENT_TOKEN"], env_enablement_fn=_env_enablement,
        allowed_users_env="WHATSAPP_AGENT_ALLOWED_USERS",
        allow_all_env="WHATSAPP_AGENT_ALLOW_ALL_USERS",
        max_message_length=MAX_LENGTH, emoji="💬", pii_safe=True,
        platform_hint="You are replying in a private WhatsApp Agent chat. Use concise plain text.",
    )
