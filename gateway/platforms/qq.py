"""
QQ platform adapter.

Uses qq-botpy for:
- Receiving C2C and group @ messages over the QQ bot gateway
- Sending text responses back via the QQ bot API
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import botpy
    from botpy.message import C2CMessage, GroupMessage
    QQ_AVAILABLE = True
except ImportError:
    botpy = None
    C2CMessage = Any
    GroupMessage = Any
    QQ_AVAILABLE = False

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)


def check_qq_requirements() -> bool:
    """Check if QQ dependencies are available."""
    return QQ_AVAILABLE


class _HermesQQClient(botpy.Client if QQ_AVAILABLE else object):
    """Thin botpy client wrapper that forwards events to the adapter."""

    def __init__(self, adapter: "QQAdapter"):
        intents = botpy.Intents(public_messages=True)
        super().__init__(
            intents=intents,
            timeout=adapter.CONNECT_TIMEOUT_SECONDS,
            is_sandbox=adapter.is_sandbox,
        )
        self._adapter = adapter

    async def on_ready(self):
        robot = getattr(self, "robot", None)
        logger.info("[QQ] Connected as %s", getattr(robot, "name", "unknown"))
        self._adapter._ready_event.set()

    async def on_group_at_message_create(self, message: GroupMessage):
        await self._adapter._handle_group_message(message)

    async def on_c2c_message_create(self, message: C2CMessage):
        await self._adapter._handle_c2c_message(message)

    async def on_error(self, event_method: str, *args: Any, **kwargs: Any) -> None:
        logger.exception("[QQ] botpy client error during %s", event_method)


class QQAdapter(BasePlatformAdapter):
    """QQ bot adapter using the official qq-botpy websocket client."""

    MAX_MESSAGE_LENGTH = 2000
    CONNECT_TIMEOUT_SECONDS = 30

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.QQ)
        self.app_id = str(config.token or "").strip()
        self.secret = str(config.api_key or "").strip()
        self.is_sandbox = bool(config.extra.get("sandbox")) or (
            os.getenv("QQ_BOT_SANDBOX", "").strip().lower() in ("1", "true", "yes")
        )
        self._client: Optional[_HermesQQClient] = None
        self._client_task: Optional[asyncio.Task] = None
        self._ready_event = asyncio.Event()

    async def connect(self) -> bool:
        """Connect to the QQ bot gateway and start receiving messages."""
        if not QQ_AVAILABLE:
            logger.error("[%s] qq-botpy not installed. Run: pip install qq-botpy", self.name)
            return False

        if not self.app_id or not self.secret:
            logger.error("[%s] QQ app_id or secret missing", self.name)
            return False

        try:
            self._ready_event.clear()
            self._client = _HermesQQClient(self)
            self._client_task = asyncio.create_task(
                self._client.start(appid=self.app_id, secret=self.secret),
                name="qq-botpy-client",
            )
            self._client_task.add_done_callback(self._on_client_done)
            await asyncio.wait_for(self._ready_event.wait(), timeout=self.CONNECT_TIMEOUT_SECONDS)
            self._running = True
            return True
        except asyncio.TimeoutError:
            logger.error("[%s] Timeout waiting for QQ gateway ready event", self.name)
        except Exception as e:
            logger.error("[%s] Failed to connect to QQ: %s", self.name, e, exc_info=True)

        await self.disconnect()
        return False

    def _on_client_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as e:
            logger.error("[%s] QQ client task exited with error: %s", self.name, e, exc_info=True)
        finally:
            self._running = False

    async def disconnect(self) -> None:
        """Disconnect from the QQ gateway."""
        self._running = False

        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.warning("[%s] Error closing QQ client: %s", self.name, e, exc_info=True)

        if self._client_task:
            self._client_task.cancel()
            try:
                await self._client_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        self._client = None
        self._client_task = None
        self._ready_event.clear()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message to a QQ group or C2C chat."""
        if not self._client:
            return SendResult(success=False, error="Not connected")

        try:
            chunks = self.truncate_message(self.format_message(content), self.MAX_MESSAGE_LENGTH)
            message_ids = []
            for index, chunk in enumerate(chunks):
                msg_id = reply_to if index == 0 else None
                response = await self._send_text_chunk(chat_id, chunk, msg_id=msg_id)
                response_id = self._extract_message_id(response)
                if response_id:
                    message_ids.append(response_id)

            return SendResult(
                success=True,
                message_id=message_ids[0] if message_ids else None,
                raw_response={"message_ids": message_ids},
            )
        except Exception as e:
            logger.error("[%s] Failed to send QQ message: %s", self.name, e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def _send_text_chunk(self, chat_id: str, content: str, msg_id: Optional[str] = None):
        target_type, target_id = self._parse_target(chat_id)
        kwargs = {
            "msg_type": 0,
            "content": content,
            "msg_id": msg_id,
        }
        if target_type == "group":
            return await self._client.api.post_group_message(group_openid=target_id, **kwargs)
        return await self._client.api.post_c2c_message(openid=target_id, **kwargs)

    @staticmethod
    def _extract_message_id(response: Any) -> Optional[str]:
        if isinstance(response, dict):
            value = response.get("id")
        else:
            value = getattr(response, "id", None)
        return str(value) if value else None

    @staticmethod
    def _parse_target(chat_id: str) -> tuple[str, str]:
        if ":" in chat_id:
            target_type, target_id = chat_id.split(":", 1)
            if target_type in ("group", "user") and target_id:
                return target_type, target_id
        return "user", chat_id

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """QQ bot API does not expose a typing indicator."""
        return None

    async def _handle_group_message(self, message: GroupMessage) -> None:
        await self.handle_message(self._event_from_group_message(message))

    async def _handle_c2c_message(self, message: C2CMessage) -> None:
        await self.handle_message(self._event_from_c2c_message(message))

    def _event_from_group_message(self, message: GroupMessage) -> MessageEvent:
        author = getattr(message, "author", None)
        user_id = (
            getattr(author, "member_openid", None)
            or getattr(author, "id", None)
            or getattr(message, "member_openid", None)
        )
        user_name = (
            getattr(author, "username", None)
            or getattr(author, "nick", None)
            or getattr(message, "member_name", None)
        )
        group_openid = getattr(message, "group_openid", None)
        text = (getattr(message, "content", None) or "").strip()
        return MessageEvent(
            text=text,
            message_type=MessageType.COMMAND if text.startswith("/") else MessageType.TEXT,
            source=self.build_source(
                chat_id=f"group:{group_openid}",
                chat_name=getattr(message, "group_name", None) or group_openid,
                chat_type="group",
                user_id=user_id,
                user_name=user_name,
            ),
            raw_message=message,
            message_id=str(getattr(message, "id", "") or ""),
        )

    def _event_from_c2c_message(self, message: C2CMessage) -> MessageEvent:
        author = getattr(message, "author", None)
        user_id = (
            getattr(author, "user_openid", None)
            or getattr(author, "id", None)
            or getattr(message, "user_openid", None)
        )
        user_name = (
            getattr(author, "username", None)
            or getattr(author, "nick", None)
            or getattr(message, "author_name", None)
        )
        text = (getattr(message, "content", None) or "").strip()
        return MessageEvent(
            text=text,
            message_type=MessageType.COMMAND if text.startswith("/") else MessageType.TEXT,
            source=self.build_source(
                chat_id=f"user:{user_id}",
                chat_name=user_name or user_id,
                chat_type="dm",
                user_id=user_id,
                user_name=user_name,
            ),
            raw_message=message,
            message_id=str(getattr(message, "id", "") or ""),
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        target_type, target_id = self._parse_target(chat_id)
        return {
            "name": target_id,
            "type": "group" if target_type == "group" else "dm",
            "chat_id": chat_id,
        }
