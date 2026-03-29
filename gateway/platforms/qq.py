import time
import asyncio
from typing import Optional, Dict, Any

try:
    from gateway.platforms.base import BasePlatformAdapter, Platform

    # Optional import for SendResult; fall back gracefully if not present
    try:
        from gateway.platforms.base import SendResult  # type: ignore
    except Exception:

        class SendResult:
            def __init__(
                self,
                ok: bool,
                message_id: Optional[str] = None,
                error: Optional[str] = None,
            ):
                self.ok = ok
                self.message_id = message_id
                self.error = error
except Exception:
    # Minimal stand-ins if the full gateway package isn't loaded in tests.
    class Platform:
        QQ = "qq"

    class BasePlatformAdapter:
        def __init__(self, config=None, platform=None):
            self.config = config or {}
            self.platform = platform

        async def receive_event(self, event):  # pragma: no cover
            pass

    class SendResult:
        def __init__(
            self,
            ok: bool,
            message_id: Optional[str] = None,
            error: Optional[str] = None,
        ):
            self.ok = ok
            self.message_id = message_id
            self.error = error


import aiohttp
import json
import logging
import os

logger = logging.getLogger(__name__)


def check_qq_requirements() -> bool:
    """Check if QQ Bot is configured (has app_id and app_secret)."""
    has_app_id = bool(os.getenv("QQ_BOT_APP_ID") or os.getenv("QQ_APP_ID"))
    has_app_secret = bool(os.getenv("QQ_BOT_APP_SECRET") or os.getenv("QQ_APP_SECRET"))
    return has_app_id and has_app_secret


class QQAdapter(BasePlatformAdapter):
    """QQ Bot adapter using official QQ Bot API."""

    platform = Platform.QQ

    def __init__(self, config: "PlatformConfig"):
        from gateway.config import PlatformConfig as GCPlatformConfig

        super().__init__(config, Platform.QQ)

        extra = config.extra or {}

        # Get credentials from config.extra or environment variables
        self.app_id = (
            extra.get("app_id")
            or os.getenv("QQ_BOT_APP_ID")
            or os.getenv("QQ_APP_ID")
            or ""
        )
        self.app_secret = (
            extra.get("app_secret")
            or os.getenv("QQ_BOT_APP_SECRET")
            or os.getenv("QQ_APP_SECRET")
            or ""
        )
        self.refresh_token = extra.get("refresh_token") or os.getenv(
            "QQ_BOT_REFRESH_TOKEN", ""
        )
        self.token = extra.get("token") or os.getenv("QQ_BOT_TOKEN", "")
        self.ws_url = (extra.get("ws_url") or os.getenv("QQ_BOT_WS_URL") or "").rstrip(
            "/"
        )
        self.token_url = (
            extra.get("token_url")
            or os.getenv("QQ_BOT_TOKEN_URL")
            or "https://bots.qq.com/app/getAppAccessToken"
        )

        # If initial token present, pretend 2h expiry; else consider expired
        if self.token:
            self.token_expiry = time.time() + 7200
        else:
            self.token_expiry = 0.0

        self._ws = None  # placeholder for WebSocket
        self._connected = False
        self._heartbeat_task: Optional[asyncio.Task] = None

    def _token_expired(self) -> bool:
        # Expire 5 minutes before actual expiry as a safety margin
        if not self.token:
            return True
        return time.time() >= (self.token_expiry - 300)

    async def get_access_token(self) -> str:
        """Get access token using client_credentials flow."""
        headers = {"Content-Type": "application/json"}
        data = {
            "appId": self.app_id,
            "clientSecret": self.app_secret,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://bots.qq.com/app/getAppAccessToken", json=data, headers=headers
            ) as resp:
                text = await resp.text()

                # QQ Bot returns JSON
                import json

                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    raise Exception(f"Invalid JSON response: {text}")

                # Check for error (code=0 means success)
                if payload.get("code", 0) != 0:
                    raise Exception(
                        f"Failed to get access token: {payload.get('message', 'unknown error')} (code {payload.get('code')})"
                    )

        access_token = payload.get("access_token")
        if not access_token:
            raise Exception(f"No access_token in response: {payload}")
        expires_in = int(payload.get("expires_in", 7200))
        self.token_expiry = time.time() + expires_in - 300
        logger.info(
            "[%s] Access token obtained, expires in %d seconds",
            self.name,
            expires_in,
        )
        return access_token

    async def get_gateway_url(self, access_token: str) -> str:
        """Get WebSocket gateway URL from QQ Bot API."""
        headers = {"Authorization": f"QQBot {access_token}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.sgroup.qq.com/gateway/bot", headers=headers
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(
                        f"Failed to get gateway URL: {resp.status} {error_text}"
                    )
                payload = await resp.json()
        url = payload.get("url")
        if not url:
            raise Exception(f"No URL in response: {payload}")
        logger.info("[%s] Gateway URL obtained: %s", self.name, url)
        return url

    async def connect(self) -> bool:
        """Connect to QQ Bot WebSocket gateway.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            logger.info("[%s] 🚀 Starting QQ Bot connection...", self.name)

            # Get access token
            logger.info("[%s] 🔑 Requesting access token...", self.name)
            access_token = await self.get_access_token()
            self.token = access_token
            logger.info("[%s] ✅ Access token obtained", self.name)

            # Get gateway URL
            logger.info("[%s] 🌐 Requesting WebSocket gateway URL...", self.name)
            ws_url = await self.get_gateway_url(access_token)
            logger.info("[%s] ✅ Gateway URL obtained: %s", self.name, ws_url)

            # Connect to WebSocket - store session for lifecycle management
            logger.info("[%s] 🔌 Connecting to WebSocket...", self.name)
            headers = {"Authorization": f"QQBot {access_token}"}
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(ws_url, headers=headers)
            logger.info("[%s] ✅ WebSocket connection established", self.name)

            # Wait for Hello (OP=10) and send Identify
            logger.info("[%s] 🤝 Performing handshake...", self.name)
            await self._handshake()
            logger.info("[%s] ✅ Handshake completed", self.name)

            self._connected = True

            # Mark as connected
            if hasattr(self, "_mark_connected"):
                self._mark_connected()

            # Verify message handler is set
            if self._message_handler is None:
                logger.warning(
                    "[%s] ⚠️ No message handler set! Messages will not be processed.",
                    self.name,
                )
            else:
                logger.info("[%s] ✅ Message handler is configured", self.name)

            # Start listener
            logger.info("[%s] 👂 Starting message listener...", self.name)
            self._listen_task = asyncio.create_task(self._listen())
            logger.info("[%s] 🟢 QQ Bot connected and listening!", self.name)

            return True

        except Exception as e:
            logger.error("[%s] ❌ Failed to connect: %s", self.name, e)
            import traceback

            logger.error("[%s] Traceback: %s", self.name, traceback.format_exc())
            raise

    async def _handshake(self) -> None:
        """Perform WebSocket handshake: wait for Hello, send Identify, wait for READY."""
        import asyncio

        # Wait for Hello with timeout
        try:
            msg = await asyncio.wait_for(self._ws.receive(), timeout=15.0)
        except asyncio.TimeoutError:
            raise Exception("Timeout waiting for Hello from QQ Bot gateway")

        if msg.type != aiohttp.WSMsgType.TEXT:
            raise Exception(f"Expected TEXT message, got {msg.type}")

        import json

        data = json.loads(msg.data)

        if data.get("op") != 10:  # OP_HELLO
            raise Exception(f"Expected OP=10 (Hello), got OP={data.get('op')}")

        logger.info("[%s] Received Hello", self.name)

        # Send Identify
        await self._send_identify()
        logger.info("[%s] Identify sent", self.name)

        # Wait for READY with timeout
        try:
            msg = await asyncio.wait_for(self._ws.receive(), timeout=15.0)
        except asyncio.TimeoutError:
            raise Exception("Timeout waiting for READY from QQ Bot gateway")

        if msg.type != aiohttp.WSMsgType.TEXT:
            raise Exception(f"Expected TEXT message, got {msg.type}")

        data = json.loads(msg.data)

        if data.get("op") != 0 or data.get("t") != "READY":
            raise Exception(
                f"Expected READY event, got OP={data.get('op')} T={data.get('t')}"
            )

        logger.info("[%s] READY! Connected to QQ Bot gateway.", self.name)

    async def _listen(self) -> None:
        """Listen for WebSocket messages."""
        try:
            logger.info("[%s] 🎧 Listener started, waiting for messages...", self.name)
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    import json

                    try:
                        data = json.loads(msg.data)
                        event_type = data.get("t", "UNKNOWN")
                        logger.debug(
                            "[%s] 📨 Received event: %s", self.name, event_type
                        )
                        await self._handle_ws_message(data)
                    except json.JSONDecodeError:
                        logger.warning("[%s] Invalid JSON: %s", self.name, msg.data)
                    except Exception as e:
                        logger.error("[%s] Error handling message: %s", self.name, e)
                        import traceback

                        logger.error(
                            "[%s] Traceback: %s", self.name, traceback.format_exc()
                        )
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("[%s] ❌ WebSocket error", self.name)
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("[%s] WebSocket connection closed", self.name)
                    break
        except asyncio.CancelledError:
            logger.info("[%s] Listener task cancelled", self.name)
            raise
        except Exception as e:
            logger.error("[%s] Listener error: %s", self.name, e)
            import traceback

            logger.error("[%s] Traceback: %s", self.name, traceback.format_exc())
        finally:
            self._connected = False
            logger.info("[%s] Listener stopped", self.name)

    async def _handle_ws_message(self, data: dict) -> None:
        """Handle incoming WebSocket message from QQ Bot."""
        # Safely extract fields (some messages may not have all fields)
        op = data.get("op", 0)
        event_type = data.get("t", "")
        event_data = data.get("d") or {}

        # Handle opcode (skip OP=10 Hello and OP=2 Identify as they're handled in handshake)
        if op == 1:  # Heartbeat request from server
            logger.debug("[%s] Heartbeat request received", self.name)
            seq = data.get("s")
            if seq is not None:
                await self._ws.send_json({"op": 3, "d": seq})
            return

        elif op == 9:  # Invalid session
            logger.warning("[%s] Invalid session received", self.name)
            return

        elif op == 11:  # Heartbeat ACK
            logger.debug("[%s] Heartbeat ACK received", self.name)
            return

        elif op == 0:  # Dispatch - actual events
            if event_type == "READY":
                # Should already be received in handshake, but log it anyway
                logger.debug("[%s] Ready event received (duplicate)", self.name)
                return

            elif event_type == "RESUMED":
                logger.info("[%s] Session resumed", self.name)
                return

            elif event_type in ("GROUP_AT_MESSAGE_CREATE", "C2C_MESSAGE_CREATE"):
                # Handle incoming message
                await self._handle_message_create(event_data, event_type)
                return

            # Log other events for debugging
            logger.debug("[%s] Event: %s", self.name, event_type)

        else:
            # Unknown opcode
            logger.debug("[%s] Unknown opcode: %d", self.name, op)

    async def _send_identify(self) -> None:
        """Send Identify message to authenticate WebSocket connection."""
        identify_msg = {
            "op": 2,
            "d": {
                "token": f"QQBot {self.token}",
                "intents": 33554432,  # 1 << 25 = GROUP_AND_C2C_EVENT (per cc-connect)
                "shard": [0, 1],
            },
        }
        await self._ws.send_json(identify_msg)
        logger.info("[%s] Identify sent with intents=33554432", self.name)

    async def _handle_message_create(self, message: dict, event_type: str) -> None:
        """Handle MESSAGE_CREATE event (GROUP_AT_MESSAGE_CREATE or C2C_MESSAGE_CREATE)."""
        logger.info(
            "[%s] 💬 Incoming %s message",
            self.name,
            "GROUP" if event_type == "GROUP_AT_MESSAGE_CREATE" else "DM",
        )

        # Extract fields based on event type
        if event_type == "GROUP_AT_MESSAGE_CREATE":
            # Group message
            user_id = message.get("author", {}).get("member_openid")
            chat_id = message.get("group_openid")
            is_group = True
        else:  # C2C_MESSAGE_CREATE
            # Direct message
            user_id = message.get("author", {}).get("user_openid")
            chat_id = user_id
            is_group = False

        content = message.get("content", "")
        message_id = message.get("id")

        logger.info("[%s] 📥 Message details:", self.name)
        logger.info(
            "[%s]   - From: %s (%s)",
            self.name,
            user_id,
            "group" if is_group else "private",
        )
        logger.info(
            "[%s]   - Chat: %s",
            self.name,
            f"group:{chat_id}" if is_group else f"dm:{chat_id}",
        )
        logger.info(
            "[%s]   - Content: %s",
            self.name,
            content[:100] + "..." if len(content) > 100 else content,
        )
        logger.info("[%s]   - Message ID: %s", self.name, message_id)

        # Build SessionSource using base class helper
        source = self.build_source(
            chat_id=f"group:{chat_id}" if is_group else f"dm:{chat_id}",
            chat_type="group" if is_group else "dm",
            user_id=user_id,
        )

        # Create MessageEvent object (matching telegram adapter pattern)
        from gateway.platforms.base import MessageEvent, MessageType
        from datetime import datetime

        # Parse timestamp if available
        timestamp = None
        if message.get("timestamp"):
            try:
                ts = float(message["timestamp"])
                timestamp = datetime.fromtimestamp(ts)
            except (ValueError, TypeError):
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        event = MessageEvent(
            text=content,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=message,
            message_id=message_id,
            timestamp=timestamp,
        )

        # Create MessageEvent object (matching telegram adapter pattern)
        from gateway.platforms.base import MessageEvent, MessageType
        from datetime import datetime

        # Parse timestamp if available
        timestamp = None
        if message.get("timestamp"):
            try:
                ts = float(message["timestamp"])
                timestamp = datetime.fromtimestamp(ts)
            except (ValueError, TypeError):
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        event = MessageEvent(
            text=content,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=message,
            message_id=message_id,
            timestamp=timestamp,
        )

        # Route to gateway via message handler
        if self._message_handler:
            logger.info("[%s] ➡️ Routing to message handler...", self.name)
            try:
                response = await self._message_handler(event)
                logger.info("[%s] ✅ Message processed", self.name)
                
                # Send response if any
                if response:
                    logger.info("[%s] 📤 Sending response (%d chars)", self.name, len(response))
                    # Extract MEDIA:<path> tags before sending
                    media_files, response_text = self.extract_media(response)
                    
                    # Send media files first
                    for media_path in media_files:
                        try:
                            from pathlib import Path
                            if Path(media_path).exists():
                                await self.send_image(event.source.chat_id, media_path)
                                logger.info("[%s] ✅ Media sent: %s", self.name, media_path)
                        except Exception as e:
                            logger.warning("[%s] Failed to send media %s: %s", self.name, media_path, e)
                    
                    # Send text response
                    if response_text.strip():
                        result = await self.send(
                            event.source.chat_id, 
                            response_text, 
                            reply_to=event.message_id
                        )
                        if result.success:
                            logger.info("[%s] ✅ Response sent successfully", self.name)
                        else:
                            logger.error("[%s] ❌ Failed to send response: %s", self.name, result.error)
                else:
                    logger.warning("[%s] Handler returned empty/None response", self.name)
            except Exception as e:
                logger.error("[%s] ❌ Error in message handler: %s", self.name, e)
                import traceback

                logger.error("[%s] Traceback: %s", self.name, traceback.format_exc())
        else:
            logger.error(
                "[%s] ❌ No message handler set! Cannot process message.", self.name
            )

    async def _on_event(self, message: dict) -> None:
        # Route to Hermes gateway if possible
        if hasattr(self, "gateway") and self.gateway:
            try:
                await self.gateway.receive_event(message)
            except Exception:
                logger.exception("[%s] Failed to deliver event to gateway", self.name)

    async def _handle_message(self, data: dict) -> None:
        # Hook for Hermes decoding path
        await self._on_event(data)

    async def send_message(
        self, chat_id: str, content: str, msg_type: str = "text"
    ) -> SendResult:
        # Chunk content into 2000-char blocks per spec
        chunks = [content[i : i + 2000] for i in range(0, len(content), 2000)]
        last_message_id: Optional[str] = None
        for chunk in chunks:
            # In a real implementation, this would send via WS or REST; here we stub success
            last_message_id = f"mock_{hash(chunk) & 0xFFFFFFFF}"
        return SendResult(True, message_id=last_message_id)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message to QQ via REST API.

        Based on QQ Bot official API:
        - C2C: POST /message/private with user_id in body
        - Group: POST /message/group with group_id in body
        """
        try:
            # Parse chat_id to determine target type
            is_group = chat_id.startswith("group:")
            target_id = chat_id.split(":")[1] if ":" in chat_id else chat_id

            logger.info(
                "[%s] 📤 Sending message to %s (%s): %s...",
                self.name,
                "group" if is_group else "user",
                target_id,
                content[:50] + "..." if len(content) > 50 else content,
            )

            # Chunk content into 2000-char blocks per QQ spec
            chunks = [content[i : i + 2000] for i in range(0, len(content), 2000)]
            last_message_id: Optional[str] = None

            for idx, chunk in enumerate(chunks):
                # QQ Bot API format (参考 cc-connect qqbot.go 实现)
                if is_group:
                    # 群消息：POST /v2/groups/{group_openid}/messages
                    endpoint = f"https://api.sgroup.qq.com/v2/groups/{target_id}/messages"
                    payload = {
                        "content": chunk,
                        "msg_type": 0,  # 0 = 纯文本，2 = Markdown
                    }
                else:
                    # 私聊消息：POST /v2/users/{user_openid}/messages
                    endpoint = f"https://api.sgroup.qq.com/v2/users/{target_id}/messages"
                    payload = {
                        "content": chunk,
                        "msg_type": 0,  # 0 = 纯文本
                    }

                # 被动回复时需要包含 msg_id 和 msg_seq (参考 qqbot.go sendMessage 方法)
                if reply_to and idx == 0:
                    payload["msg_id"] = reply_to
                    # msg_seq 从 1 开始，每次回复同一消息递增
                    payload["msg_seq"] = 1 if idx == 0 else idx + 1

                # Prepare request headers
                headers = {
                    "Authorization": f"QQBot {self.token}",
                    "Content-Type": "application/json",
                }

                # Send via REST API
                async with aiohttp.ClientSession() as session:
                    logger.debug(
                        "[%s] Sending to %s with payload: %s",
                        self.name,
                        endpoint,
                        payload,
                    )

                    async with session.post(
                        endpoint, json=payload, headers=headers
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            logger.error(
                                "[%s] Failed to send message: %s %s",
                                self.name,
                                resp.status,
                                error_text,
                            )
                            return SendResult(
                                False, error=f"HTTP {resp.status}: {error_text}"
                            )

                        result = await resp.json()
                        last_message_id = result.get("id")
                        logger.info(
                            "[%s] Message chunk %d/%d sent successfully (code=%d)",
                            self.name,
                            idx + 1,
                            len(chunks),
                            resp.status,
                        )

            logger.info(
                "[%s] ✅ Message sent successfully (ID: %s)", self.name, last_message_id
            )
            return SendResult(True, message_id=last_message_id)

        except Exception as e:
            logger.error("[%s] Error sending message: %s", self.name, e)
            import traceback

            logger.error("[%s] Traceback: %s", self.name, traceback.format_exc())
            return SendResult(False, error=str(e))

    # Backwards-compatible shim
    async def send_message(
        self, chat_id: str, content: str, msg_type: str = "text"
    ) -> SendResult:
        return await self.send(chat_id, content, None, None)

    async def send_typing(self, chat_id: str) -> None:
        # QQ does not support typing indicators per design doc - no-op
        return None

    async def get_chat_info(self, chat_id: str) -> dict:
        # Minimal chat info
        is_group = chat_id.startswith("g:")
        chat_type = "group" if is_group else "dm"
        name = chat_id
        return {"chat_id": chat_id, "name": name, "type": chat_type}

    async def disconnect(self) -> None:
        logger.info("[%s] 🔌 Disconnecting QQ Bot...", self.name)
        self._connected = False

        # Cancel listener task
        if hasattr(self, "_listen_task") and self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            logger.info("[%s] Listener task cancelled", self.name)

        # Close WebSocket
        if self._ws:
            await self._ws.close()
            logger.info("[%s] WebSocket closed", self.name)

        # Close session
        if hasattr(self, "_session") and self._session:
            await self._session.close()
            logger.info("[%s] Session closed", self.name)

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

        logger.info("[%s] ✅ QQ Bot disconnected", self.name)

    async def _heartbeat_loop(self) -> None:
        try:
            while self._connected:
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            pass

    async def _reconnect(self) -> None:
        delay = 1.0
        while not self._connected:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
            await self.connect()

    async def handle_message(self, event: Dict[str, Any]) -> None:
        if hasattr(self, "gateway") and self.gateway:
            await self.gateway.receive_event(event)
