"""
Cosmos WhatsApp Channel Adapter
====================================

WhatsApp integration for the Cosmos swarm.

Features:
- Multi-device support via Baileys protocol
- QR code pairing
- Groups and broadcasts
- Media (images, videos, audio, documents)
- Voice messages
- Reactions
- Read receipts
- Typing indicators
- Contact cards

Based on whatsapp-web.js (Node.js bridge) or Baileys protocol.

Setup:
1. Install whatsapp-web.js: npm install whatsapp-web.js
2. Run Node.js bridge server
3. Scan QR code to pair
4. Set WHATSAPP_BRIDGE_URL env var

"WhatsApp: 2 billion users. Now they can talk to AGI." - The Collective
"""

import os
import asyncio
import aiohttp
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
from pathlib import Path
import base64
from loguru import logger

from .channel_hub import (
    BaseChannel,
    ChannelConfig,
    ChannelMessage,
    ChannelType,
)


class WhatsAppChannel(BaseChannel):
    """
    WhatsApp channel adapter via Node.js bridge.

    The bridge server (whatsapp-web.js or Baileys) handles the actual
    WhatsApp Web connection. This adapter communicates via HTTP/WebSocket.

    Supports:
    - Direct messages
    - Group chats
    - Broadcast lists
    - Media messages
    - Voice notes
    - Reactions
    - Reply context
    """

    def __init__(
        self,
        config: ChannelConfig = None,
        bridge_url: str = None,
        session_name: str = "cosmos"
    ):
        """
        Initialize WhatsApp channel.

        Args:
            config: Channel configuration
            bridge_url: URL of the WhatsApp bridge server
            session_name: Session identifier for multi-account support
        """
        config = config or ChannelConfig(channel_type=ChannelType.WHATSAPP)
        super().__init__(config)

        self.bridge_url = bridge_url or os.environ.get("WHATSAPP_BRIDGE_URL", "http://localhost:3000")
        self.session_name = session_name

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._phone_number: Optional[str] = None
        self._listen_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Connect to WhatsApp via bridge server."""
        try:
            self._session = aiohttp.ClientSession()

            # Check bridge health
            async with self._session.get(f"{self.bridge_url}/health") as resp:
                if resp.status != 200:
                    logger.error("WhatsApp bridge not responding")
                    return False

            # Initialize session
            async with self._session.post(
                f"{self.bridge_url}/session/init",
                json={"session": self.session_name}
            ) as resp:
                data = await resp.json()
                if data.get("status") == "qr_needed":
                    logger.info("WhatsApp QR code needed - check bridge server")
                    # Could emit QR code event here for UI
                elif data.get("status") == "ready":
                    self._phone_number = data.get("phone_number")
                    logger.info(f"WhatsApp ready: {self._phone_number}")

            # Connect WebSocket for real-time events
            ws_url = self.bridge_url.replace("http", "ws") + f"/ws/{self.session_name}"
            self._ws = await self._session.ws_connect(ws_url)

            # Start listening for messages
            self._listen_task = asyncio.create_task(self._listen_messages())

            self._connected = True
            return True

        except Exception as e:
            logger.error(f"WhatsApp connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from WhatsApp."""
        if self._listen_task:
            self._listen_task.cancel()

        if self._ws:
            await self._ws.close()

        if self._session:
            await self._session.close()

        self._connected = False
        logger.info("WhatsApp disconnected")

    async def _listen_messages(self):
        """Listen for incoming messages via WebSocket."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = msg.json()
                    event_type = data.get("event")

                    if event_type == "message":
                        await self._handle_whatsapp_message(data.get("data", {}))
                    elif event_type == "qr":
                        logger.info(f"WhatsApp QR: {data.get('qr')}")
                    elif event_type == "ready":
                        self._phone_number = data.get("phone_number")
                        logger.info(f"WhatsApp authenticated: {self._phone_number}")
                    elif event_type == "disconnected":
                        logger.warning("WhatsApp disconnected from server")
                        self._connected = False

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WhatsApp WebSocket error: {msg.data}")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"WhatsApp listen error: {e}")

    async def send_message(
        self,
        chat_id: str,
        text: str,
        media_path: Optional[str] = None,
        reply_to: Optional[str] = None,
        mentions: List[str] = None,
        **kwargs
    ) -> bool:
        """
        Send a message to a WhatsApp chat.

        Args:
            chat_id: WhatsApp chat ID (phone@c.us or group@g.us)
            text: Message text
            media_path: Optional media file path
            reply_to: Message ID to reply to
            mentions: List of phone numbers to mention

        Returns:
            True if sent successfully
        """
        if not self._session:
            return False

        if not self._check_rate_limit(chat_id):
            logger.warning(f"Rate limit exceeded for chat {chat_id}")
            return False

        try:
            # Normalize chat ID
            if not chat_id.endswith("@c.us") and not chat_id.endswith("@g.us"):
                # Assume it's a phone number for DM
                chat_id = f"{chat_id}@c.us"

            payload = {
                "session": self.session_name,
                "chatId": chat_id,
                "text": text,
            }

            if reply_to:
                payload["quotedMessageId"] = reply_to

            if mentions:
                payload["mentions"] = mentions

            # Send typing indicator
            if self.config.typing_indicator:
                await self._session.post(
                    f"{self.bridge_url}/chat/typing",
                    json={"session": self.session_name, "chatId": chat_id}
                )
                await asyncio.sleep(1)

            # Handle media
            if media_path and Path(media_path).exists():
                media_path = Path(media_path)
                with open(media_path, "rb") as f:
                    media_data = base64.b64encode(f.read()).decode()

                suffix = media_path.suffix.lower()
                media_type = "document"
                if suffix in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                    media_type = "image"
                elif suffix in [".mp4", ".mov", ".avi", ".webm"]:
                    media_type = "video"
                elif suffix in [".mp3", ".wav", ".ogg", ".m4a"]:
                    if kwargs.get("as_voice") or kwargs.get("ptt"):
                        media_type = "voice"
                    else:
                        media_type = "audio"

                payload["media"] = {
                    "type": media_type,
                    "data": media_data,
                    "filename": media_path.name,
                    "caption": text
                }
                payload["text"] = None  # Caption is in media

            async with self._session.post(
                f"{self.bridge_url}/message/send",
                json=payload
            ) as resp:
                result = await resp.json()
                return result.get("success", False)

        except Exception as e:
            logger.error(f"WhatsApp send failed: {e}")
            return False

    async def _handle_whatsapp_message(self, data: Dict):
        """Handle incoming WhatsApp message."""
        # Skip own messages
        if data.get("fromMe"):
            return

        chat_id = data.get("from", "")
        sender = data.get("author") or data.get("from", "")
        is_group = "@g.us" in chat_id

        # Get sender name
        sender_name = data.get("notifyName") or data.get("pushName") or sender.split("@")[0]

        # Build normalized message
        message = ChannelMessage(
            message_id=data.get("id", {}).get("_serialized", ""),
            channel_type=ChannelType.WHATSAPP,
            channel_id=chat_id,
            sender_id=sender,
            sender_name=sender_name,
            text=data.get("body", ""),
            is_group=is_group,
            group_name=data.get("chatName") if is_group else None,
            timestamp=datetime.fromtimestamp(data.get("timestamp", 0)),
            raw_data=data
        )

        # Check for mentions
        if data.get("mentionedIds"):
            message.is_mention = self._phone_number in data["mentionedIds"]

        # Handle reply context
        if data.get("hasQuotedMsg"):
            quoted = data.get("quotedMsg", {})
            message.reply_to_id = quoted.get("id", {}).get("_serialized")
            message.reply_to_text = quoted.get("body")

        # Handle media
        if data.get("hasMedia"):
            media_type = data.get("type", "")
            message.media_type = {
                "image": "image",
                "video": "video",
                "audio": "audio",
                "ptt": "audio",  # voice note
                "document": "document",
                "sticker": "sticker",
            }.get(media_type, "document")

            # Media URL would need to be fetched from bridge
            message.media_url = data.get("mediaUrl")

        # Handle location
        if data.get("location"):
            loc = data["location"]
            message.raw_data["location"] = {
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude"),
                "description": loc.get("description")
            }

        # Handle contact card
        if data.get("vCards"):
            message.raw_data["contacts"] = data["vCards"]

        # Send read receipt if configured
        if self.config.auto_react:
            try:
                await self._session.post(
                    f"{self.bridge_url}/message/seen",
                    json={
                        "session": self.session_name,
                        "chatId": chat_id,
                        "messageId": message.message_id
                    }
                )
            except Exception:
                pass

        # Pass to handler
        await self._handle_inbound(message)

    async def send_reaction(self, chat_id: str, message_id: str, emoji: str) -> bool:
        """React to a message with an emoji."""
        if not self._session:
            return False

        try:
            async with self._session.post(
                f"{self.bridge_url}/message/react",
                json={
                    "session": self.session_name,
                    "chatId": chat_id,
                    "messageId": message_id,
                    "emoji": emoji
                }
            ) as resp:
                result = await resp.json()
                return result.get("success", False)
        except Exception as e:
            logger.error(f"Reaction failed: {e}")
            return False

    async def send_location(
        self,
        chat_id: str,
        latitude: float,
        longitude: float,
        description: str = None
    ) -> bool:
        """Send a location message."""
        if not self._session:
            return False

        try:
            async with self._session.post(
                f"{self.bridge_url}/message/location",
                json={
                    "session": self.session_name,
                    "chatId": chat_id,
                    "latitude": latitude,
                    "longitude": longitude,
                    "description": description
                }
            ) as resp:
                result = await resp.json()
                return result.get("success", False)
        except Exception as e:
            logger.error(f"Location send failed: {e}")
            return False

    async def send_contact(self, chat_id: str, contact_id: str) -> bool:
        """Send a contact card."""
        if not self._session:
            return False

        try:
            async with self._session.post(
                f"{self.bridge_url}/message/contact",
                json={
                    "session": self.session_name,
                    "chatId": chat_id,
                    "contactId": contact_id
                }
            ) as resp:
                result = await resp.json()
                return result.get("success", False)
        except Exception as e:
            logger.error(f"Contact send failed: {e}")
            return False

    async def get_chat_info(self, chat_id: str) -> Optional[Dict]:
        """Get information about a chat."""
        if not self._session:
            return None

        try:
            async with self._session.get(
                f"{self.bridge_url}/chat/info",
                params={"session": self.session_name, "chatId": chat_id}
            ) as resp:
                data = await resp.json()
                if data.get("success"):
                    return data.get("chat")
                return None
        except Exception as e:
            logger.error(f"Get chat info failed: {e}")
            return None

    async def get_contacts(self) -> List[Dict]:
        """Get all contacts."""
        if not self._session:
            return []

        try:
            async with self._session.get(
                f"{self.bridge_url}/contacts",
                params={"session": self.session_name}
            ) as resp:
                data = await resp.json()
                return data.get("contacts", [])
        except Exception as e:
            logger.error(f"Get contacts failed: {e}")
            return []

    async def get_groups(self) -> List[Dict]:
        """Get all groups."""
        if not self._session:
            return []

        try:
            async with self._session.get(
                f"{self.bridge_url}/groups",
                params={"session": self.session_name}
            ) as resp:
                data = await resp.json()
                return data.get("groups", [])
        except Exception as e:
            logger.error(f"Get groups failed: {e}")
            return []

    async def download_media(self, message_id: str) -> Optional[bytes]:
        """Download media from a message."""
        if not self._session:
            return None

        try:
            async with self._session.get(
                f"{self.bridge_url}/message/media",
                params={"session": self.session_name, "messageId": message_id}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("data"):
                        return base64.b64decode(data["data"])
                return None
        except Exception as e:
            logger.error(f"Media download failed: {e}")
            return None

    async def get_qr_code(self) -> Optional[str]:
        """Get QR code for pairing (if not authenticated)."""
        if not self._session:
            return None

        try:
            async with self._session.get(
                f"{self.bridge_url}/session/qr",
                params={"session": self.session_name}
            ) as resp:
                data = await resp.json()
                return data.get("qr")
        except Exception as e:
            logger.error(f"Get QR failed: {e}")
            return None

    async def logout(self) -> bool:
        """Log out and destroy session."""
        if not self._session:
            return False

        try:
            async with self._session.post(
                f"{self.bridge_url}/session/logout",
                json={"session": self.session_name}
            ) as resp:
                result = await resp.json()
                return result.get("success", False)
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False
