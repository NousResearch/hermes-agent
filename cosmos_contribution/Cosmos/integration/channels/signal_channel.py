"""
Cosmos Signal Channel Adapter
==================================

Signal Messenger integration for the Cosmos swarm.

Features:
- End-to-end encrypted messaging
- Groups and group v2
- Attachments
- Reactions
- Typing indicators
- Read receipts
- Disappearing messages
- Stickers

Based on signal-cli (Java) or signald (Go) daemon.

Setup:
1. Install signal-cli: https://github.com/AsamK/signal-cli
2. Register or link phone number
3. Run as daemon: signal-cli -a +1234567890 daemon --json
4. Set SIGNAL_CLI_URL env var (default: unix socket or TCP)

"Signal: Privacy first. Now with AGI that respects it." - The Collective
"""

import os
import asyncio
import json
from typing import Optional, Dict, List, Any
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


class SignalChannel(BaseChannel):
    """
    Signal Messenger channel adapter via signal-cli daemon.

    Supports:
    - Direct messages (1:1)
    - Groups (v1 and v2)
    - Attachments (images, videos, audio, files)
    - Reactions
    - Quotes (replies)
    - Typing indicators
    """

    def __init__(
        self,
        config: ChannelConfig = None,
        phone_number: str = None,
        socket_path: str = None,
        tcp_host: str = None,
        tcp_port: int = None
    ):
        """
        Initialize Signal channel.

        Args:
            config: Channel configuration
            phone_number: Registered Signal phone number (+1234567890)
            socket_path: Unix socket path for signal-cli
            tcp_host: TCP host for signal-cli daemon
            tcp_port: TCP port for signal-cli daemon
        """
        config = config or ChannelConfig(channel_type=ChannelType.SIGNAL)
        super().__init__(config)

        self.phone_number = phone_number or os.environ.get("SIGNAL_PHONE_NUMBER")
        self.socket_path = socket_path or os.environ.get("SIGNAL_SOCKET_PATH")
        self.tcp_host = tcp_host or os.environ.get("SIGNAL_TCP_HOST", "localhost")
        self.tcp_port = tcp_port or int(os.environ.get("SIGNAL_TCP_PORT", "7583"))

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._request_id = 0

    async def connect(self) -> bool:
        """Connect to signal-cli daemon."""
        if not self.phone_number:
            logger.error("Signal phone number not configured")
            return False

        try:
            # Connect via Unix socket or TCP
            if self.socket_path and Path(self.socket_path).exists():
                self._reader, self._writer = await asyncio.open_unix_connection(
                    self.socket_path
                )
                logger.info(f"Signal connected via Unix socket: {self.socket_path}")
            else:
                self._reader, self._writer = await asyncio.open_connection(
                    self.tcp_host, self.tcp_port
                )
                logger.info(f"Signal connected via TCP: {self.tcp_host}:{self.tcp_port}")

            # Subscribe to receive messages
            await self._send_command("subscribe", {"account": self.phone_number})

            # Start listening
            self._listen_task = asyncio.create_task(self._listen_messages())

            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Signal connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from signal-cli."""
        if self._listen_task:
            self._listen_task.cancel()

        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass

        self._connected = False
        logger.info("Signal disconnected")

    async def _send_command(self, method: str, params: Dict = None) -> Optional[Dict]:
        """Send a JSON-RPC command to signal-cli."""
        if not self._writer:
            return None

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {}
        }

        try:
            line = json.dumps(request) + "\n"
            self._writer.write(line.encode())
            await self._writer.drain()

            # Read response
            response_line = await asyncio.wait_for(
                self._reader.readline(),
                timeout=30
            )
            if response_line:
                return json.loads(response_line.decode())
            return None

        except Exception as e:
            logger.error(f"Signal command failed: {e}")
            return None

    async def _listen_messages(self):
        """Listen for incoming messages."""
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break

                try:
                    data = json.loads(line.decode())
                    await self._handle_signal_event(data)
                except json.JSONDecodeError:
                    continue

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Signal listen error: {e}")
            self._connected = False

    async def _handle_signal_event(self, data: Dict):
        """Handle signal-cli events."""
        # Check for message event
        if data.get("method") == "receive":
            envelope = data.get("params", {}).get("envelope", {})
            await self._handle_envelope(envelope)

    async def _handle_envelope(self, envelope: Dict):
        """Handle a Signal envelope (message container)."""
        source = envelope.get("source") or envelope.get("sourceNumber")
        if not source:
            return

        # Skip own messages
        if source == self.phone_number:
            return

        data_msg = envelope.get("dataMessage", {})
        sync_msg = envelope.get("syncMessage", {})

        # Handle data message (incoming)
        if data_msg:
            await self._process_data_message(envelope, data_msg)

        # Handle sync message (sent from another device)
        elif sync_msg and sync_msg.get("sentMessage"):
            sent = sync_msg["sentMessage"]
            # We sent this, so skip
            pass

    async def _process_data_message(self, envelope: Dict, data_msg: Dict):
        """Process a data message."""
        source = envelope.get("source") or envelope.get("sourceNumber")
        source_name = envelope.get("sourceName") or source
        timestamp = envelope.get("timestamp", 0)

        # Check if group message
        group_info = data_msg.get("groupInfo") or data_msg.get("groupV2")
        is_group = group_info is not None
        group_id = None
        group_name = None

        if group_info:
            group_id = group_info.get("groupId")
            group_name = group_info.get("name") or group_info.get("title")

        # Build channel ID
        channel_id = group_id if is_group else source

        # Build normalized message
        message = ChannelMessage(
            message_id=str(timestamp),
            channel_type=ChannelType.SIGNAL,
            channel_id=channel_id,
            sender_id=source,
            sender_name=source_name,
            text=data_msg.get("message", ""),
            is_group=is_group,
            group_name=group_name,
            timestamp=datetime.fromtimestamp(timestamp / 1000) if timestamp else datetime.now(),
            raw_data=envelope
        )

        # Handle quote (reply)
        quote = data_msg.get("quote")
        if quote:
            message.reply_to_id = str(quote.get("id"))
            message.reply_to_text = quote.get("text")

        # Handle attachments
        attachments = data_msg.get("attachments", [])
        if attachments:
            att = attachments[0]
            content_type = att.get("contentType", "")

            if content_type.startswith("image"):
                message.media_type = "image"
            elif content_type.startswith("video"):
                message.media_type = "video"
            elif content_type.startswith("audio"):
                message.media_type = "audio"
            else:
                message.media_type = "document"

            # Attachment file path (signal-cli stores locally)
            message.media_url = att.get("filename") or att.get("id")

        # Handle sticker
        sticker = data_msg.get("sticker")
        if sticker:
            message.media_type = "sticker"
            message.raw_data["sticker"] = sticker

        # Handle reaction (separate from message)
        reaction = data_msg.get("reaction")
        if reaction:
            message.text = f"[Reaction: {reaction.get('emoji')}]"
            message.raw_data["reaction"] = reaction

        # Pass to handler
        await self._handle_inbound(message)

    async def send_message(
        self,
        recipient: str,
        text: str,
        media_path: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Send a message via Signal.

        Args:
            recipient: Phone number or group ID
            text: Message text
            media_path: Optional attachment path
            reply_to: Timestamp of message to reply to

        Returns:
            True if sent successfully
        """
        if not self._writer:
            return False

        if not self._check_rate_limit(recipient):
            logger.warning(f"Rate limit exceeded for {recipient}")
            return False

        try:
            params = {
                "account": self.phone_number,
                "message": text,
            }

            # Determine if group or direct
            if recipient.startswith("+") or recipient.replace("-", "").isdigit():
                params["recipient"] = [recipient]
            else:
                params["groupId"] = recipient

            # Handle attachment
            if media_path and Path(media_path).exists():
                params["attachment"] = [str(media_path)]

            # Handle reply
            if reply_to:
                params["quoteTimestamp"] = int(reply_to)
                params["quoteAuthor"] = kwargs.get("quote_author", self.phone_number)

            # Send typing indicator
            if self.config.typing_indicator:
                await self._send_command("sendTyping", {
                    "account": self.phone_number,
                    "recipient": params.get("recipient"),
                    "groupId": params.get("groupId"),
                })
                await asyncio.sleep(1)

            result = await self._send_command("send", params)
            return result is not None and "error" not in result

        except Exception as e:
            logger.error(f"Signal send failed: {e}")
            return False

    async def send_reaction(
        self,
        recipient: str,
        target_timestamp: str,
        target_author: str,
        emoji: str
    ) -> bool:
        """Send a reaction to a message."""
        try:
            params = {
                "account": self.phone_number,
                "emoji": emoji,
                "targetTimestamp": int(target_timestamp),
                "targetAuthor": target_author,
            }

            if recipient.startswith("+"):
                params["recipient"] = [recipient]
            else:
                params["groupId"] = recipient

            result = await self._send_command("sendReaction", params)
            return result is not None and "error" not in result

        except Exception as e:
            logger.error(f"Signal reaction failed: {e}")
            return False

    async def send_read_receipt(self, sender: str, timestamps: List[str]) -> bool:
        """Send read receipt for messages."""
        try:
            result = await self._send_command("sendReceipt", {
                "account": self.phone_number,
                "recipient": sender,
                "type": "read",
                "targetTimestamp": [int(ts) for ts in timestamps]
            })
            return result is not None and "error" not in result
        except Exception as e:
            logger.error(f"Read receipt failed: {e}")
            return False

    async def get_contacts(self) -> List[Dict]:
        """Get Signal contacts."""
        try:
            result = await self._send_command("listContacts", {
                "account": self.phone_number
            })
            if result and "result" in result:
                return result["result"]
            return []
        except Exception as e:
            logger.error(f"Get contacts failed: {e}")
            return []

    async def get_groups(self) -> List[Dict]:
        """Get Signal groups."""
        try:
            result = await self._send_command("listGroups", {
                "account": self.phone_number
            })
            if result and "result" in result:
                return result["result"]
            return []
        except Exception as e:
            logger.error(f"Get groups failed: {e}")
            return []

    async def create_group(self, name: str, members: List[str]) -> Optional[str]:
        """Create a new Signal group."""
        try:
            result = await self._send_command("updateGroup", {
                "account": self.phone_number,
                "name": name,
                "member": members
            })
            if result and "result" in result:
                return result["result"].get("groupId")
            return None
        except Exception as e:
            logger.error(f"Create group failed: {e}")
            return None

    async def set_profile(self, name: str, avatar_path: str = None):
        """Set Signal profile name and avatar."""
        try:
            params = {
                "account": self.phone_number,
                "name": name
            }
            if avatar_path:
                params["avatar"] = avatar_path

            await self._send_command("updateProfile", params)
        except Exception as e:
            logger.error(f"Set profile failed: {e}")

    async def trust_identity(self, number: str, trust_all: bool = False):
        """Trust a contact's identity key."""
        try:
            await self._send_command("trust", {
                "account": self.phone_number,
                "recipient": number,
                "trustAllKnownKeys": trust_all
            })
        except Exception as e:
            logger.error(f"Trust identity failed: {e}")
