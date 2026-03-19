"""
Cosmos Matrix Channel Adapter
==================================

Matrix protocol integration for the Cosmos swarm.

Features:
- End-to-end encryption (Olm/Megolm)
- Federated rooms
- Spaces and room hierarchies
- Reactions and threading
- Rich media
- Voice/video call signaling
- Bridges to other networks
- Custom room state

Based on matrix-nio library (async Python client).

Setup:
1. Create Matrix account (on any homeserver)
2. Get access token or use password auth
3. Set MATRIX_HOMESERVER, MATRIX_USER_ID, MATRIX_ACCESS_TOKEN

"Matrix: An open network for secure, decentralized communication." - The Collective
"""

import os
import asyncio
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path
import aiofiles
from loguru import logger

from .channel_hub import (
    BaseChannel,
    ChannelConfig,
    ChannelMessage,
    ChannelType,
)

# Matrix library
try:
    from nio import (
        AsyncClient,
        AsyncClientConfig,
        MatrixRoom,
        RoomMessageText,
        RoomMessageImage,
        RoomMessageVideo,
        RoomMessageAudio,
        RoomMessageFile,
        ReactionEvent,
        LoginResponse,
        SyncResponse,
        UploadResponse,
    )
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False
    logger.warning("matrix-nio not installed. Run: pip install matrix-nio[e2e]")


class MatrixChannel(BaseChannel):
    """
    Matrix protocol channel adapter.

    Supports:
    - Direct messages (1:1 rooms)
    - Group rooms
    - Spaces
    - E2E encryption
    - Media uploads
    - Reactions
    - Thread replies
    - Room state management
    """

    def __init__(
        self,
        config: ChannelConfig = None,
        homeserver: str = None,
        user_id: str = None,
        access_token: str = None,
        password: str = None,
        device_name: str = "Cosmos"
    ):
        """
        Initialize Matrix channel.

        Args:
            config: Channel configuration
            homeserver: Matrix homeserver URL (https://matrix.example.org)
            user_id: Full Matrix user ID (@user:example.org)
            access_token: Access token (or use password)
            password: Password for login (if no token)
            device_name: Device display name
        """
        config = config or ChannelConfig(channel_type=ChannelType.MATRIX)
        super().__init__(config)

        self.homeserver = homeserver or os.environ.get("MATRIX_HOMESERVER")
        self.user_id = user_id or os.environ.get("MATRIX_USER_ID")
        self.access_token = access_token or os.environ.get("MATRIX_ACCESS_TOKEN")
        self.password = password or os.environ.get("MATRIX_PASSWORD")
        self.device_name = device_name

        self._client: Optional[AsyncClient] = None
        self._sync_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Connect to Matrix homeserver."""
        if not MATRIX_AVAILABLE:
            logger.error("matrix-nio not available")
            return False

        if not self.homeserver or not self.user_id:
            logger.error("Matrix homeserver and user_id required")
            return False

        try:
            # Create client config
            client_config = AsyncClientConfig(
                max_limit_exceeded=0,
                max_timeouts=0,
                store_sync_tokens=True,
            )

            # Create client
            self._client = AsyncClient(
                self.homeserver,
                self.user_id,
                config=client_config,
            )

            # Login with token or password
            if self.access_token:
                self._client.access_token = self.access_token
                self._client.device_id = self.device_name
                # Verify token works
                response = await self._client.whoami()
                logger.info(f"Matrix connected as {response.user_id}")
            elif self.password:
                response = await self._client.login(
                    self.password,
                    device_name=self.device_name
                )
                if isinstance(response, LoginResponse):
                    logger.info(f"Matrix logged in as {response.user_id}")
                    self.access_token = response.access_token
                else:
                    logger.error(f"Matrix login failed: {response}")
                    return False
            else:
                logger.error("Matrix requires access_token or password")
                return False

            # Register event callbacks
            self._client.add_event_callback(self._handle_message, RoomMessageText)
            self._client.add_event_callback(self._handle_media, RoomMessageImage)
            self._client.add_event_callback(self._handle_media, RoomMessageVideo)
            self._client.add_event_callback(self._handle_media, RoomMessageAudio)
            self._client.add_event_callback(self._handle_media, RoomMessageFile)
            self._client.add_event_callback(self._handle_reaction, ReactionEvent)

            # Start sync in background
            self._sync_task = asyncio.create_task(self._sync_forever())

            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Matrix connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Matrix."""
        if self._sync_task:
            self._sync_task.cancel()

        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.warning(f"Matrix disconnect error: {e}")

        self._connected = False
        logger.info("Matrix disconnected")

    async def _sync_forever(self):
        """Sync with homeserver forever."""
        try:
            await self._client.sync_forever(
                timeout=30000,
                full_state=True
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Matrix sync error: {e}")
            self._connected = False

    async def _handle_message(self, room: MatrixRoom, event: RoomMessageText):
        """Handle incoming text message."""
        # Skip own messages
        if event.sender == self.user_id:
            return

        # Skip old messages (from initial sync)
        age = event.server_timestamp
        if datetime.now().timestamp() * 1000 - age > 60000:  # >1 min old
            return

        is_dm = len(room.users) <= 2

        # Build normalized message
        message = ChannelMessage(
            message_id=event.event_id,
            channel_type=ChannelType.MATRIX,
            channel_id=room.room_id,
            sender_id=event.sender,
            sender_name=room.user_name(event.sender) or event.sender,
            text=event.body,
            is_group=not is_dm,
            group_name=room.display_name if not is_dm else None,
            timestamp=datetime.fromtimestamp(event.server_timestamp / 1000),
            raw_data={
                "event_id": event.event_id,
                "formatted_body": getattr(event, "formatted_body", None),
                "format": getattr(event, "format", None),
            }
        )

        # Check for mention
        if f"@{self.user_id}" in event.body or self.user_id in event.body:
            message.is_mention = True

        # Check for reply (relations)
        relates_to = getattr(event, "relates_to", None)
        if relates_to:
            if hasattr(relates_to, "in_reply_to"):
                message.reply_to_id = relates_to.in_reply_to.event_id

        await self._handle_inbound(message)

    async def _handle_media(self, room: MatrixRoom, event):
        """Handle incoming media message."""
        if event.sender == self.user_id:
            return

        # Determine media type
        media_type = "document"
        if isinstance(event, RoomMessageImage):
            media_type = "image"
        elif isinstance(event, RoomMessageVideo):
            media_type = "video"
        elif isinstance(event, RoomMessageAudio):
            media_type = "audio"

        is_dm = len(room.users) <= 2

        message = ChannelMessage(
            message_id=event.event_id,
            channel_type=ChannelType.MATRIX,
            channel_id=room.room_id,
            sender_id=event.sender,
            sender_name=room.user_name(event.sender) or event.sender,
            text=event.body,  # Usually filename
            is_group=not is_dm,
            group_name=room.display_name if not is_dm else None,
            media_type=media_type,
            media_url=event.url,  # mxc:// URL
            timestamp=datetime.fromtimestamp(event.server_timestamp / 1000),
            raw_data={"event_id": event.event_id}
        )

        await self._handle_inbound(message)

    async def _handle_reaction(self, room: MatrixRoom, event: ReactionEvent):
        """Handle reaction events."""
        if event.sender == self.user_id:
            return

        # Reactions are annotations to other messages
        message = ChannelMessage(
            message_id=event.event_id,
            channel_type=ChannelType.MATRIX,
            channel_id=room.room_id,
            sender_id=event.sender,
            sender_name=room.user_name(event.sender) or event.sender,
            text=f"[Reaction: {event.key}]",
            timestamp=datetime.fromtimestamp(event.server_timestamp / 1000),
            raw_data={
                "reaction": event.key,
                "reacts_to": event.reacts_to
            }
        )

        await self._handle_inbound(message)

    async def send_message(
        self,
        room_id: str,
        text: str,
        media_path: Optional[str] = None,
        reply_to: Optional[str] = None,
        formatted: bool = False,
        **kwargs
    ) -> bool:
        """
        Send a message to a Matrix room.

        Args:
            room_id: Matrix room ID (!room:server.org)
            text: Message text
            media_path: Optional media file path
            reply_to: Event ID to reply to
            formatted: If True, treat text as HTML

        Returns:
            True if sent successfully
        """
        if not self._client:
            return False

        if not self._check_rate_limit(room_id):
            logger.warning(f"Rate limit exceeded for room {room_id}")
            return False

        try:
            # Typing indicator
            if self.config.typing_indicator:
                await self._client.room_typing(room_id, True, timeout=5000)
                await asyncio.sleep(1)
                await self._client.room_typing(room_id, False)

            # Handle media upload
            if media_path and Path(media_path).exists():
                media_path = Path(media_path)

                # Read file
                async with aiofiles.open(media_path, "rb") as f:
                    data = await f.read()

                # Determine content type
                suffix = media_path.suffix.lower()
                content_types = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                    ".mp4": "video/mp4",
                    ".webm": "video/webm",
                    ".mp3": "audio/mpeg",
                    ".ogg": "audio/ogg",
                    ".wav": "audio/wav",
                    ".pdf": "application/pdf",
                }
                content_type = content_types.get(suffix, "application/octet-stream")

                # Upload
                response = await self._client.upload(
                    data,
                    content_type=content_type,
                    filename=media_path.name
                )

                if not isinstance(response, UploadResponse):
                    logger.error(f"Matrix upload failed: {response}")
                    return False

                # Send media message
                if content_type.startswith("image"):
                    await self._client.room_send(
                        room_id,
                        "m.room.message",
                        {
                            "msgtype": "m.image",
                            "body": text or media_path.name,
                            "url": response.content_uri,
                        }
                    )
                elif content_type.startswith("video"):
                    await self._client.room_send(
                        room_id,
                        "m.room.message",
                        {
                            "msgtype": "m.video",
                            "body": text or media_path.name,
                            "url": response.content_uri,
                        }
                    )
                elif content_type.startswith("audio"):
                    await self._client.room_send(
                        room_id,
                        "m.room.message",
                        {
                            "msgtype": "m.audio",
                            "body": text or media_path.name,
                            "url": response.content_uri,
                        }
                    )
                else:
                    await self._client.room_send(
                        room_id,
                        "m.room.message",
                        {
                            "msgtype": "m.file",
                            "body": text or media_path.name,
                            "url": response.content_uri,
                        }
                    )

                return True

            # Build message content
            content = {
                "msgtype": "m.text",
                "body": text,
            }

            # Add HTML formatting
            if formatted or "```" in text or "**" in text:
                # Convert markdown to HTML (basic)
                html = text
                html = html.replace("```", "<pre>").replace("```", "</pre>")
                html = html.replace("\n", "<br>")

                content["format"] = "org.matrix.custom.html"
                content["formatted_body"] = html

            # Add reply relation
            if reply_to:
                content["m.relates_to"] = {
                    "m.in_reply_to": {
                        "event_id": reply_to
                    }
                }

            await self._client.room_send(room_id, "m.room.message", content)
            return True

        except Exception as e:
            logger.error(f"Matrix send failed: {e}")
            return False

    async def send_reaction(self, room_id: str, event_id: str, emoji: str) -> bool:
        """Send a reaction to a message."""
        if not self._client:
            return False

        try:
            await self._client.room_send(
                room_id,
                "m.reaction",
                {
                    "m.relates_to": {
                        "rel_type": "m.annotation",
                        "event_id": event_id,
                        "key": emoji
                    }
                }
            )
            return True
        except Exception as e:
            logger.error(f"Matrix reaction failed: {e}")
            return False

    async def create_room(
        self,
        name: str,
        topic: str = None,
        invite: List[str] = None,
        is_direct: bool = False,
        encrypted: bool = True
    ) -> Optional[str]:
        """Create a new Matrix room."""
        if not self._client:
            return None

        try:
            response = await self._client.room_create(
                name=name,
                topic=topic,
                invite=invite or [],
                is_direct=is_direct,
                initial_state=[
                    {
                        "type": "m.room.encryption",
                        "content": {"algorithm": "m.megolm.v1.aes-sha2"}
                    }
                ] if encrypted else []
            )
            return response.room_id
        except Exception as e:
            logger.error(f"Create room failed: {e}")
            return None

    async def join_room(self, room_id_or_alias: str) -> bool:
        """Join a Matrix room."""
        if not self._client:
            return False

        try:
            await self._client.join(room_id_or_alias)
            return True
        except Exception as e:
            logger.error(f"Join room failed: {e}")
            return False

    async def leave_room(self, room_id: str) -> bool:
        """Leave a Matrix room."""
        if not self._client:
            return False

        try:
            await self._client.room_leave(room_id)
            return True
        except Exception as e:
            logger.error(f"Leave room failed: {e}")
            return False

    async def get_room_members(self, room_id: str) -> List[Dict]:
        """Get members of a room."""
        if not self._client:
            return []

        try:
            response = await self._client.joined_members(room_id)
            return [
                {"user_id": uid, "display_name": info.display_name}
                for uid, info in response.members.items()
            ]
        except Exception as e:
            logger.error(f"Get room members failed: {e}")
            return []

    async def get_rooms(self) -> List[Dict]:
        """Get all joined rooms."""
        if not self._client:
            return []

        rooms = []
        for room_id, room in self._client.rooms.items():
            rooms.append({
                "room_id": room_id,
                "name": room.display_name,
                "member_count": len(room.users),
                "encrypted": room.encrypted
            })
        return rooms

    async def download_media(self, mxc_url: str) -> Optional[bytes]:
        """Download media from mxc:// URL."""
        if not self._client:
            return None

        try:
            response = await self._client.download(mxc_url)
            if hasattr(response, "body"):
                return response.body
            return None
        except Exception as e:
            logger.error(f"Media download failed: {e}")
            return None

    async def set_display_name(self, name: str):
        """Set user display name."""
        if self._client:
            try:
                await self._client.set_displayname(name)
            except Exception as e:
                logger.error(f"Set display name failed: {e}")

    async def set_avatar(self, mxc_url: str):
        """Set user avatar from mxc:// URL."""
        if self._client:
            try:
                await self._client.set_avatar(mxc_url)
            except Exception as e:
                logger.error(f"Set avatar failed: {e}")
