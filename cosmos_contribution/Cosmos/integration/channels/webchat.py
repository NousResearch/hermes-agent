"""
Cosmos WebChat Channel Adapter
===================================

Built-in web chat interface for the Cosmos swarm.

Features:
- WebSocket real-time communication
- Session management
- File uploads
- Typing indicators
- Message history
- Multiple concurrent users
- Integration with ChannelHub

This is the default channel for the web interface.

"Web chat: No app required. Just intelligence." - The Collective
"""

import os
import asyncio
import uuid
from typing import Optional, Dict, List, Any, Set
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger

from .channel_hub import (
    BaseChannel,
    ChannelConfig,
    ChannelMessage,
    ChannelType,
)


@dataclass
class WebChatSession:
    """Represents a web chat session."""
    session_id: str
    user_id: str
    user_name: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[ChannelMessage] = field(default_factory=list)


class WebChatChannel(BaseChannel):
    """
    Built-in web chat channel.

    Works with FastAPI WebSocket endpoints to provide
    real-time chat functionality.

    Features:
    - Session management
    - Message history per session
    - Typing indicators
    - File/media support
    - Multi-user support
    """

    def __init__(
        self,
        config: ChannelConfig = None,
        max_history: int = 100,
        session_timeout: int = 3600
    ):
        """
        Initialize WebChat channel.

        Args:
            config: Channel configuration
            max_history: Max messages to keep per session
            session_timeout: Session timeout in seconds
        """
        config = config or ChannelConfig(channel_type=ChannelType.WEBCHAT)
        super().__init__(config)

        self.max_history = max_history
        self.session_timeout = session_timeout

        self._sessions: Dict[str, WebChatSession] = {}
        self._websockets: Dict[str, Any] = {}  # session_id -> WebSocket
        self._cleanup_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Initialize WebChat channel."""
        try:
            # Start session cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_sessions())

            self._connected = True
            logger.info("WebChat channel initialized")
            return True

        except Exception as e:
            logger.error(f"WebChat init failed: {e}")
            return False

    async def disconnect(self):
        """Shutdown WebChat channel."""
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Close all WebSocket connections
        for session_id, ws in list(self._websockets.items()):
            try:
                await ws.close()
            except Exception:
                pass

        self._sessions.clear()
        self._websockets.clear()
        self._connected = False
        logger.info("WebChat disconnected")

    async def _cleanup_sessions(self):
        """Periodically clean up expired sessions."""
        try:
            while True:
                await asyncio.sleep(300)  # Check every 5 minutes

                now = datetime.now()
                expired = []

                for session_id, session in self._sessions.items():
                    if (now - session.last_activity).total_seconds() > self.session_timeout:
                        expired.append(session_id)

                for session_id in expired:
                    await self.end_session(session_id)
                    logger.debug(f"Cleaned up expired session: {session_id}")

        except asyncio.CancelledError:
            pass

    def create_session(
        self,
        user_id: str = None,
        user_name: str = None,
        metadata: Dict = None
    ) -> WebChatSession:
        """
        Create a new chat session.

        Args:
            user_id: Optional user identifier
            user_name: Display name for the user
            metadata: Additional session metadata

        Returns:
            New WebChatSession
        """
        session_id = str(uuid.uuid4())
        user_id = user_id or f"guest_{session_id[:8]}"
        user_name = user_name or f"Guest {session_id[:4]}"

        session = WebChatSession(
            session_id=session_id,
            user_id=user_id,
            user_name=user_name,
            metadata=metadata or {}
        )

        self._sessions[session_id] = session
        logger.debug(f"Created WebChat session: {session_id}")

        return session

    def get_session(self, session_id: str) -> Optional[WebChatSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def end_session(self, session_id: str):
        """End and clean up a session."""
        if session_id in self._websockets:
            try:
                await self._websockets[session_id].close()
            except Exception:
                pass
            del self._websockets[session_id]

        if session_id in self._sessions:
            del self._sessions[session_id]

        logger.debug(f"Ended WebChat session: {session_id}")

    def register_websocket(self, session_id: str, websocket: Any):
        """Register a WebSocket connection for a session."""
        self._websockets[session_id] = websocket
        if session_id in self._sessions:
            self._sessions[session_id].last_activity = datetime.now()

    def unregister_websocket(self, session_id: str):
        """Unregister a WebSocket connection."""
        if session_id in self._websockets:
            del self._websockets[session_id]

    async def receive_message(
        self,
        session_id: str,
        text: str,
        media_url: str = None,
        media_type: str = None,
        metadata: Dict = None
    ) -> Optional[ChannelMessage]:
        """
        Process an incoming message from the web interface.

        Args:
            session_id: Session identifier
            text: Message text
            media_url: Optional media URL
            media_type: Type of media if present
            metadata: Additional message metadata

        Returns:
            Processed ChannelMessage or None
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"Message from unknown session: {session_id}")
            return None

        # Update session activity
        session.last_activity = datetime.now()

        # Build message
        message = ChannelMessage(
            message_id=str(uuid.uuid4()),
            channel_type=ChannelType.WEBCHAT,
            channel_id=session_id,
            sender_id=session.user_id,
            sender_name=session.user_name,
            text=text,
            media_url=media_url,
            media_type=media_type,
            timestamp=datetime.now(),
            raw_data=metadata or {}
        )

        # Add to history
        session.history.append(message)
        if len(session.history) > self.max_history:
            session.history = session.history[-self.max_history:]

        # Pass to handler
        await self._handle_inbound(message)

        return message

    async def send_message(
        self,
        session_id: str,
        text: str,
        media_path: Optional[str] = None,
        media_url: Optional[str] = None,
        media_type: Optional[str] = None,
        metadata: Dict = None,
        **kwargs
    ) -> bool:
        """
        Send a message to a web chat session.

        Args:
            session_id: Target session ID
            text: Message text
            media_path: Local media file path
            media_url: URL of media
            media_type: Type of media
            metadata: Additional data to include

        Returns:
            True if sent successfully
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"Cannot send to unknown session: {session_id}")
            return False

        ws = self._websockets.get(session_id)
        if not ws:
            logger.warning(f"No WebSocket for session: {session_id}")
            return False

        try:
            # Build response message
            message_data = {
                "type": "message",
                "message_id": str(uuid.uuid4()),
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "from": "cosmos",
            }

            if media_url:
                message_data["media_url"] = media_url
                message_data["media_type"] = media_type

            if metadata:
                message_data["metadata"] = metadata

            # Send via WebSocket
            await ws.send_json(message_data)

            # Add to history
            response_msg = ChannelMessage(
                message_id=message_data["message_id"],
                channel_type=ChannelType.WEBCHAT,
                channel_id=session_id,
                sender_id="cosmos",
                sender_name="Cosmos",
                text=text,
                media_url=media_url,
                media_type=media_type,
                timestamp=datetime.now(),
                raw_data=metadata or {}
            )
            session.history.append(response_msg)

            return True

        except Exception as e:
            logger.error(f"WebChat send failed: {e}")
            return False

    async def send_typing(self, session_id: str, is_typing: bool = True) -> bool:
        """Send typing indicator."""
        ws = self._websockets.get(session_id)
        if not ws:
            return False

        try:
            await ws.send_json({
                "type": "typing",
                "is_typing": is_typing,
                "timestamp": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Typing indicator failed: {e}")
            return False

    async def send_status(
        self,
        session_id: str,
        status: str,
        details: Dict = None
    ) -> bool:
        """Send status update to client."""
        ws = self._websockets.get(session_id)
        if not ws:
            return False

        try:
            await ws.send_json({
                "type": "status",
                "status": status,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Status send failed: {e}")
            return False

    async def broadcast(
        self,
        text: str,
        exclude_sessions: Set[str] = None
    ) -> int:
        """
        Broadcast a message to all connected sessions.

        Args:
            text: Message to broadcast
            exclude_sessions: Session IDs to exclude

        Returns:
            Number of sessions messaged
        """
        exclude = exclude_sessions or set()
        sent = 0

        for session_id in list(self._websockets.keys()):
            if session_id not in exclude:
                if await self.send_message(session_id, text):
                    sent += 1

        return sent

    def get_history(self, session_id: str, limit: int = None) -> List[Dict]:
        """
        Get message history for a session.

        Args:
            session_id: Session identifier
            limit: Max messages to return

        Returns:
            List of message dicts
        """
        session = self._sessions.get(session_id)
        if not session:
            return []

        history = session.history
        if limit:
            history = history[-limit:]

        return [
            {
                "message_id": msg.message_id,
                "sender_id": msg.sender_id,
                "sender_name": msg.sender_name,
                "text": msg.text,
                "media_url": msg.media_url,
                "media_type": msg.media_type,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            }
            for msg in history
        ]

    def get_active_sessions(self) -> List[Dict]:
        """Get all active sessions."""
        return [
            {
                "session_id": s.session_id,
                "user_id": s.user_id,
                "user_name": s.user_name,
                "created_at": s.created_at.isoformat(),
                "last_activity": s.last_activity.isoformat(),
                "message_count": len(s.history),
                "has_websocket": s.session_id in self._websockets
            }
            for s in self._sessions.values()
        ]

    def get_stats(self) -> Dict:
        """Get channel statistics."""
        return {
            "total_sessions": len(self._sessions),
            "active_websockets": len(self._websockets),
            "total_messages": sum(len(s.history) for s in self._sessions.values()),
            "connected": self._connected
        }
