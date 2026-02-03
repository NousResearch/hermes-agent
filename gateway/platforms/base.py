"""
Base platform adapter interface.

All platform adapters (Telegram, Discord, WhatsApp) inherit from this
and implement the required methods.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource


class MessageType(Enum):
    """Types of incoming messages."""
    TEXT = "text"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE = "voice"
    DOCUMENT = "document"
    STICKER = "sticker"
    COMMAND = "command"  # /command style


@dataclass
class MessageEvent:
    """
    Incoming message from a platform.
    
    Normalized representation that all adapters produce.
    """
    # Message content
    text: str
    message_type: MessageType = MessageType.TEXT
    
    # Source information
    source: SessionSource = None
    
    # Original platform data
    raw_message: Any = None
    message_id: Optional[str] = None
    
    # Media attachments
    media_urls: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    
    # Reply context
    reply_to_message_id: Optional[str] = None
    
    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_command(self) -> bool:
        """Check if this is a command message (e.g., /new, /reset)."""
        return self.text.startswith("/")
    
    def get_command(self) -> Optional[str]:
        """Extract command name if this is a command message."""
        if not self.is_command():
            return None
        # Split on space and get first word, strip the /
        parts = self.text.split(maxsplit=1)
        return parts[0][1:].lower() if parts else None
    
    def get_command_args(self) -> str:
        """Get the arguments after a command."""
        if not self.is_command():
            return self.text
        parts = self.text.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""


@dataclass 
class SendResult:
    """Result of sending a message."""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    raw_response: Any = None


# Type for message handlers
MessageHandler = Callable[[MessageEvent], Awaitable[Optional[str]]]


class BasePlatformAdapter(ABC):
    """
    Base class for platform adapters.
    
    Subclasses implement platform-specific logic for:
    - Connecting and authenticating
    - Receiving messages
    - Sending messages/responses
    - Handling media
    """
    
    def __init__(self, config: PlatformConfig, platform: Platform):
        self.config = config
        self.platform = platform
        self._message_handler: Optional[MessageHandler] = None
        self._running = False
    
    @property
    def name(self) -> str:
        """Human-readable name for this adapter."""
        return self.platform.value.title()
    
    @property
    def is_connected(self) -> bool:
        """Check if adapter is currently connected."""
        return self._running
    
    def set_message_handler(self, handler: MessageHandler) -> None:
        """
        Set the handler for incoming messages.
        
        The handler receives a MessageEvent and should return
        an optional response string.
        """
        self._message_handler = handler
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the platform and start receiving messages.
        
        Returns True if connection was successful.
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the platform."""
        pass
    
    @abstractmethod
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """
        Send a message to a chat.
        
        Args:
            chat_id: The chat/channel ID to send to
            content: Message content (may be markdown)
            reply_to: Optional message ID to reply to
            metadata: Additional platform-specific options
        
        Returns:
            SendResult with success status and message ID
        """
        pass
    
    async def send_typing(self, chat_id: str) -> None:
        """
        Send a typing indicator.
        
        Override in subclasses if the platform supports it.
        """
        pass
    
    async def _keep_typing(self, chat_id: str, interval: float = 2.0) -> None:
        """
        Continuously send typing indicator until cancelled.
        
        Telegram/Discord typing status expires after ~5 seconds, so we refresh every 2
        to recover quickly after progress messages interrupt it.
        """
        try:
            while True:
                await self.send_typing(chat_id)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass  # Normal cancellation when handler completes
    
    async def handle_message(self, event: MessageEvent) -> None:
        """
        Process an incoming message.
        
        Calls the registered message handler and sends the response.
        Keeps typing indicator active throughout processing.
        """
        if not self._message_handler:
            return
        
        # Start continuous typing indicator (refreshes every 2 seconds)
        typing_task = asyncio.create_task(self._keep_typing(event.source.chat_id))
        
        try:
            # Call the handler (this can take a while with tool calls)
            response = await self._message_handler(event)
            
            # Send response if any
            if response:
                result = await self.send(
                    chat_id=event.source.chat_id,
                    content=response,
                    reply_to=event.message_id
                )
                
                # Log send failures (don't raise - user already saw tool progress)
                if not result.success:
                    print(f"[{self.name}] Failed to send response: {result.error}")
                    # Try sending without markdown as fallback
                    fallback_result = await self.send(
                        chat_id=event.source.chat_id,
                        content=f"(Response formatting failed, plain text:)\n\n{response[:3500]}",
                        reply_to=event.message_id
                    )
                    if not fallback_result.success:
                        print(f"[{self.name}] Fallback send also failed: {fallback_result.error}")
        except Exception as e:
            print(f"[{self.name}] Error handling message: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Stop typing indicator
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
    
    def build_source(
        self,
        chat_id: str,
        chat_name: Optional[str] = None,
        chat_type: str = "dm",
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> SessionSource:
        """Helper to build a SessionSource for this platform."""
        return SessionSource(
            platform=self.platform,
            chat_id=str(chat_id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(user_id) if user_id else None,
            user_name=user_name,
            thread_id=str(thread_id) if thread_id else None,
        )
    
    @abstractmethod
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """
        Get information about a chat/channel.
        
        Returns dict with at least:
        - name: Chat name
        - type: "dm", "group", "channel"
        """
        pass
    
    def format_message(self, content: str) -> str:
        """
        Format a message for this platform.
        
        Override in subclasses to handle platform-specific formatting
        (e.g., Telegram MarkdownV2, Discord markdown).
        
        Default implementation returns content as-is.
        """
        return content
    
    def truncate_message(self, content: str, max_length: int = 4096) -> List[str]:
        """
        Split a long message into chunks.
        
        Args:
            content: The full message content
            max_length: Maximum length per chunk (platform-specific)
        
        Returns:
            List of message chunks
        """
        if len(content) <= max_length:
            return [content]
        
        chunks = []
        while content:
            if len(content) <= max_length:
                chunks.append(content)
                break
            
            # Try to split at a newline
            split_idx = content.rfind("\n", 0, max_length)
            if split_idx == -1:
                # No newline, split at space
                split_idx = content.rfind(" ", 0, max_length)
            if split_idx == -1:
                # No space either, hard split
                split_idx = max_length
            
            chunks.append(content[:split_idx])
            content = content[split_idx:].lstrip()
        
        return chunks
