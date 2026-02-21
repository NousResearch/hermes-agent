"""
Base platform adapter interface.

All platform adapters (Telegram, Discord, WhatsApp) inherit from this
and implement the required methods.
"""

import asyncio
import os
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Awaitable, Tuple
from enum import Enum

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Image cache utilities
#
# When users send images on messaging platforms, we download them to a local
# cache directory so they can be analyzed by the vision tool (which accepts
# local file paths). This avoids issues with ephemeral platform URLs
# (e.g. Telegram file URLs expire after ~1 hour).
# ---------------------------------------------------------------------------

# Default location: ~/.hermes/image_cache/
IMAGE_CACHE_DIR = Path(os.path.expanduser("~/.hermes/image_cache"))


def get_image_cache_dir() -> Path:
    """Return the image cache directory, creating it if it doesn't exist."""
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return IMAGE_CACHE_DIR


def cache_image_from_bytes(data: bytes, ext: str = ".jpg") -> str:
    """
    Save raw image bytes to the cache and return the absolute file path.

    Args:
        data: Raw image bytes.
        ext:  File extension including the dot (e.g. ".jpg", ".png").

    Returns:
        Absolute path to the cached image file as a string.
    """
    cache_dir = get_image_cache_dir()
    filename = f"img_{uuid.uuid4().hex[:12]}{ext}"
    filepath = cache_dir / filename
    filepath.write_bytes(data)
    return str(filepath)


async def cache_image_from_url(url: str, ext: str = ".jpg") -> str:
    """
    Download an image from a URL and save it to the local cache.

    Uses httpx for async download with a reasonable timeout.

    Args:
        url: The HTTP/HTTPS URL to download from.
        ext: File extension including the dot (e.g. ".jpg", ".png").

    Returns:
        Absolute path to the cached image file as a string.
    """
    import httpx

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)",
                "Accept": "image/*,*/*;q=0.8",
            },
        )
        response.raise_for_status()
        return cache_image_from_bytes(response.content, ext)


def cleanup_image_cache(max_age_hours: int = 24) -> int:
    """
    Delete cached images older than *max_age_hours*.

    Returns the number of files removed.
    """
    import time

    cache_dir = get_image_cache_dir()
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    for f in cache_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed


# ---------------------------------------------------------------------------
# Audio cache utilities
#
# Same pattern as image cache -- voice messages from platforms are downloaded
# here so the STT tool (OpenAI Whisper) can transcribe them from local files.
# ---------------------------------------------------------------------------

AUDIO_CACHE_DIR = Path(os.path.expanduser("~/.hermes/audio_cache"))


def get_audio_cache_dir() -> Path:
    """Return the audio cache directory, creating it if it doesn't exist."""
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return AUDIO_CACHE_DIR


def cache_audio_from_bytes(data: bytes, ext: str = ".ogg") -> str:
    """
    Save raw audio bytes to the cache and return the absolute file path.

    Args:
        data: Raw audio bytes.
        ext:  File extension including the dot (e.g. ".ogg", ".mp3").

    Returns:
        Absolute path to the cached audio file as a string.
    """
    cache_dir = get_audio_cache_dir()
    filename = f"audio_{uuid.uuid4().hex[:12]}{ext}"
    filepath = cache_dir / filename
    filepath.write_bytes(data)
    return str(filepath)


async def cache_audio_from_url(url: str, ext: str = ".ogg") -> str:
    """
    Download an audio file from a URL and save it to the local cache.

    Args:
        url: The HTTP/HTTPS URL to download from.
        ext: File extension including the dot (e.g. ".ogg", ".mp3").

    Returns:
        Absolute path to the cached audio file as a string.
    """
    import httpx

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)",
                "Accept": "audio/*,*/*;q=0.8",
            },
        )
        response.raise_for_status()
        return cache_audio_from_bytes(response.content, ext)


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
        
        # Track active message handlers per session for interrupt support
        # Key: session_key (e.g., chat_id), Value: (event, asyncio.Event for interrupt)
        self._active_sessions: Dict[str, asyncio.Event] = {}
        self._pending_messages: Dict[str, MessageEvent] = {}
    
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
    
    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """
        Send an image natively via the platform API.
        
        Override in subclasses to send images as proper attachments
        instead of plain-text URLs. Default falls back to sending the
        URL as a text message.
        """
        # Fallback: send URL as text (subclasses override for native images)
        text = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to)
    
    @staticmethod
    def extract_images(content: str) -> Tuple[List[Tuple[str, str]], str]:
        """
        Extract image URLs from markdown and HTML image tags in a response.
        
        Finds patterns like:
        - ![alt text](https://example.com/image.png)
        - <img src="https://example.com/image.png">
        - <img src="https://example.com/image.png"></img>
        
        Args:
            content: The response text to scan.
        
        Returns:
            Tuple of (list of (url, alt_text) pairs, cleaned content with image tags removed).
        """
        images = []
        cleaned = content
        
        # Match markdown images: ![alt](url)
        md_pattern = r'!\[([^\]]*)\]\((https?://[^\s\)]+)\)'
        for match in re.finditer(md_pattern, content):
            alt_text = match.group(1)
            url = match.group(2)
            # Only extract URLs that look like actual images
            if any(url.lower().endswith(ext) or ext in url.lower() for ext in
                   ['.png', '.jpg', '.jpeg', '.gif', '.webp', 'fal.media', 'fal-cdn', 'replicate.delivery']):
                images.append((url, alt_text))
        
        # Match HTML img tags: <img src="url"> or <img src="url"></img> or <img src="url"/>
        html_pattern = r'<img\s+src=["\']?(https?://[^\s"\'<>]+)["\']?\s*/?>\s*(?:</img>)?'
        for match in re.finditer(html_pattern, content):
            url = match.group(1)
            images.append((url, ""))
        
        # Remove matched image tags from content if we found images
        if images:
            cleaned = re.sub(md_pattern, '', cleaned)
            cleaned = re.sub(html_pattern, '', cleaned)
            # Clean up leftover blank lines
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        
        return images, cleaned
    
    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """
        Send an audio file as a native voice message via the platform API.
        
        Override in subclasses to send audio as voice bubbles (Telegram)
        or file attachments (Discord). Default falls back to sending the
        file path as text.
        """
        text = f"🔊 Audio: {audio_path}"
        if caption:
            text = f"{caption}\n{text}"
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to)
    
    @staticmethod
    def extract_media(content: str) -> Tuple[List[Tuple[str, bool]], str]:
        """
        Extract MEDIA:<path> tags and [[audio_as_voice]] directives from response text.
        
        The TTS tool returns responses like:
            [[audio_as_voice]]
            MEDIA:/path/to/audio.ogg
        
        Args:
            content: The response text to scan.
        
        Returns:
            Tuple of (list of (path, is_voice) pairs, cleaned content with tags removed).
        """
        media = []
        cleaned = content
        
        # Check for [[audio_as_voice]] directive
        has_voice_tag = "[[audio_as_voice]]" in content
        cleaned = cleaned.replace("[[audio_as_voice]]", "")
        
        # Extract MEDIA:<path> tags (path may contain spaces)
        media_pattern = r'MEDIA:(\S+)'
        for match in re.finditer(media_pattern, content):
            path = match.group(1).strip()
            if path:
                media.append((path, has_voice_tag))
        
        # Remove MEDIA tags from content
        if media:
            cleaned = re.sub(media_pattern, '', cleaned)
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        
        return media, cleaned
    
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
        
        This method returns quickly by spawning background tasks.
        This allows new messages to be processed even while an agent is running,
        enabling interruption support.
        """
        if not self._message_handler:
            return
        
        session_key = event.source.chat_id
        
        # Check if there's already an active handler for this session
        if session_key in self._active_sessions:
            # Store this as a pending message - it will interrupt the running agent
            print(f"[{self.name}] ⚡ New message while session {session_key} is active - triggering interrupt")
            self._pending_messages[session_key] = event
            # Signal the interrupt (the processing task checks this)
            self._active_sessions[session_key].set()
            return  # Don't process now - will be handled after current task finishes
        
        # Spawn background task to process this message
        asyncio.create_task(self._process_message_background(event, session_key))
    
    @staticmethod
    def _get_human_delay() -> float:
        """
        Return a random delay in seconds for human-like response pacing.

        Reads from env vars:
          HERMES_HUMAN_DELAY_MODE: "off" (default) | "natural" | "custom"
          HERMES_HUMAN_DELAY_MIN_MS: minimum delay in ms (default 800, custom mode)
          HERMES_HUMAN_DELAY_MAX_MS: maximum delay in ms (default 2500, custom mode)
        """
        import random

        mode = os.getenv("HERMES_HUMAN_DELAY_MODE", "off").lower()
        if mode == "off":
            return 0.0
        min_ms = int(os.getenv("HERMES_HUMAN_DELAY_MIN_MS", "800"))
        max_ms = int(os.getenv("HERMES_HUMAN_DELAY_MAX_MS", "2500"))
        if mode == "natural":
            min_ms, max_ms = 800, 2500
        return random.uniform(min_ms / 1000.0, max_ms / 1000.0)

    async def _process_message_background(self, event: MessageEvent, session_key: str) -> None:
        """Background task that actually processes the message."""
        # Create interrupt event for this session
        interrupt_event = asyncio.Event()
        self._active_sessions[session_key] = interrupt_event
        
        # Start continuous typing indicator (refreshes every 2 seconds)
        typing_task = asyncio.create_task(self._keep_typing(event.source.chat_id))
        
        try:
            # Call the handler (this can take a while with tool calls)
            response = await self._message_handler(event)
            
            # Send response if any
            if response:
                # Extract MEDIA:<path> tags (from TTS tool) before other processing
                media_files, response = self.extract_media(response)
                
                # Extract image URLs and send them as native platform attachments
                images, text_content = self.extract_images(response)
                
                # Send the text portion first (if any remains after extractions)
                if text_content:
                    result = await self.send(
                        chat_id=event.source.chat_id,
                        content=text_content,
                        reply_to=event.message_id
                    )
                    
                    # Log send failures (don't raise - user already saw tool progress)
                    if not result.success:
                        print(f"[{self.name}] Failed to send response: {result.error}")
                        # Try sending without markdown as fallback
                        fallback_result = await self.send(
                            chat_id=event.source.chat_id,
                            content=f"(Response formatting failed, plain text:)\n\n{text_content[:3500]}",
                            reply_to=event.message_id
                        )
                        if not fallback_result.success:
                            print(f"[{self.name}] Fallback send also failed: {fallback_result.error}")
                
                # Human-like pacing delay between text and media
                human_delay = self._get_human_delay()
                
                # Send extracted images as native attachments
                for image_url, alt_text in images:
                    if human_delay > 0:
                        await asyncio.sleep(human_delay)
                    try:
                        img_result = await self.send_image(
                            chat_id=event.source.chat_id,
                            image_url=image_url,
                            caption=alt_text if alt_text else None,
                        )
                        if not img_result.success:
                            print(f"[{self.name}] Failed to send image: {img_result.error}")
                    except Exception as img_err:
                        print(f"[{self.name}] Error sending image: {img_err}")
                
                # Send extracted audio/voice files as native attachments
                for audio_path, is_voice in media_files:
                    if human_delay > 0:
                        await asyncio.sleep(human_delay)
                    try:
                        voice_result = await self.send_voice(
                            chat_id=event.source.chat_id,
                            audio_path=audio_path,
                        )
                        if not voice_result.success:
                            print(f"[{self.name}] Failed to send voice: {voice_result.error}")
                    except Exception as voice_err:
                        print(f"[{self.name}] Error sending voice: {voice_err}")
            
            # Check if there's a pending message that was queued during our processing
            if session_key in self._pending_messages:
                pending_event = self._pending_messages.pop(session_key)
                print(f"[{self.name}] 📨 Processing queued message from interrupt")
                # Clean up current session before processing pending
                if session_key in self._active_sessions:
                    del self._active_sessions[session_key]
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass
                # Process pending message in new background task
                await self._process_message_background(pending_event, session_key)
                return  # Already cleaned up
                
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
            # Clean up session tracking
            if session_key in self._active_sessions:
                del self._active_sessions[session_key]
    
    def has_pending_interrupt(self, session_key: str) -> bool:
        """Check if there's a pending interrupt for a session."""
        return session_key in self._active_sessions and self._active_sessions[session_key].is_set()
    
    def get_pending_message(self, session_key: str) -> Optional[MessageEvent]:
        """Get and clear any pending message for a session."""
        return self._pending_messages.pop(session_key, None)
    
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
