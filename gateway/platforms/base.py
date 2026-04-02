"""
Base platform adapter interface.

All platform adapters (Telegram, Discord, WhatsApp) inherit from this
and implement the required methods.
"""

import asyncio
import logging
import os
import random
import re
import uuid
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Awaitable, Tuple
from enum import Enum

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource, build_session_key
from hermes_cli.config import get_hermes_home
from hermes_constants import get_hermes_dir


GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE = (
    "Secure secret entry is not supported over messaging. "
    "Load this skill in the local CLI to be prompted, or add the key to ~/.hermes/.env manually."
)


# ---------------------------------------------------------------------------
# Image cache utilities
#
# When users send images on messaging platforms, we download them to a local
# cache directory so they can be analyzed by the vision tool (which accepts
# local file paths). This avoids issues with ephemeral platform URLs
# (e.g. Telegram file URLs expire after ~1 hour).
# ---------------------------------------------------------------------------

# Default location: {HERMES_HOME}/cache/images/ (legacy: image_cache/)
IMAGE_CACHE_DIR = get_hermes_dir("cache/images", "image_cache")


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


async def cache_image_from_url(url: str, ext: str = ".jpg", retries: int = 2) -> str:
    """
    Download an image from a URL and save it to the local cache.

    Retries on transient failures (timeouts, 429, 5xx) with exponential
    backoff so a single slow CDN response doesn't lose the media.

    Args:
        url: The HTTP/HTTPS URL to download from.
        ext: File extension including the dot (e.g. ".jpg", ".png").
        retries: Number of retry attempts on transient failures.

    Returns:
        Absolute path to the cached image file as a string.
    """
    import asyncio
    import httpx
    import logging as _logging
    _log = _logging.getLogger(__name__)

    last_exc = None
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        for attempt in range(retries + 1):
            try:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)",
                        "Accept": "image/*,*/*;q=0.8",
                    },
                )
                response.raise_for_status()
                return cache_image_from_bytes(response.content, ext)
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 429:
                    raise
                if attempt < retries:
                    wait = 1.5 * (attempt + 1)
                    _log.debug("Media cache retry %d/%d for %s (%.1fs): %s",
                               attempt + 1, retries, url[:80], wait, exc)
                    await asyncio.sleep(wait)
                    continue
                raise
    raise last_exc


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

AUDIO_CACHE_DIR = get_hermes_dir("cache/audio", "audio_cache")


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


async def cache_audio_from_url(url: str, ext: str = ".ogg", retries: int = 2) -> str:
    """
    Download an audio file from a URL and save it to the local cache.

    Retries on transient failures (timeouts, 429, 5xx) with exponential
    backoff so a single slow CDN response doesn't lose the media.

    Args:
        url: The HTTP/HTTPS URL to download from.
        ext: File extension including the dot (e.g. ".ogg", ".mp3").
        retries: Number of retry attempts on transient failures.

    Returns:
        Absolute path to the cached audio file as a string.
    """
    import asyncio
    import httpx
    import logging as _logging
    _log = _logging.getLogger(__name__)

    last_exc = None
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        for attempt in range(retries + 1):
            try:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)",
                        "Accept": "audio/*,*/*;q=0.8",
                    },
                )
                response.raise_for_status()
                return cache_audio_from_bytes(response.content, ext)
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 429:
                    raise
                if attempt < retries:
                    wait = 1.5 * (attempt + 1)
                    _log.debug("Audio cache retry %d/%d for %s (%.1fs): %s",
                               attempt + 1, retries, url[:80], wait, exc)
                    await asyncio.sleep(wait)
                    continue
                raise
    raise last_exc


# ---------------------------------------------------------------------------
# Document cache utilities
#
# Same pattern as image/audio cache -- documents from platforms are downloaded
# here so the agent can reference them by local file path.
# ---------------------------------------------------------------------------

DOCUMENT_CACHE_DIR = get_hermes_dir("cache/documents", "document_cache")

SUPPORTED_DOCUMENT_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def get_document_cache_dir() -> Path:
    """Return the document cache directory, creating it if it doesn't exist."""
    DOCUMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DOCUMENT_CACHE_DIR


def cache_document_from_bytes(data: bytes, filename: str) -> str:
    """
    Save raw document bytes to the cache and return the absolute file path.

    The cached filename preserves the original human-readable name with a
    unique prefix: ``doc_{uuid12}_{original_filename}``.

    Args:
        data: Raw document bytes.
        filename: Original filename (e.g. "report.pdf").

    Returns:
        Absolute path to the cached document file as a string.

    Raises:
        ValueError: If the sanitized path escapes the cache directory.
    """
    cache_dir = get_document_cache_dir()
    # Sanitize: strip directory components, null bytes, and control characters
    safe_name = Path(filename).name if filename else "document"
    safe_name = safe_name.replace("\x00", "").strip()
    if not safe_name or safe_name in (".", ".."):
        safe_name = "document"
    cached_name = f"doc_{uuid.uuid4().hex[:12]}_{safe_name}"
    filepath = cache_dir / cached_name
    # Final safety check: ensure path stays inside cache dir
    if not filepath.resolve().is_relative_to(cache_dir.resolve()):
        raise ValueError(f"Path traversal rejected: {filename!r}")
    filepath.write_bytes(data)
    return str(filepath)


def cleanup_document_cache(max_age_hours: int = 24) -> int:
    """
    Delete cached documents older than *max_age_hours*.

    Returns the number of files removed.
    """
    import time

    cache_dir = get_document_cache_dir()
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


class MessageType(Enum):
    """Types of incoming messages."""
    TEXT = "text"
    LOCATION = "location"
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
    # media_urls: local file paths (for vision tool access)
    media_urls: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    
    # Reply context
    reply_to_message_id: Optional[str] = None
    reply_to_text: Optional[str] = None  # Text of the replied-to message (for context injection)
    
    # Auto-loaded skill for topic/channel bindings (e.g., Telegram DM Topics)
    auto_skill: Optional[str] = None
    
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
        raw = parts[0][1:].lower() if parts else None
        if raw and "@" in raw:
            raw = raw.split("@", 1)[0]
        return raw
    
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
    retryable: bool = False  # True for transient errors (network, timeout) — base will retry automatically


# Error substrings that indicate a transient network failure worth retrying
_RETRYABLE_ERROR_PATTERNS = (
    "connecterror",
    "connectionerror",
    "connectionreset",
    "connectionrefused",
    "timeout",
    "timed out",
    "network",
    "broken pipe",
    "remotedisconnected",
    "eoferror",
    "readtimeout",
    "writetimeout",
)


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
        self._fatal_error_code: Optional[str] = None
        self._fatal_error_message: Optional[str] = None
        self._fatal_error_retryable = True
        self._fatal_error_handler: Optional[Callable[["BasePlatformAdapter"], Awaitable[None] | None]] = None
        
        # Track active message handlers per session for interrupt support
        # Key: session_key (e.g., chat_id), Value: (event, asyncio.Event for interrupt)
        self._active_sessions: Dict[str, asyncio.Event] = {}
        self._pending_messages: Dict[str, MessageEvent] = {}
        # Background message-processing tasks spawned by handle_message().
        # Gateway shutdown cancels these so an old gateway instance doesn't keep
        # working on a task after --replace or manual restarts.
        self._background_tasks: set[asyncio.Task] = set()
        # Chats where auto-TTS on voice input is disabled (set by /voice off)
        self._auto_tts_disabled_chats: set = set()

    @property
    def has_fatal_error(self) -> bool:
        return self._fatal_error_message is not None

    @property
    def fatal_error_message(self) -> Optional[str]:
        return self._fatal_error_message

    @property
    def fatal_error_code(self) -> Optional[str]:
        return self._fatal_error_code

    @property
    def fatal_error_retryable(self) -> bool:
        return self._fatal_error_retryable

    def set_fatal_error_handler(self, handler: Callable[["BasePlatformAdapter"], Awaitable[None] | None]) -> None:
        self._fatal_error_handler = handler

    def _mark_connected(self) -> None:
        self._running = True
        self._fatal_error_code = None
        self._fatal_error_message = None
        self._fatal_error_retryable = True
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(platform=self.platform.value, platform_state="connected", error_code=None, error_message=None)
        except Exception:
            pass

    def _mark_disconnected(self) -> None:
        self._running = False
        if self.has_fatal_error:
            return
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(platform=self.platform.value, platform_state="disconnected", error_code=None, error_message=None)
        except Exception:
            pass

    def _set_fatal_error(self, code: str, message: str, *, retryable: bool) -> None:
        self._running = False
        self._fatal_error_code = code
        self._fatal_error_message = message
        self._fatal_error_retryable = retryable
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(
                platform=self.platform.value,
                platform_state="fatal",
                error_code=code,
                error_message=message,
            )
        except Exception:
            pass

    async def _notify_fatal_error(self) -> None:
        handler = self._fatal_error_handler
        if not handler:
            return
        result = handler(self)
        if asyncio.iscoroutine(result):
            await result
    
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

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
    ) -> SendResult:
        """
        Edit a previously sent message. Optional — platforms that don't
        support editing return success=False and callers fall back to
        sending a new message.
        """
        return SendResult(success=False, error="Not supported")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """
        Send a typing indicator.
        
        Override in subclasses if the platform supports it.
        metadata: optional dict with platform-specific context (e.g. thread_id for Slack).
        """
        pass

    async def stop_typing(self, chat_id: str) -> None:
        """Stop a persistent typing indicator (if the platform uses one).

        Override in subclasses that start background typing loops.
        Default is a no-op for platforms with one-shot typing indicators.
        """
        pass
    
    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
    
    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Send an animated GIF natively via the platform API.
        
        Override in subclasses to send GIFs as proper animations
        (e.g., Telegram send_animation) so they auto-play inline.
        Default falls back to send_image.
        """
        return await self.send_image(chat_id=chat_id, image_url=animation_url, caption=caption, reply_to=reply_to, metadata=metadata)
    
    @staticmethod
    def _is_animation_url(url: str) -> bool:
        """Check if a URL points to an animated GIF (vs a static image)."""
        lower = url.lower().split('?')[0]  # Strip query params
        return lower.endswith('.gif')

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
        
        # Remove only the matched image tags from content (not all markdown images)
        if images:
            extracted_urls = {url for url, _ in images}
            def _remove_if_extracted(match):
                url = match.group(2) if match.lastindex >= 2 else match.group(1)
                return '' if url in extracted_urls else match.group(0)
            cleaned = re.sub(md_pattern, _remove_if_extracted, cleaned)
            cleaned = re.sub(html_pattern, _remove_if_extracted, cleaned)
            # Clean up leftover blank lines
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        
        return images, cleaned
    
    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
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

    async def play_tts(
        self,
        chat_id: str,
        audio_path: str,
        **kwargs,
    ) -> SendResult:
        """
        Play auto-TTS audio for voice replies.

        Override in subclasses for invisible playback (e.g. Web UI).
        Default falls back to send_voice (shows audio player).
        """
        return await self.send_voice(chat_id=chat_id, audio_path=audio_path, **kwargs)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """
        Send a video natively via the platform API.

        Override in subclasses to send videos as inline playable media.
        Default falls back to sending the file path as text.
        """
        text = f"🎬 Video: {video_path}"
        if caption:
            text = f"{caption}\n{text}"
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """
        Send a document/file natively via the platform API.

        Override in subclasses to send files as downloadable attachments.
        Default falls back to sending the file path as text.
        """
        text = f"📎 File: {file_path}"
        if caption:
            text = f"{caption}\n{text}"
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """
        Send a local image file natively via the platform API.

        Unlike send_image() which takes a URL, this takes a local file path.
        Override in subclasses for native photo attachments.
        Default falls back to sending the file path as text.
        """
        text = f"🖼️ Image: {image_path}"
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
        
        # Extract MEDIA:<path> tags, allowing optional whitespace after the colon
        # and quoted/backticked paths for LLM-formatted outputs.
        media_pattern = re.compile(
            r'''[`"']?MEDIA:\s*(?P<path>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|(?:~/|/)\S+(?:[^\S\n]+\S+)*?\.(?:png|jpe?g|gif|webp|mp4|mov|avi|mkv|webm|ogg|opus|mp3|wav|m4a)(?=[\s`"',;:)\]}]|$)|\S+)[`"']?'''
        )
        for match in media_pattern.finditer(content):
            path = match.group("path").strip()
            if len(path) >= 2 and path[0] == path[-1] and path[0] in "`\"'":
                path = path[1:-1].strip()
            path = path.lstrip("`\"'").rstrip("`\"',.;:)}]")
            if path:
                media.append((path, has_voice_tag))

        # Remove MEDIA tags from content (including surrounding quote/backtick wrappers)
        if media:
            cleaned = media_pattern.sub('', cleaned)
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        
        return media, cleaned

    @staticmethod
    def extract_local_files(content: str) -> Tuple[List[str], str]:
        """
        Detect bare local file paths in response text for native media delivery.

        Matches absolute paths (/...) and tilde paths (~/) ending in common
        image or video extensions.  Validates each candidate with
        ``os.path.isfile()`` to avoid false positives from URLs or
        non-existent paths.

        Paths inside fenced code blocks (``` ... ```) and inline code
        (`...`) are ignored so that code samples are never mutilated.

        Returns:
            Tuple of (list of expanded file paths, cleaned text with the
            raw path strings removed).
        """
        _LOCAL_MEDIA_EXTS = (
            '.png', '.jpg', '.jpeg', '.gif', '.webp',
            '.mp4', '.mov', '.avi', '.mkv', '.webm',
        )
        ext_part = '|'.join(e.lstrip('.') for e in _LOCAL_MEDIA_EXTS)

        # (?<![/:\w.]) prevents matching inside URLs (e.g. https://…/img.png)
        #             and relative paths (./foo.png)
        # (?:~/|/)    anchors to absolute or home-relative paths
        path_re = re.compile(
            r'(?<![/:\w.])(?:~/|/)(?:[\w.\-]+/)*[\w.\-]+\.(?:' + ext_part + r')\b',
            re.IGNORECASE,
        )

        # Build spans covered by fenced code blocks and inline code
        code_spans: list = []
        for m in re.finditer(r'```[^\n]*\n.*?```', content, re.DOTALL):
            code_spans.append((m.start(), m.end()))
        for m in re.finditer(r'`[^`\n]+`', content):
            code_spans.append((m.start(), m.end()))

        def _in_code(pos: int) -> bool:
            return any(s <= pos < e for s, e in code_spans)

        found: list = []  # (raw_match_text, expanded_path)
        for match in path_re.finditer(content):
            if _in_code(match.start()):
                continue
            raw = match.group(0)
            expanded = os.path.expanduser(raw)
            if os.path.isfile(expanded):
                found.append((raw, expanded))

        # Deduplicate by expanded path, preserving discovery order
        seen: set = set()
        unique: list = []
        for raw, expanded in found:
            if expanded not in seen:
                seen.add(expanded)
                unique.append((raw, expanded))

        paths = [expanded for _, expanded in unique]

        cleaned = content
        if unique:
            for raw, _exp in unique:
                cleaned = cleaned.replace(raw, '')
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        return paths, cleaned

    async def _keep_typing(self, chat_id: str, interval: float = 2.0, metadata=None) -> None:
        """
        Continuously send typing indicator until cancelled.
        
        Telegram/Discord typing status expires after ~5 seconds, so we refresh every 2
        to recover quickly after progress messages interrupt it.
        """
        try:
            while True:
                await self.send_typing(chat_id, metadata=metadata)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass  # Normal cancellation when handler completes
        finally:
            # Ensure the underlying platform typing loop is stopped.
            # _keep_typing may have called send_typing() after an outer
            # stop_typing() cleared the task dict, recreating the loop.
            # Cancelling _keep_typing alone won't clean that up.
            if hasattr(self, "stop_typing"):
                try:
                    await self.stop_typing(chat_id)
                except Exception:
                    pass
    
    # ── Processing lifecycle hooks ──────────────────────────────────────────
    # Subclasses override these to react to message processing events
    # (e.g. Discord adds 👀/✅/❌ reactions).

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Hook called when background processing begins."""

    async def on_processing_complete(self, event: MessageEvent, success: bool) -> None:
        """Hook called when background processing completes."""

    async def _run_processing_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Run a lifecycle hook without letting failures break message flow."""
        hook = getattr(self, hook_name, None)
        if not callable(hook):
            return
        try:
            await hook(*args, **kwargs)
        except Exception as e:
            logger.warning("[%s] %s hook failed: %s", self.name, hook_name, e)

    @staticmethod
    def _is_retryable_error(error: Optional[str]) -> bool:
        """Return True if the error string looks like a transient network failure."""
        if not error:
            return False
        lowered = error.lower()
        return any(pat in lowered for pat in _RETRYABLE_ERROR_PATTERNS)

    async def _send_with_retry(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Any = None,
        max_retries: int = 2,
        base_delay: float = 2.0,
    ) -> "SendResult":
        """
        Send a message with automatic retry for transient network errors.

        On permanent failures (e.g. formatting / permission errors) falls back
        to a plain-text version before giving up. If all attempts fail due to
        network errors, sends the user a brief delivery-failure notice so they
        know to retry rather than waiting indefinitely.
        """

        result = await self.send(
            chat_id=chat_id,
            content=content,
            reply_to=reply_to,
            metadata=metadata,
        )

        if result.success:
            return result

        error_str = result.error or ""
        is_network = result.retryable or self._is_retryable_error(error_str)

        if is_network:
            # Retry with exponential backoff for transient errors
            for attempt in range(1, max_retries + 1):
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.warning(
                    "[%s] Send failed (attempt %d/%d, retrying in %.1fs): %s",
                    self.name, attempt, max_retries, delay, error_str,
                )
                await asyncio.sleep(delay)
                result = await self.send(
                    chat_id=chat_id,
                    content=content,
                    reply_to=reply_to,
                    metadata=metadata,
                )
                if result.success:
                    logger.info("[%s] Send succeeded on retry %d", self.name, attempt)
                    return result
                error_str = result.error or ""
                if not (result.retryable or self._is_retryable_error(error_str)):
                    break  # error switched to non-transient — fall through to plain-text fallback
            else:
                # All retries exhausted (loop completed without break) — notify user
                logger.error("[%s] Failed to deliver response after %d retries: %s", self.name, max_retries, error_str)
                notice = (
                    "\u26a0\ufe0f Message delivery failed after multiple attempts. "
                    "Please try again \u2014 your request was processed but the response could not be sent."
                )
                try:
                    await self.send(chat_id=chat_id, content=notice, reply_to=reply_to, metadata=metadata)
                except Exception as notify_err:
                    logger.debug("[%s] Could not send delivery-failure notice: %s", self.name, notify_err)
                return result

        # Non-network / post-retry formatting failure: try plain text as fallback
        logger.warning("[%s] Send failed: %s — trying plain-text fallback", self.name, error_str)
        fallback_result = await self.send(
            chat_id=chat_id,
            content=f"(Response formatting failed, plain text:)\n\n{content[:3500]}",
            reply_to=reply_to,
            metadata=metadata,
        )
        if not fallback_result.success:
            logger.error("[%s] Fallback send also failed: %s", self.name, fallback_result.error)
        return fallback_result

    async def handle_message(self, event: MessageEvent) -> None:
        """
        Process an incoming message.
        
        This method returns quickly by spawning background tasks.
        This allows new messages to be processed even while an agent is running,
        enabling interruption support.
        """
        if not self._message_handler:
            return
        
        session_key = build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
        )
        
        # Check if there's already an active handler for this session
        if session_key in self._active_sessions:
            # Special case: photo bursts/albums frequently arrive as multiple near-
            # simultaneous messages. Queue them without interrupting the active run,
            # then process them immediately after the current task finishes.
            if event.message_type == MessageType.PHOTO:
                logger.debug("[%s] Queuing photo follow-up for session %s without interrupt", self.name, session_key)
                existing = self._pending_messages.get(session_key)
                if existing and existing.message_type == MessageType.PHOTO:
                    existing.media_urls.extend(event.media_urls)
                    existing.media_types.extend(event.media_types)
                    if event.text:
                        if not existing.text:
                            existing.text = event.text
                        elif event.text not in existing.text:
                            existing.text = f"{existing.text}\n\n{event.text}".strip()
                else:
                    self._pending_messages[session_key] = event
                return  # Don't interrupt now - will run after current task completes

            # Default behavior for non-photo follow-ups: interrupt the running agent
            logger.debug("[%s] New message while session %s is active — triggering interrupt", self.name, session_key)
            self._pending_messages[session_key] = event
            # Signal the interrupt (the processing task checks this)
            self._active_sessions[session_key].set()
            return  # Don't process now - will be handled after current task finishes
        
        # Mark session as active BEFORE spawning background task to close
        # the race window where a second message arriving before the task
        # starts would also pass the _active_sessions check and spawn a
        # duplicate task.  (grammY sequentialize / aiogram EventIsolation
        # pattern — set the guard synchronously, not inside the task.)
        self._active_sessions[session_key] = asyncio.Event()

        # Spawn background task to process this message
        task = asyncio.create_task(self._process_message_background(event, session_key))
        try:
            self._background_tasks.add(task)
        except TypeError:
            # Some tests stub create_task() with lightweight sentinels that are not
            # hashable and do not support lifecycle callbacks.
            return
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)
    
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
        # Track delivery outcomes for the processing-complete hook
        delivery_attempted = False
        delivery_succeeded = False

        def _record_delivery(result):
            nonlocal delivery_attempted, delivery_succeeded
            if result is None:
                return
            delivery_attempted = True
            if getattr(result, "success", False):
                delivery_succeeded = True

        # Reuse the interrupt event set by handle_message() (which marks
        # the session active before spawning this task to prevent races).
        # Fall back to a new Event only if the entry was removed externally.
        interrupt_event = self._active_sessions.get(session_key) or asyncio.Event()
        self._active_sessions[session_key] = interrupt_event
        
        # Start continuous typing indicator (refreshes every 2 seconds)
        _thread_metadata = {"thread_id": event.source.thread_id} if event.source.thread_id else None
        typing_task = asyncio.create_task(self._keep_typing(event.source.chat_id, metadata=_thread_metadata))
        
        try:
            await self._run_processing_hook("on_processing_start", event)

            # Call the handler (this can take a while with tool calls)
            response = await self._message_handler(event)
            
            # Send response if any.  A None/empty response is normal when
            # streaming already delivered the text (already_sent=True) or
            # when the message was queued behind an active agent.  Log at
            # DEBUG to avoid noisy warnings for expected behavior.
            if not response:
                logger.debug("[%s] Handler returned empty/None response for %s", self.name, event.source.chat_id)
            if response:
                # Extract MEDIA:<path> tags (from TTS tool) before other processing
                media_files, response = self.extract_media(response)
                
                # Extract image URLs and send them as native platform attachments
                images, text_content = self.extract_images(response)
                # Strip any remaining internal directives from message body (fixes #1561)
                text_content = text_content.replace("[[audio_as_voice]]", "").strip()
                text_content = re.sub(r"MEDIA:\s*\S+", "", text_content).strip()
                if images:
                    logger.info("[%s] extract_images found %d image(s) in response (%d chars)", self.name, len(images), len(response))

                # Auto-detect bare local file paths for native media delivery
                # (helps small models that don't use MEDIA: syntax)
                local_files, text_content = self.extract_local_files(text_content)
                if local_files:
                    logger.info("[%s] extract_local_files found %d file(s) in response", self.name, len(local_files))
                
                # Auto-TTS: if voice message, generate audio FIRST (before sending text)
                # Skipped when the chat has voice mode disabled (/voice off)
                _tts_path = None
                if (event.message_type == MessageType.VOICE
                        and text_content
                        and not media_files
                        and event.source.chat_id not in self._auto_tts_disabled_chats):
                    try:
                        from tools.tts_tool import text_to_speech_tool, check_tts_requirements
                        if check_tts_requirements():
                            import json as _json
                            speech_text = re.sub(r'[*_`#\[\]()]', '', text_content)[:4000].strip()
                            if not speech_text:
                                raise ValueError("Empty text after markdown cleanup")
                            tts_result_str = await asyncio.to_thread(
                                text_to_speech_tool, text=speech_text
                            )
                            tts_data = _json.loads(tts_result_str)
                            _tts_path = tts_data.get("file_path")
                    except Exception as tts_err:
                        logger.warning("[%s] Auto-TTS failed: %s", self.name, tts_err)

                # Play TTS audio before text (voice-first experience)
                if _tts_path and Path(_tts_path).exists():
                    try:
                        await self.play_tts(
                            chat_id=event.source.chat_id,
                            audio_path=_tts_path,
                            metadata=_thread_metadata,
                        )
                    finally:
                        try:
                            os.remove(_tts_path)
                        except OSError:
                            pass

                # Send the text portion
                if text_content:
                    logger.info("[%s] Sending response (%d chars) to %s", self.name, len(text_content), event.source.chat_id)
                    result = await self._send_with_retry(
                        chat_id=event.source.chat_id,
                        content=text_content,
                        reply_to=event.message_id,
                        metadata=_thread_metadata,
                    )
                    _record_delivery(result)

                # Human-like pacing delay between text and media
                human_delay = self._get_human_delay()

                # Send extracted images as native attachments
                if images:
                    logger.info("[%s] Extracted %d image(s) to send as attachments", self.name, len(images))
                for image_url, alt_text in images:
                    if human_delay > 0:
                        await asyncio.sleep(human_delay)
                    try:
                        logger.info("[%s] Sending image: %s (alt=%s)", self.name, image_url[:80], alt_text[:30] if alt_text else "")
                        # Route animated GIFs through send_animation for proper playback
                        if self._is_animation_url(image_url):
                            img_result = await self.send_animation(
                                chat_id=event.source.chat_id,
                                animation_url=image_url,
                                caption=alt_text if alt_text else None,
                                metadata=_thread_metadata,
                            )
                        else:
                            img_result = await self.send_image(
                                chat_id=event.source.chat_id,
                                image_url=image_url,
                                caption=alt_text if alt_text else None,
                                metadata=_thread_metadata,
                            )
                        if not img_result.success:
                            logger.error("[%s] Failed to send image: %s", self.name, img_result.error)
                    except Exception as img_err:
                        logger.error("[%s] Error sending image: %s", self.name, img_err, exc_info=True)

                # Send extracted media files — route by file type
                _AUDIO_EXTS = {'.ogg', '.opus', '.mp3', '.wav', '.m4a'}
                _VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'}
                _IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

                for media_path, is_voice in media_files:
                    if human_delay > 0:
                        await asyncio.sleep(human_delay)
                    try:
                        ext = Path(media_path).suffix.lower()
                        if ext in _AUDIO_EXTS:
                            media_result = await self.send_voice(
                                chat_id=event.source.chat_id,
                                audio_path=media_path,
                                metadata=_thread_metadata,
                            )
                        elif ext in _VIDEO_EXTS:
                            media_result = await self.send_video(
                                chat_id=event.source.chat_id,
                                video_path=media_path,
                                metadata=_thread_metadata,
                            )
                        elif ext in _IMAGE_EXTS:
                            media_result = await self.send_image_file(
                                chat_id=event.source.chat_id,
                                image_path=media_path,
                                metadata=_thread_metadata,
                            )
                        else:
                            media_result = await self.send_document(
                                chat_id=event.source.chat_id,
                                file_path=media_path,
                                metadata=_thread_metadata,
                            )

                        if not media_result.success:
                            logger.warning("[%s] Failed to send media (%s): %s", self.name, ext, media_result.error)
                    except Exception as media_err:
                        logger.warning("[%s] Error sending media: %s", self.name, media_err)

                # Send auto-detected local files as native attachments
                for file_path in local_files:
                    if human_delay > 0:
                        await asyncio.sleep(human_delay)
                    try:
                        ext = Path(file_path).suffix.lower()
                        if ext in _IMAGE_EXTS:
                            await self.send_image_file(
                                chat_id=event.source.chat_id,
                                image_path=file_path,
                                metadata=_thread_metadata,
                            )
                        elif ext in _VIDEO_EXTS:
                            await self.send_video(
                                chat_id=event.source.chat_id,
                                video_path=file_path,
                                metadata=_thread_metadata,
                            )
                        else:
                            await self.send_document(
                                chat_id=event.source.chat_id,
                                file_path=file_path,
                                metadata=_thread_metadata,
                            )
                    except Exception as file_err:
                        logger.error("[%s] Error sending local file %s: %s", self.name, file_path, file_err)

            # Determine overall success for the processing hook
            processing_ok = delivery_succeeded if delivery_attempted else not bool(response)
            await self._run_processing_hook("on_processing_complete", event, processing_ok)

            # Check if there's a pending message that was queued during our processing
            if session_key in self._pending_messages:
                pending_event = self._pending_messages.pop(session_key)
                logger.debug("[%s] Processing queued message from interrupt", self.name)
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
                
        except asyncio.CancelledError:
            await self._run_processing_hook("on_processing_complete", event, False)
            raise
        except Exception as e:
            await self._run_processing_hook("on_processing_complete", event, False)
            logger.error("[%s] Error handling message: %s", self.name, e, exc_info=True)
            # Send the error to the user so they aren't left with radio silence
            try:
                error_type = type(e).__name__
                error_detail = str(e)[:300] if str(e) else "no details available"
                _thread_metadata = {"thread_id": event.source.thread_id} if event.source.thread_id else None
                await self.send(
                    chat_id=event.source.chat_id,
                    content=(
                        f"Sorry, I encountered an error ({error_type}).\n"
                        f"{error_detail}\n"
                        "Try again or use /reset to start a fresh session."
                    ),
                    metadata=_thread_metadata,
                )
            except Exception:
                pass  # Last resort — don't let error reporting crash the handler
        finally:
            # Stop typing indicator
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
            # Also cancel any platform-level persistent typing tasks (e.g. Discord)
            # that may have been recreated by _keep_typing after the last stop_typing()
            try:
                if hasattr(self, "stop_typing"):
                    await self.stop_typing(event.source.chat_id)
            except Exception:
                pass
            # Clean up session tracking
            if session_key in self._active_sessions:
                del self._active_sessions[session_key]
    
    async def cancel_background_tasks(self) -> None:
        """Cancel any in-flight background message-processing tasks.

        Used during gateway shutdown/replacement so active sessions from the old
        process do not keep running after adapters are being torn down.
        """
        tasks = [task for task in self._background_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()
        self._pending_messages.clear()
        self._active_sessions.clear()

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
        thread_id: Optional[str] = None,
        chat_topic: Optional[str] = None,
        user_id_alt: Optional[str] = None,
        chat_id_alt: Optional[str] = None,
    ) -> SessionSource:
        """Helper to build a SessionSource for this platform."""
        # Normalize empty topic to None
        if chat_topic is not None and not chat_topic.strip():
            chat_topic = None
        return SessionSource(
            platform=self.platform,
            chat_id=str(chat_id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(user_id) if user_id else None,
            user_name=user_name,
            thread_id=str(thread_id) if thread_id else None,
            chat_topic=chat_topic.strip() if chat_topic else None,
            user_id_alt=user_id_alt,
            chat_id_alt=chat_id_alt,
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
    
    @staticmethod
    def truncate_message(content: str, max_length: int = 4096) -> List[str]:
        """
        Split a long message into chunks, preserving code block boundaries.

        When a split falls inside a triple-backtick code block, the fence is
        closed at the end of the current chunk and reopened (with the original
        language tag) at the start of the next chunk.  Multi-chunk responses
        receive indicators like ``(1/3)``.

        Args:
            content: The full message content
            max_length: Maximum length per chunk (platform-specific)

        Returns:
            List of message chunks
        """
        if len(content) <= max_length:
            return [content]

        INDICATOR_RESERVE = 10   # room for " (XX/XX)"
        FENCE_CLOSE = "\n```"

        chunks: List[str] = []
        remaining = content
        # When the previous chunk ended mid-code-block, this holds the
        # language tag (possibly "") so we can reopen the fence.
        carry_lang: Optional[str] = None

        while remaining:
            # If we're continuing a code block from the previous chunk,
            # prepend a new opening fence with the same language tag.
            prefix = f"```{carry_lang}\n" if carry_lang is not None else ""

            # How much body text we can fit after accounting for the prefix,
            # a potential closing fence, and the chunk indicator.
            headroom = max_length - INDICATOR_RESERVE - len(prefix) - len(FENCE_CLOSE)
            if headroom < 1:
                headroom = max_length // 2

            # Everything remaining fits in one final chunk
            if len(prefix) + len(remaining) <= max_length - INDICATOR_RESERVE:
                chunks.append(prefix + remaining)
                break

            # Find a natural split point (prefer newlines, then spaces)
            region = remaining[:headroom]
            split_at = region.rfind("\n")
            if split_at < headroom // 2:
                split_at = region.rfind(" ")
            if split_at < 1:
                split_at = headroom

            # Avoid splitting inside an inline code span (`...`).
            # If the text before split_at has an odd number of unescaped
            # backticks, the split falls inside inline code — the resulting
            # chunk would have an unpaired backtick and any special characters
            # (like parentheses) inside the broken span would be unescaped,
            # causing MarkdownV2 parse errors on Telegram.
            candidate = remaining[:split_at]
            backtick_count = candidate.count("`") - candidate.count("\\`")
            if backtick_count % 2 == 1:
                # Find the last unescaped backtick and split before it
                last_bt = candidate.rfind("`")
                while last_bt > 0 and candidate[last_bt - 1] == "\\":
                    last_bt = candidate.rfind("`", 0, last_bt)
                if last_bt > 0:
                    # Try to find a space or newline just before the backtick
                    safe_split = candidate.rfind(" ", 0, last_bt)
                    nl_split = candidate.rfind("\n", 0, last_bt)
                    safe_split = max(safe_split, nl_split)
                    if safe_split > headroom // 4:
                        split_at = safe_split

            chunk_body = remaining[:split_at]
            remaining = remaining[split_at:].lstrip()

            full_chunk = prefix + chunk_body

            # Walk only the chunk_body (not the prefix we prepended) to
            # determine whether we end inside an open code block.
            in_code = carry_lang is not None
            lang = carry_lang or ""
            for line in chunk_body.split("\n"):
                stripped = line.strip()
                if stripped.startswith("```"):
                    if in_code:
                        in_code = False
                        lang = ""
                    else:
                        in_code = True
                        tag = stripped[3:].strip()
                        lang = tag.split()[0] if tag else ""

            if in_code:
                # Close the orphaned fence so the chunk is valid on its own
                full_chunk += FENCE_CLOSE
                carry_lang = lang
            else:
                carry_lang = None

            chunks.append(full_chunk)

        # Append chunk indicators when the response spans multiple messages
        if len(chunks) > 1:
            total = len(chunks)
            chunks = [
                f"{chunk} ({i + 1}/{total})" for i, chunk in enumerate(chunks)
            ]

        return chunks
