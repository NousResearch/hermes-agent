"""
WeChat (Weixin) platform adapter using the iLink Bot API.

Uses long-polling (getUpdates) for inbound messages and REST API for outbound.
Media is transmitted through the WeChat CDN with AES-128-ECB encryption.

Ported from the TypeScript weixin-agent-sdk (github.com/wong2/weixin-agent-sdk)
to integrate with the Hermes Agent gateway.

Requires:
    pip install httpx cryptography
    WEIXIN_TOKEN env var (from QR login) or config.yaml platforms.weixin.token

Configuration in config.yaml:
    platforms:
      weixin:
        enabled: true
        token: "bot_token_from_qr_login"
        extra:
          base_url: "https://ilinkai.weixin.qq.com"    # default
          cdn_base_url: "https://novac2c.cdn.weixin.qq.com/c2c"  # default
          account_id: "your-account-id"                 # from QR login
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import re
import struct
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding as crypto_padding
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    get_image_cache_dir,
    get_audio_cache_dir,
    get_document_cache_dir,
)
from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"
CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c"
CHANNEL_VERSION = "hermes-0.1.0"

# WeChat message/item type constants (from proto)
WX_MSG_TYPE_USER = 1
WX_MSG_TYPE_BOT = 2
WX_MSG_STATE_NEW = 0
WX_MSG_STATE_GENERATING = 1
WX_MSG_STATE_FINISH = 2
WX_ITEM_TEXT = 1
WX_ITEM_IMAGE = 2
WX_ITEM_VOICE = 3
WX_ITEM_FILE = 4
WX_ITEM_VIDEO = 5

# Upload media type constants
UPLOAD_MEDIA_IMAGE = 1
UPLOAD_MEDIA_VIDEO = 2
UPLOAD_MEDIA_FILE = 3
UPLOAD_MEDIA_VOICE = 4

# Session expired error code
SESSION_EXPIRED_ERRCODE = -14
SESSION_PAUSE_DURATION_S = 3600  # 1 hour

# Polling
DEFAULT_LONG_POLL_TIMEOUT_S = 35
MAX_CONSECUTIVE_FAILURES = 3
BACKOFF_DELAY_S = 30
RETRY_DELAY_S = 2

# Message limits
MAX_MESSAGE_LENGTH = 4096
DEDUP_WINDOW_S = 300
DEDUP_MAX_SIZE = 1000
MEDIA_MAX_BYTES = 100 * 1024 * 1024  # 100 MB


def check_weixin_requirements() -> bool:
    """Check if WeChat adapter dependencies are available."""
    if not HTTPX_AVAILABLE:
        logger.warning("WeChat: httpx not installed. Run: pip install httpx")
        return False
    if not CRYPTO_AVAILABLE:
        logger.warning("WeChat: cryptography not installed. Run: pip install cryptography")
        return False
    return True


# ---------------------------------------------------------------------------
# AES-128-ECB crypto (matches weixin-agent-sdk aes-ecb.ts)
# ---------------------------------------------------------------------------

def _aes_ecb_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt with AES-128-ECB + PKCS7 padding."""
    padder = crypto_padding.PKCS7(128).padder()
    padded = padder.update(plaintext) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.ECB())
    enc = cipher.encryptor()
    return enc.update(padded) + enc.finalize()


def _aes_ecb_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """Decrypt AES-128-ECB with PKCS7 padding."""
    cipher = Cipher(algorithms.AES(key), modes.ECB())
    dec = cipher.decryptor()
    padded = dec.update(ciphertext) + dec.finalize()
    unpadder = crypto_padding.PKCS7(128).unpadder()
    return unpadder.update(padded) + unpadder.finalize()


def _aes_ecb_padded_size(plaintext_size: int) -> int:
    """Compute AES-128-ECB ciphertext size (PKCS7 to 16-byte boundary)."""
    return ((plaintext_size + 1 + 15) // 16) * 16


def _parse_aes_key(aes_key_b64: str) -> bytes:
    """Parse CDNMedia.aes_key into a raw 16-byte key.

    Two encodings exist in the wild:
      - base64(raw 16 bytes)           → images
      - base64(hex string of 16 bytes) → file / voice / video
    """
    decoded = base64.b64decode(aes_key_b64)
    if len(decoded) == 16:
        return decoded
    if len(decoded) == 32:
        try:
            hex_str = decoded.decode("ascii")
            if all(c in "0123456789abcdefABCDEF" for c in hex_str):
                return bytes.fromhex(hex_str)
        except (UnicodeDecodeError, ValueError):
            pass
    raise ValueError(
        f"aes_key must decode to 16 raw bytes or 32-char hex string, got {len(decoded)} bytes"
    )


# ---------------------------------------------------------------------------
# Markdown → plain text (WeChat doesn't render markdown)
# ---------------------------------------------------------------------------

def _markdown_to_plain(text: str) -> str:
    """Strip markdown syntax for WeChat delivery (mirrors SDK markdownToPlainText)."""
    result = text
    # Code blocks: strip fences, keep content
    result = re.sub(r"```[^\n]*\n?([\s\S]*?)```", lambda m: m.group(1).strip(), result)
    # Images: remove entirely
    result = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", result)
    # Links: keep display text
    result = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", result)
    # Table separator rows
    result = re.sub(r"^\|[\s:|\-]+\|$", "", result, flags=re.MULTILINE)
    # Table rows: strip pipes
    def _table_row(m):
        inner = m.group(1)
        return "  ".join(cell.strip() for cell in inner.split("|"))
    result = re.sub(r"^\|(.+)\|$", _table_row, result, flags=re.MULTILINE)
    # Inline formatting
    result = re.sub(r"\*\*(.+?)\*\*", r"\1", result)
    result = re.sub(r"\*(.+?)\*", r"\1", result)
    result = re.sub(r"__(.+?)__", r"\1", result)
    result = re.sub(r"_(.+?)_", r"\1", result)
    result = re.sub(r"~~(.+?)~~", r"\1", result)
    result = re.sub(r"`(.+?)`", r"\1", result)
    return result


# ---------------------------------------------------------------------------
# State persistence helpers
# ---------------------------------------------------------------------------

def _weixin_state_dir() -> Path:
    """Return (and create) the WeChat state directory under HERMES_HOME."""
    d = get_hermes_home() / "weixin"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sync_buf_path(account_id: str) -> Path:
    d = _weixin_state_dir() / "sync"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{account_id}.buf"


def _load_sync_buf(account_id: str) -> str:
    p = _sync_buf_path(account_id)
    try:
        return p.read_text("utf-8").strip() if p.exists() else ""
    except Exception:
        return ""


def _save_sync_buf(account_id: str, buf: str) -> None:
    try:
        _sync_buf_path(account_id).write_text(buf, "utf-8")
    except Exception as e:
        logger.warning("[Weixin] Failed to save sync buf: %s", e)


# ---------------------------------------------------------------------------
# MIME helpers
# ---------------------------------------------------------------------------

_MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska", ".webm": "video/webm",
    ".pdf": "application/pdf", ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".zip": "application/zip", ".txt": "text/plain",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
    ".opus": "audio/opus", ".m4a": "audio/mp4",
}


def _mime_from_path(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    return _MIME_MAP.get(ext, "application/octet-stream")


# ---------------------------------------------------------------------------
# WeChat Adapter
# ---------------------------------------------------------------------------

class WeixinAdapter(BasePlatformAdapter):
    """WeChat (Weixin) chatbot adapter using the iLink Bot long-polling API.

    Message flow:
        1. connect() starts a long-poll loop calling getUpdates
        2. Inbound messages are parsed, media is downloaded/decrypted from CDN
        3. MessageEvent is created and dispatched to self.handle_message()
        4. BasePlatformAdapter routes the response back through send()/send_image()/etc.
        5. Outbound media is encrypted and uploaded to CDN before sending
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WEIXIN)

        extra = config.extra or {}
        self._token: str = config.token or os.getenv("WEIXIN_TOKEN", "")
        self._base_url: str = extra.get("base_url") or os.getenv("WEIXIN_BASE_URL", DEFAULT_BASE_URL)
        self._cdn_base_url: str = extra.get("cdn_base_url") or os.getenv("WEIXIN_CDN_BASE_URL", CDN_BASE_URL)
        self._account_id: str = extra.get("account_id") or os.getenv("WEIXIN_ACCOUNT_ID", "")

        self._http: Optional["httpx.AsyncClient"] = None
        self._poll_task: Optional[asyncio.Task] = None

        # context_token cache: from_user_id → latest context_token
        self._context_tokens: Dict[str, str] = {}
        # typing_ticket cache: from_user_id → (ticket, fetched_at)
        self._typing_tickets: Dict[str, Tuple[str, float]] = {}
        self._typing_ticket_ttl_s: float = 12 * 3600  # 12 hours

        # Message deduplication
        self._seen_messages: Dict[str, float] = {}

        # Session pause (errcode -14)
        self._paused_until: float = 0.0

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self) -> bool:
        """Start the long-poll loop for inbound messages."""
        if not HTTPX_AVAILABLE or not CRYPTO_AVAILABLE:
            logger.error("[Weixin] Missing dependencies (httpx, cryptography)")
            return False

        if not self._token:
            logger.error("[Weixin] No token configured. Run scripts/weixin_login.py or set WEIXIN_TOKEN")
            self._set_fatal_error("no_token", "WeChat token not configured", retryable=False)
            await self._notify_fatal_error()
            return False

        try:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True,
            )
            self._poll_task = asyncio.create_task(self._poll_loop())
            self._mark_connected()
            logger.info("[Weixin] Connected, starting poll loop (account=%s)", self._account_id or "default")
            return True
        except Exception as e:
            logger.error("[Weixin] Failed to connect: %s", e)
            return False

    async def disconnect(self) -> None:
        """Stop polling and clean up."""
        self._running = False
        self._mark_disconnected()

        # Cancel background tasks (typing ticket fetches, etc.)
        await self.cancel_background_tasks()

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._http:
            await self._http.aclose()
            self._http = None

        self._context_tokens.clear()
        self._typing_tickets.clear()
        self._seen_messages.clear()
        logger.info("[Weixin] Disconnected")

    # -- Long-poll loop -----------------------------------------------------

    async def _poll_loop(self) -> None:
        """Long-poll getUpdates in a loop until disconnected."""
        get_updates_buf = _load_sync_buf(self._account_id) if self._account_id else ""
        if get_updates_buf:
            logger.info("[Weixin] Resuming from saved sync buf (%d bytes)", len(get_updates_buf))

        consecutive_failures = 0
        poll_timeout_ms = DEFAULT_LONG_POLL_TIMEOUT_S * 1000

        while self._running:
            try:
                # Session pause check
                if self._paused_until > time.time():
                    remaining = int(self._paused_until - time.time())
                    logger.info("[Weixin] Session paused, %ds remaining", remaining)
                    await asyncio.sleep(min(remaining, 60))
                    continue

                resp = await self._api_fetch("ilink/bot/getupdates", {
                    "get_updates_buf": get_updates_buf,
                }, timeout_s=poll_timeout_ms / 1000 + 5)

                # Server-suggested timeout
                suggested = resp.get("longpolling_timeout_ms")
                if isinstance(suggested, (int, float)) and suggested > 0:
                    poll_timeout_ms = int(suggested)

                # Check for API errors
                ret = resp.get("ret", 0)
                errcode = resp.get("errcode", 0)
                is_error = (ret != 0) or (errcode != 0)

                if is_error:
                    if errcode == SESSION_EXPIRED_ERRCODE or ret == SESSION_EXPIRED_ERRCODE:
                        self._paused_until = time.time() + SESSION_PAUSE_DURATION_S
                        logger.warning(
                            "[Weixin] Session expired (errcode %d), pausing for %d min",
                            SESSION_EXPIRED_ERRCODE, SESSION_PAUSE_DURATION_S // 60,
                        )
                        consecutive_failures = 0
                        continue

                    consecutive_failures += 1
                    logger.warning(
                        "[Weixin] getUpdates error: ret=%s errcode=%s errmsg=%s (%d/%d)",
                        ret, errcode, resp.get("errmsg", ""), consecutive_failures, MAX_CONSECUTIVE_FAILURES,
                    )
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        consecutive_failures = 0
                        await asyncio.sleep(BACKOFF_DELAY_S + random.uniform(0, 5))
                    else:
                        await asyncio.sleep(RETRY_DELAY_S + random.uniform(0, 2))
                    continue

                consecutive_failures = 0

                # Save sync buf for resume
                new_buf = resp.get("get_updates_buf", "")
                if new_buf:
                    get_updates_buf = new_buf
                    if self._account_id:
                        _save_sync_buf(self._account_id, new_buf)

                # Process messages
                msgs = resp.get("msgs") or []
                for msg in msgs:
                    try:
                        await self._on_message(msg)
                    except Exception as e:
                        logger.error("[Weixin] Error processing message: %s", e, exc_info=True)

            except asyncio.CancelledError:
                return
            except Exception as e:
                if not self._running:
                    return
                consecutive_failures += 1
                logger.warning(
                    "[Weixin] Poll loop error (%d/%d): %s",
                    consecutive_failures, MAX_CONSECUTIVE_FAILURES, e,
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    consecutive_failures = 0
                    await asyncio.sleep(BACKOFF_DELAY_S + random.uniform(0, 5))
                else:
                    await asyncio.sleep(RETRY_DELAY_S + random.uniform(0, 2))

    # -- Inbound message processing -----------------------------------------

    async def _on_message(self, msg: Dict[str, Any]) -> None:
        """Process a single inbound WeChat message."""
        from_user = msg.get("from_user_id", "")
        if not from_user:
            return

        # Deduplication
        msg_id = str(msg.get("message_id", ""))
        seq = str(msg.get("seq", ""))
        dedup_key = f"{from_user}:{msg_id}:{seq}"
        if self._is_duplicate(dedup_key):
            return

        # Only process user messages; filter self-messages to prevent loops
        if msg.get("message_type") != WX_MSG_TYPE_USER:
            return
        if self._account_id and from_user == self._account_id:
            logger.debug("[Weixin] Skipping self-message from %s", from_user[:8])
            return

        # Cache context token (MUST be echoed in replies)
        context_token = msg.get("context_token", "")
        if context_token:
            self._context_tokens[from_user] = context_token
            self._persist_context_tokens()

        items = msg.get("item_list") or []
        text = self._extract_text(items)

        # Determine message type and download media
        hermes_msg_type = MessageType.TEXT
        media_urls: List[str] = []
        media_types: List[str] = []

        media_item = self._find_media_item(items)
        if media_item:
            try:
                path, mime, mtype = await self._download_media(media_item)
                if path:
                    media_urls.append(path)
                    media_types.append(mime)
                    hermes_msg_type = mtype
            except Exception as e:
                logger.error("[Weixin] Media download failed: %s", e)

        # Use voice-to-text if available and no other text
        if not text and hermes_msg_type == MessageType.VOICE:
            voice_item = media_item
            if voice_item and voice_item.get("type") == WX_ITEM_VOICE:
                stt = (voice_item.get("voice_item") or {}).get("text", "")
                if stt:
                    text = stt

        if not text and not media_urls:
            return

        # Fetch typing ticket for this user (async, non-blocking)
        task = asyncio.create_task(self._cache_typing_ticket(from_user, context_token))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        source = self.build_source(
            chat_id=from_user,
            chat_type="dm",
            user_id=from_user,
        )

        create_time_ms = msg.get("create_time_ms")
        try:
            timestamp = datetime.fromtimestamp(
                create_time_ms / 1000, tz=timezone.utc
            ) if create_time_ms else datetime.now(tz=timezone.utc)
        except (ValueError, OSError, TypeError):
            timestamp = datetime.now(tz=timezone.utc)

        event = MessageEvent(
            text=text or "",
            message_type=hermes_msg_type,
            source=source,
            message_id=msg_id or seq,
            raw_message=msg,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
        )

        logger.debug("[Weixin] Message from %s: %s (media=%d)", from_user[:8], text[:50] if text else "(media)", len(media_urls))
        await self.handle_message(event)

    @staticmethod
    def _extract_text(items: List[Dict[str, Any]]) -> str:
        """Extract text from item_list, handling quoted messages."""
        for item in items:
            if item.get("type") == WX_ITEM_TEXT:
                text_item = item.get("text_item") or {}
                text = text_item.get("text", "")

                ref = item.get("ref_msg")
                if not ref:
                    return text

                ref_item = ref.get("message_item")
                if ref_item and ref_item.get("type") in (WX_ITEM_IMAGE, WX_ITEM_VIDEO, WX_ITEM_FILE, WX_ITEM_VOICE):
                    return text  # Quoted media — just return the text part

                parts = []
                title = ref.get("title", "")
                if title:
                    parts.append(title)
                if ref_item:
                    ref_text = WeixinAdapter._extract_text([ref_item])
                    if ref_text:
                        parts.append(ref_text)

                if parts:
                    return f'[Quote: {" | ".join(parts)}]\n{text}'
                return text

            # Voice with speech-to-text
            if item.get("type") == WX_ITEM_VOICE:
                voice = item.get("voice_item") or {}
                if voice.get("text"):
                    return voice["text"]

        return ""

    @staticmethod
    def _find_media_item(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the first downloadable media item (priority: image > video > file > voice).

        Voice items with server-side speech-to-text (voice_item.text) are skipped
        — the transcription is used as text instead. This matches the SDK's
        process-message.ts behavior and avoids triggering Hermes's auto-TTS.
        """
        for item_type in (WX_ITEM_IMAGE, WX_ITEM_VIDEO, WX_ITEM_FILE, WX_ITEM_VOICE):
            for item in items:
                if item.get("type") == item_type:
                    # Skip voice items that have STT transcription
                    if item_type == WX_ITEM_VOICE:
                        voice_data = item.get("voice_item") or {}
                        if voice_data.get("text"):
                            continue
                    type_key = {
                        WX_ITEM_IMAGE: "image_item",
                        WX_ITEM_VIDEO: "video_item",
                        WX_ITEM_FILE: "file_item",
                        WX_ITEM_VOICE: "voice_item",
                    }[item_type]
                    media_data = item.get(type_key) or {}
                    media_ref = media_data.get("media") or {}
                    if media_ref.get("encrypt_query_param"):
                        return item

        # Check quoted media in ref_msg
        for item in items:
            if item.get("type") == WX_ITEM_TEXT:
                ref = item.get("ref_msg", {})
                ref_item = ref.get("message_item")
                if ref_item and ref_item.get("type") in (WX_ITEM_IMAGE, WX_ITEM_VIDEO, WX_ITEM_FILE, WX_ITEM_VOICE):
                    type_key = {
                        WX_ITEM_IMAGE: "image_item", WX_ITEM_VIDEO: "video_item",
                        WX_ITEM_FILE: "file_item", WX_ITEM_VOICE: "voice_item",
                    }.get(ref_item["type"])
                    if type_key:
                        media_ref = (ref_item.get(type_key) or {}).get("media") or {}
                        if media_ref.get("encrypt_query_param"):
                            return ref_item

        return None

    async def _download_media(self, item: Dict[str, Any]) -> Tuple[str, str, MessageType]:
        """Download and decrypt media from CDN. Returns (local_path, mime, message_type)."""
        item_type = item.get("type")

        if item_type == WX_ITEM_IMAGE:
            img = item.get("image_item") or {}
            media = img.get("media") or {}
            eqp = media.get("encrypt_query_param", "")
            # Prefer image_item.aeskey (hex) over media.aes_key (base64)
            if img.get("aeskey"):
                aes_key_b64 = base64.b64encode(bytes.fromhex(img["aeskey"])).decode()
            else:
                aes_key_b64 = media.get("aes_key", "")
            if not eqp:
                return ("", "", MessageType.TEXT)

            buf = await self._cdn_download_decrypt(eqp, aes_key_b64) if aes_key_b64 else await self._cdn_download_plain(eqp)
            path = cache_image_from_bytes(buf, ".jpg")
            return (path, "image/jpeg", MessageType.PHOTO)

        elif item_type == WX_ITEM_VOICE:
            voice = item.get("voice_item") or {}
            media = voice.get("media") or {}
            eqp = media.get("encrypt_query_param", "")
            aes_key_b64 = media.get("aes_key", "")
            if not eqp or not aes_key_b64:
                return ("", "", MessageType.TEXT)

            silk_buf = await self._cdn_download_decrypt(eqp, aes_key_b64)
            # Try to convert SILK to WAV (best-effort)
            wav_buf = self._silk_to_wav_simple(silk_buf)
            if wav_buf:
                path = cache_audio_from_bytes(wav_buf, ".wav")
                return (path, "audio/wav", MessageType.VOICE)
            else:
                path = cache_audio_from_bytes(silk_buf, ".silk")
                return (path, "audio/silk", MessageType.VOICE)

        elif item_type == WX_ITEM_FILE:
            file_item = item.get("file_item") or {}
            media = file_item.get("media") or {}
            eqp = media.get("encrypt_query_param", "")
            aes_key_b64 = media.get("aes_key", "")
            filename = file_item.get("file_name", "file.bin")
            if not eqp or not aes_key_b64:
                return ("", "", MessageType.TEXT)

            buf = await self._cdn_download_decrypt(eqp, aes_key_b64)
            path = cache_document_from_bytes(buf, filename)
            return (path, _mime_from_path(filename), MessageType.DOCUMENT)

        elif item_type == WX_ITEM_VIDEO:
            video = item.get("video_item") or {}
            media = video.get("media") or {}
            eqp = media.get("encrypt_query_param", "")
            aes_key_b64 = media.get("aes_key", "")
            if not eqp or not aes_key_b64:
                return ("", "", MessageType.TEXT)

            buf = await self._cdn_download_decrypt(eqp, aes_key_b64)
            # Save to document cache since there's no video cache
            path = cache_document_from_bytes(buf, f"video_{uuid.uuid4().hex[:8]}.mp4")
            return (path, "video/mp4", MessageType.VIDEO)

        return ("", "", MessageType.TEXT)

    @staticmethod
    def _silk_to_wav_simple(silk_buf: bytes) -> Optional[bytes]:
        """Best-effort SILK→WAV conversion.

        Tries two approaches:
        1. silk-decoder CLI (pip install silk-python or system package)
        2. ffmpeg with SILK input format (requires ffmpeg built with libsilk)

        Returns WAV bytes on success, None on failure (caller falls back to
        passing raw SILK or using voice-to-text from the WeChat API).
        """
        import subprocess
        import tempfile

        silk_path = None
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".silk", delete=False) as sf:
                sf.write(silk_buf)
                silk_path = sf.name
            wav_path = silk_path.replace(".silk", ".wav")

            # Try silk-decoder (from silk-python package)
            for cmd in [
                ["silk-decoder", silk_path, wav_path],
                ["ffmpeg", "-y", "-i", silk_path, "-ar", "24000", "-ac", "1", wav_path],
            ]:
                try:
                    result = subprocess.run(cmd, capture_output=True, timeout=10)
                    if result.returncode == 0 and os.path.exists(wav_path):
                        wav_data = Path(wav_path).read_bytes()
                        if len(wav_data) > 44:  # Valid WAV has at least a header
                            return wav_data
                except FileNotFoundError:
                    continue
                except subprocess.TimeoutExpired:
                    continue
        except Exception:
            pass
        finally:
            for p in (silk_path, wav_path):
                if p:
                    try:
                        os.unlink(p)
                    except Exception:
                        pass
        return None

    # -- Deduplication ------------------------------------------------------

    def _is_duplicate(self, key: str) -> bool:
        now = time.time()
        if len(self._seen_messages) > DEDUP_MAX_SIZE:
            cutoff = now - DEDUP_WINDOW_S
            self._seen_messages = {k: v for k, v in self._seen_messages.items() if v > cutoff}
        if key in self._seen_messages:
            return True
        self._seen_messages[key] = now
        return False

    # -- Outbound: text -----------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a plain text message to a WeChat user."""
        context_token = self._context_tokens.get(chat_id)
        if not context_token:
            return SendResult(success=False, error="No context_token for this user (they haven't messaged yet)")

        plain = _markdown_to_plain(content).strip()
        if not plain:
            return SendResult(success=True, message_id="skipped-empty")

        chunks = self.truncate_message(plain, self.MAX_MESSAGE_LENGTH)
        last_id = None

        try:
            for chunk in chunks:
                client_id = f"hermes-{uuid.uuid4().hex[:12]}"
                body = {
                    "msg": {
                        "from_user_id": "",
                        "to_user_id": chat_id,
                        "client_id": client_id,
                        "message_type": WX_MSG_TYPE_BOT,
                        "message_state": WX_MSG_STATE_FINISH,
                        "item_list": [{"type": WX_ITEM_TEXT, "text_item": {"text": chunk}}],
                        "context_token": context_token,
                    },
                }
                resp = await self._api_fetch("ilink/bot/sendmessage", body)
                if resp.get("ret", 0) != 0:
                    raise RuntimeError(f"ret={resp.get('ret')} {resp.get('errmsg', '')}")
                last_id = client_id
            return SendResult(success=True, message_id=last_id)
        except Exception as e:
            logger.error("[Weixin] Send failed: %s", e)
            return SendResult(success=False, error=str(e))

    # -- Outbound: typing ---------------------------------------------------

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send typing indicator via WeChat API."""
        entry = self._typing_tickets.get(chat_id)
        if not entry:
            return
        ticket, _ = entry
        try:
            await self._api_fetch("ilink/bot/sendtyping", {
                "ilink_user_id": chat_id,
                "typing_ticket": ticket,
                "status": 1,  # TYPING
            }, timeout_s=10)
        except Exception:
            pass  # Non-critical

    async def _cache_typing_ticket(self, user_id: str, context_token: str) -> None:
        """Fetch and cache the typing_ticket for a user (with TTL refresh)."""
        entry = self._typing_tickets.get(user_id)
        if entry:
            _, fetched_at = entry
            if time.time() - fetched_at < self._typing_ticket_ttl_s:
                return  # Still fresh
        try:
            resp = await self._api_fetch("ilink/bot/getconfig", {
                "ilink_user_id": user_id,
                "context_token": context_token,
            }, timeout_s=10)
            ticket = resp.get("typing_ticket", "")
            if ticket:
                self._typing_tickets[user_id] = (ticket, time.time())
        except Exception as e:
            logger.debug("[Weixin] Failed to get typing ticket: %s", e)

    # -- Outbound: media ----------------------------------------------------

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image by URL (download, encrypt, upload to CDN, then send)."""
        try:
            local_path = await self._download_remote_file(image_url)
            return await self._send_media_file(chat_id, local_path, caption, UPLOAD_MEDIA_IMAGE, WX_ITEM_IMAGE)
        except Exception as e:
            logger.error("[Weixin] send_image failed: %s", e)
            # Fallback to text
            text = f"{caption}\n{image_url}" if caption else image_url
            return await self.send(chat_id, text)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file."""
        try:
            return await self._send_media_file(chat_id, image_path, caption, UPLOAD_MEDIA_IMAGE, WX_ITEM_IMAGE)
        except Exception as e:
            logger.error("[Weixin] send_image_file failed: %s", e)
            return SendResult(success=False, error=str(e))

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local video file."""
        try:
            return await self._send_media_file(chat_id, video_path, caption, UPLOAD_MEDIA_VIDEO, WX_ITEM_VIDEO)
        except Exception as e:
            logger.error("[Weixin] send_video failed: %s", e)
            return SendResult(success=False, error=str(e))

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a file attachment."""
        try:
            return await self._send_media_file(
                chat_id, file_path, caption, UPLOAD_MEDIA_FILE, WX_ITEM_FILE,
                file_name=file_name or Path(file_path).name,
            )
        except Exception as e:
            logger.error("[Weixin] send_document failed: %s", e)
            return SendResult(success=False, error=str(e))

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send audio as a file attachment.

        Native voice bubbles (WX_ITEM_VOICE) are not supported by the
        iLink Bot API for outbound messages — even Tencent's own SDK
        sends audio as FILE attachments.

        For large audio files, attempts ffmpeg compression first to
        avoid CDN timeout on high-latency connections.
        """
        try:
            send_path = audio_path
            compressed_path = None
            # Compress large audio files to avoid CDN timeout
            if Path(audio_path).stat().st_size > 100_000:
                compressed_path = self._compress_audio(audio_path)
                if compressed_path:
                    send_path = compressed_path
            try:
                return await self._send_media_file(
                    chat_id, send_path, caption,
                    UPLOAD_MEDIA_FILE, WX_ITEM_FILE,
                    file_name=Path(audio_path).name,
                )
            finally:
                if compressed_path:
                    try:
                        os.unlink(compressed_path)
                    except OSError:
                        pass
        except Exception as e:
            logger.error("[Weixin] send_voice failed: %s", e)
            return SendResult(success=False, error=str(e))

    @staticmethod
    def _compress_audio(audio_path: str) -> Optional[str]:
        """Compress audio with ffmpeg for CDN upload. Returns temp path or None."""
        import subprocess
        import tempfile
        try:
            fd, out_path = tempfile.mkstemp(suffix=Path(audio_path).suffix)
            os.close(fd)
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1",
                 "-b:a", "64k", out_path],
                capture_output=True, timeout=15,
            )
            if result.returncode == 0 and Path(out_path).stat().st_size > 0:
                return out_path
            os.unlink(out_path)
            return None
        except Exception:
            return None

    async def _send_media_file(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str],
        upload_media_type: int,
        wx_item_type: int,
        file_name: Optional[str] = None,
    ) -> SendResult:
        """Upload a file to CDN and send it as a WeChat message."""
        context_token = self._context_tokens.get(chat_id)
        if not context_token:
            return SendResult(success=False, error="No context_token for this user")

        uploaded = await self._cdn_upload(file_path, chat_id, upload_media_type)
        client_id = f"hermes-{uuid.uuid4().hex[:12]}"

        # Outbound aes_key is always base64(hex string) for all media types.
        # This matches the SDK: Buffer.from(hexKey).toString("base64")
        # Inbound _parse_aes_key() handles both formats for decryption.
        aes_key_b64 = base64.b64encode(uploaded["aeskey"].encode()).decode()

        # Build item based on type
        if wx_item_type == WX_ITEM_IMAGE:
            media_item = {
                "type": WX_ITEM_IMAGE,
                "image_item": {
                    "media": {
                        "encrypt_query_param": uploaded["download_param"],
                        "aes_key": aes_key_b64,
                        "encrypt_type": 1,
                    },
                    "mid_size": uploaded["ciphertext_size"],
                },
            }
        elif wx_item_type == WX_ITEM_VIDEO:
            media_item = {
                "type": WX_ITEM_VIDEO,
                "video_item": {
                    "media": {
                        "encrypt_query_param": uploaded["download_param"],
                        "aes_key": aes_key_b64,
                        "encrypt_type": 1,
                    },
                    "video_size": uploaded["ciphertext_size"],
                },
            }
        elif wx_item_type == WX_ITEM_FILE:
            media_item = {
                "type": WX_ITEM_FILE,
                "file_item": {
                    "media": {
                        "encrypt_query_param": uploaded["download_param"],
                        "aes_key": aes_key_b64,
                        "encrypt_type": 1,
                    },
                    "file_name": file_name or Path(file_path).name,
                    "len": str(uploaded["ciphertext_size"]),
                },
            }
        else:
            return SendResult(success=False, error=f"Unsupported item type: {wx_item_type}")

        # Build message items: optional caption text + media
        item_list = []
        if caption:
            item_list.append({"type": WX_ITEM_TEXT, "text_item": {"text": _markdown_to_plain(caption)}})
        item_list.append(media_item)

        # Send each item as a separate message (WeChat pattern)
        for item in item_list:
            cid = f"hermes-{uuid.uuid4().hex[:12]}"
            body = {
                "msg": {
                    "from_user_id": "",
                    "to_user_id": chat_id,
                    "client_id": cid,
                    "message_type": WX_MSG_TYPE_BOT,
                    "message_state": WX_MSG_STATE_FINISH,
                    "item_list": [item],
                    "context_token": context_token,
                },
            }
            resp = await self._api_fetch("ilink/bot/sendmessage", body)
            if resp.get("ret", 0) != 0:
                raise RuntimeError(f"sendmessage ret={resp.get('ret')} {resp.get('errmsg', '')}")

        return SendResult(success=True, message_id=client_id)

    # -- Chat info ----------------------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about a WeChat conversation."""
        return {
            "name": chat_id[:12] + "..." if len(chat_id) > 12 else chat_id,
            "type": "dm",
            "chat_id": chat_id,
        }

    # -- HTTP API layer -----------------------------------------------------

    async def _api_fetch(self, endpoint: str, body: dict, timeout_s: float = 15) -> dict:
        """POST JSON to a WeChat API endpoint."""
        if not self._http:
            raise RuntimeError("HTTP client not initialized")

        url = f"{self._base_url.rstrip('/')}/{endpoint}"
        payload = {**body, "base_info": {"channel_version": CHANNEL_VERSION}}

        # X-WECHAT-UIN: random uint32 → decimal string → base64
        rand_uint32 = struct.unpack(">I", os.urandom(4))[0]
        uin_b64 = base64.b64encode(str(rand_uint32).encode()).decode()

        headers = {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "X-WECHAT-UIN": uin_b64,
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        resp = await self._http.post(
            url, json=payload, headers=headers,
            timeout=timeout_s,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"WeChat API {endpoint} HTTP {resp.status_code}: {resp.text[:200]}")

        return resp.json()

    # -- CDN download/decrypt -----------------------------------------------

    async def _cdn_download_decrypt(self, encrypted_query_param: str, aes_key_b64: str) -> bytes:
        """Download from WeChat CDN and AES-128-ECB decrypt."""
        if not self._http:
            raise RuntimeError("HTTP client not initialized")

        key = _parse_aes_key(aes_key_b64)
        url = f"{self._cdn_base_url}/download?encrypted_query_param={urllib.parse.quote(encrypted_query_param, safe='')}"

        resp = await self._http.get(url, timeout=60)
        resp.raise_for_status()
        return _aes_ecb_decrypt(resp.content, key)

    async def _cdn_download_plain(self, encrypted_query_param: str) -> bytes:
        """Download from WeChat CDN without decryption."""
        if not self._http:
            raise RuntimeError("HTTP client not initialized")
        url = f"{self._cdn_base_url}/download?encrypted_query_param={urllib.parse.quote(encrypted_query_param, safe='')}"
        resp = await self._http.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content

    # -- CDN upload/encrypt -------------------------------------------------

    async def _cdn_upload(self, file_path: str, to_user_id: str, media_type: int) -> dict:
        """Encrypt and upload a file to WeChat CDN. Returns upload info dict."""
        if not self._http:
            raise RuntimeError("HTTP client not initialized")

        file_size_check = Path(file_path).stat().st_size
        if file_size_check > MEDIA_MAX_BYTES:
            raise ValueError(f"File too large: {file_size_check} bytes (max {MEDIA_MAX_BYTES})")
        plaintext = Path(file_path).read_bytes()
        raw_size = len(plaintext)
        raw_md5 = hashlib.md5(plaintext).hexdigest()
        file_size = _aes_ecb_padded_size(raw_size)
        filekey = os.urandom(16).hex()
        aes_key = os.urandom(16)

        # Get upload URL
        upload_resp = await self._api_fetch("ilink/bot/getuploadurl", {
            "filekey": filekey,
            "media_type": media_type,
            "to_user_id": to_user_id,
            "rawsize": raw_size,
            "rawfilemd5": raw_md5,
            "filesize": file_size,
            "no_need_thumb": True,
            "aeskey": aes_key.hex(),
        })

        upload_param = upload_resp.get("upload_param")
        if not upload_param:
            raise RuntimeError("getUploadUrl returned no upload_param")

        # Encrypt and upload
        ciphertext = _aes_ecb_encrypt(plaintext, aes_key)
        cdn_url = (
            f"{self._cdn_base_url}/upload"
            f"?encrypted_query_param={urllib.parse.quote(upload_param, safe='')}"
            f"&filekey={urllib.parse.quote(filekey, safe='')}"
        )

        # Retry logic (matches SDK)
        download_param = None
        last_error = None
        for attempt in range(1, 4):
            try:
                resp = await self._http.post(
                    cdn_url,
                    content=ciphertext,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=60,
                )
                if 400 <= resp.status_code < 500:
                    raise RuntimeError(f"CDN upload client error {resp.status_code}: {resp.text[:200]}")
                if resp.status_code != 200:
                    # Extract CDN error details for actionable error messages
                    cdn_err = resp.headers.get("x-error-message", "")
                    size_kb = raw_size // 1024
                    if "timeout" in cdn_err.lower():
                        raise RuntimeError(
                            f"CDN upload timeout for {size_kb}KB file. "
                            f"Try compressing the file (e.g. reduce audio sample rate, "
                            f"resize images, or re-encode video at lower bitrate)"
                        )
                    raise RuntimeError(f"CDN upload server error {resp.status_code}: {cdn_err}")

                download_param = resp.headers.get("x-encrypted-param")
                if not download_param:
                    raise RuntimeError("CDN response missing x-encrypted-param header")
                break
            except Exception as e:
                last_error = e
                if "client error" in str(e):
                    raise
                if attempt < 3:
                    logger.warning("[Weixin] CDN upload attempt %d failed: %s", attempt, e)

        if not download_param:
            raise last_error or RuntimeError("CDN upload failed after 3 attempts")

        return {
            "filekey": filekey,
            "download_param": download_param,
            "aeskey": aes_key.hex(),
            "plaintext_size": raw_size,
            "ciphertext_size": file_size,
            "raw_md5": raw_md5,
        }

    # -- Helpers ------------------------------------------------------------

    async def _download_remote_file(self, url: str) -> str:
        """Download a remote URL to a temp file, return local path."""
        if not self._http:
            raise RuntimeError("HTTP client not initialized")
        resp = await self._http.get(url, timeout=60, follow_redirects=True)
        resp.raise_for_status()

        # Determine extension from content-type or URL
        ct = resp.headers.get("content-type", "")
        ext = ".bin"
        if "jpeg" in ct or "jpg" in ct:
            ext = ".jpg"
        elif "png" in ct:
            ext = ".png"
        elif "gif" in ct:
            ext = ".gif"
        elif "webp" in ct:
            ext = ".webp"
        elif "mp4" in ct:
            ext = ".mp4"
        else:
            url_ext = Path(url.split("?")[0]).suffix.lower()
            if url_ext in _MIME_MAP:
                ext = url_ext

        _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        if ext in _IMAGE_EXTS:
            cache_dir = get_image_cache_dir()
        else:
            cache_dir = get_document_cache_dir()
        filename = f"wx_dl_{uuid.uuid4().hex[:12]}{ext}"
        filepath = cache_dir / filename
        filepath.write_bytes(resp.content)
        return str(filepath)

    def _persist_context_tokens(self) -> None:
        """Persist context_tokens to disk for standalone send_message tool."""
        try:
            tokens_dir = _weixin_state_dir()
            tokens_file = tokens_dir / "context_tokens.json"
            tokens_file.write_text(json.dumps(self._context_tokens), "utf-8")
            tokens_file.chmod(0o600)
        except Exception as e:
            logger.debug("[Weixin] Failed to persist context_tokens: %s", e)

    def format_message(self, content: str) -> str:
        """Strip markdown for WeChat (plain text only)."""
        return _markdown_to_plain(content)
