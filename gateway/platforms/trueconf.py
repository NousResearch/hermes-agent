"""TrueConf platform adapter.

Uses the python-trueconf-bot SDK for receiving and sending messages
via persistent WebSocket connection.

Env vars:
  - TRUECONF_SERVER         TrueConf server address (e.g. message.example.com)
  - TRUECONF_USERNAME       Bot username
  - TRUECONF_PASSWORD       Bot password
  - TRUECONF_ALLOWED_USERS  Comma-separated list of allowed user emails
  - TRUECONF_ALLOW_ALL_USERS Set to "true" to allow all users
  - TRUECONF_HOME_CHANNEL  Default chat ID for cron delivery
  - TRUECONF_VERIFY_SSL     Set to "false" or "0" to skip SSL verification
  - TRUECONF_PARSE_MODE     Parse mode for messages: "html" (default), "markdown", or "text"
"""

import asyncio
import logging
import os
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from trueconf.utils import safe_split_text
from trueconf import Bot, Dispatcher, Router, F
from trueconf.types import Message
from trueconf.enums import MessageType, ParseMode
from trueconf.types.content import AttachmentContent
from trueconf.types import FSInputFile
from trueconf.exceptions import FileSizeTooLargeError
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType as GatewayMessageType,
    SendResult,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_video_from_bytes,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# Dedicated file logger for the TrueConf adapter (writes to ~/.hermes/logs/bot.log)
# --------------------------------------------------------------------------------------------------
_hermes_home = Path.home() / ".hermes"
_bot_log_path = _hermes_home / "logs" / "bot.log"
_bot_log_path.parent.mkdir(parents=True, exist_ok=True)

_file_handler = logging.FileHandler(_bot_log_path, encoding="utf-8", mode="a")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))

_adapter_logger = logging.getLogger("gateway.platforms.trueconf")
_adapter_logger.setLevel(logging.DEBUG)
_adapter_logger.addHandler(_file_handler)
_adapter_logger.propagate = True


def check_trueconf_requirements() -> bool:
    """Check if TrueConf adapter dependencies are available."""
    import importlib.util
    if importlib.util.find_spec("trueconf") is None:
        return False
    return bool(
        os.getenv("TRUECONF_SERVER")
        and os.getenv("TRUECONF_USERNAME")
        and os.getenv("TRUECONF_PASSWORD")
    )


_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I
)


def _looks_like_uuid(value: str) -> bool:
    return bool(_UUID_RE.match(value.strip()))


class TrueConfAdapter(BasePlatformAdapter):
    """
    TrueConf WebSocket bot adapter.

    Maintains a persistent WebSocket connection to receive incoming messages.
    Uses Router/Dispatcher pattern from python-trueconf-bot SDK.
    """

    MAX_MESSAGE_LENGTH = 4096

    async def _resolve_chat_id(self, chat_id: str) -> Optional[str]:
        """
        Resolve a user_id (email/username) → chat_id lookup.

        TrueConf API requires chat_id (UUID) not user_id for SendMessage/SendFile.
        The mapping is populated from every incoming message via _user_to_chat.

        If no mapping exists for a user_id (email), attempts to create a P2P chat
        via create_personal_chat() so outbound file sends don't fail just because
        the user hasn't written to the bot first.
        """
        if not chat_id:
            return chat_id
        if _looks_like_uuid(chat_id):
            return chat_id
        normalized = chat_id.strip().lower()
        resolved = self._user_to_chat.get(normalized)
        if resolved:
            logger.debug("TrueConf resolved user_id=%s → chat_id=%s", chat_id, resolved)
            return resolved
        if "@" in chat_id and self._bot:
            try:
                result = await asyncio.wait_for(
                    self._bot.create_personal_chat(user_id=chat_id),
                    timeout=20.0,
                )
                if result and hasattr(result, "chat_id") and result.chat_id:
                    logger.info("TrueConf created P2P chat for %s → chat_id=%s",
                                chat_id, result.chat_id)
                    self._user_to_chat[normalized] = result.chat_id
                    return result.chat_id
                elif result and hasattr(result, "chat_id"):
                    logger.warning("TrueConf create_personal_chat(%s) returned empty chat_id",
                                   chat_id)
            except asyncio.TimeoutError:
                logger.warning("TrueConf create_personal_chat(%s) timed out after 20s", chat_id)
            except Exception as e:
                logger.warning("TrueConf create_personal_chat(%s) failed: %s", chat_id, e)
            return None
        return chat_id

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.TRUECONF)
        self._server: str = os.getenv("TRUECONF_SERVER", "").strip()
        self._username: str = os.getenv("TRUECONF_USERNAME", "").strip()
        self._password: str = os.getenv("TRUECONF_PASSWORD", "").strip()
        self._verify_ssl: bool = os.getenv("TRUECONF_VERIFY_SSL", "true").lower() not in (
            "false", "0", "no"
        )

        self._bot: Optional[Bot] = None
        self._dispatcher: Optional[Dispatcher] = None
        self._router: Optional[Router] = None
        self._bot_task: Optional[asyncio.Task] = None
        self._allowed_users: Optional[set] = None
        self._allow_all: bool = False

        # Parse mode: read from TRUECONF_PARSE_MODE, default to "html"
        self._parse_mode: str = os.getenv("TRUECONF_PARSE_MODE", "html").strip().lower()
        if self._parse_mode not in ("markdown", "html", "text"):
            logger.warning(
                "TrueConf: invalid TRUECONF_PARSE_MODE '%s', defaulting to 'html'",
                self._parse_mode
            )
            self._parse_mode = "html"

        # Bot's favorites chat_id for echo detection
        self._favorites_chat_id: Optional[str] = None

        # user_id → chat_id lookup for outgoing sends
        self._user_to_chat: Dict[str, str] = {}

        # Last message tracking per chat: chat_id → {'msg_id': str, 'content': str, 'is_system': bool}
        # Used to edit system messages in-place instead of sending new ones.
        # System messages are merged only if the previous message was also a system message.
        self._last_message: Dict[str, Dict[str, Any]] = {}

        # Reconnection state
        self._reconnect_delay: float = 5.0
        self._reconnect_attempts: int = 0
        self._max_reconnect_delay: float = 60.0
        self._should_reconnect: bool = True

        allowed = os.getenv("TRUECONF_ALLOWED_USERS", "").strip()
        self._allow_all = os.getenv("TRUECONF_ALLOW_ALL_USERS", "").lower() in (
            "true", "1", "yes"
        )
        if allowed and not self._allow_all:
            self._allowed_users = {u.strip().lower() for u in allowed.split(",") if u.strip()}

    # ------------------------------------------------------------------
    # Connection management - persistent WebSocket connection
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Start the persistent WebSocket bot connection."""
        try:
            self._router = Router()
            self._dispatcher = Dispatcher()
            self._dispatcher.include_router(self._router)

            self._bot = Bot.from_credentials(
                self._server,
                self._username,
                self._password,
                verify_ssl=self._verify_ssl,
                dispatcher=self._dispatcher,
                receive_unread_messages=True,
            )

            # Register handlers in specific-to-generic order
            @self._router.message(F.content_type == MessageType.ATTACHMENT, F.photo.is_not(None))
            async def on_photo(msg: Message):
                await self._handle_incoming_message(msg)

            @self._router.message(F.content_type == MessageType.ATTACHMENT, F.video.is_not(None))
            async def on_video(msg: Message):
                await self._handle_incoming_message(msg)

            @self._router.message(F.content_type == MessageType.ATTACHMENT, F.document.is_not(None))
            async def on_document(msg: Message):
                await self._handle_incoming_message(msg)

            @self._router.message(F.content_type == MessageType.ATTACHMENT, F.sticker.is_not(None))
            async def on_sticker(msg: Message):
                await self._handle_incoming_message(msg)

            @self._router.message()
            async def on_message(msg: Message):
                await self._handle_incoming_message(msg)

            self._bot_task = asyncio.create_task(self._run_bot())
            await asyncio.sleep(5)

            try:
                self._favorites_chat_id = await asyncio.wait_for(self._bot.me, timeout=10.0)
                logger.info("TrueConf favorites_chat_id=%s for echo prevention",
                            self._favorites_chat_id)
            except Exception as e:
                logger.warning("TrueConf: could not get favorites_chat_id: %s", e)

            logger.info("TrueConf bot started, WebSocket connecting...")
            return True

        except Exception as e:
            logger.warning("TrueConf connect failed: %s", e)
            return False

    async def _run_bot(self) -> None:
        """Run the bot with auto-reconnect on unexpected exit."""
        RUN_KWARGS = {"handle_signals": False}

        while self._should_reconnect:
            try:
                await self._bot.run(**RUN_KWARGS)
            except asyncio.CancelledError:
                logger.info("TrueConf bot.run() cancelled")
                break
            except Exception as e:
                logger.warning("TrueConf bot.run() exited with error: %s", e)

            if not self._should_reconnect:
                logger.info("TrueConf bot shutdown complete, not reconnecting")
                break

            self._reconnect_attempts += 1
            delay = min(
                self._reconnect_delay * (2 ** (self._reconnect_attempts - 1)),
                self._max_reconnect_delay,
            )
            logger.info("TrueConf bot reconnecting in %.1fs (attempt %d)",
                        delay, self._reconnect_attempts)
            await asyncio.sleep(delay)

            if not self._should_reconnect:
                break

            try:
                from trueconf import Bot, Dispatcher, Router
                from trueconf.types import Message
                from trueconf.enums import MessageType

                self._router = Router()
                self._dispatcher = Dispatcher()
                self._dispatcher.include_router(self._router)

                @self._router.message(F.content_type == MessageType.ATTACHMENT, F.photo.is_not(None))
                async def on_photo(msg: Message):
                    await self._handle_incoming_message(msg)

                @self._router.message(F.content_type == MessageType.ATTACHMENT, F.video.is_not(None))
                async def on_video(msg: Message):
                    await self._handle_incoming_message(msg)

                @self._router.message(F.content_type == MessageType.ATTACHMENT, F.document.is_not(None))
                async def on_document(msg: Message):
                    await self._handle_incoming_message(msg)

                @self._router.message(F.content_type == MessageType.ATTACHMENT, F.sticker.is_not(None))
                async def on_sticker(msg: Message):
                    await self._handle_incoming_message(msg)

                @self._router.message()
                async def on_message(msg: Message):
                    await self._handle_incoming_message(msg)

                self._bot = Bot.from_credentials(
                    self._server,
                    self._username,
                    self._password,
                    verify_ssl=self._verify_ssl,
                    dispatcher=self._dispatcher,
                    receive_unread_messages=True,
                )

                logger.info("TrueConf bot recreated, reconnecting...")
            except Exception as create_err:
                logger.error("TrueConf bot recreation failed: %s", create_err)
                await asyncio.sleep(delay)

    async def disconnect(self) -> None:
        """Gracefully shutdown the bot."""
        self._should_reconnect = False
        if self._bot:
            try:
                await self._bot.shutdown()
            except Exception as e:
                logger.warning("TrueConf shutdown error: %s", e)
        if self._bot_task:
            self._bot_task.cancel()
            try:
                await self._bot_task
            except asyncio.CancelledError:
                pass
        self._bot = None
        self._bot_task = None
        self._favorites_chat_id = None
        logger.info("TrueConf bot disconnected")

    # ------------------------------------------------------------------
    # Inbound message handling
    # ------------------------------------------------------------------

    async def _handle_incoming_message(self, msg) -> None:
        """Process incoming TrueConf Message and send to gateway handler."""
        try:
            from trueconf.enums import ChatType
            from trueconf.types.content import AttachmentContent

            # Echo prevention
            if self._favorites_chat_id and msg.chat_id == self._favorites_chat_id:
                logger.debug("[%s] Skipping message from bot's own chat (favorites)",
                             self.name)
                return

            # Access control
            author_id = str(msg.author.id) if msg.author and hasattr(msg.author, 'id') else None
            if not self._allow_all and self._allowed_users:
                if author_id:
                    if author_id.strip().lower() not in self._allowed_users:
                        logger.debug("TrueConf message from unauthorized user: %s", author_id)
                        return
                else:
                    logger.debug("TrueConf message has no author id, skipping access check")

            # Determine chat type
            chat_type_map = {
                ChatType.P2P: "dm",
                ChatType.GROUP: "group",
                ChatType.CHANNEL: "channel",
                ChatType.FAVORITES: "dm",
                ChatType.SYSTEM: "dm",
            }
            box = msg.box
            chat_type = "dm"
            if box:
                try:
                    chat_type = chat_type_map.get(getattr(box, "type", None), "dm")
                except Exception:
                    chat_type = "dm"
            chat_name = getattr(box, "title", None) if box else None

            text = msg.text or ""
            chat_id = msg.chat_id or ""
            msg_id = msg.message_id or ""

            # Populate user_id → chat_id mapping for P2P chats only
            if author_id and chat_id and chat_type == "dm":
                self._user_to_chat[author_id.strip().lower()] = chat_id

            msg_type = GatewayMessageType.TEXT
            media_urls = []
            media_types = []

            photo = msg.photo
            if photo:
                msg_type = GatewayMessageType.PHOTO
                cached = await self._cache_file_with_download(
                    getattr(photo, "file_id", None),
                    getattr(photo, "mimetype", "") or "image/jpeg",
                    getattr(photo, "file_name", None),
                    "image",
                )
                if cached:
                    media_urls.append(cached)
                    media_types.append(getattr(photo, "mimetype", "") or "image/jpeg")
                else:
                    fn = getattr(photo, "file_name", None) or "image"
                    sz = getattr(photo, "file_size", 0) or 0
                    text = (f"[Image received but download failed: {fn}, "
                            f"{sz / 1024:.1f} KB]") if sz else "[Image received but download failed]"

            video = msg.video
            if video:
                msg_type = GatewayMessageType.VIDEO
                cached = await self._cache_file_with_download(
                    getattr(video, "file_id", None),
                    getattr(video, "mimetype", "") or "video/mp4",
                    getattr(video, "file_name", None),
                    "video",
                )
                if cached:
                    media_urls.append(cached)
                    media_types.append(getattr(video, "mimetype", "") or "video/mp4")
                else:
                    fn = getattr(video, "file_name", None) or "video"
                    sz = getattr(video, "file_size", 0) or 0
                    text = (f"[Video received but download failed: {fn}, "
                            f"{sz / 1024:.1f} KB]") if sz else "[Video received but download failed]"

            document = msg.document
            if document:
                msg_type = GatewayMessageType.DOCUMENT
                cached = await self._cache_file_with_download(
                    getattr(document, "file_id", None),
                    getattr(document, "mimetype", "") or "application/octet-stream",
                    getattr(document, "file_name", None),
                    "document",
                )
                if cached:
                    media_urls.append(cached)
                    media_types.append(
                        getattr(document, "mimetype", "") or "application/octet-stream"
                    )
                else:
                    fn = getattr(document, "file_name", None) or "document"
                    sz = getattr(document, "file_size", 0) or 0
                    text = (f"[File received but download failed: {fn}, "
                            f"{sz / 1024:.1f} KB]") if sz else "[File received but download failed]"

                # Extract text content for readable documents (cap 100KB)
                text_ext = Path(getattr(document, "file_name", "") or "").suffix.lower()
                if text_ext in (".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml",
                                ".log", ".py", ".js", ".ts", ".html", ".css", ".pyc"):
                    try:
                        tmp_dir = tempfile.mkdtemp(prefix="tc_text_")
                        doc_path = await self._bot.download_file_by_id(
                            getattr(document, "file_id", None), tmp_dir
                        )
                        if (doc_path and Path(doc_path).exists()
                                and Path(doc_path).stat().st_size <= 100 * 1024):
                            content_str = Path(doc_path).read_text("utf-8", errors="replace")
                            injection = (f"[Content of "
                                         f"{Path(getattr(document, 'file_name', 'file')).name}]:\n"
                                         f"{content_str}")
                            text = f"{injection}\n\n{text}" if text else injection
                        if doc_path and Path(doc_path).exists():
                            try:
                                Path(doc_path).unlink()
                            except Exception:
                                pass
                        shutil.rmtree(tmp_dir)
                    except Exception as e:
                        logger.debug("[TrueConf] Could not read text content: %s", e)

            sticker = msg.sticker
            if sticker and not media_urls:
                cached = await self._cache_file_with_download(
                    getattr(sticker, "file_id", None),
                    getattr(sticker, "mimetype", "") or "image/webp",
                    getattr(sticker, "file_name", None),
                    "image",
                )
                if cached:
                    media_urls.append(cached)
                    media_types.append(getattr(sticker, "mimetype", "") or "image/webp")
                else:
                    fn = getattr(sticker, "file_name", None) or "sticker"
                    text = f"[Sticker received but download failed: {fn}]"

            # Handle audio/* via msg.content
            content = msg.content
            if isinstance(content, AttachmentContent) and content.mimetype.startswith("audio/"):
                msg_type = GatewayMessageType.AUDIO
                cached = await self._cache_file_with_download(
                    getattr(content, "file_id", None),
                    content.mimetype,
                    getattr(content, "file_name", None),
                    "audio",
                )
                if cached:
                    media_urls.append(cached)
                    media_types.append(content.mimetype)
                else:
                    fn = getattr(content, "file_name", None) or "audio file"
                    sz = getattr(content, "file_size", 0) or 0
                    text = (f"[Audio received but download failed: {fn}, "
                            f"{sz / 1024:.1f} KB]") if sz else "[Audio received but download failed]"

            # Skip empty messages
            if not text and not media_urls:
                logger.debug("TrueConf: ignoring empty message type: %s", msg.type)
                return

            user_id_str = author_id or ""
            sender_name = (getattr(msg.author, 'display_name', None)
                           or getattr(msg.author, 'id', None) or "")

            source = self.build_source(
                chat_id=str(chat_id),
                chat_name=chat_name,
                chat_type=chat_type,
                user_id=user_id_str,
                user_name=sender_name,
            )

            event = MessageEvent(
                text=str(text),
                message_type=msg_type,
                source=source,
                raw_message=msg,
                message_id=str(msg_id),
                media_urls=media_urls,
                media_types=media_types,
            )

            await self.handle_message(event)

        except Exception as e:
            logger.error("TrueConf message handler error: %s", e, exc_info=True)

    async def _cache_file_with_download(
        self, file_id: str, mimetype: str, file_name: Optional[str], file_kind: str
    ) -> Optional[str]:
        """Download a file using bot.download_file_by_id()."""
        if not self._bot or not file_id:
            return None

        ext = self._ext_from_mimetype(
            mimetype, file_name,
            ".jpg" if file_kind == "image" else
            ".mp4" if file_kind == "video" else
            ".ogg" if file_kind == "audio" else ".bin"
        )

        tmp_path: Optional[str] = None
        for round_num in range(1, 16):
            try:
                result = await asyncio.wait_for(
                    self._bot.download_file_by_id(file_id, dest_path=None),
                    timeout=60.0,
                )
            except asyncio.TimeoutError:
                logger.warning("[TrueConf] download_file_by_id(%s) timeout at round %d",
                                file_id, round_num)
                await asyncio.sleep(3)
                continue
            except Exception as e:
                logger.warning("[TrueConf] download_file_by_id(%s) raised %s at round %d",
                                file_id, e, round_num)
                await asyncio.sleep(3)
                continue

            if result is None:
                logger.warning("[TrueConf] download_file_by_id(%s) returned None at round %d",
                                file_id, round_num)
                await asyncio.sleep(3)
                continue

            raw_bytes: bytes
            if isinstance(result, bytes):
                raw_bytes = result
            else:
                result_path = Path(result)
                if not result_path.exists() or result_path.stat().st_size == 0:
                    logger.warning("[TrueConf] downloaded file empty at round %d", round_num)
                    await asyncio.sleep(3)
                    continue
                raw_bytes = result_path.read_bytes()

            if not raw_bytes:
                logger.warning("zero bytes at round %d", round_num)
                await asyncio.sleep(3)
                continue

            cached_path: Optional[str] = None
            if file_kind == "image":
                cache_ext = self._ext_from_mimetype(mimetype, file_name, ".jpg")
                cached_path = cache_image_from_bytes(raw_bytes, ext=cache_ext)
            elif file_kind == "audio":
                cache_ext = self._ext_from_mimetype(mimetype, file_name, ".ogg")
                cached_path = cache_audio_from_bytes(raw_bytes, ext=cache_ext)
            elif file_kind == "video":
                cache_ext = self._ext_from_mimetype(mimetype, file_name, ".mp4")
                cached_path = cache_video_from_bytes(raw_bytes, ext=cache_ext)
            else:
                safe_name = file_name or f"file.{mimetype.split('/')[-1]}"
                cached_path = cache_document_from_bytes(raw_bytes, safe_name)

            if cached_path:
                logger.info("[TrueConf] Cached %s file (ID=%s) at %s (%d bytes)",
                             file_kind, file_id, cached_path, len(raw_bytes))
                return cached_path

            await asyncio.sleep(3)

        logger.warning("[TrueConf] File %s never downloaded successfully in 15 rounds",
                       file_id)
        return None

    @staticmethod
    def _ext_from_mimetype(mimetype: str, file_name: str, fallback: str) -> str:
        """Map mimetype to common file extension."""
        name_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "audio/ogg": ".ogg",
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            "audio/aac": ".aac",
            "audio/flac": ".flac",
            "video/mp4": ".mp4",
            "video/webm": ".webm",
            "video/quicktime": ".mov",
            "video/x-msvideo": ".avi",
        }
        return name_map.get(mimetype, Path(file_name).suffix if file_name else fallback)

    # ------------------------------------------------------------------
    # System message detection
    # ------------------------------------------------------------------

    def _is_system_message(self, content: str) -> bool:
        """
        Check if content is a system message (agent tool execution).
        System messages start with an emoji followed by tool name and colon.
        Examples: "📖 read_file: path", "💻 terminal: command", "🐍 execute_code:"
        Regular messages with emojis won't match this pattern.
        
        Handles multi-character emojis (like ⌨️ which is \u2338 + \ufe0f).
        """
        if not content:
            return False

        # Pattern: one or more non-whitespace (emoji) + whitespace + tool_name + ":"
        # \S+ matches the entire emoji (including variation selectors)
        # Tool names are typically lowercase with underscores
        return bool(re.match(r'^\S+\s+[a-z_]+:', content))

    @staticmethod
    def _markdown_to_html(text: str) -> str:
        """
        Convert basic Markdown to HTML for TrueConf.
        Handles: **bold**, *italic*, __bold__, _italic_, ~~strikethrough~~,
                 `code`, ```code block```, [link](url).

        Preserves original formatting inside code spans (`` `...` ``)
        and fenced code blocks (`` ```...``` ``) — no inner markdown
        conversion happens inside those.
        """
        import re

        # --- Placeholder system ---
        # Save code content before markdown conversion so subsequent
        # regexes can't touch it, then restore it at the end.
        _placeholders: dict[str, str] = {}
        _counter = 0

        def _save(match: re.Match) -> str:
            nonlocal _counter
            _counter += 1
            # NOTE: плейсхолдер содержит только буквы+цифры — без _, *, ~, [, ], (, )
            # чтобы ни один последующий регексп не зацепил его содержимое
            key = f"TRUECNFMD{_counter}"
            _placeholders[key] = match.group(0)
            return key

        # 1. Fenced code blocks (```…```) — longest first, with DOTALL
        text = re.sub(r'```.*?```', _save, text, flags=re.DOTALL)

        # 2. Inline code (`…`)
        text = re.sub(r'`[^`]+`', _save, text)

        # 3. Convert remaining markdown → HTML
        # Bold: **text** → <b>text</b>
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

        # Bold: __text__ → <b>text</b>
        text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

        # Italic: *text* → <i>text</i>
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)

        # Italic: _text_ → <i>text</i>
        text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)

        # Strikethrough: ~~text~~ → <s>text</s>
        text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)

        # Links: [text](url) → <a href="url">text</a>
        text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', text)

        # 4. Restore original code content
        for key, original in _placeholders.items():
            text = text.replace(key, original)

        return text

    def _get_parse_mode(self):
        """Get the ParseMode based on TRUECONF_PARSE_MODE setting."""
        if self._parse_mode == "html":
            return ParseMode.HTML
        elif self._parse_mode == "markdown":
            return ParseMode.MARKDOWN
        else:
            return ParseMode.TEXT


    # ------------------------------------------------------------------
    # Outbound sending
    # ------------------------------------------------------------------

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None,
        thread_id: Optional[str] = None, **kwargs
    ) -> SendResult:
        """Send a text message. For system messages (starting with emoji),
        edits the previous message only if it was also a system message.
        """
        if not self._bot:
            return SendResult(success=False, message_id=None, error="Bot not connected")

        # Check if this is a system message
        is_system = self._is_system_message(content)
        
        logger.info(f"TrueConf send: is_system={is_system}, chat_id={chat_id}, content_preview={content[:50]}")

        if is_system:
            # System message - check if previous message was also system
            last = self._last_message.get(chat_id)
            logger.info(f"TrueConf send system: last_message={last}")

            if last and last.get('is_system'):
                # Previous message was system - APPEND to it and EDIT
                accumulated = last['content'] + '\n' + content
                logger.info(f"TrueConf editing message {last['msg_id']}, accumulated length={len(accumulated)}")

                edit_result = await self.edit_message(
                    chat_id=chat_id,
                    message_id=last['msg_id'],
                    content=accumulated
                )

                logger.info(f"TrueConf edit result: success={edit_result.success}, error={edit_result.error}")
                
                if edit_result.success:
                    # Update tracked state with new accumulated content
                    last['content'] = accumulated
                    return edit_result
                # If edit fails, fall through to send new
                logger.warning(f"TrueConf edit failed, falling through to send new. error={edit_result.error}")

            # Send NEW system message (first one or fallback from failed edit)
            resolved = await self._resolve_chat_id(chat_id)
            if not resolved:
                return SendResult(
                    success=False, message_id=None,
                    error=f"Chat ID not resolved for {chat_id}"
                )

            clean_text = content.strip()
            if not clean_text:
                return SendResult(success=False, message_id=None, error="Empty message")

            try:
                # Convert markdown to HTML if needed
                text_to_send = clean_text
                if self._parse_mode == "html":
                    text_to_send = self._markdown_to_html(clean_text)
                
                result = await asyncio.wait_for(
                    self._bot.send_message(
                        chat_id=resolved,
                        text=text_to_send,
                        parse_mode=self._get_parse_mode(),
                        reply_message_id=reply_to,
                    ),
                    timeout=60.0,
                )
                if result and hasattr(result, "message_id"):
                    msg_id = str(result.message_id)
                    # Track this as the last message (system)
                    self._last_message[chat_id] = {
                        'msg_id': msg_id,
                        'content': content,
                        'is_system': True
                    }
                    logger.info(f"TrueConf sent NEW system message: msg_id={msg_id}")
                    return SendResult(success=True, message_id=msg_id, error=None)
                return SendResult(success=False, message_id=None, error="send_message failed")
            except asyncio.TimeoutError:
                logger.error("TrueConf send_message timeout")
                return SendResult(success=False, message_id=None, error="Timeout sending message")
            except Exception as e:
                logger.error("TrueConf send error: %s", e)
                return SendResult(success=False, message_id=None, error=str(e))

        else:
            # Non-system message - clear state and send normally with chunking
            self._last_message.pop(chat_id, None)

            resolved_chat_id = await self._resolve_chat_id(chat_id)
            if not resolved_chat_id:
                return SendResult(
                    success=False, message_id=None,
                    error=f"Chat ID not resolved for {chat_id}"
                )

            clean_text = content.strip()
            if not clean_text:
                return SendResult(
                    success=False, message_id=None,
                    error="Empty message after stripping markdown"
                )

            chunks = safe_split_text(clean_text, limit=self.MAX_MESSAGE_LENGTH)
            if not chunks:
                return SendResult(
                    success=False, message_id=None,
                    error="safe_split_text returned empty result"
                )

            sent_ids: List[str] = []
            last_error: Optional[str] = None

            for chunk in chunks:
                # Convert markdown to HTML if needed
                if self._parse_mode == "html":
                    chunk = self._markdown_to_html(chunk)

                # Log raw chunk before sending to debug markdown issues
                logger.info(f"Chunk: {chunk}")

                try:
                    result = await asyncio.wait_for(
                        self._bot.send_message(
                            chat_id=resolved_chat_id,
                            text=chunk,
                            parse_mode=self._get_parse_mode(),
                            reply_message_id=reply_to,
                        ),
                        timeout=60.0,
                    )
                    if result and hasattr(result, "message_id"):
                        sent_ids.append(str(result.message_id))
                        last_error = None
                    else:
                        last_error = str(result)
                except asyncio.TimeoutError:
                    logger.error("TrueConf send_message timeout")
                    last_error = "Timeout sending message"
                    break
                except Exception as e:
                    logger.error("TrueConf send error: %s", e)
                    last_error = str(e)
                    break

            if sent_ids:
                # Track the first message as the last message (non-system)
                self._last_message[chat_id] = {
                    'msg_id': sent_ids[0],
                    'content': clean_text[:200],  # store preview
                    'is_system': False
                }
                return SendResult(success=True, message_id=sent_ids[0], error=None)
            return SendResult(success=False, message_id=None, error=last_error)

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, thread_id: Optional[str] = None, **kwargs
    ) -> SendResult:
        """Send an image by URL."""
        return await self._send_image_url(chat_id, image_url, caption, reply_to)

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, thread_id: Optional[str] = None, **kwargs
    ) -> SendResult:
        """Send an image from local file."""
        if not self._bot:
            return SendResult(success=False, message_id=None, error="Bot not connected")

        resolved = await self._resolve_chat_id(chat_id)
        if not resolved:
            return SendResult(
                success=False, message_id=None,
                error=f"Chat ID not resolved for {chat_id}"
            )

        try:
            from trueconf.enums import ParseMode
            from trueconf.types import FSInputFile
            file_data = FSInputFile(image_path)

            # Convert caption markdown to HTML if needed
            caption_to_send = caption
            if caption and self._parse_mode == "html":
                caption_to_send = self._markdown_to_html(caption)

            result = await asyncio.wait_for(
                self._bot.send_photo(
                    chat_id=resolved,
                    file=file_data,
                    preview=None,
                    caption=caption_to_send,
                    parse_mode=self._get_parse_mode(),
                    reply_message_id=reply_to,
                ),
                timeout=120.0,
            )
            if result and hasattr(result, "message_id"):
                return SendResult(success=True, message_id=str(result.message_id), error=None)
            return SendResult(success=False, message_id=None, error="send_photo failed")
        except asyncio.TimeoutError:
            logger.error("TrueConf send_image_file timeout")
            return SendResult(success=False, message_id=None, error="Timeout sending image")
        except FileSizeTooLargeError as e:
            logger.warning("TrueConf server rejected image (too large): %s", e)
            return SendResult(
                success=False, message_id=None,
                error=f"File too large for TrueConf server: {e}"
            )
        except Exception as e:
            logger.error("TrueConf send_image_file error: %s", e)
            return SendResult(success=False, message_id=None, error=str(e))

    async def _send_image_url(
        self, chat_id: str, image_url: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None
    ) -> SendResult:
        """Download image from URL and send it."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                urllib.request.urlretrieve(image_url, f.name)
                temp_path = f.name
        except Exception as e:
            return SendResult(
                success=False, message_id=None,
                error=f"Failed to download image: {e}"
            )

        result = await self.send_image_file(chat_id, temp_path, caption, reply_to)
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        return result

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, thread_id: Optional[str] = None, **kwargs
    ) -> SendResult:
        """Send a video — delegates to send_document."""
        return await self.send_document(chat_id, video_path, caption, reply_to,
                                        thread_id, **kwargs)

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, thread_id: Optional[str] = None, **kwargs
    ) -> SendResult:
        """Send a document from local file."""
        if not self._bot:
            return SendResult(success=False, message_id=None, error="Bot not connected")

        resolved = await self._resolve_chat_id(chat_id)
        if not resolved:
            return SendResult(
                success=False, message_id=None,
                error=f"Chat ID not resolved for {chat_id}"
            )

        try:
            from trueconf.enums import ParseMode
            from trueconf.types import FSInputFile
            file_data = FSInputFile(file_path)

            # Convert caption markdown to HTML if needed
            caption_to_send = caption
            if caption and self._parse_mode == "html":
                caption_to_send = self._markdown_to_html(caption)

            result = await asyncio.wait_for(
                self._bot.send_document(
                    chat_id=resolved,
                    file=file_data,
                    caption=caption_to_send,
                    parse_mode=self._get_parse_mode(),
                    reply_message_id=reply_to,
                ),
                timeout=120.0,
            )
            if result and hasattr(result, "message_id"):
                return SendResult(success=True, message_id=str(result.message_id), error=None)
            return SendResult(success=False, message_id=None, error="send_document failed")
        except asyncio.TimeoutError:
            logger.error("TrueConf send_document timeout")
            return SendResult(success=False, message_id=None, error="Timeout sending document")
        except FileSizeTooLargeError as e:
            logger.warning("TrueConf server rejected file (too large): %s", e)
            return SendResult(
                success=False, message_id=None,
                error=f"File too large for TrueConf server: {e}"
            )
        except Exception as e:
            logger.error("TrueConf send_document error: %s | chat_id=%s file=%s",
                          e, chat_id, file_path)
            return SendResult(success=False, message_id=None, error=str(e))

    async def send_voice(
        self, chat_id: str, audio_path: str,
        reply_to: Optional[str] = None, thread_id: Optional[str] = None, **kwargs
    ) -> SendResult:
        """Send a voice message — delegates to send_document."""
        return await self.send_document(chat_id, audio_path, None, reply_to, thread_id, **kwargs)

    async def send_sticker(
        self, chat_id: str, sticker: str = "",
        reply_to: Optional[str] = None, thread_id: Optional[str] = None, **kwargs
    ) -> SendResult:
        """Send a sticker using FSInputFile."""
        if not self._bot:
            return SendResult(success=False, message_id=None, error="Bot not connected")

        resolved = await self._resolve_chat_id(chat_id)
        if not resolved:
            return SendResult(
                success=False, message_id=None,
                error=f"Chat ID not resolved for {chat_id}"
            )

        try:
            from trueconf.types import FSInputFile

            if sticker.startswith(("http://", "https://")):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as f:
                    urllib.request.urlretrieve(sticker, f.name)
                    temp_path = f.name
                try:
                    file_data = FSInputFile(temp_path)
                finally:
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
            else:
                file_data = FSInputFile(sticker)

            result = await asyncio.wait_for(
                self._bot.send_sticker(
                    chat_id=resolved,
                    file=file_data,
                    reply_message_id=reply_to,
                ),
                timeout=60.0,
            )
            if result and hasattr(result, "message_id"):
                return SendResult(success=True, message_id=str(result.message_id), error=None)
            return SendResult(success=False, message_id=None, error="send_sticker failed")
        except asyncio.TimeoutError:
            logger.error("TrueConf send_sticker timeout")
            return SendResult(success=False, message_id=None, error="Timeout sending sticker")
        except FileSizeTooLargeError as e:
            logger.warning("TrueConf server rejected sticker (too large): %s", e)
            return SendResult(
                success=False, message_id=None,
                error=f"File too large for TrueConf server: {e}"
            )
        except Exception as e:
            logger.error("TrueConf send_sticker error: %s", e)
            return SendResult(success=False, message_id=None, error=str(e))

    async def send_typing(self, chat_id: str, **kwargs) -> None:
        """TrueConf SDK does not support typing indicators."""
        pass

    async def edit_message(
        self, chat_id: str, message_id: str, content: str,
        *, finalize: bool = False,
    ) -> SendResult:
        """Edit a previously sent message (full content, no chunking)."""
        if not self._bot:
            return SendResult(success=False, message_id=None, error="Bot not connected")

        clean_text = content.strip().replace("\n\n", "\n")
        if not clean_text:
            return SendResult(success=False, message_id=None, error="Empty content")

        logger.info(f"TrueConf edit_message: msg_id={message_id}, content_length={len(clean_text)}")

        try:
            from trueconf.enums import ParseMode
            # Note: TrueConf Bot API edit_message does NOT accept chat_id parameter
            # The API identifies the message by message_id alone
            
            # Convert markdown to HTML if needed
            text_to_send = clean_text
            if self._parse_mode == "html":
                text_to_send = self._markdown_to_html(clean_text)

            result: EditMessageResponse = await asyncio.wait_for(
                self._bot.edit_message(
                    message_id=message_id,
                    text=text_to_send,
                    parse_mode=self._get_parse_mode(),
                ),
                timeout=30.0,
            )
            if result and hasattr(result, "message_id"):
                logger.info(f"TrueConf edit_message success: msg_id={result.message_id}")
                return SendResult(success=True, message_id=str(result.message_id), error=None)
            logger.error(f"TrueConf edit_message failed: result={result}")
            return SendResult(success=False, message_id=None, error=str(result))
        except asyncio.TimeoutError:
            logger.error("TrueConf edit_message timeout")
            return SendResult(success=False, message_id=None, error="Timeout editing message")
        except Exception as e:
            logger.error("TrueConf edit_message error: %s", e)
            return SendResult(success=False, message_id=None, error=str(e))

    # ------------------------------------------------------------------
    # Chat info
    # ------------------------------------------------------------------

    def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get basic chat info."""
        return {"name": chat_id, "type": "direct", "chat_id": chat_id}
