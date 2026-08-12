"""OneBot 11 platform adapter for Hermes Agent.

Connects Hermes to QQ through the OneBot 11 protocol, compatible with
NapCat / Lagrange / LLOneBot / go-cqhttp.

Transports:
- reverse (default): the adapter hosts a WebSocket server; NapCat's
  "ws-reverse" client dials in. Inbound events and outbound actions
  share that single connection.
- forward: the adapter dials NapCat's "ws" server (ws://host:port).

Configuration (config.yaml):

    gateway:
      platforms:
        onebot:
          enabled: true
          extra:
            mode: reverse              # reverse | forward
            host: "127.0.0.1"          # reverse: listen address
            port: 8643                 # reverse: listen port
            url: "ws://127.0.0.1:3001" # forward: NapCat ws endpoint
            access_token: ""           # optional OneBot access token
            bot_qq: ""                 # optional; auto-learned from meta events
            require_mention: true      # group chats: only reply when @'d
            dm_policy: open            # open | allowlist | disabled
            allow_from: []             # user ids when dm_policy=allowlist
            group_policy: open         # open | allowlist | disabled
            group_allow_from: []       # group ids when group_policy=allowlist
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import aiohttp
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - gateway always ships aiohttp
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource

# ---------------------------------------------------------------------------
# OneBot 11 constants
# ---------------------------------------------------------------------------

DEFAULT_PORT = 8643
ACTION_TIMEOUT = 30.0
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]
MAX_RECONNECT_ATTEMPTS = 100
IMAGE_MAX_BYTES = 8 * 1024 * 1024  # skip absurdly large CQ image downloads
MAX_MESSAGE_LENGTH = 4000  # QQ per-message cap (UTF-16-ish, keep safe)

_CQ_AT_RE = re.compile(r"\[CQ:at,qq=(\d+|all)\]")
_CQ_IMAGE_RE = re.compile(r"\[CQ:image,[^\]]*?url=([^,\]]+)\]")
_CQ_IMAGE_NOURL_RE = re.compile(r"\[CQ:image(?:,[^\]]*)?\]")
_CQ_RECORD_RE = re.compile(r"\[CQ:record,[^\]]*?url=([^,\]]+)\]")
_CQ_RECORD_NOURL_RE = re.compile(r"\[CQ:record(?:,[^\]]*)?\]")
_CQ_REPLY_RE = re.compile(r"\[CQ:reply,id=(\d+)\]")
_CQ_FACE_RE = re.compile(r"\[CQ:face,id=(\d+)\]")
_CQ_ANY_RE = re.compile(r"\[CQ:[^\]]*\]")

# Max bytes for a downloaded voice clip (silk/amr from QQ).
AUDIO_MAX_BYTES = 15 * 1024 * 1024

# Reply splitting: messages longer than this are sent as multiple messages,
# breaking at sentence boundaries (。！？!?；;\n) instead of mid-sentence.
DEFAULT_SPLIT_LENGTH = 100
_SENTENCE_BOUNDS = "。！？!?；;\n"

# Content longer than this is rendered as a text image instead of being
# sent as text (0 / negative disables the image path).
DEFAULT_TEXT_IMAGE_THRESHOLD = 150

# Primary font, followed by a fallback chain: Noto Sans CJK (primary) →
# WenQuanYi (CJK backup) → GNU Unifont (covers essentially the whole
# Unicode BMP + upper planes).
_TEXT_IMAGE_FALLBACK_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/unifont/unifont.otf",
    "/usr/share/fonts/opentype/unifont/unifont_upper.otf",
]
_TEXT_IMAGE_WIDTH = 720
_TEXT_IMAGE_FONT_SIZE = 26
_TEXT_IMAGE_MARGIN = 28


def _ttc_sc_index(path: str) -> int:
    """Find the Simplified-Chinese face index inside a .ttc collection."""
    from fontTools.ttLib import TTFont

    for i in range(64):
        try:
            tt = TTFont(path, fontNumber=i, lazy=True)
        except Exception:
            break
        fam = tt["name"].getDebugName(16) or tt["name"].getDebugName(1) or ""
        if "SC" in fam:
            return i
    return 0


def _build_font_chain(size: int):
    """Return [(PIL font, cmap codepoint set)] ordered by preference."""
    from fontTools.ttLib import TTFont
    from PIL import ImageFont

    chain = []
    seen = set()
    for path in _TEXT_IMAGE_FALLBACK_FONTS:
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            index = _ttc_sc_index(path) if path.endswith(".ttc") else 0
            pil_font = ImageFont.truetype(path, size, index=index)
            tt = TTFont(path, fontNumber=index, lazy=True)
            cmap = set(tt.getBestCmap().keys())
            chain.append((pil_font, cmap))
        except Exception as e:
            logger.warning("[onebot] font load failed %s: %s", path, e)
    return chain


def render_text_image(text: str) -> bytes:
    """Render *text* onto a white PNG (black text) and return PNG bytes.

    Uses a font fallback chain (Noto CJK → WenQuanYi → Unifont) so every
    glyph renders — no tofu boxes. Wraps per-character at the width cap;
    explicit newlines are preserved.
    """
    from PIL import Image, ImageDraw

    chain = _build_font_chain(_TEXT_IMAGE_FONT_SIZE)
    if not chain:
        raise RuntimeError("no usable fonts for text-image rendering")
    primary_font = chain[0][0]
    max_width = _TEXT_IMAGE_WIDTH - 2 * _TEXT_IMAGE_MARGIN

    # Resolve every char to a font; unmapped chars become '?'.
    # Explicit newlines are kept as line breaks.
    lines: List[str] = []
    cur = ""
    cur_width = 0.0
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            cur_width = 0.0
            continue
        font = None
        for pf, cmap in chain:
            if ord(ch) in cmap:
                font = pf
                break
        if font is None:
            ch = "?"
            font = primary_font
        w = font.getlength(ch)
        if cur and cur_width + w > max_width:
            lines.append(cur)
            cur = ch
            cur_width = w
        else:
            cur += ch
            cur_width += w
    if cur:
        lines.append(cur)

    ascent, descent = primary_font.getmetrics()
    line_h = int((ascent + descent) * 1.3)
    height = 2 * _TEXT_IMAGE_MARGIN + max(len(lines), 1) * line_h
    img = Image.new("RGB", (_TEXT_IMAGE_WIDTH, height), "white")
    draw = ImageDraw.Draw(img)

    y = _TEXT_IMAGE_MARGIN
    for line in lines:
        # Draw the line in runs of equal font so fallback glyphs mix cleanly.
        x = float(_TEXT_IMAGE_MARGIN)
        i = 0
        while i < len(line):
            ch = line[i]
            font = None
            for pf, cmap in chain:
                if ord(ch) in cmap:
                    font = pf
                    break
            if font is None:
                font = primary_font
            j = i + 1
            while j < len(line):
                c2 = line[j]
                f2 = None
                for pf, cmap in chain:
                    if ord(c2) in cmap:
                        f2 = pf
                        break
                if f2 is None:
                    f2 = primary_font
                if f2 is not font:
                    break
                j += 1
            run = line[i:j]
            draw.text((x, y), run, font=font, fill="black")
            x += font.getlength(run)
            i = j
        y += line_h

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _split_reply(content: str, limit: int = DEFAULT_SPLIT_LENGTH) -> List[str]:
    """Split long content into ≤limit-char chunks at sentence boundaries.

    Prefers the nearest sentence-ending punctuation inside the window;
    falls back to a hard cut at the limit only when a chunk has no
    boundary at all (keeps progress guaranteed).
    """
    if not content:
        return []
    if len(content) <= limit:
        return [content]
    parts: List[str] = []
    start = 0
    n = len(content)
    while start < n:
        end = min(start + limit, n)
        if end >= n:
            parts.append(content[start:])
            break
        window = content[start:end]
        cut = -1
        for i in range(len(window) - 1, -1, -1):
            if window[i] in _SENTENCE_BOUNDS:
                cut = i
                break
        if cut == -1:
            # No sentence boundary in the window — hard cut to stay bounded.
            cut = limit - 1
        parts.append(content[start : start + cut + 1])
        start = start + cut + 1
    # Trim stray spaces but keep sentence-boundary newlines intact.
    return [p.rstrip(" \t") for p in parts if p.strip(" \t")]

# A few common QQ faces → emoji; anything else collapses to [表情].
_FACE_EMOJI = {
    "0": "😊", "1": "😄", "2": "😁", "3": "😆", "4": "😅", "5": "🤣",
    "14": "😏", "21": "😳", "74": "😪", "107": "🐶", "108": "🐱",
    "110": "👍", "111": "👎", "116": "🎉", "171": "🍺", "173": "👌",
}


def _inline_markdown(text: str) -> str:
    """Strip inline Markdown from a single line (QQ shows raw syntax)."""
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text)
    text = re.sub(r"_{3}(.+?)_{3}", r"\1", text)
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"[\1]", text)
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)
    return text


def strip_markdown(text: str) -> str:
    """Convert Markdown to clean QQ-friendly plain text.

    QQ does not render Markdown — raw ``**bold**`` / ``## heading`` would
    appear as literal characters. Common constructs are converted to
    readable Unicode equivalents; fenced code blocks keep their contents.
    """
    lines = text.splitlines()
    out: List[str] = []
    in_code = False
    code_lang = ""
    code_lines: List[str] = []

    for line in lines:
        fence = re.match(r"^(`{3,}|~{3,})(.*)", line.strip())
        if fence:
            if not in_code:
                in_code = True
                code_lang = fence.group(2).strip()
                code_lines = []
            else:
                in_code = False
                label = f"[{code_lang}]" if code_lang else "[代码]"
                out.append(f"┌─{label}─")
                out.extend("│ " + cl for cl in code_lines)
                out.append("└──────")
                code_lines = []
            continue
        if in_code:
            code_lines.append(line)
            continue

        h = re.match(r"^(#{1,6})\s+(.*)", line)
        if h:
            level, title = len(h.group(1)), h.group(2).strip()
            title = _inline_markdown(title)
            out.append(f"【{title}】" if level <= 2 else f"▌ {title}")
            continue

        if re.match(r"^\s*[-*_]{3,}\s*$", line):
            out.append("────────────────")
            continue

        bq = re.match(r"^>\s?(.*)", line)
        if bq:
            out.append("「" + _inline_markdown(bq.group(1)) + "」")
            continue

        ul = re.match(r"^(\s*)[-*+]\s+(.*)", line)
        if ul:
            indent = len(ul.group(1)) // 2
            out.append("  " * indent + "• " + _inline_markdown(ul.group(2)))
            continue

        ol = re.match(r"^(\s*)(\d+)[.)]\s+(.*)", line)
        if ol:
            indent = len(ol.group(1)) // 2
            out.append("  " * indent + ol.group(2) + ". " + _inline_markdown(ol.group(3)))
            continue

        if re.match(r"^\s*\|", line):
            if re.match(r"^\s*\|[\s\-:|]+\|\s*$", line):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            out.append("  ".join(_inline_markdown(c) for c in cells if c))
            continue

        out.append(_inline_markdown(line))

    # Flush an unclosed fenced code block (LLM output truncated mid-block).
    if in_code:
        label = f"[{code_lang}]" if code_lang else "[代码]"
        out.append(f"┌─{label}─")
        out.extend("│ " + cl for cl in code_lines)
        out.append("└──────")

    return "\n".join(out).strip()


def _build_chat_id(message_type: str, id_: Any) -> str:
    """Canonical chat_id used by the session store and outbound sends."""
    prefix = "group" if message_type == "group" else "private"
    return f"{prefix}:{id_}"


def _split_chat_id(chat_id: str) -> Tuple[str, str]:
    """Return (kind, target) — kind is 'private' or 'group'."""
    if ":" in chat_id:
        kind, target = chat_id.split(":", 1)
        return kind, target
    # Bare numeric id → treat as private (user) chat.
    return "private", chat_id


class OneBotAdapter(BasePlatformAdapter):
    """QQ via OneBot 11 (NapCat etc.)."""

    # Per-message cap (QQ ~4000 chars). The gateway reads this class
    # attribute to split long responses into multiple messages.
    MAX_MESSAGE_LENGTH: int = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("onebot"))
        extra = config.extra or {}
        self._mode = str(extra.get("mode", "reverse")).strip().lower() or "reverse"
        self._host = str(extra.get("host", "127.0.0.1"))
        try:
            self._port = int(extra.get("port", DEFAULT_PORT))
        except (TypeError, ValueError):
            self._port = DEFAULT_PORT
        self._url = str(extra.get("url", "ws://127.0.0.1:3001"))
        self._access_token = str(extra.get("access_token", "") or "").strip()
        self._bot_qq = str(extra.get("bot_qq", "") or "").strip()
        try:
            self._split_length = int(extra.get("split_length", DEFAULT_SPLIT_LENGTH))
        except (TypeError, ValueError):
            self._split_length = DEFAULT_SPLIT_LENGTH
        if self._split_length <= 0:
            self._split_length = DEFAULT_SPLIT_LENGTH
        try:
            self._text_image_threshold = int(
                extra.get("text_image_threshold", DEFAULT_TEXT_IMAGE_THRESHOLD)
            )
        except (TypeError, ValueError):
            self._text_image_threshold = DEFAULT_TEXT_IMAGE_THRESHOLD
        try:
            self._image_max_size = int(extra.get("image_max_size", 1536))
        except (TypeError, ValueError):
            self._image_max_size = 1536
        self._require_mention = bool(extra.get("require_mention", True))
        self._dm_policy = str(extra.get("dm_policy", "open")).strip().lower()
        self._group_policy = str(extra.get("group_policy", "open")).strip().lower()
        self._allow_from = {str(v) for v in (extra.get("allow_from") or [])}
        self._group_allow_from = {str(v) for v in (extra.get("group_allow_from") or [])}

        # Runtime state
        self._ws: Optional[Any] = None  # live OneBot connection (read/write)
        self._self_id: Optional[str] = None  # bot's own QQ, learned from events
        self._pending_actions: Dict[str, asyncio.Future] = {}
        self._runner: Optional[Any] = None  # reverse-mode web runner
        self._site: Optional[Any] = None
        self._forward_session: Optional[Any] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._stopping = False
        self._last_event_ts = 0.0

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not AIOHTTP_AVAILABLE:
            logger.error("[onebot] aiohttp unavailable — cannot start adapter")
            return False
        if not self._acquire_platform_lock("onebot", self._mode, "OneBot mode"):
            return False
        self._stopping = False
        try:
            if self._mode == "forward":
                await self._connect_forward_once()
            else:
                await self._start_reverse_server()
            self._mark_connected()
            return True
        except Exception as e:
            logger.error("[onebot] connect failed: %s", e)
            self._mark_disconnected()
            return False

    async def disconnect(self) -> None:
        self._stopping = True
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        if self._forward_session is not None:
            try:
                await self._forward_session.close()
            except Exception:
                pass
            self._forward_session = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
            self._runner = None
            self._site = None
        for fut in self._pending_actions.values():
            if not fut.done():
                fut.set_exception(ConnectionError("OneBot adapter disconnected"))
        self._pending_actions.clear()
        self._mark_disconnected()

    # -- reverse mode --------------------------------------------------

    async def _start_reverse_server(self) -> None:
        app = web.Application()
        app.router.add_get("/ws", self._handle_reverse_ws)
        app.router.add_get("/onebot", self._handle_reverse_ws)
        app.router.add_get("/", self._handle_reverse_ws)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        logger.info(
            "[onebot] reverse WS server listening on ws://%s:%s/ws (NapCat ws-reverse → this URL)",
            self._host, self._port,
        )

    async def _handle_reverse_ws(self, request: web.Request) -> web.WebSocketResponse:
        # Optional auth: NapCat sends `Authorization: Bearer <token>` when an
        # access token is configured on its side.
        if self._access_token:
            auth = request.headers.get("Authorization", "")
            expected = f"Bearer {self._access_token}"
            if auth != expected:
                logger.warning("[onebot] reverse WS auth rejected")
                return web.Response(status=401, text="unauthorized")
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self._ws = ws
        logger.info("[onebot] NapCat connected via reverse WS")
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    self._handle_frame(data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.warning("[onebot] reverse WS error: %s", ws.exception())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[onebot] reverse WS loop ended: %s", e)
        finally:
            if self._ws is ws:
                self._ws = None
            logger.info("[onebot] NapCat disconnected (reverse WS)")
        return ws

    # -- forward mode ---------------------------------------------------

    async def _connect_forward_once(self) -> None:
        headers = {"Authorization": f"Bearer {self._access_token}"} if self._access_token else {}
        session = aiohttp.ClientSession()
        try:
            ws = await session.ws_connect(
                self._url, headers=headers, heartbeat=30,
                timeout=aiohttp.ClientWSTimeout(ws_close=10.0),
            )
        except Exception:
            await session.close()
            raise
        self._forward_session = session
        self._ws = ws
        logger.info("[onebot] forward WS connected to %s", self._url)
        self._reader_task = asyncio.create_task(self._forward_read_loop(ws))

    async def _forward_read_loop(self, ws) -> None:
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    self._handle_frame(data)
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[onebot] forward WS read error: %s", e)
        finally:
            if self._ws is ws:
                self._ws = None
            logger.info("[onebot] forward WS closed")
            if not self._stopping:
                self._reconnect_task = asyncio.create_task(self._forward_reconnect())

    async def _forward_reconnect(self) -> None:
        for delay in RECONNECT_BACKOFF * (MAX_RECONNECT_ATTEMPTS // len(RECONNECT_BACKOFF) + 1):
            if self._stopping:
                return
            await asyncio.sleep(delay)
            if self._stopping:
                return
            try:
                await self._connect_forward_once()
                logger.info("[onebot] forward WS reconnected")
                return
            except Exception as e:
                logger.warning("[onebot] forward WS reconnect failed: %s", e)

    # ------------------------------------------------------------------
    # Frame handling (shared by both transports)
    # ------------------------------------------------------------------

    def _handle_frame(self, data: dict) -> None:
        """Route an inbound OneBot frame: action response or event."""
        echo = data.get("echo")
        if echo is not None:
            fut = self._pending_actions.get(str(echo))
            if fut is not None and not fut.done():
                fut.set_result(data)
            return
        post_type = data.get("post_type")
        if post_type == "meta_event":
            self._learn_self_id(data.get("self_id"))
            return
        if post_type == "message":
            asyncio.create_task(self._process_message(data))
        # notice / request events are intentionally ignored for now.

    def _learn_self_id(self, self_id) -> None:
        if self_id is None:
            return
        sid = str(self_id)
        if self._self_id is None or self._self_id == sid:
            self._self_id = sid

    # ------------------------------------------------------------------
    # Inbound messages
    # ------------------------------------------------------------------

    async def _process_message(self, data: dict) -> None:
        try:
            message_type = data.get("message_type", "")
            user_id = str(data.get("user_id", "") or "")
            self._learn_self_id(data.get("self_id"))
            raw = data.get("raw_message", "") or ""
            sender = data.get("sender") or {}
            nickname = sender.get("card") or sender.get("nickname") or ""

            if message_type == "private":
                if not self._dm_allowed(user_id):
                    return
                chat_id = _build_chat_id("private", user_id)
                chat_type = "dm"
            elif message_type == "group":
                group_id = str(data.get("group_id", "") or "")
                if not self._group_allowed(group_id):
                    return
                if self._require_mention and not self._is_mentioned(raw):
                    return
                chat_id = _build_chat_id("group", group_id)
                chat_type = "group"
            else:
                return

            text, media_urls, media_types = await self._parse_content(raw)

            has_voice = any(t.startswith("audio/") for t in media_types)
            if not text and not media_urls:
                return
            event = MessageEvent(
                text=text,
                message_type=(
                    MessageType.VOICE
                    if has_voice and not text
                    else MessageType.PHOTO
                    if media_urls
                    else MessageType.TEXT
                ),
                user_id=user_id,
                user_name=nickname or user_id,
                source=SessionSource(
                    platform=Platform("onebot"),
                    chat_id=chat_id,
                    chat_type=chat_type,
                    user_id=user_id,
                    user_name=nickname or user_id,
                ),
                raw_message=data,
                message_id=str(data.get("message_id") or ""),
                media_urls=media_urls,
                media_types=media_types,
            )
            await self.handle_message(event)
        except Exception as e:
            logger.error("[onebot] failed processing message: %s", e, exc_info=True)

    def _dm_allowed(self, user_id: str) -> bool:
        policy = self._dm_policy
        if policy == "disabled":
            return False
        if policy == "allowlist":
            return user_id in self._allow_from
        return True

    def _group_allowed(self, group_id: str) -> bool:
        policy = self._group_policy
        if policy == "disabled":
            return False
        if policy == "allowlist":
            return group_id in self._group_allow_from
        return True

    def _is_mentioned(self, raw: str) -> bool:
        """True when the bot was @'d or the message replies to something.

        With an unknown bot id and no configured bot_qq we fail closed in
        group chats (no accidental reply to every message).
        """
        self_id = self._self_id or self._bot_qq or ""
        if self_id and f"[CQ:at,qq={self_id}]" in raw:
            return True
        if "[CQ:reply" in raw:
            # Replying to a message is an explicit nudge — treat as a call.
            return True
        return False

    async def _parse_content(self, raw: str) -> Tuple[str, List[str], List[str]]:
        """Convert a raw CQ-encoded message to (text, media_paths, media_types).

        Images with a downloadable url are fetched into a temp dir so the
        vision tool can read them; voice clips are downloaded and converted
        to 16 kHz mono WAV so the gateway's STT pipeline can transcribe
        them. Replies/at are normalized to plain text.
        """
        image_urls = _CQ_IMAGE_RE.findall(raw)
        record_urls = _CQ_RECORD_RE.findall(raw)
        media_urls: List[str] = []
        media_types: List[str] = []
        for url in image_urls:
            try:
                path = await self._download_image(url)
                if path:
                    media_urls.append(path)
                    media_types.append("image")
            except Exception as e:
                logger.debug("[onebot] image download failed: %s", e)
        for url in record_urls:
            try:
                path = await self._download_audio(url)
                if path:
                    media_urls.append(path)
                    media_types.append("audio/wav")
            except Exception as e:
                logger.debug("[onebot] voice download failed: %s", e)

        text = _CQ_IMAGE_RE.sub(lambda m: "[图片]", raw)
        text = _CQ_IMAGE_NOURL_RE.sub("[图片]", text)
        text = _CQ_RECORD_RE.sub(lambda m: "[语音]", text)
        text = _CQ_RECORD_NOURL_RE.sub("[语音]", text)
        text = _CQ_AT_RE.sub(
            lambda m: "@" + ("全体成员" if m.group(1) == "all" else m.group(1)), text
        )
        text = _CQ_REPLY_RE.sub(lambda m: "", text)
        text = _CQ_FACE_RE.sub(lambda m: _FACE_EMOJI.get(m.group(1), "[表情]"), text)
        text = _CQ_ANY_RE.sub("", text)
        text = re.sub(r"[ \t]+", " ", text).strip()
        return text, media_urls, media_types

    async def _download_image(self, url: str) -> Optional[str]:
        if not url or url.lower().startswith("base64://"):
            return None
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        timeout = aiohttp.ClientTimeout(total=20.0)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
        if len(data) > IMAGE_MAX_BYTES:
            logger.debug("[onebot] image too large, skipping (%d bytes)", len(data))
            return None
        ext = mimetypes.guess_extension(resp.headers.get("Content-Type", "")) or ".jpg"
        if ext == ".jpe":
            ext = ".jpg"
        tmp = Path(tempfile.gettempdir()) / "hermes_onebot"
        tmp.mkdir(exist_ok=True)
        path = tmp / f"img_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{ext}"
        path.write_bytes(data)
        # Shrink oversized images BEFORE the LLM sees them — high-res QQ
        # photos make vision calls slow or time out entirely.
        shrunk = await asyncio.to_thread(self._shrink_image, path)
        return shrunk or str(path)

    def _shrink_image(self, path: Path) -> Optional[str]:
        """Downscale an image to ≤ `image_max_size` px on its long edge.

        Returns the new path when the image was resized, None when it was
        already small enough (or processing failed — the caller keeps the
        original). Animated GIFs collapse to their first frame, which is
        fine for vision analysis.
        """
        max_size = self._image_max_size
        if max_size <= 0:
            return None
        try:
            from PIL import Image

            img = Image.open(path)
            img.load()
        except Exception as e:
            logger.debug("[onebot] image open failed, keeping original: %s", e)
            return None
        try:
            if max(img.size) <= max_size:
                return None
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            out = path.with_suffix(".png" if img.mode == "RGBA" else ".jpg")
            if img.mode == "RGBA":
                img.save(out, format="PNG", optimize=True)
            else:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                img.save(out, format="JPEG", quality=85)
            return str(out)
        except Exception as e:
            logger.debug("[onebot] image shrink failed, keeping original: %s", e)
            return None

    async def _download_audio(self, url: str) -> Optional[str]:
        """Download a voice clip and convert it to 16 kHz mono WAV.

        QQ voice messages are silk/amr — the gateway STT pipeline expects
        a standard audio file, so ffmpeg converts it (best effort; returns
        None on any failure and the caller degrades to a [语音] marker).
        """
        if not url or url.lower().startswith("base64://"):
            return None
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        timeout = aiohttp.ClientTimeout(total=25.0)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
        if len(data) > AUDIO_MAX_BYTES:
            logger.debug("[onebot] voice too large, skipping (%d bytes)", len(data))
            return None

        tmp = Path(tempfile.gettempdir()) / "hermes_onebot"
        tmp.mkdir(exist_ok=True)
        stem = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        in_path = tmp / f"voice_{stem}.bin"
        out_path = tmp / f"voice_{stem}.wav"
        in_path.write_bytes(data)
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-y",
                "-i",
                str(in_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                str(out_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout=20.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return None
            if rc != 0 or not out_path.exists():
                logger.debug("[onebot] ffmpeg voice conversion failed (rc=%s)", rc)
                return None
            return str(out_path)
        except (OSError, FileNotFoundError) as e:
            logger.debug("[onebot] ffmpeg unavailable: %s", e)
            return None
        finally:
            in_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._ws is None:
            return SendResult(
                success=False,
                error="OneBot WebSocket not connected",
                retryable=True,
            )
        kind, target = _split_chat_id(chat_id)
        try:
            params: Dict[str, Any] = {}
            if kind == "group":
                params["group_id"] = int(target)
            else:
                params["user_id"] = int(target)

            # No [CQ:reply] prefix — user prefers plain replies without a
            # quoted reference to the triggering message.
            media_segments: List[Dict[str, Any]] = []
            if metadata:
                media_files = metadata.get("media_files") or []
                for f in media_files:
                    b64 = await self._file_to_base64(f)
                    if b64:
                        media_segments.append(
                            {"type": "image", "data": {"file": f"base64://{b64}"}}
                        )

            # QQ doesn't render Markdown — convert common syntax to readable
            # plain text BEFORE splitting/rendering so chunks and text images
            # are both clean.
            content = strip_markdown(content or "")

            parts = _split_reply(content or "", self._split_length)

            # Long content → single text-image message instead of text.
            if (
                self._text_image_threshold > 0
                and len(content or "") > self._text_image_threshold
            ):
                try:
                    png_bytes = await asyncio.to_thread(render_text_image, content)
                    b64 = base64.b64encode(png_bytes).decode("ascii")
                    image_params = dict(params)
                    image_params["message"] = [
                        {"type": "image", "data": {"file": f"base64://{b64}"}}
                    ] + media_segments
                    data = await self._call_action("send_msg", image_params)
                    mid = data.get("message_id")
                    return SendResult(
                        success=True, message_id=str(mid) if mid is not None else None
                    )
                except Exception as e:
                    logger.warning(
                        "[onebot] text-image render failed (%s) — falling back to text chunks", e
                    )

            if not parts and not media_segments:
                return SendResult(success=True, message_id=None)

            last_message_id: Optional[str] = None
            for idx, part in enumerate(parts):
                chunk_segments: List[Dict[str, Any]] = [
                    {"type": "text", "data": {"text": part}}
                ]
                # Attach media to the final chunk.
                if idx == len(parts) - 1:
                    chunk_segments.extend(media_segments)
                chunk_params = dict(params)
                chunk_params["message"] = chunk_segments
                data = await self._call_action("send_msg", chunk_params)
                mid = data.get("message_id")
                if mid is not None:
                    last_message_id = str(mid)

            return SendResult(success=True, message_id=last_message_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[onebot] send failed: %s", e)
            return SendResult(success=False, error=str(e), retryable=True)

    async def _file_to_base64(self, path: str) -> Optional[str]:
        try:
            p = Path(path)
            if not p.exists() or p.stat().st_size > IMAGE_MAX_BYTES:
                return None
            data = await asyncio.to_thread(p.read_bytes)
            return base64.b64encode(data).decode("ascii")
        except Exception as e:
            logger.debug("[onebot] media read failed: %s", e)
            return None

    async def _call_action(self, action: str, params: Dict[str, Any], timeout: float = ACTION_TIMEOUT) -> Dict[str, Any]:
        ws = self._ws
        if ws is None:
            raise ConnectionError("OneBot WebSocket not connected")
        echo = str(uuid.uuid4())
        fut: "asyncio.Future[Dict[str, Any]]" = asyncio.get_event_loop().create_future()
        self._pending_actions[echo] = fut
        try:
            await ws.send_str(json.dumps({"action": action, "params": params, "echo": echo}))
            resp = await asyncio.wait_for(fut, timeout)
        finally:
            self._pending_actions.pop(echo, None)
        if resp.get("status") != "ok":
            raise RuntimeError(
                f"OneBot action {action} failed: {resp.get('wording') or resp.get('retcode')}"
            )
        return resp.get("data") or {}

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        kind, target = _split_chat_id(chat_id)
        return {"name": chat_id, "type": "group" if kind == "group" else "dm"}

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Show the QQ "typing…" bubble — NapCat `set_input_status`.

        QQ only surfaces input status for C2C (private) chats; group
        chats have no typing indicator, so those are skipped. Failures
        are silently ignored (the gateway bounds this call itself).
        """
        if self._ws is None:
            return
        kind, target = _split_chat_id(chat_id)
        if kind != "private":
            return
        try:
            await self._call_action(
                "set_input_status",
                {"user_id": target, "event_type": 1},
                timeout=3.0,
            )
        except Exception as e:
            logger.debug("[onebot] send_typing failed: %s", e)

    async def stop_typing(self, chat_id: str) -> None:
        """Clear the QQ input-status bubble (private chats only)."""
        if self._ws is None:
            return
        kind, target = _split_chat_id(chat_id)
        if kind != "private":
            return
        try:
            await self._call_action(
                "set_input_status",
                {"user_id": target, "event_type": 0},
                timeout=3.0,
            )
        except Exception as e:
            logger.debug("[onebot] stop_typing failed: %s", e)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def check_requirements() -> bool:
    return AIOHTTP_AVAILABLE


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    mode = str(extra.get("mode", "reverse")).strip().lower()
    if mode not in ("reverse", "forward"):
        return False
    return True


def _build_adapter(config):
    return OneBotAdapter(config)


def _is_connected(config) -> bool:
    # Best-effort: connected when config is present; real state lives on the
    # live adapter (checked by the gateway runner).
    return validate_config(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="onebot",
        label="QQ (OneBot)",
        adapter_factory=_build_adapter,
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=_is_connected,
        install_hint="OneBot needs a running NapCat / Lagrange / LLOneBot instance",
        allowed_users_env="ONEBOT_ALLOWED_USERS",
        allow_all_env="ONEBOT_ALLOW_ALL_USERS",
        cron_deliver_env_var="ONEBOT_HOME_CHANNEL",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🐧",
        platform_hint=(
            "You are chatting via QQ (OneBot). Plain text only — no markdown. "
            "Group chats: users @you or reply to you. Keep replies concise."
        ),
    )
