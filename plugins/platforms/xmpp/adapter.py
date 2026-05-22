"""XMPP platform adapter (Hermes plugin).

Connects to a self-hosted or hosted XMPP server using slixmpp.  Inbound
``<message type='chat'>`` and addressed MUC messages are delivered to the
Hermes agent; outbound replies are sent back as the same stanza type.

Plugin layout follows the SimpleX / IRC / LINE conventions:

* ``check_requirements`` gates on ``XMPP_JID`` + ``XMPP_PASSWORD`` + the
  ``slixmpp`` package importing.  Missing any of the three keeps the
  platform out of ``get_connected_platforms()`` so the gateway never
  instantiates the adapter.
* ``_env_enablement`` seeds ``PlatformConfig.extra`` from env so
  ``hermes gateway status`` reflects env-only setups without spinning
  up the XMPP client.
* ``_standalone_send`` opens an ephemeral session for cron jobs that run
  separately from the gateway.

Required environment variables:
    XMPP_JID              Bare JID the bot logs in as
    XMPP_PASSWORD         Password for the bot account

Optional environment variables:
    XMPP_HOST                 Server host override (default: SRV lookup)
    XMPP_PORT                 Server port override (default: 5222)
    XMPP_FORCE_STARTTLS       Require STARTTLS (default: true)
    XMPP_NICKNAME             MUC nickname (default: JID local part)
    XMPP_ROOMS                Comma-separated MUC JIDs to join
    XMPP_ALLOWED_USERS        Comma-separated bare JIDs allowlist
    XMPP_ALLOW_ALL_USERS      true = bypass the allowlist (dev only)
    XMPP_HOME_CHANNEL         Default target for cron deliveries
    XMPP_HOME_CHANNEL_NAME    Display name for the home channel
    XMPP_UPLOAD_SERVICE       Pin the XEP-0363 upload service JID
                              (default: SRV-style auto-discovery)
    XMPP_HTML_FORMATTING      Emit XEP-0071 XHTML-IM dual body (default: true)

The ``slixmpp`` Python package is imported lazily — the plugin stays
importable for discovery and setup-time prompts even when slixmpp is not
installed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH = 16_000  # XMPP has no hard limit; cap stanza body sanely
DEFAULT_PORT = 5222
CONNECT_TIMEOUT_SECONDS = 20.0
DISCONNECT_TIMEOUT_SECONDS = 5.0
MUC_PREFIX = "muc:"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in _TRUTHY


def _parse_comma_list(value: str) -> List[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _strip_resource(jid: str) -> str:
    """Return the bare JID (``user@server``) by dropping any ``/resource``."""
    return (jid or "").split("/", 1)[0]


def _strip_markdown(text: str) -> str:
    """Convert markdown to plain text for the XMPP ``<body>`` fallback.

    XHTML-IM aware clients render the parallel ``<html>`` element produced
    by :func:`_format_html`; this strip keeps the plain-body fallback
    readable on clients that do not.
    """
    # Bold / italic / strikethrough
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    # Code blocks and inline code (keep content, drop fences/ticks)
    text = re.sub(r"```\w*\n?", "", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Headings
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Images, then links (images first so ![] is consumed before [])
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\2", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text


def _strip_control_chars(text: str) -> str:
    """Strip the few characters XML 1.0 cannot encode in a message body.

    Slixmpp serializes stanzas as XML 1.0; raw NUL or stray vertical-tab
    bytes get rejected by the underlying parser.  CR/LF are valid in a
    body, but normalizing them to ``\\n`` keeps logs readable.
    """
    cleaned = []
    for ch in text or "":
        cp = ord(ch)
        if cp == 0x00:
            continue
        if cp < 0x20 and ch not in ("\t", "\n", "\r"):
            continue
        cleaned.append(ch)
    return "".join(cleaned).replace("\r\n", "\n").replace("\r", "\n")


def _split_message(content: str, max_bytes: int) -> List[str]:
    """Split ``content`` on paragraph boundaries to keep each chunk under
    ``max_bytes`` of UTF-8.  No code-block awareness — XMPP renders plain
    text bodies and the stripper above has already removed fences."""
    chunks: List[str] = []
    for paragraph in content.split("\n"):
        if not paragraph.strip():
            continue
        while True:
            data = paragraph.encode("utf-8")
            if len(data) <= max_bytes:
                chunks.append(paragraph)
                break
            # Binary search for the largest prefix that fits within max_bytes
            low, high, best = 1, len(paragraph), 0
            while low <= high:
                mid = (low + high) // 2
                if len(paragraph[:mid].encode("utf-8")) <= max_bytes:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            split_at = best
            space = paragraph.rfind(" ", 0, split_at)
            if space > split_at // 3:
                split_at = space
            chunks.append(paragraph[:split_at].rstrip())
            paragraph = paragraph[split_at:].lstrip()
    return chunks if chunks else [""]


# ---------------------------------------------------------------------------
# XHTML-IM helpers (XEP-0071)
# ---------------------------------------------------------------------------

# XHTML-IM allows a restricted subset of HTML. We target what the popular
# mobile + desktop clients (Conversations, Dino, Gajim, Profanity) all
# render: p, br, strong, em, code, pre, a, ul, ol, li, blockquote, span.
_XHTML_IM_NS = "http://www.w3.org/1999/xhtml"


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _sanitize_link_url(url: str) -> str:
    """Reject dangerous schemes (``javascript:``, ``data:``, ``vbscript:``)."""
    stripped = (url or "").strip()
    scheme = stripped.split(":", 1)[0].lower().strip() if ":" in stripped else ""
    if scheme in {"javascript", "data", "vbscript"}:
        return ""
    return stripped.replace('"', "&quot;")


def _markdown_to_xhtml_im(text: str) -> str:
    """Convert markdown to a safe XHTML-IM subset.

    Uses the ``markdown`` library when installed (matches the Matrix
    adapter's path; ``markdown`` is pinned for the ``matrix`` extra and
    is therefore commonly available in Hermes installs). Falls back to a
    regex pipeline when not.

    Returns an XHTML fragment suitable for assignment to
    ``msg['html']['body']`` in slixmpp. The caller wraps it in a single
    ``<p>...</p>`` envelope so the result is well-formed even when the
    markdown source is empty.
    """
    text = text or ""
    if not text.strip():
        return ""

    try:
        import markdown as _md  # type: ignore[import-untyped]

        md = _md.Markdown(extensions=["fenced_code", "nl2br", "sane_lists"])
        if "html_block" in md.preprocessors:
            md.preprocessors.deregister("html_block")
        html = md.convert(text)
        md.reset()
        # ``markdown`` wraps single-paragraph output in a stray <p>; XMPP
        # clients render dual ``<p>`` blocks with a leading blank line, so
        # strip the wrapper when there is only one.
        if html.count("<p>") == 1:
            html = html.replace("<p>", "", 1).replace("</p>", "", 1)
        return html.strip()
    except ImportError:
        return _markdown_to_xhtml_im_fallback(text)


def _markdown_to_xhtml_im_fallback(text: str) -> str:
    """Regex Markdown-to-XHTML-IM for installs without the ``markdown`` package."""
    placeholders: List[str] = []

    def _protect(html_fragment: str) -> str:
        placeholders.append(html_fragment)
        return f"\x00P{len(placeholders) - 1}\x00"

    # Fenced code blocks: ```lang\n...\n```
    result = re.sub(
        r"```[a-zA-Z0-9_+\-]*\n?(.*?)```",
        lambda m: _protect(f"<pre><code>{_html_escape(m.group(1))}</code></pre>"),
        text,
        flags=re.DOTALL,
    )
    # Inline code
    result = re.sub(
        r"`([^`\n]+)`",
        lambda m: _protect(f"<code>{_html_escape(m.group(1))}</code>"),
        result,
    )
    # Images (extract URL only — XHTML-IM ``<img>`` is poorly supported)
    result = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        lambda m: _protect(_html_escape(m.group(2))),
        result,
    )
    # Links: [text](url)
    result = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: _protect(
            '<a href="{}">{}</a>'.format(
                _sanitize_link_url(m.group(2)),
                _html_escape(m.group(1)),
            )
        )
        if _sanitize_link_url(m.group(2))
        else _protect(_html_escape(m.group(1))),
        result,
    )
    # HTML-escape the rest
    parts = re.split(r"(\x00P\d+\x00)", result)
    for idx, part in enumerate(parts):
        if not part.startswith("\x00P"):
            parts[idx] = _html_escape(part)
    result = "".join(parts)

    # Inline emphasis on the escaped+protected string
    result = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", result)
    result = re.sub(r"__(.+?)__", r"<strong>\1</strong>", result)
    result = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", result)
    result = re.sub(r"(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)", r"<em>\1</em>", result)

    # Block-level: headings → <strong>; lists; blockquotes; paragraphs
    lines = result.split("\n")
    out_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        hdr = re.match(r"^(#{1,6})\s+(.+)$", line)
        if hdr:
            out_lines.append(f"<p><strong>{hdr.group(2).strip()}</strong></p>")
            i += 1
            continue
        if re.match(r"^(\s*[-*]\s+)", line):
            bullets = []
            while i < len(lines) and re.match(r"^(\s*[-*]\s+)", lines[i]):
                bullets.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            out_lines.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                i += 1
            out_lines.append("<ol>" + "".join(f"<li>{it}</li>" for it in items) + "</ol>")
            continue
        if line.startswith("&gt; ") or line == "&gt;":
            bq = []
            while i < len(lines) and (lines[i].startswith("&gt; ") or lines[i] == "&gt;"):
                bq.append(lines[i][5:] if lines[i].startswith("&gt; ") else "")
                i += 1
            out_lines.append("<blockquote>" + "<br>".join(bq) + "</blockquote>")
            continue
        if not line.strip():
            i += 1
            continue
        # Plain paragraph
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not _starts_block(lines[i]):
            para.append(lines[i])
            i += 1
        out_lines.append("<p>" + "<br>".join(para) + "</p>")

    rendered = "".join(out_lines)
    # Restore protected fragments
    rendered = re.sub(
        r"\x00P(\d+)\x00",
        lambda m: placeholders[int(m.group(1))],
        rendered,
    )
    return rendered


def _starts_block(line: str) -> bool:
    return bool(
        re.match(r"^(#{1,6})\s+", line)
        or re.match(r"^(\s*[-*]\s+)", line)
        or re.match(r"^\s*\d+\.\s+", line)
        or line.startswith("&gt; ")
        or line == "&gt;"
    )


# ---------------------------------------------------------------------------
# Media / file helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic"})
_AUDIO_EXTS = frozenset({".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac", ".aac"})
_VIDEO_EXTS = frozenset({".mp4", ".webm", ".mov", ".mkv", ".avi"})

_CONTENT_TYPE_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".pdf": "application/pdf",
    ".zip": "application/zip",
    ".txt": "text/plain",
}


def _content_type_for(path: str) -> str:
    _, dot, ext = path.rpartition(".")
    if not dot:
        return "application/octet-stream"
    return _CONTENT_TYPE_BY_EXT.get("." + ext.lower(), "application/octet-stream")


class _UploadUnavailable(Exception):
    """Raised when HTTP Upload is not usable for the current request."""


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class XMPPAdapter(BasePlatformAdapter):
    """XMPP adapter using slixmpp's asyncio ClientXMPP.

    Instantiated by the ``adapter_factory`` passed to
    ``ctx.register_platform()`` in :func:`register`.
    """

    def __init__(self, config: PlatformConfig, **kwargs):
        platform = Platform("xmpp")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}

        self.jid = (os.getenv("XMPP_JID") or extra.get("jid") or "").strip()
        self.password = os.getenv("XMPP_PASSWORD") or extra.get("password", "")
        self.host = (os.getenv("XMPP_HOST") or extra.get("host") or "").strip() or None

        port_value = os.getenv("XMPP_PORT") or extra.get("port") or DEFAULT_PORT
        try:
            self.port = int(port_value)
        except (TypeError, ValueError):
            self.port = DEFAULT_PORT

        self.force_starttls = _parse_bool(
            os.getenv("XMPP_FORCE_STARTTLS", extra.get("force_starttls", True)),
            default=True,
        )

        bare = _strip_resource(self.jid)
        local_part = bare.split("@", 1)[0] if "@" in bare else (bare or "hermes")
        self.nickname = (
            os.getenv("XMPP_NICKNAME")
            or extra.get("nickname")
            or local_part
        ).strip() or "hermes"

        rooms_raw = os.getenv("XMPP_ROOMS") or extra.get("rooms", "")
        if isinstance(rooms_raw, list):
            self.rooms = [str(r).strip() for r in rooms_raw if str(r).strip()]
        else:
            self.rooms = _parse_comma_list(rooms_raw)

        self.max_message_length = int(extra.get("max_message_length") or MAX_MESSAGE_LENGTH)

        self.upload_service = (
            os.getenv("XMPP_UPLOAD_SERVICE") or extra.get("upload_service") or ""
        ).strip() or None
        self.html_formatting = _parse_bool(
            os.getenv("XMPP_HTML_FORMATTING", extra.get("html_formatting", True)),
            default=True,
        )

        # Runtime state
        self._client = None  # slixmpp.ClientXMPP
        self._lock_key: Optional[str] = None
        self._connected_event = asyncio.Event()
        self._closing = False
        # Upload service discovery: None = not probed yet, False = unavailable,
        # str = service JID. Cached on session_start to avoid per-send discovery.
        self._upload_service_resolved: Any = None

    @property
    def name(self) -> str:
        return "XMPP"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        if not self.jid or not self.password:
            logger.error("XMPP: XMPP_JID and XMPP_PASSWORD are required")
            self._set_fatal_error(
                "config_missing",
                "XMPP_JID and XMPP_PASSWORD must be set",
                retryable=False,
            )
            return False

        try:
            import slixmpp  # noqa: F401
        except ImportError:
            logger.error("XMPP: 'slixmpp' package not installed. Run: pip install slixmpp")
            self._set_fatal_error(
                "missing_dependency",
                "slixmpp package is not installed",
                retryable=False,
            )
            return False

        # Prevent two profiles from logging in as the same JID.
        try:
            from gateway.status import acquire_scoped_lock
            if not acquire_scoped_lock("xmpp", _strip_resource(self.jid)):
                logger.error("XMPP: %s already in use by another profile", _strip_resource(self.jid))
                self._set_fatal_error(
                    "lock_conflict",
                    "XMPP identity in use by another profile",
                    retryable=False,
                )
                return False
            self._lock_key = _strip_resource(self.jid)
        except ImportError:
            self._lock_key = None  # status module not available (tests)

        from slixmpp import ClientXMPP

        client = ClientXMPP(self.jid, self.password)
        client.register_plugin("xep_0030")  # Service Discovery
        client.register_plugin("xep_0199")  # XMPP Ping
        client.register_plugin("xep_0045")  # Multi-User Chat
        client.register_plugin("xep_0066")  # Out-of-band data (attachment URLs)
        client.register_plugin("xep_0363")  # HTTP File Upload
        if self.html_formatting:
            client.register_plugin("xep_0071")  # XHTML-IM

        # TLS toggles. ``enable_starttls`` controls whether slixmpp negotiates
        # STARTTLS during stream negotiation; ``enable_direct_tls`` covers
        # direct-TLS ports (5223). When ``XMPP_FORCE_STARTTLS=false`` we
        # disable both so plaintext local-loopback dev servers work.
        client.enable_starttls = self.force_starttls
        client.enable_direct_tls = self.force_starttls
        if not self.force_starttls:
            logger.warning(
                "XMPP: TLS disabled (XMPP_FORCE_STARTTLS=false). "
                "Only do this on trusted local networks."
            )

        client.add_event_handler("session_start", self._on_session_start)
        client.add_event_handler("message", self._on_message)
        client.add_event_handler("groupchat_message", self._on_groupchat_message)
        client.add_event_handler("disconnected", self._on_disconnected)
        client.add_event_handler("failed_auth", self._on_failed_auth)

        self._client = client
        self._closing = False
        self._connected_event = asyncio.Event()

        host = self.host
        port = self.port if host else None  # slixmpp does SRV when host omitted
        client.connect(host=host, port=port)

        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=CONNECT_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.error("XMPP: connect timed out after %.0fs", CONNECT_TIMEOUT_SECONDS)
            await self._safe_disconnect()
            self._set_fatal_error("connect_timeout", "XMPP session_start not received", retryable=True)
            return False

        if self.has_fatal_error:
            return False

        self._mark_connected()
        logger.info("XMPP: connected as %s", _strip_resource(self.jid))
        return True

    async def disconnect(self) -> None:
        self._closing = True
        if self._lock_key:
            try:
                from gateway.status import release_scoped_lock
                release_scoped_lock("xmpp", self._lock_key)
            except Exception:
                pass
            self._lock_key = None
        await self._safe_disconnect()
        self._client = None
        self._mark_disconnected()

    async def _safe_disconnect(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            future = client.disconnect(wait=DISCONNECT_TIMEOUT_SECONDS)
            if future is not None:
                await asyncio.wait_for(asyncio.shield(asyncio.ensure_future(future)), timeout=DISCONNECT_TIMEOUT_SECONDS + 2.0)
        except (asyncio.TimeoutError, Exception):
            # Best-effort: never raise out of disconnect.
            pass

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_session_start(self, event):
        client = self._client
        if client is None:
            return
        try:
            client.send_presence()
            await client.get_roster()
        except Exception:
            logger.exception("XMPP: error during session start")

        # Join any pre-configured MUC rooms.
        muc = client.plugin.get("xep_0045")
        if muc is not None and self.rooms:
            for room in self.rooms:
                try:
                    muc.join_muc(room, self.nickname, wait=False)
                    logger.info("XMPP: joining MUC %s as %s", room, self.nickname)
                except Exception:
                    logger.exception("XMPP: failed to join MUC %s", room)

        # Pre-warm the HTTP Upload service so the first attachment send does
        # not pay the discovery round-trip. Failure (no upload component on
        # the server) becomes a sticky False so subsequent send_image_file /
        # send_document calls fall back to text immediately.
        await self._resolve_upload_service()

        self._connected_event.set()

    async def _resolve_upload_service(self) -> None:
        client = self._client
        if client is None:
            self._upload_service_resolved = False
            return
        if self.upload_service:
            self._upload_service_resolved = self.upload_service
            logger.info("XMPP: using pinned upload service %s", self.upload_service)
            return
        try:
            upload = client.plugin["xep_0363"]
            iq = await upload.find_upload_service()
            if iq is None:
                logger.info(
                    "XMPP: no HTTP Upload service advertised by %s; "
                    "media will be sent as text", _strip_resource(self.jid).split("@", 1)[-1]
                )
                self._upload_service_resolved = False
                return
            service_jid = str(iq["from"]) if iq.get("from") else True
            self._upload_service_resolved = service_jid
            logger.info("XMPP: discovered HTTP Upload service: %s", service_jid)
        except Exception:
            logger.warning(
                "XMPP: HTTP Upload discovery failed; media will be sent as text",
                exc_info=True,
            )
            self._upload_service_resolved = False

    def _on_failed_auth(self, event):
        logger.error("XMPP: authentication failed for %s", _strip_resource(self.jid))
        self._set_fatal_error("auth_failed", "XMPP authentication failed", retryable=False)
        # Release the wait_for in connect()
        self._connected_event.set()

    async def _on_disconnected(self, event):
        if self._closing:
            return
        if self.is_connected:
            logger.warning("XMPP: lost connection, marking disconnected")
            self._set_fatal_error("connection_lost", "XMPP connection closed unexpectedly", retryable=True)
            try:
                await self._notify_fatal_error()
            except Exception:
                logger.exception("XMPP: error notifying fatal error")

    async def _on_message(self, msg):
        """Handle a 1:1 ``<message type='chat'|'normal'>``."""
        if msg["type"] not in ("chat", "normal"):
            return
        text = msg.get("body") or ""
        if not text.strip():
            return  # ignore chat-state / typing-only stanzas

        sender_jid_full = str(msg["from"])
        sender_jid = _strip_resource(sender_jid_full)
        if sender_jid == _strip_resource(self.jid):
            return  # own echo (shouldn't happen for type=chat but be defensive)

        await self._dispatch(
            text=text,
            chat_id=sender_jid,
            chat_name=sender_jid,
            chat_type="dm",
            user_id=sender_jid,
            user_name=sender_jid,
            message_id=str(msg.get("id") or ""),
        )

    async def _on_groupchat_message(self, msg):
        """Handle a MUC ``<message type='groupchat'>``."""
        if msg["type"] != "groupchat":
            return
        text = msg.get("body") or ""
        if not text.strip():
            return

        room_jid = _strip_resource(str(msg["from"]))
        sender_nick = msg.get("mucnick") or ""
        if not sender_nick or sender_nick == self.nickname:
            return  # own echo / system message

        # Only respond when addressed by our nickname (matches IRC behavior).
        prefixes = (
            f"{self.nickname}:",
            f"{self.nickname},",
            f"{self.nickname} ",
            f"@{self.nickname} ",
        )
        addressed = False
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                addressed = True
                break
        if not addressed:
            return

        chat_id = f"{MUC_PREFIX}{room_jid}"
        await self._dispatch(
            text=text,
            chat_id=chat_id,
            chat_name=room_jid,
            chat_type="group",
            user_id=sender_nick,
            user_name=sender_nick,
            message_id=str(msg.get("id") or ""),
        )

    async def _dispatch(
        self,
        *,
        text: str,
        chat_id: str,
        chat_name: str,
        chat_type: str,
        user_id: str,
        user_name: str,
        message_id: str = "",
    ) -> None:
        if not self._message_handler:
            return
        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=user_id,
            user_name=user_name,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=message_id or str(int(time.time() * 1000)),
            timestamp=datetime.now(tz=timezone.utc),
        )
        await self.handle_message(event)

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
        if self._client is None:
            return SendResult(success=False, error="Not connected")

        target, mtype = self._resolve_target(chat_id)
        if not target:
            return SendResult(success=False, error=f"Invalid XMPP target: {chat_id!r}")

        plain = _strip_control_chars(_strip_markdown(content or ""))
        if not plain.strip():
            return SendResult(success=True, message_id=str(int(time.time() * 1000)))

        chunks = _split_message(plain, self.max_message_length)
        # XHTML-IM only on single-chunk messages: splitting HTML across
        # stanzas at arbitrary byte boundaries would produce invalid
        # fragments, and most XMPP clients reject those.
        emit_html = (
            self.html_formatting
            and len(chunks) == 1
            and "xep_0071" in self._client.plugin
        )
        html_body = ""
        if emit_html:
            html_body = _markdown_to_xhtml_im(content or "")
            # Skip the html element when the rendered fragment is identical
            # to the plain body (no formatting present): nothing to gain.
            if html_body and html_body.strip() == _html_escape(plain).strip():
                html_body = ""

        last_id = ""
        try:
            for idx, chunk in enumerate(chunks):
                stanza = self._client.make_message(mto=target, mbody=chunk, mtype=mtype)
                if emit_html and idx == 0 and html_body:
                    try:
                        stanza["html"]["body"] = html_body
                    except Exception:
                        logger.debug("XMPP: failed to attach XHTML-IM body", exc_info=True)
                stanza.send()
                last_id = str(int(time.time() * 1000))
                await asyncio.sleep(0.05)
        except Exception as e:
            return SendResult(success=False, error=str(e))
        return SendResult(success=True, message_id=last_id)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """XEP-0085 chat states — best-effort, ignored if the peer does not support it."""
        if self._client is None:
            return
        target, mtype = self._resolve_target(chat_id)
        if not target or mtype == "groupchat":
            return  # composing in MUC is noisy and rarely useful
        try:
            stanza = self._client.make_message(mto=target, mtype=mtype)
            stanza["chat_state"] = "composing"
            stanza.send()
        except Exception:
            # Chat-state plugin may not be registered; that's fine.
            pass

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an HTTPS image URL with an OOB extension.

        Clients that understand XEP-0066 (Conversations, Gajim, Dino) treat
        ``<x xmlns='jabber:x:oob'><url>…</url></x>`` as an inline
        attachment and render the image; clients that ignore the extension
        still see the URL in ``<body>`` and can tap through.
        """
        return await self._send_url_with_oob(
            chat_id=chat_id,
            url=image_url,
            caption=caption,
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._upload_then_send(chat_id, image_path, caption=caption)

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        # The gateway routes local-file animated GIFs through
        # ``send_image_file`` already; this path is for hosted URLs.
        return await self._send_url_with_oob(chat_id, animation_url, caption=caption)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._upload_then_send(chat_id, audio_path, caption=caption)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._upload_then_send(chat_id, video_path, caption=caption)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._upload_then_send(chat_id, file_path, caption=caption)

    async def _send_url_with_oob(
        self,
        chat_id: str,
        url: str,
        *,
        caption: Optional[str] = None,
    ) -> SendResult:
        """Send a stanza whose ``<body>`` is *url* plus an OOB extension."""
        if self._client is None:
            return SendResult(success=False, error="Not connected")
        target, mtype = self._resolve_target(chat_id)
        if not target:
            return SendResult(success=False, error=f"Invalid XMPP target: {chat_id!r}")
        clean_url = _strip_control_chars(url or "").strip()
        if not clean_url:
            return SendResult(success=False, error="Empty URL")

        body_lines = []
        if caption:
            body_lines.append(_strip_control_chars(_strip_markdown(caption)))
        body_lines.append(clean_url)
        body = "\n".join(line for line in body_lines if line.strip())

        try:
            stanza = self._client.make_message(mto=target, mbody=body, mtype=mtype)
            # Attach OOB only for http(s) URLs — file:// or relative paths
            # would let a misconfigured caller leak local paths to peers.
            if clean_url.startswith(("https://", "http://")):
                try:
                    stanza["oob"]["url"] = clean_url
                except Exception:
                    logger.debug("XMPP: failed to attach OOB extension", exc_info=True)
            stanza.send()
        except Exception as e:
            return SendResult(success=False, error=str(e))
        return SendResult(success=True, message_id=str(int(time.time() * 1000)))

    async def _upload_then_send(
        self,
        chat_id: str,
        path: str,
        *,
        caption: Optional[str] = None,
    ) -> SendResult:
        """Upload *path* via HTTP Upload then send the resulting URL via OOB.

        Falls back to a text message describing the file when the upload
        component is unavailable or the upload fails.
        """
        if self._client is None:
            return SendResult(success=False, error="Not connected")
        if not path or not os.path.isfile(path):
            return SendResult(success=False, error=f"File not found: {path!r}")

        try:
            url = await self._upload_file(path)
        except _UploadUnavailable as exc:
            logger.info("XMPP: falling back to text for %s — %s", path, exc)
            return await self._send_upload_fallback(chat_id, path, caption, str(exc))

        return await self._send_url_with_oob(chat_id, url, caption=caption)

    async def _upload_file(self, path: str) -> str:
        """Upload a local file via XEP-0363 and return the HTTPS URL.

        Raises :class:`_UploadUnavailable` when the upload service is not
        available, the file exceeds the server's advertised limit, or the
        HTTP PUT itself fails. Callers turn the exception into a text
        fallback so a missing upload component never breaks message
        delivery.
        """
        if self._upload_service_resolved is None:
            await self._resolve_upload_service()
        if not self._upload_service_resolved:
            raise _UploadUnavailable("no HTTP Upload service available")

        client = self._client
        if client is None:
            raise _UploadUnavailable("not connected")

        from pathlib import Path as _Path

        size = os.path.getsize(path)
        content_type = _content_type_for(path)
        domain_kwarg: Dict[str, Any] = {}
        if isinstance(self._upload_service_resolved, str):
            domain_kwarg["domain"] = self._upload_service_resolved

        try:
            upload = client.plugin["xep_0363"]
            url = await upload.upload_file(
                _Path(path),
                size=size,
                content_type=content_type,
                **domain_kwarg,
            )
        except Exception as exc:  # UploadServiceNotFound / FileTooBig / HTTPError
            raise _UploadUnavailable(str(exc)) from exc
        return str(url)

    async def _send_upload_fallback(
        self,
        chat_id: str,
        path: str,
        caption: Optional[str],
        reason: str,
    ) -> SendResult:
        """Send a text bubble when an upload could not happen."""
        filename = os.path.basename(path) or path
        body = f"📎 {filename} (upload failed: {reason})"
        if caption:
            body = f"{caption}\n{body}"
        return await self.send(chat_id, body)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        if chat_id.startswith(MUC_PREFIX):
            jid = chat_id[len(MUC_PREFIX):]
            return {"chat_id": chat_id, "type": "group", "name": jid}
        return {"chat_id": chat_id, "type": "dm", "name": chat_id}

    def _resolve_target(self, chat_id: str) -> tuple[str, str]:
        """Return ``(jid, mtype)`` for a ``chat_id`` string."""
        if not chat_id:
            return "", "chat"
        if chat_id.startswith(MUC_PREFIX):
            return chat_id[len(MUC_PREFIX):], "groupchat"
        return chat_id, "chat"


# ---------------------------------------------------------------------------
# Plugin entry-point hooks
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """Plugin gate: require XMPP_JID + XMPP_PASSWORD + slixmpp importable."""
    if not os.getenv("XMPP_JID") or not os.getenv("XMPP_PASSWORD"):
        return False
    try:
        import slixmpp  # noqa: F401
    except ImportError:
        return False
    return True


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    jid = os.getenv("XMPP_JID") or extra.get("jid", "")
    password = os.getenv("XMPP_PASSWORD") or extra.get("password", "")
    return bool(jid and password)


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> Optional[dict]:
    jid = (os.getenv("XMPP_JID") or "").strip()
    if not jid:
        return None
    if not (os.getenv("XMPP_PASSWORD") or "").strip():
        return None

    seed: dict = {"jid": jid}
    host = (os.getenv("XMPP_HOST") or "").strip()
    if host:
        seed["host"] = host
    port = (os.getenv("XMPP_PORT") or "").strip()
    if port:
        try:
            seed["port"] = int(port)
        except ValueError:
            pass
    force_tls = (os.getenv("XMPP_FORCE_STARTTLS") or "").strip()
    if force_tls:
        seed["force_starttls"] = force_tls.lower() in _TRUTHY
    nickname = (os.getenv("XMPP_NICKNAME") or "").strip()
    if nickname:
        seed["nickname"] = nickname
    rooms = (os.getenv("XMPP_ROOMS") or "").strip()
    if rooms:
        seed["rooms"] = _parse_comma_list(rooms)
    upload_service = (os.getenv("XMPP_UPLOAD_SERVICE") or "").strip()
    if upload_service:
        seed["upload_service"] = upload_service
    html_formatting = (os.getenv("XMPP_HTML_FORMATTING") or "").strip()
    if html_formatting:
        seed["html_formatting"] = html_formatting.lower() in _TRUTHY

    home = (os.getenv("XMPP_HOME_CHANNEL") or "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": (os.getenv("XMPP_HOME_CHANNEL_NAME") or "").strip() or home,
        }
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Open an ephemeral XMPP session, send, and disconnect.

    Used by ``tools/send_message_tool._send_via_adapter`` when the gateway
    runner is not in this process (e.g. ``hermes cron`` running separately).
    Without this hook, ``deliver=xmpp`` cron jobs fail with "No live
    adapter for platform".

    ``thread_id``, ``media_files`` and ``force_document`` are accepted for
    signature parity with other plugin standalone senders but are not
    meaningful here: XMPP has no native thread primitive and media
    delivery would require HTTP Upload (XEP-0363) and a reachable
    upload component, which a one-shot session cannot guarantee.
    """
    del thread_id, media_files, force_document

    try:
        import slixmpp
    except ImportError:
        return {"error": "slixmpp not installed. Run: pip install slixmpp"}

    extra = getattr(pconfig, "extra", {}) or {}
    jid = (os.getenv("XMPP_JID") or extra.get("jid") or "").strip()
    password = os.getenv("XMPP_PASSWORD") or extra.get("password", "")
    if not jid or not password:
        return {"error": "XMPP standalone send: XMPP_JID and XMPP_PASSWORD are required"}

    raw_target = (chat_id or extra.get("home_channel", {}).get("chat_id") or "").strip()
    if not raw_target:
        return {"error": "XMPP standalone send: missing target JID"}
    if any(ch in raw_target for ch in ("\r", "\n", "\x00", " ")):
        return {"error": "XMPP standalone send: chat_id contains illegal characters"}

    if raw_target.startswith(MUC_PREFIX):
        target = raw_target[len(MUC_PREFIX):]
        mtype = "groupchat"
    else:
        target = raw_target
        mtype = "chat"

    host = (os.getenv("XMPP_HOST") or extra.get("host") or "").strip() or None
    port_raw = os.getenv("XMPP_PORT") or extra.get("port") or DEFAULT_PORT
    try:
        port = int(port_raw)
    except (TypeError, ValueError):
        port = DEFAULT_PORT
    force_tls = _parse_bool(
        os.getenv("XMPP_FORCE_STARTTLS", extra.get("force_starttls", True)),
        default=True,
    )

    body = _strip_control_chars(_strip_markdown(message or ""))
    if not body.strip():
        return {"error": "XMPP standalone send: empty message after stripping"}

    # Distinct resource so we don't collide with a live gateway adapter on
    # the same identity.
    full_jid = f"{_strip_resource(jid)}/hermes-cron-{int(time.time() * 1000) % 100000}"

    client = slixmpp.ClientXMPP(full_jid, password)
    client.register_plugin("xep_0030")
    client.register_plugin("xep_0045")
    client.enable_starttls = force_tls
    client.enable_direct_tls = force_tls

    ready = asyncio.Event()
    error: Dict[str, str] = {}

    async def _on_start(_event):
        try:
            client.send_presence()
            if mtype == "groupchat":
                muc = client.plugin["xep_0045"]
                nick = (os.getenv("XMPP_NICKNAME") or extra.get("nickname") or jid.split("@", 1)[0]).strip() or "hermes"
                muc.join_muc(target, nick, wait=False)
                # Give the server a moment to process the JOIN before we
                # send to a +n-style room. Real protocol ack would be
                # ``groupchat_subject``; a short sleep avoids us holding
                # the event loop here just to listen for one stanza.
                await asyncio.sleep(0.8)
            for chunk in _split_message(body, MAX_MESSAGE_LENGTH):
                client.send_message(mto=target, mbody=chunk, mtype=mtype)
                await asyncio.sleep(0.05)
        except Exception as exc:  # noqa: BLE001
            error["msg"] = str(exc)
        finally:
            ready.set()

    def _on_auth_fail(_event):
        error["msg"] = "authentication failed"
        ready.set()

    client.add_event_handler("session_start", _on_start)
    client.add_event_handler("failed_auth", _on_auth_fail)

    try:
        client.connect(host=host, port=port if host else None)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"XMPP standalone connect failed: {exc}"}

    try:
        await asyncio.wait_for(ready.wait(), timeout=CONNECT_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        try:
            client.disconnect(wait=1.0)
        except Exception:
            pass
        return {"error": "XMPP standalone send: timed out before session_start"}

    try:
        fut = client.disconnect(wait=DISCONNECT_TIMEOUT_SECONDS)
        if fut is not None:
            await asyncio.wait_for(asyncio.shield(asyncio.ensure_future(fut)), timeout=DISCONNECT_TIMEOUT_SECONDS + 2.0)
    except Exception:
        pass

    if error:
        return {"error": f"XMPP standalone send failed: {error['msg']}"}
    return {"success": True, "platform": "xmpp", "chat_id": chat_id, "message_id": str(int(time.time() * 1000))}


def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes setup gateway`` → XMPP."""
    try:
        from hermes_cli.setup import (
            prompt,
            prompt_yes_no,
            save_env_value,
            get_env_value,
            print_header,
            print_info,
            print_warning,
            print_success,
        )
    except ImportError:
        print()
        print("hermes_cli.setup not available; set XMPP_* vars manually in ~/.hermes/.env")
        return

    print_header("XMPP")
    existing = get_env_value("XMPP_JID")
    if existing:
        print_info(f"XMPP: already configured (JID: {existing})")
        if not prompt_yes_no("Reconfigure XMPP?", False):
            return

    print_info("Connect Hermes to an XMPP server (Prosody, ejabberd, hosted, etc.).")
    print_info("   Requires the slixmpp Python package: pip install slixmpp")
    print()

    jid = prompt("Bot JID (e.g. hermes@chat.example.org)", default=existing or "")
    if not jid:
        print_warning("JID is required — skipping XMPP setup")
        return
    save_env_value("XMPP_JID", jid.strip())

    password = prompt("Password for this account", password=True)
    if password:
        save_env_value("XMPP_PASSWORD", password)
    elif not get_env_value("XMPP_PASSWORD"):
        print_warning("Password is required — skipping XMPP setup")
        return

    host = prompt("Server host (blank = SRV lookup on the JID domain)",
                  default=get_env_value("XMPP_HOST") or "")
    save_env_value("XMPP_HOST", host.strip())

    port = prompt("Server port (default 5222)", default=get_env_value("XMPP_PORT") or "")
    if port:
        try:
            save_env_value("XMPP_PORT", str(int(port)))
        except ValueError:
            print_warning("Invalid port — using default 5222")

    use_tls = prompt_yes_no("Require STARTTLS (recommended)?", True)
    save_env_value("XMPP_FORCE_STARTTLS", "true" if use_tls else "false")

    nickname = prompt(
        "MUC nickname (blank = JID local part)",
        default=get_env_value("XMPP_NICKNAME") or "",
    )
    save_env_value("XMPP_NICKNAME", nickname.strip())

    rooms = prompt(
        "MUC rooms to auto-join (comma-separated, or blank)",
        default=get_env_value("XMPP_ROOMS") or "",
    )
    save_env_value("XMPP_ROOMS", rooms.replace(" ", ""))

    print()
    print_info("🔒 Access control: restrict which JIDs can DM the bot")
    allow_all = prompt_yes_no("Allow any JID to talk to the bot?", False)
    if allow_all:
        save_env_value("XMPP_ALLOW_ALL_USERS", "true")
        save_env_value("XMPP_ALLOWED_USERS", "")
        print_warning("⚠️  Open access — any JID can command the bot.")
    else:
        save_env_value("XMPP_ALLOW_ALL_USERS", "false")
        allowed = prompt(
            "Allowed bare JIDs (comma-separated, blank to deny everyone)",
            default=get_env_value("XMPP_ALLOWED_USERS") or "",
        )
        save_env_value("XMPP_ALLOWED_USERS", allowed.replace(" ", ""))
        if allowed:
            print_success("Allowlist configured")
        else:
            print_info("No JIDs allowed — the bot will ignore all DMs until you add some.")

    print()
    print_success("XMPP configuration saved to ~/.hermes/.env")
    print_info("Restart the gateway for changes to take effect: hermes gateway restart")


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="xmpp",
        label="XMPP",
        adapter_factory=lambda cfg: XMPPAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["XMPP_JID", "XMPP_PASSWORD"],
        install_hint="pip install slixmpp   # XMPP adapter requires slixmpp",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="XMPP_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="XMPP_ALLOWED_USERS",
        allow_all_env="XMPP_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="✉️",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via XMPP. JIDs look like email addresses "
            "(user@server). MUC rooms are identified by a ``muc:`` "
            "prefix; in rooms, users address you by your nickname. "
            "You can use markdown sparingly (bold, italic, code, links, "
            "lists) — the adapter renders it via XHTML-IM for clients "
            "that support it, with a plain-text fallback for the rest. "
            "Files attach natively when the server has an HTTP Upload "
            "component: include MEDIA:/absolute/path/to/file in your "
            "response and the adapter uploads it and sends the URL with "
            "an OOB attachment hint. If the server has no upload "
            "component the message arrives as text describing the file."
        ),
    )
