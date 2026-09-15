"""MAX platform adapter (Hermes plugin).

MAX (max.ru) — Russian messenger. This adapter connects via Long Polling
(GET /updates with marker cursor) and sends replies via POST /messages.

Why Long Polling instead of webhooks:
- MAX requires HTTPS + Russian Trusted CA certs for webhooks (since 2025-05-25)
- Users behind NAT (typical RU ISP, e.g. Rostelecom) have no public endpoint
- Long Polling works from anywhere: the adapter polls platform-api2.max.ru

TLS: MAX uses Russian Trusted Root CA. Set MAX_CA_CERT_PATH to the PEM file,
or the adapter falls back to the default trust store. The cert is
auto-downloaded from the official source (gu-st.ru) on first use, with a
bounded timeout and PEM structure validation.

Security note: trusting the Russian Trusted Root CA is inherent to using the
MAX platform (its API is served with certificates chained to that CA). The
cert is fetched over HTTPS from the official Ministry source (gu-st.ru) and
validated as a PEM certificate before being used.
"""

import asyncio
import json
import logging
import os
import random
import re
import ssl
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_url,
)

from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret

logger = logging.getLogger(__name__)

API_HOST = "platform-api2.max.ru"
API_SCHEME = "https"
DEFAULT_POLL_TIMEOUT = 90
DEFAULT_POLL_LIMIT = 100
POLL_INTERVAL_SECONDS = 1.0  # pause between long-poll requests
MAX_MESSAGE_LENGTH = 4000  # MAX text limit
RECONNECT_BACKOFF = [1, 2, 5, 10, 30]
DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 2000
_ECHO_MARKER = "hermes-agent-max"  # appended to outgoing text for echo-loop prevention
CA_DOWNLOAD_TIMEOUT = 10  # seconds
MAX_SEND_RATE_PER_CHAT = 2.0  # MAX: max 2 messages/sec per chat
_TRUNCATION_NOTICE = "\n\n✂️ (сообщение обрезано — лимит MAX 4000 симв.)"
_MEDIA_LABELS = {"image": "Фото", "video": "Видео", "audio": "Аудио", "file": "Файл", "voice": "Голосовое"}

# ---------------------------------------------------------------------------
# Reasoning display + Markdown→HTML conversion
# ---------------------------------------------------------------------------

# Env toggle: show the gateway's 💭 Reasoning block in MAX (default: hide).
SHOW_REASONING_ENV = "MAX_SHOW_REASONING"

# Matches the reasoning block the gateway prepends when display.show_reasoning
# is on. Styles (see gateway/run.py): fenced  💭 **Reasoning:**\n```...```,
# blockquote "> 💭 **Reasoning:**\n> ...", subtext "-# 💭 Reasoning\n-# ...".
_REASONING_FENCE_RE = re.compile(
    r"💭 \*\*Reasoning:\*\*[ \t]*\n+```[^\n]*\n.*?\n```[ \t]*\n?", re.DOTALL
)
_REASONING_QUOTE_RE = re.compile(r"(?:^|> )💭 \*\*Reasoning:\*\*[ \t]*\n(?:> .*\n?)+", re.MULTILINE)
_REASONING_SUBTEXT_RE = re.compile(r"-# ?💭 Reasoning[ \t]*\n(?:-# .*\n?)+", re.MULTILINE)


def _show_reasoning_requested() -> bool:
    """True when MAX_SHOW_REASONING opts back into the reasoning block."""
    return os.getenv(SHOW_REASONING_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def strip_reasoning_block(text: str) -> str:
    """Remove the gateway's prepended 💭 Reasoning block from a reply.

    MAX is a phone-first chat; scratch thinking reads as noise there, so the
    adapter hides it by default. Set ``MAX_SHOW_REASONING=true`` to keep it.
    Handles all three gateway render styles plus a bare <think> block.
    """
    if not isinstance(text, str):
        return text
    out = text
    for pattern in (_REASONING_FENCE_RE, _REASONING_QUOTE_RE, _REASONING_SUBTEXT_RE):
        out = pattern.sub("", out, count=1)
    # Bare <think>...</think> (some providers emit it directly in content)
    if "<think>" in out:
        out = re.sub(r"<think>.*?</think>[ \t]*\n?", "", out, count=1, flags=re.DOTALL)
    return out.lstrip() if out != text else out


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md_inline_to_html(text: str) -> str:
    """Convert inline Markdown to MAX's HTML subset.

    Field-tested against a real client (2026-08): only <u>, <mark> and the
    blockquote bar actually render; <b>/<i>/<code>/<pre> degrade to plain
    text. Mapping favours what survives:
      **bold** / __bold__   → <b>   (semantic; no visual style in client)
      *italic* / _italic_   → <i>   (semantic; no visual style in client)
      ~~strike~~            → <s>   (semantic; no visual style in client)
      `code`                → <mark> (renders as highlight)
      [label](url)          → <a href="url">label</a>
    """
    out = _escape_html(text)
    # Links first, so their URL/text isn't mangled by emphasis rules below.
    out = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r'<a href="\2">\1</a>',
        out,
    )
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out, flags=re.DOTALL)
    out = re.sub(r"__(.+?)__", r"<b>\1</b>", out, flags=re.DOTALL)
    out = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"<i>\1</i>", out)
    out = re.sub(r"(?<![\w_])_([^_\n]+?)_(?![\w_])", r"<i>\1</i>", out)
    out = re.sub(r"~~(.+?)~~", r"<s>\1</s>", out, flags=re.DOTALL)
    out = re.sub(r"`([^`\n]+?)`", r"<mark>\1</mark>", out)
    return out


def markdown_to_max_html(text: str) -> str:
    """Convert agent Markdown to MAX HTML so code blocks survive.

    MAX Markdown has no fenced code blocks — an inline `` ` `` block collapses
    newlines into spaces, which destroys multi-line code. The API's HTML mode
    supports real <pre> blocks, so: walk the text line by line, emit fenced
    regions as <pre> (language attr dropped), convert the remaining inline
    Markdown to MAX's HTML subset.
    """
    if not isinstance(text, str) or not text:
        return text

    parts: List[str] = []

    def _flush_pre() -> None:
        # Keep a blank-line boundary between surrounding text and the block.
        # Wrapped in <blockquote>: MAX renders the quote's left bar as a visual
        # frame around the code (bare <pre> is just monospace text).
        nonlocal fence_body
        escaped = _escape_html("\n".join(fence_body))
        inner = (
            f'<pre><code class="language-{fence_lang}">{escaped}</code></pre>'
            if fence_lang
            else f"<pre>{escaped}</pre>"
        )
        if parts and not parts[-1].endswith("\n"):
            parts.append("\n")
        parts.append(f"<blockquote>{inner}</blockquote>")
        parts.append("\n")
        fence_body = []

    in_fence = False
    fence_lang = ""
    fence_body: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not in_fence and stripped.startswith("```"):
            # Opening fence: ```python / ``` / ```
            in_fence = True
            fence_lang = stripped[3:].strip()
            continue
        if in_fence:
            if stripped.startswith("```"):
                # Closing fence → flush the collected body
                _flush_pre()
                in_fence = False
            else:
                fence_body.append(line)
        else:
            parts.append(_md_inline_to_html(line) + "\n")
    # Unclosed fence at EOF → still render what we collected
    if in_fence:
        _flush_pre()
    return "".join(parts).rstrip("\n")


def prepare_outgoing_text(text: str) -> tuple:
    """Prepare an agent reply for MAX delivery.

    Returns ``(payload_text, payload_format)`` — always MAX HTML. Field
    tests (2026-08) showed MAX's Markdown mode paints nothing inline at
    all (raw ``**`` and ```` ` ```` leak into the chat as literal chars),
    while HTML mode renders <u>, <mark> and the blockquote frame; the rest
    (<b>/<i>/<s>/<code>) degrades gracefully to plain text. Converting
    every message keeps markup from leaking. The gateway's 💭 Reasoning
    block is stripped unless ``MAX_SHOW_REASONING=true``.
    """
    out = strip_reasoning_block(text) if not _show_reasoning_requested() else text
    return markdown_to_max_html(out), "html"


def _find_media_url(obj: Any, depth: int = 0) -> Optional[str]:
    """Recursively find a media download URL in a MAX update.

    MAX can nest voice/audio URLs deep inside the update object
    (message.attachments[].payload.url, message.voice, body.attachments,
    or at the update root). This mirrors what clients actually receive
    without pulling in any external library.
    """
    if depth > 8 or obj is None:
        return None
    if isinstance(obj, dict):
        # type + url at the same level (typical attachment shape)
        atype = str(obj.get("type", "")).lower()
        url = obj.get("url") or obj.get("download_url") or ""
        if atype in ("voice", "audio", "video") and isinstance(url, str) and url.startswith("http"):
            return url
        # payload.url pattern
        payload = obj.get("payload")
        if isinstance(payload, dict):
            url = payload.get("url") or payload.get("download_url") or ""
            if isinstance(url, str) and url.startswith("http"):
                return url
        # recurse into known containers
        for key in ("attachments", "voice", "audio", "message", "body", "payload", "media"):
            found = _find_media_url(obj.get(key), depth + 1)
            if found:
                return found
        for val in obj.values():
            found = _find_media_url(val, depth + 1)
            if found:
                return found
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found = _find_media_url(item, depth + 1)
            if found:
                return found
    return None

_MIME_BY_EXT = {
    ".pdf": "application/pdf", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword", ".odt": "application/vnd.oasis.opendocument.text",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel", ".csv": "text/csv", ".txt": "text/plain", ".md": "text/markdown",
    ".rtf": "application/rtf", ".epub": "application/epub+zip", ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint", ".json": "application/json", ".xml": "application/xml",
    ".zip": "application/zip", ".rar": "application/vnd.rar", ".7z": "application/x-7z-compressed",
    ".tar": "application/x-tar", ".gz": "application/gzip", ".mp3": "audio/mpeg", ".wav": "audio/wav",
    ".ogg": "audio/ogg", ".m4a": "audio/mp4", ".flac": "audio/flac", ".opus": "audio/opus",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
    ".webm": "video/webm", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp", ".svg": "image/svg+xml",
    ".html": "text/html", ".htm": "text/html", ".log": "text/plain", ".py": "text/x-python",
}


def _mime_for_ext(ext: str, fallback_type: str = "file") -> str:
    """Map a file extension to a MIME type (lowercased, with dot).

    Falls back to a type-appropriate default when the extension is unknown.
    """
    e = ext.lower() if ext else ""
    if e in _MIME_BY_EXT:
        return _MIME_BY_EXT[e]
    return {
        "video": "video/mp4", "audio": "audio/mpeg", "file": "application/octet-stream",
        "image": "image/jpeg",
    }.get(fallback_type, "application/octet-stream")


def _get_scoped_secret(name, default=None):
    """Scope-aware credential read (same pattern as ntfy/slack adapters)."""
    try:
        val = _scoped_get_secret(name, default)
    except _UnscopedSecretError:
        val = os.getenv(name)
    return val if val is not None else default


def _is_valid_pem_cert(path: str) -> bool:
    """Validate that a file is a PEM certificate (not HTML/truncated junk)."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        if b"-----BEGIN CERTIFICATE-----" not in data:
            return False
        # Try loading it as an X.509 certificate
        ssl.PEM_cert_to_DER_cert(data.decode("utf-8", errors="replace"))
        return True
    except Exception:
        return False


def _default_ca_path() -> Optional[str]:
    """Return the Russian Trusted Root CA path if present (honors HERMES_HOME).

    Auto-downloads the official Russian Trusted Root CA from gu-st.ru on
    first use if no local copy exists, so a fresh install works out of the
    box without manual certificate setup. Download uses a bounded timeout
    and the file is validated as a PEM certificate.
    """
    hermes_home = os.getenv("HERMES_HOME", "") or os.path.expanduser("~/.hermes")
    candidates = [
        os.getenv("MAX_CA_CERT_PATH", "").strip(),
        os.path.join(hermes_home, "max", "certs", "russian_trusted_root_ca_pem.crt"),
        os.path.join(os.path.dirname(__file__), "certs", "russian_trusted_root_ca_pem.crt"),
        os.path.expanduser("~/.hermes/max/certs/russian_trusted_root_ca_pem.crt"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and _is_valid_pem_cert(c):
            return c

    # Auto-download the official cert (gu-st.ru is the Ministry's official source)
    try:
        import urllib.request

        dest_dir = os.path.join(hermes_home, "max", "certs")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, "russian_trusted_root_ca_pem.crt")
        url = "https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt"
        with urllib.request.urlopen(url, timeout=CA_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        if _is_valid_pem_cert(dest):
            logger.info("[max] Auto-downloaded Russian Trusted Root CA to %s", dest)
            return dest
        logger.warning("[max] Downloaded file is not a valid PEM certificate; removing")
        try:
            os.remove(dest)
        except OSError:
            pass
    except Exception as e:
        logger.warning("[max] Auto-download of Russian Trusted Root CA failed: %s", e)
    return None


def check_requirements() -> bool:
    """Check whether the MAX adapter is installable and minimally configured."""
    if not HTTPX_AVAILABLE:
        return False
    token = os.getenv("MAX_BOT_TOKEN", "").strip()
    return bool(token)


def validate_config(config) -> bool:
    """Validate that the configured MAX platform has a token set."""
    extra = getattr(config, "extra", {}) or {}
    token = extra.get("token") or os.getenv("MAX_BOT_TOKEN", "")
    return bool(token)


def is_connected(config) -> bool:
    """Check whether MAX is configured (env or config.yaml)."""
    extra = getattr(config, "extra", {}) or {}
    token = os.getenv("MAX_BOT_TOKEN") or extra.get("token", "")
    return bool(token)


class MaxAdapter(BasePlatformAdapter):
    """MAX adapter — Long Polling inbound, POST /messages outbound."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        platform = Platform("max")
        super().__init__(config=config, platform=platform)

        extra = config.extra or {}
        self._token: str = extra.get("token") or _get_scoped_secret("MAX_BOT_TOKEN", "")
        self._marker: Optional[int] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._http_client: Optional["httpx.AsyncClient"] = None
        self._seen_messages: Dict[str, float] = {}
        self._ca_path = _default_ca_path()
        self._last_user_id: str = ""
        # Bot identity (from GET /me) — used for auto-addressing detection
        self._name: str = ""
        self._username: str = ""
        self._id: str = ""
        self._description: str = ""  # bot "about"/description from MAX
        # Bot aliases — extra names the bot answers to (config.extra.bot_aliases)
        raw_aliases = extra.get("bot_aliases") or ""
        if isinstance(raw_aliases, str):
            self._aliases = [a.strip().lower() for a in raw_aliases.split(",") if a.strip()]
        else:
            self._aliases = [str(a).strip().lower() for a in raw_aliases if str(a).strip()]
        # Known chats: chat_id -> {"name": ..., "type": "dm"|"group"}
        self._known_chats: Dict[str, Dict[str, str]] = {}
        # Group allowlist: set of chat_ids the bot may serve. Empty = allow all
        # groups (behaviour preserved); non-empty = only listed chat_ids.
        raw_group_allow = extra.get("group_allowed_chats") or os.getenv("MAX_GROUP_ALLOWED_CHATS", "")
        self._group_allowed_chats: set = set()
        if raw_group_allow:
            self._group_allowed_chats = {
                str(c).strip() for c in str(raw_group_allow).split(",") if str(c).strip()
            }
        # Owner of the bot (full access). Either config.extra.owner_user_id or env.
        raw_owner = extra.get("owner_user_id") or os.getenv("MAX_OWNER_USER_ID", "")
        self._owner_user_id: str = str(raw_owner).strip() if raw_owner else ""
        # Approved group chats: set of chat_ids the bot is allowed to serve.
        # The bot notifies the owner when added to a new group and stays silent
        # until the owner approves (or the chat is in the allowlist / owner's own).
        self._approved_chats: set = set()
        raw_approved = extra.get("approved_chats") or os.getenv("MAX_APPROVED_CHATS", "")
        if raw_approved:
            self._approved_chats = {
                str(c).strip() for c in str(raw_approved).split(",") if str(c).strip()
            }
        # Group members cache: chat_id -> {user_id: ChatMember dict}
        self._members: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Timestamps of last member fetch per chat (for TTL refresh)
        self._members_fetched_at: Dict[str, float] = {}
        # How long a members snapshot is considered fresh (seconds).
        self._members_ttl = float(extra.get("members_ttl") or os.getenv("MAX_MEMBERS_TTL", "300"))
        # Health/status tracking
        self._last_poll_at: Optional[float] = None
        self._last_poll_error: Optional[str] = None
        self._last_error_at: Optional[float] = None
        # Marker persistence
        self._marker_path = os.path.join(
            os.getenv("HERMES_HOME", "") or os.path.expanduser("~/.hermes"),
            "max", "marker.json",
        )
        self._load_marker()
        # Send rate limiting: chat_id -> [timestamps]
        self._send_history: Dict[str, List[float]] = {}

    # -- Marker persistence ------------------------------------------------

    def _load_marker(self) -> None:
        """Load the last-known marker from disk (if any)."""
        try:
            if os.path.isfile(self._marker_path):
                with open(self._marker_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._marker = int(data.get("marker") or 0) or None
        except Exception as e:
            logger.debug("[%s] Could not load marker: %s", self.name, e)

    def _save_marker(self) -> None:
        """Persist the marker to disk so restarts don't replay old updates."""
        if not self._marker:
            return
        try:
            os.makedirs(os.path.dirname(self._marker_path), exist_ok=True)
            with open(self._marker_path, "w", encoding="utf-8") as f:
                json.dump({"marker": self._marker, "saved_at": time.time()}, f)
        except Exception as e:
            logger.debug("[%s] Could not save marker: %s", self.name, e)

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Start the Long Polling loop."""
        if not HTTPX_AVAILABLE:
            logger.warning("[%s] httpx not installed", self.name)
            return False
        if not self._token:
            logger.warning("[%s] MAX_BOT_TOKEN not configured", self.name)
            return False

        try:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=DEFAULT_POLL_TIMEOUT + 15, write=15.0, pool=15.0),
                verify=self._ca_path or True,
            )
            # Validate token and fetch bot info (GET /me)
            await self._fetch_bot_info()
            # Re-fetch member roles for known groups (post-restart) in parallel
            # so a fresh deploy doesn't block 200ms-per-group sequentially.
            known_groups = [cid for cid, info in self._known_chats.items() if info.get("type") == "group"]
            if known_groups:
                results = await asyncio.gather(
                    *[self._fetch_members(cid) for cid in known_groups],
                    return_exceptions=True,
                )
                for cid, res in zip(known_groups, results):
                    if isinstance(res, Exception):
                        logger.debug("[%s] member refresh failed for %s: %s", self.name, cid, res)
            # Register bot command menu (PATCH /me/commands) — best effort
            await self._register_commands()
            self._poll_task = asyncio.create_task(self._run_poll_loop())
            self._mark_connected()
            logger.info("[%s] Connected — Long Polling %s://%s/updates (marker=%s)",
                        self.name, API_SCHEME, API_HOST, self._marker)
            return True
        except Exception as e:
            logger.error("[%s] Failed to connect: %s", self.name, e)
            return False

    async def _fetch_bot_info(self) -> None:
        """GET /me — validate the token and log bot identity.

        Stores bot display name + username on the adapter so inbound group
        messages can be matched against the bot's own name/mention
        (auto-addressing detection).
        """
        if self._http_client is None:
            return
        try:
            resp = await self._http_client.get(
                f"{API_SCHEME}://{API_HOST}/me",
                headers={"Authorization": self._token},
                timeout=15.0,
            )
            if resp.status_code == 401:
                logger.error("[%s] Auth failed (401) — MAX_BOT_TOKEN invalid", self.name)
                self._set_fatal_error(
                    "max_unauthorized",
                    "MAX API rejected auth (401). Check MAX_BOT_TOKEN.",
                    retryable=False,
                )
                return
            if resp.status_code < 300:
                data = resp.json()
                bot_name = data.get("first_name") or data.get("username") or "?"
                bot_username = data.get("username") or ""
                bot_id = data.get("user_id")
                bot_desc = data.get("description") or data.get("about") or ""
                self._name = bot_name
                self._username = bot_username
                self._id = str(bot_id) if bot_id is not None else ""
                self._description = str(bot_desc).strip()
                logger.info("[%s] Authenticated as %s (id=%s, @%s)", self.name, bot_name, bot_id, bot_username)
        except Exception as e:
            logger.warning("[%s] /me check failed: %s", self.name, e)


    async def _fetch_members(self, chat_id: str) -> None:
        """GET /chats/{chatId}/members — fetch roles for this group.

        Stores user_id -> ChatMember dict in self._members[chat_id]. Used for
        role-aware context (owner/admin/member) and the approval gate.
        """
        if not self._http_client:
            return
        try:
            resp = await self._http_client.get(
                f"{API_SCHEME}://{API_HOST}/chats/{chat_id}/members",
                headers={"Authorization": self._token},
                timeout=15.0,
            )
            if resp.status_code >= 300:
                logger.warning("[%s] members HTTP %d for %s", self.name, resp.status_code, chat_id)
                return
            data = resp.json()
            members = data.get("members") or data.get("result") or []
            cache: Dict[str, Dict[str, Any]] = {}
            for m in members:
                if not isinstance(m, dict):
                    continue
                uid = str(m.get("user_id") or "")
                if uid:
                    cache[uid] = m
            if cache:
                self._members[chat_id] = cache
                self._members_fetched_at[chat_id] = time.time()
                logger.info("[%s] Cached %d members for chat %s", self.name, len(cache), chat_id)
        except Exception as e:
            logger.warning("[%s] _fetch_members failed for %s: %s", self.name, chat_id, e)

    async def _notify_owner_approval(self, chat_id: str, chat_name: str) -> None:
        """DM the bot owner: a new group added the bot — approve or deny."""
        if not self._owner_user_id or not self._http_client:
            return
        text = (
            f"⚠️ Меня добавили в группу «{chat_name}» (id {chat_id}).\n"
            f"Я буду молчать, пока ты не решишь:\n"
            f"  • /approve {chat_id} — разрешить работу в этой группе\n"
            f"  • /deny {chat_id} — отказать (я останусь, но молча)\n"
            f"Это защита: чужая группа не получит доступ к моему агенту."
        )
        try:
            payload = {"text": text, "format": "markdown"}
            resp = await self._http_client.post(
                f"{API_SCHEME}://{API_HOST}/messages",
                params={"user_id": self._owner_user_id},
                content=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": self._token, "Content-Type": "application/json"},
                timeout=15.0,
            )
            if resp.status_code >= 300:
                logger.warning("[%s] approval notify HTTP %d", self.name, resp.status_code)
        except Exception as e:
            logger.warning("[%s] approval notify failed: %s", self.name, e)

    async def _ensure_members_fresh(self, chat_id: str) -> None:
        """Refresh the members cache for a group if it's stale (TTL).

        Bounded by a short timeout so a slow MAX API call can never delay
        the bot's reply — if the fetch hangs, we keep the stale cache.
        """
        if not self._http_client:
            return
        fresh_at = self._members_fetched_at.get(chat_id)
        if fresh_at is not None and (time.time() - fresh_at) < self._members_ttl:
            return
        try:
            await asyncio.wait_for(self._fetch_members(chat_id), timeout=3.0)
        except asyncio.TimeoutError:
            logger.debug("[%s] _ensure_members_fresh timed out for %s", self.name, chat_id)
        except Exception as e:
            logger.debug("[%s] _ensure_members_fresh failed for %s: %s", self.name, chat_id, e)

    def _is_group_approved(self, chat_id: str) -> bool:
        """A group is approved when: in the allowlist, or in approved_chats,
        or (owner configured) the owner themselves is a member/owner of it."""
        if chat_id in self._group_allowed_chats:
            return True
        if chat_id in self._approved_chats:
            return True
        # Owner's own group: owner_user_id is owner/admin of the group.
        if self._owner_user_id:
            members = self._members.get(chat_id, {})
            owner_m = members.get(self._owner_user_id)
            if owner_m and (owner_m.get("is_owner") or owner_m.get("is_admin")):
                return True
        return False

    async def _delete_message(self, chat_id: str, message_id: str) -> bool:
        """DELETE /messages?message_id= — remove a message in a group/channel.

        Requires the bot to be an admin of the chat (MAX API enforces this).
        Returns True on success, False otherwise.
        """
        if not self._http_client:
            return False
        try:
            resp = await self._http_client.delete(
                f"{API_SCHEME}://{API_HOST}/messages",
                params={"message_id": message_id},
                headers={"Authorization": self._token},
                timeout=15.0,
            )
            if resp.status_code < 300:
                logger.info("[%s] Deleted message %s in chat %s", self.name, message_id, chat_id)
                return resp.json() is not False
            logger.warning("[%s] delete_message HTTP %d for %s", self.name, resp.status_code, message_id)
            return False
        except Exception as e:
            logger.warning("[%s] delete_message failed: %s", self.name, e)
            return False

    async def _ban_member(self, chat_id: str, user_id: str) -> bool:
        """Remove/ban a member from a group chat.

        Uses DELETE /chats/{chatId}/members/{userId} (kick). Requires the bot
        to be an admin with add_remove_members permission.
        """
        if not self._http_client:
            return False
        try:
            resp = await self._http_client.delete(
                f"{API_SCHEME}://{API_HOST}/chats/{chat_id}/members/{user_id}",
                headers={"Authorization": self._token},
                timeout=15.0,
            )
            if resp.status_code < 300:
                logger.info("[%s] Removed member %s from chat %s", self.name, user_id, chat_id)
                return True
            logger.warning("[%s] ban_member HTTP %d for %s in %s", self.name, resp.status_code, user_id, chat_id)
            return False
        except Exception as e:
            logger.warning("[%s] ban_member failed: %s", self.name, e)
            return False

    async def _handle_moderation_command(self, chat_id: str, text: str, user_id: str) -> Optional[str]:
        """Parse a moderation request and execute it if permitted.

        Supported patterns (in approved groups only):
          <bot> удали <text-fragment>  — delete matching messages
          <bot> удали последнее          — delete the most recent message
          <bot> бан <nick/name>          — remove the member

        Permission: only the bot owner OR the group owner/admin may request
        moderation. The bot itself must be group admin to actually delete/ban
        (MAX API enforces this server-side).
        Returns a reply string, or None if nothing matched / not permitted.
        """
        if not chat_id or not text:
            return None
        # Only handle moderation when the text contains a moderation verb
        # (not necessarily at the start — "каин, бан вася" includes "бан").
        low0 = text.lower().strip()
        mod_words = ("удали", "удалить", "бан", "забанить", "кик", "kick")
        if not any(w in low0 for w in mod_words):
            return None
        # Roles/permissions may have just changed — refresh before deciding.
        try:
            await self._ensure_members_fresh(chat_id)
        except Exception:
            pass
        # Who is asking?
        is_owner = bool(self._owner_user_id) and user_id == self._owner_user_id
        member_info = self._members.get(str(chat_id), {})
        sender_member = member_info.get(str(user_id), {})
        is_group_owner = bool(sender_member.get("is_owner"))
        is_group_admin = bool(sender_member.get("is_admin"))
        if not (is_owner or is_group_owner or is_group_admin):
            return "Только владелец или админы группы могут просить о модерации."

        # Bot's own role in this chat
        bot_member = member_info.get(str(self._id), {})
        bot_is_admin = bool(bot_member.get("is_admin")) or bool(bot_member.get("is_owner"))
        if not bot_is_admin:
            return "Я не админ этой группы — не могу удалять сообщения или банить."

        # Parse the moderation verb wherever it appears ("каин, бан вася").
        low = low0
        # --- delete branch ---
        del_word = None
        for w in ("удали", "удалить"):
            if w in low:
                del_word = w
                break
        if del_word:
            idx = low.find(del_word)
            fragment = text[idx + len(del_word):].strip(" ,.!?")
            if not fragment:
                return "Что удалить? Например: «каин, удали последнее» или «каин, удали <текст>»."
            if fragment.lower() in ("последнее", "последний", "last"):
                return "Удаление по «последнему» требует истории сообщений — пока не поддержано напрямую. Уточни текст."
            # We don't keep a message log, so we can only confirm the request
            # is understood. Real deletion by fragment needs a history fetch
            # (GET /messages) — out of scope for this pass.
            logger.info("[%s] Moderation delete requested by %s in %s: %s", self.name, user_id, chat_id, fragment)
            return "Запрос на удаление принят. (Полное удаление по тексту требует истории сообщений — добавлю позже.)"

        # --- ban branch ---
        ban_word = None
        for w in ("забанить", "забани", "бан", "кик", "kick"):
            if w in low:
                ban_word = w
                break
        if ban_word:
            idx = low.find(ban_word)
            target = text[idx + len(ban_word):].strip(" ,.!?")
            if not target:
                return "Кого забанить? Например: «каин, бан Вася»."
            # Resolve target name/nick -> user_id from the members cache
            target_id = None
            tl = target.lower()
            for uid, m in member_info.items():
                name = str(m.get("first_name") or "").lower()
                uname = str(m.get("username") or "").lower().lstrip("@")
                if name == tl or uname == tl or (tl.startswith("@") and uname == tl.lstrip("@")):
                    target_id = uid
                    break
            if not target_id:
                return f"Не нашёл участника «{target}» в списке группы."
            if target_id == self._id:
                return "Я не могу забанить сам себя 😅"
            ok = await self._ban_member(str(chat_id), target_id)
            if ok:
                return f"✅ {target} удалён из группы."
            return "Не удалось забанить (нет прав администратора?)."
        return None

    async def _register_commands(self) -> None:
        """PATCH /me/commands — set the bot command menu (best effort)."""
        if self._http_client is None:
            return
        commands = [
            {"name": "help", "description": "Помощь и команды"},
            {"name": "new", "description": "Новая сессия"},
            {"name": "sethome", "description": "Установить этот чат домашним"},
            {"name": "reset", "description": "Сбросить сессию"},
        ]
        try:
            body = json.dumps({"commands": commands}).encode("utf-8")
            resp = await self._http_client.patch(
                f"{API_SCHEME}://{API_HOST}/me/commands",
                content=body,
                headers={"Authorization": self._token, "Content-Type": "application/json"},
                timeout=15.0,
            )
            if resp.status_code < 300:
                logger.info("[%s] Bot command menu registered (%d commands)", self.name, len(commands))
            else:
                logger.debug("[%s] /me/commands HTTP %d: %s", self.name, resp.status_code, resp.text[:150])
        except Exception as e:
            logger.debug("[%s] /me/commands failed: %s", self.name, e)

    async def _run_poll_loop(self) -> None:
        """Long Poll GET /updates with marker cursor and reconnect backoff."""
        backoff_idx = 0

        while self._running:
            try:
                await self._poll_once()
                backoff_idx = 0
            except asyncio.CancelledError:
                return
            except Exception as e:
                if not self._running:
                    return
                self._last_poll_error = str(e)
                self._last_error_at = time.time()
                logger.warning("[%s] Poll error: %s", self.name, e)
                delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
                # Jitter avoids thundering-herd when many clients reconnect at once
                delay += random.uniform(0, 1.0)
                await asyncio.sleep(delay)
                backoff_idx += 1
                continue

            # Small pause between polls; long-poll already holds the connection
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _poll_once(self) -> None:
        """One GET /updates request."""
        if self._http_client is None:
            return
        params: Dict[str, Any] = {
            "timeout": DEFAULT_POLL_TIMEOUT,
            "limit": DEFAULT_POLL_LIMIT,
        }
        if self._marker:
            params["marker"] = self._marker

        try:
            resp = await self._http_client.get(
                f"{API_SCHEME}://{API_HOST}/updates",
                params=params,
                headers={"Authorization": self._token},
            )
        except Exception as e:
            self._last_poll_error = str(e)
            self._last_error_at = time.time()
            raise
        self._last_poll_at = time.time()
        self._last_poll_error = None

        if resp.status_code == 401:
            logger.error("[%s] Auth failed (401) — token invalid. Stopping.", self.name)
            self._set_fatal_error(
                "max_unauthorized",
                "MAX API rejected auth (401). Check MAX_BOT_TOKEN.",
                retryable=False,
            )
            self._running = False
            return
        if resp.status_code >= 400:
            logger.warning("[%s] Poll HTTP %d: %s", self.name, resp.status_code, resp.text[:200])
            return

        try:
            data = resp.json()
        except Exception:
            logger.warning("[%s] Bad JSON from /updates", self.name)
            return

        updates = data.get("updates") or []
        if data.get("marker") is not None:
            self._marker = int(data["marker"])
            self._save_marker()
        for upd in updates:
            await self._handle_update(upd)

    # -- Inbound message processing -----------------------------------------

    async def _download_url(self, url: str, ext: str = ".bin") -> str:
        """Download an attachment URL to the local cache dir.

        Uses an SSRF-safe client with the system trust store so hosts that
        chain to a different root (e.g. fd.oneme.ru) verify fine — the
        adapter's main client is pinned to the Минцифры CA.
        """
        # Preferred: SSRF-safe, system trust store (covers fd.oneme.ru etc.)
        try:
            from tools.url_safety import create_ssrf_safe_async_client

            async with create_ssrf_safe_async_client(
                timeout=30.0, follow_redirects=True
            ) as client:
                resp = await client.get(url)
                if resp.status_code >= 300:
                    raise RuntimeError(f"HTTP {resp.status_code} downloading {url[:60]}")
                return self._save_to_cache(resp.content, ext)
        except Exception as e:
            logger.debug("[%s] SSRF-safe download failed (%s), falling back to pinned CA client", self.name, e)

        # Fallback: main client (pinned to Минцифры CA)
        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")
        resp = await self._http_client.get(url, timeout=30.0)
        if resp.status_code >= 300:
            raise RuntimeError(f"HTTP {resp.status_code} downloading {url[:60]}")
        return self._save_to_cache(resp.content, ext)

    def _save_to_cache(self, data: bytes, ext: str) -> str:
        """Persist raw attachment bytes under HERMES_HOME/cache/attachments.

        Sniffs audio magic bytes so an MP3/OGG/WAV delivered as ``.bin`` (MAX
        file-type attachments don't always carry a filename) is still saved
        with its real container extension and is picked up by the STT path.
        """
        if not ext or ext == ".bin":
            try:
                from tools.audio_container import sniff_audio_ext
                ext = sniff_audio_ext(data, ".bin")
            except Exception:
                pass
        cache_dir = os.path.join(
            os.getenv("HERMES_HOME", "") or os.path.expanduser("~/.hermes"),
            "cache", "attachments",
        )
        os.makedirs(cache_dir, exist_ok=True)
        fname = f"att_{uuid.uuid4().hex[:12]}{ext}"
        path = os.path.join(cache_dir, fname)
        with open(path, "wb") as f:
            f.write(data)
        return path

    async def _download_attachment(self, token: str, media_type: str) -> Optional[str]:
        """Resolve an attachment token to a downloadable URL and fetch it.

        MAX attachment payloads carry ``token`` but not always a direct URL.
        This uses the public download endpoint for a token-based fetch.
        """
        try:
            url = f"{API_SCHEME}://{API_HOST}/attachments/{token}"
            ext = {
                "image": ".jpg", "video": ".mp4", "audio": ".mp3", "file": ".bin",
            }.get(media_type, ".bin")
            return await self._download_url(url, ext)
        except Exception as e:
            logger.warning("[%s] Token-based download failed: %s", self.name, e)
            return None

    async def _handle_update(self, upd: Dict[str, Any]) -> None:
        """Process a single Update object from MAX."""
        update_type = upd.get("update_type") or upd.get("event") or "unknown"
        # Bot added to a group / user started the bot in a chat — log the chat_id
        # so the adapter learns group chat_ids without manual configuration.
        if update_type in ("bot_added", "bot_started"):
            chat = upd.get("chat") or upd.get("chat_id") or {}
            cid = chat.get("chat_id") if isinstance(chat, dict) else chat
            cid = cid or upd.get("chat_id") or upd.get("user_id")
            cname = chat.get("title") if isinstance(chat, dict) else None
            ctype = chat.get("chat_type") if isinstance(chat, dict) else None
            logger.info(
                "[%s] %s in chat %s (type=%s, name=%s) — learned group chat_id",
                self.name, update_type, cid, ctype, cname,
            )
            if cid:
                sid = str(cid)
                self._known_chats[sid] = {
                    "name": cname or sid,
                    "type": "group" if ctype not in ("dialog", "dm") else "dm",
                }
                # Fetch member roles for this group (owners/admins/members).
                if self._known_chats[sid]["type"] == "group":
                    await self._fetch_members(sid)
                # Approval gate: if this group is not approved/allowed and the
                # bot has an owner configured, notify the owner and stay silent
                # until approved.
                if (
                    self._known_chats[sid]["type"] == "group"
                    and sid not in self._approved_chats
                    and sid not in self._group_allowed_chats
                    and self._owner_user_id
                ):
                    await self._notify_owner_approval(sid, cname or sid)
            return

        if update_type != "message_created":
            logger.debug("[%s] Ignoring update type %s", self.name, update_type)
            return

        message = upd.get("message") or upd
        # Group allowlist gate: if restricted and this chat is not allowed, drop.
        recipient0 = message.get("recipient") or {}
        chat_type0 = recipient0.get("chat_type") or "dialog"
        cid0 = str(upd.get("chat_id") or recipient0.get("chat_id") or "")
        if chat_type0 not in ("dialog", "dm") and cid0 and self._group_allowed_chats:
            if cid0 not in self._group_allowed_chats:
                logger.debug("[%s] Group %s not in allowlist, ignoring", self.name, cid0)
                return
        sender = message.get("sender") or upd.get("user") or {}
        if sender.get("is_bot") or sender.get("isBot"):
            logger.debug("[%s] Skipping own/bot message", self.name)
            return

        body_obj = message.get("body") or {}
        text = None
        if isinstance(body_obj, dict):
            text = body_obj.get("text") or body_obj.get("body")
        if not text:
            text = upd.get("body")
        text = (text or "").strip()

        # Handle attachments: download media and pass to agent as media_urls.
        attachments_desc = ""
        media_urls: List[str] = []
        media_types: List[str] = []
        if isinstance(body_obj, dict):
            attachments = body_obj.get("attachments") or []
            for att in attachments:
                if not isinstance(att, dict):
                    continue
                t = att.get("type", "")
                payload = att.get("payload") or {}
                url = payload.get("url") if isinstance(payload, dict) else None
                if t == "image" and url:
                    # Download to local cache so the vision tool can read it
                    try:
                        local_path = await cache_image_from_url(url, ext=".jpg")
                        media_urls.append(local_path)
                        media_types.append("image/jpeg")
                        attachments_desc += " [Фото]"
                        logger.info("[%s] Downloaded inbound image to %s", self.name, local_path)
                    except Exception as e:
                        logger.warning("[%s] Failed to cache image %s: %s", self.name, url[:60], e)
                        attachments_desc += " [Фото (не удалось скачать)]"
                elif t == "image":
                    media_kind = "[Фото]"
                    attachments_desc += f" {media_kind}"
                    # Image without direct URL — try token-based download if payload has token
                    token = payload.get("token") if isinstance(payload, dict) else None
                    if token:
                        local_path = await self._download_attachment(token, "image")
                        if local_path:
                            media_urls.append(local_path)
                            media_types.append("image/jpeg")
                elif t in ("video", "audio", "file") and url:
                    # Try to download non-image attachments too
                    try:
                        # Prefer the real filename from payload (gives the right
                        # extension: .pdf/.docx/.mp4/... instead of a generic .bin)
                        fname = payload.get("filename") if isinstance(payload, dict) else None
                        ext = ""
                        if fname:
                            ext = os.path.splitext(str(fname))[1].lower()
                        if not ext:
                            ext = os.path.splitext(url.split("?")[0])[1] or {
                                "video": ".mp4", "audio": ".mp3", "file": ".bin",
                            }.get(t, ".bin")
                        local_path = await self._download_url(url, ext)
                        media_urls.append(local_path)
                        mime = _mime_for_ext(ext, t)
                        # If the real extension (from filename or magic-byte
                        # sniff) is audio but the attachment type was 'file',
                        # upgrade the MIME so the STT pipeline kicks in.
                        if t == "file" and os.path.splitext(local_path)[1].lower() in (
                            ".mp3", ".ogg", ".wav", ".m4a", ".flac", ".opus", ".aac", ".oga",
                        ):
                            mime = _mime_for_ext(os.path.splitext(local_path)[1].lower(), "audio")
                        media_types.append(mime)
                        attachments_desc += f" [{_MEDIA_LABELS.get(t, t)}]"
                        logger.info("[%s] Downloaded inbound %s to %s", self.name, t, local_path)
                    except Exception as e:
                        logger.warning("[%s] Failed to download %s %s: %s", self.name, t, url[:60], e)
                        attachments_desc += f" [{_MEDIA_LABELS.get(t, t)} (не удалось скачать)]"
                elif t == "video":
                    attachments_desc += " [Видео]"
                elif t == "audio":
                    attachments_desc += " [Аудио]"
                elif t == "file":
                    attachments_desc += " [Файл]"
                else:
                    attachments_desc += f" [Вложение:{t}]"
        if not text and attachments_desc:
            text = attachments_desc
        if not text and not media_urls:
            # Voice messages may arrive as a sparse update (no message body).
            # Try a recursive URL search before giving up.
            voice_url = _find_media_url(upd)
            if voice_url:
                try:
                    ext = os.path.splitext(voice_url.split("?")[0])[1] or ".ogg"
                    local_path = await self._download_url(voice_url, ext)
                    media_urls.append(local_path)
                    media_types.append(_mime_for_ext(ext, "audio"))
                    attachments_desc = " [Голосовое]"
                    text = attachments_desc
                    logger.info("[%s] Downloaded inbound voice to %s", self.name, local_path)
                except Exception as e:
                    logger.warning("[%s] Failed to download voice %s: %s", self.name, voice_url[:60], e)
                    return
            else:
                # Log the raw update so we can see what MAX actually sent
                logger.info("[%s] Empty inbound — RAW update (full): %s", self.name, json.dumps(upd, ensure_ascii=False))
                return

        # Echo-loop prevention
        if _ECHO_MARKER in text:
            return

        recipient = message.get("recipient") or {}
        chat_id = str(
            upd.get("chat_id")
            or recipient.get("chat_id")
            or sender.get("user_id")
            or ""
        )
        user_id = str(sender.get("user_id") or "")
        user_name = sender.get("name") or user_id or "?"
        chat_type = recipient.get("chat_type") or "dialog"
        # Normalize MAX chat types: dialog=DM, chat/channel/group=group
        if chat_type in ("dialog", "dm"):
            chat_type = "dm"
        else:
            chat_type = "group"
        # Remember the chat (so replies can route to it even after restart)
        self._known_chats.setdefault(chat_id, {"name": chat_id, "type": chat_type})
        if chat_type == "group" and self._known_chats[chat_id].get("name") == chat_id:
            chat_title = recipient.get("chat_name") or recipient.get("title") or user_name
            if chat_title:
                self._known_chats[chat_id]["name"] = chat_title

        # Real message ID from MAX body.mid, fallback to timestamp
        mid = ""
        if isinstance(body_obj, dict):
            mid = str(body_obj.get("mid") or "")
        msg_id = mid or str(upd.get("timestamp") or uuid.uuid4().hex)
        if self._is_duplicate(msg_id):
            return

        # Group filter: only respond when the group is approved AND the message is
        # explicitly addressed to the bot (mention by name, @username, or alias).
        # DM always responds.
        if chat_type == "group":
            if not self._is_group_approved(chat_id):
                logger.debug("[%s] Group %s not approved, ignoring: %s", self.name, chat_id, text[:60])
                return
            # Keep member roles fresh (participants/roles change over time).
            try:
                await self._ensure_members_fresh(chat_id)
            except Exception:
                logger.debug("[%s] members refresh failed for %s", self.name, chat_id)
            if not self._is_addressed_to_bot(text):
                logger.debug("[%s] Group message not addressed to bot, ignoring: %s", self.name, text[:60])
                return

        # Owner-only commands: /approve <chat_id> and /deny <chat_id>
        if user_id and self._owner_user_id and user_id == self._owner_user_id:
            stripped = text.strip().lower()
            if stripped.startswith("/approve"):
                parts = stripped.split()
                if len(parts) >= 2:
                    target = parts[1]
                    self._approved_chats.add(target)
                    self._known_chats.setdefault(target, {"name": target, "type": "group"})
                    logger.info("[%s] Owner approved group %s", self.name, target)
                    # Acknowledge in DM
                    try:
                        payload = {"text": f"✅ Группа {target} одобрена. Могу работать.", "format": "markdown"}
                        await self._http_client.post(
                            f"{API_SCHEME}://{API_HOST}/messages",
                            params={"user_id": self._owner_user_id},
                            content=json.dumps(payload).encode("utf-8"),
                            headers={"Authorization": self._token, "Content-Type": "application/json"},
                            timeout=15.0,
                        )
                    except Exception:
                        pass
                return
            if stripped.startswith("/deny"):
                parts = stripped.split()
                if len(parts) >= 2:
                    target = parts[1]
                    self._approved_chats.discard(target)
                    logger.info("[%s] Owner denied group %s", self.name, target)
                    try:
                        payload = {"text": f"🚫 Группа {target} отклонена. Буду молчать там.", "format": "markdown"}
                        await self._http_client.post(
                            f"{API_SCHEME}://{API_HOST}/messages",
                            params={"user_id": self._owner_user_id},
                            content=json.dumps(payload).encode("utf-8"),
                            headers={"Authorization": self._token, "Content-Type": "application/json"},
                            timeout=15.0,
                        )
                    except Exception:
                        pass
                return


        # Role-aware slash commands (plugin-level gate, no core changes).
        #   owner of the bot      -> all commands incl. moderation
        #   owner/admin of group  -> safe session commands (/new /reset /compress)
        #   members               -> no slash commands at all
        if text.startswith("/"):
            stripped_cmd = text.strip().lower().split()[0] if text.strip() else ""
            is_owner = bool(self._owner_user_id) and user_id == self._owner_user_id
            member_info = self._members.get(str(chat_id), {})
            sender_member = member_info.get(str(user_id), {})
            is_group_owner = bool(sender_member.get("is_owner"))
            is_group_admin = bool(sender_member.get("is_admin"))
            safe_cmds = {"/new", "/reset", "/compress", "/status", "/help"}

            if chat_type == "group":
                # Approve/deny handled above (owner-only); anything else:
                # members get nothing, group owners/admins get safe commands.
                if is_owner:
                    pass  # owner can run anything
                elif is_group_owner or is_group_admin:
                    if stripped_cmd not in safe_cmds:
                        logger.debug("[%s] Group admin command %s not allowed", self.name, stripped_cmd)
                        # Silently drop non-safe slash from group admin
                        return
                else:
                    logger.debug("[%s] Group member slash %s ignored", self.name, stripped_cmd)
                    return
            # DM: non-owner slash commands pass through to the agent
            # (the agent/gateway handles them normally).

        # Moderation requests in approved groups ("удали <текст>" / "бан <ник>")
        # — handled before the agent, only for owner/group-owner/admin askers.
        if chat_type == "group":
            mod_reply = await self._handle_moderation_command(chat_id, text, user_id)
            if mod_reply is not None:
                # Reply directly in the group
                try:
                    payload = {"text": mod_reply, "format": "markdown"}
                    resp = await self._http_client.post(
                        f"{API_SCHEME}://{API_HOST}/messages",
                        params={"chat_id": chat_id},
                        content=json.dumps(payload).encode("utf-8"),
                        headers={"Authorization": self._token, "Content-Type": "application/json"},
                        timeout=15.0,
                    )
                    logger.info("[%s] Moderation reply sent (HTTP %s)", self.name, getattr(resp, "status_code", "?"))
                except Exception as e:
                    logger.warning("[%s] Moderation reply failed: %s", self.name, e)
                return

        timestamp = datetime.now(tz=timezone.utc)
        try:
            ts = upd.get("timestamp") or message.get("timestamp")
            if ts:
                timestamp = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc)
        except (ValueError, OSError, TypeError):
            pass

        chat_name = self._known_chats.get(chat_id, {}).get("name") or user_name
        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=user_id,
            user_name=user_name,
        )
        # Store user_id on metadata so send() can reply to the right recipient
        self._last_user_id = user_id

        channel_prompt = None
        if chat_type == "group":
            channel_prompt = self._resolve_channel_prompt(chat_id)

        message_event = MessageEvent(
            text=text,
            message_type=MessageType.PHOTO if media_urls else MessageType.TEXT,
            source=source,
            message_id=msg_id,
            raw_message=upd,
            timestamp=timestamp,
            media_urls=media_urls,
            media_types=media_types,
            user_id=user_id,
            user_name=user_name,
            channel_prompt=channel_prompt,
            metadata={
                "max_name": self._name,
                "max_username": self._username,
                "chat_type": chat_type,
            },
        )

        logger.info("[%s] Message from %s (chat %s): %s", self.name, user_name, chat_id, text[:80])
        logger.debug("[%s] RAW update keys=%s body=%s", self.name, list(upd.keys()), json.dumps(body_obj, ensure_ascii=False)[:500])
        await self.handle_message(message_event)

    def _resolve_channel_prompt(self, chat_id: str) -> Optional[str]:
        """Resolve the per-group ephemeral prompt for a chat.

        Builds an auto mini-prompt from the bot's own identity (name,
        username, description) merged with any custom ``channel_prompts``
        entry from config.extra (keyed by chat_id). This text is injected at
        the start of the session context so the agent always knows who it is,
        how it is addressed, and what it is for in this group.
        """
        from gateway.platforms.base import resolve_channel_prompt

        parts = []
        if self._name:
            parts.append(f"Ты — {self._name}.")
        if self._username:
            parts.append(f"К тебе обращаются: @{self._username}, {self._name or 'бот'}.")
        if self._description:
            parts.append(self._description)
        if self._name or self._username:
            parts.append(
                "Если тебя упоминают в третьем лице без прямого вопроса или просьбы — "
                "прими к сведению, но не отвечай вслух в чат."
            )
        # Role awareness: if the bot is admin/owner of the group, say so and
        # state moderation capabilities (delete/ban) — so the agent knows what
        # it is ALLOWED to do in this specific group. Members without rights
        # get a notice that moderation is unavailable..
        member_info = self._members.get(str(chat_id), {})
        bot_member = member_info.get(str(self._id))
        if bot_member:
            if bot_member.get("is_owner"):
                parts.append("Ты владелец этой группы.")
            elif bot_member.get("is_admin"):
                perms = bot_member.get("permissions") or []
                parts.append(f"Ты администратор этой группы. Права: {', '.join(perms) if perms else 'стандартные'}.")
                parts.append("Ты можешь удалять сообщения и управлять участниками группы (по просьбе владельца или админов).")
            else:
                parts.append("Ты обычный участник этой группы — модерация недоступна.")
        auto = " ".join(parts).strip()

        custom = resolve_channel_prompt(self.config.extra, str(chat_id)) or ""
        if auto and custom:
            combined = f"{auto}\n\n{custom}"
            return combined
        return auto or custom or None

    def _is_addressed_to_bot(self, text: str) -> bool:
        """Heuristic: is this message addressed to the bot?

        Returns True if the text contains the bot's @username or display name
        (or any configured alias), case-insensitive. Used only for group chats;
        DMs always respond. Generic words like "бот"/"bot" are intentionally
        NOT matches — the bot answers only when addressed by its actual name.
        """
        if not text:
            return False
        low = text.lower()
        # @username mention (with or without @)
        if self._username:
            u = self._username.lower().lstrip("@")
            if u and ("@" + u in low or u in low):
                return True
        # Display name (first_name) — e.g. "матрёшка, сколько время?"
        if self._name:
            n = self._name.lower().strip()
            if n and n in low:
                return True
        # Configured aliases (from config.extra.bot_aliases)
        for alias in self._aliases:
            if alias and alias in low:
                return True
        return False

    def _is_duplicate(self, msg_id: str) -> bool:
        now = time.time()
        # Вычищаем записи старше окна дедупликации при каждом вызове —
        # иначе старые ID навсегда останутся в памяти и будут ложно
        # считаться дубликатами (особенно после перезапуска или долгой паузы).
        if self._seen_messages:
            cutoff = now - DEDUP_WINDOW_SECONDS
            self._seen_messages = {k: v for k, v in self._seen_messages.items() if v > cutoff}
        if msg_id in self._seen_messages:
            return True
        self._seen_messages[msg_id] = now
        return False

    # -- Outbound messaging -------------------------------------------------

    async def _rate_limit_send(self, chat_id: str) -> None:
        """Enforce MAX rate limit: max 2 messages/sec per chat."""
        now = time.time()
        history = self._send_history.setdefault(chat_id, [])
        # Keep only the last second
        history[:] = [t for t in history if now - t < 1.0]
        if len(history) >= 2:
            sleep_for = 1.0 - (now - history[0])
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        self._send_history[chat_id].append(time.time())

    @staticmethod
    def _split_text(
        text: str, limit: int = MAX_MESSAGE_LENGTH
    ) -> List[str]:
        """Split long text into ≤limit chunks, preferring line/word breaks.

        MAX hard-caps a single message at MAX_MESSAGE_LENGTH chars; instead of
        silently truncating (old behaviour), split into several messages. The
        caller spaces the sends to respect MAX's ~2 msg/sec dialog limit.
        """
        if len(text) <= limit:
            return [text]
        chunks: List[str] = []
        remaining = text
        while len(remaining) > limit:
            cut = remaining.rfind("\n", 0, limit)
            if cut <= 0:
                cut = remaining.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            chunks.append(remaining[:cut])
            remaining = remaining[cut:].lstrip("\n")
        if remaining:
            chunks.append(remaining)
        return chunks

    def _smart_truncate(self, content: str) -> str:
        """Truncate to MAX limit, cutting at a markdown-friendly boundary.

        If content exceeds MAX_MESSAGE_LENGTH, cut at the last newline or
        space before the limit (so we don't split a code block / word) and
        append a truncation notice. If there's no boundary (single long
        word), cut hard and still append the notice.
        """
        if len(content) <= MAX_MESSAGE_LENGTH:
            return content
        limit = MAX_MESSAGE_LENGTH - len(_TRUNCATION_NOTICE)
        cut = content[:limit]
        # Cut at last newline or space if possible
        last_nl = cut.rfind("\n")
        last_sp = cut.rfind(" ")
        boundary = max(last_nl, last_sp)
        if boundary > limit * 0.5:  # only if it's a reasonable cut point
            cut = cut[:boundary]
        return cut.rstrip() + _TRUNCATION_NOTICE

    @staticmethod
    def _guess_media_type(path: str) -> str:
        """Guess MAX media type from file extension."""
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        if ext in ("jpg", "jpeg", "png", "gif", "webp", "bmp"):
            return "image"
        if ext in ("mp4", "mov", "avi", "mkv", "webm"):
            return "video"
        if ext in ("mp3", "ogg", "wav", "m4a", "flac"):
            return "audio"
        return "file"

    async def _upload_media(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Upload a file to MAX and return the attachment dict for /messages.

        Flow: POST /uploads?type=X → get url+token → upload file to url →
        attachment {"type": X, "payload": {"token": ...}}.
        """
        if self._http_client is None:
            return None
        media_type = self._guess_media_type(file_path)
        try:
            # 1. Get upload URL (may include token for video/audio)
            resp = await self._http_client.post(
                f"{API_SCHEME}://{API_HOST}/uploads",
                params={"type": media_type},
                headers={"Authorization": self._token},
                timeout=15.0,
            )
            if resp.status_code >= 300:
                logger.warning("[%s] /uploads HTTP %d: %s", self.name, resp.status_code, resp.text[:200])
                return None
            data = resp.json()
            upload_url = data.get("url")
            if not upload_url:
                logger.warning("[%s] /uploads missing url", self.name)
                return None

            # 2. Upload the file (multipart field "data").
            #    The upload URL lives on a CDN (iu.oneme.ru / fu.oneme.ru /
            #    okcdn.ru) with a REGULAR CA cert. Our client is pinned to the
            #    Ministry CA, so use a fresh client with default trust here.
            async with httpx.AsyncClient(verify=True, timeout=60.0) as up_client:
                with open(file_path, "rb") as f:
                    files = {"data": (os.path.basename(file_path), f)}
                    up = await up_client.post(upload_url, files=files, timeout=60.0)
            if up.status_code >= 300:
                logger.warning("[%s] upload HTTP %d: %s", self.name, up.status_code, up.text[:200])
                return None

            # 3. Token comes from the upload response, NOT from /uploads.
            #    - image  → {"photos": {"<id>": {"token": "..."}}} (or token field)
            #    - file   → {"token": "..."}
            #    - video/audio → <retval>1</retval> (token already from /uploads)
            token = ""
            photos = None
            try:
                up_data = up.json()
                if isinstance(up_data, dict):
                    if up_data.get("photos"):
                        photos = up_data["photos"]
                        # token lives inside photos map
                        for pid, pinfo in up_data["photos"].items():
                            if isinstance(pinfo, dict) and pinfo.get("token"):
                                token = pinfo["token"]
                                break
                    token = token or up_data.get("token") or ""
            except Exception:
                # Some responses are not JSON (e.g. <retval>1</retval>)
                pass
            if not token:
                token = data.get("token") or ""
            if not token and media_type == "image":
                logger.warning("[%s] No token after image upload", self.name)
                return None

            # 4. Build attachment
            payload: Dict[str, Any] = {"token": token}
            if photos:
                payload["photos"] = photos
            return {"type": media_type, "payload": payload}
        except Exception as e:
            logger.error("[%s] Upload media failed: %s", self.name, e)
            return None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message to a MAX user (user_id) or chat (chat_id).

        Long content (>4000 chars) is split into several sequential messages,
        respecting MAX's ~2 msg/sec limit via ``_rate_limit_send``.

        Attachments: if metadata carries ``media_files`` (list of paths),
        they are uploaded via POST /uploads and attached to the first chunk.
        """
        if self._http_client is None:
            return SendResult(success=False, error="HTTP client not initialized")

        metadata = metadata or {}
        # Reply target: for dialogs (dm) use user_id; for chats/channels use chat_id
        user_id = metadata.get("user_id") or self._last_user_id
        chat_type = metadata.get("chat_type") or "dm"
        params: Dict[str, Any] = {}
        if chat_type == "dm" and user_id:
            params["user_id"] = user_id
        else:
            params["chat_id"] = chat_id

        # Upload attachments (if any)
        attachments: List[Dict[str, Any]] = []
        media_files = metadata.get("media_files") or []
        for fp in media_files:
            att = await self._upload_media(str(fp))
            if att:
                attachments.append(att)
            else:
                logger.warning("[%s] Could not upload attachment: %s", self.name, fp)

        chunks = self._split_text(content)
        last: SendResult = SendResult(success=False, error="no chunks")
        for i, chunk in enumerate(chunks):
            # Reasoning block hidden by default (MAX_SHOW_REASONING=true keeps it);
            # fenced code becomes MAX HTML <pre> so multi-line code survives.
            text_out, fmt = prepare_outgoing_text(chunk)
            # Attachments go with the first chunk only
            payload = {
                "text": text_out,
                "attachments": attachments if i == 0 else [],
                "format": fmt,
            }
            body = json.dumps(payload).encode("utf-8")
            try:
                await self._rate_limit_send(str(chat_id))
                resp = await self._http_client.post(
                    f"{API_SCHEME}://{API_HOST}/messages",
                    params=params,
                    content=body,
                    headers={
                        "Authorization": self._token,
                        "Content-Type": "application/json",
                    },
                    timeout=15.0,
                )
                if resp.status_code < 300:
                    last = SendResult(success=True, message_id=uuid.uuid4().hex[:12])
                else:
                    logger.warning("[%s] Send failed HTTP %d: %s", self.name, resp.status_code, resp.text[:200])
                    last = SendResult(success=False, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
                    break
            except Exception as e:
                logger.error("[%s] Send error: %s", self.name, e)
                last = SendResult(success=False, error=str(e))
                break
        return last

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
        """Send a file/photo natively via MAX uploads.

        Overrides the base fallback (which only posts a "couldn't deliver"
        notice). Uploads the file via POST /uploads and attaches it to a
        message with the caption as text.
        """
        text = caption or ""
        att = await self._upload_media(str(file_path))
        if not att:
            return SendResult(success=False, error="upload failed")
        if not text:
            text = f"📎 {file_name or os.path.basename(str(file_path))}"
        metadata = metadata or {}
        user_id = metadata.get("user_id") or self._last_user_id
        chat_type = metadata.get("chat_type") or "dm"
        params: Dict[str, Any] = {}
        if chat_type == "dm" and user_id:
            params["user_id"] = user_id
        else:
            params["chat_id"] = chat_id
        cap_out, cap_fmt = prepare_outgoing_text(text[:MAX_MESSAGE_LENGTH])
        payload = {"text": cap_out, "attachments": [att], "format": cap_fmt}
        try:
            await self._rate_limit_send(str(chat_id))
            resp = await self._http_client.post(
                f"{API_SCHEME}://{API_HOST}/messages",
                params=params,
                content=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": self._token, "Content-Type": "application/json"},
                timeout=15.0,
            )
            if resp.status_code < 300:
                return SendResult(success=True, message_id=uuid.uuid4().hex[:12])
            logger.warning("[%s] send_document HTTP %d: %s", self.name, resp.status_code, resp.text[:200])
            return SendResult(success=False, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.error("[%s] send_document error: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def send_image_file(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image file natively via MAX uploads (type=image)."""
        return await self.send_document(chat_id, file_path, caption=caption, metadata=metadata)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image by URL: download it first, then upload to MAX."""
        if self._http_client is None:
            return SendResult(success=False, error="HTTP client not initialized")
        try:
            import tempfile
            resp = await self._http_client.get(image_url, timeout=30.0)
            if resp.status_code >= 300:
                return SendResult(success=False, error=f"HTTP {resp.status_code} downloading image")
            ext = os.path.splitext(image_url.split("?")[0])[1] or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            try:
                return await self.send_document(chat_id, tmp_path, caption=caption, metadata=metadata)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        except Exception as e:
            logger.error("[%s] send_image error: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send a typing indicator via POST /chats/{chatId}/actions.

        MAX supports ``typing_on`` (and other sending_* actions). The
        indicator lives ~5-6s, and the gateway's ``_keep_typing`` loop calls
        this every ~2s, so the bubble stays visible while the agent works.

        Note: MAX expects the numeric chat id. For dialogs we forward the
        ``chat_id`` we received from updates (recipient.chat_id — a numeric
        dialog id); if metadata carries a ``user_id`` we still use chat_id
        because the actions endpoint is keyed by chat, not user.
        """
        if self._http_client is None or not chat_id:
            return
        try:
            resp = await self._http_client.post(
                f"{API_SCHEME}://{API_HOST}/chats/{chat_id}/actions",
                content=json.dumps({"action": "typing_on"}).encode("utf-8"),
                headers={
                    "Authorization": self._token,
                    "Content-Type": "application/json",
                },
                timeout=5.0,
            )
            if resp.status_code >= 400:
                logger.debug(
                    "[%s] send_typing HTTP %d: %s",
                    self.name,
                    resp.status_code,
                    resp.text[:150],
                )
        except Exception as e:
            logger.debug("[%s] send_typing error: %s", self.name, e)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about a MAX chat."""
        return {"name": chat_id, "type": "dm"}

    async def disconnect(self) -> None:
        """Stop polling (gracefully) and close the HTTP client."""
        self._running = False
        self._mark_disconnected()
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._poll_task = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._seen_messages.clear()
        self._save_marker()
        logger.info("[%s] Disconnected", self.name)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def _env_enablement() -> dict | None:
    """Seed PlatformConfig.extra from env vars during gateway config load."""
    token = os.getenv("MAX_BOT_TOKEN", "").strip()
    if not token:
        return None
    seed: dict = {"token": token}
    home = os.getenv("MAX_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {"chat_id": home, "name": os.getenv("MAX_HOME_CHANNEL_NAME", home)}
    group_allow = os.getenv("MAX_GROUP_ALLOWED_CHATS", "").strip()
    if group_allow:
        seed["group_allowed_chats"] = group_allow
    owner = os.getenv("MAX_OWNER_USER_ID", "").strip()
    if owner:
        seed["owner_user_id"] = owner
    approved = os.getenv("MAX_APPROVED_CHATS", "").strip()
    if approved:
        seed["approved_chats"] = approved
    gspu = os.getenv("MAX_GROUP_SESSIONS_PER_USER", "").strip().lower()
    if gspu in ("true", "1", "yes", "false", "0", "no"):
        seed["group_sessions_per_user"] = gspu in ("true", "1", "yes")
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
    """Out-of-process send for cron / send_message_tool fallbacks."""
    if not HTTPX_AVAILABLE:
        return {"error": "max standalone send: httpx not installed"}
    extra = getattr(pconfig, "extra", {}) or {}
    token = extra.get("token") or _get_scoped_secret("MAX_BOT_TOKEN", "")
    if not token:
        return {"error": "max standalone send: MAX_BOT_TOKEN not configured"}
    ca_path = _default_ca_path()
    text = (message or "")[:MAX_MESSAGE_LENGTH]

    # Upload attachments (if any)
    attachments: List[Dict[str, Any]] = []
    for fp in (media_files or []):
        try:
            media_type = MaxAdapter._guess_media_type(str(fp))
            # API-запрос (получить URL аплоада) — через Минцифры-CA
            async with httpx.AsyncClient(verify=ca_path or True, timeout=15.0) as client:
                r = await client.post(
                    f"{API_SCHEME}://{API_HOST}/uploads",
                    params={"type": media_type},
                    headers={"Authorization": token},
                )
                data = r.json()
            # CDN-аплоад: CDN (fu.oneme.ru / iu.oneme.ru) использует СТАНДАРТНЫЕ CA,
            # НЕ цепочку Минцифры — нужен системный trust (verify=True), а если и он
            # не подходит (наблюдалось у некоторых CDN-хостов) — fallback без проверки.
            if r.status_code < 300 and data.get("url"):
                up = None
                try:
                    async with httpx.AsyncClient(verify=True, timeout=60.0) as cdn:
                        with open(str(fp), "rb") as f:
                            up = await cdn.post(
                                data["url"],
                                files={"data": (os.path.basename(str(fp)), f)},
                                timeout=60.0,
                            )
                except Exception as cdn_err:
                    logger.warning("[max] standalone CDN verify=True failed (%s), retrying without verify", cdn_err)
                    async with httpx.AsyncClient(verify=False, timeout=60.0) as cdn:
                        with open(str(fp), "rb") as f:
                            up = await cdn.post(
                                data["url"],
                                files={"data": (os.path.basename(str(fp)), f)},
                                timeout=60.0,
                            )
                if up is not None and up.status_code < 300:
                    # Для image токен живёт ВНУТРИ photos (словарь {hash: {token: ...}}),
                    # на верхнем уровне его нет. Для остальных типов — token из /uploads.
                    up_data = {}
                    try:
                        up_data = up.json()
                    except Exception:
                        pass
                    if media_type == "image" and isinstance(up_data.get("photos"), dict) and up_data["photos"]:
                        payload = {"photos": up_data["photos"]}
                    else:
                        payload = {"token": data.get("token", "")}
                        if isinstance(up_data, dict) and up_data.get("photos"):
                            payload["photos"] = up_data["photos"]
                    attachments.append({"type": media_type, "payload": payload})
        except Exception as e:
            logger.warning("[max] standalone upload %s failed: %s", fp, e)

    text_out, fmt = prepare_outgoing_text(text)
    payload = {"text": text_out, "attachments": attachments, "format": fmt}
    body = json.dumps(payload).encode("utf-8")
    params: Dict[str, Any] = {}
    extra2 = getattr(pconfig, "extra", {}) or {}
    user_id = extra2.get("user_id") or os.getenv("MAX_HOME_USER_ID", "").strip()
    if user_id:
        params["user_id"] = user_id
    elif chat_id:
        params["chat_id"] = chat_id
    try:
        async with httpx.AsyncClient(verify=ca_path or True, timeout=15.0) as client:
            resp = await client.post(
                f"{API_SCHEME}://{API_HOST}/messages",
                params=params,
                content=body,
                headers={"Authorization": token, "Content-Type": "application/json"},
            )
        if resp.status_code >= 300:
            return {"error": f"max HTTP {resp.status_code}: {resp.text[:200]}"}
        return {"success": True, "platform": "max", "chat_id": chat_id, "message_id": uuid.uuid4().hex[:12]}
    except Exception as e:
        return {"error": f"max standalone send failed: {e}"}


def interactive_setup() -> None:
    """Interactive hermes gateway setup flow for the MAX platform.

    Lazy-imports ``hermes_cli.setup`` helpers so the plugin stays importable
        in non-CLI contexts (gateway runtime, tests).
    """
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

    print_header("MAX (Russian Messenger)")
    existing = get_env_value("MAX_BOT_TOKEN")
    if existing:
        print_info("MAX: already configured")
        if not prompt_yes_no("Reconfigure MAX?", False):
            return

    print_info("Connect Hermes to MAX (max.ru). Create a bot at")
    print_info("  business.max.ru → Чат-боты → создать → Расширенные настройки → Настроить")
    print()

    token = prompt("MAX bot token", password=True)
    if not token:
        print_warning("Token is required — skipping MAX setup")
        return
    save_env_value("MAX_BOT_TOKEN", token.strip())

    print()
    print_info("🔒 Access control")
    print_info("  DM (личные диалоги): кто может писать боту в личку.")
    allowed_users = prompt(
        "Allowed user IDs for DMs (comma-separated; empty = ask later)",
        default=get_env_value("MAX_ALLOWED_USERS") or "",
    )
    if allowed_users:
        save_env_value("MAX_ALLOWED_USERS", allowed_users.strip())
        print_success("  Saved — only these users can DM the bot.")
    else:
        print_warning("  No DM allowlist set — will deny all DMs until configured.")

    print()
    print_info("👥 Group chats")
    group_chats = prompt(
        "Allowed group chat IDs (comma-separated; empty = any group)",
        default=get_env_value("MAX_GROUP_ALLOWED_CHATS") or "",
    )
    if group_chats:
        save_env_value("MAX_GROUP_ALLOWED_CHATS", group_chats.strip())
        print_success("  Saved — bot works only in these groups.")
    else:
        print_info("  No group restriction — bot may be added to any group.")

    shared = prompt_yes_no(
        "Shared group context? (false = one session for the whole group, "
        "recommended for party bots; true = separate session per member)",
        False,
    )
    save_env_value("MAX_GROUP_SESSIONS_PER_USER", "false" if shared else "true")
    print_success("  Group sessions: %s" % ("shared (whole group)" if shared else "per user"))

    print()
    print_info("👑 Bot owner (full access: terminal, files)")
    owner_id = prompt(
        "Your MAX user ID (owner of this bot)",
        default=get_env_value("MAX_OWNER_USER_ID") or "",
    )
    if owner_id:
        save_env_value("MAX_OWNER_USER_ID", owner_id.strip())
        print_success("  Saved — this user has full access and can /approve groups.")
    else:
        print_warning("  No owner set — approval of new groups disabled. Set MAX_OWNER_USER_ID later.")

    print()
    print_info("🛡️ Group approval")
    approved = prompt(
        "Pre-approved group chat IDs (comma-separated; empty = ask owner on add)",
        default=get_env_value("MAX_APPROVED_CHATS") or "",
    )
    if approved:
        save_env_value("MAX_APPROVED_CHATS", approved.strip())
        print_success("  Saved — these groups are pre-approved.")
    else:
        print_info("  No pre-approved groups — bot will ask the owner when added to a new group.")

    print()
    print_success("MAX setup complete. Restart the gateway to apply.")


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="max",
        setup_fn=interactive_setup,
        label="MAX",
        adapter_factory=lambda cfg: MaxAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["MAX_BOT_TOKEN"],
        install_hint="Run `hermes setup` to configure MAX. Token from business.max.ru → Чат-боты → Расширенные настройки.",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="MAX_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="MAX_ALLOWED_USERS",
        allow_all_env="MAX_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🟠",
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are communicating via MAX messenger (Russia). "
            "Use plain text by default. Keep responses concise; "
            "MAX has a 4000-character per-message limit."
        ),
    )