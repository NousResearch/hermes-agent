"""Matrix gateway adapter (any homeserver, via mautrix; optional E2EE with ``mautrix[encryption]``).

Env vars (config.yaml ``matrix:`` keys alias several — env wins):
  MATRIX_HOMESERVER, MATRIX_ACCESS_TOKEN (preferred) | MATRIX_USER_ID + MATRIX_PASSWORD;
  MATRIX_E2EE_MODE off|optional|required (legacy MATRIX_ENCRYPTION=true => required);
  MATRIX_DEVICE_ID (stable E2EE device), MATRIX_RECOVERY_KEY (cross-signing after key rotation),
  MATRIX_RECOVERY_KEY_OUTPUT_FILE (one-time 0600 write of a bootstrapped key), MATRIX_PROXY;
  MATRIX_ALLOWED_USERS, MATRIX_ALLOWED_ROOMS (whitelist; DMs exempt), MATRIX_IGNORE_USER_PATTERNS
  (regexes for bridge ghosts), MATRIX_HOME_ROOM (cron delivery), MATRIX_REACTIONS (default true);
  MATRIX_REQUIRE_MENTION (default true), MATRIX_THREAD_REQUIRE_MENTION, MATRIX_FREE_RESPONSE_ROOMS,
  MATRIX_PROCESS_NOTICES, MATRIX_ALLOW_ROOM_MENTIONS, MATRIX_ALLOW_PUBLIC_ROOMS (all default false);
  MATRIX_AUTO_THREAD (default true), MATRIX_DM_AUTO_THREAD, MATRIX_DM_MENTION_THREADS,
  MATRIX_SESSION_SCOPE auto|room|thread; MATRIX_MAX_MESSAGE_LENGTH (default 16000),
  MATRIX_MAX_MEDIA_BYTES, MATRIX_ROOM_IDENTITY_TTL_SECONDS; MATRIX_APPROVAL_REQUIRE_SENDER (default
  true), MATRIX_APPROVAL_TIMEOUT_SECONDS (default 300); MATRIX_TOOLS_ALLOW_{REDACTION,INVITES,ROOM_CREATE}.
"""

from __future__ import annotations

import asyncio
import array
import inspect
from contextlib import suppress
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import urljoin, urlsplit, urlunsplit
from dataclasses import dataclass, field

from html import escape as _html_escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Optional, Set

from agent.secret_scope import UnscopedSecretError, get_secret

try:
    from mautrix.types import (
        ContentURI, EventID, EventType, PresenceState, RoomCreatePreset, RoomID, TrustState, UserID)
except ImportError:
    # Import-safe stubs without mautrix: check_matrix_requirements() gates production use, but
    # tests exercise adapter methods so the attributes must exist.
    ContentURI = EventID = RoomID = UserID = str  # type: ignore[misc,assignment]

    EventType = type("_EventTypeStub", (), {  # type: ignore[misc,assignment]
        "ROOM_MESSAGE": "m.room.message", "REACTION": "m.reaction",
        "ROOM_ENCRYPTED": "m.room.encrypted", "ROOM_NAME": "m.room.name"})
    PresenceState = type("_PresenceStateStub", (), {  # type: ignore[misc,assignment]
        "ONLINE": "online", "OFFLINE": "offline", "UNAVAILABLE": "unavailable"})
    RoomCreatePreset = type("_RoomCreatePresetStub", (), {  # type: ignore[misc,assignment]
        "PRIVATE": "private_chat", "PUBLIC": "public_chat", "TRUSTED_PRIVATE": "trusted_private_chat"})
    TrustState = type("_TrustStateStub", (), {"UNVERIFIED": 0, "VERIFIED": 1})  # type: ignore[misc,assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    gateway_trust_env, BasePlatformAdapter, MessageEvent, MessageType, ProcessingOutcome,
    SendResult, resolve_proxy_url, proxy_kwargs_for_aiohttp, _ssrf_redirect_guard)
from gateway.platforms.helpers import ThreadParticipationTracker

logger = logging.getLogger(__name__)

_MATRIX_VOICE_WAVEFORM_BINS = 30


def _run_media_tool(cmd: list, *, timeout: int, text: bool = False):
    """Run ffmpeg/ffprobe with captured output and no stdin."""
    return subprocess.run(cmd, capture_output=True, text=text, timeout=timeout, stdin=subprocess.DEVNULL)


def _matrix_voice_metadata_for_file(path: Path) -> Dict[str, Any]:
    """Best-effort duration + MSC1767 waveform for voice bubbles; must work without ffprobe/ffmpeg."""
    metadata: Dict[str, Any] = {}
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = _run_media_tool(
                [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of",
                 "default=noprint_wrappers=1:nokey=1", str(path)], timeout=10, text=True)
            if result.returncode == 0:
                duration = float((result.stdout or "").strip() or 0)
                if duration > 0:
                    metadata["duration"] = int(duration * 1000)
        except Exception:
            logger.debug("Matrix: failed to probe voice duration for %s", path, exc_info=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            result = _run_media_tool(
                [ffmpeg, "-v", "error", "-i", str(path), "-ac", "1", "-ar", "8000", "-f", "s16le", "-"], timeout=15)
            if result.returncode == 0 and result.stdout:
                samples = array.array("h")
                samples.frombytes(result.stdout)
                if sys.byteorder != "little":
                    samples.byteswap()
                if samples:
                    count, bins = len(samples), _MATRIX_VOICE_WAVEFORM_BINS
                    waveform = []
                    for idx in range(bins):
                        start = idx * count // bins
                        peak = max(abs(v) for v in samples[start:max(start + 1, (idx + 1) * count // bins)])
                        waveform.append(min(1024, int(peak / 32767 * 1024)))
                    metadata["waveform"] = waveform
        except Exception:
            logger.debug("Matrix: failed to build voice waveform for %s", path, exc_info=True)
    return metadata

def _matrix_transcode_voice_to_ogg(path: str) -> Optional[str]:
    """Transcode to a NEW temp .ogg (caller owns cleanup); None if ffmpeg is missing/fails.
    Blocking subprocess work — call via ``asyncio.to_thread`` from async code."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    import tempfile
    fd, ogg_path = tempfile.mkstemp(prefix="matrix_voice_", suffix=".ogg")
    os.close(fd)
    try:
        result = _run_media_tool(
            [ffmpeg, "-v", "error", "-y", "-i", str(path), "-acodec", "libopus", "-ac", "1", "-b:a", "48k",
             "-vbr", "on", "-application", "voip", "-compression_level", "10", ogg_path],
            timeout=30)
        if result.returncode == 0 and os.path.getsize(ogg_path) > 0:
            return ogg_path
    except Exception:
        logger.debug("Matrix: voice transcode to Ogg/Opus failed for %s", path, exc_info=True)
    with suppress(OSError):
        os.unlink(ogg_path)
    return None


_MATRIX_BANG_COMMAND_RE = re.compile(r"^!([A-Za-z][A-Za-z0-9_-]*)(?=$|\s)(.*)$", re.DOTALL)


def _resolve_matrix_bang_command(name: str) -> str | None:
    """Resolve a ``!command`` token (Matrix clients reserve ``/``) to a dispatchable token.
    Only known gateway/skill commands resolve, so ordinary exclamations stay chat text. Returns
    whichever candidate resolved — raw lowercased first, then ``_``→``-`` — never a forced
    canonical form: aliases pass through for the dispatcher."""
    if not name:
        return None
    candidates = list(dict.fromkeys((name.lower(), name.lower().replace("_", "-"))))
    try:
        from hermes_cli.commands import is_gateway_known_command
        for candidate in candidates:
            if is_gateway_known_command(candidate):
                return candidate
    except Exception:
        logger.debug("Matrix: is_gateway_known_command failed for %r", name, exc_info=True)
    try:
        from agent.skill_commands import get_skill_commands
        skill_commands = get_skill_commands() or {}  # keys are slash-prefixed ("/arxiv")
        for candidate in candidates:
            if f"/{candidate}" in skill_commands:
                return candidate
    except Exception:
        logger.debug("Matrix: get_skill_commands failed for %r", name, exc_info=True)
    return None


def _normalize_matrix_bang_command(text: str) -> str:
    """Convert Matrix ``!command`` aliases to normal Hermes ``/command`` text."""
    if not text or not text.startswith("!"):
        return text
    match = _MATRIX_BANG_COMMAND_RE.match(text)
    resolved = _resolve_matrix_bang_command(match.group(1)) if match else None
    if resolved is None:
        return text
    return f"/{resolved}{match.group(2) or ''}"


# Reply fallback prefix: "> <@alice:example.org> quoted\n> more\n\nactual reply".
_MATRIX_REPLY_FALLBACK_PILL_RE = re.compile(r"^>\s*<(@[^>]+)>\s*(.*)$")


def _extract_reply_fallback(body: str) -> tuple[Optional[str], Optional[str]]:
    """Return (quoted_text, author_mxid) from the inline reply fallback; author from the first-line pill."""
    if not body or not body.startswith("> "):
        return None, None
    quoted_lines: list[str] = []
    author_id: Optional[str] = None
    for line in body.split("\n"):
        if not line.startswith("> "):
            break
        content = line[2:]
        if author_id is None:
            pill_match = _MATRIX_REPLY_FALLBACK_PILL_RE.match(line)
            if pill_match:
                author_id = pill_match.group(1)
                content = pill_match.group(2)  # drop the pill from the visible quote
        quoted_lines.append(content)
    quoted_text = "\n".join(quoted_lines).strip() or None
    return quoted_text, author_id


def _strip_reply_fallback(body: str) -> str:
    """Strip the inline ``> quote\\n\\nreply`` fallback prefix; unchanged if absent."""
    if not body or not body.startswith("> "):
        return body
    stripped = []
    past_fallback = False
    for line in body.split("\n"):
        if not past_fallback:
            if line.startswith("> ") or line == ">":
                continue
            past_fallback = True
            if line == "":
                continue
        stripped.append(line)
    return "\n".join(stripped) if stripped else body


class _MatrixHtmlSanitizer(HTMLParser):
    """Allowlist sanitizer for Matrix-compatible formatted HTML."""

    _ALLOWED_TAGS = {
        "a", "b", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "li", "ol",
        "p", "pre", "s", "strike", "strong", "table", "tbody", "td", "th", "thead", "tr", "ul"}
    _VOID_TAGS = {"br", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._parts: list[str] = []
        self._skip_depth = 0

    @staticmethod
    def _safe_url(value: str) -> str:
        stripped = re.sub(r"[\x00-\x1f\x7f]+", "", value or "").strip()
        match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", stripped)
        scheme = match.group(1).lower() if match else ""
        if scheme and scheme not in {"http", "https", "matrix", "mailto"}:
            return ""
        return stripped

    def _safe_attrs(self, tag: str, attrs: list[tuple[str, str | None]]) -> str:
        safe: list[str] = []
        for key, value in attrs:
            attr = str(key or "").lower()
            raw_value = "" if value is None else str(value)
            if tag == "a" and attr == "href":
                href = self._safe_url(raw_value)
                if href:
                    safe.append(f' href="{_html_escape(href, quote=True)}"')
            elif tag == "code" and attr == "class" and re.fullmatch(r"language-[A-Za-z0-9_+.-]{1,64}", raw_value):
                safe.append(f' class="{_html_escape(raw_value, quote=True)}"')
        return "".join(safe)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._skip_depth += 1
        elif not self._skip_depth and tag in self._ALLOWED_TAGS:
            self._parts.append(f"<{tag}>" if tag in self._VOID_TAGS else f"<{tag}{self._safe_attrs(tag, attrs)}>")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth or tag not in self._ALLOWED_TAGS or tag in self._VOID_TAGS:
            return
        self._parts.append(f"</{tag}>")

    def _emit(self, text: str) -> None:
        if not self._skip_depth:
            self._parts.append(text)

    def handle_data(self, data: str) -> None:
        self._emit(_html_escape(data))

    def handle_entityref(self, name: str) -> None:
        self._emit(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._emit(f"&#{name};")

    def get_html(self) -> str:
        return "".join(self._parts)


@dataclass(frozen=True)
class MatrixRoomIdentity:
    """Resolved Matrix room identity for routing and prompt context."""
    room_id: str
    room_name: str | None
    room_topic: str | None
    canonical_alias: str | None
    server_name: str | None
    joined_member_count: int | None
    is_direct_account_data: bool
    display_name: str
    has_explicit_name: bool
    chat_type: str
    conflict: bool = False


@dataclass
class _MatrixApprovalPrompt:
    """Pending reaction-based exec approval prompt."""
    session_key: str
    chat_id: str
    message_id: str
    resolved: bool = False
    requester_user_id: str | None = None
    expires_at: float | None = None
    bot_reaction_events: dict[str, str] = field(default_factory=dict, init=False)  # emoji -> event_id


@dataclass
class _MatrixPickerPrompt:
    """Pending reaction-based picker; ``choices`` maps emoji -> selection, ``on_selected`` is the callback."""
    chat_id: str
    message_id: str
    session_key: str
    choices: dict
    on_selected: Any
    requester_user_id: str | None = None
    expires_at: float | None = None
    resolved: bool = False
    bot_reaction_events: dict[str, str] = field(default_factory=dict)


_MatrixModelPickerPrompt = _MatrixChoicePickerPrompt = _MatrixPickerPrompt


# Spec allows ~65 KB events; 4000 was too small (split Markdown tables mid-row).
# Matrix message size limit. The spec allows large events (~65 KB), but very large bodies can render poorly
# in some clients. The previous 4,000-char default was overly conservative and split Markdown tables mid-row
# (#53026).
DEFAULT_MAX_MESSAGE_LENGTH = 16000
MATRIX_MAX_MESSAGE_LENGTH_CEILING = 65535


def _resolve_max_message_length(config) -> int:
    """Resolve outbound chunk size from config, env, or plugin registry."""
    raw = (getattr(config, "extra", {}) or {}).get("max_message_length")
    if raw is None:
        raw = os.getenv("MATRIX_MAX_MESSAGE_LENGTH")
    if raw is None:
        with suppress(Exception):
            from gateway.platform_registry import platform_registry
            entry = platform_registry.get("matrix")
            if entry and entry.max_message_length:
                raw = entry.max_message_length
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_MESSAGE_LENGTH
    return max(500, min(value, MATRIX_MAX_MESSAGE_LENGTH_CEILING))


# E2EE store dir is resolved per adapter in connect() (``_resolve_store_dir``), NOT at module scope:
# the multiplex gateway imports this once and a module constant would collide every profile's Olm
# identity in one crypto.db.
# Store directory for E2EE keys and sync state. Mirrors the pairing-store fix (a6397c379). See #89168.
from hermes_constants import get_hermes_dir as _get_hermes_dir

_STARTUP_GRACE_SECONDS = 5  # ignore messages older than this many seconds before startup

_OUTBOUND_MENTION_RE = re.compile(r"(?<![\w/])(@[0-9A-Za-z._=/-]+:[0-9A-Za-z.-]+(?::\d+)?)")

_E2EE_INSTALL_HINT = "Install with: pip install 'mautrix[encryption]' asyncpg aiosqlite  (requires libolm C library)"

_MATRIX_IMAGE_FILENAME_EXTS = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".heic", ".heif", ".avif"})
_MATRIX_MEDIA_FILENAME_EXTS = frozenset({
    ".ogg", ".oga", ".opus", ".m4a", ".mp3", ".wav", ".flac", ".aac", ".amr", ".mp4", ".webm", ".mov", ".mkv"})
# Keycap 1-9, 🔟; choice pickers (/reasoning, /fast) can need 12 slots, so they add 🅰️ 🅱️.
_MATRIX_MODEL_PICKER_REACTIONS = tuple(f"{d}\ufe0f\u20e3" for d in "123456789") + ("\U0001f51f",)
_MATRIX_CHOICE_PICKER_REACTIONS = _MATRIX_MODEL_PICKER_REACTIONS + ("\U0001f170\ufe0f", "\U0001f171\ufe0f")

def _looks_like_matrix_image_filename(text: str) -> bool:
    """True when an m.image body is just the uploaded filename (no caption) — not user text."""
    return _looks_like_transport_filename(text, "image/", _MATRIX_IMAGE_FILENAME_EXTS)


def _looks_like_transport_filename(text: str, mime_prefixes, exts: frozenset, reject_spaces: bool = False) -> bool:
    """Bare single-token filename with a known media extension or a matching guessed MIME type."""
    candidate = str(text or "").strip()
    if not candidate or "\n" in candidate or candidate.endswith("/"):
        return False
    # A genuine caption essentially always contains whitespace; a bare transport filename does not.
    if reject_spaces and any(ch.isspace() for ch in candidate):
        return False
    if Path(candidate).name != candidate:
        return False
    suffix = Path(candidate).suffix.lower()
    if not suffix:
        return False
    guessed_type, _ = mimetypes.guess_type(candidate)
    return bool(guessed_type and guessed_type.startswith(mime_prefixes)) or suffix in exts


def _looks_like_matrix_media_filename(text: str) -> bool:
    """True when an m.audio/m.file/m.video body is just the uploaded filename (no caption)."""
    return _looks_like_transport_filename(text, ("audio/", "video/"), _MATRIX_MEDIA_FILENAME_EXTS, True)


def _is_bare_media_filename(msgtype: str, body: str) -> bool:
    """True when a media event body is only the uploaded filename for its msgtype."""
    if msgtype == "m.image":
        return _looks_like_matrix_image_filename(body)
    return msgtype in ("m.audio", "m.file", "m.video") and _looks_like_matrix_media_filename(body)


def _matrix_event_timestamp_seconds(event: Any) -> float:
    """Return a Matrix event timestamp in seconds, accepting ms or sec values."""
    try:
        ts = float(getattr(event, "timestamp", None) or getattr(event, "server_timestamp", None) or 0)
    except (TypeError, ValueError):
        return 0.0
    # origin_server_ts is ms; some SDK objects/fakes expose seconds — keep both sane.
    return ts / 1000.0 if ts > 10_000_000_000 else ts


def _create_matrix_session(proxy_url: str | None):
    """ClientSession whose proxy applies to *all* requests: mautrix's ``HTTPAPI._send()`` never
    forwards per-request ``proxy=``, so it must be session-level (``proxy=`` for HTTP(S),
    ``ProxyConnector`` for SOCKS); with no proxy, ``trust_env`` honours HTTP(S)_PROXY."""
    import aiohttp
    if not proxy_url:
        return aiohttp.ClientSession(trust_env=gateway_trust_env())
    if proxy_url.split("://")[0].lower().startswith("socks"):
        try:
            from aiohttp_socks import ProxyConnector
            return aiohttp.ClientSession(connector=ProxyConnector.from_url(proxy_url, rdns=True))
        except ImportError:
            logger.warning(
                "aiohttp_socks not installed — SOCKS proxy %s ignored. Run: pip install aiohttp-socks", proxy_url)
            return aiohttp.ClientSession(trust_env=gateway_trust_env())
    return aiohttp.ClientSession(proxy=proxy_url)


def _check_e2ee_deps() -> bool:
    """True if all four E2EE deps import: olm, PgCryptoStore (also drives sqlite), asyncpg, aiosqlite.
    Without all four, encrypted rooms fail at connect with ``No module named 'asyncpg'``.

    Verifies python-olm (via mautrix.crypto.OlmMachine), the SQLite crypto store backend
    (mautrix.crypto.store.asyncpg.PgCryptoStore — yes, the PgCryptoStore class also drives the sqlite
    backend in mautrix 0.21), and the database drivers actually used at connect time (``asyncpg`` for the
    underlying upgrade_table machinery, ``aiosqlite`` for the ``sqlite:///`` URL we pass to
    ``Database.create``). See #31116.
    """
    try:
        from mautrix.crypto import OlmMachine  # noqa: F401
        from mautrix.crypto.store.asyncpg import PgCryptoStore  # noqa: F401
        import asyncpg  # noqa: F401
        import aiosqlite  # noqa: F401
        return True
    except (ImportError, AttributeError):
        return False


def _normalize_e2ee_mode(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in ("required", "require", "true", "1", "yes", "on"):
        return "required"
    if raw in ("optional", "prefer", "preferred"):
        return "optional"
    return "off"


def _resolve_e2ee_mode(extra: Optional[Dict[str, Any]] = None) -> str:
    """Resolve E2EE mode with MATRIX_ENCRYPTION backwards compatibility."""
    extra = extra or {}
    explicit = extra.get("e2ee_mode") or os.getenv("MATRIX_E2EE_MODE", "")
    if explicit:
        return _normalize_e2ee_mode(explicit)
    legacy_enabled = extra.get("encryption", _env_truthy("MATRIX_ENCRYPTION"))
    return "required" if legacy_enabled else "off"


def _env_truthy(name: str, default: str = "") -> bool:
    """Return True when the env var is one of true/1/yes (case-insensitive)."""
    return os.getenv(name, default).lower() in ("true", "1", "yes")


def _env_number(name: str, default, cast):
    """Parse a numeric env var, falling back to *default* on ValueError."""
    try:
        return cast(os.getenv(name, str(default)))
    except ValueError:
        return default


def _csv_set(raw: Any) -> Set[str]:
    """Normalize a comma-separated string or list into a set of stripped tokens."""
    if isinstance(raw, list):
        return {str(r).strip() for r in raw if str(r).strip()}
    return {r.strip() for r in str(raw).split(",") if r.strip()}


def _extra_csv_set(config, key: str, env_name: str) -> Set[str]:
    """Resolve a room/user list from config.extra[key], else the env var."""
    raw = config.extra.get(key)
    if raw is None:
        raw = os.getenv(env_name, "")
    return _csv_set(raw)


def _recovery_key_output_path() -> Optional[Path]:
    output_file = os.getenv("MATRIX_RECOVERY_KEY_OUTPUT_FILE", "").strip()
    return Path(output_file).expanduser() if output_file else None


def _write_matrix_recovery_key_output_file(recovery_key: str) -> Optional[Path]:
    """Write a generated recovery key to MATRIX_RECOVERY_KEY_OUTPUT_FILE (0600, never overwritten)."""
    path = _recovery_key_output_path()
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(recovery_key)
            fh.write("\n")
    except Exception:
        with suppress(OSError):
            os.close(fd)
        raise
    return path


def _get_matrix_recovery_key_output_target() -> tuple[Optional[Path], str]:
    """Return a usable one-time recovery-key output path, or a redacted reason."""
    path = _recovery_key_output_path()
    if path is None:
        return None, "not_configured"
    if path.exists():
        return None, "exists"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return None, f"unusable: {exc}"
    return path, ""


def _handle_generated_matrix_recovery_key(mxid: str, recovery_key: str) -> None:
    """Handle a freshly generated Matrix recovery key without logging it."""
    try:
        output_path = _write_matrix_recovery_key_output_file(recovery_key)
    except FileExistsError:
        logger.warning(
            "Matrix: bootstrapped cross-signing for %s. Recovery key output file "
            "already exists; refusing to overwrite. Store the generated key "
            "securely and set MATRIX_RECOVERY_KEY for future restarts.", mxid)
        return
    except Exception as exc:
        logger.warning(
            "Matrix: bootstrapped cross-signing for %s, but failed to write "
            "MATRIX_RECOVERY_KEY_OUTPUT_FILE: %s. Store the generated key "
            "securely and set MATRIX_RECOVERY_KEY for future restarts.", mxid, exc)
        return
    if output_path:
        logger.warning(
            "Matrix: bootstrapped cross-signing for %s. A new recovery key was written to %s with mode 0600. Move it "
            "to your secret store and set MATRIX_RECOVERY_KEY for future restarts.",
            mxid, output_path)
    else:
        logger.warning(
            "Matrix: bootstrapped cross-signing for %s. A new recovery key was generated but will "
            "not be logged. Set MATRIX_RECOVERY_KEY_OUTPUT_FILE to write it once with mode 0600, "
            "or configure MATRIX_RECOVERY_KEY from your Matrix client before future restarts.",
            mxid)


def _scoped_recovery_key() -> str:
    """MATRIX_RECOVERY_KEY via the profile-scoped secret store (see _startup_env_secret): a bare
    os.getenv under multiplex resolves the default profile's key and verification fails with
    "Key MAC does not match".

    We read through :func:`get_secret`, which is scope-aware. An *unscoped* read under multiplex (e.g. the
    default-profile startup loop) raises ``UnscopedSecretError``; in that context ``os.environ`` is that
    profile's own value, so we fall back to it — mirroring the established Slack app-token pattern (#59739).
    """
    return _startup_env_secret("MATRIX_RECOVERY_KEY")


def _sanitize_matrix_html(html: str) -> str:
    sanitizer = _MatrixHtmlSanitizer()
    try:
        sanitizer.feed(html or "")
        sanitizer.close()
        return sanitizer.get_html()
    except Exception:
        return _html_escape(html or "")


def _redact_url_for_log(url: str) -> str:
    """Strip query/fragment from URLs before logging signed media links."""
    try:
        parts = urlsplit(str(url))
        if not parts.scheme and not parts.netloc:
            return str(url).split("?", 1)[0].split("#", 1)[0]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:
        return "<url>"


def _pre_sanitize_matrix_markdown(text: str) -> str:
    """Remove unsafe raw HTML before Markdown conversion can escape it."""
    result = re.sub(r"(?is)<\s*(script|style)\b[^>]*>.*?<\s*/\s*\1\s*>", "", text or "")
    result = re.sub(r"""(?is)\s+on[a-z0-9_-]+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""", "", result)
    return re.sub(
        r"""(?is)\s+(href|src)\s*=\s*("[^"]*(?:javascript|data|vbscript):[^"]*"|'[^']*(?:javascript|data|vbscript):[^']*'|[^\s>]*(?:javascript|data|vbscript):[^\s>]*)""",
        "", result)


def _startup_env_secret(name: str) -> str:
    """Scope-aware credential read: a scoped miss is empty (never borrow the process env);
    only an UNSCOPED read (default-profile startup loop) falls back to os.environ.

    See #59739.
    """
    try:
        return (get_secret(name) or "").strip()
    except UnscopedSecretError:
        return os.getenv(name, "").strip()


def matrix_deps_present() -> bool:
    """PASSIVE registry ``check_fn`` — must never install; ``ensure_matrix_deps`` is the installer.

    Registry ``check_fn`` — called from status displays and config loading, so it must never install
    anything. The ACTIVE lazy-installer (``check_matrix_requirements``) is registered as ``ensure_deps_fn``
    and runs from ``create_adapter()`` when this returns False (#79812).
    """
    try:
        from tools.lazy_deps import is_available
        return is_available("platform.matrix")
    except Exception:  # pragma: no cover — defensive
        return False


def check_matrix_requirements() -> bool:
    """Credentials + deps answer for setup/status callers (credentials must NOT gate the installer)."""
    token = _startup_env_secret("MATRIX_ACCESS_TOKEN")
    password = _startup_env_secret("MATRIX_PASSWORD")
    homeserver = _startup_env_secret("MATRIX_HOMESERVER")
    if not token and not password:
        logger.debug("Matrix: neither MATRIX_ACCESS_TOKEN nor MATRIX_PASSWORD set")
        return False
    if not homeserver:
        logger.warning("Matrix: MATRIX_HOMESERVER not set")
        return False
    return ensure_matrix_deps()


def ensure_matrix_deps() -> bool:
    """ACTIVE deps-only installer (registry ``ensure_deps_fn``); rebinds the type globals. Installs the
    whole ``platform.matrix`` group when ANY declared package is missing — short-circuiting on
    ``import mautrix`` left asyncpg/aiosqlite uninstalled forever.

    Lazy-installs the full ``platform.matrix`` feature group via ``tools.lazy_deps.ensure_and_bind``
    whenever any of the declared packages (mautrix, Markdown, aiosqlite, asyncpg, aiohttp-socks) is missing
    — not just mautrix itself. Previously this short-circuited on ``import mautrix``, which left the other
    four packages uninstalled forever and broke E2EE connect with ``No module named 'asyncpg'`` (#31116).
    """
    try:
        from tools.lazy_deps import feature_missing, ensure_and_bind
        missing = feature_missing("platform.matrix")
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("Matrix: lazy_deps lookup failed: %s", exc)
        missing = ()
        ensure_and_bind = None  # type: ignore[assignment]
    if ensure_and_bind is None:
        return False
    if missing:
        def _import():
            from mautrix.types import (
                ContentURI, EventID, EventType, PresenceState, RoomCreatePreset, RoomID, TrustState, UserID)
            return {
                "ContentURI": ContentURI, "EventID": EventID, "EventType": EventType, "PresenceState": PresenceState,
                "RoomCreatePreset": RoomCreatePreset, "RoomID": RoomID, "TrustState": TrustState, "UserID": UserID}
        if not ensure_and_bind("platform.matrix", _import, globals(), prompt=False):
            logger.warning(
                "Matrix: required packages not installed (%s). Run: pip install "
                "'mautrix[encryption]' asyncpg aiosqlite Markdown aiohttp-socks",
                ", ".join(missing) if missing else "platform.matrix")
            return False
    e2ee_mode = _resolve_e2ee_mode()
    if e2ee_mode == "required" and not _check_e2ee_deps():
        logger.error(
            "Matrix: E2EE is required but dependencies are missing. %s. Without this, encrypted "
            "rooms will not work. Set MATRIX_E2EE_MODE=off to disable E2EE.",
            _E2EE_INSTALL_HINT)
        return False
    if e2ee_mode == "optional" and not _check_e2ee_deps():
        logger.warning("Matrix: E2EE optional but dependencies are missing. %s", _E2EE_INSTALL_HINT)
    return True


class _CryptoStateStore:
    """StateStore shim for OlmMachine (MemoryStateStore lacks is_encrypted/get_encryption_info/
    find_shared_rooms); falls back to a homeserver state query when the store has no info."""

    def __init__(self, client_state_store: Any, joined_rooms: set, client=None):
        self._ss = client_state_store
        self._joined_rooms = joined_rooms
        self._client = client
        # MemoryStateStore has no set_encryption_info, so cache homeserver answers here.
        self._enc_info_cache: dict = {}

    async def is_encrypted(self, room_id: str) -> bool:
        return (await self.get_encryption_info(room_id)) is not None

    async def get_encryption_info(self, room_id: str):
        info = await self._ss.get_encryption_info(room_id) if hasattr(self._ss, "get_encryption_info") else None
        if info is not None:
            return info
        if room_id in self._enc_info_cache:
            return self._enc_info_cache[room_id]
        if self._client is None:
            return None
        try:
            from mautrix.types import EventType as _ET, RoomEncryptionStateEventContent as _Enc, RoomID as _RID
            raw = await self._client.get_state_event(_RID(room_id), _ET.ROOM_ENCRYPTION)
        except Exception as exc:
            logger.debug("Matrix: homeserver encryption-info query failed for %s: %s", room_id, exc)
            return None
        if not raw:
            return None
        content = raw if isinstance(raw, _Enc) else _Enc.deserialize(
            raw.serialize() if hasattr(raw, "serialize") else raw)
        if hasattr(self._ss, "set_encryption_info"):
            with suppress(Exception):
                await self._ss.set_encryption_info(_RID(room_id), content)
        self._enc_info_cache[room_id] = content
        return content

    async def find_shared_rooms(self, user_id: str) -> list:
        return list(self._joined_rooms)  # all joined rooms: correct for a single-user bot


class MatrixAdapter(BasePlatformAdapter):
    """Gateway adapter for Matrix (any homeserver)."""

    supports_code_blocks = True  # Matrix renders fenced code blocks (HTML/markdown)
    splits_long_messages = True  # send() chunks via truncate_message(max_message_length)
    typed_command_prefix = "!"  # clients reserve typed "/" for local commands; "!command" always reaches Hermes
    # Class-level defaults keep object.__new__-built test instances working.
    max_message_length = DEFAULT_MAX_MESSAGE_LENGTH
    _split_threshold = DEFAULT_MAX_MESSAGE_LENGTH - 100

    def _resolve_store_dir(self) -> Path:
        """Pin the crypto-store dir to the active profile (connect() runs inside the profile
        scope); cached so later out-of-scope reads report the store actually in use."""
        self._store_dir = _get_hermes_dir("platforms/matrix/store", "matrix/store")
        return self._store_dir

    @property
    def _crypto_db_path(self) -> Path:
        return (self._store_dir or _get_hermes_dir("platforms/matrix/store", "matrix/store")) / "crypto.db"

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MATRIX)
        self.max_message_length = _resolve_max_message_length(config)
        self.MAX_MESSAGE_LENGTH = self.max_message_length  # mirrors other adapters for tooling
        # A chunk near the outbound limit almost certainly has a continuation.
        self._split_threshold = max(100, self.max_message_length - 100)
        self._homeserver: str = (config.extra.get("homeserver", "") or os.getenv("MATRIX_HOMESERVER", "")).rstrip("/")
        self._access_token: str = config.token or _startup_env_secret("MATRIX_ACCESS_TOKEN")
        self._user_id: str = config.extra.get("user_id", "") or os.getenv("MATRIX_USER_ID", "")
        self._password: str = config.extra.get("password", "") or _startup_env_secret("MATRIX_PASSWORD")
        self._e2ee_mode: str = _resolve_e2ee_mode(config.extra)
        self._encryption: bool = self._e2ee_mode != "off"
        self._device_id: str = config.extra.get("device_id", "") or os.getenv("MATRIX_DEVICE_ID", "")
        self._device_id_unverified: bool = False
        self._client: Any = None  # mautrix.client.Client
        self._crypto_db: Any = None  # mautrix.util.async_db.Database
        self._store_dir: Optional[Path] = None  # pinned per profile in connect()
        self._sync_task: Optional[asyncio.Task] = None
        self._invite_join_tasks: Dict[str, asyncio.Task] = {}
        self._closing = False
        self._startup_ts: float = 0.0
        self._reset_clock_skew_detector()
        self._last_sync_ts: float = 0.0
        self._dm_rooms: Dict[str, bool] = {}
        self._room_identities: Dict[str, MatrixRoomIdentity] = {}
        self._room_identity_cached_at: Dict[str, float] = {}
        self._room_identity_ttl_seconds = _env_number("MATRIX_ROOM_IDENTITY_TTL_SECONDS", 60.0, float)
        self._room_identity_cache_max = 256
        self._joined_rooms: Set[str] = set()
        from collections import deque
        self._processed_events: deque = deque(maxlen=1000)  # event dedup, newest kept
        self._processed_events_set: set = set()
        self._threads = ThreadParticipationTracker("matrix")  # require_mention bypass
        self._require_mention: bool = self._parse_require_mention(config)
        self._thread_require_mention: bool = self._parse_thread_require_mention(config)
        self._free_rooms: Set[str] = _extra_csv_set(config, "free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS")
        # If non-empty, bot ONLY responds in these rooms (whitelist); DMs exempt.
        self._allowed_rooms: Set[str] = _extra_csv_set(config, "allowed_rooms", "MATRIX_ALLOWED_ROOMS")
        self._allow_room_mentions: bool = _env_truthy("MATRIX_ALLOW_ROOM_MENTIONS", "false")
        self._auto_thread: bool = _env_truthy("MATRIX_AUTO_THREAD", "true")
        self._dm_auto_thread: bool = _env_truthy("MATRIX_DM_AUTO_THREAD", "false")
        self._dm_mention_threads: bool = _env_truthy("MATRIX_DM_MENTION_THREADS", "false")
        raw_session_scope = os.getenv("MATRIX_SESSION_SCOPE", "auto").strip().lower()
        self._matrix_session_scope = raw_session_scope if raw_session_scope in {"auto", "room", "thread"} else "auto"
        self._process_notices: bool = _env_truthy("MATRIX_PROCESS_NOTICES", "false")
        self._reactions_enabled: bool = os.getenv("MATRIX_REACTIONS", "true").lower() not in {"false", "0", "no"}
        self._pending_reactions: dict[tuple[str, str], str] = {}
        # Let the final message land before redacting reactions ("missing event" in some
        # clients). 5s is empirically safe; if it must be tunable, use config.yaml not env.
        self._reaction_redaction_delay_seconds = 5.0
        self._reaction_redaction_tasks: Set[asyncio.Task] = set()
        self._proxy_url: str | None = resolve_proxy_url(platform_env_var="MATRIX_PROXY")
        if self._proxy_url:
            logger.info("Matrix: proxy configured — %s", self._proxy_url)
        self._max_media_bytes = _env_number("MATRIX_MAX_MEDIA_BYTES", 100 * 1024 * 1024, int)
        # Text batching merges client-side splits (~4000 chars) of one long message.
        self._text_batch_delay_seconds = float(os.getenv("HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS", "0.6"))
        self._text_batch_split_delay_seconds = float(os.getenv("HERMES_MATRIX_TEXT_BATCH_SPLIT_DELAY_SECONDS", "2.0"))
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}
        self._approval_reaction_map = {
            "✅": "once", "🌀": "session", "♾️": "always", "♾": "always", "\u267e\ufe0f": "always",
            "\u267e": "always", "❌": "deny", "❎": "deny"}
        self._approval_prompts_by_event: Dict[str, _MatrixApprovalPrompt] = {}
        self._approval_prompt_by_session: Dict[str, str] = {}
        self._approval_require_sender: bool = _env_truthy("MATRIX_APPROVAL_REQUIRE_SENDER", "true")
        self._approval_timeout_seconds = _env_number("MATRIX_APPROVAL_TIMEOUT_SECONDS", 300, int)
        self._model_picker_prompts_by_event: Dict[str, _MatrixPickerPrompt] = {}
        self._choice_picker_prompts_by_event: Dict[str, _MatrixPickerPrompt] = {}
        self._allowed_user_ids: Set[str] = _csv_set(os.getenv("MATRIX_ALLOWED_USERS", ""))
        self._allowed_room_ids: Set[str] = set(self._allowed_rooms)
        self._ignored_user_patterns: list[re.Pattern[str]] = []
        for pattern in (p.strip() for p in os.getenv("MATRIX_IGNORE_USER_PATTERNS", "").split(",") if p.strip()):
            try:
                self._ignored_user_patterns.append(re.compile(pattern))
            except re.error as exc:
                logger.warning("Matrix: ignoring invalid MATRIX_IGNORE_USER_PATTERNS entry %r: %s", pattern, exc)

    def _is_duplicate_event(self, event_id) -> bool:
        """Return True if this event was already processed. Tracks the ID otherwise."""
        if not event_id:
            return False
        if event_id in self._processed_events_set:
            return True
        if len(self._processed_events) == self._processed_events.maxlen:
            self._processed_events_set.discard(self._processed_events[0])
        self._processed_events.append(event_id)
        self._processed_events_set.add(event_id)
        return False

    @staticmethod
    def _configured_bool(config, key: str) -> Optional[bool]:
        """Parse a YAML bool / "true"/"off"-style string from config.extra; None if unset."""
        configured = config.extra.get(key)
        if configured is None:
            return None
        if isinstance(configured, bool):
            return configured
        if isinstance(configured, str):
            return configured.lower() not in {"false", "0", "no", "off"}
        return bool(configured)

    @staticmethod
    def _parse_require_mention(config) -> bool:
        """require_mention from config.extra, else MATRIX_REQUIRE_MENTION (default true)."""
        configured = MatrixAdapter._configured_bool(config, "require_mention")
        if configured is not None:
            return configured
        return os.getenv("MATRIX_REQUIRE_MENTION", "true").lower() not in {"false", "0", "no", "off"}

    @staticmethod
    def _parse_thread_require_mention(config) -> bool:
        """thread_require_mention from config.extra, else MATRIX_THREAD_REQUIRE_MENTION (default false)."""
        configured = MatrixAdapter._configured_bool(config, "thread_require_mention")
        if configured is not None:
            return configured
        return os.getenv("MATRIX_THREAD_REQUIRE_MENTION", "false").lower() in {"true", "1", "yes", "on"}

    @staticmethod
    def _extract_server_ed25519(device_keys_obj: Any) -> Optional[str]:
        for kid, kval in (getattr(device_keys_obj, "keys", {}) or {}).items():
            if str(kid).startswith("ed25519:"):
                return str(kval)
        return None

    @staticmethod
    async def _query_own_device_keys(client: Any):
        """query_keys for our own device; the DeviceKeys entry or None."""
        resp = await client.query_keys({client.mxid: [client.device_id]})
        our_user_devices = (getattr(resp, "device_keys", {}) or {}).get(str(client.mxid)) or {}
        return our_user_devices.get(str(client.device_id))

    async def _reverify_keys_after_upload(self, client: Any, local_ed25519: str) -> bool:
        """Re-query the server after share_keys() and verify our ed25519 key matches."""
        if not client.device_id or self._device_id_unverified:
            logger.warning("Matrix: skipping post-upload key verification — device_id not yet established")
            return True
        try:
            dev = await self._query_own_device_keys(client)
            if dev and self._extract_server_ed25519(dev) != local_ed25519:
                logger.error(
                    "Matrix: device %s has immutable identity keys that don't match this "
                    "installation. Generate a new access token with a fresh device.", client.device_id)
                return False
        except Exception as exc:
            logger.error("Matrix: post-upload key verification failed: %s", exc, exc_info=True)
            return False
        return True

    async def _reset_crypto_store_if_device_changed(self, crypto_store: Any, device_id: str) -> bool:
        """Reset the Olm account when the token's device changed; True if reset. The store is keyed
        by user ID, so a new device would inherit the old Olm account whose identity keys can never
        be published under the new device ID."""
        if not device_id:
            return False
        try:
            stored_device_id = await crypto_store.get_device_id()
        except Exception as exc:
            logger.warning("Matrix: could not read stored device ID: %s", exc)
            return False
        if not stored_device_id or stored_device_id == device_id:
            return False
        logger.warning(
            "Matrix: access token belongs to a new device (%s -> %s) — resetting local Olm account "
            "so fresh identity keys are generated for this device", stored_device_id, device_id)
        await crypto_store.delete()
        return True

    async def _migrate_legacy_crypto_pickle(
            self, crypto_store: Any, crypto_db: Any, acct_id: str, pickle_key: str) -> bool:
        """Re-pickle the Olm account under the current pickle key when it changed. The key embeds the
        device ID; an account created before MATRIX_DEVICE_ID was set lives under ``<acct>:default``
        and later fails with BAD_ACCOUNT_KEY (silently disabling optional E2EE). False only when an
        account exists but no key opens it."""
        with suppress(Exception):
            await crypto_store.get_account()
            return True
        from mautrix.crypto.store.asyncpg import PgCryptoStore
        for legacy_key in (f"{acct_id}:default", acct_id):
            if legacy_key == pickle_key:
                continue
            try:
                account = await PgCryptoStore(account_id=acct_id, pickle_key=legacy_key, db=crypto_db).get_account()
            except Exception:
                account = None
            if account is None:
                continue
            # Sessions first, account last: the account is the commit marker (the fast path
            # above short-circuits once it reads), so an interrupted sweep is retried.
            try:
                await self._repickle_crypto_sessions(crypto_db, acct_id, legacy_key, pickle_key)
            except Exception as exc:
                logger.error(
                    "Matrix: pickle key migration failed while re-pickling sessions (%s) — leaving "
                    "the account under the legacy key so the migration is retried on the next start.", exc)
                return False
            await crypto_store.put_account(account)
            logger.info(
                "Matrix: re-pickled crypto store account and sessions under the current pickle key "
                "(device ID was configured after the account was created)")
            return True
        logger.error(
            "Matrix: crypto store account exists but cannot be unpickled with the current or any "
            "legacy pickle key. If MATRIX_DEVICE_ID was changed manually, restore its previous value.")
        return False

    async def _repickle_crypto_sessions(self, crypto_db: Any, acct_id: str, legacy_key: str, pickle_key: str) -> None:
        """Re-pickle olm/megolm sessions too — they share the key; account-only breaks key sharing."""
        import olm as olm_lib
        tables = {
            "crypto_olm_session": olm_lib.Session, "crypto_megolm_inbound_session": olm_lib.InboundGroupSession,
            "crypto_megolm_outbound_session": olm_lib.OutboundGroupSession}
        for table, session_cls in tables.items():
            rows = await crypto_db.fetch(f"SELECT session_id, session FROM {table} WHERE account_id=$1", acct_id)
            for row in rows:
                if row["session"] is None:
                    continue
                pickled = bytes(row["session"])
                with suppress(Exception):
                    session_cls.from_pickle(pickled, pickle_key)
                    continue  # already readable with the current key
                try:
                    session = session_cls.from_pickle(pickled, legacy_key)
                except Exception as exc:
                    # Readable under neither key: leave it inert rather than delete crypto material.
                    logger.warning(
                        "Matrix: %s row %s cannot be unpickled with the current or legacy key; leaving "
                        "it in place, its sessions are unrecoverable: %s", table, row["session_id"], exc)
                    continue
                await crypto_db.execute(
                    f"UPDATE {table} SET session=$1 WHERE account_id=$2 AND session_id=$3",
                    session.pickle(pickle_key), acct_id, row["session_id"])

    async def _verify_device_keys_on_server(self, client: Any, olm: Any) -> bool:
        """True if our device keys are on the server (or were re-uploaded); False ⇒ refuse E2EE."""
        if not client.device_id or self._device_id_unverified:
            logger.warning("Matrix: skipping device key verification — device_id not yet established")
            return True
        try:
            our_keys = await self._query_own_device_keys(client)
        except Exception as exc:
            logger.error("Matrix: cannot verify device keys on server: %s — refusing E2EE", exc, exc_info=True)
            return False
        local_ed25519 = olm.account.identity_keys.get("ed25519")

        async def _reupload(error_fmt: str, *error_args) -> bool:
            try:
                await olm.share_keys()
            except Exception as exc:
                logger.error(error_fmt, *error_args, exc, exc_info=True)
                return False
            return await self._reverify_keys_after_upload(client, local_ed25519)
        if not our_keys:
            logger.warning("Matrix: device keys missing from server — re-uploading")
            olm.account.shared = False
            return await _reupload("Matrix: failed to re-upload device keys: %s")
        if self._extract_server_ed25519(our_keys) == local_ed25519:
            return True
        if olm.account.shared:
            logger.error(
                "Matrix: server has different identity keys for device %s — local crypto state is "
                "stale. Delete %s and restart.", client.device_id, str(self._crypto_db_path))
            return False
        logger.warning("Matrix: server has stale keys for device %s — attempting re-upload", client.device_id)
        with suppress(Exception):
            await client.api.request(
                client.api.Method.DELETE if hasattr(client.api, "Method") else "DELETE",
                f"/_matrix/client/v3/devices/{client.device_id}")
            logger.info("Matrix: deleted stale device %s from server", client.device_id)
        return await _reupload(
            "Matrix: cannot upload device keys for %s: %s. Try generating a new access token to get a fresh device.",
            client.device_id)

    @staticmethod
    async def _abort_connect(api: Any, crypto_db: Any = None) -> bool:
        """Close what connect() opened so far; always False so callers can ``return await``."""
        if crypto_db is not None:
            await crypto_db.stop()
        await api.session.close()
        return False

    async def _connect_authenticate(self, client: Any, api: Any) -> bool:
        """Authenticate via access token (whoami) or password login; resolve user/device IDs."""
        if self._access_token:
            api.token = self._access_token
            try:
                resp = await client.whoami()
                resolved_user_id = getattr(resp, "user_id", "") or self._user_id
                resolved_device_id = str(getattr(resp, "device_id", "") or "")
                if resolved_user_id:
                    self._user_id = str(resolved_user_id)
                    client.mxid = UserID(self._user_id)
                # The configured device_id wins when whoami() reports none, but a token can
                # only upload keys for its own device — on conflict whoami() wins, loudly.
                if resolved_device_id and self._device_id and resolved_device_id != self._device_id:
                    logger.error(
                        "Matrix: MATRIX_DEVICE_ID=%s does not match the device this access token "
                        "belongs to (%s). A token can only upload keys for its own device, so the "
                        "configured value is being ignored. Unset MATRIX_DEVICE_ID, or use a token "
                        "issued for %s.", self._device_id, resolved_device_id, self._device_id)
                    effective_device_id = resolved_device_id
                else:
                    effective_device_id = self._device_id or resolved_device_id
                if effective_device_id:
                    client.device_id = effective_device_id
                if not client.device_id:
                    try:
                        dev_resp = await client.query_keys({client.mxid: []})
                        all_devices = (getattr(dev_resp, "device_keys", {}) or {}).get(str(client.mxid)) or {}
                        if len(all_devices) == 1:
                            client.device_id = next(iter(all_devices))
                        elif not all_devices:
                            logger.warning(
                                "Matrix: no devices found for %s — key verification will be skipped", client.mxid)
                    except Exception as exc:
                        logger.warning("Matrix: device list query failed: %s", exc)
                if not client.device_id:
                    logger.warning(
                        "Matrix: device_id could not be resolved for %s. Set MATRIX_DEVICE_ID for full "
                        "key verification. E2EE will proceed without server-side device key confirmation.",
                        client.mxid)
                    self._device_id_unverified = True
                logger.info(
                    "Matrix: using access token for %s%s", self._user_id or "(unknown user)",
                    f" (device {effective_device_id})" if effective_device_id else "")
            except Exception as exc:
                logger.error(
                    "Matrix: whoami failed — check MATRIX_ACCESS_TOKEN and MATRIX_HOMESERVER: %s", exc, exc_info=True)
                return await self._abort_connect(api)
        elif self._password and self._user_id:
            try:
                resp = await client.login(
                    identifier=self._user_id, password=self._password, device_name="Hermes Agent",
                    device_id=self._device_id or None)
                if resp and hasattr(resp, "device_id"):
                    client.device_id = resp.device_id
                logger.info("Matrix: logged in as %s", self._user_id)
            except Exception as exc:
                logger.error("Matrix: login failed — %s", exc)
                return await self._abort_connect(api)
        else:
            logger.error("Matrix: need MATRIX_ACCESS_TOKEN or MATRIX_USER_ID + MATRIX_PASSWORD")
            return await self._abort_connect(api)
        return True

    async def _connect_setup_e2ee(self, client: Any, api: Any, state_store: Any) -> bool:
        """Set up the Olm machine + crypto store. Returns False when connect must abort."""
        if not _check_e2ee_deps():
            if self._e2ee_mode == "optional":
                logger.warning(
                    "Matrix: E2EE optional but dependencies are missing. Continuing without "
                    "encrypted-room support. %s", _E2EE_INSTALL_HINT)
                self._encryption = False
            else:
                logger.error(
                    "Matrix: E2EE is required but dependencies are missing. %s. Refusing to connect — "
                    "encrypted rooms would silently fail.", _E2EE_INSTALL_HINT)
                return await self._abort_connect(api)
        if not self._encryption:
            return True
        phase = "import"
        try:
            from mautrix.crypto import OlmMachine
            from mautrix.crypto.store.asyncpg import PgCryptoStore
            from mautrix.util.async_db import Database
            self._store_dir.mkdir(parents=True, exist_ok=True)
            phase = "create"
            if (self._store_dir / "crypto_store.pickle").exists():  # pre-SQLite era
                logger.info("Matrix: removing legacy crypto_store.pickle (migrated to SQLite)")
                (self._store_dir / "crypto_store.pickle").unlink()
            crypto_db = Database.create(
                f"sqlite:///{self._crypto_db_path}", upgrade_table=PgCryptoStore.upgrade_table)
            await crypto_db.start()
            self._crypto_db = crypto_db
            _acct_id = self._user_id or "hermes"
            # Key on the RESOLVED client.device_id (token's real device), not the configured
            # one, or the Olm account is stored under a key that can never be looked up.
            _pickle_key = f"{_acct_id}:{client.device_id or self._device_id or 'default'}"
            crypto_store = PgCryptoStore(account_id=_acct_id, pickle_key=_pickle_key, db=crypto_db)
            await crypto_store.open()
            _store_was_reset = False
            if client.device_id:
                _store_was_reset = await self._reset_crypto_store_if_device_changed(crypto_store, client.device_id)
                await crypto_store.put_device_id(client.device_id)
            # A just-deleted store has no account to migrate.
            if not _store_was_reset and not await self._migrate_legacy_crypto_pickle(
                    crypto_store, crypto_db, _acct_id, _pickle_key):
                logger.warning("Matrix: crypto pickle migration failed — E2EE may not work correctly")
            crypto_state = _CryptoStateStore(state_store, self._joined_rooms, client)
            olm = OlmMachine(client, crypto_store, crypto_state)
            olm.share_keys_min_trust = TrustState.UNVERIFIED
            olm.send_keys_min_trust = TrustState.UNVERIFIED
            await olm.load()
            if not await self._verify_device_keys_on_server(client, olm):
                return await self._abort_connect(api, crypto_db)
            try:
                await olm.share_keys()
            except Exception as exc:
                if "already exists" in str(exc):
                    logger.error(
                        "Matrix: device %s has stale one-time keys on the server signed with a "
                        "previous identity key. Delete the device from the homeserver and restart, "
                        "or generate a new access token to get a fresh device ID.", client.device_id)
                    return await self._abort_connect(api, crypto_db)
                logger.warning("Matrix: share_keys() warning during startup: %s", exc)
            await self._verify_or_bootstrap_cross_signing(olm, client)
            client.crypto = olm
            logger.info(
                "Matrix: E2EE enabled (store: %s%s)", str(self._crypto_db_path),
                f", device_id={client.device_id}" if client.device_id else "")
        except Exception as exc:
            return await self._e2ee_setup_failed(phase, exc, api)
        return True

    async def _e2ee_setup_failed(self, what: str, exc: Exception, api: Any) -> bool:
        """Optional mode: log + disable E2EE and return True; required mode: close + return False."""
        if self._e2ee_mode == "optional":
            logger.warning(
                "Matrix: failed to %s optional E2EE client; continuing without encrypted-room "
                "support: %s. %s", what, exc, _E2EE_INSTALL_HINT)
            self._encryption = False
            return True
        logger.error("Matrix: failed to %s E2EE client: %s. %s", what, exc, _E2EE_INSTALL_HINT)
        return await self._abort_connect(api)

    async def _verify_or_bootstrap_cross_signing(self, olm: Any, client: Any) -> None:
        """Verify cross-signing via MATRIX_RECOVERY_KEY, or bootstrap a new key (non-fatal)."""
        # Honor the active profile's secret scope so a secondary profile under gateway.multiplex_profiles
        # resolves its own recovery key instead of the default profile's (which fails E2EE verification with
        # "Key MAC does not match", #69090).
        recovery_key = _scoped_recovery_key()
        if recovery_key:
            try:
                await olm.verify_with_recovery_key(recovery_key)
                logger.info("Matrix: cross-signing verified via recovery key")
            except Exception as exc:
                logger.warning("Matrix: recovery key verification failed: %s", exc)
        else:
            try:
                own_xsign = await olm.get_own_cross_signing_public_keys()
            except Exception as exc:
                own_xsign = None
                logger.warning("Matrix: cross-signing key lookup failed: %s", exc)
            if own_xsign is None:
                _, output_error = _get_matrix_recovery_key_output_target()
                if output_error:
                    reason = {
                        "not_configured": "is not configured. Configure MATRIX_RECOVERY_KEY from your Matrix client "
                                          "or set MATRIX_RECOVERY_KEY_OUTPUT_FILE to write a new recovery key once "
                                          "with mode 0600.",
                        "exists": "already exists and will not be overwritten.",
                    }.get(output_error, "is not usable: %s")
                    logger.warning(
                        "Matrix: cross-signing keys are missing, but automatic bootstrap is skipped because "
                        "MATRIX_RECOVERY_KEY_OUTPUT_FILE " + reason,
                        *([output_error] if output_error not in ("not_configured", "exists") else []))
                else:
                    try:
                        new_recovery_key = await olm.generate_recovery_key()
                        _handle_generated_matrix_recovery_key(str(client.mxid), new_recovery_key)
                    except Exception as exc:
                        logger.warning(
                            "Matrix: cross-signing bootstrap failed (non-fatal — Element will show "
                            "'not verified by its owner'): %s", exc)

    async def _connect_initial_sync(self, client: Any) -> None:
        """Full initial sync: seed joined rooms, DM cache, and dispatch queued to-device events."""
        try:
            sync_data = await client.sync(timeout=10000, full_state=True)
            if isinstance(sync_data, dict):
                self._joined_rooms.clear()
                await self._absorb_sync(client, sync_data, initial=True)
            else:
                logger.warning("Matrix: initial sync returned unexpected type %s", type(sync_data).__name__)
        except Exception as exc:
            logger.warning("Matrix: initial sync error: %s", exc)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._device_id_unverified = False
        if self._client is not None:
            try:
                await self.disconnect()
            except Exception as exc:
                logger.warning("Matrix: error disconnecting before reconnect: %s", exc)
        from mautrix.api import HTTPAPI
        from mautrix.client import Client
        from mautrix.client.state_store import MemoryStateStore, MemorySyncStore
        if not self._homeserver:
            logger.error("Matrix: homeserver URL not configured")
            return False
        # Resolved here, inside the profile scope, so multiplexed profiles never share it.
        self._resolve_store_dir().mkdir(parents=True, exist_ok=True)
        client_session = _create_matrix_session(self._proxy_url)
        api = HTTPAPI(base_url=self._homeserver, token=self._access_token or "", client_session=client_session)
        state_store = MemoryStateStore()
        sync_store = MemorySyncStore()
        client = Client(
            mxid=UserID(self._user_id) if self._user_id else UserID(""), device_id=self._device_id or None,
            api=api, state_store=state_store, sync_store=sync_store)
        self._client = client
        if not await self._connect_authenticate(client, api):
            return False
        if self._encryption and not await self._connect_setup_e2ee(client, api, state_store):
            return False
        from mautrix.client import InternalEventType as IntEvt
        from mautrix.client.dispatcher import MembershipEventDispatcher
        client.add_dispatcher(MembershipEventDispatcher)  # without this INVITE never fires
        client.add_event_handler(EventType.ROOM_MESSAGE, self._on_room_message, wait_sync=True)
        client.add_event_handler(EventType.REACTION, self._on_reaction, wait_sync=True)
        client.add_event_handler(IntEvt.INVITE, self._on_invite, wait_sync=True)
        self._startup_ts = time.time()
        self._reset_clock_skew_detector()  # a reconnect after an NTP fix starts clean
        self._closing = False
        await self._connect_initial_sync(client)
        if self._encryption and getattr(client, "crypto", None):
            try:
                await client.crypto.share_keys()
            except Exception as exc:
                logger.warning("Matrix: initial key share failed: %s", exc)
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._mark_connected()
        self._wire_plugin_handlers(self._client)  # plugin-registered native handlers
        return True

    async def disconnect(self) -> None:
        self._closing = True
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except (asyncio.CancelledError, Exception):
                pass
        for tasks in (self._invite_join_tasks.values(), self._reaction_redaction_tasks):
            pending = list(tasks)
            for task in pending:
                if not task.done():
                    task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        self._invite_join_tasks.clear()
        self._reaction_redaction_tasks.clear()
        if getattr(self, "_crypto_db", None):
            try:
                await self._crypto_db.stop()
            except Exception as exc:
                logger.debug("Matrix: could not close crypto DB on disconnect: %s", exc)
        if self._client:
            with suppress(Exception):
                await self._client.api.session.close()
            self._client = None
        logger.info("Matrix: disconnected")

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        if not content:
            return SendResult(success=True)
        last_event_id = None
        for chunk in self.truncate_message(self.format_message(content), self.max_message_length):
            msg_content = self._build_text_message_content(chunk)
            self._apply_relation_metadata(msg_content, reply_to=reply_to, metadata=metadata)
            try:
                last_event_id = await self._send_room_message(chat_id, msg_content)
                logger.info("Matrix: sent event %s to %s", last_event_id, chat_id)
            except Exception as exc:
                if not (self._encryption and getattr(self._client, "crypto", None)):
                    logger.error("Matrix: failed to send to %s: %s", chat_id, exc)
                    return SendResult(success=False, error=str(exc))
                try:  # E2EE error: retry once after sharing keys
                    await self._client.crypto.share_keys()
                    last_event_id = await self._send_room_message(chat_id, msg_content)
                    logger.info("Matrix: sent event %s to %s (after key share)", last_event_id, chat_id)
                except Exception as retry_exc:
                    logger.error("Matrix: failed to send to %s after retry: %s", chat_id, retry_exc)
                    return SendResult(success=False, error=str(retry_exc))
        return SendResult(success=True, message_id=last_event_id)

    async def _send_room_message(self, chat_id: str, msg_content: Dict[str, Any]) -> str:
        """Send one m.room.message event (45s cap) and return its event ID as str."""
        event_id = await asyncio.wait_for(
            self._client.send_message_event(RoomID(chat_id), EventType.ROOM_MESSAGE, msg_content), timeout=45)
        return str(event_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        identity = await self._resolve_room_identity(chat_id)
        return {"name": identity.display_name, "type": "dm" if identity.chat_type == "dm" else "group"}

    def get_diagnostics(self) -> Dict[str, Any]:
        now = time.time()
        token_present = bool(self._access_token)
        user_id = self._user_id or getattr(self._client, "mxid", "") or ""
        device_id = self._device_id or getattr(self._client, "device_id", "") or ""
        return {
            "platform": "matrix", "homeserver": self._homeserver,
            "auth": {
                "access_token_present": token_present, "password_present": bool(self._password),
                "token_preview": "***" if token_present else "", "user_id": user_id,
                "device_id_present": bool(device_id), "device_id_preview": "***" if str(device_id or "").strip() else ""},
            "sync": {
                "connected": self._client is not None, "joined_room_count": len(self._joined_rooms),
                "last_sync_age_seconds": max(0.0, now - self._last_sync_ts) if self._last_sync_ts else None},
            "e2ee": {
                "mode": self._e2ee_mode, "enabled": bool(self._encryption), "deps_available": _check_e2ee_deps(),
                "crypto_store_path": str(self._crypto_db_path),
                "recovery_key_configured": bool(_scoped_recovery_key().strip())},
            "policy": {
                "allowed_user_count": len(self._allowed_user_ids), "allowed_room_count": len(self._allowed_room_ids),
                "ignored_user_pattern_count": len(self._ignored_user_patterns),
                "require_mention": self._require_mention, "free_response_room_count": len(self._free_rooms),
                "allow_room_mentions": self._allow_room_mentions, "process_notices": self._process_notices,
                "allow_public_rooms": _env_truthy("MATRIX_ALLOW_PUBLIC_ROOMS")},
            "media": {"max_media_bytes": self._max_media_bytes}}

    async def _set_typing(self, chat_id: str, timeout: int) -> None:
        if self._client:
            with suppress(Exception):
                await self._client.set_typing(RoomID(chat_id), timeout=timeout)

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        await self._set_typing(chat_id, 30000)

    async def stop_typing(self, chat_id: str) -> None:
        await self._set_typing(chat_id, 0)

    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> SendResult:
        formatted = self.format_message(content)
        new_content = self._build_text_message_content(formatted)
        msg_content: Dict[str, Any] = {"msgtype": "m.text", "body": f"* {formatted}", "m.new_content": new_content}
        if "m.mentions" in new_content:
            msg_content["m.mentions"] = new_content["m.mentions"]
        if "formatted_body" in new_content:
            msg_content["format"] = "org.matrix.custom.html"
            msg_content["formatted_body"] = f'* {new_content["formatted_body"]}'
        msg_content["m.relates_to"] = {"rel_type": "m.replace", "event_id": message_id}
        return await self._send_content_event(chat_id, msg_content)

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        from tools.url_safety import is_safe_url
        if not is_safe_url(image_url):
            logger.warning("Matrix: blocked unsafe image URL (SSRF protection)")
            return await super().send_image(chat_id, image_url, caption, reply_to, metadata=metadata)
        try:
            data, ct, fname = await self._download_external_media_with_cap(image_url)
        except Exception as exc:
            logger.warning("Matrix: failed to download image %s: %s", _redact_url_for_log(image_url), exc)
            fallback = ("I couldn't download and upload the image to Matrix. "
                        "The source URL was not shown because it may contain private tokens.")
            return await self.send(chat_id, f"{caption}\n{fallback}" if caption else fallback, reply_to)
        return await self._upload_and_send(chat_id, data, fname, ct, "m.image", caption, reply_to, metadata)

    async def _download_external_media_with_cap(self, url: str) -> tuple[bytes, str, str]:
        """Download external media while enforcing redirect safety and size caps."""
        from tools.url_safety import is_safe_url
        if not is_safe_url(url):
            raise ValueError("blocked unsafe media URL")

        async def _read_capped(resp, chunks, content_type) -> tuple[bytes, str]:
            """Enforce Content-Length + streamed size caps, then require an image/* type."""
            try:
                size = int(resp.headers.get("Content-Length") or resp.headers.get("content-length"))
            except Exception:
                size = None
            if size is not None and size > self._max_media_bytes:
                raise ValueError(f"media exceeds Matrix limit ({size} > {self._max_media_bytes} bytes)")
            parts: list[bytes] = []
            total = 0
            async for chunk in chunks:
                total += len(chunk)
                if total > self._max_media_bytes:
                    raise ValueError(f"media exceeds Matrix limit (> {self._max_media_bytes} bytes)")
                parts.append(bytes(chunk))
            content_type = str(content_type or "").split(";", 1)[0].strip().lower()
            if not content_type.startswith("image/"):
                raise ValueError("external media is not an image")
            return b"".join(parts), content_type
        fname = url.rsplit("/", 1)[-1].split("?")[0] or "image.png"
        try:
            import aiohttp as _aiohttp
            _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(self._proxy_url)
            async with _aiohttp.ClientSession(**_sess_kw) as http:
                fetch_url = url
                for _ in range(20):
                    async with http.get(
                        fetch_url, timeout=_aiohttp.ClientTimeout(total=30), allow_redirects=False, **_req_kw) as resp:
                        if resp.status in {301, 302, 303, 307, 308}:
                            location = resp.headers.get("Location")
                            if not location:
                                raise ValueError("redirect missing Location")
                            # Re-validate EVERY hop: a public URL can 302 toward loopback/metadata endpoints,
                            # and checking only the final URL is too late (the hop already connected).
                            fetch_url = urljoin(fetch_url, location)
                            if not is_safe_url(fetch_url):
                                raise ValueError("blocked unsafe redirect URL")
                            continue
                        resp.raise_for_status()
                        data, ct = await _read_capped(
                            resp, resp.content.iter_chunked(65536),
                            getattr(resp, "content_type", None)
                            or resp.headers.get("content-type", "application/octet-stream"))
                        return data, ct, fname
                raise ValueError("too many redirects")
        except ImportError:
            from tools.url_safety import create_ssrf_safe_async_client
            _httpx_kw: dict = {"proxy": self._proxy_url} if self._proxy_url else {}
            _httpx_kw["event_hooks"] = {"response": [_ssrf_redirect_guard]}
            async with create_ssrf_safe_async_client(**_httpx_kw) as http:
                async with http.stream("GET", url, follow_redirects=True, timeout=30) as resp:
                    resp.raise_for_status()
                    data, ct = await _read_capped(
                        resp, resp.aiter_bytes(), resp.headers.get("content-type", "application/octet-stream"))
                    return data, ct, fname

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        return await self._send_local_file(chat_id, image_path, "m.image", caption, reply_to, metadata=metadata)

    async def send_multiple_images(
        self, chat_id: str, images: list[tuple[str, str]], metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0) -> None:
        if not images:
            return
        from urllib.parse import unquote as _unquote
        total = len(images)
        for idx, (image_url, alt_text) in enumerate(images, start=1):
            if human_delay > 0 and idx > 1:
                await asyncio.sleep(human_delay)
            caption = f"{alt_text} ({idx}/{total})" if alt_text and total > 1 else (alt_text or None)
            if image_url.startswith("file://"):
                result = await self.send_image_file(
                    chat_id=chat_id, image_path=_unquote(image_url[7:]), caption=caption, metadata=metadata)
            else:
                result = await self.send_image(chat_id=chat_id, image_url=image_url, caption=caption, metadata=metadata)
            if not result.success:
                logger.warning("Matrix: failed to send image %d/%d: %s", idx, total, result.error)

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        return await self._send_local_file(chat_id, file_path, "m.file", caption, reply_to, file_name, metadata)

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Upload audio as an MSC3245 voice message. Voice bubbles need Ogg/Opus but callers pass any
        format (e.g. TTS output), so transcode here — best-effort: without ffmpeg the original is sent."""
        converted_path: Optional[str] = None
        if not str(audio_path).lower().endswith((".ogg", ".oga", ".opus")):
            converted_path = await asyncio.to_thread(_matrix_transcode_voice_to_ogg, audio_path)
        try:
            return await self._send_local_file(
                chat_id, converted_path or audio_path, "m.audio", caption, reply_to,
                # keep the caller's basename (the temp transcode file has a generated name)
                file_name=(Path(audio_path).with_suffix(".ogg").name if converted_path else None),
                metadata=metadata, is_voice=True)
        finally:
            if converted_path:
                with suppress(OSError):
                    os.unlink(converted_path)

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        return await self._send_local_file(chat_id, video_path, "m.video", caption, reply_to, metadata=metadata)

    # Template attrs for the shared _format_exec_approval core (header + fence + reason only;
    # the smart-deny/scope wording lives in the reaction legend below).
    _EA_HEADER = "⚠️ **Dangerous command requires approval**\n"
    _EA_CMD_BUDGET = 2000

    async def _send_reaction_prompt(
        self, chat_id: str, text: str, metadata: Optional[dict], make_prompt, registry: dict, emojis,
        label: str) -> SendResult:
        """Send *text*, register ``make_prompt(message_id, requester, expires_at)`` under
        the resulting event, then seed the bot's reaction controls (recording their IDs)."""
        result = await self.send(chat_id, text, metadata=metadata)
        if not result.success or not result.message_id:
            return result
        prompt = make_prompt(
            result.message_id, str((metadata or {}).get("requester_user_id") or "") or None,
            time.monotonic() + max(self._approval_timeout_seconds, 0))
        registry[result.message_id] = prompt
        for emoji in emojis:
            try:
                reaction_event_id = await self._send_reaction(chat_id, result.message_id, emoji)
                if reaction_event_id:
                    prompt.bot_reaction_events[emoji] = str(reaction_event_id)
            except Exception as exc:
                logger.debug("Matrix: failed to add %s reaction %s: %s", label, emoji, exc)
        return result

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str, description: str = "dangerous command",
        metadata: Optional[dict] = None, allow_permanent: bool = True, allow_session: bool = True,
        smart_denied: bool = False, tool_use_id: str = "") -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Not connected")
        if smart_denied:
            scope_choices = "Smart DENY: owner override applies to this one operation only.\n"
        else:
            scope_choices = (
                ("Reply `!approve session` to approve this pattern for the session, " if allow_session else "")
                + ("`!approve always` to approve permanently, " if allow_permanent else ""))
        legend = ["✅ = approve once"]
        reactions = ["✅"]
        if allow_session:
            legend.append("🌀 = approve for this session")
            reactions.append("🌀")
            if allow_permanent:
                legend.append("♾️ = approve always")
                reactions.append("♾️")
        legend.append("❎ = deny")
        reactions.append("❌")
        text = (
            f"{self._format_exec_approval(command, description)}\n\n"
            f"{scope_choices}Reply `!approve` to execute once, or `!deny` to cancel.\n\n"
            "You can also click the reaction to approve:\n" + "\n".join(legend))

        def _make(message_id, requester, expires_at):
            old_event = self._approval_prompt_by_session.get(session_key)
            if old_event:
                self._approval_prompts_by_event.pop(old_event, None)
            self._approval_prompt_by_session[session_key] = message_id
            return _MatrixApprovalPrompt(
                session_key=session_key, chat_id=chat_id, message_id=message_id, requester_user_id=requester,
                expires_at=expires_at)
        return await self._send_reaction_prompt(
            chat_id, text, metadata, _make, self._approval_prompts_by_event, tuple(reactions), "approval")

    async def send_model_picker(
        self, chat_id: str, providers: list, current_model: str, current_provider: str, session_key: str,
        on_model_selected, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Not connected")
        flat_choices = [
            (str(model_id), str(p.get("slug") or ""), str(p.get("name") or p.get("slug") or ""))
            for p in providers or [] for model_id in (p.get("models") or [])][:len(_MATRIX_MODEL_PICKER_REACTIONS)]
        if not flat_choices:
            return await self.send(
                chat_id, "No authenticated models are available for this session.", metadata=metadata)
        try:
            from hermes_cli.providers import get_label
            provider_label = get_label(current_provider)
        except Exception:
            provider_label = current_provider
        lines = [
            "⚙ **Model Configuration**", f"Current model: `{current_model or 'unknown'}`",
            f"Provider: {provider_label or 'unknown'}", "", "React to choose a model:"]
        choices: dict[str, tuple[str, str]] = {}
        for emoji, (model_id, provider_slug, provider_name) in zip(_MATRIX_MODEL_PICKER_REACTIONS, flat_choices):
            choices[emoji] = (model_id, provider_slug)
            lines.append(f"{emoji} `{model_id}` — {provider_name}")
        return await self._send_picker(
            chat_id, lines, choices, session_key, on_model_selected, metadata, self._model_picker_prompts_by_event,
            "model picker")

    async def _send_picker(
        self, chat_id: str, lines: list, choices: dict, session_key: str, on_selected, metadata, registry: dict,
        label: str) -> SendResult:
        """Send picker *lines*, register a _MatrixPickerPrompt under the event, seed its reactions."""
        return await self._send_reaction_prompt(
            chat_id, "\n".join(lines), metadata,
            lambda message_id, requester, expires_at: _MatrixPickerPrompt(
                chat_id=chat_id, message_id=message_id, session_key=session_key, choices=choices,
                on_selected=on_selected, requester_user_id=requester, expires_at=expires_at),
            registry, choices, label)

    async def send_choice_picker(
        self, chat_id: str, title: str, choices: list, session_key: str, on_choice_selected,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Reaction-based choice picker (/reasoning, /fast); choice = {value, label, is_current}."""
        if not self._client:
            return SendResult(success=False, error="Not connected")
        emoji_choices: dict[str, str] = {}
        lines = [title, ""]
        for emoji, choice in zip(_MATRIX_CHOICE_PICKER_REACTIONS, choices):
            value = str(choice.get("value") or "")
            label = str(choice.get("label") or value)
            if choice.get("is_current"):
                label = f"{label} ← current"
            emoji_choices[emoji] = value
            lines.append(f"{emoji} {label}")
        if not emoji_choices:
            return SendResult(success=False, error="No choices")
        lines += ["", "React to choose."]
        return await self._send_picker(
            chat_id, lines, emoji_choices, session_key, on_choice_selected, metadata,
            self._choice_picker_prompts_by_event, "choice picker")

    def format_message(self, content: str) -> str:
        """Markdown passes through; strip image markdown (media is uploaded separately)."""
        return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\2", content)

    async def _upload_and_send(
        self, room_id: str, data: bytes, filename: str, content_type: str, msgtype: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
        is_voice: bool = False, voice_metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        if len(data) > self._max_media_bytes:
            return self._media_too_large(len(data))
        upload_data = data
        encrypted_file = None
        if await self._room_needs_encrypted_upload(room_id):
            try:
                from mautrix.crypto.attachments import encrypt_attachment
                upload_data, encrypted_file = encrypt_attachment(data)
            except Exception as exc:
                logger.error("Matrix: attachment encryption failed: %s", exc)
                return SendResult(success=False, error=str(exc))
        try:
            mxc_url = await self._client.upload_media(
                upload_data, mime_type=content_type, filename=filename, size=len(upload_data))
        except Exception as exc:
            logger.error("Matrix: upload failed: %s", exc)
            return SendResult(success=False, error=str(exc))
        msg_content: Dict[str, Any] = {
            "msgtype": msgtype, "body": caption or filename, "info": {"mimetype": content_type, "size": len(data)}}
        if encrypted_file is not None:
            msg_content["file"] = {**encrypted_file.serialize(), "url": str(mxc_url)}
        else:
            msg_content["url"] = str(mxc_url)
        if is_voice:  # MSC3245 native voice flag + MSC1767 audio metadata
            msg_content["org.matrix.msc3245.voice"] = {}
            audio_metadata = {
                k: v for k in ("duration", "waveform") if (v := (voice_metadata or {}).get(k)) is not None}
            if "duration" in audio_metadata:
                msg_content["info"]["duration"] = audio_metadata["duration"]
            if audio_metadata:
                msg_content["org.matrix.msc1767.audio"] = audio_metadata
        self._apply_relation_metadata(msg_content, reply_to=reply_to, metadata=metadata)
        return await self._send_content_event(room_id, msg_content)

    async def _room_needs_encrypted_upload(self, room_id: str) -> bool:
        """E2EE on, Olm machine loaded, and the state store says the room is encrypted."""
        if not (self._encryption and getattr(self._client, "crypto", None)):
            return False
        state_store = getattr(self._client, "state_store", None)
        if not state_store:
            return False
        try:
            return bool(await state_store.is_encrypted(RoomID(room_id)))
        except Exception:
            return False

    def _media_too_large(self, size: int) -> SendResult:
        return SendResult(
            success=False, error=f"Media file exceeds Matrix limit ({size} > {self._max_media_bytes} bytes)")

    async def _send_content_event(self, room_id: str, msg_content: Dict[str, Any]) -> SendResult:
        """Send a prebuilt m.room.message payload, mapping exceptions to SendResult."""
        try:
            event_id = await self._client.send_message_event(RoomID(room_id), EventType.ROOM_MESSAGE, msg_content)
            return SendResult(success=True, message_id=str(event_id))
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def _send_local_file(
        self, room_id: str, file_path: str, msgtype: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, file_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
        is_voice: bool = False) -> SendResult:
        p = Path(file_path).expanduser()
        if not p.exists():
            # file_path is host-local; never echo it into chat.
            logger.warning("[%s] upload fallback: media file not found for %s", self.name, file_path)
            text = "⚠️ Couldn't deliver the attachment."
            return await self.send(room_id, f"{caption}\n{text}" if caption else text, reply_to)
        try:
            file_size = p.stat().st_size
        except OSError:
            file_size = 0
        if file_size > self._max_media_bytes:
            return self._media_too_large(file_size)
        fname = file_name or p.name
        # ffprobe/ffmpeg probing is blocking (subprocess timeouts up to 15s) —
        # run it off the event loop so voice uploads never stall the adapter.
        voice_metadata = await asyncio.to_thread(_matrix_voice_metadata_for_file, p) if is_voice else None
        return await self._upload_and_send(
            room_id, p.read_bytes(), fname, mimetypes.guess_type(fname)[0] or "application/octet-stream", msgtype,
            caption, reply_to, metadata, is_voice, voice_metadata)

    async def _sync_loop(self) -> None:
        client = self._client
        next_batch = await client.sync_store.get_next_batch()  # resume from the initial sync
        while not self._closing:
            try:
                # 45s outer cap guards TCP-level hangs the 30s long-poll timeout can't catch.
                sync_data = await asyncio.wait_for(client.sync(since=next_batch, timeout=30000), timeout=45.0)
                # Auth failures (M_UNKNOWN_TOKEN) arrive as SyncError objects, not exceptions.
                _sync_msg = getattr(sync_data, "message", None)
                if isinstance(_sync_msg, str) and "unknown_token" in _sync_msg.lower():
                    logger.error("Matrix: permanent auth error from sync: %s — stopping", _sync_msg)
                    return
                if isinstance(sync_data, dict):
                    next_batch = await self._absorb_sync(client, sync_data) or next_batch
                    await asyncio.sleep(0)  # let fresh invite joins start before the next sync
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                if any(k in str(exc).lower() for k in ("401", "403", "unauthorized", "forbidden")):
                    logger.error("Matrix: permanent auth error: %s — stopping sync", exc)
                    return
                logger.warning("Matrix: sync error: %s — retrying in 5s", exc)
                await asyncio.sleep(5)

    async def _absorb_sync(self, client: Any, sync_data: Dict[str, Any], *, initial: bool = False) -> Optional[str]:
        """Apply one sync response: joined rooms, next_batch, event dispatch, pending invites. Returns next_batch.
        The initial (full-state) sync also seeds the DM cache and dispatches so the OlmMachine sees
        to-device key shares queued while offline."""
        self._last_sync_ts = time.time()
        rooms_join = sync_data.get("rooms", {}).get("join", {})
        if rooms_join or initial:
            self._joined_rooms.update(rooms_join.keys())
            self._invalidate_room_identities()
        nb = sync_data.get("next_batch")  # incremental syncs resume from here
        if nb:
            await client.sync_store.put_next_batch(nb)
        if initial:
            logger.info("Matrix: initial sync complete, joined %d rooms", len(self._joined_rooms))
            await self._refresh_dm_cache()
        try:
            await self._dispatch_sync(sync_data)
        except Exception as exc:
            logger.warning("Matrix: %s: %s", "initial sync event dispatch error" if initial else "sync event dispatch error", exc)
        self._schedule_pending_invite_joins(sync_data)
        return nb

    async def _dispatch_sync(self, sync_data: Dict[str, Any]) -> None:
        """Dispatch a sync response through the mautrix event machinery."""
        client = self._client
        if not client or not hasattr(client, "handle_sync"):
            return
        tasks = client.handle_sync(sync_data)
        if inspect.isawaitable(tasks):
            tasks = await tasks
        if tasks:
            # return_exceptions=True: one failing handler must not drop its SIBLING events.
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Matrix: event handler failed during sync dispatch: %s", result)

    def _is_self_sender(self, sender: str) -> bool:
        """True if *sender* is the bot itself (case-insensitive: homeservers vary localpart case). With
        no resolved user_id we can't prove a sender is NOT us, so return True — dropping our own
        events beats an echo loop ("hall of mirrors").

        Matrix user IDs are byte-compared after trimming whitespace and lowercasing — some homeservers
        normalize the localpart case differently at different API surfaces, and the reply-loop tail of the
        "hall of mirrors" bug (#15763) has been observed with the bot's own account bypassing a
        case-sensitive equality check.
        """
        own = (self._user_id or "").strip().lower()
        return not own or sender.strip().lower() == own

    @staticmethod
    def _is_system_or_bridge_sender(sender: str) -> bool:
        """True for appservice/bridge/system identities (``@_telegram_123:server``) or malformed IDs.
        Never offer these a pairing code: an approved bridge would relay every outbound message
        back as an "authorized user message" (echo loop).

        We treat these as system identities for pairing purposes: they should never be offered a pairing
        code, because an operator approving the code would hand the bridge itself permanent authorization —
        and every outbound message relayed by the bridge would then loop back into the agent as an
        "authorized user message", which is the root of issue #15763.
        """
        localpart = (sender or "").strip().lstrip("@").partition(":")[0]
        return not localpart or localpart.startswith("_")

    async def _is_allowed_matrix_room_event(self, room_id: str) -> bool:
        """MATRIX_ALLOWED_ROOMS gate; DMs are exempt so personal chats survive a project allowlist."""
        if not self._allowed_room_ids or room_id in self._allowed_room_ids:
            return True
        try:
            return await self._is_dm_room(room_id)
        except Exception as exc:
            logger.debug("Matrix: could not resolve room identity for allowlist check in %s: %s", room_id, exc)
            return False

    def _reset_clock_skew_detector(self) -> None:
        """State for _note_late_grace_drop: consecutive-drop count, their skew, and the once-only warning."""
        # Clock-skew detection: count grace-check drops that happen well after startup (i.e. not
        # initial-sync backfill). If the host's system clock is set ahead of real time, the startup grace
        # check `event_ts < startup_ts - 5` silently drops every live message. See #12614 — the symptom is
        # "bot joins rooms but never replies". Drops only count when their skew matches the first sampled
        # drop (within 60s), so varied-age backfill from freshly-invited rooms doesn't trip the heuristic.
        self._late_grace_drops: int = 0
        self._late_grace_skew: float = 0.0
        self._clock_skew_warned: bool = False

    def _note_late_grace_drop(self, event_ts: float) -> None:
        """Clock-skew heuristic for grace-check drops well after startup. A host clock set ahead of
        real time makes every live event look "older than startup" and the bot silently never
        replies. Warn once when drops keep happening >30s after startup with a *consistent* skew —
        unlike backfill from a freshly invited room, whose event ages vary widely and reset the counter."""
        if self._clock_skew_warned or time.time() - self._startup_ts <= 30:
            return
        skew = self._startup_ts - event_ts
        if not (5 < skew < 86400):  # ignore malformed/absurd timestamps
            return
        if self._late_grace_drops and abs(skew - self._late_grace_skew) < 60:
            self._late_grace_drops += 1
        else:
            self._late_grace_skew = skew
            self._late_grace_drops = 1
        if self._late_grace_drops >= 3:
            logger.warning(
                "Matrix: dropped %d consecutive live events as 'too old' more than 30s after startup "
                "(skew ≈ %.0fs). The host system clock is likely set ahead of real time, which causes "
                "the startup grace filter to silently discard every incoming message. Run "
                "`timedatectl set-ntp true` (or sync NTP) and restart the bot.", self._late_grace_drops, skew)
            self._clock_skew_warned = True

    async def _on_room_message(self, event: Any) -> None:
        room_id = str(getattr(event, "room_id", ""))
        sender = str(getattr(event, "sender", ""))
        # DEBUG-level proof the callback fires at all (silent-inbound troubleshooting).
        logger.debug(
            "Matrix: callback fired — event %s from %s in %s", getattr(event, "event_id", "?"), sender, room_id)
        if self._is_self_sender(sender):
            return
        # Bridge/system identities must never reach the pairing flow (echo loop once paired).
        # Ignore own messages (case-insensitive; also drops when our own user_id hasn't been resolved yet —
        # see _is_self_sender docstring and issue #15763).
        # Once a bridge user is paired, every outbound message it relays would loop back as an authorized
        # user message (the "hall of mirrors" in #15763).
        if self._is_system_or_bridge_sender(sender):
            logger.debug("Matrix: ignoring system/bridge sender %s in %s", sender, room_id)
            return
        if any(pattern.search(sender or "") for pattern in self._ignored_user_patterns):
            logger.debug("Matrix: ignoring sender %s in %s due to configured ignore pattern", sender, room_id)
            return
        if not await self._is_allowed_matrix_room_event(room_id):
            logger.info("Matrix: ignoring message from unauthorized room %s", room_id)
            return
        event_id = str(getattr(event, "event_id", ""))
        if self._is_duplicate_event(event_id):
            return
        # Startup grace: ignore old messages replayed by the initial sync.
        event_ts = _matrix_event_timestamp_seconds(event)
        if event_ts and event_ts < self._startup_ts - _STARTUP_GRACE_SECONDS:
            self._note_late_grace_drop(event_ts)
            return
        content = getattr(event, "content", None)
        if content is None:
            return
        if isinstance(content, dict):
            source_content, msgtype = content, content.get("msgtype", "")
        else:
            source_content = content.serialize() if hasattr(content, "serialize") else {}
            msgtype = str(content.msgtype) if hasattr(content, "msgtype") else ""
        relates_to = source_content.get("m.relates_to", {})
        if relates_to.get("rel_type") == "m.replace":  # skip edits
            return
        # m.notice is the conventional bot-response msgtype; ignoring it prevents bot-to-bot loops.
        if msgtype == "m.notice" and not self._process_notices:
            return
        if msgtype in ("m.image", "m.audio", "m.video", "m.file"):
            await self._handle_media_message(room_id, sender, event_id, event_ts, source_content, relates_to, msgtype)
        elif msgtype in ("m.text", "m.notice"):
            await self._handle_text_message(room_id, sender, event_id, event_ts, source_content, relates_to)

    async def _resolve_message_context(
        self, room_id: str, sender: str, event_id: str, body: str, source_content: dict,
        relates_to: dict) -> Optional[tuple]:
        """Shared mention/thread/DM gating. Returns (body, is_dm, chat_type, thread_id,
        display_name, source) or None when the message should be dropped."""
        identity = await self._resolve_room_identity(room_id)
        is_dm = await self._is_dm_room(room_id)
        chat_type = "dm" if is_dm else "group"
        thread_id = relates_to.get("event_id") if relates_to.get("rel_type") == "m.thread" else None
        formatted_body = source_content.get("formatted_body")
        mentions_block = source_content.get("m.mentions") or {}  # MSC3952: authoritative signal
        mention_user_ids = mentions_block.get("user_ids") if isinstance(mentions_block, dict) else None
        is_mentioned = self._is_bot_mentioned(body, formatted_body, mention_user_ids)
        if not is_dm:
            # Whitelist first: non-listed rooms are dropped even when @mentioned (DMs exempt).
            if self._allowed_rooms and room_id not in self._allowed_rooms:
                logger.debug(
                    "Matrix: ignoring message %s in %s — room not in MATRIX_ALLOWED_ROOMS whitelist", event_id, room_id)
                return None
            is_free_room = room_id in self._free_rooms
            in_bot_thread = bool(thread_id and thread_id in self._threads)
            if self._require_mention and not is_free_room and not in_bot_thread:
                if not is_mentioned and not body.startswith("/"):
                    logger.debug(
                        "Matrix: ignoring message %s in %s — no @mention "
                        "(set MATRIX_REQUIRE_MENTION=false to disable)", event_id, room_id)
                    return None
            # thread_require_mention: even inside a bot thread require @mention — prevents
            # infinite reply loops when several bots share one thread.
            elif self._thread_require_mention and in_bot_thread and not is_free_room and not is_mentioned:
                logger.debug(
                    "Matrix: ignoring message %s in thread %s — no @mention (thread_require_mention=true)",
                    event_id, thread_id)
                return None
        if is_mentioned and self._require_mention:
            body = self._strip_mention(body)
        # Real thread roots are preserved above; synthetic roots (this event) follow policy: DM
        # @mention threads / DM auto-thread, or room auto-thread unless session_scope pins the room.
        if not thread_id:
            if is_dm:
                synthetic = (self._dm_mention_threads and is_mentioned) or self._dm_auto_thread
            else:
                synthetic = self._matrix_session_scope == "thread" or (
                    self._matrix_session_scope != "room" and self._auto_thread)
            if synthetic:
                thread_id = event_id
        display_name = await self._get_display_name(room_id, sender)
        source = self.build_source(
            chat_id=room_id, chat_name=identity.display_name, chat_type=chat_type, user_id=sender,
            user_name=display_name, thread_id=thread_id, chat_topic=identity.room_topic,
            guild_id=identity.server_name, parent_chat_id=room_id if thread_id else None, message_id=event_id)
        if thread_id:
            self._threads.mark(thread_id)  # covers real roots and synthetic ones alike
        self._background_read_receipt(room_id, event_id)
        return body, is_dm, chat_type, thread_id, display_name, source

    async def _extract_reply_context(
        self, room_id: str, body: str, relates_to: dict
    ) -> tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Return (body, reply_to, reply_to_text, reply_to_author_id, reply_to_author_name). Captures
        the inline reply fallback (``> <@user:srv> text\\n\\nreply``) BEFORE stripping it, so the
        prompt layer can render "[Replying to: ...]" like Signal/Slack/Telegram."""
        reply_to = (relates_to.get("m.in_reply_to") or {}).get("event_id")
        reply_to_text = reply_to_author_id = reply_to_author_name = None
        if reply_to and body.startswith("> "):
            reply_to_text, reply_to_author_id = _extract_reply_fallback(body)
            body = _strip_reply_fallback(body)
            # Resolve the replied-to author's display name (falls back to localpart).
            if reply_to_author_id:
                reply_to_author_name = await self._get_display_name(room_id, reply_to_author_id)
        return body, reply_to, reply_to_text, reply_to_author_id, reply_to_author_name

    async def _build_inbound_event(
        self, room_id: str, sender: str, event_id: str, body: str, source_content: dict, relates_to: dict,
        **extra) -> Optional[MessageEvent]:
        """Gate + normalise an inbound event into a MessageEvent (None => drop). Text body may
        still change (reply-fallback strip); ``extra`` carries media fields / message_type."""
        ctx = await self._resolve_message_context(room_id, sender, event_id, body, source_content, relates_to)
        if ctx is None:
            return None
        body, _is_dm, _chat_type, _thread_id, display_name, source = ctx
        body, reply_to, reply_to_text, reply_to_author_id, reply_to_author_name = (
            await self._extract_reply_context(room_id, body, relates_to))
        media_msgtype = extra.pop("media_msgtype", None)
        if media_msgtype is None:
            # Re-normalize after reply stripping so ``> quoted\n\n!model`` is still a command.
            body = _normalize_matrix_bang_command(body)
            extra["message_type"] = MessageType.COMMAND if body.startswith("/") else MessageType.TEXT
        elif _is_bare_media_filename(media_msgtype, body):
            body = ""  # transport filename, not user text
        return MessageEvent(
            text=body, source=source, raw_message=source_content, message_id=event_id,
            reply_to_message_id=reply_to, reply_to_text=reply_to_text, reply_to_author_id=reply_to_author_id,
            reply_to_author_name=reply_to_author_name,
            # Top-level sender fields mirror source.* — downstream prompt code reads them.
            user_id=sender, user_name=display_name, **extra)

    async def _handle_text_message(
        self, room_id: str, sender: str, event_id: str, event_ts: float, source_content: dict,
        relates_to: dict) -> None:
        body = source_content.get("body", "") or ""
        if not body:
            return
        msg_event = await self._build_inbound_event(
            room_id, sender, event_id, _normalize_matrix_bang_command(body), source_content, relates_to)
        if msg_event is None:
            return
        if msg_event.message_type == MessageType.TEXT and self._text_batch_delay_seconds > 0:
            self._enqueue_text_event(msg_event)
        else:
            await self.handle_message(msg_event)

    async def _handle_media_message(
        self, room_id: str, sender: str, event_id: str, event_ts: float, source_content: dict,
        relates_to: dict, msgtype: str) -> None:
        body = source_content.get("body", "") or ""
        url = source_content.get("url", "")
        if url and not str(url).startswith("mxc://"):
            logger.warning("[Matrix] Rejecting inbound media %s with non-MXC URL", event_id)
            return
        content_info = source_content.get("info", {})
        if not isinstance(content_info, dict):
            content_info = {}
        event_mimetype = content_info.get("mimetype", "")
        try:
            event_size_int = int(content_info.get("size") or 0)
        except (TypeError, ValueError):
            event_size_int = 0
        if event_size_int and event_size_int > self._max_media_bytes:
            logger.warning(
                "[Matrix] Rejecting oversized inbound media %s (%d > %d bytes)", event_id, event_size_int,
                self._max_media_bytes)
            return
        file_content = source_content.get("file", {})  # encrypted media carries file.url
        if not url and isinstance(file_content, dict):
            url = file_content.get("url", "") or ""
            if url and not str(url).startswith("mxc://"):
                logger.warning("[Matrix] Rejecting inbound encrypted media %s with non-MXC URL", event_id)
                return
        is_encrypted_media = bool(file_content and isinstance(file_content, dict) and file_content.get("url"))
        msg_type, media_type, is_voice_message = self._classify_inbound_media(msgtype, event_mimetype, source_content)
        # Cache locally so downstream tools get a real file path.
        cached_path = None
        if url:
            try:
                cached_path = await self._download_and_cache_media(
                    url, event_id, file_content if is_encrypted_media else None, msg_type, media_type,
                    is_voice_message, body)
            except Exception as e:
                logger.warning("[Matrix] Failed to cache media: %s", e)
        # Unencrypted media may fall back to the HTTP download URL when caching failed.
        http_url = self._mxc_to_http(url) if url and not is_encrypted_media else ""
        media_urls = [cached_path] if cached_path else ([http_url] if http_url else None)
        msg_event = await self._build_inbound_event(
            room_id, sender, event_id, body, source_content, relates_to, message_type=msg_type,
            media_urls=media_urls, media_types=[media_type] if media_urls else None, media_msgtype=msgtype)
        if msg_event is not None:
            await self.handle_message(msg_event)

    @staticmethod
    def _classify_inbound_media(
            msgtype: str, event_mimetype: str, source_content: dict) -> tuple[MessageType, str, bool]:
        """Map a Matrix media msgtype to (MessageType, mime type, is_voice_message)."""
        if msgtype == "m.image":
            return MessageType.PHOTO, event_mimetype or "image/png", False
        if msgtype == "m.audio":
            is_voice = source_content.get("org.matrix.msc3245.voice") is not None
            return (MessageType.VOICE if is_voice else MessageType.AUDIO), event_mimetype or "audio/ogg", is_voice
        if msgtype == "m.video":
            return MessageType.VIDEO, event_mimetype or "video/mp4", False
        return MessageType.DOCUMENT, event_mimetype or "application/octet-stream", False

    async def _download_and_cache_media(
        self, url: str, event_id: str, encrypted_file: Optional[dict], msg_type: MessageType, media_type: str,
        is_voice_message: bool, body: str) -> Optional[str]:
        """Download (and decrypt, when *encrypted_file* is given) media into the local cache."""
        file_bytes = await self._client.download_media(ContentURI(url))
        if file_bytes is None:
            return None
        if encrypted_file is not None:
            from mautrix.crypto.attachments import decrypt_attachment
            hashes_value, key_value = encrypted_file.get("hashes"), encrypted_file.get("key")
            hash_value = hashes_value.get("sha256") if isinstance(hashes_value, dict) else None
            key_value = key_value.get("k") if isinstance(key_value, dict) else key_value
            iv_value = encrypted_file.get("iv")
            if not (key_value and hash_value and iv_value):
                logger.warning("[Matrix] Encrypted media event missing decryption metadata for %s", event_id)
                return None
            file_bytes = decrypt_attachment(file_bytes, key_value, hash_value, iv_value)
        from gateway.platforms.base import (
            cache_audio_from_bytes_async,
            cache_document_from_bytes_async,
            cache_image_from_bytes_async,
        )
        if msg_type == MessageType.PHOTO:
            ext_map = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}
            cached_path = await cache_image_from_bytes_async(file_bytes, ext=ext_map.get(media_type, ".jpg"))
            logger.info("[Matrix] Cached user image at %s", cached_path)
            return cached_path
        if msg_type in {MessageType.AUDIO, MessageType.VOICE}:
            ext = Path(body or ("voice.ogg" if is_voice_message else "audio.ogg")).suffix or ".ogg"
            return await cache_audio_from_bytes_async(file_bytes, ext=ext)
        filename = body or ("video.mp4" if msg_type == MessageType.VIDEO else "document")
        return await cache_document_from_bytes_async(file_bytes, filename)

    async def _on_invite(self, event: Any) -> None:
        """Auto-join rooms when invited, recording DM rooms in m.direct."""
        room_id = str(getattr(event, "room_id", ""))
        is_direct = bool(getattr(getattr(event, "content", None), "is_direct", False))
        inviter = str(getattr(event, "sender", ""))
        # Only authorized inviters — otherwise any federated user could pull the bot into rooms.
        if not self._is_authorized_user(inviter):
            logger.warning("Matrix: rejecting invite to %s from unauthorized user %s", room_id, inviter)
            return
        logger.info("Matrix: invited to %s — joining (is_direct=%s)", room_id, is_direct)
        # Join off the sync path; a declared DM is recorded in m.direct once the join lands.
        self._schedule_invite_join(room_id, is_direct=is_direct and bool(inviter), inviter=inviter)

    async def _join_room_by_id(self, room_id: str) -> bool:
        if not room_id or room_id in self._joined_rooms:
            return bool(room_id)
        try:
            await self._client.join_room(RoomID(room_id))
            self._joined_rooms.add(room_id)
            self._invalidate_room_identities(room_id)
            logger.info("Matrix: joined %s", room_id)
            await self._refresh_dm_cache()
            return True
        except Exception as exc:
            logger.warning("Matrix: error joining %s: %s", room_id, exc)
            # Abandoned rooms ("no servers ..." / "room not found") would retry every startup
            # unless we leave the invite; the match is narrow so transient errors keep retrying.
            msg = str(exc).lower()
            if ("no servers" in msg) or ("room not found" in msg):
                with suppress(Exception):
                    await self._client.leave_room(RoomID(room_id))
                    logger.info("Matrix: declined dead invite to %s", room_id)
            return False

    def _schedule_invite_join(self, room_id: str, *, is_direct: bool = False, inviter: str = "") -> None:
        """Schedule an invite join without blocking sync or gateway readiness."""
        existing = self._invite_join_tasks.get(room_id)
        if not room_id or room_id in self._joined_rooms or (existing and not existing.done()):
            return

        async def _join_invite() -> None:
            try:
                joined = await asyncio.wait_for(self._join_room_by_id(room_id), timeout=45.0)
                if joined and is_direct and inviter:
                    await self._record_dm_room(room_id, inviter)
            except asyncio.TimeoutError:
                logger.warning("Matrix: timed out joining invite %s", room_id)
            finally:
                self._invite_join_tasks.pop(room_id, None)
        self._invite_join_tasks[room_id] = asyncio.create_task(_join_invite())

    def _schedule_pending_invite_joins(self, sync_data: Dict[str, Any]) -> None:
        """Join rooms still present in rooms.invite after sync processing."""
        invites = (sync_data.get("rooms", {}) if isinstance(sync_data, dict) else {}).get("invite", {})
        if not isinstance(invites, dict):
            return
        for room_id in invites:
            if room_id in self._joined_rooms:
                continue
            logger.info("Matrix: reconciling pending invite for %s", room_id)
            self._schedule_invite_join(str(room_id))

    async def _send_reaction(self, room_id: str, event_id: str, emoji: str) -> Optional[str]:
        """Send an emoji reaction; returns the reaction event_id, or None on failure."""
        if not self._client:
            return None
        content = {"m.relates_to": {"rel_type": "m.annotation", "event_id": event_id, "key": emoji}}
        try:
            resp_event_id = await self._client.send_message_event(RoomID(room_id), EventType.REACTION, content)
            logger.debug("Matrix: sent reaction %s to %s", emoji, event_id)
            return str(resp_event_id)
        except Exception as exc:
            logger.debug("Matrix: reaction send error: %s", exc)
            return None

    async def _redact_reaction(self, room_id: str, reaction_event_id: str, reason: str = "") -> bool:
        return await self.redact_message(room_id, reaction_event_id, reason)

    def _schedule_reaction_redaction(self, room_id: str, reaction_event_id: str, reason: str = "") -> None:
        """Redact a reaction after a short delay so message delivery settles."""

        async def _redact_later() -> None:
            try:
                if self._reaction_redaction_delay_seconds:
                    await asyncio.sleep(self._reaction_redaction_delay_seconds)
                if not await self._redact_reaction(room_id, reaction_event_id, reason):
                    logger.debug("Matrix: failed to redact reaction %s", reaction_event_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Matrix: delayed reaction redaction failed for %s: %s", reaction_event_id, exc)
        task = asyncio.create_task(_redact_later())
        self._reaction_redaction_tasks.add(task)
        task.add_done_callback(self._reaction_redaction_tasks.discard)

    async def on_processing_start(self, event: MessageEvent) -> None:
        msg_id, room_id = event.message_id, event.source.chat_id
        if self._reactions_enabled and msg_id and room_id:
            reaction_event_id = await self._send_reaction(room_id, msg_id, "\U0001f440")
            if reaction_event_id:
                self._pending_reactions[(room_id, msg_id)] = reaction_event_id

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        msg_id, room_id = event.message_id, event.source.chat_id
        if not self._reactions_enabled or not msg_id or not room_id or outcome == ProcessingOutcome.CANCELLED:
            return
        eyes_event_id = self._pending_reactions.pop((room_id, msg_id), None)
        if eyes_event_id:
            self._schedule_reaction_redaction(room_id, eyes_event_id, "processing complete")
        await self._send_reaction(room_id, msg_id, "\u2705" if outcome == ProcessingOutcome.SUCCESS else "\u274c")

    async def _on_reaction(self, event: Any) -> None:
        sender = str(getattr(event, "sender", ""))
        if self._is_self_sender(sender):
            return
        event_id = str(getattr(event, "event_id", ""))
        if self._is_duplicate_event(event_id):
            return
        room_id = str(getattr(event, "room_id", ""))
        content = getattr(event, "content", None)
        if not content:
            return
        relates_to = (content.get("m.relates_to", {}) if isinstance(content, dict)
                      else getattr(content, "relates_to", {}))
        reacts_to = key = ""
        if isinstance(relates_to, dict):
            reacts_to = relates_to.get("event_id", "")
            key = relates_to.get("key", "")
        elif hasattr(relates_to, "event_id"):
            reacts_to = str(getattr(relates_to, "event_id", ""))
            key = str(getattr(relates_to, "key", ""))
        logger.info("Matrix: reaction %s from %s on %s in %s", key, sender, reacts_to, room_id)
        for handler in (self._handle_approval_reaction, self._handle_model_picker_reaction,
                        self._handle_choice_picker_reaction):
            if await handler(room_id, reacts_to, key, sender):
                return

    async def _claim_reaction_prompt(
        self, registry: dict, room_id: str, reacts_to: str, key: str, sender: str, label: str, invalid_text: str,
        on_expired, choices: Optional[dict] = None) -> tuple[bool, Any, Any]:
        """Shared gate for reaction prompts: (handled, prompt, selection). handled=False => not our
        prompt; selection=None with handled=True => consumed without action (wrong room, expired,
        unauthorized reactor, or a key that is not a choice). ``choices`` defaults to ``prompt.choices``."""
        prompt = registry.get(reacts_to)
        if not prompt or prompt.resolved:
            return False, None, None
        if room_id != prompt.chat_id:
            return True, prompt, None
        if self._matrix_prompt_expired(prompt):
            await on_expired(room_id, reacts_to, prompt)
            return True, prompt, None
        if not await self._validate_matrix_prompt_reactor(room_id, reacts_to, sender, prompt, label):
            return True, prompt, None
        selection = (prompt.choices if choices is None else choices).get(key)
        if selection is None:
            await self._send_invalid_reaction_feedback(room_id, reacts_to, invalid_text)
        return True, prompt, selection

    async def _handle_approval_reaction(self, room_id: str, reacts_to: str, key: str, sender: str) -> bool:
        """Resolve a pending exec-approval prompt from a reaction. True if it was the target."""
        handled, prompt, choice = await self._claim_reaction_prompt(
            self._approval_prompts_by_event, room_id, reacts_to, key, sender, "approval",
            "That reaction is not valid for this approval prompt.", self._expire_matrix_approval_prompt,
            choices=self._approval_reaction_map)
        if choice is None:
            return handled
        try:
            from tools.approval import resolve_gateway_approval
            count = resolve_gateway_approval(prompt.session_key, choice)
            if count:
                prompt.resolved = True
                self._approval_prompts_by_event.pop(reacts_to, None)
                self._approval_prompt_by_session.pop(prompt.session_key, None)
                logger.info(
                    "Matrix reaction resolved %d approval(s) for session %s (choice=%s, user=%s)",
                    count, prompt.session_key, choice, sender)
                await self._redact_bot_approval_reactions(room_id, prompt)
        except Exception as exc:
            logger.error("Failed to resolve gateway approval from Matrix reaction: %s", exc)
        return True

    async def _handle_model_picker_reaction(self, room_id: str, reacts_to: str, key: str, sender: str) -> bool:
        """Apply a model-picker reaction. True if the reaction targeted a pending picker."""
        return await self._handle_picker_reaction(
            self._model_picker_prompts_by_event, room_id, reacts_to, key, sender, "model picker",
            "That reaction is not one of the available model choices.", self._expire_matrix_model_picker_prompt,
            ("switch model", "switch model"), redact_bot_reactions=True)

    async def _handle_choice_picker_reaction(self, room_id: str, reacts_to: str, key: str, sender: str) -> bool:
        """Apply a choice-picker reaction. True if the reaction targeted a pending picker."""
        async def _expire(_room_id, target_event_id, _prompt):
            self._choice_picker_prompts_by_event.pop(target_event_id, None)
        return await self._handle_picker_reaction(
            self._choice_picker_prompts_by_event, room_id, reacts_to, key, sender, "choice picker",
            "That reaction is not one of the available choices.", _expire, ("apply choice", "apply selection"))

    async def _handle_picker_reaction(
        self, registry: dict, room_id: str, reacts_to: str, key: str, sender: str, label: str, invalid_text: str,
        on_expired, verbs: tuple[str, str], *, redact_bot_reactions: bool = False) -> bool:
        """Claim the picker, fire ``on_selected(room_id, *selection)`` and post its confirmation (or the error).
        ``verbs`` = (log verb, user-facing verb)."""
        handled, prompt, selection = await self._claim_reaction_prompt(
            registry, room_id, reacts_to, key, sender, label, invalid_text, on_expired)
        if selection is None:
            return handled
        prompt.resolved = True
        registry.pop(reacts_to, None)
        args = selection if isinstance(selection, tuple) else (selection,)
        try:
            confirmation = await prompt.on_selected(room_id, *args)
            if redact_bot_reactions:
                await self._redact_bot_model_picker_reactions(room_id, prompt)
            if confirmation:
                await self.send(room_id, confirmation, reply_to=reacts_to)
        except Exception as exc:
            logger.error("Failed to %s from Matrix reaction: %s", verbs[0], exc)
            await self.send(room_id, f"Failed to {verbs[1]}: {exc}", reply_to=reacts_to)
        return True

    def _matrix_prompt_expired(self, prompt: Any) -> bool:
        expires_at = getattr(prompt, "expires_at", None)
        return expires_at is not None and time.monotonic() > float(expires_at)

    def _is_authorized_user(self, user_id: str) -> bool:
        """GATEWAY_ALLOW_ALL_USERS, or membership in MATRIX_ALLOWED_USERS."""
        return _env_truthy("GATEWAY_ALLOW_ALL_USERS") or bool(
            self._allowed_user_ids and user_id in self._allowed_user_ids)

    async def _validate_matrix_prompt_reactor(
        self, room_id: str, target_event_id: str, sender: str, prompt: Any, prompt_label: str) -> bool:
        if not self._is_authorized_user(sender):
            logger.info(
                "Matrix: ignoring %s reaction from unauthorized user %s on %s", prompt_label, sender, target_event_id)
            await self._send_invalid_reaction_feedback(
                room_id, target_event_id, "Only an authorized Matrix user can use these controls.")
            return False
        requester = getattr(prompt, "requester_user_id", None)
        # getattr: object.__new__-built test doubles may lack the attribute.
        if getattr(self, "_approval_require_sender", True) and requester and sender != requester:
            logger.info("Matrix: ignoring %s reaction from %s; requester is %s", prompt_label, sender, requester)
            await self._send_invalid_reaction_feedback(
                room_id, target_event_id, "Only the user who requested this action can use these controls.")
            return False
        return True

    async def _send_invalid_reaction_feedback(self, room_id: str, target_event_id: str, text: str) -> None:
        try:
            await self.send(room_id, text, reply_to=target_event_id)
        except Exception as exc:
            logger.debug("Matrix: failed to send invalid reaction feedback: %s", exc)

    async def _expire_matrix_approval_prompt(self, room_id: str, target_event_id: str, prompt: Any) -> None:
        prompt.resolved = True
        self._approval_prompts_by_event.pop(target_event_id, None)
        self._approval_prompt_by_session.pop(prompt.session_key, None)
        await self._redact_bot_approval_reactions(room_id, prompt)
        await self._send_invalid_reaction_feedback(
            room_id, target_event_id,
            "This approval prompt has expired. Run the command again if you still want to approve it.")

    async def _expire_matrix_model_picker_prompt(self, room_id: str, target_event_id: str, prompt: Any) -> None:
        prompt.resolved = True
        self._model_picker_prompts_by_event.pop(target_event_id, None)
        await self._redact_bot_model_picker_reactions(room_id, prompt)
        await self._send_invalid_reaction_feedback(
            room_id, target_event_id, "This model picker has expired. Run `/model` again to choose a model.")

    async def _redact_bot_approval_reactions(self, room_id: str, prompt: Any) -> None:
        """Redact the bot's seeded approval reactions (delayed), leaving only the user's reaction."""
        for emoji, evt_id in prompt.bot_reaction_events.items():
            self._schedule_reaction_redaction(room_id, evt_id, "approval resolved")
            logger.debug("Matrix: scheduled bot reaction redaction %s (%s)", emoji, evt_id)

    async def _redact_bot_model_picker_reactions(self, room_id: str, prompt: Any) -> None:
        for emoji, evt_id in prompt.bot_reaction_events.items():
            try:
                await self.redact_message(room_id, evt_id, "model picker resolved")
                logger.debug("Matrix: redacted model picker reaction %s (%s)", emoji, evt_id)
            except Exception as exc:
                logger.debug("Matrix: failed to redact model picker reaction %s: %s", emoji, exc)

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch the aggregated text."""
        current_task = asyncio.current_task()
        try:
            pending = self._pending_text_batches.get(key)
            last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
            near_split = last_len >= self._split_threshold
            await asyncio.sleep(self._text_batch_split_delay_seconds if near_split else self._text_batch_delay_seconds)
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            logger.info("[Matrix] Flushing text batch %s (%d chars)", key, len(event.text or ""))
            await self.handle_message(event)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    def _background_read_receipt(self, room_id: str, event_id: str) -> None:

        async def _send() -> None:
            try:
                await self.send_read_receipt(room_id, event_id)
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug("Matrix: background read receipt failed: %s", exc)
        asyncio.ensure_future(_send())

    async def send_read_receipt(self, room_id: str, event_id: str) -> bool:
        if not self._client:
            return False
        try:
            room, event = RoomID(room_id), EventID(event_id)
            if hasattr(self._client, "set_fully_read_marker"):
                await self._client.set_fully_read_marker(room, event, event)
            elif hasattr(self._client, "send_receipt"):
                await self._client.send_receipt(room, event)
            elif hasattr(self._client, "set_read_markers"):
                await self._client.set_read_markers(room, fully_read_event=event, read_receipt=event)
            else:
                logger.debug("Matrix: client has no read receipt method")
                return False
            logger.debug("Matrix: sent read receipt for %s in %s", event_id, room_id)
            return True
        except Exception as exc:
            logger.debug("Matrix: read receipt failed: %s", exc)
            return False

    async def _client_op(self, coro_factory, ok_msg: tuple, err_msg: str, *, level: str = "warning") -> bool:
        """Run one client call when connected: log *ok_msg* and return True, or log the error and return False."""
        if not self._client:
            return False
        try:
            await coro_factory()
            getattr(logger, "debug" if level == "debug" else "info")(*ok_msg)
            return True
        except Exception as exc:
            getattr(logger, level)(err_msg, exc)
            return False

    async def redact_message(self, room_id: str, event_id: str, reason: str = "") -> bool:
        return await self._client_op(
            lambda: self._client.redact(RoomID(room_id), EventID(event_id), reason=reason or None),
            ("Matrix: redacted %s in %s", event_id, room_id), "Matrix: redact error: %s")

    async def create_room(
        self, name: str = "", topic: str = "", invite: Optional[list] = None, is_direct: bool = False,
        preset: str = "private_chat") -> Optional[str]:
        if not self._client:
            return None
        if preset == "public_chat" and not _env_truthy("MATRIX_ALLOW_PUBLIC_ROOMS"):
            logger.warning("Matrix: refusing to create public room without MATRIX_ALLOW_PUBLIC_ROOMS=true")
            return None
        try:
            preset_enum = {
                "private_chat": RoomCreatePreset.PRIVATE, "public_chat": RoomCreatePreset.PUBLIC,
                "trusted_private_chat": RoomCreatePreset.TRUSTED_PRIVATE}.get(preset, RoomCreatePreset.PRIVATE)
            room_id = await self._client.create_room(
                name=name or None, topic=topic or None, invitees=[UserID(u) for u in (invite or [])],
                is_direct=is_direct, preset=preset_enum)
            room_id_str = str(room_id)
            self._joined_rooms.add(room_id_str)
            logger.info("Matrix: created room %s (%s)", room_id_str, name or "unnamed")
            return room_id_str
        except Exception as exc:
            logger.warning("Matrix: create_room error: %s", exc)
            return None

    async def invite_user(self, room_id: str, user_id: str) -> bool:
        return await self._client_op(
            lambda: self._client.invite_user(RoomID(room_id), UserID(user_id)),
            ("Matrix: invited %s to %s", user_id, room_id), "Matrix: invite error: %s")

    _VALID_PRESENCE_STATES = frozenset(("online", "offline", "unavailable"))

    async def set_presence(self, state: str = "online", status_msg: str = "") -> bool:
        if not self._client:
            return False
        if state not in self._VALID_PRESENCE_STATES:
            logger.warning("Matrix: invalid presence state %r", state)
            return False
        presence_map = {
            "online": PresenceState.ONLINE, "offline": PresenceState.OFFLINE, "unavailable": PresenceState.UNAVAILABLE}
        return await self._client_op(
            lambda: self._client.set_presence(presence=presence_map[state], status=status_msg or None),
            ("Matrix: presence set to %s", state), "Matrix: set_presence failed: %s", level="debug")

    @staticmethod
    def _state_event_value(event: Any, key: str) -> Optional[str]:
        """Extract a simple value from a Matrix state event object or dict (top-level, then .content)."""
        if event is None:
            return None
        for obj in (event, event.get("content") if isinstance(event, dict) else getattr(event, "content", None)):
            value = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
            if value:
                return str(value)
        return None

    async def _get_room_member_count(self, room_id: str) -> Optional[int]:
        """state_store first (cached), then a direct joined_members API query."""
        state_store = getattr(self._client, "state_store", None) if self._client else None
        if state_store:
            with suppress(Exception):
                members = await state_store.get_members(room_id)
                if members is not None:
                    return len(members)
        client = getattr(self, "_client", None)  # object.__new__-built test doubles may lack it
        if client is not None and hasattr(client, "joined_members"):
            with suppress(Exception):
                resp = await client.joined_members(room_id)
                if getattr(resp, "members", None) is not None:
                    return len(resp.members)
        return None

    async def _get_room_state_value(self, room_id: str, event_type: str, key: str) -> Optional[str]:
        """Fetch a stripped string field from a room state event, or None."""
        if not self._client or not hasattr(self._client, "get_state_event"):
            return None
        try:
            event = await self._client.get_state_event(RoomID(room_id), event_type)
        except Exception:
            return None
        value = (self._state_event_value(event, key) or "").strip()
        return value or None

    def _invalidate_room_identities(self, room_id: str | None = None) -> None:
        """Drop one cached room identity (or all when *room_id* is None)."""
        if room_id is None:
            self._room_identities.clear()
            self._room_identity_cached_at.clear()
        else:
            self._room_identities.pop(room_id, None)
            self._room_identity_cached_at.pop(room_id, None)

    async def _resolve_room_identity(self, room_id: str, *, force_refresh: bool = False) -> MatrixRoomIdentity:
        """Resolve room identity; member count is the primary DM signal (see below)."""
        cached = self._room_identities.get(room_id)
        ttl = self._room_identity_ttl_seconds
        cache_fresh = ttl <= 0 or time.monotonic() - self._room_identity_cached_at.get(room_id, 0.0) <= ttl
        if cached is not None and cache_fresh and not force_refresh:
            return cached
        room_name = await self._get_room_state_value(room_id, "m.room.name", "name")
        room_topic = await self._get_room_state_value(room_id, "m.room.topic", "topic")
        canonical_alias = await self._get_room_state_value(room_id, "m.room.canonical_alias", "alias")
        member_count = await self._get_room_member_count(room_id)
        has_explicit_name = bool(room_name)
        is_direct = bool(self._dm_rooms.get(room_id, False))
        # <=2 members is necessarily a DM regardless of m.direct/name (clients auto-name DMs
        # like "Alice & Bot"); fall back to m.direct + unnamed only when the count is unknown.
        is_likely_dm = (member_count is not None and member_count <= 2) or (is_direct and not has_explicit_name)
        identity = MatrixRoomIdentity(
            room_id=room_id, room_name=room_name, room_topic=room_topic, canonical_alias=canonical_alias,
            server_name=(room_id.rsplit(":", 1)[-1].strip() or None) if ":" in room_id else None,
            joined_member_count=member_count,
            is_direct_account_data=is_direct, display_name=room_name or canonical_alias or room_id,
            has_explicit_name=has_explicit_name, chat_type="dm" if is_likely_dm else "room",
            conflict=bool(is_direct and has_explicit_name and (member_count is None or member_count > 2)))
        if len(self._room_identities) >= self._room_identity_cache_max:
            oldest = min(self._room_identity_cached_at, key=self._room_identity_cached_at.get, default=None)
            if oldest:
                self._invalidate_room_identities(oldest)
        self._room_identities[room_id] = identity
        self._room_identity_cached_at[room_id] = time.monotonic()
        return identity

    async def _is_dm_room(self, room_id: str) -> bool:
        return (await self._resolve_room_identity(room_id)).chat_type == "dm"

    async def _fetch_m_direct(self, *, log_failure: bool = False, require_dict: bool = False):
        """Return the m.direct account-data mapping, or None when absent/unreadable."""
        try:
            resp = await self._client.get_account_data("m.direct")
        except Exception as exc:
            if log_failure:
                logger.debug("Matrix: get_account_data('m.direct') failed: %s", exc)
            return None
        if hasattr(resp, "content") and (not require_dict or isinstance(resp.content, dict)):
            return resp.content
        return resp if isinstance(resp, dict) else None

    async def _refresh_dm_cache(self) -> None:
        if not self._client:
            return
        dm_data = await self._fetch_m_direct(log_failure=True)
        if dm_data is None:
            return
        dm_room_ids = {str(r) for rooms in dm_data.values() if isinstance(rooms, list) for r in rooms if isinstance(r, str)}
        self._dm_rooms = {rid: (rid in dm_room_ids) for rid in self._joined_rooms}
        self._invalidate_room_identities()

    async def _record_dm_room(self, room_id: str, inviter: str) -> None:
        """Persist a room as DM in m.direct account data after an invite. ``m.direct`` is absent (404)
        until the account has had a DM; fetch the current mapping (if any), append *room_id* under
        *inviter*, write it back so ``_refresh_dm_cache`` sees the DM."""
        if not self._client:
            return
        dm_data: Dict[str, list] = await self._fetch_m_direct(require_dict=True) or {}
        rooms_for_user = dm_data.get(inviter, [])
        rooms_for_user = rooms_for_user if isinstance(rooms_for_user, list) else []
        if room_id not in rooms_for_user:
            rooms_for_user.append(room_id)
            dm_data[inviter] = rooms_for_user
            try:
                await self._client.set_account_data("m.direct", dm_data)
                logger.info("Matrix: recorded %s as DM room (inviter=%s)", room_id, inviter)
            except Exception as exc:
                logger.warning("Matrix: failed to update m.direct: %s", exc)
        # Local cache so _resolve_room_identity sees it immediately.
        self._dm_rooms[room_id] = True
        self._invalidate_room_identities(room_id)

    def _build_text_message_content(self, text: str, msgtype: str = "m.text") -> Dict[str, Any]:
        """Build Matrix text content with HTML and outbound mention metadata."""
        msg_content: Dict[str, Any] = {"msgtype": msgtype, "body": text}
        mention_user_ids = self._extract_outbound_mentions(text)
        if mention_user_ids:
            msg_content["m.mentions"] = {"user_ids": mention_user_ids}
        if self._allow_room_mentions and self._has_outbound_room_mention(text):
            msg_content.setdefault("m.mentions", {})["room"] = True
        html = self._markdown_to_html(self._inject_outbound_mention_links(text))
        if html and html != text:
            msg_content["format"] = "org.matrix.custom.html"
            msg_content["formatted_body"] = html
        return msg_content

    def _apply_relation_metadata(
        self, msg_content: Dict[str, Any], *, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Apply Matrix reply/thread relation metadata to an outbound payload."""
        thread_id = str((metadata or {}).get("thread_id") or "")
        if reply_to:
            msg_content["m.relates_to"] = {"m.in_reply_to": {"event_id": reply_to}}
        if thread_id:
            relates_to = msg_content.get("m.relates_to", {})
            relates_to["rel_type"] = "m.thread"
            relates_to["event_id"] = thread_id
            relates_to["is_falling_back"] = True
            # Non-thread clients render the reply fallback; default it to the thread root.
            relates_to.setdefault("m.in_reply_to", {"event_id": reply_to or thread_id})
            msg_content["m.relates_to"] = relates_to

    def _extract_outbound_mentions(self, text: str) -> list[str]:
        protected, _ = self._protect_outbound_mention_regions(text)
        return list(dict.fromkeys(m.group(1) for m in _OUTBOUND_MENTION_RE.finditer(protected)))

    def _has_outbound_room_mention(self, text: str) -> bool:
        """Return True when outbound text contains @room outside protected spans."""
        protected, _ = self._protect_outbound_mention_regions(text)
        return bool(re.search(r"(?<![\w/])@room(?![\w:.-])", protected))

    def _inject_outbound_mention_links(self, text: str) -> str:
        """Wrap outbound Matrix mentions in markdown links outside code spans."""
        if not text:
            return text
        protected, placeholders = self._protect_outbound_mention_regions(text)
        linked = _OUTBOUND_MENTION_RE.sub(lambda m: f"[{m.group(1)}](https://matrix.to/#/{m.group(1)})", protected)
        for idx, original in enumerate(placeholders):
            linked = linked.replace(f"\x00MENTION_PROTECTED{idx}\x00", original)
        return linked

    def _protect_outbound_mention_regions(self, text: str) -> tuple[str, list[str]]:
        """Protect markdown regions where outbound mentions should stay literal."""
        placeholders: list[str] = []

        def _protect(fragment: str) -> str:
            idx = len(placeholders)
            placeholders.append(fragment)
            return f"\x00MENTION_PROTECTED{idx}\x00"
        protected = text or ""
        for pattern in (r"```[\s\S]*?```", r"`[^`\n]+`", r"\[[^\]]+\]\([^)]+\)"):
            protected = re.sub(pattern, lambda match: _protect(match.group(0)), protected)
        return protected, placeholders

    def _is_bot_mentioned(
        self, body: str, formatted_body: Optional[str] = None, mention_user_ids: Optional[list] = None) -> bool:
        """True if the bot is mentioned; ``m.mentions.user_ids`` (MSC3952) is authoritative
        even when the body has no ``@bot`` text (pills may live only in formatted_body)."""
        if mention_user_ids and self._user_id and self._user_id in mention_user_ids:
            return True
        if not body and not formatted_body:
            return False
        if self._user_id and self._user_id in body:
            return True
        localpart = self._user_localpart()
        if localpart and re.search(r"\b" + re.escape(localpart) + r"\b", body, re.IGNORECASE):
            return True
        return bool(formatted_body and self._user_id and f"matrix.to/#/{self._user_id}" in formatted_body)

    def _user_localpart(self) -> str:
        """``@bot:server`` -> ``bot``; empty when the user ID has no server part."""
        return self._user_id.split(":")[0].lstrip("@") if self._user_id and ":" in self._user_id else ""

    def _strip_mention(self, body: str) -> str:
        """Strip explicit ``@user:server`` / ``@localpart`` tokens only — never bare localpart
        words, or "Hermes Agent" would become "Agent"."""
        if not body:
            return ""
        if self._user_id:
            body = body.replace(self._user_id, "")
        localpart = self._user_localpart()
        if localpart:
            body = re.sub(r'(?<![\w])@' + re.escape(localpart) + r'\b', '', body, flags=re.IGNORECASE)
        # Normalize spacing after mention removal.
        body = re.sub(r'[ \t]{2,}', ' ', body)
        body = re.sub(r'\s+([,.;:!?])', r'\1', body)
        return body.strip()

    async def _get_display_name(self, room_id: str, user_id: str) -> str:
        """Get a user's display name in a room, falling back to user_id."""
        state_store = getattr(self._client, "state_store", None) if self._client else None
        if state_store:
            with suppress(Exception):
                member = await state_store.get_member(room_id, user_id)
                if member and getattr(member, "displayname", None):
                    return member.displayname
        if user_id.startswith("@") and ":" in user_id:
            return user_id[1:].split(":")[0]
        return user_id

    def _mxc_to_http(self, mxc_url: str) -> str:
        if not mxc_url.startswith("mxc://"):
            return mxc_url
        return f"{self._homeserver}/_matrix/client/v1/media/download/{mxc_url[6:]}"

    def _markdown_to_html(self, text: str) -> str:
        """Markdown → org.matrix.custom.html via ``markdown`` when installed, else the regex fallback."""
        text = _pre_sanitize_matrix_markdown(text)
        with suppress(ImportError):
            import markdown as _md
            md = _md.Markdown(extensions=["fenced_code", "tables", "nl2br", "sane_lists"])
            if "html_block" in md.preprocessors:
                md.preprocessors.deregister("html_block")
            html = md.convert(text)
            md.reset()
            if html.count("<p>") == 1:
                html = html.replace("<p>", "").replace("</p>", "")
            return _sanitize_matrix_html(html)
        return _sanitize_matrix_html(self._markdown_to_html_fallback(text))

    @staticmethod
    def _sanitize_link_url(url: str) -> str:
        stripped = url.strip()
        if ":" in stripped and stripped.split(":", 1)[0].lower().strip() in {"javascript", "data", "vbscript"}:
            return ""
        return stripped.replace('"', "&quot;")

    @staticmethod
    def _markdown_to_html_fallback(text: str) -> str:
        """Comprehensive regex Markdown-to-HTML for Matrix."""
        placeholders: list = []

        def _is_bq_line(ln: str) -> bool:
            return ln.startswith(("&gt; ", "> ")) or ln in ("&gt;", ">")

        def _protect_html(html_fragment: str) -> str:
            idx = len(placeholders)
            placeholders.append(html_fragment)
            return f"\x00PROTECTED{idx}\x00"

        result = re.sub(
            r"```(\w*)\n(.*?)```",
            lambda m: _protect_html(
                f'<pre><code class="language-{_html_escape(m.group(1))}">{_html_escape(m.group(2))}</code></pre>'
                if m.group(1) else f"<pre><code>{_html_escape(m.group(2))}</code></pre>"),
            text, flags=re.DOTALL)
        result = re.sub(r"`([^`\n]+)`", lambda m: _protect_html(f"<code>{_html_escape(m.group(1))}</code>"), result)
        # Protect markdown links before escaping.
        result = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            lambda m: _protect_html(
                f'<a href="{MatrixAdapter._sanitize_link_url(m.group(2))}">{_html_escape(m.group(1))}</a>'),
            result)
        result = "".join(p if p.startswith("\x00PROTECTED") else _html_escape(p)
                         for p in re.split(r"(\x00PROTECTED\d+\x00)", result))
        # Block-level transforms (line-oriented): hr, headers, blockquote, lists.
        lines = result.split("\n")
        out_lines: list = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if re.match(r"^[\s]*([-*_])\s*\1\s*\1[\s\-*_]*$", line):
                out_lines.append("<hr>")
                i += 1
                continue
            hdr = re.match(r"^(#{1,6})\s+(.+)$", line)
            if hdr:
                level = len(hdr.group(1))
                out_lines.append(f"<h{level}>{hdr.group(2).strip()}</h{level}>")
                i += 1
                continue
            if _is_bq_line(line):
                bq_lines = []
                while i < len(lines) and _is_bq_line(lines[i]):
                    ln = lines[i]
                    bq_lines.append(ln[5:] if ln.startswith("&gt; ") else ln[2:] if ln.startswith("> ") else "")
                    i += 1
                out_lines.append(f"<blockquote>{'<br>'.join(bq_lines)}</blockquote>")
                continue
            for item_re, tag in ((r"^[\s]*[-*+]\s+(.+)$", "ul"), (r"^[\s]*\d+[.)]\s+(.+)$", "ol")):
                if re.match(item_re, line):
                    items = []
                    while i < len(lines) and re.match(item_re, lines[i]):
                        items.append(re.match(item_re, lines[i]).group(1))
                        i += 1
                    out_lines.append(f"<{tag}>{''.join(f'<li>{item}</li>' for item in items)}</{tag}>")
                    break
            else:
                out_lines.append(line)
                i += 1
        result = "\n".join(out_lines)
        for pattern, repl in (
            (r"\*\*(.+?)\*\*", r"<strong>\1</strong>"), (r"__(.+?)__", r"<strong>\1</strong>"),
            (r"\*(.+?)\*", r"<em>\1</em>"), (r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>"),
            (r"~~(.+?)~~", r"<del>\1</del>")):
            result = re.sub(pattern, repl, result, flags=re.DOTALL)
        result = re.sub(r"\n", "<br>\n", result)
        result = re.sub(r"<br>\n(</?(?:pre|blockquote|h[1-6]|ul|ol|li|hr))", r"\n\1", result)
        result = re.sub(r"(</(?:pre|blockquote|h[1-6]|ul|ol|li)>)<br>", r"\1", result)
        for idx, original in enumerate(placeholders):
            result = result.replace(f"\x00PROTECTED{idx}\x00", original)
        return result


async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    """standalone_sender_fn: out-of-process delivery via the Client-Server API (cron without gateway)."""
    extra = getattr(pconfig, "extra", {}) or {}
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        homeserver = (extra.get("homeserver") or os.getenv("MATRIX_HOMESERVER", "")).rstrip("/")
        # In-turn read inside an installed secret scope: honor get_secret, no env fallback.
        token = getattr(pconfig, "token", None) or get_secret("MATRIX_ACCESS_TOKEN", "") or ""
        if not homeserver or not token:
            return {"error": "Matrix not configured (MATRIX_HOMESERVER, MATRIX_ACCESS_TOKEN required)"}
        txn_id = f"hermes_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
        from urllib.parse import quote
        url = f"{homeserver}/_matrix/client/v3/rooms/{quote(chat_id, safe='')}/send/m.room.message/{txn_id}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"msgtype": "m.text", "body": message}
        with suppress(ImportError):
            import markdown as _md
            html = _md.markdown(message, extensions=["fenced_code", "tables"])
            payload["format"] = "org.matrix.custom.html"
            payload["formatted_body"] = re.sub(r"<h[1-6]>(.*?)</h[1-6]>", r"<strong>\1</strong>", html)
        # asyncio.wait_for, not aiohttp.ClientTimeout: cron invokes this via
        # run_coroutine_threadsafe ("Timeout context manager should be used inside a task").
        async with aiohttp.ClientSession() as session:
            async def _do_send():
                async with session.put(url, headers=headers, json=payload) as resp:
                    if resp.status not in {200, 201}:
                        return {"error": f"Matrix API error ({resp.status}): {await resp.text()}"}
                    data = await resp.json()
                    return {"success": True, "platform": "matrix", "chat_id": chat_id,
                            "message_id": data.get("event_id")}
            try:
                return await asyncio.wait_for(_do_send(), timeout=30)
            except asyncio.TimeoutError:
                return {"error": "Matrix API timeout (30s)"}
    except Exception as e:
        return {"error": f"Matrix send failed: {e}"}


def interactive_setup() -> None:
    """Interactive credential setup (setup_fn); CLI helpers are lazy-imported."""
    from hermes_cli.config import get_env_value, remove_env_value, save_env_value
    from hermes_cli.cli_output import prompt, prompt_yes_no, print_header, print_info, print_success, print_warning
    print_header("Matrix")
    existing = get_env_value("MATRIX_ACCESS_TOKEN") or get_env_value("MATRIX_PASSWORD")
    if existing:
        print_info("Matrix: already configured")
        if not prompt_yes_no("Reconfigure Matrix?", False):
            return
    for line in ("Works with any Matrix homeserver (Synapse, Conduit, Dendrite, or matrix.org).",
                 "   1. Create a bot user on your homeserver, or use your own account",
                 "   2. Get an access token from Element, or provide user ID + password"):
        print_info(line)
    def _ask(key: str, question: str, **kw) -> str:
        value = prompt(question, **kw)
        if value:
            save_env_value(key, value.rstrip("/") if key == "MATRIX_HOMESERVER" else value)
        return value
    _ask("MATRIX_HOMESERVER", "Homeserver URL (e.g. https://matrix.example.org)")
    print_info("Auth: provide an access token (recommended), or user ID + password.")
    token = _ask("MATRIX_ACCESS_TOKEN", "Access token (leave empty for password login)", password=True)
    if token:
        _ask("MATRIX_USER_ID", "User ID (@bot:server — optional, will be auto-detected)")
        print_success("Matrix access token saved")
    else:
        _ask("MATRIX_USER_ID", "User ID (@bot:server)")
        if _ask("MATRIX_PASSWORD", "Password", password=True):
            print_success("Matrix credentials saved")
    if token or get_env_value("MATRIX_PASSWORD"):
        want_e2ee = prompt_yes_no("Enable end-to-end encryption (E2EE)?", False)
        if want_e2ee:
            save_env_value("MATRIX_ENCRYPTION", "true")
            print_success("E2EE enabled")
        matrix_pkg = "mautrix[encryption]" if want_e2ee else "mautrix"
        from tools.lazy_deps import ensure as _lazy_ensure, feature_missing
        _missing_before = feature_missing("platform.matrix")
        if _missing_before:
            print_info(f"Installing {matrix_pkg} (+ {len(_missing_before)} runtime deps)...")
            try:
                _lazy_ensure("platform.matrix", prompt=False)
                print_success(f"{matrix_pkg} installed")
            except Exception as exc:
                print_warning(
                    "Install failed — run manually: pip install "
                    "'mautrix[encryption]' asyncpg aiosqlite Markdown aiohttp-socks")
                print_info(f"  Error: {exc}")
        print_info("🔒 Security: Restrict who can use your bot")
        print_info("   Matrix user IDs look like @username:server")
        allowed_users = prompt("Allowed user IDs (comma-separated, leave empty for open access)")
        if allowed_users:
            save_env_value("MATRIX_ALLOWED_USERS", allowed_users.replace(" ", ""))
            print_success("Matrix allowlist configured")
        else:
            print_info("⚠️  No allowlist set - anyone who can message the bot can use it!")
        for line in ("📬 Home Room: where Hermes delivers cron job results and notifications.",
                     "   Room IDs look like !abc123:server (shown in Element room settings)",
                     "   You can also set this later by typing /set-home in a Matrix room.",
                     "Leave blank to clear a previously saved home room (cron / notifications)."):
            print_info(line)
        home_room = prompt("Home room ID (leave empty to set later with /set-home)").strip()
        if home_room:
            save_env_value("MATRIX_HOME_ROOM", home_room)
        elif remove_env_value("MATRIX_HOME_ROOM"):
            print_info("Home room cleared.")


_YAML_LOWER_KEYS = (
    ("require_mention", "MATRIX_REQUIRE_MENTION"), ("process_notices", "MATRIX_PROCESS_NOTICES"),
    ("session_scope", "MATRIX_SESSION_SCOPE"), ("auto_thread", "MATRIX_AUTO_THREAD"),
    ("dm_mention_threads", "MATRIX_DM_MENTION_THREADS"))
_YAML_LIST_KEYS = (
    ("allowed_users", "MATRIX_ALLOWED_USERS"), ("free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS"),
    ("allowed_rooms", "MATRIX_ALLOWED_ROOMS"), ("ignore_user_patterns", "MATRIX_IGNORE_USER_PATTERNS"))


def _apply_yaml_config(yaml_cfg: dict, matrix_cfg: dict) -> dict | None:
    """apply_yaml_config_fn: config.yaml matrix: keys → MATRIX_* env (env wins). Returns None. Lowercased
    flags apply whenever the key is present (None still writes "none"); list-valued keys skip None.

    Implements the apply_yaml_config_fn contract (#24849). Mirrors the legacy matrix_cfg block from
    gateway/config.py::load_gateway_config(). Env vars take precedence over YAML. Returns None — everything
    flows through env.
    """
    for key, env_name in _YAML_LOWER_KEYS:
        if key in matrix_cfg and not os.getenv(env_name):
            os.environ[env_name] = str(matrix_cfg[key]).lower()
    for key, env_name in _YAML_LIST_KEYS:
        value = matrix_cfg.get(key)
        if value is not None and not os.getenv(env_name):
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)
            os.environ[env_name] = str(value)
    if "max_message_length" in matrix_cfg and not os.getenv("MATRIX_MAX_MESSAGE_LENGTH"):
        os.environ["MATRIX_MAX_MESSAGE_LENGTH"] = str(matrix_cfg["max_message_length"])
    return None


def _is_connected(config) -> bool:
    """Connected = homeserver + token (or password). Reads via hermes_cli.gateway.get_env_value so
    setup-status callers that patch it see the same value; PlatformConfig extras are honored."""
    extra = getattr(config, "extra", {}) or {}
    import hermes_cli.gateway as gateway_mod
    homeserver = extra.get("homeserver") or gateway_mod.get_env_value("MATRIX_HOMESERVER") or ""
    token = (getattr(config, "token", None) or gateway_mod.get_env_value("MATRIX_ACCESS_TOKEN")
             or gateway_mod.get_env_value("MATRIX_PASSWORD") or "")
    return bool(str(homeserver).strip() and str(token).strip())


def _build_adapter(config):
    """Factory wrapper that constructs MatrixAdapter from a PlatformConfig."""
    return MatrixAdapter(config)


def register(ctx) -> None:
    ctx.register_platform(
        name="matrix", label="Matrix", adapter_factory=_build_adapter, check_fn=matrix_deps_present,
        ensure_deps_fn=ensure_matrix_deps, is_connected=_is_connected,
        required_env=["MATRIX_HOMESERVER", "MATRIX_ACCESS_TOKEN"], install_hint="pip install 'mautrix[encryption]'",
        setup_fn=interactive_setup, apply_yaml_config_fn=_apply_yaml_config, allowed_users_env="MATRIX_ALLOWED_USERS",
        allow_all_env="MATRIX_ALLOW_ALL_USERS", cron_deliver_env_var="MATRIX_HOME_ROOM",
        standalone_sender_fn=_standalone_send, max_message_length=DEFAULT_MAX_MESSAGE_LENGTH, emoji="🔐",
        allow_update_command=True)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

MAX_MESSAGE_LENGTH = DEFAULT_MAX_MESSAGE_LENGTH

_MATRIX_CAPABILITIES: Dict[str, str] = {
    "text": "yes",
    "threads": "yes",
    "reactions": "yes",
    "approvals": "yes",
    "model picker": "yes",
    "thinking panes": "yes",
    "images": "yes",
    "multiple images": "yes",
    "files": "yes",
    "voice/audio": "yes",
    "video": "yes",
    "E2EE": "off / optional / required",
    "diagnostics": "yes",
}

def get_matrix_capabilities() -> Dict[str, str]:
    """Return Matrix gateway capabilities for docs and release checks."""
    return dict(_MATRIX_CAPABILITIES)
# ---- END PLUGIN-COMPAT ----
