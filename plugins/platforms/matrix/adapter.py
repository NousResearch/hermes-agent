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
from .choice_picker import MatrixChoicePickerPrompt

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


from .adapter_lifecycle import MatrixLifecycleMixin
from .adapter_delivery import MatrixDeliveryMixin
from .adapter_prompts import MatrixPromptsMixin
from .adapter_inbound import MatrixInboundMixin


class MatrixAdapter(
    MatrixLifecycleMixin, MatrixDeliveryMixin, MatrixPromptsMixin, MatrixInboundMixin,
    BasePlatformAdapter,
):
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
        self._choice_picker_prompts_by_event: Dict[str, MatrixChoicePickerPrompt] = {}
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

    # Template attrs for the shared _format_exec_approval core (header + fence + reason only;
    # the smart-deny/scope wording lives in the reaction legend below).
    _EA_HEADER = "⚠️ **Dangerous command requires approval**\n"
    _EA_CMD_BUDGET = 2000

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
