from __future__ import annotations

"""
Discord platform adapter.

Uses discord.py library for:
- Receiving messages from servers and DMs
- Sending responses back
- Handling threads and channels
"""

import asyncio
import datetime as dt
import hashlib
import inspect
import json
import logging
import math
import os
import re
import struct
import subprocess
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from contextlib import suppress
from typing import Callable, Dict, List, Optional, Any, Tuple
from urllib.parse import quote, urljoin

from agent.async_utils import (consume_detached_task_result as _consume_background_task_result)
from agent.display import ToolPreview

logger = logging.getLogger(__name__)

_DISCORD_MARKDOWN_LINK_LABEL_RE = re.compile(r"([\\\[\]])")
_DISCORD_URL_LABEL_SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)


def _voice_mixer_module():
    """Sibling ``voice_mixer`` module: flat import (plugin dir on sys.path) else package-relative."""
    try:
        import voice_mixer
        return voice_mixer
    except ImportError:
        from . import voice_mixer
        return voice_mixer


def _image_ext_from_content_type(content_type: str) -> str:
    """Attachment extension for a downloaded image (png unless jpeg/gif/webp is evident)."""
    if "jpeg" in content_type or "jpg" in content_type:
        return "jpg"
    if "gif" in content_type:
        return "gif"
    if "webp" in content_type:
        return "webp"
    return "png"


def _format_discord_markdown_link(label: str, url: str) -> str:
    """Return a Discord Markdown link whose label is not itself a URL (URL-shaped labels can
    win as a broken link; the ``<url>`` angle brackets stop Discord unfurling an embed)."""
    label = _DISCORD_URL_LABEL_SCHEME_RE.sub("", label, count=1)
    escaped_label = _DISCORD_MARKDOWN_LINK_LABEL_RE.sub(r"\\\1", label)
    escaped_url = quote(url, safe=":/?#[]@!$&'*+,;=%")
    return f"[{escaped_label}](<{escaped_url}>)"


class _Snowflake:
    """``.id``-only Snowflake stand-in for ``channel.history(before=...)``; avoids
    ``discord.Object``, which stubbed discord test doubles cannot build."""

    __slots__ = ("id",)

    def __init__(self, id: int) -> None:  # noqa: A002 - matches discord API
        self.id = id

VALID_THREAD_AUTO_ARCHIVE_MINUTES = {60, 1440, 4320, 10080}
_DISCORD_COMMAND_SYNC_POLICIES = {"safe", "bulk", "off"}
_DISCORD_COMMAND_SYNC_STATE_SUBDIR = "gateway"
_DISCORD_COMMAND_SYNC_STATE_FILENAME = "discord_command_sync_state.json"
_DISCORD_NONCONVERSATIONAL_STATE_FILENAME = "discord_nonconversational_messages.json"

_DISCORD_COMMAND_SYNC_MUTATION_INTERVAL_SECONDS = 4.5
_DISCORD_COMMAND_SYNC_MAX_RATE_LIMIT_SLEEP_SECONDS = 30.0
# Discord caps global slash commands at 100/app; exceeding it fails the ENTIRE sync (error 30032).
_DISCORD_MAX_APP_COMMANDS = 100
# Native slash commands (registered before COMMAND_REGISTRY/plugins so they survive the 100 cap):
#   (discord name, description, [(arg, type, default-or-_REQUIRED, arg description,
#   [(choice label, value), ...] or None)], command-text template, follow-up message)
# Placeholders are the arg names; text is `.strip()`ped unless ``strip`` is False.
_REQUIRED = object()
_NATIVE_SLASH_COMMANDS: tuple = (
    # /thread: template None -> registered by _register_thread_slash (auth-gated defer).
    ("thread", "Create a new thread and start a Hermes session in it", (), None, None),
)
_DISCORD_SELECT_FIELD_LIMIT = 100
# Discord caps a single select menu at 25 options; a View holds at most 5 rows.
_DISCORD_SELECT_MAX_OPTIONS = 25
_DISCORD_SELECT_MAX_ROWS = 5
# Model-select capacity: keep 2 rows for Back/Cancel, fill the rest with selects.
_DISCORD_MODEL_SELECT_CAPACITY = (_DISCORD_SELECT_MAX_ROWS - 2) * _DISCORD_SELECT_MAX_OPTIONS
_DISCORD_BUTTON_LABEL_LIMIT = 80
_DISCORD_ELLIPSIS = "\u2026"
_DISCORD_NONCONVERSATIONAL_METADATA_KEYS = frozenset({
    "non_conversational", "non_conversational_history",
})
_DISCORD_IMAGE_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_DISCORD_IMAGE_MAX_REDIRECTS = 10
# Upgrade-bridge fallback: recognizes status bumps from gateway versions pre-dating
# metadata["non_conversational"]. New emitters must set the metadata flag, not add regexes.
_DISCORD_NONCONVERSATIONAL_HISTORY_MESSAGE_PATTERNS = (
    re.compile(r"^\s*💾\s*Self-improvement review:\s+\S[\s\S]*$", re.IGNORECASE),
    # Shorter legacy form still used by background-review test doubles.
    re.compile(
        r"^\s*💾\s+Skill\s+['\"].+?['\"]\s+(?:created|updated|improved|patched)\.?\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*⏳\s+Working\s+—\s+\d+\s+min(?:\s|$)", re.IGNORECASE),
    re.compile(
        r"^\s*\[Background process\s+\S+\s+"
        r"(?:finished with exit code|is still running~)[\s\S]*\]\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:✅|❌)\s+Hermes update\s+"
        r"(?:finished|failed|timed out)[\s\S]*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*♻️?\s+Gateway\s+(?:restarted successfully|online\b)[\s\S]*$", re.IGNORECASE),
)
try:
    import discord
    from discord import Message as DiscordMessage, Intents
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None
    DiscordMessage = Any
    Intents = Any
    commands = None

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))


def _is_discord_transport_error(exc: BaseException) -> bool:
    """True for connection-shaped send failures (dead/dropping WS) that never reached Discord, so
    the delivery ledger can replay them; timeouts excluded (a timed-out send may have landed).

    These are the failures where the message demonstrably did NOT reach Discord because the transport itself
    was down — the delivery-obligation ledger can safely replay them after reconnect (#95382). HTTP-level
    rejections (permissions, formatting, 4xx) are NOT transport errors and must keep their original error
    string.
    """
    if isinstance(exc, asyncio.TimeoutError):
        return False
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    if DISCORD_AVAILABLE and discord is not None:
        _transport_types = tuple(
            t
            for t in (
                getattr(discord, "ConnectionClosed", None),
                getattr(discord, "GatewayNotFound", None),
                getattr(discord, "DiscordServerError", None),
            )
            if isinstance(t, type)
        )
        if _transport_types and isinstance(exc, _transport_types):
            return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "websocket closed", "connection reset", "connection closed", "session is closed",
            "cannot write to closing transport", "not connected",
        )
    )


try:
    from .ffmpeg_utils import resolve_ffmpeg_executable
except ImportError:
    from ffmpeg_utils import resolve_ffmpeg_executable

from .voice import VoiceReceiver
from .commands.generated import GeneratedCommandMixin

from gateway.config import Platform, PlatformConfig

from gateway.platforms.helpers import (
    MessageDeduplicator, ThreadParticipationTracker, convert_table_to_bullets,
)
from utils import atomic_json_write, env_float
from gateway.platforms.base import (
    BasePlatformAdapter, SendResult,
    cache_image_from_url, cache_image_from_bytes_async, cache_audio_from_url, cache_audio_from_bytes_async,
    cache_document_from_bytes_async, SUPPORTED_DOCUMENT_TYPES, _TEXT_INJECT_EXTENSIONS,
    _prefix_within_utf16_limit, utf16_len, validate_inbound_media_size,
)
from .services.authorization import AuthorizationMixin
from .services.threads import ThreadLifecycleMixin
from .services.recovery import RecoveryMixin
from .services.lifecycle import LifecycleMixin
from .services.command_sync import CommandSyncMixin
from .services.messaging import MessagingMixin
from .services.gates import GatesMixin
from .services.ingress import IngressMixin
from .services.interactions import InteractionsMixin
from .services.state import MessageStateMixin
from .services.channel_context import ChannelContextMixin
from .services.config_runtime import RuntimeConfigMixin
from .services.initialization import InitializationMixin
from .events.normalization import EventNormalizationMixin
from .services.voice import VoiceLifecycleMixin
from .services.standalone import _standalone_send
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from tools.url_safety import is_safe_url
from gateway.platforms._shared import yaml_env_setter as _yaml_env_setter


async def _read_url_image_with_redirect_guard(
    session: Any, url: str, *, timeout: Any, request_kwargs: Dict[str, Any],
) -> Tuple[int, bytes, Dict[str, str]]:
    """Read an image URL while re-checking every redirect target for SSRF."""
    current_url = url
    for _ in range(_DISCORD_IMAGE_MAX_REDIRECTS + 1):
        if not is_safe_url(current_url):
            raise ValueError("Blocked unsafe image URL redirect")
        async with session.get(
            current_url, timeout=timeout, allow_redirects=False, **request_kwargs,
        ) as resp:
            raw_headers = getattr(resp, "headers", {}) or {}
            headers = {str(key).lower(): value for key, value in dict(raw_headers).items()}
            status = int(getattr(resp, "status", 0))
            if status in _DISCORD_IMAGE_REDIRECT_STATUSES:
                location = headers.get("location")
                if not location:
                    return status, b"", headers
                next_url = urljoin(current_url, str(location))
                if not is_safe_url(next_url):
                    raise ValueError("Blocked redirect to private/internal address")
                current_url = next_url
                continue
            return status, await resp.read(), headers
    raise ValueError("Too many image URL redirects")


def _truncate_discord_component_text(text: str, limit: int) -> str:
    """Return text within Discord's UTF-16 component field budget."""
    return _prefix_within_utf16_limit(str(text or ""), max(0, limit))


def _abort_discord_websocket_transport(websocket: Any) -> bool:
    """Abort the active aiohttp transport after a bounded close times out."""
    socket = getattr(websocket, "socket", None)
    response = getattr(socket, "_response", None)
    connection = getattr(socket, "_conn", None)
    if connection is None:
        connection = getattr(response, "connection", None)
    protocol = getattr(connection, "protocol", None)
    writer = getattr(socket, "_writer", None)
    transport = getattr(writer, "transport", None)
    if transport is None:
        transport = getattr(protocol, "transport", None)
    abort = getattr(transport, "abort", None)
    if not callable(abort):
        return False
    abort()
    return True


async def _wait_for_ready_or_bot_exit(
    ready_event: asyncio.Event, bot_task: asyncio.Task, timeout: Optional[float],
) -> None:
    """Wait until Discord is ready, or surface early bot startup failure (``Bot.start()`` errors
    would otherwise burn the full timeout on a dead task; racing preserves the exception)."""
    ready_task = asyncio.create_task(ready_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {ready_task, bot_task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            raise asyncio.TimeoutError
        if bot_task in done:
            exc = bot_task.exception()
            if exc is not None:
                raise exc
            if not ready_task.done():
                raise RuntimeError("Discord bot task exited before ready")
        await ready_task
    finally:
        if not ready_task.done():
            ready_task.cancel()
            with suppress(asyncio.CancelledError):
                await ready_task


def _needs_server_members_intent(
    allowed_user_ids: set[str] | list[str] | None, allowed_role_ids: set[str] | list[str] | None,
) -> bool:
    """True when Server Members intent is needed: username allowlist entries (not IDs / ``*``)
    or role allowlists needing member lookups. Message Content is always requested."""
    entries = allowed_user_ids or ()
    if any(entry != "*" and not str(entry).isdigit() for entry in entries):
        return True
    return bool(allowed_role_ids)


def _format_privileged_intents_guidance(*, needs_members: bool) -> str:
    """Actionable fix text when Discord rejects privileged Gateway Intents."""
    lines = [
        "Discord rejected the connection because privileged Gateway Intents "
        "are not enabled for this bot in the Developer Portal.",
        "Hermes is requesting:",
        "  - Message Content Intent (required to read message text)",
    ]
    if needs_members:
        lines.append(
            "  - Server Members Intent (required for username allowlists "
            "and/or DISCORD_ALLOWED_ROLES)"
        )
    lines.extend(
        [
            "Fix: https://discord.com/developers/applications → your application "
            "→ Bot → Privileged Gateway Intents → enable the intent(s) listed "
            "above → Save Changes, then restart the gateway.",
            "Docs: https://hermes-agent.nousresearch.com/docs/user-guide/messaging/discord",
        ]
    )
    return "\n".join(lines)


def _load_opus_codec() -> None:
    """Try bundled (Windows) opus, then ``ctypes.util.find_library``, then Homebrew paths
    (find_library misses Homebrew libs on macOS); warn once if none loads."""
    import ctypes.util
    opus_candidates = []
    bundled_opus = _find_discord_windows_bundled_opus(discord)
    if bundled_opus:
        opus_candidates.append(bundled_opus)
    opus_path = ctypes.util.find_library("opus")
    if opus_path:
        opus_candidates.append(opus_path)
    elif sys.platform == "darwin":
        for _hp in ("/opt/homebrew/lib/libopus.dylib", "/usr/local/lib/libopus.dylib"):  # Apple Silicon, Intel
            if os.path.isfile(_hp):
                opus_candidates.append(_hp)
                break
    for opus_path in opus_candidates:
        try:
            discord.opus.load_opus(opus_path)
            if discord.opus.is_loaded():
                break
        except Exception:
            logger.warning("Opus codec found at %s but failed to load", opus_path)
    if not discord.opus.is_loaded():
        logger.warning("Opus codec not found — voice channel playback disabled")


def _find_discord_windows_bundled_opus(discord_module: Any = None) -> Optional[str]:
    """Return discord.py's bundled Windows opus DLL path when present."""
    if sys.platform != "win32":
        return None
    discord_module = discord if discord_module is None else discord_module
    if discord_module is None:
        return None
    opus_module = getattr(discord_module, "opus", None)
    opus_file = getattr(opus_module, "__file__", None)
    if not opus_file:
        return None
    target = "x64" if struct.calcsize("P") * 8 > 32 else "x86"
    bundled = _Path(opus_file).resolve().parent / "bin" / f"libopus-0.{target}.dll"
    if bundled.is_file():
        return str(bundled)
    return None


class _DiscordNonConversationalMessageTracker:
    """Persistent bounded set of Discord message IDs that are status noise."""

    _MAX_TRACKED = 2000

    def __init__(self, max_tracked: int = _MAX_TRACKED):
        self._max_tracked = max_tracked
        self._ids: dict[str, None] = dict.fromkeys(self._load())
        # Serializes the offloaded flushes so two concurrent mark_many() calls
        # cannot land their writes out of order (last-writer-wins would drop
        # the newer ids from disk).
        self._persist_lock = asyncio.Lock()

    def _state_path(self) -> _Path:
        from hermes_constants import get_hermes_home
        return (
            get_hermes_home()
            / _DISCORD_COMMAND_SYNC_STATE_SUBDIR
            / _DISCORD_NONCONVERSATIONAL_STATE_FILENAME
        )

    def _load(self) -> list[str]:
        path = self._state_path()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(message_id) for message_id in data if str(message_id).strip()]
        except Exception:
            logger.debug("[%s] Failed to load non-conversational Discord IDs", "Discord")
        return []

    def _snapshot(self) -> list[str]:
        """Trim in-memory state and return the ids to persist (loop-side)."""
        ids = list(self._ids)
        if len(ids) > self._max_tracked:
            ids = ids[-self._max_tracked:]
            self._ids = dict.fromkeys(ids)
        return ids

    def _save(self, ids: list[str]) -> None:
        try:
            atomic_json_write(self._state_path(), ids, indent=None)
        except Exception:
            logger.debug("[%s] Failed to save non-conversational Discord IDs", "Discord", exc_info=True)

    async def mark_many(self, message_ids: List[str]) -> None:
        changed = False
        for message_id in message_ids:
            key = str(message_id or "").strip()
            if key and key not in self._ids:
                self._ids[key] = None
                changed = True
        if changed:
            # atomic_json_write() calls os.fsync(), which blocks until the
            # write reaches stable storage. Both callers of mark_many() run
            # on the event loop, so offload the flush the same way #83906
            # did for the other gateway persist paths. The snapshot (and the
            # trim that reassigns ``_ids``) stays on the loop so the worker
            # never touches the dict while another task mutates it; the lock
            # keeps flushes in mutation order.
            async with self._persist_lock:
                ids = self._snapshot()
                await asyncio.to_thread(self._save, ids)

    def __contains__(self, message_id: str) -> bool:
        return str(message_id or "") in self._ids


def _metadata_marks_nonconversational(metadata: Optional[Dict[str, Any]]) -> bool:
    """Return True when an outbound send was explicitly marked as status-only."""
    if not isinstance(metadata, dict):
        return False
    return any(bool(metadata.get(key)) for key in _DISCORD_NONCONVERSATIONAL_METADATA_KEYS)


def _prompt_target_id(chat_id: str, metadata: Optional[dict]) -> str:
    """Interactive prompts post into ``metadata["thread_id"]`` when present, else ``chat_id``."""
    if metadata and metadata.get("thread_id"):
        return metadata["thread_id"]
    return chat_id


def _looks_like_nonconversational_history_message(content: str) -> bool:
    """Fallback recognizer for legacy status bumps missing persisted IDs."""
    text = content or ""
    return any(pattern.match(text) for pattern in _DISCORD_NONCONVERSATIONAL_HISTORY_MESSAGE_PATTERNS)


def _clean_discord_id(entry: str) -> str:
    """Strip pasted prefixes (``user:123``, ``<@123>``, ``<@!123>``) to a bare ID/username."""
    entry = entry.strip()
    if entry.startswith("<@") and entry.endswith(">"):
        entry = entry.lstrip("<@!").rstrip(">")
    if entry.lower().startswith("user:"):
        entry = entry[5:]
    return entry.strip()


# Under gateway.multiplex_profiles os.environ is process-global and first-writer-wins, so raw
# os.getenv() can return ANOTHER profile's value; _scoped_gate_env reads the active profile's
# secret scope (contextvar propagates into connect()) and falls back to os.getenv outside multiplex.

# Authorization/gate env vars snapshotted per-adapter at connect() time.
# ── per-profile gate env reads (issue #72348) ──────────────────────────── Under
# gateway.multiplex_profiles, os.environ is process-global and the YAML→env bridge in _apply_yaml_config is
# first-writer-wins, so a raw os.getenv() on an allow/deny gate can return ANOTHER profile's value.
# _scoped_gate_env reads the active profile's secret scope when one is installed (secondary adapters connect
# — and their discord.py event tasks are created — inside _profile_runtime_scope, so the contextvar
# propagates) and falls back to os.getenv only outside multiplex.
_GATE_ENV_KEYS = (
    "DISCORD_ALLOWED_USERS", "DISCORD_ALLOWED_ROLES", "DISCORD_ALLOWED_CHANNELS",
    "DISCORD_IGNORED_CHANNELS", "DISCORD_NO_THREAD_CHANNELS", "DISCORD_FREE_RESPONSE_CHANNELS",
    "DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS", "DISCORD_ALLOW_ALL_USERS", "DISCORD_ALLOW_BOTS",
    "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS",
)


def _scoped_gate_env(name: str, default: str = "") -> str:
    """Scope-aware gate env read: profile secret scope first under multiplex."""
    try:
        from gateway.authz_mixin import _platform_gate_env
        return _platform_gate_env(name, default)
    except Exception:
        return (os.getenv(name) or default).strip()


def _multiplex_active() -> bool:
    """True when the gateway is running in multiplex_profiles mode."""
    try:
        from agent.secret_scope import is_multiplex_active
        return bool(is_multiplex_active())
    except Exception:
        return False


def discord_deps_present() -> bool:
    """PASSIVE probe: is discord.py importable? Registry ``check_fn`` — must never install
    (the ACTIVE installer ``check_discord_requirements`` runs as ``ensure_deps_fn``).

    Registry ``check_fn`` — called from status displays and config loading, so it must never install
    anything. The ACTIVE lazy-installer (``check_discord_requirements``) is registered as ``ensure_deps_fn``
    and runs from ``create_adapter()`` when this returns False (#79812).
    """
    return DISCORD_AVAILABLE


def check_discord_requirements() -> bool:
    """Check Discord deps; lazy-installs discord.py on first call and re-binds
    module globals so ``DISCORD_AVAILABLE`` becomes True."""
    global DISCORD_AVAILABLE, discord, DiscordMessage, Intents, commands
    if DISCORD_AVAILABLE:
        return True
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("platform.discord", prompt=False)
    except Exception:
        return False
    try:
        import discord as _discord
        from discord import Message as _DM, Intents as _Intents
        from discord.ext import commands as _commands
    except ImportError:
        return False
    discord = _discord
    DiscordMessage = _DM
    Intents = _Intents
    commands = _commands
    DISCORD_AVAILABLE = True
    _define_discord_view_classes()
    return True


def _build_allowed_mentions(extra: Optional[dict] = None):
    """Build Discord ``AllowedMentions`` denying @everyone/@here/roles by default (any LLM output
    with ``@everyone`` would otherwise ping the server); user / replied-user pings stay on.

    Override via ``discord.allow_mentions.*`` in config.yaml (``extra["allow_mentions"]``, per profile)
    or env — a secondary multiplex profile never sees the default profile's env (#72348):

        DISCORD_ALLOW_MENTION_EVERYONE      default false  — @everyone + @here
        DISCORD_ALLOW_MENTION_ROLES         default false  — @role pings
        DISCORD_ALLOW_MENTION_USERS         default true   — @user pings
        DISCORD_ALLOW_MENTION_REPLIED_USER  default true   — reply-ping author
    """
    if not DISCORD_AVAILABLE:
        return None
    configured = (extra or {}).get("allow_mentions")
    configured = configured if isinstance(configured, dict) else {}

    def _b(name: str, key: str, default: bool) -> bool:
        if (raw := configured.get(key)) is not None:
            return str(raw).strip().lower() in {"true", "1", "yes", "on"}
        return _env_bool(name, default)

    return discord.AllowedMentions(
        everyone=_b("DISCORD_ALLOW_MENTION_EVERYONE", "everyone", False),
        roles=_b("DISCORD_ALLOW_MENTION_ROLES", "roles", False),
        users=_b("DISCORD_ALLOW_MENTION_USERS", "users", True),
        replied_user=_b("DISCORD_ALLOW_MENTION_REPLIED_USER", "replied_user", True),
    )


def _discord_ready_timeout_seconds() -> float:
    """Return the Discord ready wait timeout during gateway startup."""
    raw = os.getenv("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            logger.warning("Ignoring invalid HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT=%r", raw)
    return 30.0



def _read_dm_role_auth_guild() -> Optional[int]:
    """Return the guild ID opted-in for DM role-based auth, or None (secure default). Read from
    config.yaml ``discord.dm_role_auth_guild`` only (behavioral, not a secret); int or numeric string."""
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config() or {}
        discord_cfg = cfg.get("discord", {}) or {}
        raw = discord_cfg.get("dm_role_auth_guild")
    except Exception:
        return None
    if raw is None or raw == "":
        return None
    try:
        guild_id = int(raw)
    except (TypeError, ValueError):
        return None
    return guild_id if guild_id > 0 else None


# Default timeout for Discord button views when ``approvals.discord_prompt_timeout`` is unset;
# Discord interaction tokens expire at ~15 minutes, so 900s is the practical ceiling.
_DISCORD_PROMPT_TIMEOUT_DEFAULT = 300
_DISCORD_PROMPT_TIMEOUT_MIN = 30
_DISCORD_PROMPT_TIMEOUT_MAX = 900


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _scoped_gate_env(name).lower()
    if not raw:
        return default
    return raw in {"true", "1", "yes", "on"}


def _read_discord_prompt_timeout() -> int:
    """Timeout (seconds) for Discord button views from ``approvals.discord_prompt_timeout``
    (default 300), clamped to [MIN, MAX] so a typo can't make prompts vanish or outlive tokens."""
    raw: Any = None
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config() or {}
        approvals_cfg = cfg.get("approvals", {}) or {}
        raw = approvals_cfg.get("discord_prompt_timeout")
    except Exception:
        return _DISCORD_PROMPT_TIMEOUT_DEFAULT
    if raw is None or raw == "":
        return _DISCORD_PROMPT_TIMEOUT_DEFAULT
    try:
        seconds = int(raw)
    except (TypeError, ValueError):
        return _DISCORD_PROMPT_TIMEOUT_DEFAULT
    if seconds < _DISCORD_PROMPT_TIMEOUT_MIN:
        return _DISCORD_PROMPT_TIMEOUT_MIN
    if seconds > _DISCORD_PROMPT_TIMEOUT_MAX:
        return _DISCORD_PROMPT_TIMEOUT_MAX
    return seconds


from plugins.platforms.discord.adapter_media import DiscordMediaMixin


class DiscordAdapter(InitializationMixin, GeneratedCommandMixin, AuthorizationMixin, ThreadLifecycleMixin, RecoveryMixin, VoiceLifecycleMixin, LifecycleMixin, CommandSyncMixin, MessagingMixin, GatesMixin, IngressMixin, InteractionsMixin, MessageStateMixin, ChannelContextMixin, RuntimeConfigMixin, EventNormalizationMixin, DiscordMediaMixin, BasePlatformAdapter):
    """Discord bot adapter: guild/DM messages, threads, slash commands, button approvals, reactions."""

    MAX_MESSAGE_LENGTH = 2000
    _SPLIT_THRESHOLD = 1900  # near the 2000-char split point
    supports_code_blocks = True  # Discord markdown renders fenced code blocks natively
    splits_long_messages = True  # send() chunks via truncate_message(MAX_MESSAGE_LENGTH)
    # Safety ceiling on split deliveries: chunks beyond the cap become a notice (degenerate turns).
    # Safety ceiling on split deliveries (#86581): a degenerate turn can produce tens of thousands of
    # characters — without a cap the adapter posts every 2000-char chunk back-to-back and floods the channel
    # (the incident delivered 60,698 chars as 31 messages).
    MAX_SPLIT_MESSAGES = 8

    # Voice auto-disconnect after N idle seconds (discord.voice_channel_inactivity_timeout_seconds; 0 off).
    VOICE_TIMEOUT = 300
    # Minimum wait for one voice playback; the effective limit scales with clip duration.
    PLAYBACK_TIMEOUT = 120
    PLAYBACK_TIMEOUT_PADDING = 30

    def format_tool_preview(self, preview: ToolPreview) -> str:
        """Keep a truncated URL preview clickable in Discord markdown."""
        if not preview.url:
            return preview.text
        return _format_discord_markdown_link(preview.text, preview.url)



    async def _on_discord_ready(self) -> None:
        """Compatibility entrypoint; readiness orchestration lives in ``events.ready``."""
        from .events.ready import handle
        await handle(self)

    def _discord_message_admission(self, message: Any, *, claim: bool) -> tuple[bool, bool]:
        """Return ``(admitted, role_authorized)`` for one Discord event."""
        message_id = str(getattr(message, "id", ""))
        if claim:
            if self._dedup.is_duplicate(message_id):
                return False, False
        elif self._dedup.contains(message_id):
            return False, False
        if message.author == self._client.user:
            return False, False
        if message.type not in {discord.MessageType.default, discord.MessageType.reply}:
            return False, False
        role_authorized = False
        if getattr(message.author, "bot", False):
            allow_bots = self._get_allow_bots()
            if allow_bots == "none":
                return False, False
            if allow_bots == "mentions" and not self._self_is_explicitly_mentioned(message):
                return False, False
            if (
                self._discord_bots_require_inline_mention()
                and not self._self_is_raw_mentioned(message)
            ):
                return False, False
        else:
            msg_guild = getattr(message, "guild", None)
            is_dm = isinstance(message.channel, discord.DMChannel) or msg_guild is None
            msg_channel_ids = None
            if not is_dm:
                msg_channel_ids = {str(message.channel.id)}
                parent_id = self._get_parent_channel_id(message.channel)
                if parent_id:
                    msg_channel_ids.add(parent_id)
            if not self._is_allowed_user(
                str(message.author.id), message.author, guild=msg_guild, is_dm=is_dm,
                channel_ids=msg_channel_ids,
            ):
                self._warn_if_fail_closed_default()
                return False, False
            role_authorized = bool(getattr(self, "_allowed_role_ids", set()))
        raw_self_mention = self._self_is_explicitly_mentioned(message)
        if not isinstance(message.channel, discord.DMChannel) and (
            message.mentions or raw_self_mention
        ):
            other_bots_mentioned = any(
                mentioned.bot and mentioned != self._client.user
                for mentioned in message.mentions
            )
            if other_bots_mentioned and not raw_self_mention:
                return False, False
            ignore_no_mention = _scoped_gate_env("DISCORD_IGNORE_NO_MENTION", "true").lower() in {"true", "1", "yes"}
            if ignore_no_mention and not raw_self_mention and not other_bots_mentioned:
                parent_id = None
                if hasattr(message.channel, "parent_id") and message.channel.parent_id:
                    parent_id = str(message.channel.parent_id)
                free_channels = self._discord_free_response_channels()
                channel_keys = self._discord_channel_keys(message, parent_id)
                if "*" not in free_channels and not (channel_keys & free_channels):
                    return False, False
        return True, role_authorized

    async def _dispatch_discord_message(self, message: Any) -> bool:
        """Compatibility entrypoint; ingress orchestration lives in ``events.message_create``."""
        from .events.message_create import handle
        return await handle(message, self)

    # --- gateway_platform_event fire-sites ---

    async def _on_platform_message_edit(self, before, after) -> None:
        """Compatibility entrypoint; normalization lives in ``events.message_edit``."""
        from .events.message_edit import handle
        await handle(before, after, self)

    async def _on_platform_message_delete(self, message) -> None:
        """Compatibility entrypoint; normalization lives in ``events.message_delete``."""
        from .events.message_delete import handle
        await handle(message, self)

    async def _on_platform_thread_create(self, thread) -> None:
        """Compatibility entrypoint; normalization lives in ``events.thread_create``."""
        from .events.thread_create import handle
        await handle(thread, self)

    async def _on_platform_thread_update(self, before, after) -> None:
        """Compatibility entrypoint; normalization lives in ``events.thread_update``."""
        from .events.thread_update import handle
        await handle(before, after, self)




    # Attachment download helpers
    # Prefer the authenticated bot session (``att.read()``): CDN URLs increasingly 403 without
    # bot auth and some VPN DNS setups make ``is_safe_url`` flag the CDN as SSRF. If ``read()``
    # is missing or fails, fall back to the SSRF-gated URL downloaders (defense-in-depth).
    # ------------------------------------------------------------------


def _component_check_auth(
    interaction, allowed_user_ids: Optional[set], allowed_role_ids: Optional[set],
) -> bool:
    """Shared user-or-role OR authorization for component button clicks.
    Allow on: DISCORD/GATEWAY_ALLOW_ALL_USERS, user in DISCORD/GATEWAY_ALLOWED_USERS, a role in the
    role allowlist, or pairing-store approval. Role allowlist with no ``roles`` (DM) rejects (fail closed).
    """
    user = getattr(interaction, "user", None)
    if user is None or getattr(user, "id", None) is None:
        return False
    # Scope-aware reads: interaction tasks inherit the owning profile's secret-scope contextvar;
    # under multiplex a raw os.getenv could return ANOTHER profile's allow-all flag.
    # Scope-aware reads (issue #72348): component interactions are dispatched from discord.py tasks
    # descended from the task created inside the owning profile's runtime scope, so the profile's
    # secret-scope contextvar is inherited here.
    if _scoped_gate_env("DISCORD_ALLOW_ALL_USERS").strip().lower() in {"true", "1", "yes"}:
        return True
    if _scoped_gate_env("GATEWAY_ALLOW_ALL_USERS").strip().lower() in {"true", "1", "yes"}:
        return True
    user_set = {str(uid).strip() for uid in (allowed_user_ids or set()) if str(uid).strip()}
    global_allowed = {
        uid.strip()
        for uid in _scoped_gate_env("GATEWAY_ALLOWED_USERS").split(",")
        if uid.strip()
    }
    user_set.update(global_allowed)
    role_set = set(allowed_role_ids or set())
    has_users = bool(user_set)
    has_roles = bool(role_set)
    try:
        uid = str(user.id)
    except AttributeError:
        uid = ""
    if has_users:
        if "*" in user_set or (uid and uid in user_set):
            return True
    if has_roles:
        roles_attr = getattr(user, "roles", None)
        if roles_attr is None:
            # Role policy configured but no role data (DM Member, raw User): fail closed.
            return False
        try:
            user_role_ids = {getattr(r, "id", None) for r in roles_attr}
        except TypeError:
            return False
        if user_role_ids & role_set:
            return True
    # Pairing store (mirrors ``authz_mixin._check_authorization``): paired users click without allowlist.
    if uid:
        try:
            from gateway.pairing import PairingStore
            store = PairingStore()
            if store.is_approved("discord", uid):
                return True
        except Exception:
            pass
    return False


def _resolve_exec_approval_admin_gate(config_extra: Optional[dict]) -> Tuple[bool, set]:
    """Resolve the exec-approval admin gate from ``extra``; returns ``(require_admin, admin_user_ids)``.
    Default OFF (user-scope buttons). When ``require_admin_for_exec_approval`` is true only
    ``allow_admin_from`` ids may click; on with no admins -> ``(True, set())`` (fail closed, log once).
    """
    extra = config_extra if isinstance(config_extra, dict) else {}
    raw_toggle = extra.get("require_admin_for_exec_approval", False)
    require_admin = str(raw_toggle).strip().lower() in {"true", "1", "yes"}
    if not require_admin:
        return (False, set())
    try:
        from gateway.slash_access import _coerce_id_list
        admin_ids = set(_coerce_id_list(extra.get("allow_admin_from")))
    except Exception:
        admin_ids = set()
    return (True, admin_ids)


def _define_discord_view_classes() -> None:
    """Register Discord UI view classes as module globals.
    Called at module load and after a lazy install so the classes exist whenever DISCORD_AVAILABLE."""
    global ExecApprovalView, SlashConfirmView, UpdatePromptView, ModelPickerView, ClarifyChoiceView, ChoicePickerView
    from .views.exec_approval import ExecApprovalView as _ExecApprovalView
    from .views.slash_confirm import SlashConfirmView as _SlashConfirmView
    from .views.update_prompt import UpdatePromptView as _UpdatePromptView
    from .views.model_picker import ModelPickerView as _ModelPickerView
    from .views.choice_picker import ChoicePickerView as _ChoicePickerView
    from .views.clarify_choice import ClarifyChoiceView as _ClarifyChoiceView
    ExecApprovalView = _ExecApprovalView
    SlashConfirmView = _SlashConfirmView
    UpdatePromptView = _UpdatePromptView
    ModelPickerView = _ModelPickerView
    ChoicePickerView = _ChoicePickerView
    ClarifyChoiceView = _ClarifyChoiceView


if DISCORD_AVAILABLE:
    _define_discord_view_classes()


# ── Standalone (out-of-process) sender ────────────────────────────────────────
# Used by ``tools/send_message_tool._send_via_adapter`` when no live DiscordAdapter is in this
# process (e.g. standalone ``hermes cron``); same forum/thread/multipart logic via Discord REST.

# Process-local channel-type probe cache: avoids re-probing every send when the directory cache misses.
_DISCORD_CHANNEL_TYPE_PROBE_CACHE: Dict[str, bool] = {}
_DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES = 1 * 1024 * 1024
_DISCORD_STANDALONE_ERROR_BODY_LIMIT_BYTES = 8 * 1024


def _remember_channel_is_forum(chat_id: str, is_forum: bool) -> None:
    _DISCORD_CHANNEL_TYPE_PROBE_CACHE[str(chat_id)] = bool(is_forum)


def _probe_is_forum_cached(chat_id: str) -> Optional[bool]:
    return _DISCORD_CHANNEL_TYPE_PROBE_CACHE.get(str(chat_id))


def _derive_forum_thread_name(message: str) -> str:
    """Derive a thread name from the first line of the message, capped at 100 chars."""
    first_line = message.strip().split("\n", 1)[0].strip()
    first_line = first_line.lstrip("#").strip()
    if not first_line:
        first_line = "New Post"
    return first_line[:100]


# Plugin setup is kept separate from the runtime adapter; import these aliases only
# after the adapter definitions are complete so setup helpers can resolve lazily.
from .setup import (  # noqa: E402
    _clean_discord_user_ids, interactive_setup, _apply_yaml_config,
    _is_connected, _build_adapter, register,
)




# ── Plugin entry point ────────────────────────────────────────────────────────




# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'env_int': ('utils', 'env_int'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
