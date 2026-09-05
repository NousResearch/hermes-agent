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
    ("new", "Start a new conversation", (), "/reset", "New conversation started~"),
    ("reset", "Reset your Hermes session", (), "/reset", "Session reset~"),
    ("model", "Show or change the model",
     (("name", str, "", "Model name (e.g. anthropic/claude-sonnet-4). Leave empty to see current.", None),),
     "/model {name}", None),
    ("reasoning", "Show/change reasoning effort, or toggle showing it",
     (("effort", str, "", "Pick a level, reset the override, or show/hide reasoning. Leave empty to see current.",
       # One `/reasoning <arg>` handler; Discord has no free-text subcommand, so list every value.
       (("none — disable reasoning", "none"), ("minimal", "minimal"), ("low", "low"),
        ("medium", "medium"), ("high", "high"), ("xhigh", "xhigh"), ("max", "max"),
        ("ultra — maximum reasoning", "ultra"), ("reset — clear this session's override", "reset"),
        ("show — reveal reasoning in replies", "show"), ("hide — hide reasoning from replies", "hide"))),),
     "/reasoning {effort}", None),
    ("personality", "Set a personality",
     (("name", str, "", "Personality name. Leave empty to list available.", None),),
     "/personality {name}", None),
    ("retry", "Retry your last message", (), "/retry", "Retrying~"),
    ("undo", "Remove the last exchange", (), "/undo", None),
    ("status", "Show Hermes session status", (), "/status", "Status sent~"),
    ("sethome", "Set this chat as the home channel", (), "/sethome", None),
    ("stop", "Stop the running Hermes agent", (), "/stop", "Stop requested~"),
    ("steer", "Inject a message after the next tool call (no interrupt)",
     (("prompt", str, _REQUIRED, "Text to inject into the agent's next tool result", None),),
     "/steer {prompt}", None),
    ("plan", "Write a markdown implementation plan (no execution)",
     (("task", str, "", "What to plan. Leave empty to infer from the conversation.", None),),
     "/plan {task}", None),
    ("compress", "Compress conversation context", (), "/compress", None),
    ("title", "Set or show the session title",
     (("name", str, "", "Session title. Leave empty to show current.", None),),
     "/title {name}", None),
    ("resume", "Resume a previously-named session",
     (("name", str, "", "Session name to resume. Leave empty to list sessions.", None),),
     "/resume {name}", None),
    ("usage", "Show token usage for this session", (), "/usage", None),
    ("help", "Show available commands", (), "/help", None),
    ("insights", "Show usage insights and analytics",
     (("days", int, 7, "Number of days to analyze (default: 7)", None),),
     "/insights {days}", None),
    ("reload-mcp", "Reload MCP servers from config", (), "/reload-mcp", None),
    ("reload-skills", "Re-scan ~/.hermes/skills/ for new or removed skills", (), "/reload-skills", None),
    ("voice", "Toggle voice reply mode",
     (("mode", str, "", "Voice mode: join, channel, leave, on, tts, off, or status",
       # `join` and `channel` both hit _handle_voice_channel_join; expose both to match docs.
       (("join — join your voice channel", "join"), ("channel — join your voice channel (alias)", "channel"),
        ("leave — leave voice channel", "leave"), ("on — voice reply to voice messages", "on"),
        ("tts — voice reply to all messages", "tts"), ("off — text only", "off"),
        ("status — show current mode", "status"))),),
     "/voice {mode}", None),
    ("update", "Update Hermes Agent to the latest version", (), "/update", "Update initiated~"),
    ("restart", "Gracefully restart the Hermes gateway", (), "/restart", "Restart requested~"),
    ("approve", "Approve a pending dangerous command",
     (("scope", str, "", "Optional: 'all', 'session', 'always', 'all session', 'all always'", None),),
     "/approve {scope}", None),
    ("deny", "Deny a pending dangerous command",
     (("scope", str, "", "Optional: 'all' to deny all pending commands", None),),
     "/deny {scope}", None),
    # /thread: template None -> registered by _register_thread_slash (auth-gated defer).
    ("thread", "Create a new thread and start a Hermes session in it", (), None, None),
    ("queue", "Queue a prompt for the next turn (doesn't interrupt)",
     (("prompt", str, _REQUIRED, "The prompt to queue", None),),
     "/queue {prompt}", "Queued for the next turn."),
    ("bg", "Run a prompt in a separate background session",
     (("prompt", str, _REQUIRED, "The prompt to run in the background", None),),
     "/bg {prompt}", "Background task started~"),
    ("btw", "Ask a side question about the current conversation",
     (("question", str, _REQUIRED, "The side question to answer without interrupting", None),),
     "/btw {question}", "Side question dispatched~"),
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

from gateway.config import Platform, PlatformConfig

from gateway.platforms.helpers import (
    MessageDeduplicator, ThreadParticipationTracker, convert_table_to_bullets,
)
from utils import atomic_json_write, env_float
from gateway.platforms.base import (
    BasePlatformAdapter, MessageEvent, MessageType, ProcessingOutcome, SendResult,
    cache_image_from_url, cache_image_from_bytes_async, cache_audio_from_url, cache_audio_from_bytes_async,
    cache_document_from_bytes_async, SUPPORTED_DOCUMENT_TYPES, _TEXT_INJECT_EXTENSIONS,
    _prefix_within_utf16_limit, utf16_len, validate_inbound_media_size,
)
from tools.url_safety import is_safe_url
from gateway.platforms._shared import profile_scoped as _profile_scoped_config_load


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


def _build_allowed_mentions():
    """Build Discord ``AllowedMentions`` denying @everyone/@here/roles by default (any LLM output
    with ``@everyone`` would otherwise ping the server); user / replied-user pings stay on.

    Override via env (or ``discord.allow_mentions.*`` in config.yaml):

        DISCORD_ALLOW_MENTION_EVERYONE      default false  — @everyone + @here
        DISCORD_ALLOW_MENTION_ROLES         default false  — @role pings
        DISCORD_ALLOW_MENTION_USERS         default true   — @user pings
        DISCORD_ALLOW_MENTION_REPLIED_USER  default true   — reply-ping author
    """
    if not DISCORD_AVAILABLE:
        return None
    _b = _env_bool
    return discord.AllowedMentions(
        everyone=_b("DISCORD_ALLOW_MENTION_EVERYONE", False),
        roles=_b("DISCORD_ALLOW_MENTION_ROLES", False),
        users=_b("DISCORD_ALLOW_MENTION_USERS", True),
        replied_user=_b("DISCORD_ALLOW_MENTION_REPLIED_USER", True),
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


class VoiceReceiver:
    """Captures voice audio from a Discord voice channel: hooks the VoiceClient socket, decrypts
    RTP (NaCl + DAVE E2EE), decodes Opus per user; a polling loop delivers utterances on silence."""

    SILENCE_THRESHOLD = 1.5    # seconds of silence → end of utterance
    MIN_SPEECH_DURATION = 0.5  # minimum seconds to process (skip noise)
    SAMPLE_RATE = 48000        # Discord native rate
    CHANNELS = 2               # Discord sends stereo

    def __init__(self, voice_client, allowed_user_ids: set = None):
        self._vc = voice_client
        self._allowed_user_ids = allowed_user_ids or set()
        self._running = False
        self._secret_key: Optional[bytes] = None
        self._dave_session = None
        self._bot_ssrc: int = 0
        self._ssrc_to_user: Dict[int, int] = {}
        self._lock = threading.Lock()
        self._buffers: Dict[int, bytearray] = defaultdict(bytearray)
        self._last_packet_time: Dict[int, float] = {}
        # Opus decoder per SSRC (each user needs own decoder state)
        self._decoders: Dict[int, object] = {}
        # Pause flag: don't capture while bot is playing TTS
        self._paused = False
        # Debug logging counter (instance-level to avoid cross-instance races)
        self._packet_debug_count = 0

    # --- Lifecycle ---

    def start(self):
        """Start listening for voice packets."""
        conn = self._vc._connection
        self._secret_key = bytes(conn.secret_key)
        self._dave_session = conn.dave_session
        self._bot_ssrc = conn.ssrc
        self._install_speaking_hook(conn)
        conn.add_socket_listener(self._on_packet)
        self._running = True
        logger.info("VoiceReceiver started (bot_ssrc=%d)", self._bot_ssrc)

    def stop(self):
        """Stop listening and clean up."""
        self._running = False
        try:
            self._vc._connection.remove_socket_listener(self._on_packet)
        except Exception:
            pass
        with self._lock:
            self._buffers.clear()
            self._last_packet_time.clear()
            self._decoders.clear()
            self._ssrc_to_user.clear()
        logger.info("VoiceReceiver stopped")

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    # --- SSRC -> user_id mapping via SPEAKING opcode hook ---

    def map_ssrc(self, ssrc: int, user_id: int):
        with self._lock:
            self._ssrc_to_user[ssrc] = user_id

    def _install_speaking_hook(self, conn):
        """Wrap the voice websocket hook to capture SPEAKING events (op 5); ``conn.hook`` is
        re-passed on each (re)connect, so wrap it on the state AND the live websocket."""
        original_hook = conn.hook
        receiver_self = self

        async def wrapped_hook(ws, msg):
            if isinstance(msg, dict) and msg.get("op") == 5:
                data = msg.get("d", {})
                ssrc = data.get("ssrc")
                user_id = data.get("user_id")
                if ssrc and user_id:
                    logger.info("SPEAKING event: ssrc=%d -> user=%s", ssrc, user_id)
                    receiver_self.map_ssrc(int(ssrc), int(user_id))
            if original_hook:
                await original_hook(ws, msg)
        conn.hook = wrapped_hook
        try:
            from discord.utils import MISSING
            if hasattr(conn, 'ws') and conn.ws is not MISSING:
                conn.ws._hook = wrapped_hook
                logger.info("Speaking hook installed on live websocket")
        except Exception as e:
            logger.warning("Could not install hook on live ws: %s", e)

    # --- Packet handler (called from SocketReader thread) ---

    def _on_packet(self, data: bytes):
        if not self._running or self._paused:
            return
        self._packet_debug_count += 1
        if self._packet_debug_count <= 5:
            logger.debug(
                "Raw UDP packet: len=%d, first_bytes=%s",
                len(data), data[:4].hex() if len(data) >= 4 else "short",
            )
        if len(data) < 16:
            return
        # RTP v2: top 2 bits 10 (rest varies); voice payload type (byte 1 & 0x7F) is 0x78.
        if (data[0] >> 6) != 2 or (data[1] & 0x7F) != 0x78:
            if self._packet_debug_count <= 5:
                logger.debug("Skipped non-RTP: byte0=0x%02x byte1=0x%02x", data[0], data[1])
            return
        first_byte = data[0]
        _, _, seq, timestamp, ssrc = struct.unpack_from(">BBHII", data, 0)
        if ssrc == self._bot_ssrc:
            return
        # Calculate dynamic RTP header size (RFC 9335 / rtpsize mode)
        cc = first_byte & 0x0F  # CSRC count
        has_extension = bool(first_byte & 0x10)  # extension bit
        has_padding = bool(first_byte & 0x20)  # padding bit (RFC 3550 §5.1)
        header_size = 12 + (4 * cc) + (4 if has_extension else 0)
        if len(data) < header_size + 4:  # need at least header + nonce
            return
        # Read extension length from preamble (for skipping after decrypt)
        ext_data_len = 0
        if has_extension:
            ext_preamble_offset = 12 + (4 * cc)
            ext_words = struct.unpack_from(">H", data, ext_preamble_offset + 2)[0]
            ext_data_len = ext_words * 4
        if self._packet_debug_count <= 10:
            with self._lock:
                known_user = self._ssrc_to_user.get(ssrc, "unknown")
            logger.debug(
                "RTP packet: ssrc=%d, seq=%d, user=%s, hdr=%d, ext_data=%d",
                ssrc, seq, known_user, header_size, ext_data_len,
            )
        header = bytes(data[:header_size])
        payload_with_nonce = data[header_size:]
        # --- NaCl transport decrypt (aead_xchacha20_poly1305_rtpsize) ---
        if len(payload_with_nonce) < 4:
            return
        nonce = bytearray(24)
        nonce[:4] = payload_with_nonce[-4:]
        encrypted = bytes(payload_with_nonce[:-4])
        try:
            import nacl.secret  # noqa: E402 — delayed import, only in voice path
            box = nacl.secret.Aead(self._secret_key)
            decrypted = box.decrypt(encrypted, header, bytes(nonce))
        except Exception as e:
            if self._packet_debug_count <= 10:
                logger.warning("NaCl decrypt failed: %s (hdr=%d, enc=%d)", e, header_size, len(encrypted))
            return
        # Skip encrypted extension data to get the actual opus payload
        if ext_data_len and len(decrypted) > ext_data_len:
            decrypted = decrypted[ext_data_len:]
        # Strip RTP padding (RFC 3550 §5.1): last payload byte is the count; leaving it corrupts DAVE/Opus.
        if has_padding:
            if not decrypted:
                if self._packet_debug_count <= 10:
                    logger.warning("RTP padding bit set but no payload (ssrc=%d)", ssrc)
                return
            pad_len = decrypted[-1]
            if pad_len == 0 or pad_len > len(decrypted):
                if self._packet_debug_count <= 10:
                    logger.warning(
                        "Invalid RTP padding length %d for payload size %d (ssrc=%d)",
                        pad_len, len(decrypted), ssrc,
                    )
                return
            decrypted = decrypted[:-pad_len]
            if not decrypted:
                return
        # --- DAVE E2EE decrypt ---
        if self._dave_session:
            with self._lock:
                user_id = self._ssrc_to_user.get(ssrc, 0)
            if user_id:
                try:
                    import davey
                    decrypted = self._dave_session.decrypt(
                        user_id, davey.MediaType.audio, decrypted
                    )
                except Exception as e:
                    # Unencrypted passthrough — use NaCl-decrypted data as-is
                    if "Unencrypted" not in str(e):
                        if self._packet_debug_count <= 10:
                            logger.warning("DAVE decrypt failed for ssrc=%d: %s", ssrc, e)
                        return
            # Unknown SSRC (no SPEAKING yet): skip DAVE, try Opus directly; user_id arrives with SPEAKING.
        try:
            if ssrc not in self._decoders:
                self._decoders[ssrc] = discord.opus.Decoder()
            pcm = self._decoders[ssrc].decode(decrypted)
            with self._lock:
                self._buffers[ssrc].extend(pcm)
                self._last_packet_time[ssrc] = time.monotonic()
        except Exception as e:
            with self._lock:
                self._decoders.pop(ssrc, None)
            logger.debug("Opus decode error for SSRC %s; reset decoder: %s", ssrc, e)
            return

    # --- Silence detection ---

    def _infer_user_for_ssrc(self, ssrc: int) -> int:
        """Infer user_id for an unmapped SSRC: after a bot rejoin Discord may not resend
        SPEAKING, so if exactly one allowed user is in the channel, map the SSRC to them."""
        try:
            channel = self._vc.channel
            if not channel:
                return 0
            bot_id = self._vc.user.id if self._vc.user else 0
            allowed = self._allowed_user_ids
            candidates = [
                m.id for m in channel.members
                if m.id != bot_id and (not allowed or str(m.id) in allowed)
            ]
            if len(candidates) == 1:
                uid = candidates[0]
                self._ssrc_to_user[ssrc] = uid
                logger.info("Auto-mapped ssrc=%d -> user=%d (sole allowed member)", ssrc, uid)
                return uid
        except Exception:
            pass
        return 0

    def check_silence(self) -> list:
        """Return list of (user_id, pcm_bytes) for completed utterances."""
        now = time.monotonic()
        completed = []
        with self._lock:
            ssrc_user_map = dict(self._ssrc_to_user)
            ssrc_list = list(self._buffers.keys())
            for ssrc in ssrc_list:
                last_time = self._last_packet_time.get(ssrc, now)
                silence_duration = now - last_time
                buf = self._buffers[ssrc]
                # 48kHz, 16-bit, stereo = 192000 bytes/sec
                buf_duration = len(buf) / (self.SAMPLE_RATE * self.CHANNELS * 2)
                if silence_duration >= self.SILENCE_THRESHOLD and buf_duration >= self.MIN_SPEECH_DURATION:
                    user_id = ssrc_user_map.get(ssrc, 0)
                    if not user_id:
                        # SSRC unmapped (SPEAKING missing after rejoin) — infer from channel.
                        user_id = self._infer_user_for_ssrc(ssrc)
                    if user_id:
                        completed.append((user_id, bytes(buf)))
                    self._buffers[ssrc] = bytearray()
                    self._last_packet_time.pop(ssrc, None)
                elif silence_duration >= self.SILENCE_THRESHOLD * 2:
                    # Stale buffer with no valid user — discard
                    self._buffers.pop(ssrc, None)
                    self._last_packet_time.pop(ssrc, None)
        return completed

    def flush_pending(self) -> list:
        """Return buffered utterances that have not yet reached silence."""
        completed = []
        with self._lock:
            ssrc_user_map = dict(self._ssrc_to_user)
            for ssrc, buf in list(self._buffers.items()):
                # 48kHz, 16-bit, stereo = 192000 bytes/sec
                buf_duration = len(buf) / (self.SAMPLE_RATE * self.CHANNELS * 2)
                if buf_duration >= self.MIN_SPEECH_DURATION:
                    user_id = ssrc_user_map.get(ssrc, 0)
                    if not user_id:
                        user_id = self._infer_user_for_ssrc(ssrc)
                    if user_id:
                        completed.append((user_id, bytes(buf)))
                self._buffers.pop(ssrc, None)
                self._last_packet_time.pop(ssrc, None)
        return completed

    # --- PCM -> WAV conversion (for Whisper STT) ---

    @staticmethod
    def pcm_to_wav(pcm_data: bytes, output_path: str, src_rate: int = 48000, src_channels: int = 2):
        """Convert raw PCM to 16kHz mono WAV via ffmpeg into *output_path* (not stdout: ffmpeg
        can't seek a pipe, so piped WAV carries placeholder RIFF sizes strict readers misreport)."""
        from hermes_cli._subprocess_compat import windows_hide_flags
        subprocess.run(
            [
                resolve_ffmpeg_executable(), "-y", "-loglevel", "error", "-f", "s16le",
                "-ar", str(src_rate), "-ac", str(src_channels), "-i", "pipe:0", "-ar", "16000",
                "-ac", "1", output_path,
            ],
            input=pcm_data,
            check=True,
            timeout=10,
            # Capture stderr so a failure's CalledProcessError carries ffmpeg's real message.
            stderr=subprocess.PIPE,
            creationflags=windows_hide_flags(),
        )


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
    raw = os.getenv(name, "").strip().lower()
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


from .adapter_lifecycle import DiscordLifecycleMixin
from .adapter_recovery import DiscordRecoveryMixin
from .adapter_commands import DiscordCommandsMixin
from .adapter_delivery import DiscordDeliveryMixin
from .adapter_voice import DiscordVoiceMixin
from .adapter_routing import DiscordRoutingMixin
from .adapter_prompts import DiscordPromptsMixin
from .adapter_inbound import DiscordInboundMixin


class DiscordAdapter(
    DiscordLifecycleMixin, DiscordRecoveryMixin, DiscordCommandsMixin, DiscordDeliveryMixin, DiscordVoiceMixin, DiscordRoutingMixin, DiscordPromptsMixin, DiscordInboundMixin,
    BasePlatformAdapter,
):
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

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.DISCORD)
        self._client: Optional[commands.Bot] = None
        self._ready_event = asyncio.Event()
        self._allowed_user_ids: set = set()  # For button approval authorization
        self._allowed_role_ids: set = set()  # For DISCORD_ALLOWED_ROLES filtering
        # Gate env snapshot captured in connect() inside the owning profile's scope; None until then.
        # None until then; accessors fall back to live scope-aware reads (issue #72348).
        self._gate_env_snapshot: Optional[Dict[str, str]] = None
        self.gateway_runner = None  # Set by gateway/run.py for cross-platform delivery
        self._voice_clients: Dict[int, Any] = {}  # guild_id -> VoiceClient
        self._voice_locks: Dict[int, asyncio.Lock] = {}  # guild_id -> serialize join/leave
        # Text batching: merge rapid successive messages (Telegram-style)
        self._text_batch_delay_seconds = env_float("HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS", 0.6)
        self._text_batch_split_delay_seconds = env_float("HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS", 2.0)
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}
        self._voice_text_channels: Dict[int, int] = {}  # guild_id -> text_channel_id
        self._voice_sources: Dict[int, Dict[str, Any]] = {}  # guild_id -> linked text channel source metadata
        self._voice_timeout_tasks: Dict[int, asyncio.Task] = {}  # guild_id -> timeout task
        self._voice_timeout_seconds = self._load_voice_timeout()
        self._playback_timeout_seconds = self._load_playback_timeout()
        self._voice_receivers: Dict[int, VoiceReceiver] = {}  # guild_id -> VoiceReceiver
        self._voice_listen_tasks: Dict[int, asyncio.Task] = {}  # guild_id -> listen loop
        self._voice_input_callback: Optional[Callable] = None  # set by run.py
        self._on_voice_disconnect: Optional[Callable] = None  # set by run.py
        # Voice-reply mode ("off"|"voice_only"|"all") per linked text-channel id (set by run.py) so
        # the inactivity timer keeps the bot in channel for /voice off, unlike /voice leave.
        self._voice_mode_getter: Optional[Callable] = None  # set by run.py
        # Continuous voice mixer per guild (ambient bed + ducked speech) so acks/TTS/thinking overlap.
        self._voice_mixers: Dict[int, Any] = {}  # guild_id -> VoiceMixer
        self._ambient_pcm_cache: Optional[bytes] = None  # decoded ambient bed
        self._voice_fx_cfg: Dict[str, Any] = self._load_voice_fx_config()
        # Threads the bot participated in (no @mention needed there); persisted across restarts.
        self._threads = ThreadParticipationTracker("discord")
        # Persistent typing loops per channel (DMs don't reliably show bot typing events).
        self._typing_tasks: Dict[str, asyncio.Task] = {}
        self._bot_task: Optional[asyncio.Task] = None
        # Background task that runs post-connect housekeeping (command-menu registration + DM-topic setup)
        # off the connect path so a slow Bot API call (e.g. a set_my_commands stall for certain tokens)
        # cannot blow the gateway's connect timeout (#46298).
        self._post_connect_task: Optional[asyncio.Task] = None
        # WS liveness probe: REST 200 can't prove Gateway events still arrive, so sample WS
        # ready/open/ACK + heartbeat latency; consecutive failures -> retryable-fatal. 0 disables.
        self._liveness_interval_seconds = self._finite_positive_config_float(
            "websocket_liveness_interval_seconds", 15.0,
            env_key="HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS",
        )
        self._liveness_failure_threshold = self._config_int(
            "websocket_liveness_failure_threshold", 2,
            env_key="HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD",
        )
        self._heartbeat_ack_max_age_seconds = self._finite_positive_config_float(
            "websocket_heartbeat_ack_max_age_seconds", 60.0,
        )
        self._max_latency_seconds = self._finite_positive_config_float(
            "websocket_max_latency_seconds", 30.0,
        )
        self._liveness_task: Optional[asyncio.Task] = None
        self._liveness_notification_task: Optional[asyncio.Task] = None
        # True while disconnect() intentionally closes discord.py (done callback: shutdown vs crash).
        self._disconnecting = False
        self._missed_message_backfill_task: Optional[asyncio.Task] = None
        from hermes_constants import get_hermes_home
        from plugins.platforms.discord.recovery import DiscordRecoveryStore
        self._discord_recovery_store = DiscordRecoveryStore(get_hermes_home())
        # Dedup cache: Discord RESUME replays events after reconnects.
        self._dedup = MessageDeduplicator()
        # Reply threading mode: "off", "first" (default; first chunk only), "all" (every chunk).
        self._reply_to_mode: str = getattr(config, 'reply_to_mode', 'first') or 'first'
        self._slash_commands: bool = self.config.extra.get("slash_commands", True)
        # Bot's last message ID per channel: lets history backfill skip the full channel.history() scan.
        self._last_self_message_id: Dict[str, str] = {}
        # Bot-authored lifecycle/status message IDs that must not bound history after restart.
        self._nonconversational_messages = _DiscordNonConversationalMessageTracker()
        # Last truncated mid-stream preview per (chat_id, message_id): past the 2000 cap every edit
        # truncates to the SAME text, and re-sending only burns edit rate limit. Dropped on finalize.
        # Once an oversized streaming edit saturates at the 2000-char preview cap, every subsequent
        # progressive edit truncates to the SAME text; re-sending it is a no-op that still counts against
        # Discord's edit rate limit (~1 edit per stream tick for the rest of a long reply). Mirrors the
        # Telegram #58563 fix.
        self._last_overflow_preview: Dict[tuple, str] = {}
        self._warned_fail_closed_default = False

    # --- gateway_platform_event fire-sites ---

    # --- Voice channel methods (join / leave / play) ---

    # --- Voice listening (Phase 2) ---

    # UDP keepalive interval; Discord drops the UDP route after ~60s of silence.
    _KEEPALIVE_INTERVAL = 15

    # ── Slash command authorization ─────────────────────────────────────
    # ``_check_slash_authorization`` mirrors the on_message gates one-for-one. No allowlist =>
    # fail closed unless allow-all; DISCORD_ALLOWED_CHANNELS alone authorizes per validated channel.

    # --- Thread creation helpers ---

    # ── per-adapter authorization gates ──────────────────────────────────
    # Under multiplex_profiles os.environ is process-global (first-writer-wins), so raw os.getenv
    # would leak profile A into B. Order: connect()-time env snapshot, config.extra, scoped env read.

    # ------------------------------------------------------------------
    # Auto-thread helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Attachment download helpers
    # Prefer the authenticated bot session (``att.read()``): CDN URLs increasingly 403 without
    # bot auth and some VPN DNS setups make ``is_safe_url`` flag the CDN as SSRF. If ``read()``
    # is missing or fails, fall back to the SSRF-gated URL downloaders (defense-in-depth).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Text message aggregation (handles Discord client-side splits)
    # ------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Discord UI Components (outside the adapter class)
# ---------------------------------------------------------------------------


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
    """Bind the SDK views after initial import or lazy dependency installation."""
    from .adapter_views import define_discord_view_classes

    global ExecApprovalView, SlashConfirmView, UpdatePromptView, ModelPickerView, ClarifyChoiceView, ChoicePickerView
    (ExecApprovalView, SlashConfirmView, UpdatePromptView, ModelPickerView,
     ClarifyChoiceView, ChoicePickerView) = define_discord_view_classes()

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


def _standalone_sanitize_error(text) -> str:
    """Local copy of tools.send_message_tool._sanitize_error_text (strips bot tokens); avoids hard dep."""
    s = str(text)
    import re as _re_san
    return _re_san.sub(r"(Authorization:\s*Bot\s+)\S+", r"\1***", s, flags=_re_san.IGNORECASE)


def _standalone_close_response(resp: Any) -> None:
    close = getattr(resp, "close", None)
    if callable(close):
        close()
        return
    release = getattr(resp, "release", None)
    if callable(release):
        release()


async def _standalone_read_response_bytes_limited(
    resp: Any, limit_bytes: int,
) -> Tuple[Optional[bytes], bool]:
    """Read at most *limit_bytes*; returns ``(body, truncated)``. ``(None, False)`` when the object
    has no streaming ``content.read`` coroutine (proxy/test double) — callers use ``json()``/``text()``."""
    content = getattr(resp, "content", None)
    read = getattr(content, "read", None)
    if content is None or not inspect.iscoroutinefunction(read):
        return None, False
    try:
        chunks: list[bytes] = []
        total = 0
        while total <= limit_bytes:
            chunk = await read(limit_bytes + 1 - total)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", "replace")
            total += len(chunk)
            chunks.append(chunk)
            if total > limit_bytes:
                _standalone_close_response(resp)
                return b"".join(chunks)[:limit_bytes], True
        return b"".join(chunks), False
    except (TypeError, AttributeError):
        # Quacked like a stream but wasn't — caller uses native json()/text().
        return None, False


def _standalone_response_encoding(resp: Any) -> str:
    get_encoding = getattr(resp, "get_encoding", None)
    if callable(get_encoding):
        try:
            return get_encoding() or "utf-8"
        except Exception:
            return "utf-8"
    return "utf-8"


async def _standalone_read_text_limited(resp: Any, limit_bytes: int) -> str:
    body, _truncated = await _standalone_read_response_bytes_limited(resp, limit_bytes)
    if body is None:
        return await resp.text()
    return body.decode(_standalone_response_encoding(resp), "replace")


async def _standalone_read_json_limited(resp: Any, limit_bytes: int) -> dict:
    body, truncated = await _standalone_read_response_bytes_limited(resp, limit_bytes)
    if body is None:
        return await resp.json()
    if truncated:
        raise ValueError(f"Discord API JSON response exceeds {limit_bytes} bytes")
    if not body:
        return {}
    data = json.loads(body.decode(_standalone_response_encoding(resp), "replace"))
    return data if isinstance(data, dict) else {}


def _standalone_warn_missing_media(media_path: str) -> str:
    warning = f"Media file not found, skipping: {media_path}"
    logger.warning(warning)
    return warning


async def _standalone_response_json_or_error(resp: Any, error_prefix: str):
    """``(data, None)`` for a 200/201 JSON response, else ``(None, {"error": ...})``
    with the (size-capped) body text appended to ``error_prefix``."""
    if resp.status not in {200, 201}:
        body = await _standalone_read_text_limited(resp, _DISCORD_STANDALONE_ERROR_BODY_LIMIT_BYTES)
        return None, {"error": f"{error_prefix} ({resp.status}): {body}"}
    return await _standalone_read_json_limited(resp, _DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES), None


async def _standalone_is_forum(aiohttp, chat_id: str, json_headers: dict, sess_kw: dict, req_kw: dict) -> bool:
    """Forum detection: channel directory → process-local probe cache → memoized ``GET /channels/{id}``."""
    _channel_type = None
    try:
        from gateway.channel_directory import lookup_channel_type
        _channel_type = lookup_channel_type("discord", chat_id)
    except Exception:
        pass
    if _channel_type is not None:
        return _channel_type == "forum"
    cached = _probe_is_forum_cached(chat_id)
    if cached is not None:
        return cached
    is_forum = False
    try:
        info_url = f"https://discord.com/api/v10/channels/{chat_id}"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15), **sess_kw) as info_sess:
            async with info_sess.get(info_url, headers=json_headers, **req_kw) as info_resp:
                if info_resp.status == 200:
                    info = await _standalone_read_json_limited(info_resp, _DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES)
                    is_forum = info.get("type") == 15
                    _remember_channel_is_forum(chat_id, is_forum)
    except Exception:
        logger.debug("Failed to probe channel type for %s", chat_id, exc_info=True)
    return is_forum


async def _standalone_send(
    pconfig, chat_id: str, message: str, *, thread_id: Optional[str] = None,
    media_files: Optional[list] = None, force_document: bool = False, caption: Optional[str] = None,
) -> Dict[str, Any]:
    """Send via Discord REST without a live gateway adapter (token: ``pconfig.token`` then env var).
    Forum channels (type 15) reject ``POST /messages``, so a thread post is created via
    ``POST /channels/{id}/threads`` with media as multipart attachments. Channel type: directory
    cache → process-local probe cache → memoized GET. ``force_document`` accepted but unused."""
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    token = (getattr(pconfig, "token", None) or "").strip()
    if not token:
        # Profile-scoped read: under multiplex the env may hold another profile's token.
        from agent.secret_scope import get_secret
        token = (get_secret("DISCORD_BOT_TOKEN", "") or "").strip()
    if not token:
        return {"error": "Discord standalone send: DISCORD_BOT_TOKEN is not set"}
    try:
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
        _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
        auth_headers = {"Authorization": f"Bot {token}"}
        json_headers = {**auth_headers, "Content-Type": "application/json"}
        media_files = media_files or []
        last_data = None
        warnings = []
        if thread_id:
            url = f"https://discord.com/api/v10/channels/{thread_id}/messages"
        else:
            # Forum channels (type 15) reject POST /messages — create a thread post.
            if await _standalone_is_forum(aiohttp, chat_id, json_headers, _sess_kw, _req_kw):
                thread_name = _derive_forum_thread_name(message)
                thread_url = f"https://discord.com/api/v10/channels/{chat_id}/threads"
                # Filter readable media first to pick JSON vs multipart before opening a session.
                valid_media = []
                for media_path, _is_voice in media_files:
                    if not os.path.exists(media_path):
                        warnings.append(_standalone_warn_missing_media(media_path))
                        continue
                    valid_media.append(media_path)
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60), **_sess_kw) as session:
                    if valid_media:
                        # Multipart payload_json + files[N]: thread + starter + attachments in one call.
                        attachments_meta = [
                            {"id": str(idx), "filename": os.path.basename(path)}
                            for idx, path in enumerate(valid_media)
                        ]
                        starter_message = {"content": (caption or message), "attachments": attachments_meta}
                        payload_json = json.dumps({"name": thread_name, "message": starter_message})
                        form = aiohttp.FormData()
                        form.add_field("payload_json", payload_json, content_type="application/json")
                        try:
                            for idx, media_path in enumerate(valid_media):
                                with open(media_path, "rb") as fh:
                                    form.add_field(
                                        f"files[{idx}]", fh.read(),
                                        filename=os.path.basename(media_path),
                                    )
                            async with session.post(thread_url, headers=auth_headers, data=form, **_req_kw) as resp:
                                data, err = await _standalone_response_json_or_error(resp, "Discord forum thread creation error")
                                if err:
                                    return err
                        except Exception as e:
                            return {"error": _standalone_sanitize_error(f"Discord forum thread upload failed: {e}")}
                    else:
                        # No media: JSON POST creates the thread with the text starter.
                        async with session.post(
                            thread_url, headers=json_headers,
                            json={"name": thread_name, "message": {"content": message}}, **_req_kw,
                        ) as resp:
                            data, err = await _standalone_response_json_or_error(resp, "Discord forum thread creation error")
                            if err:
                                return err
                thread_id_created = data.get("id")
                starter_msg_id = (data.get("message") or {}).get("id", thread_id_created)
                result = {
                    "success": True, "platform": "discord", "chat_id": chat_id,
                    "thread_id": thread_id_created, "message_id": starter_msg_id,
                }
                if warnings:
                    result["warnings"] = warnings
                return result
            url = f"https://discord.com/api/v10/channels/{chat_id}/messages"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), **_sess_kw) as session:
            if message.strip() or not media_files:
                async with session.post(url, headers=json_headers, json={"content": message}, **_req_kw) as resp:
                    last_data, err = await _standalone_response_json_or_error(resp, "Discord API error")
                    if err:
                        return err
            # One multipart upload per file; a MEDIA:<path> caption rides as the attachment message's
            # content, and caption_pending makes a missing file fall back to a plain message.
            caption_pending = bool(caption)
            for media_path, _is_voice in media_files:
                if not os.path.exists(media_path):
                    warnings.append(_standalone_warn_missing_media(media_path))
                    if caption_pending:
                        try:
                            async with session.post(
                                url, headers=json_headers, json={"content": caption}, **_req_kw,
                            ) as resp:
                                if resp.status in {200, 201}:
                                    last_data = await _standalone_read_json_limited(
                                        resp, _DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES,
                                    )
                                    caption_pending = False
                        except Exception:
                            logger.warning("Discord caption-fallback send failed for missing media")
                    continue
                try:
                    form = aiohttp.FormData()
                    filename = os.path.basename(media_path)
                    if caption_pending:
                        form.add_field(
                            "payload_json", json.dumps({"content": caption}),
                            content_type="application/json",
                        )
                        caption_pending = False
                    with open(media_path, "rb") as f:
                        form.add_field("files[0]", f, filename=filename)
                        async with session.post(url, headers=auth_headers, data=form, **_req_kw) as resp:
                            data, err = await _standalone_response_json_or_error(resp, "Discord API error")
                            if err:
                                warning = _standalone_sanitize_error(f"Failed to send media {media_path}: {err['error']}")
                                logger.error(warning)
                                warnings.append(warning)
                                continue
                            last_data = data
                except Exception as e:
                    warning = _standalone_sanitize_error(f"Failed to send media {media_path}: {e}")
                    logger.error(warning)
                    warnings.append(warning)
        if last_data is None:
            error = "No deliverable text or media remained after processing"
            if warnings:
                return {"error": error, "warnings": warnings}
            return {"error": error}
        result = {"success": True, "platform": "discord", "chat_id": chat_id, "message_id": last_data.get("id")}
        if warnings:
            result["warnings"] = warnings
        return result
    except Exception as e:
        # Include the exception type: str(TimeoutError()) is empty.
        logger.error("Discord standalone send failed", exc_info=True)
        return {"error": _standalone_sanitize_error(f"Discord send failed: {type(e).__name__}: {e}")}


# ── Plugin entry point ────────────────────────────────────────────────────────


def _clean_discord_user_ids(raw: str) -> list:
    """Strip common Discord mention prefixes from a comma-separated ID string."""
    cleaned = []
    for uid in raw.replace(" ", "").split(","):
        uid = uid.strip()
        if uid.startswith("<@") and uid.endswith(">"):
            uid = uid.lstrip("<@!").rstrip(">")
        if uid.lower().startswith("user:"):
            uid = uid[5:]
        if uid:
            cleaned.append(uid)
    return cleaned


def interactive_setup() -> None:
    """Guide the user through Discord bot setup: token, allowlist, home channel (lazy CLI imports)."""
    from hermes_cli.config import get_env_value, remove_env_value, save_env_value
    from hermes_cli.cli_output import (
        prompt, prompt_yes_no, print_header, print_info, print_success,
    )
    def _info_lines(*lines: str) -> None:
        for line in lines:
            print_info(line)

    def _save_allowlist(allowed_users: str) -> None:
        save_env_value("DISCORD_ALLOWED_USERS", ",".join(_clean_discord_user_ids(allowed_users)))
        print_success("Discord allowlist configured")

    print_header("Discord")
    existing = get_env_value("DISCORD_BOT_TOKEN")
    if existing:
        print_info("Discord: already configured")
        if not prompt_yes_no("Reconfigure Discord?", False):
            if not get_env_value("DISCORD_ALLOWED_USERS"):
                print_info(
                    "⚠️  Discord has no user allowlist. With the fail-closed default, "
                    "messages are denied unless you configure allowed users, roles, "
                    "or channels, or set DISCORD_ALLOW_ALL_USERS=true."
                )
                if prompt_yes_no("Add allowed users now?", True):
                    print_info("   To find Discord ID: Enable Developer Mode, right-click name → Copy ID")
                    allowed_users = prompt("Allowed user IDs (comma-separated)")
                    if allowed_users:
                        _save_allowlist(allowed_users)
            return
    _info_lines(
        "Create a bot at https://discord.com/developers/applications",
        "On Bot → Privileged Gateway Intents, enable:",
        "  - Message Content Intent (required — without it Discord rejects the connection)",
        "  - Server Members Intent (required if you use usernames or role allowlists)",
        "Save Changes in the Developer Portal before starting the gateway.",
        "Docs: https://hermes-agent.nousresearch.com/docs/user-guide/messaging/discord",
    )
    token = prompt("Discord bot token", password=True)
    if not token:
        return
    save_env_value("DISCORD_BOT_TOKEN", token)
    print_success("Discord token saved")
    print()
    _info_lines(
        "🔒 Security: Restrict who can use your bot", "   To find your Discord user ID:",
        "   1. Enable Developer Mode in Discord settings", "   2. Right-click your name → Copy ID",
    )
    print()
    print_info("   You can also use Discord usernames (resolved on gateway start).")
    print()
    allowed_users = prompt("Allowed user IDs or usernames (comma-separated, leave empty for open access)")
    if allowed_users:
        _save_allowlist(allowed_users)
    else:
        print_info(
            "⚠️  No allowlist set. Discord will deny messages until you set "
            "DISCORD_ALLOWED_USERS, DISCORD_ALLOWED_ROLES, DISCORD_ALLOWED_CHANNELS, "
            "or DISCORD_ALLOW_ALL_USERS=true for open access."
        )
    print()
    _info_lines(
        "📬 Home Channel: where Hermes delivers cron job results,",
        "   cross-platform messages, and notifications.",
        "   To get a channel ID: right-click a channel → Copy Channel ID",
        "   (requires Developer Mode in Discord settings)",
        "   You can also set this later by typing /set-home in a Discord channel.",
    )
    home_channel = prompt("Home channel ID (leave empty to set later with /set-home)").strip()
    if home_channel:
        save_env_value("DISCORD_HOME_CHANNEL", home_channel)
    elif remove_env_value("DISCORD_HOME_CHANNEL"):
        print_info("Home channel cleared.")


_YAML_BOOL_ENV_KEYS = (
    ("require_mention", "DISCORD_REQUIRE_MENTION"),
    ("thread_require_mention", "DISCORD_THREAD_REQUIRE_MENTION"),
    ("bots_require_inline_mention", "DISCORD_BOTS_REQUIRE_INLINE_MENTION"),
)
# (public websocket_* key, legacy liveness_* alias, env bridge var)
_YAML_WEBSOCKET_LIVENESS_KEYS = (
    ("websocket_liveness_interval_seconds", "liveness_interval_seconds", "HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS"),
    ("websocket_liveness_failure_threshold", "liveness_failure_threshold", "HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD"),
    ("websocket_heartbeat_ack_max_age_seconds", None, None),
    ("websocket_max_latency_seconds", None, None),
)


def _apply_yaml_config(yaml_cfg: dict, discord_cfg: dict) -> dict | None:
    """Translate ``config.yaml`` ``discord:`` keys into env vars (``apply_yaml_config_fn``).
    The adapter reads ``DISCORD_*`` via ``os.getenv()`` at ~50 sites, so this hook owns YAML→env;
    ``extra`` stays the per-adapter truth for liveness (multiplex isolation). Returns liveness settings.

    Implements the ``apply_yaml_config_fn`` contract (#24836). Mirrors the legacy ``discord_cfg`` block that
    used to live in ``gateway/config.py::load_gateway_config()`` before this migration.
    """
    def _env_default(env_key: str, value) -> None:
        # First-writer-wins: an explicit env var always beats the YAML value.
        if not os.getenv(env_key):
            os.environ[env_key] = value

    def _csv(value) -> str:
        return ",".join(str(v) for v in value) if isinstance(value, list) else str(value)

    for key, env_key in _YAML_BOOL_ENV_KEYS:
        if key in discord_cfg:
            _env_default(env_key, str(discord_cfg[key]).lower())
    platforms_cfg = yaml_cfg.get("platforms")
    platform_extra_cfg = {}
    if isinstance(platforms_cfg, dict):
        discord_platform_cfg = platforms_cfg.get("discord")
        if isinstance(discord_platform_cfg, dict):
            candidate_extra = discord_platform_cfg.get("extra")
            if isinstance(candidate_extra, dict):
                platform_extra_cfg = candidate_extra
    seeded_extra = {}
    # Gate keys are ALWAYS seeded into PlatformConfig.extra (per-profile lists); the os.environ writes
    # below are first-writer-wins for legacy consumers and skipped for profile-scoped multiplex loads.
    # The os.environ writes below remain first-writer-wins for legacy env-only consumers, but are skipped
    # for profile-scoped loads under multiplex — a secondary profile's gates must never land in
    # process-global env where they'd become another profile's policy. See #72348.
    _skip_env_bridge = _profile_scoped_config_load()

    def _gate(key: str, env_key: str, *, from_platform_extra: bool, lower: bool = False) -> None:
        value = discord_cfg[key] if key in discord_cfg else (platform_extra_cfg.get(key) if from_platform_extra else None)
        if value is None:
            return
        text = str(value).lower() if lower else _csv(value)
        seeded_extra[key] = text
        if not _skip_env_bridge:
            _env_default(env_key, text)

    _gate("allow_from", "DISCORD_ALLOWED_USERS", from_platform_extra=True)
    _gate("allowed_roles", "DISCORD_ALLOWED_ROLES", from_platform_extra=True)
    _gate("allow_all_users", "DISCORD_ALLOW_ALL_USERS", from_platform_extra=True, lower=True)
    approval_mentions_cfg = (
        discord_cfg["approval_mentions"] if "approval_mentions" in discord_cfg
        else platform_extra_cfg.get("approval_mentions")
    )
    if approval_mentions_cfg is not None:
        _env_default("DISCORD_APPROVAL_MENTIONS", str(approval_mentions_cfg).lower())
    _gate("free_response_channels", "DISCORD_FREE_RESPONSE_CHANNELS", from_platform_extra=False)
    for key, env_key in (("auto_thread", "DISCORD_AUTO_THREAD"), ("reactions", "DISCORD_REACTIONS")):
        if key in discord_cfg:
            _env_default(env_key, str(discord_cfg[key]).lower())
    backfill_cfg = discord_cfg.get("missed_message_backfill")
    if isinstance(backfill_cfg, dict):
        seeded_extra["missed_message_backfill"] = dict(backfill_cfg)
    _gate("ignored_channels", "DISCORD_IGNORED_CHANNELS", from_platform_extra=False)
    _gate("allowed_channels", "DISCORD_ALLOWED_CHANNELS", from_platform_extra=False)
    _gate("no_thread_channels", "DISCORD_NO_THREAD_CHANNELS", from_platform_extra=False)
    # history_backfill: recover mention-gated channel messages between bot turns.
    if "history_backfill" in discord_cfg:
        _env_default("DISCORD_HISTORY_BACKFILL", str(discord_cfg["history_backfill"]).lower())
    hbl = discord_cfg.get("history_backfill_limit")
    if hbl is not None:
        _env_default("DISCORD_HISTORY_BACKFILL_LIMIT", str(hbl))
    # allow_mentions: safe defaults live in the adapter; these keys only override when set.
    allow_mentions_cfg = discord_cfg.get("allow_mentions")
    if isinstance(allow_mentions_cfg, dict):
        for yaml_key in ("everyone", "roles", "users", "replied_user"):
            if yaml_key in allow_mentions_cfg:
                _env_default(f"DISCORD_ALLOW_MENTION_{yaml_key.upper()}", str(allow_mentions_cfg[yaml_key]).lower())
    # reply_to_mode: top-level preferred, falls back to extra; YAML 1.1 parses bare 'off' as False.
    _discord_extra = discord_cfg.get("extra") if isinstance(discord_cfg.get("extra"), dict) else {}
    _discord_rtm = discord_cfg["reply_to_mode"] if "reply_to_mode" in discord_cfg else _discord_extra.get("reply_to_mode")
    if _discord_rtm is not None:
        _env_default("DISCORD_REPLY_TO_MODE", "off" if _discord_rtm is False else str(_discord_rtm).lower())
    # Public config keys win over the generic ``extra`` form.
    _websocket_liveness_cfg = {**_discord_extra, **discord_cfg}
    # WebSocket health knobs (REST 200 is not Gateway health); legacy liveness_* aliases accepted.
    for primary_key, legacy_key, env_key in _YAML_WEBSOCKET_LIVENESS_KEYS:
        value = _websocket_liveness_cfg.get(primary_key)
        if value is None and legacy_key:
            value = _websocket_liveness_cfg.get(legacy_key)
        if value is not None:
            seeded_extra[primary_key] = value
            if env_key and not os.getenv(env_key):
                os.environ[env_key] = str(value)
    return seeded_extra or None


def _is_connected(config) -> bool:
    """Connected when DISCORD_BOT_TOKEN is set.
    Looks up ``hermes_cli.gateway.get_env_value`` at call time so tests can patch it (ambient env)."""
    import hermes_cli.gateway as gateway_mod
    return bool((gateway_mod.get_env_value("DISCORD_BOT_TOKEN") or "").strip())


def _build_adapter(config):
    """Factory wrapper that constructs DiscordAdapter from a PlatformConfig."""
    return DiscordAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="discord",
        label="Discord",
        adapter_factory=_build_adapter,
        check_fn=discord_deps_present,
        ensure_deps_fn=check_discord_requirements,
        is_connected=_is_connected,
        required_env=["DISCORD_BOT_TOKEN"],
        install_hint="Run `hermes setup` to install Discord support.",
        setup_fn=interactive_setup,
        # YAML→env bridge: ``discord:`` config keys → ``DISCORD_*`` env vars read via os.getenv().
        # YAML→env config bridge — owns the translation of ``config.yaml`` ``discord:`` keys
        # (require_mention, free_response_channels, auto_thread, reactions, ignored_channels,
        # allowed_channels, no_thread_channels, allow_mentions.*, reply_to_mode, thread_require_mention)
        # into ``DISCORD_*`` env vars that the adapter reads via ``os.getenv()``. Replaces the hardcoded
        # block that used to live in ``gateway/config.py``. Hook contract: #24836.
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="DISCORD_ALLOWED_USERS",
        allow_all_env="DISCORD_ALLOW_ALL_USERS",
        cron_deliver_env_var="DISCORD_HOME_CHANNEL",
        # Out-of-process cron delivery via REST, else ``deliver=discord`` jobs fail with "No live adapter".
        standalone_sender_fn=_standalone_send,
        max_message_length=2000,
        emoji="🎮",
        allow_update_command=True,
    )


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
