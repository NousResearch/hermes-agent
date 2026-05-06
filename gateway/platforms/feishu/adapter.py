"""
Feishu/Lark platform adapter.

Supports:
- WebSocket long connection and Webhook transport
- Direct-message and group @mention-gated text receive/send
- Inbound image/file/audio/media caching
- Gateway allowlist integration via FEISHU_ALLOWED_USERS
- Persistent dedup state across restarts
- Per-chat serial message processing (matches openclaw createChatQueue)
- Processing status reactions: Typing while working, removed on success,
  swapped for CrossMark on failure
- Reaction events routed as synthetic text events (matches openclaw)
- Interactive card button-click events routed as synthetic COMMAND events
- Webhook anomaly tracking (matches openclaw createWebhookAnomalyTracker)
- Verification token validation as second auth layer (matches openclaw)

Feishu identity model
---------------------
Feishu uses three user-ID tiers (official docs:
https://open.feishu.cn/document/home/user-identity-introduction/introduction):

  open_id  (ou_xxx)  — **App-scoped**.  The same person gets a different
                        open_id under each Feishu app.  Always available in
                        event payloads without extra permissions.
  user_id  (u_xxx)   — **Tenant-scoped**.  Stable within a company but
                        requires the ``contact:user.employee_id:readonly``
                        scope.  May not be present.
  union_id (on_xxx)  — **Developer-scoped**.  Same across all apps owned by
                        one developer/ISV.  Best cross-app stable ID.

For bots specifically:

  app_id              — The application's canonical credential identifier.
  bot open_id         — Returned by ``/bot/v3/info``.  This is the bot's own
                        open_id *within its app context* and is what Feishu
                        puts in ``mentions[].id.open_id`` when someone
                        @-mentions the bot.  Used for mention gating only.

In single-bot mode (what Hermes currently supports), open_id works as a
de-facto unique user identifier since there is only one app context.

Session-key participant isolation prefers ``union_id`` (via user_id_alt)
over ``open_id`` (via user_id) so that sessions stay stable if the same
user is seen through different apps in the future.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import json
import logging
import mimetypes
import os
import re
import struct
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

# aiohttp/websockets are independent optional deps — import outside lark_oapi
# so they remain available for tests and webhook mode even if lark_oapi is missing.
try:
    import aiohttp
    from aiohttp import web
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]

try:
    # ``lark`` is kept because tools/send_message_tool.py calls
    # FeishuAdapter._build_lark_client to lazy-build a raw ``lark.Client``
    # for one-off attachments outside the main outbound path. FEISHU_DOMAIN /
    # LARK_DOMAIN are re-exported through gateway/platforms/feishu/__init__.py
    # for external consumers. SDK FeishuChannel owns all wire payload
    # construction; per-symbol imports from lark_oapi.api.* / lark_oapi.core.*
    # are no longer needed at module load.
    import lark_oapi as lark
    from lark_oapi.channel import FeishuChannel  # noqa: F401 — FEISHU_AVAILABLE probe
    from lark_oapi.core.const import FEISHU_DOMAIN, LARK_DOMAIN

    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None  # type: ignore[assignment]
    FEISHU_DOMAIN = None  # type: ignore[assignment]
    LARK_DOMAIN = None  # type: ignore[assignment]

FEISHU_WEBSOCKET_AVAILABLE = websockets is not None
FEISHU_WEBHOOK_AVAILABLE = aiohttp is not None


def _sync_package_exports() -> None:
    """Keep package-level dependency probes aligned after lazy installs."""
    package = sys.modules.get(__package__)
    if package is None:
        return
    for name in (
        "FEISHU_AVAILABLE",
        "FEISHU_DOMAIN",
        "FEISHU_WEBHOOK_AVAILABLE",
        "FEISHU_WEBSOCKET_AVAILABLE",
        "LARK_DOMAIN",
    ):
        setattr(package, name, globals()[name])


from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    SUPPORTED_DOCUMENT_TYPES,
    cache_document_from_bytes,
    cache_image_from_url,
)
from gateway.platforms.feishu.webhook_guard import WebhookAnomaly
from gateway.session import SessionSource
from gateway.status import acquire_scoped_lock, release_scoped_lock
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Media type sets (still consumed by _normalize_media_extension below)
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
_AUDIO_EXTENSIONS = {".ogg", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".opus", ".webm"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}


def _parse_opus_duration_ms(data: bytes) -> Optional[int]:
    if not data or len(data) < 27:
        return None
    offset = data.rfind(b"OggS")
    while offset >= 0:
        if offset + 27 <= len(data):
            try:
                granule = struct.unpack_from("<q", data, offset + 6)[0]
            except struct.error:
                return None
            if granule >= 0:
                return int(round(granule / 48))
        offset = data.rfind(b"OggS", 0, offset)
    return None


def _parse_mp4_duration_ms(data: bytes) -> Optional[int]:
    moov = _find_mp4_box(data, 0, len(data), b"moov")
    if moov is None:
        return None
    mvhd = _find_mp4_box(data, moov[0], moov[1], b"mvhd")
    if mvhd is None:
        return None
    start, end = mvhd
    if end - start < 32:
        return None

    version = data[start]
    try:
        if version == 0:
            timescale = struct.unpack_from(">I", data, start + 12)[0]
            duration = struct.unpack_from(">I", data, start + 16)[0]
        else:
            timescale = struct.unpack_from(">I", data, start + 20)[0]
            duration = struct.unpack_from(">Q", data, start + 24)[0]
    except struct.error:
        return None
    if not timescale or duration <= 0:
        return None
    return int(round((duration / timescale) * 1000))


def _find_mp4_box(
    data: bytes,
    start: int,
    end: int,
    wanted: bytes,
) -> Optional[tuple[int, int]]:
    offset = start
    while offset + 8 <= end:
        try:
            size = struct.unpack_from(">I", data, offset)[0]
            box_type = data[offset + 4 : offset + 8]
        except struct.error:
            return None

        header_len = 8
        if size == 1:
            if offset + 16 > end:
                return None
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            header_len = 16
        elif size == 0:
            size = end - offset

        if size < header_len or offset + size > end:
            return None

        payload_start = offset + header_len
        payload_end = offset + size
        if box_type == wanted:
            return payload_start, payload_end
        offset += size
    return None


# ---------------------------------------------------------------------------
# Connection, retry and batching tuning
# ---------------------------------------------------------------------------

_FEISHU_CONNECT_ATTEMPTS = 3
_FEISHU_APP_LOCK_SCOPE = "feishu-app-id"
_DEFAULT_TEXT_BATCH_DELAY_SECONDS = 0.6
_DEFAULT_TEXT_BATCH_MAX_MESSAGES = 8
_DEFAULT_TEXT_BATCH_MAX_CHARS = 4000
_DEFAULT_MEDIA_BATCH_DELAY_SECONDS = 0.8
_DEFAULT_WEBHOOK_HOST = "127.0.0.1"
_DEFAULT_WEBHOOK_PORT = 8765
_DEFAULT_WEBHOOK_PATH = "/feishu/webhook"
# ---------------------------------------------------------------------------
# TTL, rate-limit and webhook security constants
# ---------------------------------------------------------------------------

# Persistent dedup-store tunables consumed by JsonFileDedupStore ctor and
# the SDK DedupConfig builder. Webhook server tunables (rate-limit window/
# cap, body-size, anomaly tracker) live in webhook_guard.py.
_FEISHU_DEDUP_DEFAULT_TTL_SECONDS = 24 * 3600
_FEISHU_DEDUP_DEFAULT_CACHE_SIZE = 2048
_FEISHU_CARD_ACTION_DEDUP_TTL_SECONDS = 15 * 60   # card action token dedup window (15 min)
_MIN_LARK_OAPI_VERSION = (1, 6, 5)


# JsonFileDedupStore lives in gateway/platforms/feishu/dedup_store.py; re-import
# locally so connect() can refer to it under its historical name.
from gateway.platforms.feishu.dedup_store import JsonFileDedupStore  # noqa: F401


# _APPROVAL_CHOICE_MAP / _APPROVAL_LABEL_MAP live in
# gateway/platforms/feishu/approvals.py; re-imported here so existing
# references inside this file keep working.
from gateway.platforms.feishu.approvals import (  # noqa: F401
    _APPROVAL_CHOICE_MAP,
    _APPROVAL_LABEL_MAP,
)

# Feishu reactions render as prominent badges, unlike Discord/Telegram's
# small footer emoji — a success badge on every message would add noise, so
# we only mark start (Typing) and failure (CrossMark); the reply itself is
# the success signal.
_FEISHU_REACTION_IN_PROGRESS = "Typing"
_FEISHU_REACTION_FAILURE = "CrossMark"
# Bound on the (message_id → reaction_id) handle cache. Happy-path entries
# drain on completion; the cap is a safeguard against unbounded growth from
# delete-failures, not a capacity plan.
_FEISHU_PROCESSING_REACTION_CACHE_SIZE = 1024

# Markdown table detector. The Feishu client does not render markdown tables
# inside post ``tag:md`` nodes (the format produced by SDK ``tag_md_mode=
# "native"``), so messages containing a table arrive blank. When this regex
# matches, ``send`` bypasses the markdown converter and ships the chunk as
# plain text so the table source stays visible.
_FEISHU_MARKDOWN_TABLE_RE = re.compile(r"^\|.*\|\n\|[-|: ]+\|", re.MULTILINE)
_FEISHU_MARKDOWN_TABLE_DELIM_RE = re.compile(r"^\|[-|: ]+\|\s*$")
_FEISHU_MARKDOWN_TABLE_LINE_RE = re.compile(r"^\|.*\|\s*$")


def _split_feishu_table_chunks(text: str) -> List[tuple[str, bool]]:
    """Split markdown so only table blocks are downgraded to plain text."""
    if not _FEISHU_MARKDOWN_TABLE_RE.search(text):
        return [(text, False)] if text else []

    lines = text.splitlines()
    parts: List[tuple[str, bool]] = []
    markdown_lines: List[str] = []
    idx = 0

    def _flush_markdown() -> None:
        nonlocal markdown_lines
        chunk = "\n".join(markdown_lines).strip("\n")
        if chunk.strip():
            parts.append((chunk, False))
        markdown_lines = []

    while idx < len(lines):
        is_table_start = (
            idx + 1 < len(lines)
            and _FEISHU_MARKDOWN_TABLE_LINE_RE.match(lines[idx])
            and _FEISHU_MARKDOWN_TABLE_DELIM_RE.match(lines[idx + 1])
        )
        if not is_table_start:
            markdown_lines.append(lines[idx])
            idx += 1
            continue

        _flush_markdown()
        table_lines = [lines[idx], lines[idx + 1]]
        idx += 2
        while idx < len(lines) and _FEISHU_MARKDOWN_TABLE_LINE_RE.match(lines[idx]):
            table_lines.append(lines[idx])
            idx += 1
        parts.append(("\n".join(table_lines).strip("\n"), True))

    _flush_markdown()
    return parts


@dataclass
class FeishuGroupRule:
    """Per-group policy rule for controlling which users may interact with the bot."""

    policy: str  # "open" | "allowlist" | "blacklist" | "admin_only" | "disabled"
    allowlist: set[str] = field(default_factory=set)
    blacklist: set[str] = field(default_factory=set)
    require_mention: Optional[bool] = None  # None = inherit global setting


@dataclass(frozen=True)
class FeishuAdapterSettings:
    app_id: str  # Canonical bot/app identifier (credential, not from event payloads)
    app_secret: str
    domain_name: str
    connection_mode: str
    encrypt_key: str
    verification_token: str
    group_policy: str
    allowed_group_users: frozenset[str]
    allow_all_users: bool
    # Manual fallback for bot identity. SDK hydration refreshes these after
    # connect() when available.
    bot_open_id: str
    bot_user_id: str
    bot_name: str
    dedup_cache_size: int
    text_batch_delay_seconds: float
    text_batch_split_delay_seconds: float
    text_batch_max_messages: int
    text_batch_max_chars: int
    media_batch_delay_seconds: float
    webhook_host: str
    webhook_port: int
    webhook_path: str
    # WS tuning (reconnect/ping intervals etc.) is owned by SDK
    # TransportConfig (server-authoritative ClientConfig). The corresponding
    # HERMES_FEISHU_WS_* env vars are tolerated in _load_settings for
    # backward compatibility but otherwise unused.
    admins: frozenset[str] = frozenset()
    default_group_policy: str = ""
    group_rules: Dict[str, FeishuGroupRule] = field(default_factory=dict)
    # Bot-admission and mention-policy toggles.
    # ``allow_bots`` ∈ {"none","mentions","all"} controls whether peer-bot
    # messages are admitted. ``require_mention`` is the global default for
    # group messages requiring an @-mention; per-chat overrides via
    # ``group_rules.<chat_id>.require_mention``.
    allow_bots: str = "none"
    require_mention: bool = True


def check_feishu_requirements() -> bool:
    """Check if Feishu/Lark dependencies are available.

    Feishu is excluded from the eager ``all`` extra and must lazy-install at
    first gateway use. A forced install is used when the SDK import probe
    failed at module load, because an older ``lark-oapi`` distribution can be
    present while still missing the ``lark_oapi.channel`` package required by
    this adapter.
    """
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        force_install = not FEISHU_AVAILABLE or _lark_oapi_needs_forced_install()
        _lazy_ensure("platform.feishu", prompt=False, force=force_install)
    except Exception:
        return False

    sdk_available = _load_feishu_sdk()
    transport_available = _load_feishu_transport_deps()
    return sdk_available and transport_available


def check_feishu_send_requirements() -> bool:
    """Check dependencies needed for one-shot outbound Feishu sends."""
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        force_install = not FEISHU_AVAILABLE or _lark_oapi_needs_forced_install()
        _lazy_ensure("platform.feishu", prompt=False, force=force_install)
    except Exception:
        return False
    return _load_feishu_sdk()


def _lark_oapi_needs_forced_install() -> bool:
    try:
        from importlib import metadata
        installed = metadata.version("lark-oapi")
    except metadata.PackageNotFoundError:
        return True
    except Exception:
        return not FEISHU_AVAILABLE

    parts = []
    for part in re.split(r"[^0-9]+", installed):
        if part:
            parts.append(int(part))
    return tuple(parts or [0]) < _MIN_LARK_OAPI_VERSION


def _load_feishu_sdk() -> bool:
    """Import and bind the Feishu SDK globals after lazy installation."""
    global FEISHU_AVAILABLE, FEISHU_DOMAIN, LARK_DOMAIN, lark
    try:
        import lark_oapi as _lark
        from lark_oapi.channel import FeishuChannel as _FeishuChannel
        from lark_oapi.core.const import (
            FEISHU_DOMAIN as _FEISHU_DOMAIN,
            LARK_DOMAIN as _LARK_DOMAIN,
        )
    except ImportError:
        FEISHU_AVAILABLE = False
        lark = None  # type: ignore[assignment]
        FEISHU_DOMAIN = None  # type: ignore[assignment]
        LARK_DOMAIN = None  # type: ignore[assignment]
        _sync_package_exports()
        return False

    _ = _FeishuChannel
    lark = _lark
    FEISHU_DOMAIN = _FEISHU_DOMAIN
    LARK_DOMAIN = _LARK_DOMAIN
    FEISHU_AVAILABLE = True
    _sync_package_exports()
    return True


def _load_feishu_transport_deps() -> bool:
    """Import and bind Feishu transport deps after lazy installation."""
    global FEISHU_WEBHOOK_AVAILABLE, FEISHU_WEBSOCKET_AVAILABLE
    global aiohttp, web, websockets
    try:
        import aiohttp as _aiohttp
        from aiohttp import web as _web
    except ImportError:
        _aiohttp = None
        _web = None

    try:
        import websockets as _websockets
    except ImportError:
        _websockets = None

    aiohttp = _aiohttp  # type: ignore[assignment]
    web = _web  # type: ignore[assignment]
    websockets = _websockets  # type: ignore[assignment]
    FEISHU_WEBHOOK_AVAILABLE = _aiohttp is not None and _web is not None
    FEISHU_WEBSOCKET_AVAILABLE = _websockets is not None
    webhook_guard = sys.modules.get("gateway.platforms.feishu.webhook_guard")
    if webhook_guard is not None:
        setattr(webhook_guard, "aiohttp", _aiohttp)
        setattr(webhook_guard, "web", _web)
        setattr(webhook_guard, "WEBHOOK_AVAILABLE", FEISHU_WEBHOOK_AVAILABLE)
    _sync_package_exports()
    return FEISHU_WEBHOOK_AVAILABLE or FEISHU_WEBSOCKET_AVAILABLE


# events_mapping helpers live in gateway/platforms/feishu/events_mapping.py.
# Re-imported here so FeishuAdapter handler bodies can reference them under
# the original names.
from gateway.platforms.feishu.events_mapping import (  # noqa: F401, E402
    _build_mention_hint,
    _strip_edge_self_mentions,
    _sdk_content_to_message_type,
    self_get_chat_info_safe,
    _resolve_source_chat_type_for_event,
)


from gateway.platforms.feishu.events_mapping import (  # noqa: F401, E402
    to_message_event,
    _to_command_event_from_card_action,
    _to_text_event_from_reaction,
    _sdk_comment_to_legacy_dict,
)



class FeishuAdapter(BasePlatformAdapter):
    """Feishu/Lark bot adapter bridging Hermes to ``lark_oapi.channel.FeishuChannel``.

    Adapts ``BasePlatformAdapter`` (Hermes) to the lark-oapi FeishuChannel
    SDK. The SDK owns transport (WS or webhook), inbound dedup/policy/
    mention/lock/batch/queue filtering, identity hydration, post/card
    parsing, markdown rendering, retry, SSRF and most chunking. This
    class is the projection layer between Hermes' generic platform
    contract and the SDK's typed events/configs.

    Inbound flow:
        SDK ``channel.on("message"|"cardAction"|"reaction"|"comment"|...)``
        → ``_on_sdk_*`` → ``events_mapping.to_message_event`` (or the
        equivalent synthesizer for cardAction / reaction / comment) →
        ``MessageEvent`` → ``BasePlatformAdapter.handle_message``.

    Outbound flow:
        ``send`` / ``edit_message`` / ``send_image`` / ``send_voice`` /
        ``send_document`` / ``send_video`` / ``send_image_file`` →
        ``self._channel.send`` (or the equivalent SDK method). Audio/file
        captions preserve Hermes' existing behavior by sending caption text
        first, then the native attachment.

    Adapter-side responsibilities (not the SDK's):
        - Drive document comment LLM flow (``comments.handle_drive_comment_event``)
        - QR-based bot onboarding (``qr_register`` / ``probe_bot``)
        - Webhook aiohttp server + rate-limit / anomaly guards (``webhook_guard``)
        - Persistent dedup store (``JsonFileDedupStore``)
        - Exec-approval card flow (``approvals._send_exec_approval_impl``)
        - Processing-status reactions (Typing on start, CrossMark on failure)

    Configuration env vars: ``FEISHU_APP_ID``, ``FEISHU_APP_SECRET``,
    ``FEISHU_DOMAIN``, ``FEISHU_CONNECTION_MODE``, ``FEISHU_ENCRYPT_KEY``,
    ``FEISHU_VERIFICATION_TOKEN``, ``FEISHU_GROUP_POLICY``,
    ``FEISHU_ALLOWED_USERS``, ``FEISHU_ALLOW_ALL_USERS``,
    ``FEISHU_BOT_OPEN_ID/USER_ID/NAME``,
    ``FEISHU_WEBHOOK_HOST/PORT/PATH``, ``FEISHU_REACTIONS``, plus
    ``HERMES_FEISHU_*`` tunables for batching and dedup.
    """

    MAX_MESSAGE_LENGTH = 8000
    # Threshold for detecting Feishu client-side message splits.
    # When a chunk is near the ~4096-char practical limit, a continuation
    # is almost certain.
    _SPLIT_THRESHOLD = 4000

    # =========================================================================
    # Lifecycle — init / settings / connect / disconnect
    # =========================================================================

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.FEISHU)

        self._settings = self._load_settings(config.extra or {})
        self._apply_settings(self._settings)
        self._client: Optional[Any] = None
        self._ws_client: Optional[Any] = None
        self._channel: Optional["FeishuChannel"] = None
        # Persistent dedup store. Instantiated in connect(); flushed and
        # cleared in disconnect(). Backs SDK Deduper.check_and_mark.
        self._dedup_store: Optional["JsonFileDedupStore"] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._webhook_runner: Optional[Any] = None
        # Partial-bound aiohttp handler created by webhook_guard at startup;
        # ``_handle_webhook_request`` delegates to it so contract tests can
        # exercise the guard pipeline directly.
        self._webhook_handler: Optional[Callable[[Any], Awaitable[Any]]] = None
        self._dedup_state_path = get_hermes_home() / "feishu_seen_message_ids.json"
        self._chat_info_cache: Dict[str, Dict[str, Any]] = {}
        self._app_lock_identity: Optional[str] = None
        # Exec approval button state (approval_id → {session_key, message_id, chat_id})
        self._approval_state: Dict[int, Dict[str, str]] = {}
        self._approval_counter = itertools.count(1)
        # Update prompt button state (prompt_id → {session_key, message_id, chat_id})
        self._update_prompt_state: Dict[int, Dict[str, str]] = {}
        self._update_prompt_counter = itertools.count(1)
        # Feishu reaction deletion requires the opaque reaction_id returned
        # by create, so we cache it per message_id.
        self._pending_processing_reactions: "OrderedDict[str, str]" = OrderedDict()
        self._known_peer_bot_open_ids: set[str] = set()

    @staticmethod
    def _load_settings(extra: Dict[str, Any]) -> FeishuAdapterSettings:
        # Deprecated WS tuning vars: WS reconnect/ping behavior is owned by
        # SDK TransportConfig (server-authoritative ClientConfig). Parse and
        # discard for backward compatibility.
        _DEPRECATED_ENV_VARS: Sequence[tuple[str, str]] = (
            ("HERMES_FEISHU_WS_RECONNECT_NONCE",
             "SDK TransportConfig does not expose this (server-authoritative ClientConfig)"),
            ("HERMES_FEISHU_WS_RECONNECT_INTERVAL",
             "SDK TransportConfig does not expose this (server-authoritative ClientConfig)"),
            ("HERMES_FEISHU_WS_PING_INTERVAL",
             "SDK TransportConfig does not expose this (server-authoritative ClientConfig)"),
            ("HERMES_FEISHU_WS_PING_TIMEOUT",
             "SDK TransportConfig does not expose this (server-authoritative ClientConfig)"),
        )
        for env_var, reason in _DEPRECATED_ENV_VARS:
            if os.environ.get(env_var):
                logger.debug(
                    "[Feishu] %s is set but no longer used (%s); ignoring.",
                    env_var,
                    reason,
                )
        # Per-config (extra) keys for the same deprecated fields are also
        # tolerated — e.g. test fixtures still pass ws_reconnect_nonce=0.
        for legacy_key in (
            "ws_reconnect_nonce",
            "ws_reconnect_interval",
            "ws_ping_interval",
            "ws_ping_timeout",
        ):
            if legacy_key in extra:
                logger.debug(
                    "[Feishu] config key '%s' is set but no longer used "
                    "(SDK TransportConfig owns WS tuning); ignoring.",
                    legacy_key,
                )

        # Parse per-group rules from config
        raw_group_rules = extra.get("group_rules", {})
        group_rules: Dict[str, FeishuGroupRule] = {}
        if isinstance(raw_group_rules, dict):
            for chat_id, rule_cfg in raw_group_rules.items():
                if not isinstance(rule_cfg, dict):
                    continue
                # Only override when ``require_mention`` is explicitly set —
                # missing vs false must not collapse (None = inherit global).
                per_chat_require_mention: Optional[bool] = None
                if "require_mention" in rule_cfg:
                    raw_rm = rule_cfg.get("require_mention")
                    if isinstance(raw_rm, bool):
                        per_chat_require_mention = raw_rm
                    else:
                        per_chat_require_mention = (
                            str(raw_rm).strip().lower() not in {"false", "0", "no"}
                        )
                group_rules[str(chat_id)] = FeishuGroupRule(
                    policy=str(rule_cfg.get("policy", "open")).strip().lower(),
                    allowlist=set(str(u).strip() for u in rule_cfg.get("allowlist", []) if str(u).strip()),
                    blacklist=set(str(u).strip() for u in rule_cfg.get("blacklist", []) if str(u).strip()),
                    require_mention=per_chat_require_mention,
                )

        # Bot-level admins
        raw_admins = extra.get("admins", [])
        admins = frozenset(str(u).strip() for u in raw_admins if str(u).strip())

        # Default group policy (for groups not in group_rules)
        default_group_policy = str(extra.get("default_group_policy", "")).strip().lower()

        # Bot admission: env-only so adapter and gateway auth bypass share one
        # source of truth (yaml feishu.allow_bots is bridged to this env var
        # at config load).
        allow_bots_raw = os.environ.get("FEISHU_ALLOW_BOTS", "").strip().lower()
        allow_bots = allow_bots_raw or "none"
        if allow_bots not in {"none", "mentions", "all"}:
            logger.warning(
                "[Feishu] Unknown allow_bots=%r, falling back to 'none'. "
                "Valid: none, mentions, all.",
                allow_bots_raw,
            )
            allow_bots = "none"

        # Global default for group @-mention requirement; default True.
        # ``extra["require_mention"]`` (yaml) overrides the env var when
        # explicitly set; otherwise fall back to FEISHU_REQUIRE_MENTION.
        if "require_mention" in extra:
            raw_rm = extra.get("require_mention")
            if isinstance(raw_rm, bool):
                require_mention = raw_rm
            else:
                require_mention = (
                    str(raw_rm).strip().lower() not in {"false", "0", "no"}
                )
        else:
            require_mention_raw = os.environ.get("FEISHU_REQUIRE_MENTION", "").strip().lower()
            require_mention = (
                require_mention_raw not in {"false", "0", "no"}
                if require_mention_raw
                else True
            )

        return FeishuAdapterSettings(
            app_id=str(extra.get("app_id") or os.getenv("FEISHU_APP_ID", "")).strip(),
            app_secret=str(extra.get("app_secret") or os.getenv("FEISHU_APP_SECRET", "")).strip(),
            domain_name=str(extra.get("domain") or os.getenv("FEISHU_DOMAIN", "feishu")).strip().lower(),
            connection_mode=str(
                extra.get("connection_mode") or os.getenv("FEISHU_CONNECTION_MODE", "websocket")
            ).strip().lower(),
            encrypt_key=os.getenv("FEISHU_ENCRYPT_KEY", "").strip(),
            verification_token=os.getenv("FEISHU_VERIFICATION_TOKEN", "").strip(),
            group_policy=os.getenv("FEISHU_GROUP_POLICY", "allowlist").strip().lower(),
            allowed_group_users=frozenset(
                item.strip()
                for item in os.getenv("FEISHU_ALLOWED_USERS", "").split(",")
                if item.strip()
            ),
            allow_all_users=os.getenv("FEISHU_ALLOW_ALL_USERS", "").strip().lower() in (
                "true", "1", "yes",
            ),
            bot_open_id=str(
                extra.get("bot_open_id") or os.getenv("FEISHU_BOT_OPEN_ID", "")
            ).strip(),
            bot_user_id=str(
                extra.get("bot_user_id") or os.getenv("FEISHU_BOT_USER_ID", "")
            ).strip(),
            bot_name=str(
                extra.get("bot_name") or os.getenv("FEISHU_BOT_NAME", "")
            ).strip(),
            dedup_cache_size=max(
                32,
                int(os.getenv("HERMES_FEISHU_DEDUP_CACHE_SIZE", str(_FEISHU_DEDUP_DEFAULT_CACHE_SIZE))),
            ),
            text_batch_delay_seconds=float(
                os.getenv("HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS", str(_DEFAULT_TEXT_BATCH_DELAY_SECONDS))
            ),
            text_batch_split_delay_seconds=float(
                os.getenv("HERMES_FEISHU_TEXT_BATCH_SPLIT_DELAY_SECONDS", "2.0")
            ),
            text_batch_max_messages=max(
                1,
                int(os.getenv("HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES", str(_DEFAULT_TEXT_BATCH_MAX_MESSAGES))),
            ),
            text_batch_max_chars=max(
                1,
                int(os.getenv("HERMES_FEISHU_TEXT_BATCH_MAX_CHARS", str(_DEFAULT_TEXT_BATCH_MAX_CHARS))),
            ),
            media_batch_delay_seconds=float(
                os.getenv("HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS", str(_DEFAULT_MEDIA_BATCH_DELAY_SECONDS))
            ),
            webhook_host=str(
                extra.get("webhook_host") or os.getenv("FEISHU_WEBHOOK_HOST", _DEFAULT_WEBHOOK_HOST)
            ).strip(),
            webhook_port=int(
                extra.get("webhook_port") or os.getenv("FEISHU_WEBHOOK_PORT", str(_DEFAULT_WEBHOOK_PORT))
            ),
            webhook_path=(
                str(extra.get("webhook_path") or os.getenv("FEISHU_WEBHOOK_PATH", _DEFAULT_WEBHOOK_PATH)).strip()
                or _DEFAULT_WEBHOOK_PATH
            ),
            admins=admins,
            default_group_policy=default_group_policy,
            group_rules=group_rules,
            allow_bots=allow_bots,
            require_mention=require_mention,
        )

    def _apply_settings(self, settings: FeishuAdapterSettings) -> None:
        self._app_id = settings.app_id
        self._app_secret = settings.app_secret
        self._domain_name = settings.domain_name
        self._connection_mode = settings.connection_mode
        self._encrypt_key = settings.encrypt_key
        self._verification_token = settings.verification_token
        self._group_policy = settings.group_policy
        self._allowed_group_users = set(settings.allowed_group_users)
        self._allow_all_users = settings.allow_all_users
        self._admins = set(settings.admins)
        self._default_group_policy = settings.default_group_policy or settings.group_policy
        self._group_rules = settings.group_rules
        # dedup_cache_size flows directly into JsonFileDedupStore ctor in
        # connect(); SDK hydration can refresh bot identity after connect.
        self._webhook_host = settings.webhook_host
        self._webhook_port = settings.webhook_port
        self._webhook_path = settings.webhook_path
        # Admission policy state projected into SDK policy config; a small
        # Hermes-side fallback still uses _allow_group_message when needed.
        # Bot-identity is hydrated post-connect by the SDK channel and copied
        # onto the adapter for self-message and peer-bot checks.
        self._allow_bots = settings.allow_bots
        self._require_mention = settings.require_mention
        self._bot_open_id = settings.bot_open_id or getattr(self, "_bot_open_id", "") or ""
        self._bot_user_id = settings.bot_user_id or getattr(self, "_bot_user_id", "") or ""
        self._bot_name = settings.bot_name or getattr(self, "_bot_name", "") or ""

    def _build_sdk_policy_config(
        self, settings: "FeishuAdapterSettings"
    ) -> Any:
        """Project Hermes settings → SDK PolicyConfig.

        Semantic alignment between Hermes' historic ``allowed_group_users``
        (a PER-USER allowlist, despite the field name) and SDK
        ``PolicyConfig.group_allowlist`` (a PER-CHAT chat_id allowlist):
        we map the per-user list onto SDK ``allow_from`` (the sender
        open_id gate, applied after policy_kind branches) and force
        ``group_policy="open"`` so the chat_id check does not pre-reject
        groups Hermes intended to allow.

            FEISHU_GROUP_POLICY=allowlist + FEISHU_ALLOWED_USERS=A,B
              → SDK group_policy=open, allow_from=[A,B]

        ``FEISHU_ALLOW_ALL_USERS=true`` is Hermes' top-level skip-user-auth
        flag (see ``gateway/run.py``); when set, we relax SDK to fully
        open so the SDK's policy gate doesn't pre-reject anything that
        Hermes' upper layer would have authorized.

        Mapping summary:
        - default_group_policy / group_policy → SDK group_policy
          ("blacklist" → "blocklist")
        - allowed_group_users → SDK allow_from (per-user filter)
        - admins → SDK admins
        - group_rules → SDK group_overrides (per-chat-id rules; per-rule
          allowlist/blacklist ARE per-user, matching GroupOverride semantics)
        - DM stays "open" (Hermes has no DM-policy fields).

        Always-on invariants: dm_policy="open", respond_to_mention_all=True.
        require_mention is taken from settings.require_mention (operator
        override via FEISHU_REQUIRE_MENTION; default True).
        """
        from lark_oapi.channel import PolicyConfig, GroupOverride

        policy_config_fields = set(inspect.signature(PolicyConfig).parameters)
        supports_sender_identity_fields = "sender_identity_fields" in policy_config_fields
        self._sdk_policy_supports_sender_identity_fields = supports_sender_identity_fields

        def _policy_config(**kwargs: Any) -> Any:
            return PolicyConfig(**{k: v for k, v in kwargs.items() if k in policy_config_fields})

        # Map Hermes' historical "blacklist" alias onto SDK's "blocklist".
        def _normalize_policy(p: str) -> str:
            return "blocklist" if p == "blacklist" else (p or "open")

        peer_bots_need_hermes_gate = settings.allow_bots in {"mentions", "all"}
        group_overrides: Dict[str, Any] = {}
        for chat_id, rule in (settings.group_rules or {}).items():
            rule_policy = _normalize_policy(rule.policy) if rule.policy else None
            sdk_rule_policy = rule_policy
            sdk_rule_allowlist = list(rule.allowlist) if rule.allowlist else None
            sdk_rule_blocklist = list(rule.blacklist) if rule.blacklist else None
            if peer_bots_need_hermes_gate and rule_policy in {"allowlist", "blocklist"}:
                sdk_rule_policy = "open"
                sdk_rule_allowlist = None
                sdk_rule_blocklist = None
            group_overrides[chat_id] = GroupOverride(
                policy=sdk_rule_policy,
                allowlist=sdk_rule_allowlist,
                blocklist=sdk_rule_blocklist,
                require_mention=rule.require_mention,
            )

        # FEISHU_ALLOW_ALL_USERS=true overrides everything — Hermes top-level
        # auth allows any sender, so SDK policy gate must not pre-reject.
        if settings.allow_all_users:
            return _policy_config(
                dm_policy="open",
                group_policy="open",
                require_mention=settings.require_mention,
                respond_to_mention_all=True,
                admins=list(settings.admins) if settings.admins else None,
                sender_identity_fields=["open_id", "user_id", "union_id"],
                group_overrides=group_overrides,
            )

        # Translate Hermes' "allowlist" semantics: per-user filter, not
        # per-chat. SDK's `group_allowlist` field is per-chat — wrong shape.
        # SDK's `allow_from` is the per-user gate, applied after policy_kind.
        hermes_policy = _normalize_policy(
            settings.default_group_policy or settings.group_policy or "open"
        )
        sdk_group_policy: str
        sdk_allow_from: Optional[list]
        sdk_group_allowlist: Optional[list]
        if hermes_policy == "allowlist":
            # Per-user gate via allow_from; group_policy=open lets all chats
            # through, then allow_from filters by sender identity. SDK 1.6.0
            # only checks sender.open_id; when it lacks multi-identity
            # support, Hermes applies the per-user fallback in _on_sdk_message.
            # If peer bots are allowed, keep allow_from empty so the SDK
            # cannot drop them before Hermes' bot admission gate runs.
            sdk_group_policy = "open"
            sdk_allow_from = (
                list(settings.allowed_group_users)
                if (
                    settings.allowed_group_users
                    and supports_sender_identity_fields
                    and not peer_bots_need_hermes_gate
                )
                else None
            )
            sdk_group_allowlist = None
        else:
            sdk_group_policy = hermes_policy
            sdk_allow_from = None
            sdk_group_allowlist = None  # Hermes settings has no per-chat-id allowlist field

        return _policy_config(
            dm_policy="open",
            group_policy=sdk_group_policy,
            require_mention=settings.require_mention,
            respond_to_mention_all=True,
            admins=list(settings.admins) if settings.admins else None,
            allow_from=sdk_allow_from,
            group_allowlist=sdk_group_allowlist,
            sender_identity_fields=["open_id", "user_id", "union_id"],
            group_overrides=group_overrides,
        )

    def _build_sdk_safety_config(
        self, settings: "FeishuAdapterSettings"
    ) -> Any:
        """Project Hermes settings → SDK SafetyConfig.

        - MediaBatchConfig defaults to ``enabled=False`` in the SDK;
          Hermes opts in explicitly to preserve same-chat consecutive-image
          merging.
        - DedupConfig.ttl comes from the module-level
          ``_FEISHU_DEDUP_DEFAULT_TTL_SECONDS`` constant; ``max_entries``
          from ``settings.dedup_cache_size``.
        - TextBatchConfig.delay_ms = ``settings.text_batch_delay_seconds * 1000``;
          ``max_messages`` / ``max_chars`` map directly from settings.
        - ``long_threshold_chars`` / ``long_delay_ms`` / ``sweep_seconds`` /
          ``chat_queue`` / ``stale_message_window_ms`` use SDK defaults.
        - ``safety_cache`` is not a SafetyConfig field; it is a FeishuChannel
          ctor parameter, left None here so the SDK uses its in-memory
          SeenCache (cross-process caching is a future need).
        """
        from lark_oapi.channel import (
            SafetyConfig,
            DedupConfig,
            TextBatchConfig,
            MediaBatchConfig,
        )

        return SafetyConfig(
            dedup=DedupConfig(
                enabled=True,
                ttl_seconds=int(_FEISHU_DEDUP_DEFAULT_TTL_SECONDS),
                max_entries=settings.dedup_cache_size,
            ),
            text_batch=TextBatchConfig(
                delay_ms=int(settings.text_batch_delay_seconds * 1000),
                max_messages=settings.text_batch_max_messages,
                max_chars=settings.text_batch_max_chars,
            ),
            media_batch=MediaBatchConfig(
                enabled=True,
                delay_ms=int(settings.media_batch_delay_seconds * 1000),
                max_items=9,
            ),
        )

    def _sdk_transport_kind(self) -> str:
        mode = self._settings.connection_mode or "websocket"
        return "webhook" if mode == "webhook" else "ws"

    def _resolve_sdk_domain(self) -> str:
        """Map Hermes' historical short domain setting to SDK URL form."""
        domain_short = (self._settings.domain_name or "feishu").strip().lower()
        if domain_short.startswith("http"):
            return domain_short
        if domain_short == "lark":
            return LARK_DOMAIN or "https://open.larksuite.com"
        return FEISHU_DOMAIN or "https://open.feishu.cn"

    def _build_sdk_outbound_config(self) -> Any:
        from lark_oapi.channel import (
            MarkdownConverter,
            OutboundConfig,
            RetryConfig,
        )

        return OutboundConfig(
            text_chunk_limit=self.MAX_MESSAGE_LENGTH,
            retry=RetryConfig(max_attempts=3),
            markdown_converter=MarkdownConverter(
                enabled=True,
                table_mode="bullets",
                # tag_md_mode="native" emits tag:md AST, which the
                # Feishu client renders with native markdown styling.
                tag_md_mode="native",
            ),
            # Open SSRF allowlist matches the historical Hermes behavior;
            # revisit when we tighten the outbound guard.
            ssrf_allowlist=["*"],
            # on_oversize left None — no paste_rs_fallback yet.
        )

    def _register_sdk_channel_handlers(self, channel: Any) -> None:
        """Bridge SDK channel events to Hermes handlers."""
        channel.on("message", self._on_sdk_message)
        channel.on("reject", self._on_sdk_reject)
        channel.on("error", self._on_sdk_error)
        channel.on("reconnecting", self._on_sdk_reconnecting)
        channel.on("reconnected", self._on_sdk_reconnected)
        channel.on("cardAction", self._on_sdk_card_action)
        channel.on("reaction", self._on_sdk_reaction)
        channel.on("comment", self._on_sdk_comment)
        channel.on("botAdded", self._on_sdk_bot_added)
        channel.on("botLeave", self._on_sdk_bot_leave)
        # messageRead is intentionally not registered — Hermes does not
        # maintain message-read receipts.

    def _install_comment_dispatch_override(self, channel: Any) -> None:
        """Patch SDK comment action identity so reply_id participates in dedup."""
        try:
            from lark_oapi.channel.normalize.comment import normalize_comment
        except Exception:
            logger.debug("[Feishu] SDK normalize_comment unavailable; using SDK comment handler")
            return

        async def _handle_comment_event(data: Any) -> None:
            try:
                raw_event = getattr(data, "event", None)
                header = getattr(data, "header", None)
                envelope_ts = (
                    getattr(header, "create_time", None)
                    if header is not None
                    else getattr(data, "ts", None)
                )
                normalized = normalize_comment(
                    raw_event if raw_event is not None else data,
                    bot_open_id=getattr(channel, "_bot_open_id", None),
                    envelope_timestamp=envelope_ts,
                )
                if normalized is None:
                    return
                event_id = (
                    f"comment:{normalized.file_token}:{normalized.comment_id}"
                    + (f":{normalized.reply_id}" if normalized.reply_id else "")
                )
                await channel._through_action_safety(
                    event_id=event_id,
                    queue_scope=normalized.file_token,
                    handler=lambda: channel._invoke("comment", normalized),
                )
            except Exception as exc:
                logger.exception("FeishuChannel comment dispatch failed: %s", exc)

        channel._handle_comment_event = _handle_comment_event

    def _build_sdk_channel(
        self,
        *,
        transport_kind: Optional[str] = None,
        dedup_store: Optional[Any] = None,
        register_handlers: bool = True,
    ) -> Any:
        from lark_oapi import LogLevel
        from lark_oapi.channel import (
            FeishuChannel,
            InboundConfig,
            TransportConfig,
        )

        channel = FeishuChannel(
            app_id=self._settings.app_id,
            app_secret=self._settings.app_secret,
            domain=self._resolve_sdk_domain(),
            log_level=LogLevel.WARNING,
            encrypt_key=self._settings.encrypt_key or None,
            verification_token=self._settings.verification_token or None,
            transport=TransportConfig(
                kind=transport_kind or self._sdk_transport_kind(),
                auto_reconnect=True,
            ),
            policy=self._build_sdk_policy_config(self._settings),
            safety=self._build_sdk_safety_config(self._settings),
            inbound=InboundConfig(
                # Deliver all reactions and verify target-message authorship
                # in Hermes. SDK "own" depends on this process' sent-message
                # cache, so it drops reactions after gateway restarts.
                reaction_notifications="all",
            ),
            outbound=self._build_sdk_outbound_config(),
            dedup_store=dedup_store,
        )
        self._install_comment_dispatch_override(channel)
        if register_handlers:
            self._register_sdk_channel_handlers(channel)
        return channel

    def _build_send_only_channel(self) -> Any:
        """Build a FeishuChannel for one-shot outbound sends.

        This intentionally does not call ``start()`` / ``connect()`` and does
        not acquire the Feishu app lock. The send_message tool only needs the
        SDK outbound sender/upload helpers, not an inbound WS/webhook runtime.
        """
        from lark_oapi import LogLevel
        from lark_oapi.channel import FeishuChannel

        return FeishuChannel(
            app_id=self._settings.app_id,
            app_secret=self._settings.app_secret,
            domain=self._resolve_sdk_domain(),
            log_level=LogLevel.WARNING,
            outbound=self._build_sdk_outbound_config(),
        )

    def _isolate_sdk_websocket_loop(self) -> None:
        """Ensure the SDK websocket client owns an executor-safe event loop.

        ``lark_oapi.ws.client`` keeps a module-level ``loop`` and later drives
        it with ``run_until_complete()`` from the SDK's blocking websocket
        starter. If that module was imported while the gateway's main asyncio
        loop was already running, the cached loop points at the running gateway
        loop and websocket startup fails with "This event loop is already
        running".
        """
        try:
            import lark_oapi.ws.client as ws_client
        except Exception:
            return

        sdk_loop = getattr(ws_client, "loop", None)
        try:
            loop_needs_replacement = (
                sdk_loop is None
                or sdk_loop.is_closed()
                or sdk_loop.is_running()
            )
        except Exception:
            loop_needs_replacement = True

        if loop_needs_replacement:
            ws_client.loop = asyncio.new_event_loop()

    async def _connect_websocket_once(self) -> None:
        """Start the SDK websocket channel and wait for transport readiness."""
        if self._channel is None:
            raise RuntimeError("FeishuChannel not constructed")

        self._isolate_sdk_websocket_loop()

        connect_until_ready = getattr(self._channel, "connect_until_ready", None)
        if callable(connect_until_ready):
            await connect_until_ready(timeout=30.0)
            return

        start_background = getattr(self._channel, "start_background", None)
        if callable(start_background):
            await start_background(timeout=30.0)
            return

        connect = getattr(self._channel, "connect", None)
        wait_ready = getattr(self._channel, "wait_ready", None)
        if callable(connect) and callable(wait_ready):
            await connect()
            await wait_ready(timeout=30.0)
            return

        raise RuntimeError(
            "lark-oapi channel SDK missing websocket lifecycle methods; "
            "expected connect_until_ready/start_background or connect/wait_ready"
        )

    async def _cleanup_failed_channel_start(self) -> None:
        """Best-effort cleanup between channel startup retries."""
        await self._stop_webhook_server()
        channel = self._channel
        if channel is not None:
            try:
                await channel.disconnect()
            except Exception:
                logger.debug(
                    "[Feishu] channel.disconnect failed after startup error",
                    exc_info=True,
                )

    async def _cleanup_failed_websocket_attempt(self) -> None:
        """Best-effort cleanup between websocket startup retries."""
        await self._cleanup_failed_channel_start()

    async def _connect_websocket_with_retry(self) -> None:
        """Run websocket startup with the historical connect retry profile."""
        last_error: Optional[Exception] = None
        for attempt in range(1, _FEISHU_CONNECT_ATTEMPTS + 1):
            try:
                if self._channel is None:
                    self._channel = self._build_sdk_channel(
                        transport_kind="ws",
                        dedup_store=self._dedup_store,
                    )
                await self._connect_websocket_once()
                return
            except Exception as exc:
                last_error = exc
                await self._cleanup_failed_websocket_attempt()
                self._channel = None
                if attempt >= _FEISHU_CONNECT_ATTEMPTS:
                    break
                wait_seconds = 2 ** (attempt - 1)
                logger.warning(
                    "[Feishu] channel.connect / wait_ready attempt %d/%d failed; retrying in %ds: %s",
                    attempt,
                    _FEISHU_CONNECT_ATTEMPTS,
                    wait_seconds,
                    exc,
                )
                await asyncio.sleep(wait_seconds)

        raise last_error or TimeoutError(
            "Feishu WS did not establish connection within 30s"
        )

    async def connect(self) -> bool:
        """Connect to Feishu/Lark."""
        if not FEISHU_AVAILABLE:
            logger.error("[Feishu] lark-oapi not installed")
            return False
        if not self._app_id or not self._app_secret:
            logger.error("[Feishu] FEISHU_APP_ID or FEISHU_APP_SECRET not set")
            return False
        if self._connection_mode not in {"websocket", "webhook"}:
            logger.error(
                "[Feishu] Unsupported FEISHU_CONNECTION_MODE=%s. Supported modes: websocket, webhook.",
                self._connection_mode,
            )
            return False

        try:
            self._app_lock_identity = self._app_id
            acquired, existing = acquire_scoped_lock(
                _FEISHU_APP_LOCK_SCOPE,
                self._app_lock_identity,
                metadata={"platform": self.platform.value},
            )
            if not acquired:
                owner_pid = existing.get("pid") if isinstance(existing, dict) else None
                message = (
                    "Another local Hermes gateway is already using this Feishu app_id"
                    + (f" (PID {owner_pid})." if owner_pid else ".")
                    + " Stop the other gateway before starting a second Feishu websocket client."
                )
                logger.error("[Feishu] %s", message)
                self._set_fatal_error("feishu_app_lock", message, retryable=False)
                return False

            # SDK TransportConfig accepts only "ws" | "webhook"; Hermes'
            # historical ``connection_mode`` uses "websocket", so we map it
            # here. WS tuning fields (ping/reconnect intervals) are not
            # exposed by SDK TransportConfig.
            _hermes_mode = self._settings.connection_mode or "websocket"
            _sdk_kind = "webhook" if _hermes_mode == "webhook" else "ws"

            # Persistent dedup store — preserves dedup state across restarts
            # (without it the SDK falls back to InMemoryDedupStore).
            self._dedup_store = JsonFileDedupStore(
                path=self._dedup_state_path,
                max_entries=(
                    self._settings.dedup_cache_size
                    or _FEISHU_DEDUP_DEFAULT_CACHE_SIZE
                ),
                default_ttl_seconds=_FEISHU_DEDUP_DEFAULT_TTL_SECONDS,
                account_id=self._app_id,
            )

            self._loop = asyncio.get_running_loop()

            if _hermes_mode == "websocket":
                try:
                    await self._connect_websocket_with_retry()
                except Exception as e:
                    logger.error(
                        "[Feishu] channel.connect / wait_ready failed: %s",
                        e,
                        exc_info=True,
                    )
                    self._set_fatal_error(
                        "feishu_channel_connect",
                        f"channel.connect failed: {e}",
                        retryable=True,
                    )
                    self._channel = None
                    if self._dedup_store is not None:
                        try:
                            self._dedup_store.flush()
                        except Exception:
                            logger.warning(
                                "[Feishu] dedup_store.flush failed during failed startup",
                                exc_info=True,
                            )
                    self._dedup_store = None
                    await self._release_app_lock()
                    return False
            else:
                self._channel = self._build_sdk_channel(
                    transport_kind="webhook",
                    dedup_store=self._dedup_store,
                )
                await self._connect_with_retry()

            self._mark_connected()
            logger.info("[Feishu] Connected in %s mode (%s)", self._connection_mode, self._domain_name)
            return True
        except Exception as exc:
            await self._cleanup_failed_channel_start()
            self._channel = None
            await self._release_app_lock()
            message = f"Feishu startup failed: {exc}"
            self._set_fatal_error("feishu_connect_error", message, retryable=True)
            logger.error("[Feishu] Failed to connect: %s", exc, exc_info=True)
            return False

    async def disconnect(self) -> None:
        """Disconnect from Feishu/Lark."""
        self._running = False
        self._disable_websocket_auto_reconnect()
        await self._stop_webhook_server()
        self._loop = None

        try:
            if self._channel is not None:
                try:
                    await self._channel.disconnect()
                except Exception as e:
                    logger.warning("[Feishu] channel.disconnect raised: %s", e)

            # Flush pending dedup-store writes before releasing the app lock.
            # ``flush()`` is idempotent (no-op if already clean).
            if self._dedup_store is not None:
                try:
                    self._dedup_store.flush()
                except Exception:
                    logger.warning(
                        "[Feishu] dedup_store.flush failed during disconnect",
                        exc_info=True,
                    )
        finally:
            self._dedup_store = None
            self._channel = None
            await self._release_app_lock()
            self._mark_disconnected()

        logger.info("[Feishu] Disconnected")

    # =========================================================================
    # SDK channel event handlers
    # =========================================================================

    async def _dispatch_handle_message_on_adapter_loop(
        self,
        event: MessageEvent,
        *,
        label: str,
        context: str = "",
    ) -> None:
        """Dispatch Hermes message handling on the adapter-owned event loop.

        SDK channel callbacks run on the SDK background loop.  Hermes'
        BasePlatformAdapter lifecycle tracks background message-processing tasks
        from the adapter/gateway loop, so only the final handoff into
        handle_message crosses back to ``self._loop``.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        target_loop = self._loop or current_loop
        if target_loop is None:
            logger.warning(
                "[Feishu] Dropping %s event because no adapter loop is available",
                label,
            )
            return
        if bool(getattr(target_loop, "is_closed", lambda: False)()):
            logger.warning(
                "[Feishu] Dropping %s event because adapter loop is closed",
                label,
            )
            return
        if not bool(getattr(target_loop, "is_running", lambda: True)()):
            logger.warning(
                "[Feishu] Dropping %s event because adapter loop is not running",
                label,
            )
            return

        def _log_with_context(future: Any) -> None:
            try:
                exc = future.exception()
            except (asyncio.CancelledError, Exception):
                # Fallback to the static logger if exception() itself raises.
                self._log_background_failure(future)
                return
            if exc is not None:
                logger.error(
                    "[Feishu] handle_message raised for %s (%s): %s",
                    label, context or "<no-context>", exc,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        if current_loop is target_loop:
            # Same loop -- schedule as a task so the SDK callback caller is
            # not blocked on the Hermes handler. handle_message exceptions
            # are logged by _log_with_context with message-ID context.
            task = asyncio.ensure_future(self.handle_message(event))
            task.add_done_callback(_log_with_context)
            return

        # SDK callback runs on the SDK loop; hand off to the adapter loop
        # without blocking the SDK callback. Errors are surfaced via
        # _log_with_context (carries message-ID context).
        scheduled = self._submit_on_loop_with_callback(
            target_loop, self.handle_message(event), _log_with_context,
        )
        if not scheduled:
            logger.warning(
                "[Feishu] Dropped %s event (%s): failed to schedule on adapter loop",
                label, context or "<no-context>",
            )

    async def _on_sdk_message(self, msg: "InboundMessage") -> None:
        """SDK has applied: dedup, stale, policy (admin/blocklist/allowlist),
        require_mention, processing_lock, text_batch, media_batch, per-chat queue,
        drop_self_sent. Hermes only:
            1) Application-layer peer-bot admission gate (FEISHU_ALLOW_BOTS).
            2) SDK InboundMessage → Hermes MessageEvent (events_mapping)
            3) handle_message (BasePlatformAdapter main flow)
        """
        # Always-on self-echo guard + peer-bot admission gate. SDK
        # ``drop_self_sent`` already drops echoes when the bot identity is
        # known to the SDK; this layer fails closed if Hermes' own
        # ``_bot_open_id`` is unresolved (preventing feedback loops when a
        # fresh adapter has not yet hydrated identity).
        sender = getattr(msg, "sender", None)
        if sender is not None and bool(getattr(sender, "is_bot", False)):
            self_open_id = self._resolve_self_open_id()
            sender_open_id = str(getattr(sender, "open_id", "") or "")
            if not self_open_id:
                logger.debug(
                    "[Feishu] Drop peer-bot msg (own identity unresolved); fail-closed."
                )
                return
            if sender_open_id and sender_open_id == self_open_id:
                return  # self-echo
            if sender_open_id:
                self._known_peer_bot_open_ids.add(sender_open_id)
            mode = (self._allow_bots or "none").lower()
            if mode not in ("mentions", "all"):
                logger.debug(
                    "[Feishu] Drop peer-bot msg (allow_bots=%s).", self._allow_bots,
                )
                return
            if mode == "mentions":
                mentions_self = bool(getattr(msg, "mentioned_all", False)) or any(
                    str(getattr(m, "open_id", "") or "") == self_open_id
                    for m in (getattr(msg, "mentions", None) or [])
                )
                if not mentions_self:
                    logger.debug(
                        "[Feishu] Drop peer-bot msg (allow_bots=mentions, "
                        "but bot was not mentioned)."
                    )
                    return
        elif self._needs_sdk_user_policy_fallback(msg):
            if not self._allow_group_message(
                getattr(msg, "sender", None),
                getattr(getattr(msg, "conversation", None), "chat_id", "") or "",
                is_bot=False,
            ):
                logger.debug("[Feishu] dropping inbound event: group_policy_rejected")
                return

        conversation = getattr(msg, "conversation", None)
        chat_type = str(getattr(conversation, "chat_type", "") or "")
        chat_id = str(getattr(conversation, "chat_id", "") or "")
        if chat_type in {"group", "topic"} and self._require_mention_for(chat_id):
            if not self._sdk_message_mentions_self(msg):
                logger.debug("[Feishu] Drop group msg (bot was not mentioned).")
                return

        try:
            event = await to_message_event(msg, channel=self._channel)
        except Exception as e:
            logger.error(
                "[Feishu] events_mapping.to_message_event failed for msg %s: %s",
                getattr(msg, "id", "<unknown>"),
                e,
                exc_info=True,
            )
            return
        if (
            event.message_type == MessageType.TEXT
            and not (event.text or "").strip()
            and not event.media_urls
        ):
            logger.debug(
                "[Feishu] Ignoring empty text message id=%s",
                getattr(msg, "id", "<unknown>"),
            )
            return

        await self._dispatch_handle_message_on_adapter_loop(
            event,
            label="message",
            context=f"msg {getattr(msg, 'id', '<unknown>')}",
        )

    def _needs_sdk_user_policy_fallback(self, msg: Any) -> bool:
        """Return True when SDK policy lacks multi-identity user filtering."""
        sdk_supports_sender_fields = getattr(
            self, "_sdk_policy_supports_sender_identity_fields", True
        )
        peer_bots_need_hermes_gate = (self._allow_bots or "none") in {"mentions", "all"}
        global_group_policy = self._default_group_policy or self._group_policy
        sdk_allow_from_disabled_for_peer_bots = (
            peer_bots_need_hermes_gate
            and global_group_policy == "allowlist"
            and bool(self._allowed_group_users)
        )
        conversation = getattr(msg, "conversation", None)
        chat_id = getattr(conversation, "chat_id", "") or ""
        rule = self._group_rules.get(chat_id) if chat_id else None
        sdk_group_rule_filter_disabled_for_peer_bots = bool(
            peer_bots_need_hermes_gate
            and rule
            and rule.policy in {"allowlist", "blacklist"}
        )
        if getattr(conversation, "chat_type", "") not in ("group", "topic"):
            return False

        if getattr(self, "_allow_all_users", False):
            return False

        if global_group_policy == "allowlist" and not self._allowed_group_users:
            return True

        if sdk_supports_sender_fields and not sdk_allow_from_disabled_for_peer_bots:
            if not sdk_group_rule_filter_disabled_for_peer_bots:
                return False
        if global_group_policy == "allowlist":
            return bool(self._allowed_group_users)
        return bool(rule and rule.policy in {"allowlist", "blacklist", "admin_only", "disabled"})

    async def _on_sdk_reject(self, evt) -> None:
        """Bridge SDK RejectEvent to Hermes metrics.

        SDK reject reasons (lark_oapi/channel/safety/types.py:26-42):
          stale | duplicate | lock_contention | self_sent
          | policy_dm_disabled | policy_group_disabled
          | policy_dm_not_in_allowlist | policy_group_not_in_allowlist
          | policy_blocklist | policy_admin_only
          | policy_no_mention | policy_mention_all_blocked | policy_sender_not_allowed

        Reject metrics emit the SDK literal names directly; the Hermes
        legacy aliases (``group_not_allowed`` / ``sender_not_allowed`` /
        ``no_mention`` / ``dm_disabled`` / etc.) are no longer used.
        """
        reason = getattr(evt, "reason", "<unknown>")
        msg_id = getattr(evt, "message_id", "")
        chat_id = getattr(evt, "chat_id", "")
        sender_id = getattr(evt, "sender_id", "")

        logger.info(
            "[Feishu] message rejected: reason=%s message_id=%s chat_id=%s sender_id=%s",
            reason, msg_id, chat_id, sender_id,
        )
        # Best-effort metric emission. Hermes does not currently expose a
        # central inbound-drop recorder on the adapter; if a future refactor
        # adds ``_record_inbound_drop(reason=..., chat_id=..., sender_id=...)``
        # this handler will pick it up automatically without further changes.
        try:
            recorder = getattr(self, "_record_inbound_drop", None)
            if recorder:
                recorder(reason=reason, chat_id=chat_id, sender_id=sender_id)
        except Exception as e:
            logger.warning("[Feishu] _on_sdk_reject metric emit failed: %s", e)

    async def _on_sdk_error(self, err) -> None:
        """Bridge SDK error events to Hermes fatal-error tracking.

        Default classification is lenient (``retryable=True``) so the SDK's
        auto_reconnect can recover transient failures. Per-exception-type
        refinement is a future hardening pass.
        """
        logger.error("[Feishu] SDK channel error: %s", err, exc_info=True)
        self._set_fatal_error(
            "feishu_channel_error",
            f"SDK channel error: {err}",
            retryable=True,
        )

    async def _on_sdk_reconnecting(self) -> None:
        """SDK WS reconnecting — bridge to logging + metrics if Hermes has them."""
        logger.warning("[Feishu] SDK channel WS reconnecting...")
        # Optional anomaly tracker hook; falls back to log-only when absent.
        tracker = getattr(self, "_record_ws_anomaly", None)
        if tracker:
            try:
                tracker(kind="reconnecting")
            except Exception:
                logger.debug("[Feishu] _record_ws_anomaly call failed", exc_info=True)

    async def _on_sdk_reconnected(self) -> None:
        """SDK WS reconnected — clear any anomaly state, log."""
        logger.info("[Feishu] SDK channel WS reconnected.")
        clearer = getattr(self, "_clear_ws_anomaly", None)
        if clearer:
            try:
                clearer()
            except Exception:
                logger.debug("[Feishu] _clear_ws_anomaly call failed", exc_info=True)

    async def _on_sdk_card_action(self, action: "CardActionEvent") -> None:
        """SDK has applied: dedup (event_id=card:{message_id}:{op}:{action_id}),
        lock + serial-by-chat (push_action). Hermes only:
            1) value.hermes_action present → approval flow
               (resolve_gateway_approval + channel.update_card)
            2) otherwise → synthesize a COMMAND MessageEvent and dispatch
               via BasePlatformAdapter.handle_message
        """
        payload = action.action if action.action else None
        value = payload.value if payload is not None else None
        hermes_action = (
            value.get("hermes_action")
            if isinstance(value, dict) else None
        )
        update_prompt_action = (
            value.get("hermes_update_prompt_action")
            if isinstance(value, dict) else None
        )

        if hermes_action:
            await self._resolve_approval_via_sdk(action, hermes_action)
            return
        if update_prompt_action:
            await self._resolve_update_prompt_via_sdk(action, update_prompt_action)
            return

        # Non-approval card action: synthesize COMMAND.
        try:
            event = _to_command_event_from_card_action(action, channel=self._channel)
            # Refine chat_type via chat_info (P2P cards also flow here).
            try:
                chat_info = await self.get_chat_info(action.chat_id) if action.chat_id else {}
                event.source = SessionSource(
                    platform=event.source.platform,
                    chat_id=event.source.chat_id,
                    chat_name=chat_info.get("name") or event.source.chat_id or "Feishu Chat",
                    chat_type=self._resolve_source_chat_type(
                        chat_info=chat_info,
                        event_chat_type=chat_info.get("raw_type") or event.source.chat_type,
                    ),
                    user_id=event.source.user_id,
                    user_id_alt=event.source.user_id_alt,
                    user_name=event.source.user_name,
                    thread_id=event.source.thread_id,
                )
            except Exception as e:
                logger.debug("[Feishu] _on_sdk_card_action: chat_type refinement failed: %s", e)
        except Exception as e:
            logger.error(
                "[Feishu] _on_sdk_card_action event mapping failed: %s",
                e, exc_info=True,
            )
            return
        await self._dispatch_handle_message_on_adapter_loop(
            event,
            label="cardAction",
            context=f"card {getattr(action, 'message_id', '<unknown>')}",
        )

    async def _resolve_approval_via_sdk(
        self, action: "CardActionEvent", hermes_action: str,
    ) -> None:
        """Delegate to ``approvals._resolve_approval_via_sdk_impl``."""
        from gateway.platforms.feishu.approvals import _resolve_approval_via_sdk_impl
        await _resolve_approval_via_sdk_impl(self, action, hermes_action)

    async def _resolve_update_prompt_via_sdk(
        self, action: "CardActionEvent", answer: str,
    ) -> None:
        """Delegate to ``approvals._resolve_update_prompt_via_sdk_impl``."""
        from gateway.platforms.feishu.approvals import _resolve_update_prompt_via_sdk_impl
        await _resolve_update_prompt_via_sdk_impl(self, action, answer)

    @staticmethod
    def _sender_value(sender: Any, key: str) -> Any:
        if isinstance(sender, dict):
            return sender.get(key)
        return getattr(sender, key, None)

    @staticmethod
    def _sender_id_part(sender_id: Any, key: str) -> str:
        if isinstance(sender_id, dict):
            return str(sender_id.get(key) or "")
        return str(getattr(sender_id, key, "") or "")

    def _message_payload_sender_is_self(self, payload: Any) -> bool:
        sender = getattr(payload, "sender", None)
        if isinstance(payload, dict):
            sender = payload.get("sender") or sender
        if sender is None:
            return False

        sender_type = str(self._sender_value(sender, "sender_type") or "").lower()
        sender_id = (
            self._sender_value(sender, "sender_id")
            or self._sender_value(sender, "id")
        )
        id_type = str(self._sender_value(sender, "id_type") or "").lower()

        open_id = str(self._sender_value(sender, "open_id") or "")
        user_id = str(self._sender_value(sender, "user_id") or "")
        if sender_id is not None and not isinstance(sender_id, str):
            open_id = open_id or self._sender_id_part(sender_id, "open_id")
            user_id = user_id or self._sender_id_part(sender_id, "user_id")
        elif id_type == "open_id":
            open_id = str(sender_id or "")
        elif id_type == "user_id":
            user_id = str(sender_id or "")

        self_ids = {
            self._resolve_self_open_id(),
            getattr(self, "_bot_user_id", ""),
        } - {"", None}
        if self_ids and ({open_id, user_id} & self_ids):
            return True

        # im.v1.message.get reports messages authored by this bot as
        # sender_type=app with sender.id=<app_id>, not the bot open_id.
        sender_id_text = sender_id if isinstance(sender_id, str) else ""
        return bool(
            sender_type in {"app", "bot"}
            and self._app_id
            and sender_id_text == self._app_id
        )

    @staticmethod
    def _message_payload_value(payload: Any, key: str) -> Any:
        if isinstance(payload, dict):
            return payload.get(key)
        return getattr(payload, key, None)

    def _message_payload_chat_context(self, payload: Any) -> Dict[str, str]:
        chat_id = str(
            self._message_payload_value(payload, "chat_id")
            or self._message_payload_value(payload, "open_chat_id")
            or self._message_payload_value(payload, "share_chat_id")
            or ""
        )
        chat_type = str(
            self._message_payload_value(payload, "chat_type")
            or self._message_payload_value(payload, "raw_type")
            or ""
        )
        return {"chat_id": chat_id, "chat_type": chat_type}

    async def _reaction_target_context(self, message_id: str) -> Optional[Dict[str, str]]:
        if not self._channel or not message_id:
            return None
        fetch_message = getattr(self._channel, "fetch_message", None)
        if not callable(fetch_message):
            return None
        try:
            target = fetch_message(message_id)
            if inspect.isawaitable(target):
                target = await target
        except Exception:
            logger.debug(
                "[Feishu] failed to fetch reaction target message %s",
                message_id,
                exc_info=True,
            )
            return None

        candidates: list[Any] = [target]
        if isinstance(target, dict):
            data = target.get("data")
            if isinstance(data, dict):
                items = data.get("items")
                if isinstance(items, list):
                    candidates.extend(items)
                else:
                    candidates.append(data)

        for candidate in candidates:
            if self._message_payload_sender_is_self(candidate):
                return self._message_payload_chat_context(candidate)
        return None

    async def _reaction_targets_self_message(self, message_id: str) -> bool:
        return await self._reaction_target_context(message_id) is not None

    @classmethod
    def _payload_marks_bot_identity(cls, payload: Any) -> bool:
        is_bot = cls._message_payload_value(payload, "is_bot")
        if isinstance(is_bot, bool):
            return is_bot
        if str(is_bot or "").strip().lower() in {"true", "1", "yes"}:
            return True

        for key in ("sender_type", "operator_type", "user_type", "actor_type", "type"):
            value = str(cls._message_payload_value(payload, key) or "").strip().lower()
            if value in {"app", "bot", "robot", "application"}:
                return True

        id_type = str(cls._message_payload_value(payload, "id_type") or "").strip().lower()
        return id_type in {"app_id", "bot_id"}

    @classmethod
    def _reaction_operator_open_id(cls, evt: Any) -> str:
        operator = getattr(evt, "operator", None)
        open_id = str(getattr(operator, "open_id", "") or "")
        if open_id:
            return open_id

        raw = getattr(evt, "raw", None) or {}
        raw_event = cls._message_payload_value(raw, "event")
        candidates = [raw, raw_event]
        for container in candidates:
            if container is None:
                continue
            for key in ("operator", "operator_id", "user", "user_id", "actor", "actor_id"):
                identity = cls._message_payload_value(container, key)
                candidate_open_id = str(
                    cls._message_payload_value(identity, "open_id") or ""
                )
                if candidate_open_id:
                    return candidate_open_id
        return ""

    @classmethod
    def _reaction_operator_has_bot_marker(cls, evt: Any) -> bool:
        operator = getattr(evt, "operator", None)
        if cls._payload_marks_bot_identity(operator):
            return True

        raw = getattr(evt, "raw", None) or {}
        raw_event = cls._message_payload_value(raw, "event")
        candidates = [raw, raw_event]
        for container in candidates:
            if container is None:
                continue
            if cls._reaction_container_marks_operator_bot(container):
                return True
            for key in ("operator", "operator_id", "user", "user_id", "actor", "actor_id"):
                identity = cls._message_payload_value(container, key)
                if cls._payload_marks_bot_identity(identity):
                    return True
        return False

    @classmethod
    def _reaction_container_marks_operator_bot(cls, payload: Any) -> bool:
        is_bot = cls._message_payload_value(payload, "is_bot")
        if isinstance(is_bot, bool):
            return is_bot
        if str(is_bot or "").strip().lower() in {"true", "1", "yes"}:
            return True

        for key in ("operator_type", "user_type", "actor_type"):
            value = str(cls._message_payload_value(payload, key) or "").strip().lower()
            if value in {"app", "bot", "robot", "application"}:
                return True
        return False

    def _reaction_operator_is_peer_bot(self, evt: Any) -> bool:
        operator_open_id = self._reaction_operator_open_id(evt)
        self_open_id = self._resolve_self_open_id()
        if operator_open_id and self_open_id and operator_open_id == self_open_id:
            return False
        if operator_open_id and operator_open_id in self._known_peer_bot_open_ids:
            return True
        if not self._reaction_operator_has_bot_marker(evt):
            return False
        if operator_open_id:
            self._known_peer_bot_open_ids.add(operator_open_id)
        return True

    async def _on_sdk_reaction(self, evt: "ReactionEvent") -> None:
        """Route user reactions on bot-authored messages as synthetic text."""
        operator_is_peer_bot = self._reaction_operator_is_peer_bot(evt)
        if operator_is_peer_bot and (self._allow_bots or "none").lower() != "all":
            logger.debug(
                "[Feishu] Drop peer-bot reaction (allow_bots=%s).", self._allow_bots,
            )
            return

        target_context = await self._reaction_target_context(
            getattr(evt, "message_id", "") or ""
        )
        if target_context is None:
            logger.debug(
                "[Feishu] Drop reaction for non-self target msg %s",
                getattr(evt, "message_id", ""),
            )
            return
        try:
            event = await _to_text_event_from_reaction(
                evt,
                channel=self._channel,
                bot_open_id_fallback=getattr(self, "_bot_open_id", "") or "",
                chat_id_fallback=target_context.get("chat_id", ""),
                chat_type_fallback=target_context.get("chat_type", ""),
                operator_is_bot=operator_is_peer_bot,
            )
        except Exception as e:
            logger.error(
                "[Feishu] _to_text_event_from_reaction failed for msg %s: %s",
                getattr(evt, "message_id", "<unknown>"), e, exc_info=True,
            )
            return
        if event is None:
            return
        await self._dispatch_handle_message_on_adapter_loop(
            event,
            label="reaction",
            context=f"reaction {getattr(evt, 'message_id', '<unknown>')}",
        )

    async def _on_sdk_comment(self, evt: "CommentEvent") -> None:
        """SDK has applied: dedup (event_id=comment:{file_token}:{comment_id})
        + lock + serial-by-file_token via push_action. Hermes delegates to
        ``comments.handle_drive_comment_event`` with a legacy-shaped envelope
        so the existing comment LLM pipeline keeps working unchanged.
        """
        if self._channel is None:
            logger.warning("[Feishu] _on_sdk_comment: channel is None, dropping event")
            return

        try:
            from gateway.platforms.feishu.comments import handle_drive_comment_event
        except Exception as e:
            logger.error("[Feishu] comments import failed: %s", e, exc_info=True)
            return

        legacy_data = _sdk_comment_to_legacy_dict(evt)

        # FeishuChannel exposes the underlying lark_oapi.Client via its public
        # ``client`` property (channel.py:387); fall back to ``self._client``
        # for webhook/non-SDK code paths.
        client = getattr(self._channel, "client", None) or getattr(self, "_client", None)
        if client is None:
            logger.warning(
                "[Feishu] _on_sdk_comment: no underlying lark_oapi.Client available; dropping"
            )
            return

        bot_identity = getattr(self._channel, "bot_identity", None)
        self_open_id = (
            getattr(bot_identity, "open_id", "") if bot_identity else ""
        ) or (getattr(self, "_bot_open_id", "") or "")

        try:
            await handle_drive_comment_event(
                client, legacy_data, self_open_id=self_open_id,
            )
        except Exception as e:
            logger.error(
                "[Feishu] handle_drive_comment_event failed for file %s comment %s: %s",
                evt.file_token, evt.comment_id, e, exc_info=True,
            )

    async def _on_sdk_bot_added(self, evt: "BotAddedEvent") -> None:
        """Bot was added to a chat. Mirror legacy _on_bot_added_to_chat behavior:
        log + invalidate chat_info_cache.
        """
        chat_id = str(evt.chat_id or "")
        operator_open_id = evt.operator.open_id if evt.operator else ""
        logger.info(
            "[Feishu] Bot added to chat: %s (operator=%s)",
            chat_id, operator_open_id,
        )
        if chat_id and hasattr(self, "_chat_info_cache"):
            try:
                self._chat_info_cache.pop(chat_id, None)
            except Exception:
                pass

    async def _on_sdk_bot_leave(self, evt: "BotLeaveEvent") -> None:
        """Bot was removed from a chat. Mirror legacy _on_bot_removed_from_chat
        behavior: log + invalidate chat_info_cache.
        """
        chat_id = str(evt.chat_id or "")
        operator_open_id = evt.operator.open_id if evt.operator else ""
        logger.info(
            "[Feishu] Bot removed from chat: %s (operator=%s)",
            chat_id, operator_open_id,
        )
        if chat_id and hasattr(self, "_chat_info_cache"):
            try:
                self._chat_info_cache.pop(chat_id, None)
            except Exception:
                pass

    def _disable_websocket_auto_reconnect(self) -> None:
        if self._ws_client is None:
            return
        try:
            setattr(self._ws_client, "_auto_reconnect", False)
        except Exception:
            pass
        finally:
            self._ws_client = None

    async def _stop_webhook_server(self) -> None:
        if self._webhook_runner is None:
            return
        runner = self._webhook_runner
        try:
            await runner.cleanup()
        finally:
            try:
                from gateway.platforms.feishu.webhook_guard import clear_webhook_handler

                clear_webhook_handler(runner)
            except Exception:
                pass
            self._webhook_runner = None
            self._webhook_handler = None

    # =========================================================================
    # Outbound — send / edit / send_image / send_voice / …
    # =========================================================================

    @staticmethod
    def _build_reply_opts(
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        effective_reply_to = reply_to
        if not effective_reply_to and metadata and metadata.get("thread_id"):
            effective_reply_to = FeishuAdapter._thread_reply_anchor(None, metadata)
        if not effective_reply_to:
            return None
        opts: Dict[str, Any] = {"reply_to": effective_reply_to}
        if (metadata or {}).get("thread_id"):
            opts["reply_in_thread"] = True
            opts["reply_target_gone"] = "fail"
        return opts

    @classmethod
    def _target_and_send_opts(
        cls,
        chat_id: str,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Return SDK send target/options, preserving Feishu topic routing."""
        opts = cls._build_reply_opts(reply_to, metadata)
        if opts is not None:
            return chat_id, opts
        thread_id = str((metadata or {}).get("thread_id") or "").strip()
        if thread_id:
            return thread_id, {"receive_id_type": "thread_id"}
        return chat_id, None

    @staticmethod
    def _thread_reply_anchor(
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if reply_to:
            return reply_to
        if not metadata:
            return None
        return (
            metadata.get("reply_to_message_id")
            or metadata.get("reply_to")
            or metadata.get("root_message_id")
            or None
        )

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a Feishu message via FeishuChannel SDK.

        Delegates to the SDK with ``tag_md_mode='native'`` configured at the
        channel level so markdown renders via the Feishu client's native
        markdown parser (tag:md), preserving H1/H2/H3 size hierarchy,
        blockquote rendering, and native list controls.

        SDK provides retry / SSRF / format-error fallback / chunking. Hermes
        still pre-chunks via ``truncate_message`` because ``format_message``
        may produce content longer than the SDK's ``text_chunk_limit``;
        SDK chunking is fence-aware but doesn't know about Hermes'
        MAX_MESSAGE_LENGTH semantics.
        """
        if not self._channel:
            return SendResult(success=False, error="Not connected")

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
        thread_reply_anchor = self._thread_reply_anchor(reply_to, metadata)
        preserve_thread_anchor = bool((metadata or {}).get("thread_id") and thread_reply_anchor)

        last_result: Optional[SendResult] = None
        for chunk in chunks:
            for subchunk, is_table in _split_feishu_table_chunks(chunk):
                # Markdown tables don't render inside post ``tag:md`` nodes on
                # the Feishu client; ship only table blocks as plain text so
                # surrounding markdown keeps native styling.
                payload: Any = {"text": subchunk} if is_table else subchunk
                effective_reply_to = thread_reply_anchor if preserve_thread_anchor else reply_to
                try:
                    target_id, send_opts = self._target_and_send_opts(
                        chat_id, effective_reply_to, metadata,
                    )
                    sdk_result = await self._channel.send(
                        target_id,
                        payload,
                        opts=send_opts,
                    )
                    if sdk_result.success:
                        last_result = SendResult(success=True, message_id=sdk_result.message_id)
                    else:
                        err = sdk_result.error
                        last_result = SendResult(
                            success=False,
                            error=str(err) if err else "channel.send failed",
                        )
                except Exception as e:
                    logger.warning("[Feishu] channel.send raised: %s", e)
                    last_result = SendResult(success=False, error=str(e))

                if not preserve_thread_anchor:
                    # Multi-part plain replies should only attach the first
                    # delivered message to the triggering Feishu message.
                    reply_to = None

        return last_result or SendResult(success=False, error="No content to send")

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """Edit a previously sent Feishu text/post message.

        Delegates to ``channel.edit_message``; the SDK handles markdown
        rendering via ``MarkdownConverter(tag_md_mode='native')``, preserving
        Feishu native rendering across the streaming-edit lifecycle.
        """
        if not self._channel or not message_id:
            return SendResult(
                success=False,
                error="Not connected" if not self._channel else "Missing message_id",
            )

        formatted = self.format_message(content)
        truncated = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
        text = truncated[0] if truncated else formatted

        try:
            sdk_result = await self._channel.edit_message(message_id, text)
            if sdk_result.success:
                return SendResult(success=True, message_id=message_id)
            err = sdk_result.error
            return SendResult(
                success=False,
                error=str(err) if err else "edit failed",
            )
        except Exception as e:
            logger.warning("[Feishu] channel.edit_message raised: %s", e)
            return SendResult(success=False, error=str(e))

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Delegate to ``approvals._send_exec_approval_impl``."""
        from gateway.platforms.feishu.approvals import _send_exec_approval_impl
        return await _send_exec_approval_impl(
            self, chat_id, command, session_key,
            description=description, metadata=metadata,
        )

    async def send_update_prompt(
        self,
        chat_id: str,
        prompt: str,
        default: str = "",
        session_key: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Delegate to ``approvals._send_update_prompt_impl``."""
        from gateway.platforms.feishu.approvals import _send_update_prompt_impl
        return await _send_update_prompt_impl(
            self, chat_id, prompt, default=default,
            session_key=session_key, metadata=metadata,
        )

    @staticmethod
    def _build_resolved_approval_card(*, choice: str, user_name: str) -> Dict[str, Any]:
        """Thin staticmethod shim around ``approvals._build_resolved_approval_card``."""
        from gateway.platforms.feishu.approvals import _build_resolved_approval_card
        return _build_resolved_approval_card(choice=choice, user_name=user_name)

    @staticmethod
    def _validate_nonempty_file(file_path: str, label: str) -> Optional[SendResult]:
        if not os.path.exists(file_path):
            return SendResult(success=False, error=f"{label} not found: {file_path}")
        try:
            if os.path.getsize(file_path) <= 0:
                return SendResult(success=False, error=f"{label} is empty: {file_path}")
        except OSError as exc:
            return SendResult(success=False, error=f"Failed to access {label.lower()}: {exc}")
        return None

    def _probe_upload_duration_ms(self, file_path: str, file_type: str) -> Optional[int]:
        try:
            data = Path(file_path).read_bytes()
            if file_type == "mp4":
                return _parse_mp4_duration_ms(data)
            if file_type == "opus":
                return _parse_opus_duration_ms(data)
        except Exception as exc:
            logger.debug("[Feishu] Failed to probe media duration for %s: %s", file_path, exc)
        return None

    async def _upload_file_with_duration(
        self,
        file_path: str,
        *,
        file_type: str,
    ) -> Optional[str]:
        duration_ms = await asyncio.to_thread(
            self._probe_upload_duration_ms,
            file_path,
            file_type,
        )
        if not duration_ms:
            return None

        client = getattr(self._channel, "client", None) if self._channel else None
        client = client or self._client
        if client is None:
            return None

        try:
            from lark_oapi.api.im.v1.model.create_file_request import CreateFileRequest
            from lark_oapi.api.im.v1.model.create_file_request_body import CreateFileRequestBody

            name = os.path.basename(file_path) or "upload"
            with open(file_path, "rb") as fh:
                body = (
                    CreateFileRequestBody.builder()
                    .file_type(file_type)
                    .file_name(name)
                    .duration(duration_ms)
                    .file(fh)
                    .build()
                )
                req = CreateFileRequest.builder().request_body(body).build()
                create = getattr(client.im.v1.file, "acreate", None) or getattr(client.im.v1.file, "create")
                resp = create(req)
                if inspect.isawaitable(resp):
                    resp = await resp

            success = resp.success() if callable(getattr(resp, "success", None)) else getattr(resp, "success", False)
            if not success:
                logger.warning(
                    "[Feishu] duration-aware file upload failed: code=%s msg=%s",
                    getattr(resp, "code", None),
                    getattr(resp, "msg", ""),
                )
                return None

            data = getattr(resp, "data", None)
            return getattr(data, "file_key", None) if data is not None else None
        except Exception as exc:
            logger.warning("[Feishu] duration-aware file upload raised: %s", exc)
            return None

    async def _send_sdk_message(
        self,
        *,
        chat_id: str,
        message: Dict[str, Any],
        default_error: str,
        log_label: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raise_exceptions: bool = False,
    ) -> SendResult:
        """Send through the channel SDK and project its result to Hermes."""
        if not self._channel:
            return SendResult(success=False, error="Not connected")
        try:
            target_id, send_opts = self._target_and_send_opts(chat_id, reply_to, metadata)
            sdk_result = await self._channel.send(
                target_id,
                message,
                opts=send_opts,
            )
            if sdk_result.success:
                return SendResult(success=True, message_id=sdk_result.message_id)
            err = sdk_result.error
            return SendResult(success=False, error=str(err) if err else default_error)
        except Exception as exc:
            if raise_exceptions:
                raise
            logger.warning("[Feishu] %s raised: %s", log_label, exc)
            return SendResult(success=False, error=str(exc))

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send an audio file.

        Feishu does not natively render an "audio + caption" atomic message.
        Send the caption first, then send the audio as a native attachment.
        """
        if not self._channel:
            return SendResult(success=False, error="Not connected")
        invalid_file = self._validate_nonempty_file(audio_path, "Audio")
        if invalid_file:
            return invalid_file
        audio_source = await self._upload_file_with_duration(audio_path, file_type="opus") or audio_path

        if not caption:
            return await self._send_sdk_message(
                chat_id=chat_id,
                message={"audio": {"source": audio_source}},
                default_error="send_voice failed",
                log_label="channel.send(audio)",
                reply_to=reply_to,
                metadata=metadata,
            )

        thread_reply_anchor = self._thread_reply_anchor(reply_to, metadata)
        preserve_thread_anchor = bool((metadata or {}).get("thread_id") and thread_reply_anchor)

        caption_result = await self.send(
            chat_id=chat_id,
            content=caption,
            reply_to=thread_reply_anchor or reply_to,
            metadata=metadata,
        )
        if not caption_result.success:
            return caption_result
        return await self.send_voice(
            chat_id=chat_id,
            audio_path=audio_path,
            caption=None,
            reply_to=thread_reply_anchor if preserve_thread_anchor else None,
            metadata=metadata,
        )

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
        """Send a generic file.

        Feishu does not natively render a "file + caption" atomic message.
        Send the caption first, then send the file as a native attachment.
        """
        if not self._channel:
            return SendResult(success=False, error="Not connected")
        if not os.path.exists(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")

        if not caption:
            file_payload: Dict[str, Any] = {"source": file_path}
            if file_name:
                file_payload["file_name"] = file_name
            return await self._send_sdk_message(
                chat_id=chat_id,
                message={"file": file_payload},
                default_error="send_document failed",
                log_label="channel.send(file)",
                reply_to=reply_to,
                metadata=metadata,
            )

        thread_reply_anchor = self._thread_reply_anchor(reply_to, metadata)
        preserve_thread_anchor = bool((metadata or {}).get("thread_id") and thread_reply_anchor)

        caption_result = await self.send(
            chat_id=chat_id,
            content=caption,
            reply_to=thread_reply_anchor or reply_to,
            metadata=metadata,
        )
        if not caption_result.success:
            return caption_result
        return await self.send_document(
            chat_id=chat_id,
            file_path=file_path,
            caption=None,
            file_name=file_name,
            reply_to=thread_reply_anchor if preserve_thread_anchor else None,
            metadata=metadata,
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a video file."""
        if not self._channel:
            return SendResult(success=False, error="Not connected")
        invalid_file = self._validate_nonempty_file(video_path, "Video")
        if invalid_file:
            return invalid_file

        video_source = await self._upload_file_with_duration(video_path, file_type="mp4") or video_path
        message: Dict[str, Any] = {"video": {"source": video_source}}
        if caption:
            message["caption"] = self.format_message(caption)

        return await self._send_sdk_message(
            chat_id=chat_id,
            message=message,
            default_error="send_video failed",
            log_label="channel.send(video)",
            reply_to=reply_to,
            metadata=metadata,
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
        """Send a local image file to Feishu via SDK."""
        if not self._channel:
            return SendResult(success=False, error="Not connected")
        if not os.path.exists(image_path):
            return SendResult(success=False, error=f"Image file not found: {image_path}")

        message: Dict[str, Any] = {"image": {"source": image_path}}
        if caption:
            message["caption"] = self.format_message(caption)

        return await self._send_sdk_message(
            chat_id=chat_id,
            message=message,
            default_error="send_image_file failed",
            log_label="channel.send(image_file)",
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Feishu bot API does not expose a typing indicator."""
        return None

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image by URL — SDK handles download + upload."""
        if not self._channel:
            return SendResult(success=False, error="Not connected")

        message: Dict[str, Any] = {"image": {"source": image_url}}
        if caption:
            message["caption"] = self.format_message(caption)

        try:
            return await self._send_sdk_message(
                chat_id=chat_id,
                message=message,
                default_error="send_image failed",
                log_label="channel.send(image)",
                reply_to=reply_to,
                metadata=metadata,
                raise_exceptions=True,
            )
        except Exception as e:
            logger.warning("[Feishu] channel.send(image) raised: %s", e)
            # Fallback: degrade to base class generic implementation
            return await super().send_image(
                chat_id=chat_id, image_url=image_url, caption=caption,
                reply_to=reply_to, metadata=metadata,
            )

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Animation degrades to document send.

        URL → tmp-file download is preserved (original Hermes behavior):
        SDK send_document accepts a local path, so we download first then
        delegate to send_document.
        """
        try:
            file_path, file_name = await self._download_remote_document(
                animation_url,
                default_ext=".gif",
                preferred_name="animation.gif",
            )
        except Exception as exc:
            logger.error("[Feishu] Failed to download animation %s: %s", animation_url, exc, exc_info=True)
            return await super().send_animation(
                chat_id=chat_id,
                animation_url=animation_url,
                caption=caption,
                reply_to=reply_to,
                metadata=metadata,
            )
        degraded_caption = f"[GIF downgraded to file]\n{caption}" if caption else "[GIF downgraded to file]"
        return await self.send_document(
            chat_id=chat_id,
            file_path=file_path,
            file_name=file_name,
            caption=degraded_caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return real chat metadata from Feishu when available."""
        fallback = {
            "chat_id": chat_id,
            "name": chat_id,
            "type": "dm",
        }
        if not self._channel:
            return fallback

        cached = self._chat_info_cache.get(chat_id)
        if cached is not None:
            return dict(cached)

        try:
            sdk_chat = await self._channel.get_chat_info(chat_id)
            if sdk_chat is None:
                return fallback
            raw_chat_type = str(getattr(sdk_chat, "chat_type", "") or "").strip().lower()
            info = {
                "chat_id": getattr(sdk_chat, "chat_id", None) or chat_id,
                "name": str(getattr(sdk_chat, "name", None) or chat_id),
                "type": self._map_chat_type(raw_chat_type),
                "raw_type": raw_chat_type or None,
            }
            self._chat_info_cache[chat_id] = info
            return dict(info)
        except Exception as e:
            logger.warning("[Feishu] channel.get_chat_info raised: %s", e)
            return fallback

    def format_message(self, content: str) -> str:
        """Feishu text messages are plain text by default."""
        return content.strip()

    @staticmethod
    def _loop_accepts_callbacks(loop: Any) -> bool:
        """Return True when the adapter loop can accept thread-safe submissions."""
        return loop is not None and not bool(getattr(loop, "is_closed", lambda: False)())

    def _submit_on_loop(self, loop: Any, coro: Any) -> bool:
        """Schedule background work on the adapter loop with shared failure logging."""
        from agent.async_utils import safe_schedule_threadsafe

        future = safe_schedule_threadsafe(
            coro,
            loop,
            logger=logger,
            log_message="[Feishu] Failed to schedule background callback work",
            log_level=logging.WARNING,
        )
        if future is None:
            return False
        future.add_done_callback(self._log_background_failure)
        return True

    def _submit_on_loop_with_callback(
        self, loop: Any, coro: Any, done_callback: Callable[[Any], None],
    ) -> bool:
        """Like _submit_on_loop but uses a caller-supplied done_callback
        instead of the default _log_background_failure (so callers can
        attach context like message_id to error logs)."""
        from agent.async_utils import safe_schedule_threadsafe

        future = safe_schedule_threadsafe(
            coro,
            loop,
            logger=logger,
            log_message="[Feishu] Failed to schedule background callback work",
            log_level=logging.WARNING,
        )
        if future is None:
            return False
        future.add_done_callback(done_callback)
        return True

    # =========================================================================
    # Processing status reactions
    # =========================================================================

    def _reactions_enabled(self) -> bool:
        return os.getenv("FEISHU_REACTIONS", "true").strip().lower() not in {"false", "0", "no"}

    async def _add_reaction(self, message_id: str, emoji_type: str) -> Optional[str]:
        """Return the reaction_id on success, else None. The id is needed later for deletion."""
        if not self._channel or not message_id or not emoji_type:
            return None
        try:
            sdk_result = await self._channel.add_reaction(message_id, emoji_type)
            if not sdk_result.success:
                return None
            # SDK SendResult.message_id is the message we reacted to, NOT the reaction_id.
            # The reaction_id is in raw["data"]["reaction_id"] per Feishu API response shape.
            raw = sdk_result.raw or {}
            data = raw.get("data") if isinstance(raw, dict) else {}
            if not isinstance(data, dict):
                return None
            return data.get("reaction_id") or data.get("id")
        except Exception as e:
            logger.warning("[Feishu] channel.add_reaction raised: %s", e)
            return None

    async def _remove_reaction(self, message_id: str, reaction_id: str) -> bool:
        if not self._channel or not message_id or not reaction_id:
            return False
        try:
            sdk_result = await self._channel.remove_reaction(message_id, reaction_id)
            return sdk_result.success
        except Exception as e:
            logger.warning("[Feishu] channel.remove_reaction raised: %s", e)
            return False

    def _remember_processing_reaction(self, message_id: str, reaction_id: str) -> None:
        cache = self._pending_processing_reactions
        cache[message_id] = reaction_id
        cache.move_to_end(message_id)
        while len(cache) > _FEISHU_PROCESSING_REACTION_CACHE_SIZE:
            cache.popitem(last=False)

    def _pop_processing_reaction(self, message_id: str) -> Optional[str]:
        return self._pending_processing_reactions.pop(message_id, None)

    async def on_processing_start(self, event: MessageEvent) -> None:
        if not self._reactions_enabled():
            return
        message_id = event.message_id
        if not message_id or message_id in self._pending_processing_reactions:
            return
        reaction_id = await self._add_reaction(message_id, _FEISHU_REACTION_IN_PROGRESS)
        if reaction_id:
            self._remember_processing_reaction(message_id, reaction_id)

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome
    ) -> None:
        if not self._reactions_enabled():
            return
        message_id = event.message_id
        if not message_id:
            return

        start_reaction_id = self._pending_processing_reactions.get(message_id)
        if start_reaction_id:
            if not await self._remove_reaction(message_id, start_reaction_id):
                # Don't stack a second badge on top of a Typing we couldn't
                # remove — UI would read as both "working" and "done/failed"
                # simultaneously. Keep the handle so LRU eventually evicts it.
                return
            self._pop_processing_reaction(message_id)

        if outcome is ProcessingOutcome.FAILURE:
            await self._add_reaction(message_id, _FEISHU_REACTION_FAILURE)

    # =========================================================================
    # Webhook server and security
    # =========================================================================

    def _hermes_log_webhook_anomaly(self, anomaly: WebhookAnomaly) -> None:
        """on_anomaly hook for webhook_guard.

        Synchronous callback invoked by ``_AnomalyTracker._fire_hook`` every
        time the anomaly counter increments. Hermes uses it for log-level
        surfacing only; deeper integrations (metrics counters, alerting) can
        be added later by extending this method without changing the
        webhook_guard public API.
        """
        if anomaly.count == 1:
            logger.info(
                "[Feishu] Webhook anomaly first seen: ip=%s status=%d",
                anomaly.remote_ip,
                anomaly.status_code,
            )
        # The threshold WARNING is already emitted by _AnomalyTracker itself;
        # here we only optionally surface a structured info-level entry.

    # =========================================================================
    # Inbound processing pipeline
    # =========================================================================

    async def _download_remote_image(self, image_url: str) -> str:
        ext = self._guess_remote_extension(image_url, default=".jpg")
        return await cache_image_from_url(image_url, ext=ext)

    async def _download_remote_document(
        self,
        file_url: str,
        *,
        default_ext: str,
        preferred_name: str,
    ) -> tuple[str, str]:
        from tools.url_safety import is_safe_url
        if not is_safe_url(file_url):
            raise ValueError(f"Blocked unsafe URL (SSRF protection): {file_url[:80]}")

        import httpx

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(
                file_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)",
                    "Accept": "*/*",
                },
            )
            response.raise_for_status()
            # Snapshot Content-Type and body while the client context is
            # still active so pooled connections fully release on exit.
            content_type_hdr = str(response.headers.get("Content-Type", ""))
            body = response.content
        filename = self._derive_remote_filename(
            file_url,
            content_type=content_type_hdr,
            default_name=preferred_name,
            default_ext=default_ext,
        )
        cached_path = cache_document_from_bytes(body, filename)
        return cached_path, filename

    @staticmethod
    def _guess_remote_extension(url: str, *, default: str) -> str:
        ext = Path((url or "").split("?", 1)[0]).suffix.lower()
        return ext if ext in (_IMAGE_EXTENSIONS | _AUDIO_EXTENSIONS | _VIDEO_EXTENSIONS | set(SUPPORTED_DOCUMENT_TYPES)) else default

    @staticmethod
    def _derive_remote_filename(file_url: str, *, content_type: str, default_name: str, default_ext: str) -> str:
        candidate = Path((file_url or "").split("?", 1)[0]).name or default_name
        ext = Path(candidate).suffix.lower()
        if not ext:
            guessed = mimetypes.guess_extension((content_type or "").split(";", 1)[0].strip().lower() or "") or default_ext
            candidate = f"{candidate}{guessed}"
        return candidate

    async def _handle_webhook_request(self, request: Any) -> Any:
        """Thin shim that delegates to webhook_guard's aiohttp handler.

        Kept as a method so contract tests in
        ``tests/gateway/feishu/test_webhook_security.py`` can invoke the
        webhook guard pipeline directly with a mock aiohttp request.
        Production aiohttp routing via ``webhook_guard.start_webhook_server``
        bypasses this method entirely — the real request lands on the
        handler stored in ``self._webhook_handler``.

        Tests that exercise the webhook pipeline before ``_connect_webhook``
        runs must populate ``self._webhook_handler`` themselves.
        """
        if self._webhook_handler is None:
            raise RuntimeError(
                "webhook server not started (self._webhook_handler is None); "
                "this method must be called only after _connect_webhook(), "
                "or with a test fixture that injects the handler"
            )
        return await self._webhook_handler(request)

    # =========================================================================
    # Message content extraction and resource download
    # =========================================================================

    @staticmethod
    def _map_chat_type(raw_chat_type: str) -> str:
        """Delegate; kept as staticmethod for callers that reach for FeishuAdapter._map_chat_type."""
        from gateway.platforms.feishu.types import map_chat_type
        return map_chat_type(raw_chat_type)

    @staticmethod
    def _resolve_source_chat_type(*, chat_info: Dict[str, Any], event_chat_type: str) -> str:
        from gateway.platforms.feishu.types import resolve_source_chat_type
        return resolve_source_chat_type(chat_info=chat_info, event_chat_type=event_chat_type)

    @staticmethod
    def _log_background_failure(future: Any) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("[Feishu] Background inbound processing failed")

    # =========================================================================
    # Deduplication is owned by ``JsonFileDedupStore`` injected into
    # ``FeishuChannel(dedup_store=...)``. SDK ``Deduper.check_and_mark``
    # drives the store from the inbound pipeline.
    # =========================================================================

    # =========================================================================
    # Connection internals — websocket / webhook setup
    # =========================================================================

    async def _connect_with_retry(self) -> None:
        for attempt in range(_FEISHU_CONNECT_ATTEMPTS):
            try:
                # Websocket mode is owned by the SDK channel via
                # connect()/disconnect(); only webhook mode reaches this loop.
                if self._channel is None:
                    self._channel = self._build_sdk_channel(
                        transport_kind="webhook",
                        dedup_store=self._dedup_store,
                    )
                await self._connect_webhook()
                return
            except Exception as exc:
                self._running = False
                self._disable_websocket_auto_reconnect()
                await self._cleanup_failed_channel_start()
                self._channel = None
                if attempt >= _FEISHU_CONNECT_ATTEMPTS - 1:
                    raise
                wait_seconds = 2 ** attempt
                logger.warning(
                    "[Feishu] Connect attempt %d/%d failed; retrying in %ds: %s",
                    attempt + 1,
                    _FEISHU_CONNECT_ATTEMPTS,
                    wait_seconds,
                    exc,
                )
                await asyncio.sleep(wait_seconds)

    async def _connect_webhook(self) -> None:
        """Start the webhook server via ``webhook_guard.start_webhook_server``.

        SDK responsibilities: signature verification, verification_token,
        URL-verification challenge, encrypted payload decryption, dispatcher
        + event routing, bot identity hydration (driven by
        ``channel.connect()``).
        Hermes responsibilities (in webhook_guard): aiohttp server lifecycle,
        rate-limit, anomaly tracker, body-size / Content-Type /
        body-read-timeout / JSON-parse guards.

        ``await self._channel.connect()`` must run before
        ``handle_webhook_request`` is invoked, otherwise the SDK raises
        ``NOT_CONNECTED`` (channel.py:463-467); connect() also hydrates
        the bot identity needed by the inbound pipeline.
        """
        from gateway.platforms.feishu.webhook_guard import (
            RateLimit,
            get_webhook_handler,
            start_webhook_server,
            WEBHOOK_AVAILABLE,
            RATE_WINDOW_SECONDS,
            RATE_LIMIT_MAX,
        )

        if not WEBHOOK_AVAILABLE:
            raise RuntimeError("aiohttp not installed; webhook mode unavailable")

        if self._channel is None:
            raise RuntimeError(
                "FeishuChannel not constructed; _connect_webhook called outside connect() lifecycle"
            )

        # SDK lifecycle drives bot identity hydration + dispatcher construction.
        # handle_webhook_request would raise NOT_CONNECTED otherwise.
        await self._channel.connect()

        self._webhook_runner = await start_webhook_server(
            host=self._webhook_host,
            port=self._webhook_port,
            path=self._webhook_path,
            app_id=self._app_id,
            handle_request=self._channel.handle_webhook_request,
            rate_limit=RateLimit(
                window_seconds=RATE_WINDOW_SECONDS,
                max_requests=RATE_LIMIT_MAX,
            ),
            on_anomaly=self._hermes_log_webhook_anomaly,
            encrypt_key=self._settings.encrypt_key,
            verification_token=self._settings.verification_token,
        )

        # webhook_guard keeps the partial-bound aiohttp handler in a module
        # registry so this thin shim can delegate without reaching into
        # aiohttp internals (this is the entry point contract tests use).
        self._webhook_handler = get_webhook_handler(self._webhook_runner)
        if self._webhook_handler is None:
            raise RuntimeError("webhook_guard runner missing registered handler")

    def _build_lark_client(self, domain: Any) -> Any:
        return (
            lark.Client.builder()
            .app_id(self._app_id)
            .app_secret(self._app_secret)
            .domain(domain)
            .log_level(lark.LogLevel.WARNING)
            .build()
        )

    async def _release_app_lock(self) -> None:
        if not self._app_lock_identity:
            return
        try:
            release_scoped_lock(_FEISHU_APP_LOCK_SCOPE, self._app_lock_identity)
        except Exception as exc:
            logger.warning("[Feishu] Failed to release app lock: %s", exc, exc_info=True)
        finally:
            self._app_lock_identity = None

    # =========================================================================
    # Inbound admission compatibility
    # =========================================================================
    #
    # Production inbound now enters through SDK typed events. The SDK owns
    # dedup, self-drop, allowlist/blacklist, mention gating, lock and queueing.
    # Hermes keeps only the policy helper required by the SDK fallback path
    # when the installed SDK cannot enforce every Hermes-specific predicate.

    def _require_mention_for(self, chat_id: str) -> bool:
        rule = self._group_rules.get(chat_id) if chat_id else None
        if rule and rule.require_mention is not None:
            return bool(rule.require_mention)
        return bool(self._require_mention)

    def _sdk_message_mentions_self(self, msg: Any) -> bool:
        if bool(getattr(msg, "mentioned_all", False)):
            return True

        bot_identity = getattr(getattr(self, "_channel", None), "bot_identity", None)
        bot_open_id = self._bot_open_id or str(
            getattr(bot_identity, "open_id", "") or ""
        )
        bot_user_id = self._bot_user_id or str(
            getattr(bot_identity, "user_id", "") or ""
        )
        bot_name = (
            self._bot_name
            or str(getattr(bot_identity, "name", "") or "")
            or str(getattr(bot_identity, "display_name", "") or "")
        )

        for mention in getattr(msg, "mentions", None) or []:
            mention_open_id = str(getattr(mention, "open_id", "") or "")
            mention_user_id = str(getattr(mention, "user_id", "") or "")
            mention_name = str(getattr(mention, "name", "") or "")
            if bot_open_id and mention_open_id == bot_open_id:
                return True
            if bot_user_id and mention_user_id == bot_user_id:
                return True
            if bot_name and mention_name == bot_name:
                return True

        content_text = str(getattr(msg, "content_text", "") or "")
        return "@_all" in content_text or "@all" in content_text

    def _allow_group_message(
        self,
        sender_id: Any,
        chat_id: str = "",
        *,
        is_bot: bool = False,
    ) -> bool:
        """Per-group policy gate for non-DM traffic.

        Bot-bypass semantics: admitted bots skip allowlist/blacklist
        (parallel human-scope filters), but channel-level locks (disabled,
        admin_only) and admin short-circuits still apply.
        """
        sender_open_id = getattr(sender_id, "open_id", None)
        sender_user_id = getattr(sender_id, "user_id", None)
        sender_union_id = getattr(sender_id, "union_id", None)
        sender_ids = {sender_open_id, sender_user_id, sender_union_id} - {None, ""}

        if sender_ids and self._admins and (sender_ids & self._admins):
            return True

        rule = self._group_rules.get(chat_id) if chat_id else None
        if rule:
            policy = rule.policy
            allowlist = rule.allowlist
            blacklist = rule.blacklist
        else:
            policy = self._default_group_policy or self._group_policy
            allowlist = self._allowed_group_users
            blacklist = set()

        # Channel locks apply to everyone; allowlist/blacklist only gate
        # humans (bots were already cleared upstream by FEISHU_ALLOW_BOTS).
        if policy == "disabled":
            return False
        if policy == "open":
            return True
        if policy == "admin_only":
            return False
        if is_bot:
            return True

        if policy == "allowlist":
            return bool(sender_ids and (sender_ids & allowlist))
        if policy == "blacklist":
            return bool(sender_ids and not (sender_ids & blacklist))

        return bool(sender_ids and (sender_ids & self._allowed_group_users))

    def _resolve_self_open_id(self) -> str:
        """Best-effort self open_id resolution for SDK-driven inbound paths.

        Order: explicit ``_bot_open_id`` (hydrated from /bot/v3/info or
        env) → SDK channel.bot_identity → empty string. Callers that need
        fail-closed semantics should treat empty as "self identity
        unresolved" and reject peer-bot traffic accordingly.
        """
        if self._bot_open_id:
            return self._bot_open_id
        bot_identity = getattr(getattr(self, "_channel", None), "bot_identity", None)
        if bot_identity:
            oid = str(getattr(bot_identity, "open_id", "") or "")
            if oid:
                return oid
        return ""

# QR onboarding lives in gateway/platforms/feishu/qr_register.py.
# Re-imported here so in-file references keep working; external callers
# pull these via gateway/platforms/feishu/__init__.py.
from gateway.platforms.feishu.qr_register import (  # noqa: F401, E402
    probe_bot,
    qr_register,
)
