"""Kimi (kimi.com / Moonshot AI) platform adapter.

Bridges two channels under one bot identity using a single `X-Kimi-Bot-Token`
credential:

- DM channel:    ``wss://www.kimi.com/api-claw/bots/agent-ws``
  Kimi speaks Zed Agent Client Protocol (ACP) JSON-RPC over WebSocket. The
  adapter responds to ACP handshake frames locally (``initialize``,
  ``session/new``) and converts ``session/prompt`` frames into MessageEvent
  for dispatch through the normal gateway pipeline.

- Group channel: ``https://www.kimi.com/api-ws/``
  Kimi exposes a Connect RPC (https://connectrpc.com) surface for group IM.
  The adapter opens a server-streamed ``Subscribe`` call (one HTTP/1.1
  long-poll POST per session; all-rooms firehose),
  translates inbound ``ChatMessageEvent`` payloads into MessageEvent, and
  sends outbound replies via unary ``SendMessage``. Both use
  ``application/json`` / ``application/connect+json`` content types — no
  protobuf bindings required on our side.

The adapter owns no subprocess. Messages flow directly into
``BasePlatformAdapter.handle_message`` via ``self._message_handler``.

Group-room participation requires OpenClaw runtime metadata headers on the IM
RPC path. Kimi Claw gathers these from ``openclaw --version``,
``openclaw skills list --json``, and ``openclaw plugins list --json``; this
adapter supplies conservative defaults and lets deployments override them via
``config.extra``.

References:
    - Connect protocol spec: https://connectrpc.com/docs/protocol
    - Zed ACP spec: https://github.com/zed-industries/agent-client-protocol
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import struct
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

try:
    import aiohttp  # type: ignore
    _AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - hermes core already depends on aiohttp
    aiohttp = None  # type: ignore
    _AIOHTTP_AVAILABLE = False

try:
    import websockets  # type: ignore
    from websockets.exceptions import ConnectionClosed  # type: ignore
    _WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None  # type: ignore
    ConnectionClosed = Exception  # type: ignore
    _WEBSOCKETS_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.session import build_session_key
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from hermes_constants import get_hermes_dir

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_BASE_URL = "https://www.kimi.com/api-ws"
_DEFAULT_KIMIAPI_HOST = "https://www.kimi.com/api-claw"
_DEFAULT_DM_WS_URL = "wss://www.kimi.com/api-claw/bots/agent-ws"
_IM_SERVICE = "kimi.gateway.im.v1.IMService"

# Connect protocol envelope flags (https://connectrpc.com/docs/protocol)
_CONNECT_FLAG_COMPRESSED = 0x01
_CONNECT_FLAG_END_STREAM = 0x02

# Kimi's DM channel uses a single sentinel sessionId across all WS lifecycles
_DM_SESSION_SENTINEL = "im:kimi:main"

# Chat-id prefix scheme used inside MessageEvent.source.chat_id
_CHATID_DM_PREFIX = "dm:"
_CHATID_ROOM_PREFIX = "room:"

# WS close codes that indicate permanent auth failure
_PERMANENT_WS_CODES = {4001}  # kimi-claw's auth-failed sentinel

# Group-gate header defaults. Kimi's IM service refuses group-room
# participation unless these OpenClaw runtime headers meet the minimum CalVer
# (2026.3.13). Defaults here unlock group access; cosmetic fields like
# claw_id auto-generate a unique value per adapter instance.
_GROUP_GATE_DEFAULTS = {
    "claw_version": "0.25.0",
    "openclaw_version": "2026.3.13",
    "openclaw_plugins": [{"id": "kimi-claw", "version": "0.25.0"}],
    "openclaw_skills": [],
}

_SLASH_COMMAND_RE = re.compile(r"^/[a-z0-9_-]+$", re.IGNORECASE)
_DEFAULT_USER_MESSAGE_PREFIX = "User Message From Kimi:\n"

# kimi-claw's prompt-adapter injects sender identity into the prompt text as
# a `[sender_short_id: <short_id>]` line for group-routed-over-ACP messages
# (see kimi-claw's user-message-prefix.js::withGroupRoomSenderShortId). The
# structured session/prompt.params carries only sessionId + prompt by design.
# We parse this prefix line so per-user routing works when Kimi routes a
# group message through the DM WS instead of through Subscribe.
_SENDER_SHORT_ID_LINE_RE = re.compile(
    r"^\s*\[sender_short_id:\s*([^\]\s][^\]]*?)\s*\]\s*$",
    re.MULTILINE,
)

# Kimi's WS frames are large but finite; 4MB matches the bridge setting.
_WS_MAX_FRAME_SIZE = 4 * 1024 * 1024

# DM outbound chunking — Kimi's UI renders progressively so split long replies.
_DM_CHUNK_SIZE = 3500

# Unary RPC default timeout (seconds).
_RPC_TIMEOUT_S = 30.0

# Reconnect backoff bounds.
_RECONNECT_MIN_S = 2.0
_RECONNECT_MAX_S_DEFAULT = 60.0

# DM application-level keepalive interval. Kimi's server idle-closes the WS
# at ~60s when no ACP frames flow; WS-protocol PING frames do NOT satisfy
# its liveness check (observed code=1006 close at exactly 60s post-connect
# during an idle window). We emit `$/ping` JSON-RPC notifications well under
# the 60s window to reset the server's idle timer. `$/`-prefixed methods are
# the LSP/JSON-RPC convention for implementation-specific notifications and
# MUST be ignored by peers that don't recognize them.
_DM_APP_KEEPALIVE_S_DEFAULT = 25.0

# DM health tripwire: after this many seconds since first connect, log a
# one-shot summary of DM prompt traffic. In live testing, Kimi routes what
# users call "DMs" through the group Subscribe stream as room:<uuid>; the
# ACP WS connects and keepalives but never receives session/prompt frames.
# Zero traffic after an hour flags that the ACP WS path may be dead weight
# — the operator can toggle config.extra.enable_dms=false to drop it.
_DM_HEALTH_SUMMARY_S_DEFAULT = 3600.0

# Dedup ring buffer size — covers Kimi's replay window on Subscribe reconnect.
_DEDUP_MAXLEN = 2000

# Default cap for per-room state dicts (``_rooms``, ``_last_message_id_per_
# room``, ``_probe_msg_id_room_counts``). Bloom in production sees ~4-10 unique
# rooms over weeks; 500 is two orders of magnitude above that, so the cap is
# inert for normal operation. The point is to bound the worst case for
# operators running this plugin at higher cardinality (Bloom-as-a-Service,
# multi-org deployments) without forcing them to tune anything. Override via
# ``config.extra["room_cache_max_entries"]``.
_ROOM_CACHE_DEFAULT_MAX = 500

# Phase 0 burst-drop instrumentation: arrival-time cache is bounded too,
# but at a much higher ceiling than the other per-room dicts. Codex review
# #58 noted that sharing the 500-entry cap with ``_rooms`` /
# ``_last_message_id_per_room`` / ``_probe_msg_id_room_counts`` would
# *silently neuter* the gap-candidate log under cardinality pressure: an
# evicted arrival-time entry yields ``prev_arrival=None`` on the next
# message → no INFO log fires regardless of actual delay. Since gap
# detection is correctness-critical for observability (the very signal
# Phase 0 ships to gather), this dict gets its own cap that's effectively
# unbounded for any realistic Kimi deployment. 10000 entries × ~16 bytes
# (room_id key + monotonic float) ≈ 160KB — negligible. Override via
# ``config.extra["arrival_time_cache_max_entries"]`` on the rare deployment
# that exceeds 10000 distinct rooms; consider Phase 1 recovery dispatch
# at that point because 10000+ rooms is a different scale problem entirely.
_ARRIVAL_TIME_CACHE_DEFAULT_MAX = 10000

# Kimi file upload/download tuning.
_FILE_UPLOAD_MAX_PATHS = 5
_FILE_UPLOAD_TIMEOUT_S_DEFAULT = 120.0
_KIMI_FILE_URI_RE = re.compile(r"^kimi-file://([^/?#\s]+)$")

_CHAT_MESSAGE_COMPLETED_STATUSES = {"2", "COMPLETED", "STATUS_COMPLETED"}
_CHAT_MESSAGE_INCOMPLETE_STATUSES = {
    "0",
    "1",
    "UNSPECIFIED",
    "GENERATING",
    "STATUS_UNSPECIFIED",
    "STATUS_GENERATING",
}
# NOTE: `role` classifies message CONTENT (USER/ASSISTANT/SYSTEM per OpenAI chat
# semantics), not sender IDENTITY. A human using AI-drafting tools may emit
# `role=ASSISTANT`; a bot may emit `role=USER`. Use `group_trusted_senders` for
# authoritative sender-identity gating; this role axis is a best-effort content
# signal only.
_USER_MESSAGE_ROLES = {"USER", "ROLE_USER", "MESSAGE_ROLE_USER"}
_NON_USER_MESSAGE_ROLES = {
    "ASSISTANT",
    "BOT",
    "MODEL",
    "SYSTEM",
    "ROLE_ASSISTANT",
    "ROLE_BOT",
    "ROLE_MODEL",
    "ROLE_SYSTEM",
    "MESSAGE_ROLE_ASSISTANT",
    "MESSAGE_ROLE_BOT",
    "MESSAGE_ROLE_MODEL",
    "MESSAGE_ROLE_SYSTEM",
}

MAX_MESSAGE_LENGTH = 8000  # Kimi UI handles long messages, but chunking is kinder


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class KimiAdapterError(Exception):
    """Base for all Kimi adapter errors."""


class KimiAuthError(KimiAdapterError):
    """Permanent authentication failure (HTTP 401/403, WS 4001)."""


class KimiTransientError(KimiAdapterError):
    """Transient error — caller should retry with backoff."""


class KimiRpcError(KimiAdapterError):
    """RPC returned a structured error (4xx with grpc-style body)."""


class KimiProtocolError(KimiAdapterError):
    """Malformed wire frame — treat as terminal for this connection."""


# ──────────────────────────────────────────────────────────────────────────────
# Requirements check
# ──────────────────────────────────────────────────────────────────────────────

def check_kimi_requirements() -> bool:
    """Return True if dependencies for the Kimi adapter are available.

    Mirrors other platforms' ``check_*_requirements`` pattern. We require:
    - ``websockets`` (for DM channel)
    - ``aiohttp`` (for group Connect RPC channel)
    Both ship via the ``hermes-agent[messaging]`` extra; this check exists
    for defensive symmetry with other platforms and to fail fast with a
    clear message if someone strips the runtime.
    """
    if not _WEBSOCKETS_AVAILABLE:
        logger.error("Kimi adapter: websockets package not installed")
        return False
    if not _AIOHTTP_AVAILABLE:
        logger.error("Kimi adapter: aiohttp package not installed")
        return False
    return True


def _check_for_registry() -> bool:
    """``check_fn`` for the platform-registry pass — stricter than the
    deps-only :func:`check_kimi_requirements`.

    ``gateway/config.py`` enables every plugin whose ``check_fn`` returns
    True. This PR declares ``websockets`` in the ``[messaging]`` extra,
    so a deps-only check would auto-enable KimiClaw on every install
    that already has the ``[messaging]`` extra (Telegram, LINE, …) even
    when ``KIMI_BOT_TOKEN`` is not set. Gate on credentials in addition
    to deps so unconfigured users do not see the platform light up.
    Mirrors the LINE / SimpleX / Google Chat precedent (e.g.
    ``plugins/platforms/google_chat/adapter.py::_check_for_registry``).
    """
    if not check_kimi_requirements():
        return False
    if not os.getenv("KIMI_BOT_TOKEN"):
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_standalone_slash_command(text: str) -> bool:
    return bool(_SLASH_COMMAND_RE.match(text.strip()))


def _first_text_block(params: Any) -> Optional[Dict[str, Any]]:
    """Return the first text block dict inside an ACP session/prompt params."""
    if not isinstance(params, dict):
        return None
    prompt = params.get("prompt")
    if not isinstance(prompt, list):
        return None
    for block in prompt:
        if isinstance(block, dict) and block.get("type") == "text":
            return block
    return None


def _extract_user_identity(params: Any) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort extract ``(user_id, user_name)`` from ACP ``session/prompt``.

    Kimi's public ACP contract defines ``session/prompt.params`` as carrying
    only ``sessionId`` and ``prompt`` — DMs are 1:1 so sender identity is
    implicit in the WS session, and groups are normally delivered via the
    Subscribe stream (which carries structured sender metadata). There is no
    documented sender-identity field on ``params`` for either case.

    Defensive probes we still run (for future schema additions and any
    non-standard deployments):

      - Nested ``params["sender"]`` / ``params["user"]`` / ``params["author"]``
        with ``id`` / ``userId`` / ``name``
      - Flat ``params["userId"]`` / ``params["user_id"]``

    Returns ``(None, None)`` if none of these shapes are present. Callers
    should then probe the prompt text for a ``[sender_short_id: X]`` prefix
    via :func:`_extract_short_id_from_text` (kimi-claw's injection convention
    for group-routed-over-ACP messages).
    """
    if not isinstance(params, dict):
        return None, None
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    for key in ("sender", "user", "author"):
        obj = params.get(key)
        if isinstance(obj, dict):
            user_id = user_id or (
                obj.get("id") or obj.get("userId") or obj.get("user_id")
            )
            user_name = user_name or (
                obj.get("name") or obj.get("display_name") or obj.get("displayName")
            )
    user_id = user_id or params.get("userId") or params.get("user_id")
    return user_id, user_name


def _extract_short_id_from_text(text: str) -> Optional[str]:
    """Return the ``sender_short_id`` from kimi-claw's text prefix, if present.

    kimi-claw's client injects a ``[sender_short_id: <short_id>]`` line into
    the prompt text when forwarding a group-room message over the DM ACP WS
    (see kimi-claw ``src/user-message-prefix.js::withGroupRoomSenderShortId``).
    We treat this as the authoritative sender identity for that routing
    mode. The structured `params` surface does not carry it.
    """
    if not isinstance(text, str) or not text:
        return None
    m = _SENDER_SHORT_ID_LINE_RE.search(text)
    if not m:
        return None
    return m.group(1).strip() or None


def _field(obj: Dict[str, Any], *names: str) -> Any:
    """Return the first non-empty field value across camelCase/snake_case names."""
    for name in names:
        value = obj.get(name)
        if value is not None and value != "":
            return value
    return None


def _normalize_status(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped.upper()
    return None


def _chat_message_is_complete(value: Any) -> bool:
    """Return whether a Kimi ChatMessageEvent should be dispatched.

    Kimi's Connect JSON uses protobuf enum names (``STATUS_COMPLETED``), while
    decoded generated-message tests may carry the numeric enum value (2). Legacy
    hand-written fixtures often omit status entirely; keep those dispatchable.
    """
    status = _normalize_status(value)
    if status is None:
        return True
    if status in _CHAT_MESSAGE_COMPLETED_STATUSES:
        return True
    if status in _CHAT_MESSAGE_INCOMPLETE_STATUSES:
        return False
    # Unknown future status: prefer visibility over silent drops.
    return True


def _chat_message_is_user_role(value: Any) -> bool:
    """Return whether a Kimi ChatMessage role is dispatchable as user input."""
    role = _normalize_status(value)
    if role is None:
        return True
    if role in _USER_MESSAGE_ROLES:
        return True
    if role in _NON_USER_MESSAGE_ROLES:
        return False
    # Unknown future roles are not safe to treat as user prompts.
    return False


def _event_payload(event: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    """Normalize Kimi IM event oneof shapes to ``(case, value)``.

    On the wire, Connect/protobuf JSON uses top-level oneof field names like
    ``{"chatMessage": {...}}``. Kimi's generated JS objects use
    ``{"payload": {"case": "chatMessage", "value": {...}}}``. Older local
    tests used ``{"message": {...}}``. Accept all three.
    """
    payload = event.get("payload")
    if isinstance(payload, dict):
        case = payload.get("case")
        value = payload.get("value")
        if isinstance(case, str) and isinstance(value, dict):
            return case, value

    for case in ("chatMessage", "chat_message", "message", "ping", "reconnect", "typing"):
        value = event.get(case)
        if isinstance(value, dict):
            normalized = "chatMessage" if case in ("chat_message", "message") else case
            return normalized, value

    # Legacy tests sometimes pass the message fields at the top level.
    if _field(event, "chatId", "chat_id") and _field(event, "messageId", "message_id"):
        return "chatMessage", event

    return None, {}


def _redact_sender(short_id_or_id: Optional[str]) -> str:
    """Redact a Kimi sender identifier for safe INFO-level logging.

    Keeps the ``u_`` / ``b_`` type prefix plus the first 4 chars of the
    body, and masks the rest. Enough signal for an operator to spot a
    misconfigured ``group_trusted_senders`` (unique-ish prefix + known
    peer list → identification by pattern) without bleeding full
    short_ids into log aggregators.

    Examples::

        _redact_sender("u_gs5ri2l5dpytlap") -> "u_gs5r****"
        _redact_sender("b_ipt7azbrrljvjsu") -> "b_ipt7****"
        _redact_sender("kimi")              -> "kimi"  # <= 4 chars untouched
        _redact_sender(None)                -> "<none>"
    """
    if not short_id_or_id:
        return "<none>"
    s = str(short_id_or_id)
    if len(s) <= 4:
        return s
    # Preserve "u_" / "b_" prefix if present → "u_" + 4 chars + ****
    if len(s) >= 6 and s[1] == "_":
        return f"{s[:6]}****"
    return f"{s[:4]}****"


# Crockford base32 alphabet used by ULID (RFC: 0-9, A-Z minus I, L, O, U).
_ULID_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_CROCKFORD_INDEX = {c: i for i, c in enumerate(_ULID_CROCKFORD)}


def _ulid_time_ms(ulid_str: Optional[str]) -> Optional[int]:
    """Extract the 48-bit Crockford-base32 timestamp prefix from a ULID.

    Returns None on malformed input. **Important production caveat:** Kimi's
    actual message ids in production are UUID v6/v7/v8-shaped (36-char dashed
    hex like ``19dc9c4f-1262-8c1b-8000-0a4a6c626bbb``), not Crockford base32
    ULIDs. The first 48 bits of those UUIDs ARE time-monotonic but their
    units are NOT unix-ms — captured deltas show ~16× the wall-clock seconds,
    suggesting a Snowflake-style or otherwise non-standard epoch encoding.

    For correctness, this decoder accepts only the canonical Crockford ULID
    format. Production gap-detection should NOT rely on this — use wall-
    clock ``time.time()`` arrival tracking instead (see
    ``_last_arrival_time_per_room`` and the Phase 0 #18 gap-candidate logger).

    Used by Probe-3 timing DEBUG log and the test suite. Returns None for
    UUID-shaped input so callers must fall back to a different time source.
    """
    if not ulid_str or not isinstance(ulid_str, str) or len(ulid_str) < 10:
        return None
    prefix = ulid_str[:10].upper()
    total = 0
    for c in prefix:
        v = _ULID_CROCKFORD_INDEX.get(c)
        if v is None:
            return None
        total = total * 32 + v
    return total


def _block_text(block: Any) -> Optional[str]:
    if not isinstance(block, dict):
        return None

    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        case = content.get("case")
        value = content.get("value")
        if case == "text" and isinstance(value, dict):
            text = _field(value, "content", "text")
            return text if isinstance(text, str) else None
        for key in ("text", "textBlock"):
            nested = content.get(key)
            if isinstance(nested, dict):
                text = _field(nested, "content", "text")
                return text if isinstance(text, str) else None

    for key in ("text", "textBlock"):
        nested = block.get(key)
        if isinstance(nested, dict):
            text = _field(nested, "content", "text")
            return text if isinstance(text, str) else None
        if isinstance(nested, str):
            return nested

    text = _field(block, "contentText", "content_text")
    return text if isinstance(text, str) else None


def _resource_link_uri(block: Any) -> Optional[str]:
    if not isinstance(block, dict):
        return None

    candidates: List[Any] = []
    content = block.get("content")
    if isinstance(content, dict):
        if content.get("case") == "resourceLink" and isinstance(content.get("value"), dict):
            candidates.append(content["value"])
        for key in ("resourceLink", "resource_link"):
            nested = content.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)

    for key in ("resourceLink", "resource_link"):
        nested = block.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)

    for candidate in candidates:
        uri = _field(candidate, "uri", "url", "downloadUrl", "download_url")
        if isinstance(uri, str) and uri:
            return uri
    return None


def _extract_blocks_payload(msg: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    text_parts: List[str] = []
    media_urls: List[str] = []
    media_types: List[str] = []

    # Probe (H-A): collect non-text block shapes for a post-loop DEBUG
    # summary with envelope-length oracle fields. Observability-only —
    # removable standalone.
    non_text_blocks: List[Dict[str, Any]] = []

    blocks = msg.get("blocks") if isinstance(msg.get("blocks"), list) else []
    for block in blocks:
        text = _block_text(block)
        if text:
            text_parts.append(text)
        elif logger.isEnabledFor(logging.DEBUG):
            block_info: Dict[str, Any] = {}
            if isinstance(block, dict):
                block_info["keys"] = sorted(block.keys())
                content = block.get("content")
                if isinstance(content, dict):
                    block_info["content_case"] = content.get("case")
                    block_info["content_keys"] = sorted(content.keys())
                block_info["has_uri"] = _resource_link_uri(block) is not None
            else:
                block_info["type"] = type(block).__name__
            logger.debug(
                "Kimi groups: non-text block (no extracted text): %r",
                block_info,
            )
            non_text_blocks.append(block_info)
        uri = _resource_link_uri(block)
        if uri:
            media_urls.append(uri)
            media_types.append("resource_link")

    text = "\n".join(text_parts).strip()

    # Probe (H-A) oracle: one summary line per message that had any
    # non-text blocks, pairing the extracted length with the envelope
    # preview lengths so operators can distinguish "legitimate image
    # attachment" from "orphaned fragmented tail" without correlating
    # against the sender's intent manually.
    if non_text_blocks and logger.isEnabledFor(logging.DEBUG):
        envelope_text = _field(msg, "text")
        envelope_summary = _field(msg, "summary")
        envelope_text_len = len(envelope_text) if isinstance(envelope_text, str) else 0
        envelope_summary_len = (
            len(envelope_summary) if isinstance(envelope_summary, str) else 0
        )
        extracted_len = sum(len(p) for p in text_parts)
        logger.debug(
            "Kimi groups: %d non-text block(s) — extracted_text=%d, envelope_text=%d, envelope_summary=%d, shapes=%r",
            len(non_text_blocks), extracted_len, envelope_text_len,
            envelope_summary_len, non_text_blocks,
        )

    return text, media_urls, media_types


def _build_text_block(text: str) -> Dict[str, Any]:
    """Build Kimi SendMessageRequest block JSON using protobuf JSON names."""
    return {
        "id": f"hermes_{uuid.uuid4().hex}",
        "text": {"content": text},
    }


def _build_resource_link_block(resource: Dict[str, Any]) -> Dict[str, Any]:
    uri = str(_field(resource, "uri", "url", "downloadUrl", "download_url") or "")
    title = str(_field(resource, "title", "name", "fileName", "file_name") or uri)
    return {
        "id": f"hermes_{uuid.uuid4().hex}",
        "resourceLink": {
            "title": title,
            "uri": uri,
            "downloadUrl": str(_field(resource, "downloadUrl", "download_url") or uri),
            "etag": str(_field(resource, "etag") or ""),
            "sizeBytes": int(_field(resource, "sizeBytes", "size_bytes") or 0),
        },
    }


def _infer_mime_type(path: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def _sanitize_kimi_file_name(name: str) -> str:
    base = Path(name).name.strip()
    base = re.sub(r"[\x00-\x1f\x7f/\\]+", "_", base)
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("._")
    return (base[:120] or "file")


def _parse_kimi_file_id(uri: str) -> Optional[str]:
    match = _KIMI_FILE_URI_RE.match(uri.strip())
    return match.group(1) if match else None


def _origin_from_url(url: str, fallback: str = "https://www.kimi.com") -> str:
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        pass
    return fallback.rstrip("/")


def _upload_endpoint(kimiapi_host: str) -> str:
    host = (kimiapi_host or _DEFAULT_KIMIAPI_HOST).strip().rstrip("/")
    if host.endswith("/files:upload"):
        return host
    origin = _origin_from_url(host)
    return f"{origin}/api-claw/files:upload"


def _file_metadata_endpoint(kimiapi_host: str, file_id: str) -> str:
    origin = _origin_from_url(kimiapi_host or _DEFAULT_KIMIAPI_HOST)
    return f"{origin}/api-claw/files/{file_id}"


async def _upload_kimi_file(
    session: Any,
    *,
    path: str,
    bot_token: str,
    upload_url: str,
    timeout_s: float,
) -> Dict[str, Any]:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise KimiProtocolError(f"Kimi upload path is not a readable file: {path}")

    file_name = file_path.name
    mime_type = _infer_mime_type(str(file_path))
    form = aiohttp.FormData()
    with file_path.open("rb") as handle:
        form.add_field("file", handle, filename=file_name, content_type=mime_type)
        async with session.post(
            upload_url,
            data=form,
            headers={"X-Kimi-Bot-Token": bot_token},
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            raw = await resp.read()
            if resp.status in (401, 403):
                raise KimiAuthError(f"upload auth failed HTTP {resp.status}")
            if resp.status >= 400:
                retryable = resp.status >= 500 or resp.status == 429
                exc = KimiTransientError if retryable else KimiRpcError
                raise exc(f"upload failed HTTP {resp.status}: {raw[:200]!r}")
            try:
                data = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise KimiProtocolError(f"upload returned bad JSON: {exc}") from exc

    file_obj = data.get("file") if isinstance(data, dict) else None
    if not isinstance(file_obj, dict):
        raise KimiProtocolError("upload response missing file object")
    file_id = _field(file_obj, "id")
    if not isinstance(file_id, str) or not file_id:
        raise KimiProtocolError("upload response missing file.id")
    meta = file_obj.get("meta") if isinstance(file_obj.get("meta"), dict) else {}
    return {
        "uri": f"kimi-file://{file_id}",
        "name": _field(meta, "name") or file_name,
        "mimeType": _field(meta, "contentType", "content_type") or mime_type,
        "sizeBytes": file_path.stat().st_size,
    }


async def _upload_kimi_files(
    session: Any,
    *,
    paths: List[str],
    bot_token: str,
    upload_url: str,
    timeout_s: float,
) -> List[Dict[str, Any]]:
    if len(paths) > _FILE_UPLOAD_MAX_PATHS:
        raise KimiProtocolError(
            f"Kimi upload supports at most {_FILE_UPLOAD_MAX_PATHS} files per message"
        )
    uploaded: List[Dict[str, Any]] = []
    for path in paths:
        uploaded.append(await _upload_kimi_file(
            session,
            path=path,
            bot_token=bot_token,
            upload_url=upload_url,
            timeout_s=timeout_s,
        ))
    return uploaded


def _header_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (list, dict)):
        if not value:
            return None
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return str(value)


def _resolve_env_template(value: Any) -> str:
    """Resolve ``${VAR}`` template strings against the environment.

    Defensive against Hermes config-loading paths that do not invoke
    env-var substitution for external-plugin platforms.  In particular,
    when ``platforms.kimi.token: ${KIMI_BOT_TOKEN}`` appears in
    ``~/.hermes/config.yaml``, the kimi plugin (registered via
    ``ctx.register_platform()``) receives the LITERAL string
    ``"${KIMI_BOT_TOKEN}"`` rather than the resolved env value.  The
    adapter's bot-token fallback chain previously short-circuited on
    that truthy-but-bogus value, sending the template string to
    kimi.com and getting 401 ``unauthenticated``.

    Returns:
        - For a ``"${NAME}"`` literal: the value of env var ``NAME``
          (``""`` if unset).
        - For any other string: the string unchanged.
        - For ``None``: ``""``.
        - For non-strings: ``str(value)``.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    stripped = value.strip()
    if (
        stripped.startswith("${")
        and stripped.endswith("}")
        and len(stripped) >= 4  # at least ${X}
    ):
        var_name = stripped[2:-1].strip()
        if var_name:
            return os.getenv(var_name, "")
    return value


def _normalize_openclaw_plugins(value: Any, claw_version: Any) -> Any:
    """Accept legacy string config while emitting Kimi Claw's JSON shape."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] in "[{":
        return value
    return [{
        "id": stripped,
        "version": str(claw_version or _GROUP_GATE_DEFAULTS["claw_version"]),
    }]


def _normalize_openclaw_skills(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] in "[{":
        return value
    return [item.strip() for item in stripped.split(",") if item.strip()]


def _runtime_headers(
    *,
    bot_token: Optional[str] = None,
    claw_version: Any = None,
    openclaw_version: Any = None,
    claw_id: Any = None,
    openclaw_plugins: Any = None,
    openclaw_skills: Any = None,
) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if bot_token:
        headers["X-Kimi-Bot-Token"] = bot_token
    plugins = _normalize_openclaw_plugins(openclaw_plugins, claw_version)
    skills = _normalize_openclaw_skills(openclaw_skills)
    for name, value in (
        ("X-Kimi-Claw-Version", claw_version),
        ("X-Kimi-OpenClaw-Version", openclaw_version),
        ("X-Kimi-Claw-ID", claw_id),
        ("X-Kimi-OpenClaw-Plugins", plugins),
        ("X-Kimi-OpenClaw-Skills", skills),
    ):
        header = _header_value(value)
        if header:
            headers[name] = header
    return headers


def _split_for_streaming(text: str, chunk_size: int) -> List[str]:
    """Split ``text`` for progressive DM streaming.

    Prefer paragraph boundaries, fall back to line boundaries, fall back to
    hard cuts. Avoids splitting mid-word where possible.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    remaining = text
    while len(remaining) > chunk_size:
        # Try paragraph break first
        split_at = remaining.rfind("\n\n", 0, chunk_size)
        if split_at < chunk_size // 2:
            # Too early — try single newline
            split_at = remaining.rfind("\n", 0, chunk_size)
        if split_at < chunk_size // 2:
            # Fall back to last space in the window
            split_at = remaining.rfind(" ", 0, chunk_size)
        if split_at < chunk_size // 2:
            split_at = chunk_size
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n").lstrip(" ")
    if remaining:
        chunks.append(remaining)
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Internal state dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _DMInflight:
    """Tracks one in-flight DM session/prompt awaiting a final end_turn response."""
    kimi_sid: str
    req_id: Any
    started_at: float = field(default_factory=time.time)


@dataclass
class _ChatInfoCache:
    """Cached metadata for one Kimi room (TTL-refreshed on demand)."""
    room_id: str
    name: Optional[str] = None
    members: List[Dict[str, Any]] = field(default_factory=list)
    last_refresh_ts: float = 0.0


class _BoundedLRU(OrderedDict):
    """``OrderedDict`` with a hard size cap; oldest entry evicted on overflow.

    Used for the adapter's per-room state dicts (``_rooms``,
    ``_last_message_id_per_room``, ``_probe_msg_id_room_counts``, and
    ``_last_arrival_time_per_room``) so the plugin stays bounded under
    arbitrary room cardinality. Reads do **not** refresh order — only
    writes. "Least recently used" → "least recently *updated*", which is
    the right semantics for our consumers:

    - ``_last_message_id_per_room`` is written every inbound group message,
      so a busy room naturally refreshes its position; idle rooms drift
      toward the LRU end. If reads refreshed too, the cap could never
      bite a single chatty room (read every message + write every message
      → permanent residency for *any* room with traffic).
    - ``_rooms`` owns its own freshness via the 300s TTL re-fetch in
      ``get_chat_info``; eviction recency is decoupled from that on
      purpose. Update-LRU treats "least recently re-fetched" as the
      eviction key, which is exactly what we want for a TTL-managed cache.
    - ``_probe_msg_id_room_counts`` reads and writes at the same call site
      (increment-then-store), so the read/write distinction is irrelevant.
    - ``_last_arrival_time_per_room`` writes monotonic timestamps every
      inbound chatMessage. The same write-on-message pattern as
      ``_last_message_id_per_room`` keeps the access semantics symmetric.
      **It uses its own larger cap** (``_ARRIVAL_TIME_CACHE_DEFAULT_MAX``,
      default 10000) rather than the shared 500-entry cap because eviction
      blinds the Phase 0 gap-candidate log — a re-cached room produces
      ``prev_arrival=None`` and the gap log won't fire no matter how
      large the actual delay was. Codex review #58 flagged this as a
      contradiction with the "no load-bearing state" disclaimer below;
      the separate cap resolves it.

    None of these dicts hold *message-dispatch* correctness state — replay
    dedup is owned by ``_processed_set``, which is independently bounded by
    ``_DEDUP_MAXLEN``. So eviction never causes a duplicate or dropped
    message at the agent layer. The arrival-time dict above is the lone
    *observability*-critical user; it gets a higher cap to keep the
    "bounded" promise meaningful in practice.

    Eviction *does* have observable side effects under cardinality pressure
    that we accept as the cost of bounded growth:

    1. **Resumed-room ``first-seen`` DEBUG log.** ``_last_message_id_per_
       room`` is hoisted out of the DEBUG gate (line ~2412) specifically
       so toggling DEBUG on later doesn't produce a misleading
       ``first-seen`` for an active room. Eviction of a quiet room
       reintroduces a similar (but narrower) failure mode: the next
       message from a resumed-after-eviction room logs ``first-seen``
       instead of a delta. Misleading observability, not misleading
       state. At the default cap (500) and Bloom's typical cardinality
       (~10 rooms) this never fires.
    2. **Probe sample-phase reset.** ``_probe_msg_id_room_counts`` keys
       its sampling phase off the count modulo ``probe_msg_id_sample_
       rate``. Eviction resets to count=1 so the first ``sample_rate-1``
       resumed messages are skipped from the DEBUG sample. Same scale
       gating as #1.
    3. **``_rooms`` cold-resume + RPC failure.** ``get_chat_info`` re-
       fetches a missing entry via ``GetRoom``/``ListMembers``; on
       ``KimiAdapterError`` the fallback path returns
       ``{"name": room_id, "type": "group"}`` with no members. Without
       eviction, that fallback only runs for never-cached rooms; with
       eviction, a transient RPC failure on a previously-cached room
       can briefly degrade display name + members until the next
       successful refresh. Real degradation, but bounded to one
       inbound per failed re-fetch and recovered on next success.

    None of these triggers at the default cap unless room cardinality
    exceeds 500, and operators can raise ``room_cache_max_entries`` if
    they routinely run more rooms than that. The README's "Bounded room
    state" entry summarises this for users.
    """

    def __init__(self, *, maxsize: int):
        super().__init__()
        if maxsize < 1:
            raise ValueError(f"maxsize must be ≥ 1, got {maxsize!r}")
        self._maxsize = maxsize

    def __setitem__(self, key: Any, value: Any) -> None:
        # Update existing key in-place but move it to the most-recent end so
        # the LRU ordering reflects the latest activity — important for
        # ``_last_message_id_per_room``, which writes the same key on every
        # inbound message in a busy room and must NOT count that room as
        # stale just because it was first inserted long ago.
        if key in self:
            self.move_to_end(key, last=True)
            super().__setitem__(key, value)
            return
        super().__setitem__(key, value)
        # Evict from the oldest end until we're under cap. ``last=False``
        # pops the *first*-inserted key (FIFO from the LRU end).
        while len(self) > self._maxsize:
            self.popitem(last=False)


# ──────────────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────────────

class KimiAdapter(BasePlatformAdapter):
    """Platform adapter for Kimi (kimi.com / Moonshot AI).

    Architecture: two concurrent long-lived tasks under one ``connect()``:
      - ``_dm_ws_loop``: maintains ACP WebSocket, synthesises handshake
        responses, dispatches ``session/prompt`` frames as MessageEvents,
        emits replies as ``agent_message_chunk`` updates.
      - ``_group_subscribe_loop``: maintains ``Subscribe {}`` Connect stream,
        translates inbound ChatMessageEvents into MessageEvents, routes
        outbound ``send()`` to ``SendMessage`` unary RPC.

    Both tasks reconnect independently with exponential backoff. A fatal
    auth failure stops only the affected task — DM and groups degrade
    independently.

    Chat-id scheme in MessageEvent.source.chat_id:
      - ``dm:im:kimi:main`` for the single DM channel (Kimi's sentinel
        sessionId)
      - ``room:<uuid>`` for group rooms. Inbound thread-like metadata is kept
        on ``SessionSource.thread_id`` when present, but current Kimi
        ``SendMessageRequest`` does not accept an outbound thread field.

    Loop-affinity invariant (load-bearing — see ``_session_for_current_loop``
    at the http-session helpers):
        ``self._http_session`` is created in ``connect()`` and is therefore
        bound to whichever event loop is running at connect time (the
        gateway's main loop in normal deployment).  Hermes's
        ``send_message_tool`` bridges back to async from a sync entry point
        via a worker-thread loop (``model_tools._run_async`` →
        ``worker_loop.run_until_complete``).  Calling ``self._http_session``
        from that worker loop trips aiohttp's
        ``current_task(loop=self._loop)`` guard and raises ``Timeout context
        manager should be used inside a task``.  Every async helper that
        touches the session goes through ``_session_for_current_loop()``,
        which transparently yields an ephemeral session bound to the current
        loop when the loops don't match.  Future maintainers: do not call
        ``self._http_session.post(...)`` directly from any code path that
        might be reached via the tool dispatcher.  Always go through the
        helper.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def __init__(self, config: PlatformConfig) -> None:
        # Platform("kimiclaw") creates a pseudo-member via Platform._missing_()
        # once the registry has been populated by our register() call below.
        # Identity-stable: Platform("kimiclaw") is Platform("kimiclaw"). See
        # upstream gateway/config.py::Platform._missing_ for the mechanism.
        super().__init__(config, Platform("kimiclaw"))

        # Credentials — ``_resolve_env_template`` guards against the
        # ``token: ${KIMI_BOT_TOKEN}`` config.yaml pattern arriving
        # unsubstituted (Hermes does not run env-var substitution for
        # external-plugin platform config).  Without it, the literal
        # ``"${KIMI_BOT_TOKEN}"`` would short-circuit the ``or`` chain
        # before the env-var fallback fires, and that template string
        # would be sent to kimi.com as the bot token (HTTP 401).
        self._bot_token: str = (
            _resolve_env_template(config.token)
            or _resolve_env_template(config.extra.get("bot_token", ""))
            or os.getenv("KIMI_BOT_TOKEN", "")
        )

        # Endpoints
        self._base_url: str = config.extra.get("base_url", _DEFAULT_BASE_URL).rstrip("/")
        self._kimiapi_host: str = config.extra.get(
            "kimiapi_host",
            config.extra.get("kimiapiHost", _DEFAULT_KIMIAPI_HOST),
        ).rstrip("/")
        self._upload_url: str = config.extra.get(
            "upload_url",
            _upload_endpoint(self._kimiapi_host),
        )
        self._dm_ws_url: str = config.extra.get("dm_ws_url", _DEFAULT_DM_WS_URL)
        self._file_timeout_s: float = float(
            config.extra.get("file_timeout_s", _FILE_UPLOAD_TIMEOUT_S_DEFAULT)
        )
        self._file_download_dir: Path = Path(
            config.extra.get(
                "file_download_dir",
                str(get_hermes_dir("cache/kimi_files", "kimi_file_cache")),
            )
        ).expanduser()

        # Channel enable flags
        self._enable_dms: bool = bool(config.extra.get("enable_dms", True))
        self._enable_groups: bool = bool(config.extra.get("enable_groups", True))

        # Group-gate runtime metadata headers (OpenClaw version, plugins,
        # skills) — kimi.com gates group-room frames at OpenClaw v2026.3.13.
        # See `website/docs/user-guide/messaging/kimiclaw.md` ("Deployment
        # model") for the disclosure of why we send these without running
        # an OpenClaw runtime locally.
        self._claw_version: str = config.extra.get(
            "claw_version", _GROUP_GATE_DEFAULTS["claw_version"]
        )
        self._openclaw_version: str = config.extra.get(
            "openclaw_version", _GROUP_GATE_DEFAULTS["openclaw_version"]
        )
        self._claw_id: str = config.extra.get("claw_id") or (
            f"hermes-kimi-{uuid.uuid4().hex[:16]}"
        )
        self._openclaw_plugins: Any = config.extra.get(
            "openclaw_plugins", _GROUP_GATE_DEFAULTS["openclaw_plugins"]
        )
        self._openclaw_skills: Any = config.extra.get(
            "openclaw_skills", _GROUP_GATE_DEFAULTS["openclaw_skills"]
        )

        # Message handling
        self._user_message_prefix: str = config.extra.get(
            "user_message_prefix", _DEFAULT_USER_MESSAGE_PREFIX
        )
        self._disable_prefix: bool = bool(config.extra.get("disable_prefix", False))
        self._auto_skill: Optional[Any] = config.extra.get("auto_skill")
        self._channel_prompt: Optional[str] = config.extra.get("channel_prompt")
        self._group_require_mention: bool = bool(
            config.extra.get("group_require_mention", False)
        )

        # Per-chat exemption from the mention gate. Mirrors the Matrix/DingTalk
        # `free_response_chats` pattern: when `group_require_mention=True`, rooms
        # listed here bypass the mention requirement and respond to every message.
        # Necessary because Kimi has no wire-level distinction between 1:1 DM and
        # group rooms — both are `room:<uuid>` — so a global mention requirement
        # silently swallows DM traffic without an exempt list. Accepts both keys
        # for cross-platform familiarity; `kimi_free_response_chats` is the
        # documented name, the other is an alias.
        #
        # Wire shape note: `_on_group_event` reads `chatId` directly from the
        # subscribe-stream envelope, where it arrives as a raw UUID with no
        # `room:` prefix.  Gateway-side identifiers (KIMI_HOME_CHANNEL,
        # /sethome output, `Session.chat_id` after handler enrichment) all
        # carry the `room:` prefix.  Users naturally copy-paste the prefixed
        # form from those surfaces.  We normalise the set entries to the raw
        # UUID shape that the gate's `chat_id` actually carries, so a config
        # of `room:<uuid>` and a wire frame with `chatId=<uuid>` correctly
        # compare equal.  Both formats are accepted on input.
        _exempt_raw = (
            config.extra.get("kimi_free_response_chats")
            or config.extra.get("group_require_mention_exempt_rooms")
            or []
        )

        def _normalise_exempt_entry(value: object) -> Optional[str]:
            if not isinstance(value, str) or not value:
                return None
            return value[len("room:"):] if value.startswith("room:") else value

        self._group_require_mention_exempt_rooms: frozenset[str] = frozenset(
            normalised
            for normalised in (_normalise_exempt_entry(r) for r in _exempt_raw)
            if normalised
        )

        # v2.2.0: room-distinguishability via list_group_members.  When True,
        # the mention gate auto-bypasses rooms that the ListMembers RPC reports
        # have exactly 2 members (bot + 1 user) — kimi.com delivers 1:1 DMs
        # as ``room:<uuid>`` indistinguishable from groups at the envelope
        # layer we consume, so without explicit classification a global
        # ``group_require_mention=true`` swallows DM traffic.  The detector
        # replaces use case 1 of ``kimi_free_response_chats`` (DM
        # misdetection); use case 2 (explicit free-response groups via
        # operator policy) is NOT auto-detectable and remains served by the
        # exempt list.  See ``.review/b1-room-distinguishability-spike.md``
        # for the cost analysis and race-window handling.
        #
        # Default OFF for v2.2.0 — opt-in until stabilization.  After
        # validation in production, a future release may flip the default.
        self._kimi_dm_autodetect: bool = bool(
            config.extra.get("kimi_dm_autodetect", False)
        )

        # Short_id / id allowlist — authoritative identity-based bypass of role filter
        trusted = config.extra.get("group_trusted_senders") or []
        self._group_trusted_senders: frozenset[str] = frozenset(
            str(s) for s in trusted if isinstance(s, (str, int))
        )

        # Policy for non-user-role senders: "off" | "trusted_only" | "mentions" | "all"
        raw = config.extra.get("group_allow_bot_senders", "off")
        if raw is True:
            raw = "all"
        elif raw is False:
            raw = "off"
        if raw not in ("off", "trusted_only", "mentions", "all"):
            logger.warning(
                "Kimi: invalid group_allow_bot_senders=%r, defaulting to 'off'", raw
            )
            raw = "off"
        self._group_allow_bot_senders: str = raw

        self._hydrate_missing_text: bool = bool(
            config.extra.get("hydrate_missing_text", True)
        )

        # Reconnect tuning
        self._reconnect_max_s: float = float(
            config.extra.get("reconnect_max_s", _RECONNECT_MAX_S_DEFAULT)
        )
        # Subscribe (group) backoff state — instance-scoped so a successful
        # stream (first processed frame post-connect) can reset it without
        # driving bot-thrash: after reset, the next reconnect delay starts
        # from the floor (not base), preventing oscillation hammering Kimi's
        # infra when the stream flaps every ~30-60s.
        self._group_subscribe_backoff_base: float = _RECONNECT_MIN_S
        self._group_subscribe_backoff_floor: float = 10.0
        self._group_subscribe_backoff: float = self._group_subscribe_backoff_base
        self._group_subscribe_frame_since_connect: bool = False
        # Monotonic counter of successful subscribe-stream reconnects,
        # incremented at the first chatMessage post-connect. Phase 0 burst-
        # drop instrumentation: operators correlate this against per-room
        # gap-candidate INFOs in the same log to see whether drops cluster
        # around reconnects (favours Phase 1 design) or appear mid-stream
        # (favours Phase 2/3). Cold start counts as #1.
        self._group_subscribe_reconnect_count: int = 0
        # Wall-clock timestamp (``time.time()``) of the most recent
        # increment, captured at the same line. Used by the gap-candidate
        # INFO log to compute ``since_reconnect_s`` so the correlation
        # operators want is in a single line rather than across two
        # journalctl streams. ``-1`` sentinel = no reconnect observed yet.
        self._group_subscribe_last_reconnect_ts: float = -1.0
        self._ws_ping_interval: int = int(config.extra.get("ws_ping_interval", 15))
        self._ws_ping_timeout: int = int(config.extra.get("ws_ping_timeout", 60))
        self._dm_app_keepalive_s: float = float(
            config.extra.get("dm_app_keepalive_s", _DM_APP_KEEPALIVE_S_DEFAULT)
        )
        self._startup_grace_s: float = float(config.extra.get("startup_grace_s", 30))
        self._dm_health_summary_s: float = float(
            config.extra.get("dm_health_summary_s", _DM_HEALTH_SUMMARY_S_DEFAULT)
        )

        # Session-key knobs hoisted out of the _dm_cancel_session hot path.
        # These are semantically config flags — read once at __init__ so the
        # session-cancel call site stays a straight function call, matching
        # the pattern used by `_group_require_mention` et al.
        self._group_sessions_per_user: bool = bool(
            config.extra.get("group_sessions_per_user", True)
        )
        self._thread_sessions_per_user: bool = bool(
            config.extra.get("thread_sessions_per_user", False)
        )

        # Runtime state
        self._closing: bool = False
        self._startup_ts: float = 0.0
        self._dm_task: Optional[asyncio.Task] = None
        self._group_task: Optional[asyncio.Task] = None
        self._dm_health_task: Optional[asyncio.Task] = None
        self._http_session: Optional[Any] = None  # aiohttp.ClientSession
        # The event loop ``self._http_session`` was constructed on.  aiohttp
        # binds ``ClientSession`` to ``asyncio.get_running_loop()`` at
        # ``__init__`` time; using the session from a different loop later
        # makes ``TimerContext`` raise ``RuntimeError("Timeout context
        # manager should be used inside a task")`` because
        # ``asyncio.current_task(loop=session._loop)`` returns ``None``.
        # See ``_session_for_current_loop`` for the cross-loop workaround.
        self._http_session_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws: Optional[Any] = None  # active DM WS

        # DM traffic observability — count successfully dispatched
        # session/prompt frames. A one-shot summary fires
        # `_dm_health_summary_s` after connect() to flag zero-traffic DMs
        # (Kimi routing DMs via the group Subscribe path instead of ACP).
        self._dm_prompt_count: int = 0

        # Bot identity (populated by GetMe on connect)
        self._me_id: Optional[str] = None
        self._me_short_id: Optional[str] = None
        self._me_name: Optional[str] = None

        # DM state
        # ACP synthetic session id (we generate; Kimi treats it as opaque)
        self._dm_fake_session_id: Optional[str] = None
        # In-flight DM prompts per sid (FIFO). Overlapping prompts are queued
        # so end_turn responses match their originating req_id in order.
        self._dm_inflight: Dict[str, "deque[_DMInflight]"] = {}
        # Kimi's actual sessionId parameter, observed from inbound frames
        self._dm_observed_kimi_sid: Optional[str] = None
        # 409 "bot already connected" strike count — resets on successful connect.
        # First strike sleeps 60s, subsequent strikes 300s to let Kimi's server-
        # side routing clear any ghost WS state from prior thrash cycles.
        self._dm_409_strikes: int = 0
        # One-shot warning guard for the multi-user DM collapse limitation.
        self._warned_dm_collapse: bool = False
        # One-shot warning guards for outbound group surfaces that Kimi Claw
        # v0.25.0's SendMessageRequest doesn't currently carry on the wire.
        # Kept as simple bools because the warning is a one-per-process
        # operator tripwire, not per-room metering.
        self._warned_outbound_thread_drop: bool = False
        self._warned_outbound_mentions_drop: bool = False

        # Dedup of inbound events (keyed by (source_tag, message_id))
        self._processed: deque = deque(maxlen=_DEDUP_MAXLEN)
        self._processed_set: Set[Tuple[str, str]] = set()

        # Bound the per-room state dicts so cardinality can't grow without
        # ceiling on long-running deployments (see ``_ROOM_CACHE_DEFAULT_MAX``
        # for rationale). All three dicts share the same key space (room_id)
        # and the same eviction semantics, so a single cap is sufficient.
        _raw_room_cap = config.extra.get(
            "room_cache_max_entries", _ROOM_CACHE_DEFAULT_MAX
        )
        try:
            _room_cap = max(1, int(_raw_room_cap))
        except (TypeError, ValueError):
            logger.warning(
                "Kimi: invalid room_cache_max_entries=%r in config.extra — "
                "expected positive integer, falling back to %d",
                _raw_room_cap, _ROOM_CACHE_DEFAULT_MAX,
            )
            _room_cap = _ROOM_CACHE_DEFAULT_MAX

        # Per-room cache
        self._rooms: _BoundedLRU = _BoundedLRU(maxsize=_room_cap)

        # Probe (H-C): per-room last-seen message_id, for DEBUG timing
        # correlation against conductor wall-clock. Observability only —
        # removable standalone with the Probe-3 log block.
        self._last_message_id_per_room: _BoundedLRU = _BoundedLRU(maxsize=_room_cap)
        # Probe (H-C) sample-rate knob: log 1-in-N per-room DEBUG records
        # under busy groups so operators can cap log volume without flipping
        # DEBUG off entirely. Default 1 (log every message — prior behavior).
        # Tracker-update itself is NEVER sampled — every inbound populates
        # _last_message_id_per_room so Fix A's invariant survives sampling.
        _raw_sample_rate = config.extra.get("probe_msg_id_sample_rate", 1)
        try:
            self._probe_msg_id_sample_rate: int = max(1, int(_raw_sample_rate or 1))
        except (TypeError, ValueError):
            logger.warning(
                "Kimi: invalid probe_msg_id_sample_rate=%r in config.extra — "
                "expected positive integer, falling back to 1",
                _raw_sample_rate,
            )
            self._probe_msg_id_sample_rate = 1
        self._probe_msg_id_room_counts: _BoundedLRU = _BoundedLRU(maxsize=_room_cap)

        # Phase 0 burst-drop instrumentation: per-room MONOTONIC arrival
        # tracking + INFO threshold. Tracks ``time.monotonic()`` at the
        # moment each chatMessage is processed in ``_on_group_event``;
        # logs at INFO when the per-room delta meets the threshold.
        # Monotonic (not wall-clock) because deltas need to be safe
        # against NTP step / leap-second / VM-suspend resume — Codex and
        # Kimi reviews #58 independently flagged that ``time.time()``-
        # based deltas can produce false-positive gap logs when the
        # system clock jumps. Operator correlation with journalctl
        # timestamps still works via the log line's own leading ISO
        # timestamp; the embedded delta describes "process-time since
        # last delivery" which is the honest signal. Higher cap than
        # the shared LRU (see ``_ARRIVAL_TIME_CACHE_DEFAULT_MAX``) to
        # keep gap detection unblinded under cardinality pressure.
        _raw_arrival_cap = config.extra.get(
            "arrival_time_cache_max_entries", _ARRIVAL_TIME_CACHE_DEFAULT_MAX
        )
        try:
            _arrival_cap = max(1, int(_raw_arrival_cap))
        except (TypeError, ValueError):
            logger.warning(
                "Kimi: invalid arrival_time_cache_max_entries=%r in config.extra "
                "— expected positive integer, falling back to %d",
                _raw_arrival_cap, _ARRIVAL_TIME_CACHE_DEFAULT_MAX,
            )
            _arrival_cap = _ARRIVAL_TIME_CACHE_DEFAULT_MAX
        self._last_arrival_time_per_room: _BoundedLRU = _BoundedLRU(maxsize=_arrival_cap)

        # Threshold in seconds for emitting the gap-candidate INFO log. 0
        # disables. Default 30 — well above conversational cadence (humans
        # often reply within 1-5s), conservative enough that an idle room
        # resuming after a coffee break naturally crosses it. False-positive
        # rate is acceptable: the operator-facing premise is "look for
        # CLUSTERS of these correlating against reconnects", not "any single
        # one is a problem". Codex review #1 noted this risks flooding INFO
        # in idle rooms — the wall-clock pivot keeps the signal honest;
        # threshold tuning per-deployment is the operator's call.
        _raw_gap_threshold = config.extra.get(
            "burst_drop_gap_log_threshold_s", 30.0
        )
        try:
            self._burst_drop_gap_log_threshold_s: float = max(0.0, float(_raw_gap_threshold))
        except (TypeError, ValueError):
            logger.warning(
                "Kimi: invalid burst_drop_gap_log_threshold_s=%r in config.extra "
                "— expected non-negative number, falling back to 30.0",
                _raw_gap_threshold,
            )
            self._burst_drop_gap_log_threshold_s = 30.0

        # ── Lift 3a: interrupt-and-drain queue improvements ───────────────
        # Pending-slot TTL (seconds). ``None`` = never expire (default,
        # matches Bloom's ``session_reset.mode: none`` — sessions are held
        # indefinitely). When set, a queued pending message older than this
        # many seconds is silently evicted and not dispatched to the agent,
        # preventing stale follow-ups from re-entering the conversation after
        # a long tool-call turn. Hakimi hard-codes 5 minutes; we make it
        # configurable so operators tune to their session-lifetime preference.
        _raw_ttl = config.extra.get("pending_message_ttl_seconds")
        self._pending_message_ttl: Optional[float] = (
            float(_raw_ttl) if _raw_ttl is not None else None
        )
        # Timestamps of when each pending message was *first* enqueued,
        # keyed by session_key. Used for TTL eviction and drop-log metadata.
        self._pending_enqueued_at: Dict[str, float] = {}

        # ── Lift 3b: output_mode flag ─────────────────────────────────────
        #
        # Two-path architecture: Hermes routes outbound messages to Kimi via
        # two distinct entry points, and ``output_mode`` controls only the
        # first.
        #
        #   1. ``adapter.send()`` ← gateway run-loop calls this for each agent
        #      prose chunk during a streaming turn. This is what generates
        #      "the agent typing back to you in chat" UX. Gated by
        #      ``output_mode``.
        #
        #   2. ``send_kimi_message()`` ← module-level helper at the bottom of
        #      this file.  Registered via ``standalone_sender_fn=_standalone_send``
        #      in ``register()``.  Invoked by Hermes's cron scheduler for
        #      out-of-process delivery, and by ``send_message_tool`` as the
        #      fallback path when no live adapter is in-process (per the
        #      upstream contract at ``tools/send_message_tool.py::_send_via_adapter``,
        #      which prefers the live ``adapter.send()`` when one exists).
        #      Bypasses ``adapter.send()`` entirely → never gated by
        #      ``output_mode``.
        #
        # Modes:
        #   ``passthrough`` (default) — both paths emit. Agent prose streams
        #     to Kimi as it's generated; explicit tool calls + cron deliveries
        #     also work. Production-default; matches every other Hermes
        #     platform adapter.
        #   ``tool_only`` — path (1) is suppressed (``send()`` returns early
        #     with success=True). The agent's prose still appears in Hermes
        #     logs but is not relayed to Kimi. Path (2) is unaffected — tool-
        #     driven sends and cron deliveries work normally. The user sees
        #     output only when the agent explicitly invokes
        #     ``send_message_tool``.
        #
        # When to use ``tool_only``:
        #   • Group rooms where streaming prose would be noisy and the agent
        #     should emit a single curated reply via the tool.
        #   • Multi-step agents where intermediate "thinking out loud" is
        #     undesirable platform-side but useful in logs.
        #   • Cron-only or tool-only deployments (no interactive turns).
        #
        # When NOT to use ``tool_only``:
        #   • 1:1 DMs where the user expects streaming response UX — silence
        #     looks like the bot hung.
        #   • Setups where the agent isn't reliably guided (system prompt or
        #     skill nudge) to call ``send_message_tool``. Without that
        #     guidance, the bot will appear mute on every turn.
        #
        # Background — this flag exists because the bridge's
        # ``HIDE_TOOL_CALLS=1`` filter, the prior workaround, hung Hermes
        # over stdio. The adapter's in-process coupling lets us suppress at
        # the right layer without that deadlock.
        _raw_mode = config.extra.get("output_mode", "passthrough")
        if _raw_mode not in ("passthrough", "tool_only"):
            logger.warning(
                "Kimi: invalid output_mode=%r — expected 'passthrough' or "
                "'tool_only'; defaulting to 'passthrough'",
                _raw_mode,
            )
            _raw_mode = "passthrough"
        self._output_mode: str = _raw_mode

    async def connect(self) -> bool:
        """Open HTTP session, fetch bot identity, spawn channel loops.

        Returns ``True`` if at least one enabled channel is viable.
        """
        if not self._bot_token:
            logger.error("Kimi: no bot_token configured (set config.token or KIMI_BOT_TOKEN)")
            return False
        if not check_kimi_requirements():
            return False

        if not self._acquire_platform_lock(
            "kimi-bot-token", self._bot_token, "Kimi bot token"
        ):
            return False

        self._closing = False
        self._startup_ts = time.time()
        # trust_env=True so that HTTP_PROXY / HTTPS_PROXY / ALL_PROXY env
        # vars are honoured. aiohttp's default is False, which silently
        # ignores them and breaks corporate-proxy deployments. Symmetric
        # with the in-tree Yuanbao / WeCom / Weixin / Matrix adapters and
        # with the SMS / Slack / Teams / Google-Chat sweep that landed
        # upstream in commit c1ae18ee8.  Must be re-applied at EVERY
        # ClientSession() construction site below; aiohttp does not pick
        # this up from a module-level default.
        self._http_session = aiohttp.ClientSession(trust_env=True)
        self._http_session_loop = asyncio.get_running_loop()

        # Belt-and-braces sweep of parallel TTL state at the start of
        # every connect cycle. Standard teardown paths (disconnect,
        # cancel_background_tasks) already clear this, but partial-init
        # failures or future code paths that bypass them could leave
        # stale entries from a prior session — and the gateway reuses
        # this same adapter instance on reconnect. Clearing here
        # guarantees every connect starts from a known-empty state.
        self._pending_enqueued_at.clear()

        # Fetch bot identity once — needed to filter self-authored group messages.
        try:
            me = await self._rpc_unary("GetMe", {})
            self._me_id = me.get("id")
            self._me_short_id = me.get("shortId")
            self._me_name = me.get("name")
            logger.info(
                "Kimi: connected as %s (shortId=%s, id=%s)",
                self._me_name, self._me_short_id, self._me_id,
            )
        except KimiAuthError as exc:
            logger.error("Kimi: GetMe auth failed: %s", exc)
            await self._cleanup_http()
            self._release_platform_lock()
            return False
        except Exception as exc:
            logger.warning("Kimi: GetMe failed (%s); continuing — loops will retry", exc)

        if self._enable_dms:
            self._dm_task = asyncio.create_task(self._dm_ws_loop(), name="kimi-dm")
            # Arm the one-shot DM health tripwire in parallel. If disconnect()
            # fires before _dm_health_summary_s elapses, the task is cancelled
            # cleanly and never logs.
            if self._dm_health_summary_s > 0:
                self._dm_health_task = asyncio.create_task(
                    self._log_dm_health_summary(), name="kimi-dm-health",
                )
        if self._enable_groups:
            self._group_task = asyncio.create_task(
                self._group_subscribe_loop(), name="kimi-group"
            )

        if not (self._dm_task or self._group_task):
            logger.error("Kimi: both channels disabled — nothing to do")
            await self._cleanup_http()
            self._release_platform_lock()
            return False

        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Cancel both loops, close WS + HTTP session."""
        self._closing = True
        self._mark_disconnected()

        tasks = [
            t for t in (self._dm_task, self._group_task, self._dm_health_task)
            if t is not None
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._dm_health_task = None

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        await self._cleanup_http()
        self._release_platform_lock()

        # Defense-in-depth: clear our parallel TTL state. The gateway's
        # standard shutdown path calls cancel_background_tasks() before
        # disconnect() (gateway/run.py:2725-2729), and that override
        # already clears _pending_enqueued_at via super(). But other
        # call sites — gateway/run.py:_safe_adapter_disconnect at
        # ~line 953 + line 1145, plus the error-recovery branch at
        # ~line 1110-1112 in connect() — call disconnect() directly
        # without a prior drain. Clearing here ensures
        # _pending_enqueued_at never outlives the connection,
        # regardless of teardown path.
        #
        # Known limitation (out of scope for this fix): direct-
        # disconnect paths don't clear the base class's
        # ``_pending_messages`` or ``_active_sessions`` either — those
        # only get cleaned up via cancel_background_tasks(). If a
        # future code path reuses an adapter after a direct disconnect
        # with real pending messages, those would also need clearing.
        # The pre-existing behaviour is unchanged by this commit.
        self._pending_enqueued_at.clear()

    async def cancel_background_tasks(self) -> None:
        """Mirror base behaviour for our parallel TTL state.

        BasePlatformAdapter.cancel_background_tasks (gateway/platforms/
        base.py:2553-2554) clears ``_pending_messages`` and
        ``_active_sessions`` at the end of its drain. Our subclass
        maintains a parallel ``_pending_enqueued_at`` dict that is
        only meaningful while the corresponding ``_pending_messages``
        slot is live; once base clears its state, our timestamps are
        orphaned. Without this override they leak across reconnects
        (the gateway typically reuses the adapter instance).

        Correctness note: a stale-only timestamp is benign — the TTL
        guard in ``handle_message`` keys off ``_pending_messages.get
        (session_key)`` first (see line ~1247) and bails if no slot
        exists, so a phantom ``_pending_enqueued_at`` entry can't
        evict a real later message. The leak is a memory-hygiene
        issue, not a correctness one — relevant for long-running pi
        deployments that reconnect repeatedly over weeks.

        Order: ``super()`` first, then our ``clear()``. Reversed,
        an in-flight handler whose ``finally`` block runs during the
        drain's ``await asyncio.gather`` could re-insert a key after
        our clear, leaving us with a single stray entry per drain.
        With this order, the base awaits all such handlers to
        completion (their ``finally`` blocks see ``_pending_messages``
        empty and pop their own timestamp via the guard), so our
        clear is a final sweep over a known-empty dict.

        Other parallel dicts on this adapter (``_last_message_id_per_
        room`` for replay dedup, ``_probe_msg_id_room_counts`` for
        debug counters) intentionally persist across reconnects or
        carry no semantic state — they're not in scope here.
        """
        await super().cancel_background_tasks()
        self._pending_enqueued_at.clear()

    @asynccontextmanager
    async def _session_for_current_loop(self) -> AsyncIterator[Any]:
        """Yield an ``aiohttp.ClientSession`` bound to the running event loop.

        The cached ``self._http_session`` is bound to whichever loop ran
        ``connect()`` (the gateway main loop).  Hermes's
        ``send_message_tool`` dispatches ``adapter.send()`` from a
        worker-thread loop via ``_run_async`` →
        ``worker_loop.run_until_complete``, which crosses event loops with
        the gateway-bound session.  aiohttp's ``TimerContext.__enter__``
        then raises ``RuntimeError("Timeout context manager should be used
        inside a task")`` because
        ``asyncio.current_task(loop=session._loop)`` looks at the
        gateway loop while we're on the worker loop.

        Strategy:
        - Same loop as the cached session → yield the cached session
          (connection pool preserved for normal traffic).
        - No cached session yet → create one on the current loop and cache
          it for reuse.
        - Different loop than the cached session → yield an ephemeral
          session bound to the current loop, closed when the ``async
          with`` block exits.  Single-connection cost on cross-loop calls
          is acceptable; the alternative is the bug.

        Counterpart upstream issue: ``send_message_tool`` should marshal
        ``adapter.send()`` back to the gateway loop via
        ``asyncio.run_coroutine_threadsafe`` rather than running the
        coroutine on the worker loop.  Until then, this helper is the
        plugin-side workaround.
        """
        current_loop = asyncio.get_running_loop()
        cached = self._http_session
        if cached is None:
            # First-ever lazy creation — cache on the current loop.
            # See ``connect()`` for the trust_env=True rationale.
            self._http_session = aiohttp.ClientSession(trust_env=True)
            self._http_session_loop = current_loop
            yield self._http_session
            return
        if getattr(cached, "closed", False):
            # Stale closed session — replace on the current loop.
            self._http_session = aiohttp.ClientSession(trust_env=True)
            self._http_session_loop = current_loop
            yield self._http_session
            return
        if self._http_session_loop is None or self._http_session_loop is current_loop:
            # Either same loop, or caller installed the session directly
            # (e.g. a test fixture that bypassed ``connect()``).  Either
            # way, trust it; cross-loop protection only kicks in when we
            # have a known-bound loop that differs from the current one.
            yield cached
            return
        # Cross-loop: ephemeral session scoped to this call.
        async with aiohttp.ClientSession(trust_env=True) as ephemeral:
            yield ephemeral

    async def _cleanup_http(self) -> None:
        if self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception:
                pass
            self._http_session = None
            self._http_session_loop = None

    # ──────────────────────────────────────────────────────────────────────
    # ── Lift 3a: handle_message override ─────────────────────────────────

    async def handle_message(self, event: MessageEvent) -> None:  # type: ignore[override]
        """Augment base dispatch with pending-slot drop-logging and TTL eviction.

        Hakimi's ``processMessage`` pattern (``chatRouter.ts:395-433``) uses a
        single-slot pending buffer and silently drops messages when a second
        follow-up arrives before the first processes. ``BasePlatformAdapter``
        already implements the slot + interrupt-and-drain correctly (including
        the error-path drain via ``finally``). This override adds two
        improvements over hakimi:

        1. **WARN on pending-slot overwrite** — when a new message overwrites
           an existing pending slot (hakimi silently drops). Logs chat_id and
           a 80-char redacted preview of the dropped message text.
        2. **Configurable TTL** — if ``pending_message_ttl_seconds`` is set,
           evict an expired pending message (too stale to be useful) before
           queuing the new one. Hakimi hard-codes 5 minutes; default here is
           ``None`` (no expiry) to match Bloom's indefinite session config.

        The error-path drain fix (drain runs in ``finally``, not just in
        ``try``) is already correct in ``BasePlatformAdapter`` via the
        late-arrival drain in ``_process_message_background``.
        """
        from gateway.session import build_session_key as _bsk

        session_key = _bsk(
            event.source,
            group_sessions_per_user=self._group_sessions_per_user,
            thread_sessions_per_user=self._thread_sessions_per_user,
        )

        # Only intervene when there is already an active session — this is
        # the exact condition under which the base class would overwrite the
        # pending slot.
        if session_key in self._active_sessions:
            now = time.monotonic()

            # ── TTL eviction ────────────────────────────────────────────────
            existing = self._pending_messages.get(session_key)
            if existing is not None and self._pending_message_ttl is not None:
                enqueued_at = self._pending_enqueued_at.get(session_key)
                if enqueued_at is not None:
                    age = now - enqueued_at
                    if age > self._pending_message_ttl:
                        logger.info(
                            "Kimi [%s]: evicting expired pending message "
                            "(age=%.1fs > ttl=%.1fs) — slot freed for new message",
                            session_key,
                            age,
                            self._pending_message_ttl,
                        )
                        self._pending_messages.pop(session_key, None)
                        self._pending_enqueued_at.pop(session_key, None)
                        existing = None

            # ── Drop-log when overwriting a non-expired slot ────────────────
            existing = self._pending_messages.get(session_key)
            if existing is not None:
                dropped_text = getattr(existing, "text", "") or ""
                preview = dropped_text[:80].replace("\n", " ")
                if len(dropped_text) > 80:
                    preview += "..."
                logger.warning(
                    "Kimi [%s]: overwriting pending slot — dropping message "
                    "(preview: %r). Latest message will be queued instead "
                    "(last-wins semantics).",
                    session_key,
                    preview,
                )
                self._pending_enqueued_at.pop(session_key, None)

            # Record enqueue timestamp for the new pending message.
            # Set BEFORE calling super() so the slot timestamp is consistent
            # with what super() puts in _pending_messages.
            self._pending_enqueued_at[session_key] = now

        try:
            await super().handle_message(event)
        finally:
            # Clean up timestamp when the session finishes (slot consumed
            # or not needed). Wrapped in finally so any unexpected
            # exception from super() — most relevantly a CancelledError
            # propagating up from base.handle_message itself — doesn't
            # skip the cleanup and leak a timestamp into a future
            # invocation. (Note: gateway/run.py's task-drain at
            # cancel_background_tasks cancels ``_background_tasks`` —
            # the spawned ``_process_message_background`` workers —
            # not direct ``handle_message`` callers, so the
            # cancellation pressure here is from other paths.)
            #
            # Why the guard is deterministic under cancellation: base
            # writes to ``_pending_messages[session_key]`` happen
            # synchronously with no ``await`` between the write and
            # the function return (see gateway/platforms/base.py
            # interrupt-queue path). So by the time our ``finally``
            # observes ``_pending_messages``, the slot is in one of
            # two known states: empty (no follow-up landed → safe to
            # pop) or owned by a follow-up (write completed before
            # our await unwound → preserve the fresh timestamp the
            # follow-up's own pre-super block set).
            if session_key not in self._pending_messages:
                self._pending_enqueued_at.pop(session_key, None)

    # Public send / platform-surface overrides
    # ──────────────────────────────────────────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Route outbound message by chat_id prefix.

        - ``dm:<sid>``   → emit ACP ``agent_message_chunk`` frames + end_turn
        - ``room:<id>``  → POST ``SendMessage`` with Kimi text blocks

        ``metadata`` keys:
          - ``thread_id``: accepted for compatibility with the Hermes surface.
            Kimi Claw v0.25.0's ``SendMessageRequest`` has no thread field, so
            threaded sends currently collapse to the underlying room with a
            one-shot WARNING log (see ``_send_group``).
          - ``mentions``: list of member short_ids to @-mention. Currently
            emits a one-shot WARNING and falls through to plain text; Kimi's
            mention-block wire shape is not yet confirmed via the surface
            check.
        """
        # ── Lift 3b: output_mode gate ─────────────────────────────────────
        # In ``tool_only`` mode the agent's prose responses are suppressed.
        #
        # Known behaviour against current upstream Hermes (≤ 0.14.0):
        # ``send_message_tool._send_via_adapter`` prefers the live adapter
        # over ``standalone_sender_fn`` when both exist (this is the
        # documented dispatch contract).  In ``tool_only`` mode, explicit
        # tool sends therefore route through this method and are suppressed
        # alongside streaming prose.  The dispatch contract itself is
        # correct; the limitation is that this plugin doesn't currently
        # differentiate prose vs explicit tool sends at the gate.  See
        # README "tool_only and send_message_tool — known limitation" for
        # the user-facing guidance + when each output_mode is appropriate.
        #
        # Default is ``passthrough`` so production behavior is unchanged.
        if self._output_mode == "tool_only":
            logger.debug(
                "Kimi: output_mode=tool_only — suppressing prose send to %s",
                chat_id,
            )
            # Bug fix: still need to close the inflight DM round-trip even when
            # we suppress the prose send, otherwise Kimi's UI spinner hangs
            # waiting for end_turn until WS reconnect. See _close_dm_inflight.
            if chat_id.startswith(_CHATID_DM_PREFIX):
                await self._close_dm_inflight(chat_id[len(_CHATID_DM_PREFIX):])
            return SendResult(success=True)

        if not content:
            # Same fix as the tool_only path: empty content still needs to
            # close any pending DM inflight round-trip.
            if chat_id.startswith(_CHATID_DM_PREFIX):
                await self._close_dm_inflight(chat_id[len(_CHATID_DM_PREFIX):])
            return SendResult(success=True)

        metadata = metadata or {}
        formatted = self.format_message(content)

        try:
            if chat_id.startswith(_CHATID_DM_PREFIX):
                kimi_sid = chat_id[len(_CHATID_DM_PREFIX):]
                return await self._send_dm(kimi_sid, formatted, reply_to, metadata)
            if chat_id.startswith(_CHATID_ROOM_PREFIX):
                room_and_thread = chat_id[len(_CHATID_ROOM_PREFIX):]
                if "/" in room_and_thread:
                    room_id, thread_id = room_and_thread.split("/", 1)
                else:
                    room_id, thread_id = room_and_thread, metadata.get("thread_id")
                return await self._send_group(
                    room_id, formatted, reply_to, thread_id, metadata
                )
            return SendResult(
                success=False,
                error=f"Kimi: unknown chat_id format: {chat_id!r}",
                retryable=False,
            )
        except KimiAuthError as exc:
            # Send-path auth failure (token expired mid-session, revoked, etc.)
            # — surface via runtime status so the gateway's per-platform
            # circuit breaker (upstream commit 518f39557, May 2026) sees the
            # platform as fatally broken and stops dispatching new sends
            # against this adapter.  Symmetric with the connect-loop auth
            # paths at _dm_ws_loop (≈L2269) and _group_subscribe_loop
            # (≈L2680), which already call _set_fatal_error on permanent
            # auth failure (rc==3).  Without this, the gateway would keep
            # accepting send requests against a dead adapter — silent
            # outage from the operator's perspective.
            #
            # retryable=True (not False) because a send-time auth failure
            # is ambiguous: it might be a refreshable token expiry or a
            # truly permanent revoke.  Let the reconnect machinery decide:
            # the connect attempt either succeeds (transient was real) or
            # hits the rc==3 path and re-fires _set_fatal_error with
            # retryable=False for the permanent case.
            logger.error(
                "Kimi: send-time auth failure on %s — marking platform "
                "fatal (retryable, reconnect will re-evaluate): %s",
                chat_id, exc,
            )
            self._set_fatal_error(
                "kimi_send_auth",
                f"Kimi: send-time auth failure: {exc}",
                retryable=True,
            )
            return SendResult(success=False, error=str(exc), retryable=False)
        except asyncio.TimeoutError:
            # Kimi.com server occasionally accepts a SendMessage POST internally
            # but holds the HTTP response on that TCP connection, causing
            # aiohttp.ClientTimeout(total=_RPC_TIMEOUT_S) to fire client-side at
            # exactly 30 s.  Production capture 2026-05-17:  one SendMessage
            # entered `_rpc_unary` at T+0 and never received RESPONSE_HEADERS,
            # while a concurrent SendMessage on the same adapter / same session
            # completed in 697 ms during the same window — proving the hang is
            # per-connection, not per-pool.  The pre-timeout request is still
            # processed server-side, so the previous policy of treating timeout
            # as retryable delivered the message twice (user-visible duplicate).
            # Mark non-retryable so the gateway does not double-send.  Genuine
            # network failures still propagate as KimiTransientError below.
            logger.warning(
                "Kimi: SendMessage to %s timed out after %.1fs at the client; "
                "kimi.com may still have accepted the message internally. "
                "Marking non-retryable to avoid duplicate delivery.",
                chat_id, _RPC_TIMEOUT_S,
            )
            return SendResult(
                success=False,
                error=(
                    f"Kimi: send timed out after {_RPC_TIMEOUT_S:.0f}s "
                    "(message may have been delivered server-side; not retrying)"
                ),
                retryable=False,
            )
        except KimiTransientError as exc:
            return SendResult(success=False, error=str(exc), retryable=True)
        except Exception as exc:
            logger.exception("Kimi: send failed")
            return SendResult(success=False, error=str(exc), retryable=False)

    async def _is_dm_room(self, room_id: str) -> Optional[bool]:
        """Classify a Kimi room as DM (2 members) or group (3+).

        Returns:
            ``True``  — 2 members (bot + 1 user); this is a 1:1 DM.
            ``False`` — 3+ members; this is a group room.
            ``None``  — classification failed (RPC error, etc.).  Callers
                        must fall back to existing behaviour (treat as group).

        Reads / populates ``self._rooms[room_id].members`` with the same 5
        minute TTL as ``get_chat_info``.  First call per new room costs one
        ``ListMembers`` RPC (≈50-200 ms over WAN to kimi.com); subsequent
        calls are O(1) until TTL expiry.

        Background: Kimi delivers 1:1 DMs as ``room:<uuid>`` indistinguishable
        from group rooms at the wire envelope our adapter consumes — see the
        v2.1.2 / v2.1.3 release notes for the original debugging episode that
        produced the ``kimi_free_response_chats`` workaround.  This helper is
        the v2.2.0 detector that lets us auto-bypass the mention gate for
        rooms structurally identifiable as DMs without operator config.
        """
        cached = self._rooms.get(room_id)
        if cached is not None and (time.time() - cached.last_refresh_ts) <= 300:
            members = cached.members
        else:
            try:
                members = await self.list_group_members(room_id)
            except (KimiAdapterError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "Kimi: DM-autodetect classification failed for %s — "
                    "falling back to exempt-list behaviour: %s",
                    room_id, exc,
                )
                return None
            # Refresh the cache.  Preserve any existing name (separate RPC).
            prior_name = cached.name if cached is not None else None
            self._rooms[room_id] = _ChatInfoCache(
                room_id=room_id,
                name=prior_name,
                members=members,
                last_refresh_ts=time.time(),
            )
        n = len(members)
        if n == 2:
            return True
        if n >= 3:
            return False
        # Degenerate (0 or 1 member) — shouldn't happen for a room that
        # actually sent us a message.  Defensive: log + return None so the
        # caller falls back to legacy behaviour rather than over-bypassing.
        logger.warning(
            "Kimi: room %s has %d members (expected ≥2) — cannot classify",
            room_id, n,
        )
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return ``{"name", "type", "chat_id", "members"?}``."""
        if chat_id.startswith(_CHATID_DM_PREFIX):
            return {
                "chat_id": chat_id,
                "name": "Kimi DM",
                "type": "dm",
            }
        if chat_id.startswith(_CHATID_ROOM_PREFIX):
            room_id = chat_id[len(_CHATID_ROOM_PREFIX):].split("/", 1)[0]
            cached = self._rooms.get(room_id)
            if cached is None or (time.time() - cached.last_refresh_ts) > 300:
                try:
                    room = await self.get_group(room_id)
                    members = await self.list_group_members(room_id)
                except KimiAdapterError:
                    return {"chat_id": chat_id, "name": room_id, "type": "group"}
                cached = _ChatInfoCache(
                    room_id=room_id,
                    name=room.get("name"),
                    members=members,
                    last_refresh_ts=time.time(),
                )
                self._rooms[room_id] = cached
            return {
                "chat_id": chat_id,
                "name": cached.name or room_id,
                "type": "group",
                "members": cached.members,
            }
        return {"chat_id": chat_id, "name": chat_id, "type": "unknown"}

    async def get_me(self) -> Dict[str, Any]:
        """Return the current Kimi bot/user identity."""
        return await self._rpc_unary("GetMe", {})

    async def get_group(self, room_id: str) -> Dict[str, Any]:
        """Return one Kimi room object using GetRoom."""
        room_resp = await self._rpc_unary("GetRoom", {"roomId": room_id})
        room = (
            room_resp.get("room")
            if isinstance(room_resp.get("room"), dict)
            else room_resp
        )
        return room if isinstance(room, dict) else {}

    async def list_group_members(
        self,
        room_id: str,
        *,
        page_size: int = 100,
        max_pages: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return room members using Kimi's ListMembers RPC."""
        members: List[Dict[str, Any]] = []
        page_token = ""
        for _ in range(max_pages):
            body: Dict[str, Any] = {
                "roomId": room_id,
                "pageSize": page_size,
            }
            if page_token:
                body["pageToken"] = page_token
            resp = await self._rpc_unary("ListMembers", body)
            page_members = resp.get("members")
            if isinstance(page_members, list):
                members.extend(m for m in page_members if isinstance(m, dict))
            page_token = str(resp.get("nextPageToken") or "")
            if not page_token:
                break
        return members

    async def list_group_messages(
        self,
        chat_id: str,
        *,
        limit: int = 20,
        start_message_id: Optional[str] = None,
        end_message_id: Optional[str] = None,
        include_start_message: bool = True,
        include_end_message: bool = True,
        direction: str = "BACKWARD",
        max_pages: int = 1,
    ) -> List[Dict[str, Any]]:
        """Return Kimi IM message wrappers for a group or thread chat.

        Wraps ``kimi.gateway.im.v1.IMService.ListMessages`` (unary
        Connect-RPC at ``https://www.kimi.com/api-ws/``).

        ## Direction (anchor-relative)

        - ``BACKWARD`` (default, wire enum value 2) — returns messages going
          *older in time* from the anchor. With no ``startMessageId``, returns
          the latest ``limit`` messages newest-first. With ``startMessageId``,
          returns messages older than (or including, per
          ``includeStartMessage``) that id. This is the "show me history" mode.
          ``_fetch_group_message`` (below) relies on this.
        - ``FORWARD`` — returns messages going *newer in time* from the
          anchor (and including the anchor when ``includeStartMessage=True``).
          Use this for catch-up / recovery patterns: pass the last message id
          you saw to fetch what came after. **Probed live 2026-04-27** via
          kimiim-cli: anchor of an oldest known id returned newer messages
          (CreateTimes ascending forward in time) plus the anchor itself in
          newest-first display order.

        ## Pagination (max_pages)

        The response carries ``nextPageToken``; echo it back as ``pageToken``
        on the next request. Empty/missing token = end of stream. Tokens are
        opaque server-issued cursors (NOT message ids — kimiim-cli treats
        ``--start-id`` and ``--page-token`` as separate modes).

        ``max_pages`` (default 1) caps the loop. Default keeps backward-
        compatible single-page behaviour with every existing caller. Set
        higher to recover gaps larger than ``limit`` messages; the cap
        prevents runaway fetches if Kimi keeps issuing tokens.

        Mirrors ``list_group_files`` (below) and ``list_group_members`` —
        same ``pageToken`` round-trip pattern.

        ## Wrapper schema (per item in returned list)

        ::

            {
                "senderId": str,
                "senderShortId": str,           # optional
                "senderName": str,              # optional
                "messageId": str,               # OPTIONAL at top level
                "message": {                    # canonical Message object
                    "id": str,                  # canonical message id lives HERE
                    "blocks": [...],
                    "threadId": str,            # optional
                    ...
                },
            }

        **Top-level ``messageId`` is sometimes absent** — the canonical id is
        ``wrapper["message"]["id"]``. Use ``_field`` to walk both possibilities
        safely (see ``_fetch_group_message`` for the precedent).

        ## Edge cases

        - Unknown ``chatId``: HTTP 400 → ``KimiRpcError``.
        - Malformed/zero ``startMessageId``: HTTP 400 with body
          ``"value does not match any of the specified id_kinds"`` (Kimi
          validates against ``id_kind=uuidv8`` — confirms the production id
          format). Treat as permanent error.
        - Stale-but-shape-valid ``startMessageId``: behaviour unproven for
          actual deleted/non-existent ids; recommend wrapping recovery calls
          in ``try/except KimiRpcError`` and falling back to no-anchor.
        - Empty room: returns ``[]`` after one RPC call.

        Sources for the contract:
        - kimiim-cli ``list-messages -h`` (definitive, since kimiim-cli is
          Kimi's own binary for this RPC)
        - mitm-captured request body for direction default
        - existing wrapper consumers ``_fetch_group_message`` and the
          hydration tests
        """
        if max_pages < 1:
            raise ValueError(f"max_pages must be ≥ 1, got {max_pages!r}")
        wrappers: List[Dict[str, Any]] = []
        page_token = ""
        for _ in range(max_pages):
            body: Dict[str, Any] = {
                "chatId": chat_id,
                "pageSize": limit,
                "direction": direction,
                "includeStartMessage": include_start_message,
                "includeEndMessage": include_end_message,
            }
            # ``pageToken`` and ``startMessageId``/``endMessageId`` are
            # mutually-exclusive cursoring modes: on page 0 (no token yet)
            # the anchors define the window; on page 1+ the server-issued
            # ``pageToken`` encodes everything needed to continue.
            # Echoing anchors alongside ``pageToken`` is what Kimi review
            # #58 flagged as risking duplicate results / undefined ordering
            # — and it's the precondition for ``_fetch_group_message``
            # to safely raise ``max_pages`` above 1. Mirrors the conditional
            # ``pageToken`` injection in ``list_group_files`` below.
            if page_token:
                body["pageToken"] = page_token
            else:
                if start_message_id:
                    body["startMessageId"] = start_message_id
                if end_message_id:
                    body["endMessageId"] = end_message_id
            resp = await self._rpc_unary("ListMessages", body)
            messages = resp.get("messages")
            if isinstance(messages, list):
                wrappers.extend(m for m in messages if isinstance(m, dict))
            page_token = str(resp.get("nextPageToken") or "")
            if not page_token:
                break
        return wrappers

    async def list_group_files(
        self,
        room_id: str,
        *,
        page_size: int = 100,
        max_pages: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return files shared in a Kimi room using ListRoomFiles."""
        files: List[Dict[str, Any]] = []
        page_token = ""
        for _ in range(max_pages):
            body: Dict[str, Any] = {"roomId": room_id, "pageSize": page_size}
            if page_token:
                body["pageToken"] = page_token
            resp = await self._rpc_unary("ListRoomFiles", body)
            page_files = resp.get("files")
            if isinstance(page_files, list):
                files.extend(f for f in page_files if isinstance(f, dict))
            page_token = str(resp.get("nextPageToken") or "")
            if not page_token:
                break
        return files

    def format_message(self, content: str) -> str:
        """Kimi renders markdown natively; pass through unchanged."""
        return content

    async def send_typing(
        self, chat_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Kimi has no native typing indicator RPC. DMs already show a spinner
        from the open ACP session/prompt; groups have no API surface. No-op."""
        return None

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
    ) -> SendResult:
        """Send image URLs as Kimi resource-link blocks for groups.

        Local images are handled by ``send_image_file`` through Kimi's upload
        endpoint. DM image upload is not supported by Kimi's ACP chunk channel,
        so DMs fall back to a text URL.
        """
        if chat_id.startswith(_CHATID_ROOM_PREFIX):
            room_and_thread = chat_id[len(_CHATID_ROOM_PREFIX):]
            room_id, thread_id = (
                room_and_thread.split("/", 1)
                if "/" in room_and_thread
                else (room_and_thread, None)
            )
            return await self._send_group(
                room_id,
                caption or "",
                reply_to=None,
                thread_id=thread_id,
                metadata={
                    "attachments": [{
                        "uri": image_url,
                        "title": caption or image_url,
                    }]
                },
            )
        text = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id, text)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        return await self._send_uploaded_file(
            chat_id,
            image_path,
            caption=caption,
            reply_to=reply_to,
            metadata=kwargs.get("metadata"),
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        metadata = dict(kwargs.get("metadata") or {})
        if file_name:
            metadata["file_name"] = file_name
        return await self._send_uploaded_file(
            chat_id,
            file_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def _send_uploaded_file(
        self,
        chat_id: str,
        file_path: str,
        *,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a local file to Kimi and send it as a resource link."""
        if not chat_id.startswith(_CHATID_ROOM_PREFIX):
            return await self.send(
                chat_id,
                f"{caption + chr(10) if caption else ''}File: {file_path}",
                reply_to=reply_to,
                metadata=metadata,
            )
        try:
            async with self._session_for_current_loop() as session:
                uploaded = await _upload_kimi_files(
                    session,
                    paths=[file_path],
                    bot_token=self._bot_token,
                    upload_url=self._upload_url,
                    timeout_s=self._file_timeout_s,
                )
        except KimiAuthError as exc:
            return SendResult(success=False, error=str(exc), retryable=False)
        except KimiTransientError as exc:
            return SendResult(success=False, error=str(exc), retryable=True)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=False)

        if metadata and metadata.get("file_name") and uploaded:
            uploaded[0]["name"] = metadata["file_name"]
        room_and_thread = chat_id[len(_CHATID_ROOM_PREFIX):]
        room_id, thread_id = (
            room_and_thread.split("/", 1)
            if "/" in room_and_thread
            else (room_and_thread, None)
        )
        return await self._send_group(
            room_id,
            caption or "",
            reply_to=reply_to,
            thread_id=thread_id,
            metadata={"attachments": uploaded},
        )

    # ──────────────────────────────────────────────────────────────────────
    # DM WebSocket loop
    # ──────────────────────────────────────────────────────────────────────

    async def _log_dm_health_summary(self) -> None:
        """One-shot observability tripwire for the DM ACP WS path.

        After ``_dm_health_summary_s`` elapses since ``connect()``, log an
        INFO summary of DM prompts processed. If the count is zero, log at
        WARNING — Kimi may be routing what users call "DMs" through the
        group Subscribe stream (we've seen this live) and the ACP WS path
        is effectively dead weight that can be disabled via
        ``config.extra.enable_dms = false``.

        Cancellation-safe: if the adapter disconnects before the delay
        elapses, the sleep raises ``CancelledError`` and we exit without
        logging anything. No recurrence — this is a tripwire, not a
        heartbeat.
        """
        try:
            await asyncio.sleep(self._dm_health_summary_s)
        except asyncio.CancelledError:
            return
        count = self._dm_prompt_count
        if count > 0:
            logger.info(
                "Kimi DM: received %d prompts in first hour", count,
            )
        else:
            logger.warning(
                "Kimi DM: zero prompts in first hour since connect — Kimi "
                "may be routing DMs via group Subscribe. Consider "
                "config.extra.enable_dms: false to disable ACP path.",
            )

    async def _dm_ws_loop(self) -> None:
        """Maintain the DM ACP WebSocket with exponential reconnect backoff."""
        backoff = _RECONNECT_MIN_S
        while not self._closing:
            rc = await self._dm_ws_connect_once()
            if self._closing or rc == 3:
                # Permanent auth failure — surface it via runtime status so the
                # gateway supervisor can stop retrying and alert the operator.
                if rc == 3 and not self._closing:
                    logger.error("Kimi DM: permanent auth failure, stopping loop")
                    self._set_fatal_error(
                        "kimi_dm_auth",
                        "Kimi DM WebSocket permanent auth failure",
                        retryable=False,
                    )
                return
            if rc == 1:
                # Other terminal error — stop trying but don't claim auth.
                logger.error("Kimi DM: terminal error, stopping loop")
                return
            # rc == 0: transient — back off + retry.
            logger.info("Kimi DM: reconnecting in %.1fs", backoff)
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                return
            backoff = min(backoff * 2, self._reconnect_max_s)

    async def _dm_ws_connect_once(self) -> int:
        """One WS connection attempt.

        Return codes:
          0 → transient (retry)
          1 → other terminal error
          3 → permanent auth failure (don't retry)
        """
        headers = self._ws_upgrade_headers()
        logger.info("Kimi DM: dialing %s", self._dm_ws_url)
        try:
            async with websockets.connect(
                self._dm_ws_url,
                additional_headers=headers,
                ping_interval=self._ws_ping_interval,
                ping_timeout=self._ws_ping_timeout,
                max_size=_WS_MAX_FRAME_SIZE,
            ) as ws:
                logger.info("Kimi DM: connected")
                self._ws = ws
                self._dm_fake_session_id = None
                self._dm_observed_kimi_sid = None
                self._dm_inflight.clear()
                # Successful upgrade — clear any accumulated 409 cooldown strikes.
                self._dm_409_strikes = 0
                keepalive_task = asyncio.create_task(
                    self._dm_app_keepalive(ws), name="kimi-dm-keepalive"
                )
                try:
                    async for frame in ws:
                        if isinstance(frame, bytes):
                            logger.debug("Kimi DM: dropping %d-byte binary frame", len(frame))
                            continue
                        try:
                            msg = json.loads(frame)
                        except json.JSONDecodeError:
                            logger.warning("Kimi DM: non-JSON frame: %.200r", frame)
                            continue
                        if isinstance(msg, dict):
                            await self._dm_on_inbound_frame(msg)
                finally:
                    keepalive_task.cancel()
                    try:
                        await keepalive_task
                    except asyncio.CancelledError:
                        # Expected — we just cancelled the keepalive. Outer-scope
                        # cancellation will propagate via the next await point.
                        pass
                    except Exception:
                        logger.debug(
                            "Kimi DM: keepalive task ended with exception",
                            exc_info=True,
                        )
                    self._ws = None
            return 0
        except ConnectionClosed as exc:
            code = getattr(exc, "code", None)
            logger.info("Kimi DM: WS closed code=%s reason=%r", code, getattr(exc, "reason", None))
            if code in _PERMANENT_WS_CODES:
                return 3
            return 0
        except asyncio.CancelledError:
            return 1
        except Exception as exc:
            status = (
                getattr(exc, "status_code", None)
                or getattr(getattr(exc, "response", None), "status_code", None)
            )
            if status == 401:
                logger.error("Kimi DM: WS upgrade 401 — bot token rejected")
                return 3
            if status == 403:
                logger.error("Kimi DM: WS upgrade 403 — bot forbidden")
                return 3
            if status == 409:
                # "Bot already connected" — Kimi's single-WS-per-token constraint.
                # Using the default 2s→60s exponential here produces reconnect
                # thrash that Kimi's routing layer can interpret as misbehavior
                # and silently throttle DM delivery to this bot for hours. Hard
                # cooldown so any ghost WS on the server side ages out first.
                self._dm_409_strikes += 1
                cooldown = 60.0 if self._dm_409_strikes == 1 else 300.0
                logger.warning(
                    "Kimi DM: WS upgrade 409 (ghost WS, strike %d) — cooling "
                    "off %.0fs before retry",
                    self._dm_409_strikes, cooldown,
                )
                try:
                    await asyncio.sleep(cooldown)
                except asyncio.CancelledError:
                    return 1
                return 0
            logger.warning("Kimi DM: connection error: %r", exc)
            return 0

    async def _dm_app_keepalive(self, ws: Any) -> None:
        """Emit `$/ping` JSON-RPC notifications to prevent Kimi's 60s idle close.

        Kimi's server idle-closes the WebSocket at ~60 seconds when no
        application-level ACP frames flow; WS-protocol PING frames (handled
        automatically by the `websockets` library) do NOT satisfy its
        liveness check. This was confirmed by observing code=1006 closes at
        exactly 60s post-connect during idle windows (no user messages, no
        outbound session/update notifications).

        `$/`-prefixed methods are the LSP / JSON-RPC convention for
        implementation-specific notifications that peers MUST silently
        ignore when unrecognized, so this is safe for any ACP-aware
        counterparty.
        """
        try:
            while True:
                await asyncio.sleep(self._dm_app_keepalive_s)
                try:
                    await ws.send(json.dumps(
                        {"jsonrpc": "2.0", "method": "$/ping", "params": {}},
                        separators=(",", ":"),
                    ))
                    logger.debug("Kimi DM: $/ping keepalive sent")
                except ConnectionClosed:
                    return
        except asyncio.CancelledError:
            return

    def _ws_upgrade_headers(self) -> Dict[str, str]:
        """Build headers for the DM WS upgrade, including group-gate runtime metadata."""
        return _runtime_headers(
            bot_token=self._bot_token,
            claw_version=self._claw_version,
            openclaw_version=self._openclaw_version,
            claw_id=self._claw_id,
            openclaw_plugins=self._openclaw_plugins,
            openclaw_skills=self._openclaw_skills,
        )

    async def _dm_on_inbound_frame(self, msg: Dict[str, Any]) -> None:
        """Dispatch one ACP JSON-RPC frame from Kimi.

        Kimi's client sends:
          - ``initialize``          → respond with agent info
          - ``session/new``         → respond with a synthetic sessionId
          - ``session/prompt``      → convert to MessageEvent, dispatch
          - ``session/cancel``      → cancel in-flight reply (best-effort)
        """
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}

        # Observe Kimi's sessionId unconditionally — needed for outbound rewrites.
        sid = params.get("sessionId") if isinstance(params, dict) else None
        if isinstance(sid, str) and sid != self._dm_observed_kimi_sid:
            self._dm_observed_kimi_sid = sid
            logger.info("Kimi DM: observed sessionId=%s", sid)

        if method == "initialize":
            await self._dm_respond(req_id, {
                "protocolVersion": 1,
                "agentInfo": {"name": "hermes-agent", "version": "1.0"},
            })
            return

        if method == "session/new":
            self._dm_fake_session_id = str(uuid.uuid4())
            await self._dm_respond(req_id, {"sessionId": self._dm_fake_session_id})
            logger.info("Kimi DM: created synthetic session %s", self._dm_fake_session_id)
            return

        if method == "session/cancel":
            logger.info("Kimi DM: session/cancel for sid=%s", sid)
            await self._dm_cancel_session(sid if isinstance(sid, str) else None)
            if req_id is not None:
                await self._dm_respond(req_id, None)
            return

        if method == "session/prompt":
            await self._dm_handle_prompt(msg)
            return

        # Unknown methods with ids: reply with method-not-found so the peer
        # doesn't hang. Notifications (no id): silently ignore.
        if req_id is not None:
            await self._dm_respond_error(
                req_id,
                code=-32601,
                message=f"method not found: {method}",
            )
        else:
            logger.debug("Kimi DM: ignoring notification method=%s", method)

    async def _dm_handle_prompt(self, msg: Dict[str, Any]) -> None:
        """Convert a session/prompt into a MessageEvent and dispatch."""
        req_id = msg.get("id")
        params = msg.get("params") or {}
        text_block = _first_text_block(params)
        if text_block is None:
            logger.warning("Kimi DM: session/prompt with no text block")
            if req_id is not None:
                await self._dm_respond(req_id, {"stopReason": "end_turn"})
            return

        text = text_block.get("text") if isinstance(text_block, dict) else ""
        if not isinstance(text, str):
            text = ""

        # Apply user-message prefix unless it's a standalone slash command.
        if (
            not self._disable_prefix
            and text
            and not _is_standalone_slash_command(text)
        ):
            text = self._user_message_prefix + text

        kimi_sid = self._dm_observed_kimi_sid or _DM_SESSION_SENTINEL
        chat_id = f"{_CHATID_DM_PREFIX}{kimi_sid}"
        message_id = str(req_id) if req_id is not None else f"dm-{uuid.uuid4().hex[:12]}"

        # Queue req_id so overlapping prompts get FIFO end_turn responses
        # instead of clobbering a previous in-flight (which would leave the
        # original prompt's Kimi UI spinner hanging forever).
        self._dm_inflight.setdefault(kimi_sid, deque()).append(
            _DMInflight(kimi_sid=kimi_sid, req_id=req_id)
        )

        # Resolve a per-user identity. Kimi's ACP contract only carries
        # sessionId + prompt on `params` by design; identity is implicit on
        # 1:1 DMs. For group-routed-over-ACP messages kimi-claw injects a
        # `[sender_short_id: X]` prefix into the prompt text, so we fall
        # through to that surface before giving up and using a sid-derived id.
        user_id, user_name = _extract_user_identity(params)
        if not user_id:
            short_id = _extract_short_id_from_text(text)
            if short_id:
                user_id = f"kimi:{short_id}"
        if not user_id:
            if not self._warned_dm_collapse:
                logger.warning(
                    "Kimi DM: session/prompt carries no user identity and "
                    "prompt text has no [sender_short_id: X] prefix — this "
                    "is expected for 1:1 DMs (single user). Multi-user "
                    "bots reading this WS will collapse all users into "
                    "one Hermes session; session state won't be isolated "
                    "per user. See docs § Known limitations."
                )
                self._warned_dm_collapse = True
            user_id = f"kimi:dm:{kimi_sid}"

        event = self._build_message_event(
            kind="dm",
            text=text,
            message_id=message_id,
            chat_id=chat_id,
            chat_name="Kimi DM",
            user_id=user_id,
            user_name=user_name,
            raw=msg,
        )
        # Count before dispatch — semantically "the adapter processed one
        # session/prompt frame from Kimi's WS". Not conditional on handler
        # success; the signal we care about is Kimi-side routing behaviour.
        self._dm_prompt_count += 1
        await self.handle_message(event)

    async def _dm_cancel_session(self, kimi_sid: Optional[str]) -> None:
        """Cancel active Hermes processing for a Kimi DM ACP session."""
        sid = kimi_sid or self._dm_observed_kimi_sid or _DM_SESSION_SENTINEL
        chat_id = f"{_CHATID_DM_PREFIX}{sid}"
        source = self.build_source(
            chat_id=chat_id,
            chat_name="Kimi DM",
            chat_type="dm",
            user_id=f"kimi:dm:{sid}",
            user_name=None,
        )
        session_key = build_session_key(
            source,
            group_sessions_per_user=self._group_sessions_per_user,
            thread_sessions_per_user=self._thread_sessions_per_user,
        )
        await self.cancel_session_processing(
            session_key,
            release_guard=True,
            discard_pending=True,
        )
        self._dm_inflight.pop(sid, None)

    async def _dm_respond(self, req_id: Any, result: Any) -> None:
        """Send a JSON-RPC result back over the DM WS."""
        if self._ws is None or req_id is None:
            return
        payload = {"jsonrpc": "2.0", "id": req_id, "result": result}
        try:
            await self._ws.send(json.dumps(payload, separators=(",", ":")))
        except ConnectionClosed:
            logger.info("Kimi DM: WS closed while sending response")

    async def _dm_respond_error(self, req_id: Any, code: int, message: str) -> None:
        if self._ws is None or req_id is None:
            return
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }
        try:
            await self._ws.send(json.dumps(payload, separators=(",", ":")))
        except ConnectionClosed:
            pass

    async def _dm_emit_chunk(self, kimi_sid: str, text: str) -> None:
        """Emit one ACP ``agent_message_chunk`` update over the WS."""
        if self._ws is None:
            return
        payload = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": kimi_sid,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text},
                },
            },
        }
        try:
            await self._ws.send(json.dumps(payload, separators=(",", ":")))
        except ConnectionClosed:
            logger.info("Kimi DM: WS closed while emitting chunk")

    async def _send_dm(
        self,
        kimi_sid: str,
        content: str,
        reply_to: Optional[str],
        metadata: Dict[str, Any],
    ) -> SendResult:
        """Emit streamed ``agent_message_chunk`` frames + end_turn response.

        ``reply_to`` is not used on the DM channel (Kimi's ACP doesn't expose
        reply-to semantics for DMs — the UI threads all responses to the
        in-flight prompt). Ignored gracefully.
        """
        del reply_to, metadata  # unused on DM path
        if self._ws is None:
            return SendResult(success=False, error="Kimi DM: WS not connected", retryable=True)

        chunks = _split_for_streaming(content, _DM_CHUNK_SIZE)
        for chunk in chunks:
            await self._dm_emit_chunk(kimi_sid, chunk)

        # Pop the oldest in-flight prompt for this sid (FIFO) and close its
        # round-trip via _close_dm_inflight (shared with send()'s short-
        # circuit paths so tool_only / empty-content don't hang the spinner).
        await self._close_dm_inflight(kimi_sid)

        return SendResult(
            success=True,
            message_id=f"dm-{uuid.uuid4().hex[:12]}",
        )

    async def _close_dm_inflight(self, kimi_sid: str) -> None:
        """Pop the oldest in-flight DM prompt and close its JSON-RPC round-trip.

        Called both by ``_send_dm`` (after streaming the prose chunks) and by
        ``send()``'s short-circuit paths (``output_mode='tool_only'`` and
        empty-content). Without this in the short-circuits, the Kimi UI
        spinner hangs waiting for ``end_turn`` until the next WS reconnect
        clears ``_dm_inflight``.

        Safe to call when no inflight prompt exists (no-op).
        """
        queue = self._dm_inflight.get(kimi_sid)
        if not queue:
            return
        inflight = queue.popleft()
        if not queue:
            self._dm_inflight.pop(kimi_sid, None)
        if inflight.req_id is not None:
            await self._dm_respond(inflight.req_id, {"stopReason": "end_turn"})

    # ──────────────────────────────────────────────────────────────────────
    # Group Subscribe loop
    # ──────────────────────────────────────────────────────────────────────

    async def _group_subscribe_loop(self) -> None:
        """Maintain the global ``Subscribe`` stream with reconnect backoff.

        Backoff state lives on the adapter (``_group_subscribe_backoff``) so a
        successful stream (first processed ``chatMessage`` frame post-connect)
        can reset it. The reset is conditional on the current backoff already
        exceeding the floor:

        - Cold start (backoff == base == 2s) → no reset, no log.
        - Grown state (e.g. 32s after churn) → clamps to floor (10s) and
          emits ``"stream recovered after N.Ns backoff"`` once per cycle.

        Only ``chatMessage`` events arm the hook — keepalive pings, typing,
        and control events cannot reset backoff (a degraded stream emitting
        only pings would otherwise thrash back to the floor every cycle).
        The floor (10s) rather than the base (2s) stays oscillation-safe: a
        flap-every-30s pattern would otherwise hammer Kimi's infra at
        2→4→8→... on every cycle.
        """
        while not self._closing:
            rc = await self._group_subscribe_once()
            if self._closing or rc == 3:
                if rc == 3 and not self._closing:
                    logger.error("Kimi groups: permanent auth failure, stopping loop")
                    self._set_fatal_error(
                        "kimi_groups_auth",
                        "Kimi Subscribe stream permanent auth failure",
                        retryable=False,
                    )
                return
            if rc == 1:
                logger.error("Kimi groups: terminal error, stopping loop")
                return
            logger.info(
                "Kimi groups: reconnecting in %.1fs", self._group_subscribe_backoff
            )
            try:
                await asyncio.sleep(self._group_subscribe_backoff)
            except asyncio.CancelledError:
                return
            self._group_subscribe_backoff = min(
                self._group_subscribe_backoff * 2, self._reconnect_max_s
            )

    async def _group_subscribe_once(self) -> int:
        """One Subscribe stream session.

        Return codes match ``_dm_ws_connect_once``.
        """
        url = f"{self._base_url}/{_IM_SERVICE}/Subscribe"
        headers = self._http_headers(streaming=True)
        body = self._encode_envelope(b"{}")

        try:
            async with self._session_for_current_loop() as session, session.post(
                url,
                data=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=None, sock_read=None),
            ) as resp:
                if resp.status == 401 or resp.status == 403:
                    logger.error("Kimi groups: Subscribe auth failure HTTP %s", resp.status)
                    return 3
                if resp.status != 200:
                    logger.warning("Kimi groups: Subscribe HTTP %s", resp.status)
                    return 0
                # Fresh connection — arm the first-frame hook. Reset + log
                # fire exactly once per reconnect cycle (see below).
                self._group_subscribe_frame_since_connect = False
                try:
                    async for event in self._connect_envelope_parser(resp.content):
                        # Classify BEFORE dispatch so ping/control/unsupported
                        # events can't satisfy the first-frame hook.
                        case, _ = _event_payload(event) if isinstance(event, dict) else (None, {})
                        is_chat_message = case == "chatMessage"
                        # Detect "first chatMessage of this cycle" BEFORE
                        # dispatch so the gap-candidate log inside
                        # ``_on_group_event`` reads the *correct* connect#
                        # and ``last_reconnect_ts`` for in-line correlation.
                        # Claude code-reviewer #58 + Codex challenge #58 both
                        # flagged the prior post-dispatch ordering as an
                        # off-by-one: gap log carrying ``connect#=N-1`` while
                        # the subscribe-stream-live log emitted moments later
                        # carried ``connect#=N`` for the same boundary
                        # event — defeats the design goal of in-line
                        # correlation without grepping two log streams.
                        is_first_chat_post_connect = (
                            is_chat_message
                            and not self._group_subscribe_frame_since_connect
                        )
                        if is_first_chat_post_connect:
                            # Counter + ts represent "stream received its
                            # first non-control frame this cycle". That
                            # fact is true even if the dispatch below
                            # raises and the cycle ends — the subsequent
                            # reconnect+first-frame produces another bump,
                            # giving honest cycle counts. Frame gate flips
                            # synchronously to prevent re-entry within the
                            # same cycle if a second chatMessage arrives.
                            self._group_subscribe_frame_since_connect = True
                            self._group_subscribe_reconnect_count += 1
                            self._group_subscribe_last_reconnect_ts = (
                                time.monotonic()
                            )
                            prev_backoff = self._group_subscribe_backoff
                        await self._on_group_event(event)
                        # Subscribe-live log + backoff clamp run AFTER
                        # dispatch. They gate on processing success
                        # (a raise inside ``_on_group_event`` exits via
                        # the except clauses below, skipping these). This
                        # preserves the original semantic that operator-
                        # facing "stream recovered" messages only fire on
                        # genuine recovery — degraded streams emitting
                        # only pings still won't reach this block because
                        # the first-frame check above requires
                        # ``is_chat_message``.
                        if is_first_chat_post_connect:
                            # Phase 0 burst-drop instrumentation: snapshot the
                            # adapter's per-room state at the first chatMessage
                            # post-connect. This is the candidate "Phase 1
                            # recovery hook point" — operators reading
                            # journalctl can correlate gap-candidate INFOs
                            # against this line to see whether suspicious gaps
                            # cluster around reconnects (favouring Phase 1
                            # recovery design) or appear mid-stream (favouring
                            # Phase 2 / 3). The counter is monotonic over the
                            # adapter lifetime; a sudden cluster of increments
                            # is itself signal.
                            logger.info(
                                "Kimi groups: subscribe stream live "
                                "(connect#%d, rooms_tracked=%d, prev_backoff=%.1fs)",
                                self._group_subscribe_reconnect_count,
                                len(self._last_message_id_per_room),
                                prev_backoff,
                            )
                            # Conditional reset: only clamp to the floor when
                            # backoff actually grew beyond it. Cold start
                            # (backoff=base=2s) stays at 2s with no spurious
                            # "stream recovered" log on every process boot.
                            # Grown state (e.g. 32s) clamps to floor (10s) —
                            # never below, to stay oscillation-safe.
                            if prev_backoff > self._group_subscribe_backoff_floor:
                                self._group_subscribe_backoff = (
                                    self._group_subscribe_backoff_floor
                                )
                                logger.info(
                                    "Kimi groups: stream recovered after %.1fs backoff",
                                    prev_backoff,
                                )
                except KimiAuthError as exc:
                    logger.error("Kimi groups: %s", exc)
                    return 3
                except (KimiTransientError, KimiProtocolError, KimiRpcError) as exc:
                    logger.warning("Kimi groups: stream error: %s", exc)
                    return 0
            return 0
        except asyncio.CancelledError:
            return 1
        except aiohttp.ClientError as exc:
            logger.warning("Kimi groups: HTTP client error: %r", exc)
            return 0
        except Exception:
            logger.exception("Kimi groups: unexpected error in Subscribe")
            return 0

    def _is_mention_of_me(self, msg: Dict[str, Any]) -> bool:
        """Check whether a ChatMessage event explicitly @-mentions this bot.

        Reads the `mentions` array and `mentioned` field, matching against
        self._me_id and self._me_short_id. Pure function — no side effects.

        NOTE: `group_allow_bot_senders="mentions"` is EXPERIMENTAL. Kimi's
        mention metadata may be client-provided rather than server-enriched
        (unverified as of this commit). Until verified via probe, a malicious
        sender could spoof mentions to bypass this gate. For production
        authorization, prefer `trusted_only` with an explicit
        `group_trusted_senders` allowlist.
        """
        if bool(_field(msg, "mentioned")):
            return True
        mentions = msg.get("mentions") or []
        if not isinstance(mentions, list):
            return False
        for m in mentions:
            if not isinstance(m, dict):
                continue
            for key in ("short_id", "shortId"):
                v = m.get(key)
                if v and self._me_short_id and str(v) == self._me_short_id:
                    return True
            v = m.get("id")
            if v and self._me_id and str(v) == self._me_id:
                return True
        return False

    async def _on_group_event(self, event: Dict[str, Any]) -> None:
        """Handle one decoded envelope from the Subscribe firehose.

        Events observed from kimi-claw:
          - ``{"ping": {}}`` — keepalive, ignored
          - ``{"chatMessage": {...}}`` — protobuf JSON oneof
          - ``{"payload": {"case": "chatMessage", "value": {...}}}`` —
            generated JS shape used by kimi-claw internals

        ChatMessage fields are normalized across protobuf JSON names and older
        local fixtures:
          ``{
            "chatId":        "<room-uuid>",
            "messageId":     "<uuid>",
            "status":        "STATUS_COMPLETED",
            "senderId":      "<uuid>"?,
            "senderShortId": "<short>"?,
            "summary":       "...",
            "blocks":        [{"text": {"content": "..."}}]?
          }``

        We tolerate schema drift — missing fields just drop details silently.
        """
        if not isinstance(event, dict):
            return
        case, msg = _event_payload(event)
        if case == "ping":
            logger.debug("Kimi groups: keepalive ping")
            return
        if case in ("reconnect", "typing"):
            logger.debug("Kimi groups: ignoring %s event", case)
            return
        if case != "chatMessage" or not msg:
            logger.info("Kimi groups: unsupported event shape, skipping: %.200r", event)
            return

        if not _chat_message_is_complete(_field(msg, "status")):
            logger.info(
                "Kimi groups: skipping incomplete chatMessage status=%r",
                _field(msg, "status"),
            )
            return

        chat_id = _field(msg, "chatId", "chat_id")
        message_id = _field(msg, "messageId", "message_id")
        if not (chat_id and message_id):
            logger.info("Kimi groups: event missing chat_id/message_id, skipping: %.200r", msg)
            return

        # Probe (H-C): per-room message_id timing for post-hoc correlation
        # against conductor wall-clock. Updated BEFORE the filter chain so
        # self-drops, trust drops, and dedup still count as "what Kimi sent
        # us" — burst drops would show up as gaps in this trace.
        # Observability only — removable standalone.
        #
        # Tracker update is hoisted OUT of the DEBUG gate: if DEBUG is off at
        # process start and toggled on later, we must not falsely report
        # `first-seen` on the first post-toggle message (real bug fix).
        room_key = str(chat_id)
        this_id_str = str(message_id)
        prev_id = self._last_message_id_per_room.get(room_key)
        self._last_message_id_per_room[room_key] = this_id_str

        # Phase 0 burst-drop instrumentation: track MONOTONIC arrival time
        # per room, not the ULID-embedded timestamp. Two reasons: (1) Kimi's
        # production message ids are UUID v8 with a non-standard epoch (their
        # first 48 bits aren't unix-ms — captured deltas show ~16× the
        # wall-clock interval), so id-derived deltas are nonsense in
        # production. (2) ``time.monotonic()`` is the right primitive for
        # "how long since last delivery": it never goes backward, never
        # jumps on NTP sync or VM suspend/resume, and is immune to leap-
        # second adjustments. Codex review #1 + Kimi review #2 (#58)
        # independently flagged ``time.time()``-based deltas as a false-
        # positive risk under clock skew. The log line's own leading ISO
        # timestamp gives operators wall-clock correlation; the embedded
        # delta is honestly "process-time since last delivery".
        now_mono_s = time.monotonic()
        prev_arrival = self._last_arrival_time_per_room.get(room_key)
        self._last_arrival_time_per_room[room_key] = now_mono_s

        arrival_delta_s: Optional[float] = None
        if prev_arrival is not None:
            arrival_delta_s = now_mono_s - prev_arrival
            if (
                self._burst_drop_gap_log_threshold_s > 0
                and arrival_delta_s >= self._burst_drop_gap_log_threshold_s
            ):
                # Include since-last-reconnect so operators can correlate gap
                # candidates against reconnect events without grepping two
                # log streams. ``-1.0`` sentinel = "no reconnect observed
                # yet this process" (cold start before first chatMessage).
                # Both timestamps are monotonic so the subtraction is safe.
                if self._group_subscribe_last_reconnect_ts > 0:
                    since_reconnect_s = (
                        now_mono_s - self._group_subscribe_last_reconnect_ts
                    )
                    correlation = f"since_reconnect_s={since_reconnect_s:.1f}"
                else:
                    correlation = "since_reconnect_s=N/A"
                logger.info(
                    "Kimi groups: gap candidate room=%s id=%s prev=%s "
                    "delta_s=%.1f (>=%.1fs threshold) %s connect#=%d",
                    chat_id, message_id, prev_id, arrival_delta_s,
                    self._burst_drop_gap_log_threshold_s,
                    correlation,
                    self._group_subscribe_reconnect_count,
                )

        # Probe-3 DEBUG path uses ULID-decoded magnitudes for backward
        # compatibility with the existing test fixtures. In production these
        # consistently return None because Kimi sends UUIDs, so the DEBUG
        # log path falls through to "first-seen" — that's expected and
        # not the gap-candidate signal anyway.
        if logger.isEnabledFor(logging.DEBUG):
            # Sample-rate gate: only count + conditionally emit inside DEBUG,
            # so INFO and above pay nothing for the per-room counter dict
            # either.
            count = self._probe_msg_id_room_counts.get(room_key, 0) + 1
            self._probe_msg_id_room_counts[room_key] = count
            if count % self._probe_msg_id_sample_rate == 0:
                this_ts = _ulid_time_ms(this_id_str)
                prev_ts = _ulid_time_ms(prev_id) if prev_id else None
                if this_ts is not None and prev_ts is not None:
                    legacy_delta_ms = this_ts - prev_ts
                    logger.debug(
                        "Kimi groups: message_id timing room=%s id=%s prev=%s delta_ms=%d",
                        chat_id, message_id, prev_id, legacy_delta_ms,
                    )
                else:
                    logger.debug(
                        "Kimi groups: message_id first-seen room=%s id=%s",
                        chat_id, message_id,
                    )

        sender = msg.get("sender") or {}
        sender_id = (
            sender.get("id") if isinstance(sender, dict) else None
        ) or _field(msg, "senderId", "sender_id")
        sender_short_id = (
            sender.get("short_id") if isinstance(sender, dict) else None
        ) or (
            sender.get("shortId") if isinstance(sender, dict) else None
        ) or _field(msg, "senderShortId", "sender_short_id")
        sender_name = (
            sender.get("name") if isinstance(sender, dict) else None
        ) or _field(msg, "senderName", "sender_name")

        # Self-message filter.
        if sender_id and self._me_id and sender_id == self._me_id:
            return
        if sender_short_id and self._me_short_id and sender_short_id == self._me_short_id:
            return

        # Startup grace — ignore events older than startup_ts - grace.
        sent_at = _field(msg, "sentAt", "sent_at", "createTime", "create_time")
        if sent_at:
            event_ts = _parse_iso8601(sent_at)
            if event_ts and event_ts < (self._startup_ts - self._startup_grace_s):
                logger.info(
                    "Kimi groups: skipping stale event %s (sent_at=%s)",
                    message_id,
                    sent_at,
                )
                return

        block_text, media_urls, media_types = _extract_blocks_payload(msg)
        message_role = _field(msg, "role", "messageRole", "message_role")

        # Resolve text in three stages:
        #   1. inline body (blocks → text field) from the Subscribe event
        #   2. hydration if inline body is empty (Subscribe sometimes ships
        #      preview-only events for long messages — empty blocks/text and
        #      a truncated `summary`)
        #   3. summary as a last-resort fallback for graceful degradation
        #
        # H-B fix (2026-04-26): the prior fallback chain
        #     text = block_text or text_field or summary or ""
        # treated `summary` as equivalent to inline body, so a truthy
        # 50-char preview bypassed the `if not text` hydration gate and
        # the agent only ever saw the truncated server-side preview.
        # We now gate hydration on inline body being empty regardless of
        # summary, and only fall through to summary if hydration is
        # unavailable or fails. Production confirmation: Probe 2 log at
        # 2026-04-26 11:21:36 BST showed
        #   blocks=0, text=0, summary=50, chose=summary, miss_candidate=none
        # for a ~150-char inbound message, with the agent answering against
        # the 50-char preview rather than the full body.
        text_field = _field(msg, "text") or ""
        if not isinstance(text_field, str):
            text_field = ""
        summary_field = _field(msg, "summary") or ""
        if not isinstance(summary_field, str):
            summary_field = ""
        inline_text = block_text or text_field
        text = inline_text

        hydrated: Optional[Dict[str, Any]] = None
        # `hydration_state` drives Probe 2's `hydrated=` field. Five values
        # so operators can distinguish operationally distinct states when
        # debugging "why didn't hydration fire / what did it produce":
        #   skipped:inline   — inline body present, hydration not needed
        #   skipped:disabled — _hydrate_missing_text=False (operator policy)
        #   true             — hydration ran AND populated text non-empty
        #   false            — hydration ran but raised, returned empty,
        #                      or returned a payload that yielded no text
        #
        # `text_from_hydration` is the authoritative "did hydration provide
        # the chosen text?" flag — Probe 2 uses it instead of string-equality
        # against summary, so a hydrated body that happens to equal the
        # summary preview verbatim is still correctly labeled `chose=hydrated`.
        text_from_hydration = False
        if inline_text:
            hydration_state = "skipped:inline"
        elif not self._hydrate_missing_text:
            hydration_state = "skipped:disabled"
        else:
            hydration_state = "false"
            try:
                hydrated = await self._fetch_group_message(str(chat_id), str(message_id))
            except KimiAdapterError as exc:
                logger.debug(
                    "Kimi groups: failed to hydrate message %s/%s: %s",
                    chat_id,
                    message_id,
                    exc,
                )
            else:
                if hydrated:
                    hydrated_text, hydrated_urls, hydrated_types = _extract_blocks_payload(
                        hydrated
                    )
                    new_text = hydrated_text or _field(hydrated, "text") or ""
                    if new_text:
                        text = new_text
                        text_from_hydration = True
                        hydration_state = "true"
                    # else: hydrated payload was truthy but yielded no text
                    # (e.g. wrapper-only). Keep hydration_state="false" so
                    # Probe 2 doesn't claim a hydration win on an empty
                    # payload (Codex MINOR #2 — Fix E).
                    media_urls.extend(url for url in hydrated_urls if url not in media_urls)
                    media_types.extend(hydrated_types)
                    sender_id = sender_id or _field(hydrated, "senderId", "sender_id")
                    sender_short_id = sender_short_id or _field(
                        hydrated, "senderShortId", "sender_short_id"
                    )
                    sender_name = sender_name or _field(
                        hydrated, "senderName", "sender_name"
                    )
                    message_role = message_role or _field(
                        hydrated, "role", "messageRole", "message_role"
                    )

        # Final fallback: hydration unavailable, failed, or returned an
        # empty body → use summary as better-than-nothing so the agent
        # at least sees SOMETHING. Without this, a degraded Kimi backend
        # would silently drop messages we could have shown a preview of.
        # Annotate so the agent knows the body is truncated and can
        # acknowledge rather than confidently answer against half a
        # sentence (Fix B — same H-B failure mode as the original bug,
        # just less frequent).
        if not text and summary_field:
            text = (
                "[message truncated — full text unavailable, "
                "preview only]\n"
                + summary_field
            )

        # Probe (H-B): which source populated `text`, candidate lengths,
        # and whether hydration ran. Logged after the full resolution
        # cascade so the chosen source reflects the final outcome
        # (blocks / text / hydrated / summary / none).
        #
        # `miss_candidate` flags non-chosen *raw-event* candidates whose
        # length exceeds the chosen one — the precise hydration-miss
        # signature. After the H-B fix this should never report
        # `chose=summary, miss_candidate=none` for a long inbound; if it
        # does, hydration was disabled or failed silently.
        if logger.isEnabledFor(logging.DEBUG):
            # Order matters: hydration check before block/text comparisons
            # would mislabel a hydrated body that equals the inline candidate
            # verbatim. Inline candidates win when they actually populated
            # the chosen text (text_from_hydration is False in that case).
            if text_from_hydration:
                chosen = "hydrated"
            elif block_text and text == block_text:
                chosen = "blocks"
            elif text_field and text == text_field:
                chosen = "text"
            elif summary_field and summary_field in (text or ""):
                # Fix B prepends a truncation marker, so `text` is no
                # longer == summary_field — use containment.
                chosen = "summary"
            else:
                chosen = "none"
            candidate_lens = {
                "blocks": len(block_text or ""),
                "text": len(text_field),
                "summary": len(summary_field),
            }
            # Length oracle covers raw-event candidates only; `hydrated`
            # is reported via the `hydrated=` field rather than length
            # (the hydrated payload may itself be a wrapper of arbitrary
            # shape and isn't directly comparable to inline candidates).
            if chosen in candidate_lens:
                chosen_len = candidate_lens[chosen]
            else:
                # `hydrated` or `none` — compare misses against the chosen
                # text's length so a longer inline candidate still flags.
                chosen_len = len(text or "")
            miss_candidates = [
                name for name, length in candidate_lens.items()
                if name != chosen and length > chosen_len
            ]
            miss_str = ",".join(miss_candidates) if miss_candidates else "none"
            logger.debug(
                "Kimi groups: text source for %s/%s — blocks=%d, text=%d, summary=%d, chose=%s, hydrated=%s, miss_candidate=%s",
                chat_id, message_id,
                candidate_lens["blocks"], candidate_lens["text"],
                candidate_lens["summary"], chosen, hydration_state, miss_str,
            )

        if sender_id and self._me_id and sender_id == self._me_id:
            return
        if sender_short_id and self._me_short_id and sender_short_id == self._me_short_id:
            return

        # Trusted-sender allowlist — authoritative bypass of role filter
        is_trusted = False
        if sender_short_id and sender_short_id in self._group_trusted_senders:
            is_trusted = True
        elif sender_id and sender_id in self._group_trusted_senders:
            is_trusted = True

        # Role filter — gated by allow_bot_senders policy
        if not is_trusted and not _chat_message_is_user_role(message_role):
            policy = self._group_allow_bot_senders
            if policy == "off":
                logger.info(
                    "Kimi groups: dropping non-user message %s/%s role=%r (group_allow_bot_senders=off)",
                    chat_id, message_id, message_role,
                )
                return
            elif policy == "trusted_only":
                # INFO with redacted sender: operators need a grep-able
                # signal that messages are being dropped (e.g. after
                # forgetting to add a new teammate to group_trusted_senders).
                # Full short_ids at INFO would leak hundreds of user
                # identifiers into log aggregators in a kimi-claw group
                # where every user message has role='assistant', so we
                # redact to prefix + 4 chars. Full identifiers are still
                # available at DEBUG via the raw event dumps.
                logger.info(
                    "Kimi groups: dropping non-user message %s/%s role=%r sender=%s (not in group_trusted_senders)",
                    chat_id, message_id, message_role,
                    _redact_sender(sender_short_id or sender_id),
                )
                return
            elif policy == "mentions":
                if not self._is_mention_of_me(msg):
                    logger.info(
                        "Kimi groups: dropping non-user message %s/%s role=%r (group_allow_bot_senders=mentions, no @mention of us)",
                        chat_id, message_id, message_role,
                    )
                    return
                # falls through
            elif policy == "all":
                pass  # falls through

        if not text and not media_urls:
            logger.debug(
                "Kimi groups: message %s/%s has no dispatchable content",
                chat_id,
                message_id,
            )
            return

        # Dedup (chat_id, message_id) after hydration/content checks. If a
        # lightweight event fails hydration, a later replay can still deliver
        # the full message instead of being suppressed as already processed.
        if self._dedup_is_duplicate("group", str(chat_id), str(message_id)):
            return

        thread_id = _field(msg, "threadId", "thread_id")
        if not thread_id and hydrated:
            thread_id = _field(hydrated, "threadId", "thread_id")
        reply_to = msg.get("reply_to") or msg.get("replyTo") or {}
        reply_to_message_id = reply_to.get("message_id") if isinstance(reply_to, dict) else None
        reply_to_text = reply_to.get("text") if isinstance(reply_to, dict) else None

        attachments = msg.get("attachments") or []
        for att in attachments if isinstance(attachments, list) else []:
            if not isinstance(att, dict):
                continue
            url = att.get("url")
            if url:
                media_urls.append(url)
                media_types.append(att.get("type", "file"))

        if media_urls:
            media_urls, media_types = await self._resolve_kimi_file_media(
                media_urls,
                media_types,
                message_id=str(message_id),
            )

        # Mention gate: if configured to require mentions and the message
        # doesn't reference us, ignore.  Supports both numeric id and
        # short_id.  Rooms in ``kimi_free_response_chats`` bypass this —
        # historically used for two purposes:
        #
        #   1. Working around DM misdetection (Kimi delivers 1:1 DMs as
        #      ``room:<uuid>`` indistinguishable from groups at the wire
        #      envelope we consume).  v2.2.0 introduces the
        #      ``kimi_dm_autodetect`` flag which auto-bypasses 2-member
        #      rooms via ``_is_dm_room`` — operators no longer need to
        #      list DM UUIDs manually when the flag is on.
        #   2. Explicit operator policy ("this small trusted group room
        #      should be free-response despite the global mention gate").
        #      NOT auto-detectable; remains served by
        #      ``kimi_free_response_chats``.
        #
        # The detector is the LAST bypass chance — only consulted if the
        # other checks already decided to drop.  This preserves the fast
        # path: messages headed to dispatch never incur the detector RPC,
        # only messages about to be silenced.
        if (
            self._group_require_mention
            and chat_id not in self._group_require_mention_exempt_rooms
            and not self._is_mention_of_me(msg)
        ):
            if self._kimi_dm_autodetect:
                is_dm = await self._is_dm_room(chat_id)
                if is_dm is True:
                    logger.debug(
                        "Kimi groups: auto-detected DM, bypassing mention "
                        "gate for %s/%s",
                        chat_id, message_id,
                    )
                    # Fall through to dispatch.
                else:
                    # is_dm is False (group) or None (classification failed
                    # — fail closed and let the legacy gate fire).
                    logger.info(
                        "Kimi groups: dropping message %s/%s "
                        "(group_require_mention=true, no @mention, "
                        "auto-detect=%s)",
                        chat_id, message_id,
                        "group" if is_dm is False else "unknown",
                    )
                    return
            else:
                logger.info(
                    "Kimi groups: dropping message %s/%s "
                    "(group_require_mention=true, no @mention of us)",
                    chat_id, message_id,
                )
                return

        # Preserve thread identity in the dispatched chat_id so the gateway's
        # session routing keeps distinct threads inside one Kimi room in
        # separate Hermes sessions. Without this, every thread in a busy room
        # collapses to the same session key and threads crosstalk.
        if thread_id:
            chat_id_prefixed = f"{_CHATID_ROOM_PREFIX}{chat_id}/{thread_id}"
        else:
            chat_id_prefixed = f"{_CHATID_ROOM_PREFIX}{chat_id}"

        event_obj = self._build_message_event(
            kind="group",
            text=str(text),
            message_id=str(message_id),
            chat_id=chat_id_prefixed,
            chat_name=None,  # populated lazily via get_chat_info if needed
            user_id=sender_id or (f"kimi:{sender_short_id}" if sender_short_id else None),
            user_name=sender_name or sender_short_id,
            thread_id=thread_id,
            reply_to_message_id=reply_to_message_id,
            reply_to_text=reply_to_text,
            media_urls=media_urls,
            media_types=media_types,
            raw=msg,
        )
        await self.handle_message(event_obj)

    async def _fetch_group_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Fetch a full message wrapper when Subscribe only carries a summary.

        Sets ``max_pages=2`` so that if Kimi paginates around a tight
        ``start=end=message_id`` window (Codex impl #4 in #18), we still
        find the message rather than silently returning ``None``. Safe
        because ``list_group_messages`` no longer echoes the anchor on
        follow-up pages (Kimi review #58 fix above) — page 1 follows the
        opaque cursor only, so the second request has the same effective
        scope without anchor/cursor ambiguity.
        """
        wrappers = await self.list_group_messages(
            chat_id,
            limit=20,
            start_message_id=message_id,
            end_message_id=message_id,
            include_start_message=True,
            include_end_message=True,
            max_pages=2,
        )
        for wrapper in wrappers:
            message = wrapper.get("message") if isinstance(wrapper.get("message"), dict) else {}
            candidate_id = (
                _field(message, "id", "messageId", "message_id")
                or _field(wrapper, "messageId", "message_id")
            )
            if str(candidate_id) != str(message_id):
                continue
            merged = dict(message)
            for source_key, target_key in (
                ("senderId", "senderId"),
                ("sender_id", "senderId"),
                ("senderShortId", "senderShortId"),
                ("sender_short_id", "senderShortId"),
                ("senderName", "senderName"),
                ("sender_name", "senderName"),
            ):
                value = wrapper.get(source_key)
                if value is not None and target_key not in merged:
                    merged[target_key] = value
            return merged
        return None

    async def _resolve_kimi_file_media(
        self,
        media_urls: List[str],
        media_types: List[str],
        *,
        message_id: str,
    ) -> Tuple[List[str], List[str]]:
        resolved_urls: List[str] = []
        resolved_types: List[str] = []
        for idx, uri in enumerate(media_urls):
            media_type = media_types[idx] if idx < len(media_types) else ""
            if isinstance(uri, str) and uri.startswith("kimi-file://"):
                resolved = await self._resolve_kimi_file_uri(uri, message_id=message_id)
                if resolved:
                    resolved_urls.append(resolved["localPath"])
                    resolved_types.append(
                        resolved.get("contentType") or media_type or "application/octet-stream"
                    )
                    continue
            resolved_urls.append(uri)
            resolved_types.append(media_type)
        return resolved_urls, resolved_types

    async def _resolve_kimi_file_uri(
        self,
        uri: str,
        *,
        message_id: str,
    ) -> Optional[Dict[str, Any]]:
        file_id = _parse_kimi_file_id(uri)
        if not file_id:
            logger.warning("Kimi groups: invalid kimi-file URI in message %s: %s", message_id, uri)
            return None

        self._file_download_dir.mkdir(parents=True, exist_ok=True)
        existing = self._find_cached_kimi_file(file_id)
        if existing:
            return existing
        metadata_url = _file_metadata_endpoint(self._kimiapi_host, file_id)
        try:
            async with self._session_for_current_loop() as session, session.get(
                metadata_url,
                headers={"X-Kimi-Bot-Token": self._bot_token, "Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=self._file_timeout_s),
            ) as resp:
                raw = await resp.read()
                if resp.status in (401, 403):
                    raise KimiAuthError(f"kimi-file metadata auth failed HTTP {resp.status}")
                if resp.status >= 400:
                    logger.warning(
                        "Kimi groups: kimi-file metadata failed file_id=%s HTTP %s",
                        file_id,
                        resp.status,
                    )
                    return None
                metadata = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception as exc:
            logger.warning("Kimi groups: kimi-file metadata failed file_id=%s: %s", file_id, exc)
            return None

        if not isinstance(metadata, dict):
            return None
        meta = metadata.get("meta") if isinstance(metadata.get("meta"), dict) else {}
        name = str(_field(meta, "name") or file_id)
        content_type = str(_field(meta, "contentType", "content_type") or "application/octet-stream")
        blob = metadata.get("blob") if isinstance(metadata.get("blob"), dict) else {}
        download_url = _field(blob, "signUrl", "sign_url") or self._preview_download_url(metadata)
        if not isinstance(download_url, str) or not download_url:
            logger.warning("Kimi groups: kimi-file metadata has no download URL file_id=%s", file_id)
            return None

        local_name = f"{file_id}_{_sanitize_kimi_file_name(name)}"
        local_path = self._file_download_dir / local_name
        try:
            async with self._session_for_current_loop() as session, session.get(
                download_url,
                timeout=aiohttp.ClientTimeout(total=self._file_timeout_s),
            ) as resp:
                data = await resp.read()
                if resp.status >= 400 or not data:
                    logger.warning(
                        "Kimi groups: kimi-file download failed file_id=%s HTTP %s",
                        file_id,
                        resp.status,
                    )
                    return None
            local_path.write_bytes(data)
        except Exception as exc:
            logger.warning("Kimi groups: kimi-file download failed file_id=%s: %s", file_id, exc)
            return None

        return {
            "fileId": file_id,
            "name": name,
            "contentType": content_type,
            "localPath": str(local_path),
        }

    def _find_cached_kimi_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        prefix = f"{file_id}_"
        try:
            candidates = sorted(self._file_download_dir.iterdir(), key=lambda p: p.name)
        except OSError:
            return None
        for path in candidates:
            if path.is_file() and path.name.startswith(prefix) and path.stat().st_size > 0:
                return {
                    "fileId": file_id,
                    "name": path.name[len(prefix):],
                    "contentType": _infer_mime_type(str(path)),
                    "localPath": str(path),
                }
        return None

    def _preview_download_url(self, metadata: Dict[str, Any]) -> Optional[str]:
        parse_job = metadata.get("parseJob") or metadata.get("parse_job")
        if not isinstance(parse_job, dict):
            return None
        result = parse_job.get("result")
        image = result.get("image") if isinstance(result, dict) else None
        thumbnail = image.get("thumbnail") if isinstance(image, dict) else None
        if isinstance(thumbnail, dict):
            url = _field(thumbnail, "previewUrl", "preview_url")
            return url if isinstance(url, str) else None
        return None

    async def _send_group(
        self,
        room_id: str,
        content: str,
        reply_to: Optional[str],
        thread_id: Optional[str],
        metadata: Dict[str, Any],
    ) -> SendResult:
        """POST unary ``SendMessage`` with text + optional attachments.

        Kimi Claw v0.25.0's ``SendMessageRequest`` wire protocol exposes only
        ``chatId`` and ``blocks``. Two caller-visible affordances therefore
        don't round-trip today:

        - ``thread_id`` (either from ``room:<uuid>/<tid>`` or
          ``metadata["thread_id"]``): collapses to the underlying room. The
          reply shows up at the room level instead of in the intended thread.
        - ``metadata["mentions"]``: ignored. The call falls through to the
          plain-text block, which means Kimi renders ``@u_foo`` as ordinary
          characters rather than a mention pill — no notification, and
          mention-gated bots won't see the reply.

        Both were previously silently dropped. They now emit a one-shot
        WARNING so operators can see the gap. Explicit wire support for
        either will be added here when the Kimi Claw surface script confirms
        the field shapes (thread_id on ``SendMessageRequest``; a mention
        block variant alongside ``text`` / ``resourceLink``).
        """
        del reply_to  # SendMessageRequest has no reply-to field on the wire.
        if thread_id and not self._warned_outbound_thread_drop:
            logger.warning(
                "Kimi groups: outbound thread_id=%r on room=%r is not "
                "representable on SendMessageRequest (Kimi Claw v0.25.0 — "
                "chatId + blocks only). Reply will target the underlying "
                "room, not the thread. Further drops suppressed.",
                thread_id, room_id,
            )
            self._warned_outbound_thread_drop = True

        mentions = metadata.get("mentions") if metadata else None
        if isinstance(mentions, list) and mentions and not self._warned_outbound_mentions_drop:
            logger.warning(
                "Kimi groups: metadata.mentions=%r on room=%r falls through "
                "to plain text — Kimi Claw v0.25.0's block protobuf has no "
                "confirmed mention variant yet. Reply renders @short_id as "
                "text, producing no mention pill and no push notification. "
                "Further drops suppressed.",
                mentions, room_id,
            )
            self._warned_outbound_mentions_drop = True

        blocks: List[Dict[str, Any]] = []
        if content:
            blocks.append(_build_text_block(content))
        if metadata and "attachments" in metadata:
            for attachment in metadata["attachments"]:
                if not isinstance(attachment, dict):
                    continue
                uri = _field(attachment, "uri", "url", "downloadUrl", "download_url")
                if isinstance(uri, str) and uri:
                    blocks.append(_build_resource_link_block(attachment))

        if not blocks:
            return SendResult(
                success=False,
                error="Kimi: SendMessage requires text or attachments",
                retryable=False,
            )

        body: Dict[str, Any] = {"chatId": room_id, "blocks": blocks}
        resp = await self._rpc_unary("SendMessage", body)
        return SendResult(
            success=True,
            message_id=resp.get("message_id") or resp.get("messageId"),
            raw_response=resp,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Connect RPC — unary + envelope streaming
    # ──────────────────────────────────────────────────────────────────────

    async def _rpc_unary(
        self,
        method: str,
        body: Dict[str, Any],
        *,
        timeout_s: float = _RPC_TIMEOUT_S,
    ) -> Dict[str, Any]:
        """POST ``/api-ws/{service}/{method}`` with JSON body, return response.

        Raises:
            KimiAuthError     on HTTP 401/403.
            KimiTransientError on HTTP 429/5xx/network errors.
            KimiRpcError      on other 4xx with JSON error body.
        """
        url = f"{self._base_url}/{_IM_SERVICE}/{method}"
        headers = self._http_headers(streaming=False)
        try:
            async with self._session_for_current_loop() as session, session.post(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                raw = await resp.read()
                if resp.status in (401, 403):
                    raise KimiAuthError(
                        f"{method}: auth failed (HTTP {resp.status}): {raw[:200]!r}"
                    )
                if resp.status == 429 or 500 <= resp.status < 600:
                    raise KimiTransientError(
                        f"{method}: HTTP {resp.status}: {raw[:200]!r}"
                    )
                if resp.status >= 400:
                    err_msg = raw.decode("utf-8", errors="replace")[:500]
                    raise KimiRpcError(f"{method}: HTTP {resp.status}: {err_msg}")
                try:
                    return json.loads(raw.decode("utf-8")) if raw else {}
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise KimiProtocolError(
                        f"{method}: invalid JSON response: {exc}"
                    )
        except aiohttp.ClientError as exc:
            raise KimiTransientError(f"{method}: network error: {exc}")

    def _http_headers(self, *, streaming: bool) -> Dict[str, str]:
        """Shared HTTP headers for all Connect RPCs."""
        headers = {
            "Content-Type": (
                "application/connect+json" if streaming else "application/json"
            ),
            "Connect-Protocol-Version": "1",
            "Accept-Encoding": "identity",
            "User-Agent": "hermes-kimi-adapter/1.0",
        }
        if streaming:
            headers["Accept"] = "application/connect+json"
        headers.update(_runtime_headers(
            bot_token=self._bot_token,
            claw_version=self._claw_version,
            openclaw_version=self._openclaw_version,
            claw_id=self._claw_id,
            openclaw_plugins=self._openclaw_plugins,
            openclaw_skills=self._openclaw_skills,
        ))
        return headers

    def _encode_envelope(self, payload: bytes, *, end_stream: bool = False) -> bytes:
        """Encode one outbound Connect envelope: ``[flag:1B][len:4B BE][body]``."""
        flag = _CONNECT_FLAG_END_STREAM if end_stream else 0
        return bytes([flag]) + struct.pack(">I", len(payload)) + payload

    async def _connect_envelope_parser(
        self,
        reader: Any,  # aiohttp.StreamReader
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield JSON dicts from a chunked Connect streaming body.

        Handles:
          - partial envelopes (readexactly blocks correctly)
          - compressed flag bit (rejected — we negotiate uncompressed)
          - end-stream frame with optional ``error`` payload (raises appropriate
            KimiAuthError / KimiRpcError, or returns cleanly)
        """
        while True:
            try:
                header = await reader.readexactly(5)
            except asyncio.IncompleteReadError:
                # Stream closed mid-envelope — treat as transient.
                raise KimiTransientError("Subscribe stream closed mid-envelope")
            except aiohttp.ClientPayloadError as exc:
                raise KimiTransientError(f"Subscribe payload error: {exc}")

            flag = header[0]
            length = struct.unpack(">I", header[1:5])[0]
            # Defensive cap: the length prefix is 4 bytes big-endian (up to
            # 4 GB). An unbounded readexactly here would OOM on a hostile or
            # buggy upstream. Mirror the WS max-frame cap.
            if length > _WS_MAX_FRAME_SIZE:
                raise KimiProtocolError(
                    f"envelope length {length} exceeds max frame size "
                    f"{_WS_MAX_FRAME_SIZE}"
                )
            payload = b""
            if length:
                try:
                    payload = await reader.readexactly(length)
                except asyncio.IncompleteReadError:
                    raise KimiTransientError("Subscribe truncated envelope body")
                except aiohttp.ClientPayloadError as exc:
                    raise KimiTransientError(f"Subscribe payload error: {exc}")

            if flag & _CONNECT_FLAG_COMPRESSED:
                raise KimiProtocolError("Kimi sent compressed frame (unsupported)")

            try:
                msg = json.loads(payload.decode("utf-8")) if payload else {}
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise KimiProtocolError(f"malformed envelope JSON: {exc}")

            if flag & _CONNECT_FLAG_END_STREAM:
                err = msg.get("error") if isinstance(msg, dict) else None
                if err:
                    code = err.get("code", "unknown")
                    message = err.get("message") or err.get("details") or ""
                    if code in ("unauthenticated", "permission_denied"):
                        raise KimiAuthError(f"{code}: {message}")
                    raise KimiRpcError(f"{code}: {message}")
                return  # clean end-of-stream

            if isinstance(msg, dict):
                yield msg

    # ──────────────────────────────────────────────────────────────────────
    # MessageEvent synthesis + dedup
    # ──────────────────────────────────────────────────────────────────────

    def _build_message_event(
        self,
        *,
        kind: str,  # "dm" | "group"
        text: str,
        message_id: str,
        chat_id: str,
        chat_name: Optional[str],
        user_id: Optional[str],
        user_name: Optional[str],
        thread_id: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        reply_to_text: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
        raw: Any = None,
    ) -> MessageEvent:
        msg_type = MessageType.TEXT
        if text.strip().startswith("/"):
            msg_type = MessageType.COMMAND
        if media_urls:
            # Best-effort mapping from MIME prefix to MessageType.
            first = (media_types or [""])[0].lower()
            if first.startswith("image"):
                msg_type = MessageType.PHOTO
            elif first.startswith("video"):
                msg_type = MessageType.VIDEO
            elif first.startswith("audio"):
                msg_type = MessageType.AUDIO
            else:
                msg_type = MessageType.DOCUMENT

        return MessageEvent(
            text=text,
            message_type=msg_type,
            source=self.build_source(
                chat_id=chat_id,
                chat_name=chat_name,
                chat_type=("dm" if kind == "dm" else "group"),
                user_id=user_id,
                user_name=user_name,
                thread_id=thread_id,
            ),
            raw_message=raw,
            message_id=message_id,
            media_urls=list(media_urls or []),
            media_types=list(media_types or []),
            reply_to_message_id=reply_to_message_id,
            reply_to_text=reply_to_text,
            auto_skill=self._auto_skill,
            channel_prompt=self._channel_prompt,
            internal=False,
        )

    def _dedup_is_duplicate(self, kind: str, chat_id: str, message_id: str) -> bool:
        key = (f"{kind}:{chat_id}", str(message_id))
        if key in self._processed_set:
            return True
        # Evict oldest when full.
        if len(self._processed) >= self._processed.maxlen:
            old = self._processed[0]
            self._processed_set.discard(old)
        self._processed.append(key)
        self._processed_set.add(key)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Module helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_iso8601(text: str) -> Optional[float]:
    """Parse an ISO-8601 timestamp to unix-seconds, or None on failure."""
    if not isinstance(text, str) or not text:
        return None
    try:
        from datetime import datetime
        # Normalize 'Z' suffix to +00:00 for fromisoformat.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Standalone send helper (for send_message_tool / cron paths outside gateway)
# ──────────────────────────────────────────────────────────────────────────────

async def send_kimi_message(
    config: PlatformConfig,
    chat_id: str,
    text: str,
    *,
    thread_id: Optional[str] = None,
    media_paths: Optional[List[str]] = None,
) -> SendResult:
    """Send a message via Kimi without instantiating the full adapter.

    Used by cron delivery and ``send_message_tool`` when a live gateway
    adapter isn't available. Supports only group rooms (``room:<id>``) for
    now — DM sends require an active WS session.
    """
    if not chat_id.startswith(_CHATID_ROOM_PREFIX):
        return SendResult(
            success=False,
            error="Kimi: standalone send supports only group rooms",
            retryable=False,
        )
    # Mirror the ``__init__`` chain so the standalone send path (used by
    # cron delivery and ``send_message_tool`` when no live adapter exists)
    # resolves ``${KIMI_BOT_TOKEN}`` literals the same way the live adapter
    # does. Without this, a config.yaml of the form
    # ``platforms.kimi.token: ${KIMI_BOT_TOKEN}`` would 401 silently on
    # cron-driven kimi deliveries while the live bot path worked fine.
    token = (
        _resolve_env_template(config.token)
        or _resolve_env_template(config.extra.get("bot_token", ""))
        or os.getenv("KIMI_BOT_TOKEN", "")
    )
    if not token:
        return SendResult(success=False, error="Kimi: no bot_token configured", retryable=False)
    base_url = config.extra.get("base_url", _DEFAULT_BASE_URL).rstrip("/")
    kimiapi_host = config.extra.get(
        "kimiapi_host",
        config.extra.get("kimiapiHost", _DEFAULT_KIMIAPI_HOST),
    )
    upload_url = config.extra.get("upload_url", _upload_endpoint(kimiapi_host))
    file_timeout_s = float(config.extra.get("file_timeout_s", _FILE_UPLOAD_TIMEOUT_S_DEFAULT))
    room_and_thread = chat_id[len(_CHATID_ROOM_PREFIX):]
    if "/" in room_and_thread:
        room_id, inline_thread = room_and_thread.split("/", 1)
    else:
        room_id, inline_thread = room_and_thread, None
    effective_thread = thread_id or inline_thread
    if effective_thread:
        logger.warning(
            "Kimi groups: standalone send_kimi_message thread_id=%r on "
            "room=%r is not representable on SendMessageRequest (Kimi "
            "Claw v0.25.0 — chatId + blocks only). Reply will target the "
            "underlying room, not the thread.",
            effective_thread, room_id,
        )
    media_paths = list(media_paths or [])
    blocks: List[Dict[str, Any]] = []
    if text:
        blocks.append(_build_text_block(text))
    url = f"{base_url}/{_IM_SERVICE}/SendMessage"
    headers = {
        "Content-Type": "application/json",
        "Connect-Protocol-Version": "1",
        "User-Agent": "hermes-kimi-adapter/1.0",
    }
    headers.update(_runtime_headers(
        bot_token=token,
        claw_version=config.extra.get("claw_version", _GROUP_GATE_DEFAULTS["claw_version"]),
        openclaw_version=config.extra.get(
            "openclaw_version", _GROUP_GATE_DEFAULTS["openclaw_version"]
        ),
        claw_id=config.extra.get("claw_id"),
        openclaw_plugins=config.extra.get(
            "openclaw_plugins", _GROUP_GATE_DEFAULTS["openclaw_plugins"]
        ),
        openclaw_skills=config.extra.get(
            "openclaw_skills", _GROUP_GATE_DEFAULTS["openclaw_skills"]
        ),
    ))
    try:
        # trust_env=True: see KimiAdapter.connect() for rationale.
        async with aiohttp.ClientSession(trust_env=True) as session:
            if media_paths:
                uploaded = await _upload_kimi_files(
                    session,
                    paths=media_paths,
                    bot_token=token,
                    upload_url=upload_url,
                    timeout_s=file_timeout_s,
                )
                blocks.extend(_build_resource_link_block(item) for item in uploaded)
            if not blocks:
                return SendResult(
                    success=False,
                    error="Kimi: SendMessage requires text or attachments",
                    retryable=False,
                )
            body: Dict[str, Any] = {"chatId": room_id, "blocks": blocks}
            async with session.post(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=_RPC_TIMEOUT_S),
            ) as resp:
                raw = await resp.read()
                if resp.status == 401 or resp.status == 403:
                    return SendResult(
                        success=False,
                        error=f"auth failed HTTP {resp.status}",
                        retryable=False,
                    )
                if resp.status >= 400:
                    return SendResult(
                        success=False,
                        error=f"HTTP {resp.status}: {raw[:200]!r}",
                        retryable=(resp.status >= 500 or resp.status == 429),
                    )
                try:
                    data = json.loads(raw.decode("utf-8")) if raw else {}
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    return SendResult(success=False, error=f"bad JSON: {exc}", retryable=False)
                return SendResult(
                    success=True,
                    message_id=data.get("message_id") or data.get("messageId"),
                    raw_response=data,
                )
    except asyncio.TimeoutError:
        # Mirrors the live-adapter v2.1.4 fix at line ~1807.  Standalone path
        # also hits the kimi.com response-hang phenomenon (per-connection
        # server-side response held past 30s while concurrent requests succeed
        # in <1s).  Without this clause, bare asyncio.TimeoutError propagates
        # uncaught through _standalone_send and any retry layer above can
        # double-deliver — the same duplicate-send bug v2.1.4 fixed for the
        # live arm.  Return retryable=False so cron / send_message_tool don't
        # re-POST the same body.
        logger.warning(
            "Kimi: standalone SendMessage to %s timed out after %.1fs at the client; "
            "kimi.com may still have accepted the message internally. "
            "Marking non-retryable to avoid duplicate delivery.",
            chat_id, _RPC_TIMEOUT_S,
        )
        return SendResult(
            success=False,
            error=(
                f"Kimi: send timed out after {_RPC_TIMEOUT_S:.0f}s "
                "(message may have been delivered server-side; not retrying)"
            ),
            retryable=False,
        )
    except aiohttp.ClientError as exc:
        return SendResult(success=False, error=f"network error: {exc}", retryable=True)
    except KimiAuthError as exc:
        # Upload path raised: token revoked / expired. Mirror live
        # _send_uploaded_file mapping (line ~2302) — non-retryable.
        return SendResult(success=False, error=str(exc), retryable=False)
    except KimiTransientError as exc:
        # Upload path raised: 429 / 5xx from the upload endpoint.
        return SendResult(success=False, error=str(exc), retryable=True)
    except (KimiRpcError, KimiProtocolError) as exc:
        # Upload path raised: 4xx from upload, or malformed response.
        return SendResult(success=False, error=str(exc), retryable=False)


async def _standalone_send(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Registry wrapper for out-of-process ``send_message_tool`` / cron sends."""
    _ = force_document  # Kimi has no document-vs-native attachment split.
    # ``send_message_tool`` runs ``filter_media_delivery_paths()``, which yields
    # ``(path, is_voice)`` tuples, before invoking ``standalone_sender_fn``.
    # Normalize to bare paths; KimiClaw uploads every attachment generically and
    # has no voice-note concept, so the flag is dropped (mirrors the discord
    # adapter's ``for media_path, _is_voice in media_files``).
    media_paths = [media_path for media_path, _is_voice in (media_files or [])]
    result = await send_kimi_message(
        pconfig,
        chat_id,
        message,
        thread_id=thread_id,
        media_paths=media_paths,
    )
    if result.success:
        payload: Dict[str, Any] = {"success": True}
        if result.message_id:
            payload["message_id"] = result.message_id
        return payload
    return {
        "error": result.error or "Kimi standalone send failed",
        "retryable": result.retryable,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plugin registration helpers (upstream ctx.register_platform API)
# Pattern reference: upstream plugins/platforms/simplex/adapter.py
# ──────────────────────────────────────────────────────────────────────────────


def validate_config(config: PlatformConfig) -> bool:
    """Return True if a Kimi bot token is reachable.

    Mirrors the three sources the adapter ``__init__`` consults
    (see :class:`KimiAdapter`): env var ``KIMI_BOT_TOKEN``,
    ``PlatformConfig.token``, or ``PlatformConfig.extra["bot_token"]``.
    The registry calls this once during gateway setup-status display
    and again before adapter instantiation; permissive on purpose so
    env-only setups validate when the YAML/extra path is empty.

    ``${VAR}`` template strings on ``token`` / ``extra["bot_token"]`` are
    resolved against the environment before the truthiness check, so a
    YAML entry like ``token: ${KIMI_BOT_TOKEN}`` with the env var unset
    is reported as not-configured rather than as a false-positive truthy
    literal that would later 401 at ``connect()``.
    """
    if os.getenv("KIMI_BOT_TOKEN"):
        return True
    if _resolve_env_template(getattr(config, "token", None)).strip():
        return True
    extra = getattr(config, "extra", {}) or {}
    return bool(_resolve_env_template(extra.get("bot_token")).strip())


def is_connected(config: PlatformConfig) -> bool:
    """Return True if Kimi is configured (token present).

    Used by GatewayConfig.get_connected_platforms() and the setup UI.
    """
    return validate_config(config)


def _env_enablement() -> Optional[Dict[str, Any]]:
    """Read KIMI_* env vars, return a PlatformConfig.extra seed dict.

    Called during gateway startup BEFORE the adapter is constructed. Returning
    a non-None dict marks Kimi as 'env-configured' in `hermes gateway status`
    even when the adapter has not yet been instantiated.
    """
    token = os.getenv("KIMI_BOT_TOKEN")
    if not token:
        return None
    extra: Dict[str, Any] = {"bot_token": token}
    home = os.getenv("KIMI_HOME_CHANNEL")
    if home:
        extra["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("KIMI_HOME_CHANNEL_NAME", "Home"),
        }
    return extra


def _apply_yaml_config(
    yaml_cfg: Dict[str, Any], platform_cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Translate top-level ``kimiclaw:`` config.yaml keys into ``KIMI_*`` env vars.

    Env precedence is preserved via ``not os.getenv(...)`` guards — if the env
    var is already set (e.g. from ~/.hermes/.env), the YAML value is ignored.
    Lets Mimi-style profile configs put a top-level ``kimiclaw:`` block (the
    registered platform name; ``load_gateway_config()`` dispatches this hook
    via ``yaml_cfg.get(entry.name)``) directly in their profile ``config.yaml``
    instead of needing a per-profile .env.

    Called from ``gateway.config.load_gateway_config()`` after the generic
    shared-key loop and before ``_apply_env_overrides``. Returns ``None`` —
    we mutate env directly so the rest of the gateway sees the values
    through the standard env-var path.
    """
    if not isinstance(platform_cfg, dict):
        return None
    _yaml_to_env = {
        "bot_token": "KIMI_BOT_TOKEN",
        "home_channel": "KIMI_HOME_CHANNEL",
        "allowed_users": "KIMI_ALLOWED_USERS",
        "allow_all_users": "KIMI_ALLOW_ALL_USERS",
    }
    for yaml_key, env_key in _yaml_to_env.items():
        if yaml_key not in platform_cfg or os.getenv(env_key):
            continue
        value = platform_cfg[yaml_key]
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        os.environ[env_key] = str(value)
    return None


def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes gateway setup`` → Kimi.

    Prompts for the bot token and optional allowlist / home channel. Writes
    via ``hermes_cli.config`` so the values land in ``~/.hermes/.env``.
    """
    print()
    print("Kimi platform setup")
    print("-------------------")
    print("Requirements:")
    print("  1. A Kimi bot token (km_b_prod_*) — generate via kimi.com")
    print("     'Link existing OpenClaw' flow.")
    print("  2. Python packages: websockets, aiohttp (installed via the")
    print("     hermes-agent[messaging] extra; the bundled kimiclaw plugin")
    print("     declares websockets there).")
    print()

    try:
        from hermes_cli.config import get_env_value, save_env_value
    except ImportError:
        print(
            "hermes_cli.config not available; set KIMI_* vars manually in "
            "~/.hermes/.env"
        )
        return

    def _prompt(var: str, prompt_text: str, *, secret: bool = False) -> None:
        existing = get_env_value(var) if callable(get_env_value) else None
        suffix = " [keep current]" if existing else ""
        try:
            if secret:
                import getpass
                value = getpass.getpass(f"{prompt_text}{suffix}: ")
            else:
                value = input(f"{prompt_text}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value:
            save_env_value(var, value)

    _prompt("KIMI_BOT_TOKEN", "Kimi bot token (km_b_prod_*)", secret=True)
    _prompt(
        "KIMI_ALLOWED_USERS",
        "Allowed user UUIDs (comma-separated; blank=skip)",
    )
    _prompt(
        "KIMI_HOME_CHANNEL",
        "Home channel for cron delivery (room:<uuid>; DM cron not supported)",
    )
    print("Done. Restart `hermes gateway` to apply.")


# Inlined here rather than as a class attribute because the prompt builder
# reads it via the registry, not from KimiAdapter directly.
_PLATFORM_HINT = (
    "You are chatting on Kimi (kimi.com / Moonshot AI). "
    "Users may be on the kimi.com web app or mobile. "
    "Direct messages arrive over a Zed ACP WebSocket — one sentinel session "
    "per user with sessionId ``im:kimi:main``. Group rooms arrive with "
    "chat_id ``room:<uuid>`` over a separate Connect RPC stream. User "
    "identities are opaque UUIDs (``km_u_*``); display names may be absent. "
    "Kimi supports standard markdown formatting. Be concise in group rooms; "
    "verbose responses are fine in DMs."
)

_INSTALL_HINT = (
    "Requires hermes-agent with the platform-registry API (commit 2e20f6ae2 "
    "'plugin platform parity', merged ~Apr 11 2026). "
    "Run: `uv tool upgrade 'hermes-agent[messaging]'` (or `pip install --upgrade "
    "'hermes-agent[messaging]'`) — the `[messaging]` extra provides "
    "`websockets` (DM ACP socket) and `aiohttp` (group-room Connect RPC stream)."
)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup.

    Uses the upstream ``ctx.register_platform()`` API introduced by Teknium's
    ``2e20f6ae2`` "plugin platform parity" commit. All canonical Kimi wiring
    (env-var loading, auth env-maps, prompt hint, cron delivery, display)
    that previously lived as in-tree patches on the fork branches now lives
    as parameters on this single call.
    """
    ctx.register_platform(
        name="kimiclaw",
        label="KimiClaw",
        adapter_factory=lambda cfg: KimiAdapter(cfg),
        check_fn=_check_for_registry,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["KIMI_BOT_TOKEN"],
        install_hint=_INSTALL_HINT,
        setup_fn=interactive_setup,
        # Env-driven auto-configuration: seeds PlatformConfig.extra so
        # env-only setups appear in `hermes gateway status` without
        # needing to instantiate the adapter.
        env_enablement_fn=_env_enablement,
        # YAML→env bridge for profile configs (e.g. Mimi) that prefer
        # a top-level `kimiclaw:` block in config.yaml over per-profile
        # .env. The block key matches the registered platform name
        # (``entry.name``); see ``_apply_yaml_config`` for details.
        apply_yaml_config_fn=_apply_yaml_config,
        # Cron home-channel delivery — `deliver=kimi` cron jobs route to
        # KIMI_HOME_CHANNEL when set.
        cron_deliver_env_var="KIMI_HOME_CHANNEL",
        # Out-of-process cron / send_message delivery for group rooms.
        standalone_sender_fn=_standalone_send,
        # Auth env vars for _is_user_authorized() integration.
        allowed_users_env="KIMI_ALLOWED_USERS",
        allow_all_env="KIMI_ALLOW_ALL_USERS",
        # Kimi has no hard message-length cap; we chunk for readability.
        max_message_length=MAX_MESSAGE_LENGTH,
        # Display.
        emoji="🌙",  # Moonshot AI
        # Kimi uses opaque UUIDs only — no phone numbers, no emails.
        pii_safe=True,
        allow_update_command=True,
        # LLM guidance.
        platform_hint=_PLATFORM_HINT,
    )
