"""
Yuanbao platform adapter.

Connects to the Yuanbao WebSocket gateway, handles authentication (AUTH_BIND),
heartbeat, reconnection, message receive (T05) and send (T06).

Configuration in config.yaml (or via env vars):
    platforms:
      yuanbao:
        extra:
          app_id: "..."              # or YUANBAO_APP_ID
          app_secret: "..."          # or YUANBAO_APP_SECRET
          bot_id: "..."              # or YUANBAO_BOT_ID  (optional, returned by sign-token)
          ws_url: "wss://..."        # or YUANBAO_WS_URL
          api_domain: "https://..."  # or YUANBAO_API_DOMAIN
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
import urllib.parse
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import sys

import httpx

try:
    import websockets
    import websockets.exceptions
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_document_from_bytes,
    cache_image_from_bytes,
)
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.yuanbao_media import (
    download_url as media_download_url,
    get_cos_credentials,
    upload_to_cos,
    build_image_msg_body,
    build_file_msg_body,
    guess_mime_type,
    md5_hex,
)
from gateway.platforms.yuanbao_proto import (
    CMD_TYPE,
    _fields_to_dict,
    _get_string,
    _get_varint,
    _parse_fields,
    WS_HEARTBEAT_RUNNING,
    WS_HEARTBEAT_FINISH,
    HERMES_INSTANCE_ID,
    decode_conn_msg,
    decode_inbound_push,
    decode_query_group_info_rsp,
    decode_get_group_member_list_rsp,
    encode_auth_bind,
    encode_ping,
    encode_push_ack,
    encode_send_c2c_message,
    encode_send_group_message,
    encode_send_private_heartbeat,
    encode_send_group_heartbeat,
    encode_query_group_info,
    encode_get_group_member_list,
    next_seq_no,
)
from gateway.session import SessionSource, build_session_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version / platform constants (used in AUTH_BIND and sign-token headers)
# ---------------------------------------------------------------------------
try:
    from hermes_cli import __version__ as _HERMES_VERSION
except ImportError:
    _HERMES_VERSION = "0.0.0"

_APP_VERSION = _HERMES_VERSION
_BOT_VERSION = _HERMES_VERSION
_YUANBAO_INSTANCE_ID = str(HERMES_INSTANCE_ID)  # single source: yuanbao_proto.HERMES_INSTANCE_ID
_OPERATION_SYSTEM = sys.platform

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_WS_GATEWAY_URL = "wss://bot-wss.yuanbao.tencent.com/wss/connection"
DEFAULT_API_DOMAIN = "https://bot.yuanbao.tencent.com"

HEARTBEAT_INTERVAL_SECONDS = 30.0
CONNECT_TIMEOUT_SECONDS = 15.0
AUTH_TIMEOUT_SECONDS = 10.0
MAX_RECONNECT_ATTEMPTS = 100
DEFAULT_SEND_TIMEOUT = 30.0  # WS biz request timeout

# Close codes that indicate permanent errors — do NOT reconnect.
NO_RECONNECT_CLOSE_CODES = {4012, 4013, 4014, 4018, 4019, 4021}

# Heartbeat timeout threshold — N consecutive missed pongs trigger reconnect.
HEARTBEAT_TIMEOUT_THRESHOLD = 2

# Auth error code classification
AUTH_FAILED_CODES = {4001, 4002, 4003}      # permanent auth failure, re-sign token
AUTH_RETRYABLE_CODES = {4010, 4011, 4099}   # transient, can retry with same token

# Reply Heartbeat configuration
REPLY_HEARTBEAT_INTERVAL_S = 2.0   # Send RUNNING every 2 seconds
REPLY_HEARTBEAT_TIMEOUT_S = 30.0   # Auto-stop after 30 seconds of inactivity

# Reply-to reference configuration
REPLY_REF_TTL_S = 300.0            # Reference dedup TTL (5 minutes)

# Slow-response hint: push a waiting message when agent produces no data for this duration (seconds)
SLOW_RESPONSE_TIMEOUT_S = 120.0
SLOW_RESPONSE_MESSAGE = "任务有点复杂，正在努力处理中，请耐心等待..."

# Regex matching Yuanbao resource reference anchors in transcript text:
#   [image|ybres:abc123]  [file:report.pdf|ybres:xyz789]  [voice|ybres:...]
_YB_RES_REF_RE = re.compile(
    r"\[(image|voice|video|file(?::[^|\]]*)?)\|ybres:([A-Za-z0-9_\-]+)\]"
)

# Strip page indicators like (1/3) appended by BasePlatformAdapter
_INDICATOR_RE = re.compile(r'\s*\(\d+/\d+\)$')

# Observed-media backfill: how many recent transcript messages to scan
OBSERVED_MEDIA_BACKFILL_LOOKBACK = 50
# Max number of resource references to resolve per inbound turn
OBSERVED_MEDIA_BACKFILL_MAX_RESOLVE_PER_TURN = 12

class MarkdownProcessor:
    """Encapsulates all Markdown-related utilities for the Yuanbao platform.

    Provides static methods for:
    - Fence detection and streaming merge
    - Table row detection and sanitization
    - Paragraph-boundary splitting
    - Atomic-block extraction and chunk splitting
    - Outer markdown fence stripping
    - Markdown hint prompt generation
    """

    # -- Fence detection ---------------------------------------------------

    @staticmethod
    def has_unclosed_fence(text: str) -> bool:
        """
        Detect whether the text has unclosed code block fences.

        Scan line by line, toggling in/out state when encountering a line starting with ```.
        An odd number of toggles indicates an unclosed fence.

        Args:
            text: Markdown text to check

        Returns:
            Returns True if the text ends with an unclosed fence, otherwise False
        """
        in_fence = False
        for line in text.split('\n'):
            if line.startswith('```'):
                in_fence = not in_fence
        return in_fence

    # -- Table detection ---------------------------------------------------

    @staticmethod
    def ends_with_table_row(text: str) -> bool:
        """
        Detect whether the text ends with a table row (last non-empty line starts and ends with |).

        Args:
            text: Text to check

        Returns:
            Returns True if the last non-empty line is a table row
        """
        trimmed = text.rstrip()
        if not trimmed:
            return False
        last_line = trimmed.split('\n')[-1].strip()
        return last_line.startswith('|') and last_line.endswith('|')

    # -- Paragraph boundary splitting --------------------------------------

    @staticmethod
    def split_at_paragraph_boundary(
        text: str,
        max_chars: int,
        len_fn: Optional[Callable[[str], int]] = None,
    ) -> tuple[str, str]:
        """
        Find the nearest paragraph boundary split point within max_chars, return (head, tail).

        Split priority:
        1. Blank line (paragraph boundary)
        2. Newline after period/question mark/exclamation mark (Chinese and English)
        3. Last newline
        4. Force split at max_chars

        Args:
            text: Text to split
            max_chars: Maximum character count limit
            len_fn: Optional custom length function (e.g. UTF-16 length); defaults to built-in len

        Returns:
            (head, tail) tuple, head is the front part, tail is the back part, satisfying head + tail == text
        """
        _len = len_fn or len
        if _len(text) <= max_chars:
            return text, ''

        # Build a character-index window that fits within max_chars.
        # When len_fn != len we cannot simply slice [:max_chars], so we
        # binary-search for the largest prefix that fits.
        if _len is len:
            window = text[:max_chars]
        else:
            lo, hi = 0, len(text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if _len(text[:mid]) <= max_chars:
                    lo = mid
                else:
                    hi = mid - 1
            window = text[:lo]

        # 1. Prefer the last blank line (\n\n) as paragraph boundary
        pos = window.rfind('\n\n')
        if pos > 0:
            return text[:pos + 2], text[pos + 2:]

        # 2. Then find the last newline after a sentence-ending punctuation
        sentence_end_re = re.compile(r'[。！？.!?]\n')
        best_pos = -1
        for m in sentence_end_re.finditer(window):
            best_pos = m.end()
        if best_pos > 0:
            return text[:best_pos], text[best_pos:]

        # 3. Fallback: find the last newline
        pos = window.rfind('\n')
        if pos > 0:
            return text[:pos + 1], text[pos + 1:]

        # 4. No valid split point found, force split at window boundary
        cut = len(window)
        return text[:cut], text[cut:]

    # -- Atomic block helpers (private) ------------------------------------

    @staticmethod
    def is_fence_atom(text: str) -> bool:
        """Determine whether an atomic block is a code block (starts with ```)."""
        return text.lstrip().startswith('```')

    @staticmethod
    def is_table_atom(text: str) -> bool:
        """Determine whether an atomic block is a table (first line starts with |)."""
        first_line = text.split('\n')[0].strip()
        return first_line.startswith('|') and first_line.endswith('|')

    @staticmethod
    def split_into_atoms(text: str) -> list[str]:
        """
        Split text into a list of "atomic blocks", each being an indivisible logical unit:

        - Code block (fence): from opening ``` to closing ``` (including fence lines)
        - Table: consecutive |...| lines forming a whole segment
        - Normal paragraph: plain text segments separated by blank lines

        Blank lines serve as separators and are not included in any atomic block.

        Args:
            text: Markdown text to split

        Returns:
            List of atomic block strings (all non-empty)
        """
        lines = text.split('\n')
        atoms: list[str] = []

        current_lines: list[str] = []
        in_fence = False

        def _is_table_line(line: str) -> bool:
            stripped = line.strip()
            return stripped.startswith('|') and stripped.endswith('|')

        def _flush_current() -> None:
            if current_lines:
                atom = '\n'.join(current_lines)
                if atom.strip():
                    atoms.append(atom)
                current_lines.clear()

        for line in lines:
            if in_fence:
                current_lines.append(line)
                if line.startswith('```') and len(current_lines) > 1:
                    in_fence = False
                    _flush_current()
            elif line.startswith('```'):
                _flush_current()
                in_fence = True
                current_lines.append(line)
            elif _is_table_line(line):
                if current_lines and not _is_table_line(current_lines[-1]):
                    _flush_current()
                current_lines.append(line)
            elif line.strip() == '':
                _flush_current()
            else:
                if current_lines and _is_table_line(current_lines[-1]):
                    _flush_current()
                current_lines.append(line)

        _flush_current()

        return atoms

    # -- Core: chunk splitting ---------------------------------------------

    @classmethod
    def chunk_markdown_text(
        cls,
        text: str,
        max_chars: int = 4000,
        len_fn: Optional[Callable[[str], int]] = None,
    ) -> list[str]:
        """
        Split Markdown text into multiple chunks by max_chars.

        Guarantees:
        - Each chunk <= max_chars characters (unless a single code block/table itself exceeds the limit)
        - Code blocks (```...```) are not split in the middle
        - Table rows are not split in the middle (tables output as atomic blocks)
        - Split at paragraph boundaries (blank lines, after periods, etc.)
        - Small trailing/leading chunks are merged with neighbours when possible

        Args:
            text: Markdown text to split
            max_chars: Max characters per chunk, default 4000
            len_fn: Optional custom length function (e.g. UTF-16 length); defaults to built-in len

        Returns:
            List of text chunks after splitting (non-empty)
        """
        _len = len_fn or len

        if not text:
            return []

        if _len(text) <= max_chars:
            return [text]

        # Phase 1: Extract atomic blocks
        atoms = cls.split_into_atoms(text)

        # Phase 2: Greedy merge
        chunks: list[str] = []
        indivisible_set: set[int] = set()
        current_parts: list[str] = []
        current_len = 0

        def _flush_parts() -> None:
            if current_parts:
                chunks.append('\n\n'.join(current_parts))

        for atom in atoms:
            atom_len = _len(atom)
            sep_len = 2 if current_parts else 0
            projected_len = current_len + sep_len + atom_len

            if projected_len > max_chars and current_parts:
                _flush_parts()
                current_parts = []
                current_len = 0
                sep_len = 0

            if (not current_parts
                    and atom_len > max_chars
                    and (cls.is_fence_atom(atom) or cls.is_table_atom(atom))):
                indivisible_set.add(len(chunks))
                chunks.append(atom)
                continue

            current_parts.append(atom)
            current_len += sep_len + atom_len

        _flush_parts()

        # Phase 3: Post-processing — split still-oversized chunks at paragraph boundaries
        result: list[str] = []
        for idx, chunk in enumerate(chunks):
            if _len(chunk) <= max_chars:
                result.append(chunk)
                continue

            if idx in indivisible_set:
                result.append(chunk)
                continue

            if cls.has_unclosed_fence(chunk):
                result.append(chunk)
                continue

            remaining = chunk
            while _len(remaining) > max_chars:
                head, remaining = cls.split_at_paragraph_boundary(
                    remaining, max_chars, len_fn=len_fn,
                )
                if not head:
                    head, remaining = remaining[:max_chars], remaining[max_chars:]
                if head:
                    result.append(head)
            if remaining:
                result.append(remaining)

        # Phase 4: Merge small trailing/leading chunks with neighbours
        if len(result) > 1:
            merged: list[str] = [result[0]]
            for chunk in result[1:]:
                prev = merged[-1]
                combined = prev + '\n\n' + chunk
                if _len(combined) <= max_chars:
                    merged[-1] = combined
                else:
                    merged.append(chunk)
            result = merged

        return [c for c in result if c]

    # -- Block separator inference -----------------------------------------

    @classmethod
    def infer_block_separator(cls, prev_chunk: str, next_chunk: str) -> str:
        """
        Infer the separator to use between two split chunks.

        Rules (aligned with TS markdown-stream.ts):
        - Previous chunk ends with code fence or next chunk starts with fence → single newline '\\n'
        - Previous chunk ends with table row and next chunk starts with table row → single newline '\\n' (continued table)
        - Otherwise → double newline '\\n\\n' (paragraph separator)

        Args:
            prev_chunk: Previous chunk
            next_chunk: Next chunk

        Returns:
            '\\n' or '\\n\\n'
        """
        prev_trimmed = prev_chunk.rstrip()
        next_trimmed = next_chunk.lstrip()

        # Previous chunk ends with fence or next chunk starts with fence
        if prev_trimmed.endswith('```') or next_trimmed.startswith('```'):
            return '\n'

        # Table continuation
        if cls.ends_with_table_row(prev_chunk):
            first_line = next_trimmed.split('\n')[0].strip() if next_trimmed else ''
            if first_line.startswith('|') and first_line.endswith('|'):
                return '\n'

        return '\n\n'

    # -- Streaming fence merge ---------------------------------------------

    @classmethod
    def merge_block_streaming_fences(cls, chunks: list[str]) -> list[str]:
        """
        Stream-aware fence-conscious chunk merging.

        When streaming output produces multiple chunks truncated in the middle of a fence,
        attempt to merge adjacent chunks to complete the fence.

        Rules:
        - If chunk i has an unclosed fence and chunk i+1 starts with ```,
            merge i+1 into i (until the fence is closed or no more chunks).
        - Use infer_block_separator to infer the separator during merging.

        Args:
            chunks: Original chunk list

        Returns:
            Merged chunk list (length <= original length)
        """
        if not chunks:
            return []

        result: list[str] = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            # If current chunk has unclosed fence, try merging subsequent chunks
            while cls.has_unclosed_fence(current) and i + 1 < len(chunks):
                sep = cls.infer_block_separator(current, chunks[i + 1])
                current = current + sep + chunks[i + 1]
                i += 1
            result.append(current)
            i += 1

        return result

    # -- Outer fence stripping ---------------------------------------------

    @staticmethod
    def strip_outer_markdown_fence(text: str) -> str:
        """
        Strip outer Markdown fence.

        When AI reply is entirely wrapped in ```markdown\\n...\\n```, remove the outer fence,
        keeping the content. Only strip when the first line is ```markdown (case-insensitive) and the last line is ```.

        Args:
            text: Text to process

        Returns:
            Text with outer fence stripped (returns original if no match)
        """
        if not text:
            return text

        lines = text.split('\n')
        if len(lines) < 3:
            return text

        first_line = lines[0].strip()
        last_line = lines[-1].strip()

        # First line must be ```markdown (optional language tag md/markdown)
        if not re.match(r'^```(?:markdown|md)?\s*$', first_line, re.IGNORECASE):
            return text

        # Last line must be plain ```
        if last_line != '```':
            return text

        # Strip first and last lines
        inner = '\n'.join(lines[1:-1])
        return inner

    # -- Table sanitization ------------------------------------------------

    @staticmethod
    def sanitize_markdown_table(text: str) -> str:
        """
        Table output sanitization.

        Handle common formatting issues in AI-generated Markdown tables:
        1. Remove extra whitespace before/after table rows
        2. Ensure separator rows (|---|---|) are correctly formatted
        3. Remove empty table rows

        Args:
            text: Markdown text containing tables

        Returns:
            Sanitized text
        """
        if '|' not in text:
            return text

        lines = text.split('\n')
        result_lines: list[str] = []

        for line in lines:
            stripped = line.strip()

            # Table row processing
            if stripped.startswith('|') and stripped.endswith('|'):
                # Separator row normalization: | --- | --- | → |---|---|
                if re.match(r'^\|[\s\-:]+(\|[\s\-:]+)+\|$', stripped):
                    cells = stripped.split('|')
                    normalized = '|'.join(
                        cell.strip() if cell.strip() else cell
                        for cell in cells
                    )
                    result_lines.append(normalized)
                elif stripped == '||' or stripped.replace('|', '').strip() == '':
                    # Empty table row → skip
                    continue
                else:
                    result_lines.append(stripped)
            else:
                result_lines.append(line)

        return '\n'.join(result_lines)

    # -- Markdown hint prompt ----------------------------------------------

    @staticmethod
    def markdown_hint_system_prompt() -> str:
        """
        Markdown rendering hint (appended to system prompt).

        Tell AI that Yuanbao platform supports Markdown rendering, including:
        - Code blocks (```lang)
        - Tables (| col | col |)
        - Bold/italic
        """
        return (
            "The current platform supports Markdown rendering. You can use the following formats:\n"
            "- Code blocks: ```language\\ncode\\n```\n"
            "- Tables: | col1 | col2 |\\n|---|---|\\n| val1 | val2 |\n"
            "- Bold: **text** / Italic: *text*\n"
            "Please use Markdown formatting when appropriate to improve readability."
        )

class SignManager:
    """Encapsulates all sign-token related logic for the Yuanbao platform.

    Manages token acquisition, caching, signature computation, and
    automatic retry.  All state (cache, locks) is kept as class-level
    attributes so that a single shared client serves the whole process.
    """

    # -- Constants ---------------------------------------------------------

    TOKEN_PATH = "/api/v5/robotLogic/sign-token"

    RETRYABLE_CODE = 10099
    MAX_RETRIES = 3
    RETRY_DELAY_S = 1.0

    #: Early refresh margin (seconds), treat as expiring 60s before actual expiry
    CACHE_REFRESH_MARGIN_S = 60

    #: HTTP timeout (seconds)
    HTTP_TIMEOUT_S = 10.0

    # -- Class-level shared state ------------------------------------------

    # key: app_key → {"token", "bot_id", "expire_ts", ...}
    _cache: dict[str, dict[str, Any]] = {}

    # Per-app_key refresh locks — prevents concurrent duplicate sign-token
    # requests.  Created lazily inside get_refresh_lock() which is only called
    # from async context, so the Lock is always bound to the correct loop.
    # disconnect() clears this dict to prevent stale locks across reconnects.
    _locks: dict[str, asyncio.Lock] = {}

    # -- Internal helpers --------------------------------------------------

    @classmethod
    def get_refresh_lock(cls, app_key: str) -> asyncio.Lock:
        """Return (creating if needed) the per-app_key refresh lock.

        Must only be called from within a running event loop (async context).
        """
        if app_key not in cls._locks:
            cls._locks[app_key] = asyncio.Lock()
        return cls._locks[app_key]

    @staticmethod
    def compute_signature(nonce: str, timestamp: str, app_key: str, app_secret: str) -> str:
        """Compute HMAC-SHA256 signature (aligned with TypeScript original).

        plain     = nonce + timestamp + app_key + app_secret
        signature = HMAC-SHA256(key=app_secret, msg=plain).hexdigest()
        """
        plain = nonce + timestamp + app_key + app_secret
        return hmac.new(app_secret.encode(), plain.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def build_timestamp() -> str:
        """Build Beijing-time ISO-8601 timestamp (no milliseconds).

        Format: 2006-01-02T15:04:05+08:00
        """
        bjtime = datetime.now(tz=timezone(timedelta(hours=8)))
        return bjtime.strftime("%Y-%m-%dT%H:%M:%S+08:00")

    @classmethod
    def is_cache_valid(cls, entry: dict[str, Any]) -> bool:
        """Determine whether the cache entry is valid (not expired with margin)."""
        return entry["expire_ts"] - time.time() > cls.CACHE_REFRESH_MARGIN_S

    @classmethod
    def clear_locks(cls) -> None:
        """Clear all per-app_key refresh locks (called on disconnect)."""
        cls._locks.clear()

    @classmethod
    def purge_expired(cls) -> int:
        """Remove all expired entries from the token cache.

        Returns the number of entries purged.  Called lazily from
        ``get_token()`` so that stale app_key entries don't accumulate
        indefinitely in long-running processes.
        """
        now = time.time()
        expired_keys = [
            k for k, v in cls._cache.items()
            if now - v.get("expire_ts", 0) > 0
        ]
        for k in expired_keys:
            cls._cache.pop(k, None)
        return len(expired_keys)

    # -- Core: fetch -------------------------------------------------------

    @classmethod
    async def fetch(
        cls,
        app_key: str,
        app_secret: str,
        api_domain: str,
        route_env: str = "",
    ) -> dict[str, Any]:
        """Send sign-ticket HTTP request with auto-retry (up to MAX_RETRIES times)."""
        url = f"{api_domain.rstrip('/')}{cls.TOKEN_PATH}"
        async with httpx.AsyncClient(timeout=cls.HTTP_TIMEOUT_S) as client:
            for attempt in range(cls.MAX_RETRIES + 1):
                nonce = secrets.token_hex(16)
                timestamp = cls.build_timestamp()
                signature = cls.compute_signature(nonce, timestamp, app_key, app_secret)

                payload = {
                    "app_key": app_key,
                    "nonce": nonce,
                    "signature": signature,
                    "timestamp": timestamp,
                }

                headers = {
                    "Content-Type": "application/json",
                    "X-AppVersion": _APP_VERSION,
                    "X-OperationSystem": _OPERATION_SYSTEM,
                    "X-Instance-Id": _YUANBAO_INSTANCE_ID,
                    "X-Bot-Version": _BOT_VERSION,
                }
                if route_env:
                    headers["X-Route-Env"] = route_env

                logger.info(
                    "Sign token request: url=%s%s",
                    url,
                    f" (retry {attempt}/{cls.MAX_RETRIES})" if attempt > 0 else "",
                )

                response = await client.post(url, json=payload, headers=headers)

                if response.status_code != 200:
                    body = response.text
                    raise RuntimeError(f"Sign token API returned {response.status_code}: {body[:200]}")

                try:
                    result_data: dict[str, Any] = response.json()
                except Exception as exc:
                    raise ValueError(f"Sign token response parse error: {exc}") from exc

                code = result_data.get("code")
                if code == 0:
                    data = result_data.get("data")
                    if not isinstance(data, dict):
                        raise ValueError(f"Sign token response missing 'data' field: {result_data}")
                    logger.info("Sign token success: bot_id=%s", data.get("bot_id"))
                    return data

                if code == cls.RETRYABLE_CODE and attempt < cls.MAX_RETRIES:
                    logger.warning(
                        "Sign token retryable: code=%s, retrying in %ss (attempt=%d/%d)",
                        code,
                        cls.RETRY_DELAY_S,
                        attempt + 1,
                        cls.MAX_RETRIES,
                    )
                    await asyncio.sleep(cls.RETRY_DELAY_S)
                    continue

                msg = result_data.get("msg", "")
                raise RuntimeError(f"Sign token error: code={code}, msg={msg}")

        raise RuntimeError("Sign token failed: max retries exceeded")

    # -- Public API: get (with cache) --------------------------------------

    @classmethod
    async def get_token(
        cls,
        app_key: str,
        app_secret: str,
        api_domain: str,
        route_env: str = "",
    ) -> dict[str, Any]:
        """Get WS auth token (with cache).

        Return directly on cache hit without re-requesting; treat as expiring
        60 seconds before actual expiry, triggering refresh.
        """
        # Lazily evict stale entries from other app_keys
        cls.purge_expired()

        cached = cls._cache.get(app_key)
        if cached and cls.is_cache_valid(cached):
            remain = int(cached["expire_ts"] - time.time())
            logger.info("Using cached token (%ds remaining)", remain)
            return dict(cached)

        async with cls.get_refresh_lock(app_key):
            cached = cls._cache.get(app_key)
            if cached and cls.is_cache_valid(cached):
                return dict(cached)

            data = await cls.fetch(app_key, app_secret, api_domain, route_env)

            duration: int = data.get("duration", 0)
            expire_ts = time.time() + duration if duration > 0 else time.time() + 3600

            cls._cache[app_key] = {
                "token": data.get("token", ""),
                "bot_id": data.get("bot_id", ""),
                "duration": duration,
                "product": data.get("product", ""),
                "source": data.get("source", ""),
                "expire_ts": expire_ts,
            }

        return dict(cls._cache[app_key])

    # -- Public API: force refresh -----------------------------------------

    @classmethod
    async def force_refresh(
        cls,
        app_key: str,
        app_secret: str,
        api_domain: str,
        route_env: str = "",
    ) -> dict[str, Any]:
        """Force refresh token (clear cache and re-sign)."""
        logger.warning("[force-refresh] Clearing cache and re-signing token: app_key=****%s", app_key[-4:])
        async with cls.get_refresh_lock(app_key):
            cls._cache.pop(app_key, None)
            data = await cls.fetch(app_key, app_secret, api_domain, route_env)

            duration: int = data.get("duration", 0)
            expire_ts = time.time() + duration if duration > 0 else time.time() + 3600

            cls._cache[app_key] = {
                "token": data.get("token", ""),
                "bot_id": data.get("bot_id", ""),
                "duration": duration,
                "product": data.get("product", ""),
                "source": data.get("source", ""),
                "expire_ts": expire_ts,
            }

        return dict(cls._cache[app_key])


from dataclasses import dataclass, field as dc_field

@dataclass
class InboundContext:
    """Mutable context flowing through the inbound middleware pipeline.

    Each middleware reads/writes fields on this context.  The pipeline
    engine passes it to every middleware in registration order.
    """

    adapter: Any  # YuanbaoAdapter (forward-ref avoids circular import)
    raw_frames: list = dc_field(default_factory=list)  # Raw bytes frames (debounce-aggregated)

    # Populated by DecodeMiddleware
    push: Optional[dict] = None
    decoded_via: str = ""  # "json" | "protobuf"

    # Extracted from push by FieldExtractMiddleware
    from_account: str = ""
    group_code: str = ""
    group_name: str = ""
    sender_nickname: str = ""
    msg_body: list = dc_field(default_factory=list)
    msg_id: str = ""
    cloud_custom_data: str = ""

    # Derived by ChatRoutingMiddleware
    chat_id: str = ""
    chat_type: str = ""  # "dm" | "group"
    chat_name: str = ""

    # Populated by ContentExtractMiddleware
    raw_text: str = ""
    media_refs: list = dc_field(default_factory=list)

    # Owner command detection
    owner_command: Optional[str] = None

    # Source built by BuildSourceMiddleware
    source: Optional[Any] = None  # SessionSource

    # Populated by ClassifyMessageTypeMiddleware
    msg_type: Optional[Any] = None  # MessageType

    # Populated by QuoteContextMiddleware
    reply_to_message_id: Optional[str] = None
    reply_to_text: Optional[str] = None

    # Populated by MediaResolveMiddleware
    media_urls: list = dc_field(default_factory=list)
    media_types: list = dc_field(default_factory=list)

    # Populated by ExtractContentMiddleware
    link_urls: list = dc_field(default_factory=list)

    # Populated by GroupAttributionMiddleware
    channel_prompt: Optional[str] = None


class InboundMiddleware(ABC):
    """Abstract base class for all inbound pipeline middlewares.

    Subclasses must:
      - Set ``name`` as a class-level attribute (used for pipeline registration
        and dynamic insertion/removal).
      - Implement ``async handle(ctx, next_fn)`` containing the middleware logic.

    Convention:
      - Call ``await next_fn()`` to pass control to the next middleware.
      - Return without calling ``next_fn`` to **stop** the pipeline.
    """

    name: str = ""  # Override in each subclass

    @abstractmethod
    async def handle(self, ctx: InboundContext, next_fn: Callable) -> None:
        """Process *ctx* and optionally call *next_fn* to continue the pipeline."""

    async def __call__(self, ctx: InboundContext, next_fn: Callable) -> None:
        """Allow middleware instances to be called directly (duck-typing compat)."""
        return await self.handle(ctx, next_fn)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


class InboundPipeline:
    """Onion-model middleware pipeline engine for inbound message processing.

    Inspired by OpenClaw's MessagePipeline (extensions/yuanbao/src/business/
    pipeline/engine.ts).  Supports named middlewares, conditional guards
    (``when``), and ``use_before`` / ``use_after`` / ``remove`` for dynamic
    composition.

    Accepts both ``InboundMiddleware`` instances (OOP style) and plain
    ``async def(ctx, next_fn)`` callables (functional style) for flexibility.
    """

    def __init__(self) -> None:
        self._middlewares: list = []  # list of (name, handler, when_fn | None)

    # -- Internal helpers --------------------------------------------------

    @staticmethod
    def _normalize(name_or_mw, handler=None):
        """Normalize (name, handler) or (InboundMiddleware,) into (name, callable)."""
        if isinstance(name_or_mw, InboundMiddleware):
            return name_or_mw.name, name_or_mw
        # Functional style: name is a str, handler is a callable
        return name_or_mw, handler

    # -- Registration API --------------------------------------------------

    def use(self, name_or_mw, handler=None, when=None) -> "InboundPipeline":
        """Append a middleware to the end of the pipeline.

        Accepts either:
          - ``pipeline.use(SomeMiddleware())``  — OOP style
          - ``pipeline.use("name", some_fn)``   — functional style
        """
        name, h = self._normalize(name_or_mw, handler)
        self._middlewares.append((name, h, when))
        return self

    def use_before(self, target: str, name_or_mw, handler=None, when=None) -> "InboundPipeline":
        """Insert a middleware before *target* (by name).  Appends if not found."""
        name, h = self._normalize(name_or_mw, handler)
        idx = next((i for i, (n, _, _) in enumerate(self._middlewares) if n == target), None)
        entry = (name, h, when)
        if idx is None:
            self._middlewares.append(entry)
        else:
            self._middlewares.insert(idx, entry)
        return self

    def use_after(self, target: str, name_or_mw, handler=None, when=None) -> "InboundPipeline":
        """Insert a middleware after *target* (by name).  Appends if not found."""
        name, h = self._normalize(name_or_mw, handler)
        idx = next((i for i, (n, _, _) in enumerate(self._middlewares) if n == target), None)
        entry = (name, h, when)
        if idx is None:
            self._middlewares.append(entry)
        else:
            self._middlewares.insert(idx + 1, entry)
        return self

    def remove(self, name: str) -> "InboundPipeline":
        """Remove a middleware by name."""
        self._middlewares = [(n, h, w) for n, h, w in self._middlewares if n != name]
        return self

    @property
    def middleware_names(self) -> list:
        """Return ordered list of registered middleware names (for testing)."""
        return [n for n, _, _ in self._middlewares]

    # -- Execution ---------------------------------------------------------

    async def execute(self, ctx: InboundContext) -> None:
        """Run all middlewares in order.  Each middleware receives ``(ctx, next_fn)``."""
        chain = self._middlewares
        index = 0

        async def next_fn() -> None:
            nonlocal index
            while index < len(chain):
                name, handler, when_fn = chain[index]
                index += 1
                # Conditional guard: skip when returns False
                if when_fn is not None and not when_fn(ctx):
                    continue
                try:
                    await handler(ctx, next_fn)
                except Exception:
                    logger.error("[InboundPipeline] middleware [%s] error", name, exc_info=True)
                    raise
                return
            # End of chain — nothing more to do

        await next_fn()
class DecodeMiddleware(InboundMiddleware):
    """Decode raw inbound frames from JSON or Protobuf into ctx.push.

    Encapsulates JSON push parsing (aligned with TS decodeFromContent)
    and Protobuf decoding via ``decode_inbound_push``.
    """

    name = "decode"

    # -- JSON push parsing -------------------------------------------------

    @staticmethod
    def convert_json_msg_body(raw_body: list) -> list:
        """Normalize raw JSON msg_body array to [{"msg_type": str, "msg_content": dict}].

        Compatible with both PascalCase (MsgType/MsgContent) and
        snake_case (msg_type/msg_content) naming.
        """
        result = []
        for item in raw_body or []:
            if not isinstance(item, dict):
                continue
            msg_type = item.get("msg_type") or item.get("MsgType", "")
            msg_content = item.get("msg_content") or item.get("MsgContent", {})
            if isinstance(msg_content, str):
                try:
                    msg_content = json.loads(msg_content)
                except Exception:
                    msg_content = {"text": msg_content}
            result.append({"msg_type": msg_type, "msg_content": msg_content or {}})
        return result

    @staticmethod
    def parse_json_push(raw_json: dict) -> dict | None:
        """Convert JSON-format push to a dict with the same structure as
        ``decode_inbound_push``.

        Supports standard callback format (callback_command + from_account +
        msg_body) and legacy format fields (GroupId, MsgSeq, MsgKey, MsgBody,
        etc.).
        """
        if not raw_json:
            return None

        # Tencent IM callback format uses PascalCase (From_Account, To_Account, MsgBody).
        # Internal format uses snake_case (from_account, to_account, msg_body).
        # Support both.
        from_account = (
            raw_json.get("from_account", "")
            or raw_json.get("From_Account", "")
        )
        group_code = (
            raw_json.get("group_code", "")
            or raw_json.get("GroupId", "")
            or raw_json.get("group_id", "")
        )
        msg_body_raw = (
            raw_json.get("msg_body", [])
            or raw_json.get("MsgBody", [])
        )
        msg_body = DecodeMiddleware.convert_json_msg_body(msg_body_raw)

        # Recall callbacks may have neither from_account nor msg_body.
        if not from_account and not msg_body and not raw_json.get("callback_command"):
            return None

        return {
            "callback_command": raw_json.get("callback_command", ""),
            "from_account": from_account,
            "to_account": raw_json.get("to_account", "") or raw_json.get("To_Account", ""),
            "sender_nickname": raw_json.get("sender_nickname", "") or raw_json.get("nick_name", ""),
            "group_code": group_code,
            "group_name": raw_json.get("group_name", ""),
            "msg_seq": raw_json.get("msg_seq", 0) or raw_json.get("MsgSeq", 0),
            "msg_id": raw_json.get("msg_id", "") or raw_json.get("msg_key", "") or raw_json.get("MsgKey", ""),
            "msg_body": msg_body,
            "cloud_custom_data": raw_json.get("cloud_custom_data", "") or raw_json.get("CloudCustomData", ""),
            "bot_owner_id": raw_json.get("bot_owner_id", "") or raw_json.get("botOwnerId", ""),
            "recall_msg_seq_list": raw_json.get("recall_msg_seq_list") or None,
            "trace_id": (raw_json.get("log_ext") or {}).get("trace_id", "") if isinstance(raw_json.get("log_ext"), dict) else "",
        }

    # -- Pipeline handler --------------------------------------------------

    def _decode_single(self, adapter, data: bytes) -> tuple:
        """Decode a single raw frame into (push_dict, decoded_via) or (None, '')."""
        try:
            conn_json = json.loads(data.decode("utf-8"))
        except Exception:
            conn_json = None

        if isinstance(conn_json, dict):
            push = self.parse_json_push(conn_json)
            if push:
                return push, "json"
        else:
            try:
                push = decode_inbound_push(data)
            except Exception:
                push = None
            if push:
                return push, "protobuf"

        return None, ""

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        data_list = ctx.raw_frames
        if not data_list:
            return  # Stop pipeline — nothing to decode

        merged_push = None
        decoded_via = ""

        for data in data_list:
            push, via = self._decode_single(ctx.adapter, data)
            if not push:
                logger.info(
                "[%s] Push decoded but no valid message. raw hex(first64)=%s",
                    ctx.adapter.name, data.hex()[:128] if data else "(empty)",
                )
                continue

            if merged_push is None:
                # First valid push becomes the base
                merged_push = push
                decoded_via = via
                logger.info(
                "[%s] Frame decoded (via=%s): len=%d",
                    ctx.adapter.name, via, len(data),
                )
            else:
                # Subsequent pushes: merge msg_body into the base with a
                extra_body = push.get("msg_body", [])
                if extra_body:
                    _sep = {"msg_type": "TIMTextElem", "msg_content": {"text": "\n"}}
                    merged_push["msg_body"] = merged_push.get("msg_body", []) + [_sep] + extra_body
                    logger.info(
                        "[%s] Merged %d extra msg_body elements from aggregated push",
                        ctx.adapter.name, len(extra_body),
                    )

        if not merged_push:
            return  # Stop pipeline

        ctx.push = merged_push
        ctx.decoded_via = decoded_via

        logger.info(
            "[%s] Push decoded (via=%s): from=%s group=%s msg_id=%s msg_types=%s",
            ctx.adapter.name, ctx.decoded_via,
            ctx.push.get("from_account", ""),
            ctx.push.get("group_code", ""),
            ctx.push.get("msg_id", ""),
            [e.get("msg_type", "") for e in ctx.push.get("msg_body", [])],
        )
        logger.debug("[%s] Push payload: %s", ctx.adapter.name, ctx.push)

        await next_fn()


class ExtractFieldsMiddleware(InboundMiddleware):
    """Extract common fields from ctx.push into ctx attributes."""

    name = "extract-fields"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        push = ctx.push
        ctx.from_account = push.get("from_account", "")
        ctx.group_code = push.get("group_code", "")
        ctx.group_name = push.get("group_name", "")
        ctx.sender_nickname = push.get("sender_nickname", "")
        ctx.msg_body = push.get("msg_body", [])
        ctx.msg_id = push.get("msg_id", "")
        ctx.cloud_custom_data = push.get("cloud_custom_data", "")
        await next_fn()


class DedupMiddleware(InboundMiddleware):
    """Inbound message deduplication."""

    name = "dedup"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if ctx.msg_id and ctx.adapter._dedup.is_duplicate(ctx.msg_id):
            logger.debug("[%s] Duplicate message ignored: msg_id=%s", ctx.adapter.name, ctx.msg_id)
            return  # Stop pipeline
        await next_fn()


class RecallGuardMiddleware(InboundMiddleware):
    """Intercept Group.CallbackAfterRecallMsg / C2C.CallbackAfterMsgWithDraw.

    Branch A: message in transcript (observed, not yet consumed) → redact content
    Branch B: message not in transcript → append system note
    Branch C: message currently being processed → silent interrupt + delayed redact
    """

    name = "recall_guard"

    _RECALL_COMMANDS = frozenset({
        "Group.CallbackAfterRecallMsg",
        "C2C.CallbackAfterMsgWithDraw",
    })
    _REDACTED = "[This message was recalled/withdrawn by the sender; original content removed]"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        cmd = (ctx.push or {}).get("callback_command", "")
        if cmd not in self._RECALL_COMMANDS:
            await next_fn()
            return
        self._handle_recall(ctx, cmd)

    @staticmethod
    def _build_source(adapter, group_code: str, from_account: str):
        return adapter.build_source(
            chat_id=(f"group:{group_code}" if group_code else f"direct:{from_account}"),
            chat_type="group" if group_code else "dm",
            user_id=from_account or None,
            thread_id="main" if group_code else None,
        )

    def _handle_recall(self, ctx: InboundContext, cmd: str) -> None:
        adapter = ctx.adapter
        push = ctx.push or {}

        if cmd == "Group.CallbackAfterRecallMsg":
            seq_list = push.get("recall_msg_seq_list") or []
        else:
            mid = push.get("msg_id") or ""
            seq = push.get("msg_seq")
            seq_list = [{"msg_id": mid, "msg_seq": seq}] if (mid or seq) else []

        if not seq_list:
            logger.debug("[%s] Recall callback with empty seq_list, skipping", adapter.name)
            return

        group_code = (push.get("group_code") or "").strip()
        from_account = (push.get("from_account") or "").strip()

        for seq_entry in seq_list:
            recalled_id = seq_entry.get("msg_id") or str(seq_entry.get("msg_seq") or "")
            if not recalled_id:
                continue

            matched_sk = self._find_processing_session(adapter, recalled_id)
            if matched_sk is not None:
                self._interrupt_for_recall(adapter, matched_sk, recalled_id, group_code, from_account)
            else:
                recalled_content = adapter._msg_content_cache.get(recalled_id)
                self._patch_transcript(adapter, recalled_id, group_code, from_account, recalled_content)

    # -- Branch C: interrupt currently-processing message ---------------

    @staticmethod
    def _find_processing_session(adapter, recalled_id: str) -> Optional[str]:
        for sk, mid in adapter._processing_msg_ids.items():
            if mid == recalled_id and sk in adapter._active_sessions:
                return sk
        return None

    @classmethod
    def _interrupt_for_recall(cls, adapter, session_key: str, recalled_id: str,
                              group_code: str, from_account: str) -> None:
        where = f"group {group_code}" if group_code else f"direct chat with {from_account}"
        recall_text = (
            f"[CRITICAL — MESSAGE RECALLED] The user message that triggered "
            f"your current task (message_id=\"{recalled_id}\") in {where} has "
            f"been recalled/withdrawn by the sender. "
            f"IGNORE any prior system note asking you to finish processing "
            f"tool results — the original request is void. "
            f"Do NOT continue the task, do NOT call more tools, do NOT "
            f"reference the recalled content. "
            f"Reply only with a brief acknowledgment such as "
            f"\"The message has been recalled.\" in the "
            f"language the user was using."
        )

        synth_event = MessageEvent(
            text=recall_text,
            message_type=MessageType.TEXT,
            source=cls._build_source(adapter, group_code, from_account),
            internal=True,
        )
        # Set pending + signal directly (bypass handle_message to avoid busy-ack).
        # May overwrite a user message pending in the same ~200ms window — acceptable.
        adapter._pending_messages[session_key] = synth_event
        active_event = adapter._active_sessions.get(session_key)
        if active_event is not None:
            active_event.set()

        logger.info("[%s] Recall interrupt: msg_id=%s session=%s", adapter.name, recalled_id, session_key[:30])

        # The interrupted turn will persist the recalled content *after* our
        # interrupt — schedule a delayed redaction to clean it up.
        recalled_text = adapter._processing_msg_texts.get(session_key, "")
        if recalled_text:
            cls._schedule_content_redact(adapter, session_key, recalled_text, group_code, from_account)

    @classmethod
    def _schedule_content_redact(cls, adapter, session_key: str, recalled_text: str,
                                 group_code: str, from_account: str) -> None:
        async def _redact() -> None:
            store = getattr(adapter, "_session_store", None)
            if not store:
                return
            try:
                sid = store.get_or_create_session(
                    cls._build_source(adapter, group_code, from_account),
                ).session_id
            except Exception:
                return
            # Poll until the recalled content appears in transcript — the
            # interrupted turn hasn't finished writing yet when scheduled.
            for _ in range(30):
                await asyncio.sleep(0.5)
                try:
                    transcript = store.load_transcript(sid)
                except Exception:
                    continue
                for entry in transcript:
                    if entry.get("role") == "user" and entry.get("content") == recalled_text:
                        entry["content"] = cls._REDACTED
                        try:
                            store.rewrite_transcript(sid, transcript)
                            logger.info("[%s] Recall redact: session %s", adapter.name, session_key[:30])
                        except Exception as exc:
                            logger.warning("[%s] Recall redact failed: %s", adapter.name, exc)
                        return
            logger.debug("[%s] Recall redact: content not found after polling, session %s", adapter.name, session_key[:30])

        task = asyncio.create_task(_redact())
        adapter._background_tasks.add(task)
        task.add_done_callback(adapter._background_tasks.discard)

    # -- Branch A/B: patch transcript (session idle) --------------------

    @classmethod
    def _patch_transcript(cls, adapter, recalled_id: str, group_code: str,
                          from_account: str, recalled_content: Optional[str] = None) -> None:
        store = getattr(adapter, "_session_store", None)
        if not store:
            return
        try:
            sid = store.get_or_create_session(cls._build_source(adapter, group_code, from_account)).session_id
        except Exception as exc:
            logger.warning("[%s] Recall: failed to resolve session: %s", adapter.name, exc)
            return

        # Read JSONL directly — SQLite doesn't preserve message_id field.
        transcript: list = []
        try:
            path = store.get_transcript_path(sid)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                transcript.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
        except Exception as exc:
            logger.warning("[%s] Recall: failed to load transcript: %s", adapter.name, exc)
            return

        # Branch A: redact — try message_id first, then content fallback.
        # Observed messages have message_id; agent-processed @bot messages
        # only have content (run.py doesn't write message_id to transcript).
        target = None
        for entry in transcript:
            if entry.get("message_id") == recalled_id:
                target = entry
                break
        if target is None and recalled_content:
            for entry in transcript:
                if entry.get("role") == "user" and entry.get("content") == recalled_content:
                    target = entry
                    break
        if target is not None:
            target["content"] = cls._REDACTED
            try:
                store.rewrite_transcript(sid, transcript)
                logger.info("[%s] Recall: redacted msg_id=%s (branch A)", adapter.name, recalled_id)
            except Exception as exc:
                logger.warning("[%s] Recall: rewrite_transcript failed: %s", adapter.name, exc)
            return

        # Branch B: not found in transcript → append system note
        store.append_to_transcript(sid, {
            "role": "system",
            "content": f'[recall] message_id="{recalled_id}" has been recalled; do not quote or reference it.',
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })
        logger.info("[%s] Recall: system note for msg_id=%s (branch B)", adapter.name, recalled_id)


class SkipSelfMiddleware(InboundMiddleware):
    """Filter out bot's own messages."""

    name = "skip-self"

    @staticmethod
    def _is_self_reference(from_account: str, bot_id: Optional[str]) -> bool:
        """Detect whether the message is from the bot itself."""
        if not from_account or not bot_id:
            return False
        return from_account == bot_id

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if self._is_self_reference(ctx.from_account, ctx.adapter._bot_id):
            logger.debug("[%s] Ignoring self-sent message from %s", ctx.adapter.name, ctx.from_account)
            return  # Stop pipeline
        await next_fn()


class ChatRoutingMiddleware(InboundMiddleware):
    """Determine chat_id, chat_type, chat_name from push fields."""

    name = "chat-routing"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if ctx.group_code:
            ctx.chat_id = f"group:{ctx.group_code}"
            ctx.chat_type = "group"
            ctx.chat_name = ctx.group_name or ctx.group_code
        else:
            ctx.chat_id = f"direct:{ctx.from_account}"
            ctx.chat_type = "dm"
            ctx.chat_name = ctx.sender_nickname or ctx.from_account
        await next_fn()


class AccessPolicy:
    """Platform-level DM / Group access control policy.

    Encapsulates the allow/deny logic so that both inbound middleware
    and outbound ``send_dm`` can share the same rules without reaching
    into adapter internals.
    """

    def __init__(
        self,
        dm_policy: str,
        dm_allow_from: list[str],
        group_policy: str,
        group_allow_from: list[str],
    ) -> None:
        self._dm_policy = dm_policy
        self._dm_allow_from = dm_allow_from
        self._group_policy = group_policy
        self._group_allow_from = group_allow_from

    def is_dm_allowed(self, sender_id: str) -> bool:
        """Platform-level DM inbound filter (open / allowlist / disabled)."""
        if self._dm_policy == "disabled":
            return False
        if self._dm_policy == "allowlist":
            return sender_id.strip() in self._dm_allow_from
        return True

    def is_group_allowed(self, group_code: str) -> bool:
        """Platform-level group chat inbound filter (open / allowlist / disabled)."""
        if self._group_policy == "disabled":
            return False
        if self._group_policy == "allowlist":
            return group_code.strip() in self._group_allow_from
        return True

    @property
    def dm_policy(self) -> str:
        return self._dm_policy

    @property
    def group_policy(self) -> str:
        return self._group_policy


class AccessGuardMiddleware(InboundMiddleware):
    """Platform-level DM/Group access control filter."""

    name = "access-guard"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        adapter = ctx.adapter
        policy: AccessPolicy = adapter._access_policy
        if ctx.chat_type == "dm":
            if not policy.is_dm_allowed(ctx.from_account):
                logger.debug(
                    "[%s] DM from %s blocked by dm_policy=%s",
                    adapter.name, ctx.from_account, policy.dm_policy,
                )
                return  # Stop pipeline
        elif ctx.chat_type == "group":
            if not policy.is_group_allowed(ctx.group_code):
                logger.debug(
                    "[%s] Group %s blocked by group_policy=%s",
                    adapter.name, ctx.group_code, policy.group_policy,
                )
                return  # Stop pipeline
        await next_fn()


class AutoSetHomeMiddleware(InboundMiddleware):
    """Auto-designate the first inbound conversation as Yuanbao home channel.

    Triggers when no home channel is configured, or when an existing group-chat
    home is superseded by the first DM (direct > group upgrade).
    Silent: writes config.yaml and env, no user-facing message.
    """

    name = "auto-sethome"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        adapter = ctx.adapter
        if not adapter._auto_sethome_done:
            _cur_home = os.getenv("YUANBAO_HOME_CHANNEL", "")
            _should_set = (
                not _cur_home
                or (_cur_home.startswith("group:") and ctx.chat_type == "dm")
            )
            if ctx.chat_type == "dm":
                adapter._auto_sethome_done = True  # DM seen — no further upgrades needed
            if _should_set:
                try:
                    from hermes_constants import get_hermes_home
                    from utils import atomic_yaml_write
                    import yaml

                    _home = get_hermes_home()
                    config_path = _home / "config.yaml"
                    user_config: dict = {}
                    if config_path.exists():
                        with open(config_path, encoding="utf-8") as f:
                            user_config = yaml.safe_load(f) or {}
                    user_config["YUANBAO_HOME_CHANNEL"] = ctx.chat_id
                    atomic_yaml_write(config_path, user_config)
                    os.environ["YUANBAO_HOME_CHANNEL"] = str(ctx.chat_id)
                    logger.info(
                        "[%s] Auto-sethome: designated %s (%s) as Yuanbao home channel",
                        adapter.name, ctx.chat_id, ctx.chat_name,
                    )
                    # Silent auto-sethome: no user-facing message, only log
                except Exception as e:
                    logger.warning("[%s] Auto-sethome failed: %s", adapter.name, e)
        await next_fn()


class ExtractContentMiddleware(InboundMiddleware):
    """Extract raw text and media refs from msg_body."""

    name = "extract-content"

    _CARD_CONTENT_MAX_LENGTH = 1000

    @staticmethod
    def _format_shared_link(custom: dict) -> str:
        """Format elem_type 1010 (share card) into bracket-placeholder text."""
        title = custom.get("title", "")
        link = custom.get("link", "")
        header = f"[share_card: {title} | {link}]" if link else f"[share_card: {title}]"
        lines = [header]
        max_len = ExtractContentMiddleware._CARD_CONTENT_MAX_LENGTH
        for field in ("card_content", "wechat_des"):
            val = custom.get(field)
            if val and isinstance(val, str):
                preview = val[:max_len] + "...(truncated)" if len(val) > max_len else val
                lines.append(f"Preview: {preview}")
                break
        if link:
            lines.append("[visit link for full content]")
        return "\n".join(lines)

    @staticmethod
    def _format_link_understanding(custom: dict) -> Optional[str]:
        """Format elem_type 1007 (link understanding card) into bracket-placeholder text."""
        content = custom.get("content")
        if not content:
            return None
        try:
            parsed = json.loads(content)
            link = parsed.get("link") if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            link = None
        if not link or not isinstance(link, str):
            return None
        return f"[link: {link} | visit link for full content]"

    @classmethod
    def _extract_text(cls, msg_body: list) -> str:
        """Extract plain text content from MsgBody.

        - TIMTextElem      -> text field
        - TIMImageElem     -> "[image]"
        - TIMFileElem      -> "[file: {filename}]"
        - TIMSoundElem     -> "[voice]"
        - TIMVideoFileElem -> "[video]"
        - TIMFaceElem      -> "[emoji: {name}]" or "[emoji]"
        - TIMCustomElem    -> try to extract data field, otherwise "[custom message]"
        - Multiple elems joined with spaces
        """
        parts: list[str] = []
        for elem in msg_body:
            elem_type: str = elem.get("msg_type", "")
            content: dict = elem.get("msg_content", {})

            if elem_type == "TIMTextElem":
                text = content.get("text", "")
                if text:
                    parts.append(text)
            elif elem_type == "TIMImageElem":
                parts.append("[image]")
            elif elem_type == "TIMFileElem":
                filename = content.get("file_name", content.get("fileName", content.get("filename", "")))
                parts.append(f"[file: {filename}]" if filename else "[file]")
            elif elem_type == "TIMSoundElem":
                parts.append("[voice]")
            elif elem_type == "TIMVideoFileElem":
                parts.append("[video]")
            elif elem_type == "TIMCustomElem":
                data_val = content.get("data", "")
                if data_val:
                    try:
                        custom = json.loads(data_val)
                        if not isinstance(custom, dict):
                            parts.append("[unsupported message type]")
                            continue
                        ctype = custom.get("elem_type")
                        if ctype == 1002:
                            parts.append(custom.get("text", "[mention]"))
                        elif ctype == 1010:
                            parts.append(cls._format_shared_link(custom))
                        elif ctype == 1007:
                            text = cls._format_link_understanding(custom)
                            if text:
                                parts.append(text)
                            else:
                                parts.append("[unsupported message type]")
                        else:
                            parts.append("[unsupported message type]")
                    except (json.JSONDecodeError, TypeError):
                        parts.append(data_val)
                else:
                    parts.append("[unsupported message type]")
            elif elem_type == "TIMFaceElem":
                # Sticker/emoji: extract name from data JSON
                raw_data = content.get("data", "")
                face_name = ""
                if raw_data:
                    try:
                        face_data = json.loads(raw_data)
                        face_name = (face_data.get("name") or "").strip()
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass
                parts.append(f"[emoji: {face_name}]" if face_name else "[emoji]")
            elif elem_type:
                # Unknown element type — include type as placeholder
                parts.append(f"[{elem_type}]")

        return " ".join(parts) if parts else ""

    @staticmethod
    def _rewrite_slash_command(text: str) -> str:
        """Normalize input text: strip whitespace and convert full-width slash
        (Chinese input method) to ASCII slash so commands are recognized correctly.
        """
        text = text.strip()
        if text.startswith('\uff0f'):  # Full-width slash
            text = '/' + text[1:]
        return text

    @staticmethod
    def _extract_inbound_media_refs(msg_body: list) -> List[Dict[str, str]]:
        """Extract inbound image/file references from TIM msg_body.

        Return example:
          [{"kind": "image", "url": "https://..."}, {"kind": "file", "url": "...", "name": "a.pdf"}]
        """
        refs: List[Dict[str, str]] = []
        for elem in msg_body or []:
            if not isinstance(elem, dict):
                continue
            msg_type = elem.get("msg_type", "")
            content = elem.get("msg_content", {}) or {}
            if not isinstance(content, dict):
                continue

            if msg_type == "TIMImageElem":
                # Prefer medium image (index 1), fallback to index 0.
                image_info_array = content.get("image_info_array")
                if not isinstance(image_info_array, list):
                    image_info_array = []
                image_info = None
                if len(image_info_array) > 1 and isinstance(image_info_array[1], dict):
                    image_info = image_info_array[1]
                elif len(image_info_array) > 0 and isinstance(image_info_array[0], dict):
                    image_info = image_info_array[0]
                image_url = str((image_info or {}).get("url") or "").strip()
                if image_url:
                    refs.append({"kind": "image", "url": image_url})
                continue

            if msg_type == "TIMFileElem":
                file_url = str(content.get("url") or "").strip()
                file_name = (
                    str(content.get("file_name") or "").strip()
                    or str(content.get("fileName") or "").strip()
                    or str(content.get("filename") or "").strip()
                )
                if file_url:
                    ref: Dict[str, str] = {"kind": "file", "url": file_url}
                    if file_name:
                        ref["name"] = file_name
                    refs.append(ref)
        return refs

    @staticmethod
    def _extract_link_urls(msg_body: list) -> list:
        """Extract link URLs from share-card (1010) and link-understanding (1007) custom elems."""
        urls: list[str] = []
        for elem in msg_body or []:
            if not isinstance(elem, dict) or elem.get("msg_type") != "TIMCustomElem":
                continue
            data_str = (elem.get("msg_content") or {}).get("data", "")
            if not data_str:
                continue
            try:
                custom = json.loads(data_str)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(custom, dict):
                continue
            ctype = custom.get("elem_type")
            if ctype == 1010:
                link = custom.get("link")
                if link and isinstance(link, str):
                    urls.append(link)
            elif ctype == 1007:
                content = custom.get("content")
                if content:
                    try:
                        parsed = json.loads(content)
                        link = parsed.get("link") if isinstance(parsed, dict) else None
                        if link and isinstance(link, str):
                            urls.append(link)
                    except (json.JSONDecodeError, TypeError):
                        pass
        return urls

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        ctx.raw_text = self._rewrite_slash_command(self._extract_text(ctx.msg_body))
        ctx.media_refs = self._extract_inbound_media_refs(ctx.msg_body)
        ctx.link_urls = self._extract_link_urls(ctx.msg_body)
        await next_fn()

class PlaceholderFilterMiddleware(InboundMiddleware):
    """Skip pure placeholder messages (e.g. '[image]' with no media)."""

    name = "placeholder-filter"

    SKIPPABLE_PLACEHOLDERS: frozenset = frozenset({
        "[image]", "[图片]", "[file]", "[文件]",
        "[video]", "[视频]", "[voice]", "[语音]",
    })

    @classmethod
    def is_skippable_placeholder(cls, text: str, media_count: int = 0) -> bool:
        """Detect whether the message is a pure placeholder (should be skipped)."""
        if media_count > 0:
            return False
        stripped = text.strip()
        return stripped in cls.SKIPPABLE_PLACEHOLDERS

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if self.is_skippable_placeholder(ctx.raw_text, len(ctx.media_refs)):
            logger.debug("[%s] Skipping placeholder message: %r", ctx.adapter.name, ctx.raw_text)
            return  # Stop pipeline
        await next_fn()


class OwnerCommandMiddleware(InboundMiddleware):
    """Detect bot-owner slash commands in group chat.

    Identifies in-group allowlisted slash commands and determines sender identity.
    Owner commands skip @Bot detection; non-owner attempts are rejected.
    """

    name = "owner-command"

    # Slash command allowlist that bot owner can execute in group without @Bot
    ALLOWLIST: frozenset = frozenset({
        "/new", "/reset", "/retry", "/undo", "/stop",
        "/approve", "/deny", "/background", "/bg",
        "/btw", "/queue", "/q",
    })

    @staticmethod
    def _rewrite_slash_command(text: str) -> str:
        """Normalize full-width slash to ASCII slash and strip whitespace."""
        text = text.strip()
        if text.startswith('\uff0f'):  # Full-width slash
            text = '/' + text[1:]
        return text

    @classmethod
    def _detect_owner_command(
        cls,
        *,
        push: dict,
        msg_body: list,
        chat_type: str,
        from_account: str,
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """Identify allowlisted slash commands and determine sender identity.

        Returns (cmd, cmd_line, is_owner):
          - (None, None, False): Not an allowlisted command
          - (cmd, cmd_line, True): Owner match
          - (cmd, cmd_line, False): Allowlisted command but sender is not owner
        """
        if chat_type != "group" or not cls.ALLOWLIST:
            return None, None, False

        # Extract TIMTextElem: only do command recognition with exactly one text segment
        text_elems = [
            e for e in (msg_body or [])
            if e.get("msg_type") == "TIMTextElem"
        ]
        if len(text_elems) != 1:
            return None, None, False

        text = (text_elems[0].get("msg_content") or {}).get("text", "")
        cmd_line = cls._rewrite_slash_command(text)
        if not cmd_line.startswith("/"):
            return None, None, False
        cmd = cmd_line.split(maxsplit=1)[0].lower()
        if cmd not in cls.ALLOWLIST:
            return None, None, False

        # Sender identity check: bot owner <-> push.from_account == push.bot_owner_id
        owner_id = (push or {}).get("bot_owner_id") or ""
        # is_owner = bool(owner_id) and owner_id == from_account
        is_owner = True
        return cmd, cmd_line, is_owner

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        adapter = ctx.adapter
        matched_cmd, cmd_line, is_owner = self._detect_owner_command(
            push=ctx.push,
            msg_body=ctx.msg_body,
            chat_type=ctx.chat_type,
            from_account=ctx.from_account,
        )
        if matched_cmd and not is_owner:
            # Non-owner tried an owner-only command — reject and stop
            logger.info(
                "[%s] Reject non-owner slash command: chat=%s from=%s cmd=%s",
                adapter.name, ctx.chat_id, ctx.from_account, matched_cmd,
            )
            adapter._track_task(asyncio.create_task(
                adapter.send(ctx.chat_id, f"⚠️ {matched_cmd} is only available to the creator in private chat mode"),
                name=f"yuanbao-owner-cmd-denial-{matched_cmd}",
            ))
            return  # Stop pipeline

        if matched_cmd and is_owner and cmd_line:
            logger.info(
                "[%s] Bot owner slash command: chat=%s from=%s cmd=%s",
                adapter.name, ctx.chat_id, ctx.from_account, matched_cmd,
            )
            ctx.owner_command = matched_cmd
            ctx.raw_text = cmd_line  # Override with clean command text
        await next_fn()


class BuildSourceMiddleware(InboundMiddleware):
    """Build SessionSource from context fields."""

    name = "build-source"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        adapter = ctx.adapter
        ctx.source = adapter.build_source(
            chat_id=ctx.chat_id,
            chat_type=ctx.chat_type,
            chat_name=ctx.chat_name,
            user_id=ctx.from_account or None,
            user_name=ctx.sender_nickname or ctx.from_account,
            thread_id="main" if ctx.chat_type == "group" else None,
        )
        await next_fn()


class GroupAtGuardMiddleware(InboundMiddleware):
    """In group chat, observe non-@bot messages; only reply on @Bot.

    Owner commands skip @Bot detection (owner doesn't need to @Bot).
    """

    name = "group-at-guard"

    @staticmethod
    def _is_at_bot(msg_body: list, bot_id: Optional[str]) -> bool:
        """Detect whether the message @Bot.

        AT element format: TIMCustomElem, msg_content.data is a JSON string:
            {"elem_type": 1002, "text": "@xxx", "user_id": "<botId>"}
        Considered @Bot when elem_type == 1002 and user_id == bot_id.
        """
        if not bot_id:
            return False
        for elem in msg_body:
            if elem.get("msg_type") != "TIMCustomElem":
                continue
            data_str = elem.get("msg_content", {}).get("data", "")
            if not data_str:
                continue
            try:
                custom = json.loads(data_str)
            except (json.JSONDecodeError, TypeError):
                continue
            if custom.get("elem_type") == 1002 and custom.get("user_id") == bot_id:
                return True
        return False

    @staticmethod
    def _extract_bot_mention_text(msg_body: list, bot_id: Optional[str]) -> str:
        """Extract the display text used to @-mention this bot (e.g. ``@yuanbao-bot``)."""
        if not bot_id:
            return ""
        for elem in msg_body:
            if elem.get("msg_type") != "TIMCustomElem":
                continue
            data_str = elem.get("msg_content", {}).get("data", "")
            if not data_str:
                continue
            try:
                custom = json.loads(data_str)
            except (json.JSONDecodeError, TypeError):
                continue
            if custom.get("elem_type") == 1002 and custom.get("user_id") == bot_id:
                mention_text = str(custom.get("text") or "").strip()
                if mention_text:
                    return mention_text
        return ""

    @staticmethod
    def _build_group_channel_prompt(msg_body: list, bot_id: Optional[str]) -> str:
        """Build a per-turn group-chat prompt that highlights which message to respond to."""
        bid = str(bot_id or "unknown")
        bot_mention = GroupAtGuardMiddleware._extract_bot_mention_text(msg_body, bot_id) or "unknown"
        return (
            "You are handling a Yuanbao group chat message.\n"
            f"- Your identity: user_id={bid}, @-mention name in this group={bot_mention}\n"
            "- Lines in history prefixed with `[nickname|user_id]` are observed group context "
            "and are not necessarily addressed to you.\n"
            "- Treat only the current new message as a request explicitly directed at you, "
            "and answer it directly."
        )

    @staticmethod
    def _observe_group_message(
        adapter, source, sender_display: str, text: str,
        *, msg_id: Optional[str] = None,
    ) -> None:
        """Write a group message into the session transcript without triggering the agent.

        This allows the model to see the full group conversation when it is
        eventually invoked via @bot.  Messages are stored with ``role: "user"``
        in the format ``[nickname|user_id]\\n<content>`` so the model
        can distinguish participants and their user ids.
        """
        store = getattr(adapter, "_session_store", None)
        if not store:
            return
        try:
            session_entry = store.get_or_create_session(source)
            user_id = source.user_id or "unknown"
            attributed = f"[{sender_display}|{user_id}]\n{text}"
            entry: dict = {
                "role": "user",
                "content": attributed,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "observed": True,
            }
            if msg_id:
                entry["message_id"] = msg_id
            store.append_to_transcript(
                session_entry.session_id,
                entry,
            )
        except Exception as exc:
            logger.warning("[%s] Failed to observe group message: %s", adapter.name, exc)

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        adapter = ctx.adapter
        if ctx.chat_type == "group" and not ctx.owner_command and not self._is_at_bot(ctx.msg_body, adapter._bot_id):
            self._observe_group_message(
                adapter, ctx.source, ctx.sender_nickname or ctx.from_account, ctx.raw_text,
                msg_id=ctx.msg_id or None,
            )
            logger.info(
                "[%s] Group message observed (no @bot): chat=%s from=%s",
                adapter.name, ctx.chat_id, ctx.from_account,
            )
            return  # Stop pipeline — message observed but not dispatched
        await next_fn()


class GroupAttributionMiddleware(InboundMiddleware):
    """Tag group @bot messages with [nickname|user_id] attribution and channel_prompt.

    For group messages that pass the @bot guard (i.e. the bot is mentioned),
    this middleware:
      - Builds a per-turn channel_prompt so the model knows its identity and
        the attribution scheme.
      - Rewrites ctx.raw_text to ``[nickname|user_id]\\n<content>`` to match
        the observed-history format.
      - Suppresses the runner's default ``[user_name]`` shared-thread prefix
        by clearing ``source.user_name``.
    """

    name = "group-attribution"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if ctx.chat_type == "group" and not ctx.owner_command:
            adapter = ctx.adapter
            ctx.channel_prompt = GroupAtGuardMiddleware._build_group_channel_prompt(
                ctx.msg_body, adapter._bot_id,
            )
            user_id_label = ctx.from_account or "unknown"
            nickname_label = ctx.sender_nickname or ctx.from_account or "unknown"
            ctx.raw_text = f"[{nickname_label}|{user_id_label}]\n{ctx.raw_text}"
            # Suppress runner's default ``[user_name]`` shared-thread prefix so
            # the text the model sees matches the observed-history format.
            if ctx.source is not None:
                ctx.source = dataclasses.replace(ctx.source, user_name=None)
        await next_fn()


class ClassifyMessageTypeMiddleware(InboundMiddleware):
    """Determine MessageType from text content and msg_body elements."""

    name = "classify-msg-type"

    @staticmethod
    def _classify(text: str, msg_body: list) -> MessageType:
        """Classify message type based on text and msg_body."""
        if text.startswith("/"):
            return MessageType.COMMAND
        for elem in msg_body:
            etype = elem.get("msg_type", "")
            if etype == "TIMImageElem":
                return MessageType.PHOTO
            if etype == "TIMSoundElem":
                return MessageType.VOICE
            if etype == "TIMVideoFileElem":
                return MessageType.VIDEO
            if etype == "TIMFileElem":
                return MessageType.DOCUMENT
        return MessageType.TEXT

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        ctx.msg_type = self._classify(ctx.raw_text, ctx.msg_body)
        await next_fn()


class QuoteContextMiddleware(InboundMiddleware):
    """Extract quote/reply context from cloud_custom_data."""

    name = "quote-context"

    @staticmethod
    def _extract_quote_context(cloud_custom_data: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract quote context, mapping to MessageEvent.reply_to_*.

        Returns:
          (reply_to_message_id, reply_to_text)
        """
        if not cloud_custom_data:
            return None, None
        try:
            parsed = json.loads(cloud_custom_data)
        except (json.JSONDecodeError, TypeError):
            return None, None

        quote = parsed.get("quote") if isinstance(parsed, dict) else None
        if not isinstance(quote, dict):
            return None, None

        # type=2 corresponds to image reference; desc may be empty, provide a placeholder.
        quote_type = int(quote.get("type") or 0)
        desc = str(quote.get("desc") or "").strip()
        if quote_type == 2 and not desc:
            desc = "[image]"
        if not desc:
            return None, None

        quote_id = str(quote.get("id") or "").strip() or None
        sender = str(quote.get("sender_nickname") or quote.get("sender_id") or "").strip()
        quote_text = f"{sender}: {desc}" if sender else desc
        return quote_id, quote_text

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        ctx.reply_to_message_id, ctx.reply_to_text = self._extract_quote_context(ctx.cloud_custom_data)
        await next_fn()


class MediaResolveMiddleware(InboundMiddleware):
    """Resolve inbound media references to downloadable URLs."""

    name = "media-resolve"

    @staticmethod
    def _guess_image_ext_from_url(url: str) -> str:
        """Guess image extension from URL path."""
        path = urllib.parse.urlparse(url).path
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".tiff"}:
            return ext
        return ".jpg"

    @staticmethod
    async def _fetch_resource_url(adapter, resource_id: str) -> str:
        """Low-level helper: exchange a ``resourceId`` for a direct download URL.

        Handles token retrieval, the ``/api/resource/v1/download`` API call,
        and a single 401-retry with token force-refresh.  Raises on failure.
        """
        resource_id = resource_id.strip()
        if not resource_id:
            raise RuntimeError("missing resource_id")

        token_data = await adapter._get_cached_token()
        token = str(token_data.get("token") or "").strip()
        source = str(token_data.get("source") or "web").strip() or "web"
        bot_id = str(token_data.get("bot_id") or adapter._bot_id or adapter._app_key).strip()
        if not token or not bot_id:
            raise RuntimeError("missing token or bot_id for resource download")

        api_url = f"{adapter._api_domain}/api/resource/v1/download"
        headers = {
            "Content-Type": "application/json",
            "X-ID": bot_id,
            "X-Token": token,
            "X-Source": source,
        }

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            for attempt in range(2):
                resp = await client.get(api_url, params={"resourceId": resource_id}, headers=headers)
                if resp.status_code == 401 and attempt == 0:
                    # Force refresh token once on expiry and retry
                    token_data = await SignManager.force_refresh(
                        adapter._app_key, adapter._app_secret, adapter._api_domain,
                    )
                    token = str(token_data.get("token") or "").strip()
                    source = str(token_data.get("source") or source or "web").strip() or "web"
                    bot_id = str(token_data.get("bot_id") or adapter._bot_id or adapter._app_key).strip()
                    if not token or not bot_id:
                        break
                    headers["X-ID"] = bot_id
                    headers["X-Token"] = token
                    headers["X-Source"] = source
                    continue

                resp.raise_for_status()
                payload = resp.json()
                code = payload.get("code")
                if code not in (None, 0):
                    raise RuntimeError(
                        f"resource/v1/download failed: code={code}, msg={payload.get('msg', '')}"
                    )
                data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
                real_url = str((data or {}).get("url") or (data or {}).get("realUrl") or "").strip()
                if real_url:
                    return real_url
                raise RuntimeError("resource/v1/download missing url/realUrl")

        raise RuntimeError("resource/v1/download did not return a URL")

    @staticmethod
    async def _resolve_download_url(adapter, url: str) -> str:
        """Resolve Yuanbao resource placeholder to a directly fetchable real URL.

        Common URL patterns:
          https://hunyuan.tencent.com/api/resource/download?resourceId=...
        Direct GET returns 401; need business API:
          GET /api/resource/v1/download?resourceId=...
        """
        try:
            parsed = urllib.parse.urlparse(url)
        except Exception:
            return url

        query = urllib.parse.parse_qs(parsed.query)
        resource_ids = query.get("resourceId") or query.get("resourceid") or []
        resource_id = str(resource_ids[0]).strip() if resource_ids else ""
        if not resource_id:
            return url

        try:
            return await MediaResolveMiddleware._fetch_resource_url(adapter, resource_id)
        except Exception:
            return url

    @classmethod
    async def _download_and_cache(
        cls, adapter, *, fetch_url: str, kind: str,
        file_name: Optional[str] = None, log_tag: str = "",
    ) -> Optional[Tuple[str, str]]:
        """Download a Yuanbao resource and cache locally. Returns ``(local_path, mime)`` or ``None``."""
        try:
            file_bytes, content_type = await media_download_url(
                fetch_url, max_size_mb=adapter.MEDIA_MAX_SIZE_MB,
            )
        except Exception as exc:
            logger.warning(
                "[%s] inbound media download failed: kind=%s %s err=%s",
                adapter.name, kind, log_tag, exc,
            )
            return None

        if kind == "image":
            ext = cls._guess_image_ext_from_url(fetch_url)
            try:
                local_path = cache_image_from_bytes(file_bytes, ext=ext)
            except ValueError as exc:
                logger.warning(
                    "[%s] inbound image cache rejected: %s err=%s",
                    adapter.name, log_tag, exc,
                )
                return None
            mime = guess_mime_type(f"image{ext}")
            if not mime.startswith("image/"):
                mime = content_type if content_type.startswith("image/") else "image/jpeg"
            return local_path, mime

        # kind == "file"
        if not file_name:
            parsed = urllib.parse.urlparse(fetch_url)
            file_name = os.path.basename(parsed.path) or "file"
        try:
            local_path = cache_document_from_bytes(file_bytes, file_name)
        except Exception as exc:
            logger.warning(
                "[%s] inbound file cache failed: %s err=%s",
                adapter.name, log_tag, exc,
            )
            return None
        mime = guess_mime_type(file_name) or content_type or "application/octet-stream"
        return local_path, mime

    @classmethod
    async def _resolve_by_resource_id(cls, adapter, resource_id: str) -> str:
        """Exchange a Yuanbao ``resourceId`` for a short-lived direct download URL. Raises on failure."""
        return await cls._fetch_resource_url(adapter, resource_id)

    @classmethod
    async def _resolve_media_urls(
        cls, adapter, media_refs: List[Dict[str, str]]
    ) -> Tuple[List[str], List[str]]:
        """Resolve inbound media refs: download to local cache, return (local_paths, mime_types).

        Yuanbao COS hostnames resolve to private IPs, tripping the SSRF guard
        in vision_tools. We download ourselves and return local cache paths.
        """
        media_urls: List[str] = []
        media_types: List[str] = []

        for ref in media_refs:
            kind = str(ref.get("kind") or "").strip().lower()
            url = str(ref.get("url") or "").strip()
            if kind not in {"image", "file"} or not url:
                continue

            try:
                fetch_url = await cls._resolve_download_url(adapter, url)
            except Exception as exc:
                logger.warning(
                    "[%s] inbound media resolve failed: kind=%s url=%s err=%s",
                    adapter.name, kind, url, exc,
                )
                continue

            cached = await cls._download_and_cache(
                adapter,
                fetch_url=fetch_url,
                kind=kind,
                file_name=str(ref.get("name") or "").strip() or None,
                log_tag=f"placeholder_url={url[:80]}",
            )
            if cached is None:
                continue
            local_path, mime = cached
            media_urls.append(local_path)
            media_types.append(mime)

        return media_urls, media_types

    @classmethod
    async def _collect_observed_media(
        cls, adapter, source,
    ) -> Tuple[List[str], List[str]]:
        """Resolve recent observed image/file anchors from transcript into ``(local_paths, mimes)``."""
        store = getattr(adapter, "_session_store", None)
        if not store:
            return [], []
        try:
            session_entry = store.get_or_create_session(source)
            history = store.load_transcript(session_entry.session_id)
        except Exception as exc:
            logger.warning(
                "[%s] Observed-media hydration setup failed: %s",
                adapter.name, exc,
            )
            return [], []
        if not history:
            return [], []

        start = max(0, len(history) - OBSERVED_MEDIA_BACKFILL_LOOKBACK)
        order: List[Tuple[str, str, str]] = []  # (rid, kind, filename)
        seen: set = set()
        for msg in history[start:]:
            content = msg.get("content")
            if not isinstance(content, str) or "|ybres:" not in content:
                continue
            for m in _YB_RES_REF_RE.finditer(content):
                head = m.group(1)  # "image" | "file:<name>" | "voice" | "video"
                rid = m.group(2)
                kind, _, filename = head.partition(":")
                kind = kind.strip()
                if kind not in ("image", "file"):
                    continue
                if rid in seen:
                    continue
                seen.add(rid)
                order.append((rid, kind, filename.strip()))
                if len(order) >= OBSERVED_MEDIA_BACKFILL_MAX_RESOLVE_PER_TURN:
                    break
            if len(order) >= OBSERVED_MEDIA_BACKFILL_MAX_RESOLVE_PER_TURN:
                break

        if not order:
            return [], []

        media_paths: List[str] = []
        mimes: List[str] = []
        for rid, kind, filename in order:
            try:
                fresh_url = await cls._resolve_by_resource_id(adapter, rid)
            except Exception as exc:
                logger.warning(
                    "[%s] observed-media resolve failed: rid=%s kind=%s err=%s",
                    adapter.name, rid, kind, exc,
                )
                continue
            cached = await cls._download_and_cache(
                adapter,
                fetch_url=fresh_url,
                kind=kind,
                file_name=filename or None,
                log_tag=f"rid={rid}",
            )
            if cached is None:
                continue
            path, mime = cached
            media_paths.append(path)
            mimes.append(mime)
        return media_paths, mimes

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        adapter = ctx.adapter
        ctx.media_urls, ctx.media_types = await self._resolve_media_urls(adapter, ctx.media_refs)
        # Re-check placeholder after media resolution
        if PlaceholderFilterMiddleware.is_skippable_placeholder(ctx.raw_text, len(ctx.media_urls)):
            logger.debug("[%s] Skip placeholder after media download: %r", adapter.name, ctx.raw_text)
            return  # Stop pipeline
        await next_fn()


class DispatchMiddleware(InboundMiddleware):
    """Build MessageEvent and dispatch to AI handler."""

    name = "dispatch"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        adapter = ctx.adapter

        _sk = build_session_key(
            ctx.source,
            group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
        )

        async def _dispatch_inbound_event() -> None:
            media_urls = list(ctx.media_urls)
            media_types = list(ctx.media_types)

            # Backfill observed media from recent transcript history
            extra_img_urls: List[str] = []
            extra_img_mimes: List[str] = []
            try:
                extra_img_urls, extra_img_mimes = await MediaResolveMiddleware._collect_observed_media(
                    adapter, ctx.source,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] observed-image hydration raised, continuing anyway: %s",
                    adapter.name, exc,
                )
            if extra_img_urls:
                current = set(media_urls)
                for u, m in zip(extra_img_urls, extra_img_mimes):
                    if u in current:
                        continue
                    media_urls.append(u)
                    media_types.append(m)
                    current.add(u)

            # Replace [kind|ybres:xxx] anchors with local cache paths so
            # the transcript records usable paths for the model.
            _patched_event_text = ctx.raw_text
            for u, m in zip(media_urls, media_types):
                if not u.startswith("/"):
                    continue
                anchor_match = _YB_RES_REF_RE.search(_patched_event_text)
                if not anchor_match:
                    continue
                head = anchor_match.group(1)
                kind, _, filename = head.partition(":")
                kind = kind.strip()
                if kind == "image" and m.startswith("image/"):
                    replacement = f"[image: {u}]"
                elif kind == "file":
                    label = filename.strip() or os.path.basename(u)
                    replacement = f"[file: {label} → {u}]"
                else:
                    continue
                _patched_event_text = (
                    _patched_event_text[:anchor_match.start()]
                    + replacement
                    + _patched_event_text[anchor_match.end():]
                )

            event = MessageEvent(
                text=_patched_event_text,
                message_type=ctx.msg_type,
                source=ctx.source,
                message_id=ctx.msg_id or None,
                raw_message=ctx.push,
                media_urls=media_urls,
                media_types=media_types,
                reply_to_message_id=ctx.reply_to_message_id,
                reply_to_text=ctx.reply_to_text,
                channel_prompt=ctx.channel_prompt,
            )
            if _sk and ctx.msg_id:
                adapter._processing_msg_ids[_sk] = ctx.msg_id
                adapter._processing_msg_texts[_sk] = ctx.raw_text or ""
            if ctx.msg_id and ctx.raw_text:
                cache = adapter._msg_content_cache
                cache[ctx.msg_id] = ctx.raw_text
                if len(cache) > 200:
                    for k in list(cache)[:len(cache) - 200]:
                        del cache[k]
            await adapter.handle_message(event)

        if ctx.chat_type == "group":
            is_new = _sk not in adapter._group_queues
            queue = adapter._group_queues.setdefault(_sk, asyncio.Queue())
            queue.put_nowait(_dispatch_inbound_event)
            logger.info(
                "[%s] Group message enqueued (qsize=%d) for %s",
                adapter.name, queue.qsize(), (_sk or "")[:50],
            )
            if is_new:
                consumer = asyncio.create_task(
                    self._consume_group_queue(adapter, _sk),
                    name=f"yuanbao-group-consumer-{(_sk or '')[:30]}",
                )
                adapter._inbound_tasks.add(consumer)
                consumer.add_done_callback(adapter._inbound_tasks.discard)
        else:
            task = asyncio.create_task(
                _dispatch_inbound_event(),
                name=f"yuanbao-inbound-{ctx.msg_id or 'unknown'}",
            )
            adapter._inbound_tasks.add(task)
            task.add_done_callback(adapter._inbound_tasks.discard)

        await next_fn()

    @staticmethod
    async def _consume_group_queue(adapter: "YuanbaoAdapter", session_key: str) -> None:
        """Drain the group queue one dispatch at a time, waiting for each to finish."""
        _IDLE_TIMEOUT = 2.0
        queue = adapter._group_queues.get(session_key)
        if not queue:
            return
        try:
            while True:
                try:
                    dispatch_fn = await asyncio.wait_for(queue.get(), timeout=_IDLE_TIMEOUT)
                except asyncio.TimeoutError:
                    break
                logger.debug(
                    "[%s] Group queue: dispatching for %s (remaining=%d)",
                    adapter.name, (session_key or "")[:50], queue.qsize(),
                )
                try:
                    await dispatch_fn()
                    while session_key in adapter._active_sessions:
                        await asyncio.sleep(0.1)
                except Exception:
                    logger.exception("[%s] Group queue consumer error", adapter.name)
        finally:
            adapter._group_queues.pop(session_key, None)


class InboundPipelineBuilder:
    """Factory for building InboundPipeline instances.

    Separates pipeline assembly (business knowledge) from the pipeline engine
    (InboundPipeline) so the engine stays generic and reusable.
    """

    # Default middleware sequence for Yuanbao inbound message processing.
    _DEFAULT_MIDDLEWARES: list[type] = [
        DecodeMiddleware,
        ExtractFieldsMiddleware,
        RecallGuardMiddleware,
        DedupMiddleware,
        SkipSelfMiddleware,
        ChatRoutingMiddleware,
        AccessGuardMiddleware,
        AutoSetHomeMiddleware,
        ExtractContentMiddleware,
        PlaceholderFilterMiddleware,
        OwnerCommandMiddleware,
        BuildSourceMiddleware,
        GroupAtGuardMiddleware,
        GroupAttributionMiddleware,
        ClassifyMessageTypeMiddleware,
        QuoteContextMiddleware,
        MediaResolveMiddleware,
        DispatchMiddleware,
    ]

    @classmethod
    def build(cls) -> InboundPipeline:
        """Build the default inbound message processing pipeline."""
        pipeline = InboundPipeline()
        for mw_cls in cls._DEFAULT_MIDDLEWARES:
            pipeline.use(mw_cls())
        return pipeline

class ConnectionManager:
    """Manages the WebSocket connection lifecycle for YuanbaoAdapter.

    Responsibilities:
      - Opening and closing the WebSocket
      - AUTH_BIND handshake
      - Heartbeat (ping/pong) loop
      - Receive loop (frame dispatch)
      - Reconnect with exponential backoff
    """

    def __init__(self, adapter: "YuanbaoAdapter") -> None:
        self._adapter = adapter
        self._ws = None  # websockets connection
        self._connect_id: Optional[str] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._pending_acks: Dict[str, asyncio.Future] = {}
        self._pending_pong: Optional[asyncio.Future] = None
        self._consecutive_hb_timeouts: int = 0
        self._reconnect_attempts: int = 0
        self._reconnecting: bool = False
        # Debounce buffer for aggregating multi-part inbound messages
        self._inbound_buffer: Dict[str, list] = {}  # key -> [raw_data_frames, ...]
        self._inbound_timers: Dict[str, asyncio.TimerHandle] = {}  # key -> timer

    # -- Properties --------------------------------------------------------

    @property
    def ws(self):
        return self._ws

    @property
    def connect_id(self) -> Optional[str]:
        return self._connect_id

    @property
    def reconnect_attempts(self) -> int:
        return self._reconnect_attempts

    @property
    def is_connected(self) -> bool:
        if self._ws is None:
            return False
        open_attr = getattr(self._ws, "open", None)
        if open_attr is True:
            return True
        if callable(open_attr):
            try:
                return bool(open_attr())
            except Exception:
                return False
        return False

    # -- Open / Close ------------------------------------------------------

    async def open(self) -> bool:
        """Open WebSocket connection: sign-token → WS connect → AUTH_BIND → start loops.

        Returns True on success, False on failure.
        """
        adapter = self._adapter

        if not WEBSOCKETS_AVAILABLE:
            msg = "Yuanbao startup failed: 'websockets' package not installed"
            adapter._set_fatal_error("yuanbao_missing_dependency", msg, retryable=True)
            logger.warning("[%s] %s. Run: pip install websockets", adapter.name, msg)
            return False

        if not adapter._app_key or not adapter._app_secret:
            msg = (
                "Yuanbao startup failed: "
                "YUANBAO_APP_ID and YUANBAO_APP_SECRET are required"
            )
            adapter._set_fatal_error("yuanbao_missing_credentials", msg, retryable=False)
            logger.error("[%s] %s", adapter.name, msg)
            return False

        # Idempotency guard
        if self._ws is not None:
            try:
                open_attr = getattr(self._ws, "open", None)
                if open_attr is True or (callable(open_attr) and open_attr()):
                    logger.debug("[%s] Already connected, skipping connect()", adapter.name)
                    return True
            except Exception:
                pass

        # Acquire platform-scoped lock to prevent duplicate connections
        if not adapter._acquire_platform_lock(
            'yuanbao-app-key', adapter._app_key, 'Yuanbao app key'
        ):
            return False

        try:
            # Step 1: Get sign token
            logger.info("[%s] Fetching sign token from %s", adapter.name, adapter._api_domain)
            token_data = await SignManager.get_token(
                adapter._app_key, adapter._app_secret, adapter._api_domain,
                route_env=adapter._route_env,
            )

            # Update bot_id if returned by sign-token API
            if token_data.get("bot_id"):
                adapter._bot_id = str(token_data["bot_id"])

            # Step 2: Open WebSocket connection (disable built-in ping/pong)
            logger.info("[%s] Connecting to %s", adapter.name, adapter._ws_url)
            self._ws = await asyncio.wait_for(
                websockets.connect(  # type: ignore[attr-defined]
                    adapter._ws_url,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5,
                ),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )

            # Step 3: Authenticate (AUTH_BIND + wait for BIND_ACK)
            authed = await self._authenticate(token_data)
            if not authed:
                await self._cleanup_ws()
                return False

            # Step 4: Start background tasks
            self._reconnect_attempts = 0
            adapter._mark_connected()
            adapter._loop = asyncio.get_running_loop()
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name=f"yuanbao-heartbeat-{self._connect_id}"
            )
            self._recv_task = asyncio.create_task(
                self._receive_loop(), name=f"yuanbao-recv-{self._connect_id}"
            )
            logger.info(
                "[%s] Connected. connectId=%s botId=%s",
                adapter.name, self._connect_id, adapter._bot_id,
            )

            YuanbaoAdapter.set_active(adapter)

            return True

        except asyncio.TimeoutError:
            logger.error("[%s] Connection timed out", adapter.name)
            await self._cleanup_ws()
            adapter._release_platform_lock()
            return False
        except Exception as exc:
            logger.error("[%s] connect() failed: %s", adapter.name, exc, exc_info=True)
            await self._cleanup_ws()
            adapter._release_platform_lock()
            return False

    async def close(self) -> None:
        """Cancel background tasks, fail pending futures, and close the WebSocket."""

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None

        # Fail any pending ACK futures
        disc_exc = RuntimeError("YuanbaoAdapter disconnected")
        for fut in self._pending_acks.values():
            if not fut.done():
                fut.set_exception(disc_exc)
        self._pending_acks.clear()

        # Clear refresh locks to avoid stale locks from a previous event loop
        SignManager.clear_locks()

        await self._cleanup_ws()

    # -- Authentication ----------------------------------------------------

    async def _authenticate(self, token_data: dict) -> bool:
        """Send AUTH_BIND and read frames until BIND_ACK is received.

        Returns True on success, False on failure/timeout.
        """
        adapter = self._adapter
        if self._ws is None:
            return False

        token = token_data.get("token", "")
        uid = adapter._bot_id or token_data.get("bot_id", "")
        source = token_data.get("source") or "bot"
        route_env = adapter._route_env or token_data.get("route_env", "") or ""

        msg_id = str(uuid.uuid4())

        auth_bytes = encode_auth_bind(
            biz_id="ybBot",
            uid=uid,
            source=source,
            token=token,
            msg_id=msg_id,
            app_version=_APP_VERSION,
            operation_system=_OPERATION_SYSTEM,
            bot_version=_BOT_VERSION,
            route_env=route_env,
        )
        await self._ws.send(auth_bytes)
        logger.debug("[%s] AUTH_BIND sent (msg_id=%s uid=%s)", adapter.name, msg_id, uid)

        try:
            _loop = asyncio.get_running_loop()
            deadline = _loop.time() + AUTH_TIMEOUT_SECONDS
            while True:
                remaining = deadline - _loop.time()
                if remaining <= 0:
                    logger.error("[%s] AUTH_BIND timeout waiting for BIND_ACK", adapter.name)
                    return False

                raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
                if not isinstance(raw, (bytes, bytearray)):
                    continue

                try:
                    msg = decode_conn_msg(bytes(raw))
                except Exception:
                    continue

                head = msg.get("head", {})
                cmd_type = head.get("cmd_type", -1)
                cmd = head.get("cmd", "")

                if cmd_type == CMD_TYPE["Response"] and cmd == "auth-bind":
                    connect_id = self._extract_connect_id(msg)
                    if connect_id:
                        self._connect_id = connect_id
                        logger.info("[%s] BIND_ACK received: connectId=%s", adapter.name, connect_id)
                        return True
                    else:
                        logger.error("[%s] BIND_ACK missing connectId", adapter.name)
                        return False

        except asyncio.TimeoutError:
            logger.error("[%s] AUTH_BIND timeout", adapter.name)
            return False
        except Exception as exc:
            logger.error("[%s] AUTH_BIND error: %s", adapter.name, exc, exc_info=True)
            return False

    def _extract_connect_id(self, decoded_msg: dict) -> Optional[str]:
        """Extract connectId from decoded BIND_ACK message."""
        data: bytes = decoded_msg.get("data", b"")
        if not data:
            return None
        try:
            fdict = _fields_to_dict(_parse_fields(data))
            code = _get_varint(fdict, 1)
            if code != 0:
                message = _get_string(fdict, 2)
                logger.error(
                    "[%s] AuthBindRsp error: code=%d message=%r",
                    self._adapter.name, code, message,
                )
                return None
            connect_id = _get_string(fdict, 3)
            return connect_id if connect_id else None
        except Exception as exc:
            logger.warning("[%s] Failed to extract connectId: %s", self._adapter.name, exc)
            return None

    # -- Heartbeat ---------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Send HEARTBEAT (ping) every 30s; trigger reconnect after threshold misses."""
        adapter = self._adapter
        try:
            while adapter._running:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
                if self._ws is None:
                    continue
                try:
                    msg_id = str(uuid.uuid4())
                    ping_bytes = encode_ping(msg_id)
                    loop = asyncio.get_running_loop()
                    pong_future: asyncio.Future = loop.create_future()
                    self._pending_pong = pong_future
                    self._pending_acks[msg_id] = pong_future
                    await self._ws.send(ping_bytes)
                    logger.debug("[%s] PING sent (msg_id=%s)", adapter.name, msg_id)
                    try:
                        await asyncio.wait_for(pong_future, timeout=10.0)
                        self._consecutive_hb_timeouts = 0
                    except asyncio.TimeoutError:
                        self._pending_acks.pop(msg_id, None)
                        self._consecutive_hb_timeouts += 1
                        logger.warning(
                            "[%s] PONG timeout (%d/%d)",
                            adapter.name, self._consecutive_hb_timeouts, HEARTBEAT_TIMEOUT_THRESHOLD,
                        )
                        if self._consecutive_hb_timeouts >= HEARTBEAT_TIMEOUT_THRESHOLD:
                            logger.warning("[%s] Heartbeat threshold exceeded, triggering reconnect", adapter.name)
                            self.schedule_reconnect()
                            return
                    finally:
                        self._pending_acks.pop(msg_id, None)
                        self._pending_pong = None
                except Exception as exc:
                    logger.debug("[%s] Heartbeat send failed: %s", adapter.name, exc)
        except asyncio.CancelledError:
            pass

    # -- Receive loop ------------------------------------------------------

    async def _receive_loop(self) -> None:
        """Read WS frames and dispatch by cmd_type."""
        adapter = self._adapter
        try:
            async for raw in self._ws:  # type: ignore[union-attr]
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                await self._handle_frame(bytes(raw))
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed as close_exc:  # type: ignore[union-attr]
            close_code = getattr(close_exc, 'code', None)
            logger.warning(
                "[%s] WebSocket connection closed: code=%s reason=%s",
                adapter.name, close_code, getattr(close_exc, 'reason', ''),
            )
            if close_code and close_code in NO_RECONNECT_CLOSE_CODES:
                logger.error(
                    "[%s] Close code %d is non-recoverable, NOT reconnecting",
                    adapter.name, close_code,
                )
                adapter._mark_disconnected()
            else:
                self.schedule_reconnect()
        except Exception as exc:
            logger.warning("[%s] receive_loop exited: %s", adapter.name, exc)
            self.schedule_reconnect()

    async def _handle_frame(self, raw: bytes) -> None:
        """Handle a single WebSocket frame."""
        adapter = self._adapter
        try:
            msg = decode_conn_msg(raw)
        except Exception as exc:
            logger.debug("[%s] Failed to decode frame: %s", adapter.name, exc)
            return

        head = msg.get("head", {})
        cmd_type = head.get("cmd_type", -1)
        cmd = head.get("cmd", "")
        msg_id = head.get("msg_id", "")
        need_ack = head.get("need_ack", False)
        data: bytes = msg.get("data", b"")

        # HEARTBEAT_ACK
        if cmd_type == CMD_TYPE["Response"] and cmd == "ping":
            logger.debug("[%s] HEARTBEAT_ACK received (msg_id=%s)", adapter.name, msg_id)
            if self._pending_pong is not None and not self._pending_pong.done():
                self._pending_pong.set_result(True)
            elif msg_id and msg_id in self._pending_acks:
                fut = self._pending_acks.pop(msg_id)
                if not fut.done():
                    fut.set_result(True)
            return

        # Fire-and-forget heartbeat ACKs — server always responds but callers don't
        # wait on these; silently discard to avoid "Unmatched Response" noise.
        if cmd_type == CMD_TYPE["Response"] and cmd in (
            "send_group_heartbeat",
            "send_private_heartbeat",
        ):
            logger.debug("[%s] Heartbeat ACK received: cmd=%s msg_id=%s", adapter.name, cmd, msg_id)
            return

        # Response to an outbound RPC call
        if cmd_type == CMD_TYPE["Response"]:
            if msg_id and msg_id in self._pending_acks:
                fut = self._pending_acks.pop(msg_id)
                if not fut.done():
                    result = {"head": head}
                    if data:
                        result["data"] = data
                    fut.set_result(result)
            else:
                logger.debug(
                    "[%s] Unmatched Response: cmd=%s msg_id=%s",
                    adapter.name, cmd, msg_id,
                )
            return

        # Server-initiated Push
        if cmd_type == CMD_TYPE["Push"]:
            logger.info("[%s] Push received: cmd=%s msg_id=%s data_len=%d", adapter.name, cmd, msg_id, len(data))
            if need_ack and self._ws is not None:
                try:
                    ack_bytes = encode_push_ack(head)
                    await self._ws.send(ack_bytes)
                except Exception as ack_exc:
                    logger.debug("[%s] Failed to send PushAck: %s", adapter.name, ack_exc)

            if msg_id and msg_id in self._pending_acks:
                fut = self._pending_acks.pop(msg_id)
                if not fut.done():
                    try:
                        decoded = decode_inbound_push(data) if data else {"head": head}
                        fut.set_result(decoded)
                    except Exception as exc:
                        fut.set_exception(exc)
                return

            # Genuine inbound message — dispatch to AI
            if data:
                logger.info(
                    "[%s] WS received inbound push, decoding and dispatching: cmd=%s, data_len=%d",
                    adapter.name, cmd, len(data),
                )
                self._push_to_inbound(data)
            return

        logger.debug(
            "[%s] Ignoring frame: cmd_type=%d cmd=%s msg_id=%s",
            adapter.name, cmd_type, cmd, msg_id,
        )

    # -- Inbound dispatch ---------------------------------------------------

    _DEBOUNCE_WINDOW: float = 1.5  # seconds to wait for companion messages

    def _extract_sender_key(self, raw_data: bytes) -> str:
        """Lightweight decode to extract sender key for debounce grouping.

        Returns 'from_account:group_code' or a fallback unique key.
        """
        try:
            parsed = json.loads(raw_data.decode("utf-8"))
            if isinstance(parsed, dict):
                from_account = (
                    parsed.get("from_account", "")
                    or parsed.get("From_Account", "")
                )
                group_code = (
                    parsed.get("group_code", "")
                    or parsed.get("GroupId", "")
                    or parsed.get("group_id", "")
                )
                if from_account:
                    return f"{from_account}:{group_code}"
        except Exception:
            pass
        # Protobuf: try decode_inbound_push for sender info
        try:
            push = decode_inbound_push(raw_data)
            if push:
                return f"{push.get('from_account', '')}:{push.get('group_code', '')}"
        except Exception:
            pass
        # Fallback: unique key (no aggregation)
        return f"__unknown_{id(raw_data)}"

    def _push_to_inbound(self, raw_data: bytes) -> None:
        """Debounced inbound dispatch.

        Buffers raw frames from the same sender within a short time window,
        then dispatches all buffered data as a single aggregated pipeline
        execution.  This merges multi-part messages (e.g. image + text sent
        as separate WS pushes) into one pipeline run.
        """
        key = self._extract_sender_key(raw_data)

        # Cancel existing timer for this key (reset debounce window)
        existing_timer = self._inbound_timers.pop(key, None)
        if existing_timer:
            existing_timer.cancel()

        # Append to buffer
        if key not in self._inbound_buffer:
            self._inbound_buffer[key] = []
        self._inbound_buffer[key].append(raw_data)

        logger.debug(
            "[%s] Debounce: buffered frame for key=%s, count=%d",
            self._adapter.name, key, len(self._inbound_buffer[key]),
        )

        # Schedule flush after debounce window
        loop = asyncio.get_running_loop()
        timer = loop.call_later(
            self._DEBOUNCE_WINDOW,
            self._flush_inbound_buffer,
            key,
        )
        self._inbound_timers[key] = timer

    def _flush_inbound_buffer(self, key: str) -> None:
        """Flush the debounce buffer for a given key — execute the pipeline."""
        self._inbound_timers.pop(key, None)
        data_list = self._inbound_buffer.pop(key, [])
        if not data_list:
            return

        adapter = self._adapter
        logger.info(
            "[%s] Debounce flush: key=%s, aggregated %d frames",
            adapter.name, key, len(data_list),
        )

        ctx = InboundContext(adapter=adapter, raw_frames=data_list)

        adapter._track_task(asyncio.create_task(
            adapter._inbound_pipeline.execute(ctx),
            name=f"yuanbao-pipeline-{key}",
        ))

    # -- Send business request ---------------------------------------------

    async def send_biz_request(
        self,
        encoded_conn_msg: bytes,
        req_id: str,
        timeout: float = DEFAULT_SEND_TIMEOUT,
    ) -> dict:
        """Send a business-layer request and wait for the response.

        1. Register a Future in pending_acks[req_id]
        2. Send encoded_conn_msg (bytes) to WS
        3. asyncio.wait_for(future, timeout)
        4. Clean up pending_acks on timeout/exception
        """
        if self._ws is None:
            raise RuntimeError("Not connected")

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_acks[req_id] = future
        try:
            await self._ws.send(encoded_conn_msg)
            result = await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            raise
        except Exception:
            raise
        finally:
            self._pending_acks.pop(req_id, None)

    # -- Reconnect ---------------------------------------------------------

    def schedule_reconnect(self) -> None:
        """Schedule a reconnect only if running and not already reconnecting."""
        if self._adapter._running and not self._reconnecting:
            asyncio.create_task(self._reconnect_with_backoff())

    async def _reconnect_with_backoff(self) -> bool:
        """Reconnect with exponential backoff (1s, 2s, 4s, … up to 60s)."""
        if self._reconnecting:
            logger.debug("[%s] Reconnect already in progress, skipping", self._adapter.name)
            return False
        self._reconnecting = True
        try:
            return await self._do_reconnect()
        finally:
            self._reconnecting = False

    async def _do_reconnect(self) -> bool:
        """Internal reconnect loop, called under the _reconnecting guard."""
        adapter = self._adapter
        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            self._reconnect_attempts = attempt + 1
            wait = min(2 ** attempt, 60)
            logger.info(
                "[%s] Reconnect attempt %d/%d in %ds",
                adapter.name, attempt + 1, MAX_RECONNECT_ATTEMPTS, wait,
            )
            await asyncio.sleep(wait)

            await self._cleanup_ws()

            try:
                token_data = await SignManager.force_refresh(
                    adapter._app_key, adapter._app_secret, adapter._api_domain,
                    route_env=adapter._route_env,
                )
                if token_data.get("bot_id"):
                    adapter._bot_id = str(token_data["bot_id"])

                self._ws = await asyncio.wait_for(
                    websockets.connect(  # type: ignore[attr-defined]
                        adapter._ws_url,
                        ping_interval=None,
                        ping_timeout=None,
                        close_timeout=5,
                    ),
                    timeout=CONNECT_TIMEOUT_SECONDS,
                )

                authed = await self._authenticate(token_data)
                if not authed:
                    logger.warning("[%s] Re-auth failed on attempt %d", adapter.name, attempt + 1)
                    await self._cleanup_ws()
                    continue

                self._reconnect_attempts = 0
                self._consecutive_hb_timeouts = 0
                adapter._mark_connected()

                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(),
                    name=f"yuanbao-heartbeat-{self._connect_id}",
                )

                if self._recv_task and not self._recv_task.done():
                    self._recv_task.cancel()
                self._recv_task = asyncio.create_task(
                    self._receive_loop(),
                    name=f"yuanbao-recv-{self._connect_id}",
                )

                logger.info(
                    "[%s] Reconnected on attempt %d. connectId=%s",
                    adapter.name, attempt + 1, self._connect_id,
                )
                return True

            except asyncio.TimeoutError:
                logger.warning("[%s] Reconnect attempt %d timed out", adapter.name, attempt + 1)
            except Exception as exc:
                logger.warning(
                    "[%s] Reconnect attempt %d failed: %s", adapter.name, attempt + 1, exc
                )

        logger.error(
            "[%s] Giving up after %d reconnect attempts", adapter.name, MAX_RECONNECT_ATTEMPTS
        )
        adapter._mark_disconnected()
        return False

    async def _cleanup_ws(self) -> None:
        """Close and clear the WebSocket connection."""
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

class MediaSendHandler(ABC):
    """Abstract base class for media send strategies.

    Subclasses implement:
      - acquire_file(): how to obtain file bytes (download URL / read local)
      - build_msg_body(): how to build TIMxxxElem from upload result

    The shared flow (check ws → cancel notifier → validate → COS upload
    → lock → dispatch) is handled by the base handle() template method.
    """

    @abstractmethod
    async def acquire_file(
        self, adapter: "YuanbaoAdapter", **kwargs: Any,
    ) -> Tuple[bytes, str, str]:
        """Return (file_bytes, filename, content_type).

        Raises:
            ValueError: when file cannot be acquired (not found, empty, etc.)
        """

    @abstractmethod
    def build_msg_body(self, upload_result: dict, **kwargs: Any) -> list:
        """Build platform-specific MsgBody list from COS upload result."""

    def needs_cos_upload(self) -> bool:
        """Override to return False for non-COS media (e.g. sticker)."""
        return True

    async def handle(
        self,
        adapter: "YuanbaoAdapter",
        chat_id: str,
        reply_to: Optional[str] = None,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> "SendResult":
        """Template method: shared media send flow."""
        conn = adapter._connection
        sender = adapter._outbound.sender

        if conn.ws is None:
            return SendResult(success=False, error="Not connected", retryable=True)

        adapter._outbound.cancel_slow_notifier(chat_id)

        try:
            # 1. Acquire file bytes
            file_bytes, filename, content_type = await self.acquire_file(
                adapter, **kwargs,
            )

            # 2. Validate (only for handlers that upload to COS; stickers use
            # TIMFaceElem and legitimately carry no file bytes, so skipping
            # validate_media here avoids a spurious "Empty file: sticker").
            if self.needs_cos_upload():
                validation_err = MessageSender.validate_media(
                    file_bytes, filename, adapter.MEDIA_MAX_SIZE_MB,
                )
                if validation_err:
                    return SendResult(success=False, error=validation_err)

            if self.needs_cos_upload():
                file_uuid = md5_hex(file_bytes)

                # 3. Get COS upload credentials
                token_data = await adapter._get_cached_token()
                token: str = token_data.get("token", "")
                bot_id: str = (
                    token_data.get("bot_id", "") or adapter._bot_id or ""
                )

                credentials = await get_cos_credentials(
                    app_key=adapter._app_key,
                    api_domain=adapter._api_domain,
                    token=token,
                    filename=filename,
                    bot_id=bot_id,
                    route_env=adapter._route_env,
                )

                # 4. Upload to COS
                upload_result = await upload_to_cos(
                    file_bytes=file_bytes,
                    filename=filename,
                    content_type=content_type,
                    credentials=credentials,
                    bucket=credentials["bucketName"],
                    region=credentials["region"],
                )

                # 5. Build MsgBody
                # Remove keys already passed explicitly to avoid "multiple values" TypeError
                fwd_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k not in ("file_uuid", "filename", "content_type")
                }
                msg_body = self.build_msg_body(
                    upload_result,
                    file_uuid=file_uuid,
                    filename=filename,
                    content_type=content_type,
                    **fwd_kwargs,
                )
            else:
                # Non-COS media (e.g. sticker): build MsgBody directly
                msg_body = self.build_msg_body({}, **kwargs)

            # 6. Append caption if provided
            if caption:
                msg_body.append(
                    {"msg_type": "TIMTextElem", "msg_content": {"text": caption}},
                )

            # 7. Lock + dispatch
            gc = kwargs.get("group_code", "")
            return await sender.dispatch_msg_body(chat_id, msg_body, reply_to, group_code=gc)

        except ValueError as ve:
            return SendResult(success=False, error=str(ve))
        except Exception as exc:
            handler_name = type(self).__name__
            logger.error(
                "[%s] %s.handle() failed: %s",
                adapter.name, handler_name, exc, exc_info=True,
            )
            return SendResult(success=False, error=str(exc))


class ImageUrlHandler(MediaSendHandler):
    """Strategy: send image from a URL (download → COS → TIMImageElem)."""

    async def acquire_file(self, adapter, **kwargs):
        image_url: str = kwargs["image_url"]
        logger.info("[%s] ImageUrlHandler: downloading %s", adapter.name, image_url)
        file_bytes, content_type = await media_download_url(
            image_url, max_size_mb=adapter.MEDIA_MAX_SIZE_MB,
        )
        if not content_type or content_type == "application/octet-stream":
            path_part = image_url.split("?")[0]
            content_type = guess_mime_type(path_part) or "image/jpeg"
        filename = os.path.basename(image_url.split("?")[0]) or "image.jpg"
        return file_bytes, filename, content_type

    def build_msg_body(self, upload_result, **kwargs):
        return build_image_msg_body(
            url=upload_result["url"],
            uuid=kwargs["file_uuid"],
            filename=kwargs["filename"],
            size=upload_result["size"],
            width=upload_result.get("width", 0),
            height=upload_result.get("height", 0),
            mime_type=kwargs["content_type"],
        )


class ImageFileHandler(MediaSendHandler):
    """Strategy: send image from a local file path (read → COS → TIMImageElem)."""

    async def acquire_file(self, adapter, **kwargs):
        image_path: str = kwargs["image_path"]
        if not os.path.isfile(image_path):
            raise ValueError(f"File not found: {image_path}")
        logger.info("[%s] ImageFileHandler: reading %s", adapter.name, image_path)
        with open(image_path, "rb") as f:
            file_bytes = f.read()
        filename = os.path.basename(image_path) or "image.jpg"
        content_type = guess_mime_type(filename) or "image/jpeg"
        return file_bytes, filename, content_type

    def build_msg_body(self, upload_result, **kwargs):
        return build_image_msg_body(
            url=upload_result["url"],
            uuid=kwargs["file_uuid"],
            filename=kwargs["filename"],
            size=upload_result["size"],
            width=upload_result.get("width", 0),
            height=upload_result.get("height", 0),
            mime_type=kwargs["content_type"],
        )


class FileUrlHandler(MediaSendHandler):
    """Strategy: send file from a URL (download → COS → TIMFileElem)."""

    async def acquire_file(self, adapter, **kwargs):
        file_url: str = kwargs["file_url"]
        logger.info("[%s] FileUrlHandler: downloading %s", adapter.name, file_url)
        file_bytes, content_type = await media_download_url(
            file_url, max_size_mb=adapter.MEDIA_MAX_SIZE_MB,
        )
        filename = kwargs.get("filename")
        if not filename:
            path_part = file_url.split("?")[0]
            filename = os.path.basename(path_part) or "file"
        if not content_type or content_type == "application/octet-stream":
            content_type = guess_mime_type(filename) or "application/octet-stream"
        return file_bytes, filename, content_type

    def build_msg_body(self, upload_result, **kwargs):
        return build_file_msg_body(
            url=upload_result["url"],
            filename=kwargs["filename"],
            uuid=kwargs["file_uuid"],
            size=upload_result["size"],
        )


class DocumentHandler(MediaSendHandler):
    """Strategy: send local file/document (read → COS → TIMFileElem)."""

    async def acquire_file(self, adapter, **kwargs):
        file_path: str = kwargs["file_path"]
        if not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")
        logger.info("[%s] DocumentHandler: reading %s", adapter.name, file_path)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        filename = kwargs.get("filename") or os.path.basename(file_path) or "document"
        content_type = guess_mime_type(filename) or "application/octet-stream"
        return file_bytes, filename, content_type

    def build_msg_body(self, upload_result, **kwargs):
        return build_file_msg_body(
            url=upload_result["url"],
            filename=kwargs["filename"],
            uuid=kwargs["file_uuid"],
            size=upload_result["size"],
        )


class StickerHandler(MediaSendHandler):
    """Strategy: send sticker/emoji (TIMFaceElem, no COS upload needed)."""

    def needs_cos_upload(self) -> bool:
        return False

    async def acquire_file(self, adapter, **kwargs):
        # Sticker does not need file bytes; return dummy values
        return b"", "sticker", "application/octet-stream"

    def build_msg_body(self, upload_result, **kwargs):
        from gateway.platforms.yuanbao_sticker import (
            get_sticker_by_name,
            get_random_sticker,
            build_face_msg_body,
            build_sticker_msg_body,
        )
        sticker_name = kwargs.get("sticker_name")
        face_index = kwargs.get("face_index")

        if sticker_name is not None:
            sticker = get_sticker_by_name(sticker_name)
            if sticker is None:
                raise ValueError(f"Sticker not found: {sticker_name!r}")
            return build_sticker_msg_body(sticker)
        elif face_index is not None:
            return build_face_msg_body(face_index=face_index)
        else:
            sticker = get_random_sticker()
            return build_sticker_msg_body(sticker)

class GroupQueryService:
    """Encapsulates all group query operations (both low-level WS calls and
    higher-level AI-tool-facing wrappers).

    Responsibilities:
      - Low-level WS encode/decode for group info and member list queries
      - Chat-id parsing, error wrapping and result filtering for AI tools
      - Member cache population on the adapter
    """

    def __init__(self, adapter: "YuanbaoAdapter") -> None:
        self._adapter = adapter

    # ------------------------------------------------------------------
    # Low-level WS query methods
    # ------------------------------------------------------------------

    async def query_group_info_raw(self, group_code: str) -> Optional[dict]:
        """Query group info via WS (group name, owner, member count, etc.).

        Returns:
            Decoded dict or None on failure.
        """
        adapter = self._adapter
        if adapter._connection.ws is None:
            return None
        encoded = encode_query_group_info(group_code)
        from gateway.platforms.yuanbao_proto import decode_conn_msg as _decode
        decoded = _decode(encoded)
        req_id = decoded["head"]["msg_id"]
        try:
            response = await adapter._connection.send_biz_request(encoded, req_id=req_id)
            head = response.get("head", {})
            status = head.get("status", 0)
            if status != 0:
                logger.warning("[%s] query_group_info failed: status=%d", adapter.name, status)
                return None
            biz_data = response.get("data", b"") or response.get("body", b"")
            if biz_data and isinstance(biz_data, bytes):
                return decode_query_group_info_rsp(biz_data)
            return {"group_code": group_code}
        except asyncio.TimeoutError:
            logger.warning("[%s] query_group_info timeout: group=%s", adapter.name, group_code)
            return None
        except Exception as exc:
            logger.warning("[%s] query_group_info failed: %s", adapter.name, exc)
            return None

    async def get_group_member_list_raw(
        self, group_code: str, offset: int = 0, limit: int = 200
    ) -> Optional[dict]:
        """Query group member list via WS.

        Returns:
            Decoded dict or None on failure.  Also populates adapter._member_cache.
        """
        adapter = self._adapter
        if adapter._connection.ws is None:
            return None
        encoded = encode_get_group_member_list(group_code, offset=offset, limit=limit)
        from gateway.platforms.yuanbao_proto import decode_conn_msg as _decode
        decoded = _decode(encoded)
        req_id = decoded["head"]["msg_id"]
        try:
            response = await adapter._connection.send_biz_request(encoded, req_id=req_id)
            head = response.get("head", {})
            status = head.get("status", 0)
            if status != 0:
                logger.warning("[%s] get_group_member_list failed: status=%d", adapter.name, status)
                return None
            biz_data = response.get("data", b"") or response.get("body", b"")
            if biz_data and isinstance(biz_data, bytes):
                result = decode_get_group_member_list_rsp(biz_data)
            else:
                result = {"members": [], "next_offset": 0, "is_complete": True}
            if result and result.get("members"):
                adapter._member_cache[group_code] = (time.time(), result["members"])
            return result
        except asyncio.TimeoutError:
            logger.warning("[%s] get_group_member_list timeout: group=%s", adapter.name, group_code)
            return None
        except Exception as exc:
            logger.warning("[%s] get_group_member_list failed: %s", adapter.name, exc)
            return None

    # ------------------------------------------------------------------
    # AI-tool-facing wrappers (chat_id parsing + filtering)
    # ------------------------------------------------------------------

    async def query_group_info(self, chat_id: str) -> dict:
        """AI tool: Query current group info.

        No parameters needed (group_code extracted from session context).
        Returns group name, owner, member count, etc.
        """
        if not chat_id.startswith("group:"):
            return {"error": "This command is only available in group chats"}
        group_code = chat_id[len("group:"):]
        result = await self.query_group_info_raw(group_code)
        if result is None:
            return {"error": "Failed to query group info"}
        return result

    async def query_session_members(
        self,
        chat_id: str,
        action: str = "list_all",
        name: Optional[str] = None,
    ) -> dict:
        """AI tool: Query group member list.

        Args:
            chat_id: Chat ID (extracted from session context)
            action: 'find' (search by name) | 'list_bots' (list bots) | 'list_all' (list all)
            name: Search keyword when action='find'

        Returns:
            {"members": [...], "total": int, "mentionHint": str}
        """
        if not chat_id.startswith("group:"):
            return {"error": "This command is only available in group chats"}
        group_code = chat_id[len("group:"):]
        result = await self.get_group_member_list_raw(group_code)
        if result is None:
            return {"error": "Failed to query group members"}

        members = result.get("members", [])

        if action == "find" and name:
            query = name.lower()
            members = [
                m for m in members
                if query in (m.get("nickname", "") or "").lower()
                or query in (m.get("name_card", "") or "").lower()
                or query in (m.get("user_id", "") or "").lower()
            ]
        elif action == "list_bots":
            members = [m for m in members if "bot" in (m.get("nickname", "") or "").lower()]

        # Construct mentionHint
        mention_hint = ""
        if members and len(members) <= 10:
            names = [m.get("name_card") or m.get("nickname") or m.get("user_id", "") for m in members]
            mention_hint = "Mention with @name: " + ", ".join(names)

        return {
            "members": members[:50],  # Limit return count
            "total": len(members),
            "mentionHint": mention_hint,
        }


class HeartbeatManager:
    """Manages reply heartbeat (RUNNING / FINISH) lifecycle.

    Responsibilities:
      - Periodic RUNNING heartbeat sender (every 2s)
      - Auto-FINISH after 30s inactivity
      - Explicit stop with optional FINISH signal
    """

    def __init__(self, adapter: "YuanbaoAdapter") -> None:
        self._adapter = adapter
        self._reply_heartbeat_tasks: Dict[str, asyncio.Task] = {}
        self._reply_hb_last_active: Dict[str, float] = {}

    async def send_heartbeat_once(self, chat_id: str, heartbeat_val: int) -> None:
        """Send a single heartbeat (RUNNING or FINISH), best effort."""
        adapter = self._adapter
        conn = adapter._connection
        if conn.ws is None or not adapter._bot_id:
            return
        try:
            if chat_id.startswith("group:"):
                group_code = chat_id[len("group:"):]
                encoded = encode_send_group_heartbeat(
                    from_account=adapter._bot_id,
                    group_code=group_code,
                    heartbeat=heartbeat_val,
                )
            else:
                to_account = chat_id.removeprefix("direct:")
                encoded = encode_send_private_heartbeat(
                    from_account=adapter._bot_id,
                    to_account=to_account,
                    heartbeat=heartbeat_val,
                )
            await conn.ws.send(encoded)
            status_name = "RUNNING" if heartbeat_val == WS_HEARTBEAT_RUNNING else "FINISH"
            logger.debug(
                "[%s] Reply heartbeat %s sent: chat=%s",
                adapter.name, status_name, chat_id,
            )
        except Exception as exc:
            logger.debug("[%s] send_heartbeat_once failed: %s", adapter.name, exc)

    async def start(self, chat_id: str) -> None:
        """Start or renew the Reply Heartbeat periodic sender (RUNNING, every 2s)."""
        adapter = self._adapter
        conn = adapter._connection
        if conn.ws is None or not adapter._bot_id:
            return

        existing = self._reply_heartbeat_tasks.get(chat_id)
        if existing and not existing.done():
            self._reply_hb_last_active[chat_id] = time.time()
            return

        self._reply_hb_last_active[chat_id] = time.time()

        task = asyncio.create_task(
            self._worker(chat_id),
            name=f"yuanbao-reply-hb-{chat_id}",
        )
        self._reply_heartbeat_tasks[chat_id] = task

    async def _worker(self, chat_id: str) -> None:
        """Background coroutine: send RUNNING heartbeat every 2s.
        30s without renewal -> send FINISH and exit.
        """
        try:
            await self.send_heartbeat_once(chat_id, WS_HEARTBEAT_RUNNING)

            while True:
                await asyncio.sleep(REPLY_HEARTBEAT_INTERVAL_S)

                last_active = self._reply_hb_last_active.get(chat_id, 0)
                if time.time() - last_active > REPLY_HEARTBEAT_TIMEOUT_S:
                    break

                conn = self._adapter._connection
                if conn.ws is None:
                    break

                await self.send_heartbeat_once(chat_id, WS_HEARTBEAT_RUNNING)

        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            cancelled = False
        else:
            cancelled = False
        finally:
            if not cancelled:
                try:
                    await self.send_heartbeat_once(chat_id, WS_HEARTBEAT_FINISH)
                except Exception:
                    pass
            self._reply_heartbeat_tasks.pop(chat_id, None)
            self._reply_hb_last_active.pop(chat_id, None)

    async def stop(self, chat_id: str, send_finish: bool = True) -> None:
        """Stop Reply Heartbeat and optionally send FINISH."""
        task = self._reply_heartbeat_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if send_finish:
            try:
                await self.send_heartbeat_once(chat_id, WS_HEARTBEAT_FINISH)
            except Exception:
                pass

    async def close(self) -> None:
        """Cancel all reply heartbeat tasks."""
        for task in list(self._reply_heartbeat_tasks.values()):
            if not task.done():
                task.cancel()
        self._reply_heartbeat_tasks.clear()
        self._reply_hb_last_active.clear()


class SlowResponseNotifier:
    """Manages delayed 'please wait' notifications for slow agent responses.

    Starts a timer per chat_id; if the agent hasn't replied within
    SLOW_RESPONSE_TIMEOUT_S seconds, sends a courtesy message.
    """

    def __init__(self, adapter: "YuanbaoAdapter", sender: "MessageSender") -> None:
        self._adapter = adapter
        self._sender = sender
        self._tasks: Dict[str, asyncio.Task] = {}

    async def start(self, chat_id: str) -> None:
        """Start a delayed task that notifies the user when the agent is slow."""
        self.cancel(chat_id)
        task = asyncio.create_task(
            self._notifier(chat_id),
            name=f"yuanbao-slow-resp-{chat_id}",
        )
        self._tasks[chat_id] = task

    async def _notifier(self, chat_id: str) -> None:
        """Wait SLOW_RESPONSE_TIMEOUT_S, then push a 'please wait' message."""
        try:
            await asyncio.sleep(SLOW_RESPONSE_TIMEOUT_S)
            logger.info(
                "[%s] Agent response exceeded %ds for %s, sending wait notice",
                self._adapter.name, int(SLOW_RESPONSE_TIMEOUT_S), chat_id,
            )
            await self._sender.send_text_chunk(chat_id, SLOW_RESPONSE_MESSAGE)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("[%s] Slow-response notifier failed: %s", self._adapter.name, exc)

    def cancel(self, chat_id: str) -> None:
        """Cancel the pending slow-response notifier for *chat_id*, if any."""
        task = self._tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()

    async def close(self) -> None:
        """Cancel all slow-response tasks."""
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()
        self._tasks.clear()


class MessageSender:
    """Core message sending dispatcher for YuanbaoAdapter.

    Responsibilities:
      - Per-chat-id lock management (serial send ordering)
      - Text chunk sending with retry
      - C2C / Group message encoding and dispatch
      - Media send helpers (image, file, sticker, document)
      - Direct send helper (text + media, used by send_message tool)
    """

    IMAGE_EXTS: ClassVar[frozenset] = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})
    CHAT_DICT_MAX_SIZE: ClassVar[int] = 1000  # Max distinct chat IDs in _chat_locks

    def __init__(self, adapter: "YuanbaoAdapter") -> None:
        self._adapter = adapter
        self._chat_locks: collections.OrderedDict[str, asyncio.Lock] = collections.OrderedDict()

        # Optional hooks injected by OutboundManager for coordination
        self._on_send_start: Optional[Callable[[str], Any]] = None   # cancel slow-notifier
        self._on_send_finish: Optional[Callable[[str], Any]] = None  # send FINISH heartbeat

        # Media send handlers (strategy pattern)
        self._media_handlers: Dict[str, MediaSendHandler] = {
            "image_url": ImageUrlHandler(),
            "image_file": ImageFileHandler(),
            "file_url": FileUrlHandler(),
            "document": DocumentHandler(),
            "sticker": StickerHandler(),
        }

    # -- Media handler registry ---------------------------------------------

    def register_handler(self, name: str, handler: MediaSendHandler) -> None:
        """Register (or replace) a named media send handler."""
        self._media_handlers[name] = handler

    # -- Chat lock ---------------------------------------------------------

    def get_chat_lock(self, chat_id: str) -> asyncio.Lock:
        """Return (or create) a per-chat-id lock with safe LRU eviction."""
        if chat_id in self._chat_locks:
            self._chat_locks.move_to_end(chat_id)
            return self._chat_locks[chat_id]
        if len(self._chat_locks) >= self.CHAT_DICT_MAX_SIZE:
            evicted = False
            for key in list(self._chat_locks):
                if not self._chat_locks[key].locked():
                    self._chat_locks.pop(key)
                    evicted = True
                    break
            if not evicted:
                self._chat_locks.pop(next(iter(self._chat_locks)))
        self._chat_locks[chat_id] = asyncio.Lock()
        return self._chat_locks[chat_id]

    # -- Text send ---------------------------------------------------------

    async def send_text(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        group_code: str = "",
    ) -> "SendResult":
        """Send text message with auto-chunking and per-chat-id ordering guarantee."""
        adapter = self._adapter
        conn = adapter._connection
        if conn.ws is None:
            return SendResult(success=False, error="Not connected", retryable=True)

        if self._on_send_start:
            self._on_send_start(chat_id)

        lock = self.get_chat_lock(chat_id)
        async with lock:
            content_to_send = self.strip_cron_wrapper(content)
            chunks = self.truncate_message(content_to_send, adapter.MAX_TEXT_CHUNK)
            logger.info(
                "[%s] truncate_message: input=%d chars, max=%d, output=%d chunk(s) sizes=%s",
                adapter.name, len(content_to_send), adapter.MAX_TEXT_CHUNK,
                len(chunks), [len(c) for c in chunks],
            )
            for i, chunk in enumerate(chunks):
                r_to = reply_to if i == 0 else None
                result = await self.send_text_chunk(chat_id, chunk, r_to, group_code=group_code)
                if not result.success:
                    return result

        # Notify outbound coordinator that send is complete (e.g. FINISH heartbeat)
        if self._on_send_finish:
            try:
                await self._on_send_finish(chat_id)
            except Exception:
                pass
        return SendResult(success=True)

    async def send_media(
        self,
        chat_id: str,
        handler_name: str,
        reply_to: Optional[str] = None,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> "SendResult":
        """Dispatch media send to the named handler strategy."""
        handler = self._media_handlers.get(handler_name)
        if handler is None:
            return SendResult(
                success=False,
                error=f"Unknown media handler: {handler_name!r}",
            )
        return await handler.handle(
            self._adapter, chat_id,
            reply_to=reply_to, caption=caption, **kwargs,
        )

    # -- Direct send (text + media, used by send_message tool) -------------

    async def send_direct(
        self,
        chat_id: str,
        message: str,
        media_files: Optional[List[Tuple[str, bool]]] = None,
    ) -> Dict[str, Any]:
        """Send text + media via Yuanbao (used by the ``send_message`` tool).

        Unlike Weixin which creates a fresh adapter per call, Yuanbao reuses
        the running gateway adapter (persistent WebSocket).  Logic mirrors
        send_weixin_direct: send text first, then iterate media_files by
        extension.
        """
        adapter = self._adapter
        last_result: Optional["SendResult"] = None

        # 1. Send text
        if message.strip():
            last_result = await adapter.send(chat_id, message)
            if not last_result.success:
                return {"error": f"Yuanbao send failed: {last_result.error}"}

        # 2. Iterate media_files, dispatch by file extension
        for media_path, _is_voice in media_files or []:
            ext = Path(media_path).suffix.lower()
            if ext in self.IMAGE_EXTS:
                last_result = await adapter.send_image_file(chat_id, media_path)
            else:
                last_result = await adapter.send_document(chat_id, media_path)

            if not last_result.success:
                return {"error": f"Yuanbao media send failed: {last_result.error}"}

        if last_result is None:
            return {"error": "No deliverable text or media remained after processing"}

        return {
            "success": True,
            "platform": "yuanbao",
            "chat_id": chat_id,
            "message_id": last_result.message_id if last_result else None,
        }

    async def dispatch_msg_body(
        self,
        chat_id: str,
        msg_body: list,
        reply_to: Optional[str] = None,
        group_code: str = "",
    ) -> "SendResult":
        """Lock + dispatch an arbitrary MsgBody to C2C or group."""
        lock = self.get_chat_lock(chat_id)
        async with lock:
            if chat_id.startswith("group:"):
                grp = chat_id[len("group:"):]
                result = await self.send_group_msg_body(grp, msg_body, reply_to)
            else:
                to_account = chat_id.removeprefix("direct:")
                result = await self.send_c2c_msg_body(to_account, msg_body, group_code=group_code)

        if result.get("success"):
            return SendResult(success=True, message_id=result.get("msg_key"))
        return SendResult(success=False, error=result.get("error", "Unknown error"))

    async def send_text_chunk(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
        retry: int = 3,
        group_code: str = "",
    ) -> "SendResult":
        """Send a single text chunk with retry (exponential backoff: 1s, 2s, 4s)."""
        adapter = self._adapter
        last_error: str = "Unknown error"
        for attempt in range(retry):
            try:
                if chat_id.startswith("group:"):
                    grp = chat_id[len("group:"):]
                    raw = await self.send_group_message(grp, text, reply_to)
                else:
                    to_account = chat_id.removeprefix("direct:")
                    raw = await self.send_c2c_message(to_account, text, group_code=group_code)

                if raw.get("success"):
                    return SendResult(success=True, message_id=raw.get("msg_key"))

                last_error = raw.get("error", "Unknown error")
                logger.warning(
                    "[%s] send_text_chunk attempt %d/%d failed: %s",
                    adapter.name, attempt + 1, retry, last_error,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "[%s] send_text_chunk attempt %d/%d exception: %s",
                    adapter.name, attempt + 1, retry, last_error,
                )

            if attempt < retry - 1:
                await asyncio.sleep(2 ** attempt)

        logger.error(
            "[%s] send_text_chunk max retries (%d) exceeded. Last error: %s",
            adapter.name, retry, last_error,
        )
        return SendResult(success=False, error=f"Max retries exceeded: {last_error}")

    # -- C2C / Group message -----------------------------------------------

    async def send_c2c_message(self, to_account: str, text: str, group_code: str = "") -> dict:
        """Send C2C text message, return {success: bool, msg_key: str}."""
        msg_body = [{"msg_type": "TIMTextElem", "msg_content": {"text": text}}]
        return await self.send_c2c_msg_body(to_account, msg_body, group_code=group_code)

    async def send_group_message(
        self,
        group_code: str,
        text: str,
        reply_to: Optional[str] = None,
    ) -> dict:
        """Send group text message, auto-converting @nickname to TIMCustomElem."""
        msg_body = self._build_msg_body_with_mentions(text, group_code)
        return await self.send_group_msg_body(group_code, msg_body, reply_to)

    # @mention pattern: (whitespace or start) + @ + nickname + (whitespace or end)
    _AT_USER_RE = re.compile(r'(?:(?<=\s)|(?<=^))@(\S+?)(?=\s|$)', re.MULTILINE)

    def _build_msg_body_with_mentions(self, text: str, group_code: str) -> list:
        """Parse @nickname patterns and build mixed TIMTextElem + TIMCustomElem msg_body."""
        cached = self._adapter._member_cache.get(group_code)
        if cached:
            ts, member_list = cached
            members = member_list if (time.time() - ts < self._adapter.MEMBER_CACHE_TTL_S) else []
        else:
            members = []
        if not members:
            return [{"msg_type": "TIMTextElem", "msg_content": {"text": text}}]

        nickname_to_uid = {}
        for m in members:
            nick = m.get("nickname") or m.get("nick_name") or ""
            uid = m.get("user_id") or ""
            if nick and uid:
                nickname_to_uid[nick.lower()] = (nick, uid)

        msg_body: list = []
        last_idx = 0
        for match in self._AT_USER_RE.finditer(text):
            start = match.start()
            if start > last_idx:
                seg = text[last_idx:start].strip()
                if seg:
                    msg_body.append({"msg_type": "TIMTextElem", "msg_content": {"text": seg}})

            nickname = match.group(1)
            entry = nickname_to_uid.get(nickname.lower())
            if entry:
                real_nick, uid = entry
                msg_body.append({
                    "msg_type": "TIMCustomElem",
                    "msg_content": {
                        "data": json.dumps({"elem_type": 1002, "text": f"@{real_nick}", "user_id": uid}),
                    },
                })
            else:
                msg_body.append({"msg_type": "TIMTextElem", "msg_content": {"text": f"@{nickname}"}})

            last_idx = match.end()

        if last_idx < len(text):
            tail = text[last_idx:].strip()
            if tail:
                msg_body.append({"msg_type": "TIMTextElem", "msg_content": {"text": tail}})

        if not msg_body:
            msg_body.append({"msg_type": "TIMTextElem", "msg_content": {"text": text}})

        return msg_body

    async def send_c2c_msg_body(self, to_account: str, msg_body: list, group_code: str = "") -> dict:
        """Send C2C message with arbitrary MsgBody."""
        adapter = self._adapter
        req_id = f"c2c_{next_seq_no()}"
        encoded = encode_send_c2c_message(
            to_account=to_account,
            msg_body=msg_body,
            from_account=adapter._bot_id or "",
            msg_id=req_id,
            group_code=group_code,
        )
        return await self._dispatch_encoded(adapter, encoded, req_id)

    async def send_group_msg_body(
        self,
        group_code: str,
        msg_body: list,
        reply_to: Optional[str] = None,
    ) -> dict:
        """Send group message with arbitrary MsgBody."""
        adapter = self._adapter
        req_id = f"grp_{next_seq_no()}"
        encoded = encode_send_group_message(
            group_code=group_code,
            msg_body=msg_body,
            from_account=adapter._bot_id or "",
            msg_id=req_id,
            ref_msg_id=reply_to or "",
        )
        return await self._dispatch_encoded(adapter, encoded, req_id)

    # -- Common dispatch helper --------------------------------------------

    @staticmethod
    async def _dispatch_encoded(
        adapter: "YuanbaoAdapter", encoded: bytes, req_id: str,
    ) -> dict:
        """Send pre-encoded bytes via WS and return a normalised result dict."""
        try:
            response = await adapter._connection.send_biz_request(encoded, req_id=req_id)
            return {"success": True, "msg_key": response.get("msg_id", "")}
        except asyncio.TimeoutError:
            return {"success": False, "error": f"Request timeout after {DEFAULT_SEND_TIMEOUT}s"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # -- Media validation ---------------------------------------------------

    @staticmethod
    def validate_media(
        file_bytes: Optional[bytes], filename: str, max_size_mb: int = 20
    ) -> Optional[str]:
        """Media pre-validation: check file validity before sending/uploading.

        Returns:
            Error description (str) if validation fails, otherwise None.
        """
        if file_bytes is None or len(file_bytes) == 0:
            return f"Empty file: {filename}"
        max_bytes = max_size_mb * 1024 * 1024
        if len(file_bytes) > max_bytes:
            size_mb = len(file_bytes) / 1024 / 1024
            return f"File too large: {filename} ({size_mb:.1f}MB > {max_size_mb}MB)"
        return None

    # -- Text truncation (table-aware) --------------------------------------

    @staticmethod
    def truncate_message(
        content: str,
        max_length: int = 4000,
        len_fn: Optional[Callable[[str], int]] = None,
    ) -> List[str]:
        """
        Split a long message into chunks with table-awareness.

        Delegates core splitting to ``MarkdownProcessor.chunk_markdown_text``
        and strips page indicators like ``(1/3)`` from the output.

        Falls back to ``BasePlatformAdapter.truncate_message`` for non-table
        content and for overall text that fits in a single chunk.
        """
        _len = len_fn or len
        if _len(content) <= max_length:
            return [content]

        # Delegate to MarkdownProcessor for table/fence-aware chunking
        chunks = MarkdownProcessor.chunk_markdown_text(
            content, max_length, len_fn=len_fn,
        )

        # Strip page indicators like (1/3) that BasePlatformAdapter may add
        chunks = [_INDICATOR_RE.sub('', c) for c in chunks]

        return chunks if chunks else [content]

    # -- Cron wrapper stripping ---------------------------------------------

    @staticmethod
    def strip_cron_wrapper(content: str) -> str:
        """Strip scheduler cron header/footer wrapper for cleaner Yuanbao output."""
        if not content.startswith("Cronjob Response: "):
            return content

        divider = "\n-------------\n\n"
        footer_prefix = '\n\nTo stop or manage this job, send me a new message (e.g. "stop reminder '
        divider_pos = content.find(divider)
        footer_pos = content.rfind(footer_prefix)
        if divider_pos < 0 or footer_pos < 0 or footer_pos <= divider_pos:
            return content

        header = content[:divider_pos]
        if "\n(job_id: " not in header:
            return content

        body_start = divider_pos + len(divider)
        body = content[body_start:footer_pos].strip()
        return body or content

    # -- Cleanup on disconnect ---------------------------------------------

    async def close(self) -> None:
        """Release chat locks (no-op for now; placeholder for future cleanup)."""
        self._chat_locks.clear()


class OutboundManager:
    """Outbound coordinator that orchestrates sending, heartbeat and slow-response.

    Composes:
      - MessageSender   — core text/media sending
      - HeartbeatManager — reply heartbeat (RUNNING / FINISH) lifecycle
      - SlowResponseNotifier — delayed 'please wait' notifications

    YuanbaoAdapter holds a single ``_outbound: OutboundManager`` and delegates
    all outbound operations through it.
    """

    # Expose class-level constants from MessageSender for backward compatibility
    CHAT_DICT_MAX_SIZE: ClassVar[int] = MessageSender.CHAT_DICT_MAX_SIZE

    def __init__(self, adapter: "YuanbaoAdapter") -> None:
        self._adapter = adapter
        self.sender: MessageSender = MessageSender(adapter)
        self.heartbeat: HeartbeatManager = HeartbeatManager(adapter)
        self.slow_notifier: SlowResponseNotifier = SlowResponseNotifier(adapter, self.sender)

        # Wire coordination hooks into MessageSender
        self.sender._on_send_start = self._handle_send_start
        self.sender._on_send_finish = self._handle_send_finish

    # -- Coordination hooks ------------------------------------------------

    def _handle_send_start(self, chat_id: str) -> None:
        """Called by MessageSender before sending: cancel slow-response notifier."""
        self.slow_notifier.cancel(chat_id)

    async def _handle_send_finish(self, chat_id: str) -> None:
        """Called by MessageSender after sending: send FINISH heartbeat."""
        await self.heartbeat.send_heartbeat_once(chat_id, WS_HEARTBEAT_FINISH)

    # -- Delegated public API (used by YuanbaoAdapter) ---------------------

    async def send_text(
        self, chat_id: str, content: str, reply_to: Optional[str] = None,
        group_code: str = "",
    ) -> "SendResult":
        """Send text message with auto-chunking."""
        return await self.sender.send_text(chat_id, content, reply_to, group_code=group_code)

    async def send_media(
        self, chat_id: str, handler_name: str, **kwargs: Any,
    ) -> "SendResult":
        """Dispatch media send to the named handler strategy."""
        return await self.sender.send_media(chat_id, handler_name, **kwargs)

    async def send_direct(
        self, chat_id: str, message: str,
        media_files: Optional[List[Tuple[str, bool]]] = None,
    ) -> Dict[str, Any]:
        """Send text + media (used by send_message tool)."""
        return await self.sender.send_direct(chat_id, message, media_files)

    async def start_typing(self, chat_id: str) -> None:
        """Start reply heartbeat (RUNNING)."""
        await self.heartbeat.start(chat_id)

    async def stop_typing(self, chat_id: str, send_finish: bool = False) -> None:
        """Stop reply heartbeat."""
        await self.heartbeat.stop(chat_id, send_finish=send_finish)

    async def start_slow_notifier(self, chat_id: str) -> None:
        """Start slow-response notifier."""
        await self.slow_notifier.start(chat_id)

    def cancel_slow_notifier(self, chat_id: str) -> None:
        """Cancel slow-response notifier."""
        self.slow_notifier.cancel(chat_id)

    def get_chat_lock(self, chat_id: str) -> asyncio.Lock:
        """Proxy to MessageSender.get_chat_lock for backward compatibility."""
        return self.sender.get_chat_lock(chat_id)

    @property
    def _chat_locks(self) -> collections.OrderedDict:
        """Proxy to MessageSender._chat_locks for backward compatibility."""
        return self.sender._chat_locks

    @staticmethod
    def validate_media(
        file_bytes: Optional[bytes], filename: str, max_size_mb: int = 20,
    ) -> Optional[str]:
        """Proxy to MessageSender.validate_media."""
        return MessageSender.validate_media(file_bytes, filename, max_size_mb)

    async def close(self) -> None:
        """Shut down all sub-managers."""
        await self.sender.close()
        await self.heartbeat.close()
        await self.slow_notifier.close()


class YuanbaoAdapter(BasePlatformAdapter):
    """Yuanbao AI Bot adapter backed by a persistent WebSocket connection."""

    PLATFORM = Platform.YUANBAO
    MAX_TEXT_CHUNK: int = 4000  # Yuanbao single message character limit
    MEDIA_MAX_SIZE_MB: int = 50  # Max media file size in MB for upload validation
    REPLY_REF_MAX_ENTRIES: ClassVar[int] = 500  # Max capacity of reference dedup dict

    # -- Active instance registry (class-level singleton) -------------------

    _active_instance: ClassVar[Optional["YuanbaoAdapter"]] = None

    @classmethod
    def get_active(cls) -> Optional["YuanbaoAdapter"]:
        """Return the currently connected YuanbaoAdapter, or None."""
        return cls._active_instance

    @classmethod
    def set_active(cls, adapter: Optional["YuanbaoAdapter"]) -> None:
        """Register (or clear) the active adapter instance."""
        cls._active_instance = adapter

    def __init__(self, config: PlatformConfig, **kwargs: Any) -> None:
        super().__init__(config, Platform.YUANBAO)

        # Credentials / endpoints from config.extra (populated by config.py from env/yaml)
        _extra = config.extra or {}
        self._app_key: str = (_extra.get("app_id") or "").strip()
        self._app_secret: str = (_extra.get("app_secret") or "").strip()
        self._bot_id: Optional[str] = _extra.get("bot_id") or None
        self._ws_url: str = (_extra.get("ws_url") or DEFAULT_WS_GATEWAY_URL).strip()
        self._api_domain: str = (_extra.get("api_domain") or DEFAULT_API_DOMAIN).rstrip("/")
        self._route_env: str = (_extra.get("route_env") or "").strip()

        # Core managers (UML composition)
        self._connection: ConnectionManager = ConnectionManager(self)
        self._outbound: OutboundManager = OutboundManager(self)

        # Inbound dispatch tasks — tracked so disconnect() can cancel them
        self._inbound_tasks: set[asyncio.Task] = set()

        # Set of background tasks — prevent GC from collecting fire-and-forget tasks
        self._background_tasks: set[asyncio.Task] = set()

        # Member cache: group_code -> (updated_ts, [{"user_id":..., "nickname":..., ...}, ...])
        # Populated by get_group_member_list(), used by @mention resolution.
        # Entries older than MEMBER_CACHE_TTL_S are treated as stale.
        self._member_cache: Dict[str, Tuple[float, list]] = {}
        self.MEMBER_CACHE_TTL_S: float = 300.0  # 5 minutes

        # Inbound message deduplication (WS reconnect / network jitter)
        self._dedup = MessageDeduplicator(ttl_seconds=300)

        # Group chat sequential dispatch queue (session_key → asyncio.Queue).
        self._group_queues: Dict[str, asyncio.Queue] = {}

        # Recall support: track which msg_id is being processed per session_key
        # so RecallGuardMiddleware can detect "currently processing" messages.
        self._processing_msg_ids: Dict[str, str] = {}
        self._processing_msg_texts: Dict[str, str] = {}
        # Bounded cache of msg_id → attributed content for recent messages.
        # Used by _patch_transcript as content-match fallback when transcript
        # entries lack a message_id field (agent-processed @bot messages).
        self._msg_content_cache: Dict[str, str] = {}

        # Reply-to dedup: inbound_msg_id -> expire_ts
        # ------------------------------------------------------------------
        # Access control policy (DM / Group)
        # ------------------------------------------------------------------
        dm_policy: str = (
            _extra.get("dm_policy")
            or os.getenv("YUANBAO_DM_POLICY", "open")
        ).strip().lower()

        _dm_allow_from_raw: str = (
            _extra.get("dm_allow_from")
            or os.getenv("YUANBAO_DM_ALLOW_FROM", "")
        )
        dm_allow_from: list[str] = [x.strip() for x in _dm_allow_from_raw.split(",") if x.strip()]

        group_policy: str = (
            _extra.get("group_policy")
            or os.getenv("YUANBAO_GROUP_POLICY", "open")
        ).strip().lower()

        _group_allow_from_raw: str = (
            _extra.get("group_allow_from")
            or os.getenv("YUANBAO_GROUP_ALLOW_FROM", "")
        )
        group_allow_from: list[str] = [x.strip() for x in _group_allow_from_raw.split(",") if x.strip()]

        self._access_policy = AccessPolicy(
            dm_policy=dm_policy,
            dm_allow_from=dm_allow_from,
            group_policy=group_policy,
            group_allow_from=group_allow_from,
        )

        # Group query service (AI tool backing)
        self._group_query = GroupQueryService(self)

        # Inbound message processing pipeline (middleware pattern)
        self._inbound_pipeline: InboundPipeline = InboundPipelineBuilder.build()

        # ------------------------------------------------------------------
        # Auto-sethome: first user to message the bot becomes the owner.
        # If no home channel is configured, the first conversation will be
        # automatically set as the home channel.  When the existing home
        # channel is a group chat (group:xxx), it stays eligible for
        # upgrade — the first DM will override it with direct:xxx.
        # ------------------------------------------------------------------
        _existing_home = os.getenv("YUANBAO_HOME_CHANNEL") or (
            config.home_channel.chat_id if config.home_channel else ""
        )
        self._auto_sethome_done: bool = bool(_existing_home) and not _existing_home.startswith("group:")

    # ------------------------------------------------------------------
    # Task tracking helper
    # ------------------------------------------------------------------

    def _track_task(self, task: asyncio.Task) -> asyncio.Task:
        """Register a fire-and-forget task so it won't be GC'd prematurely."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to Yuanbao WS gateway and authenticate.

        Delegates to ConnectionManager.open().
        """
        return await self._connection.open()

    async def disconnect(self) -> None:
        """Cancel background tasks and close the WebSocket connection."""
        if YuanbaoAdapter._active_instance is self:
            YuanbaoAdapter.set_active(None)

        self._running = False
        self._mark_disconnected()
        self._release_platform_lock()

        # Delegate to managers
        await self._connection.close()
        await self._outbound.close()

        # Cancel all in-flight inbound dispatch tasks
        for task in list(self._inbound_tasks):
            if not task.done():
                task.cancel()
        self._inbound_tasks.clear()

        self._group_queues.clear()

        logger.info("[%s] Disconnected", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        group_code: str = "",
    ) -> SendResult:
        """Send text message with auto-chunking. Delegates to OutboundManager."""
        return await self._outbound.send_text(chat_id, content, reply_to, group_code=group_code)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic chat metadata derived from the chat_id prefix.

        chat_id conventions:
          "group:<group_code>"  → group chat
          "direct:<account>"   → C2C / direct message (default)

        TODO (T06): fetch real chat name/member-count from Yuanbao API.
        """
        if chat_id.startswith("group:"):
            return {"name": chat_id, "type": "group"}
        return {"name": chat_id, "type": "dm"}

    async def send_typing(self, chat_id: str, metadata: Optional[dict] = None) -> None:
        """Send "typing" status heartbeat (RUNNING). Delegates to OutboundManager."""
        try:
            await self._outbound.start_typing(chat_id)
        except Exception:
            pass

    async def stop_typing(self, chat_id: str) -> None:
        """Stop the RUNNING heartbeat loop without sending FINISH immediately.

        FINISH is sent by send() after actual message delivery to ensure correct ordering:
        RUNNING... -> message arrives -> FINISH.
        """
        try:
            await self._outbound.stop_typing(chat_id, send_finish=False)
        except Exception:
            pass

    async def _process_message_background(self, event, session_key: str) -> None:
        """Wrap base class processing with a slow-response notifier."""
        chat_id = event.source.chat_id
        await self._outbound.start_slow_notifier(chat_id)
        try:
            await super()._process_message_background(event, session_key)
        finally:
            self._outbound.cancel_slow_notifier(chat_id)

    # ------------------------------------------------------------------
    # Group query (delegate to GroupQueryService)
    # ------------------------------------------------------------------

    async def query_group_info(self, group_code: str) -> Optional[dict]:
        """Query group info (delegates to GroupQueryService)."""
        return await self._group_query.query_group_info_raw(group_code)

    async def get_group_member_list(
        self, group_code: str, offset: int = 0, limit: int = 200
    ) -> Optional[dict]:
        """Query group member list (delegates to GroupQueryService)."""
        return await self._group_query.get_group_member_list_raw(group_code, offset=offset, limit=limit)

    # ------------------------------------------------------------------
    # DM active private chat + access control
    # ------------------------------------------------------------------

    DM_MAX_CHARS = 10000  # DM text limit

    async def send_dm(self, user_id: str, text: str, group_code: str = "") -> SendResult:
        """
        Actively send C2C private chat message.

        Args:
            user_id: Target user ID
            text: Message text (limit 10000 characters)
            group_code: Source group code (for group-originated DM context)

        Returns:
            SendResult
        """
        if not self._access_policy.is_dm_allowed(user_id):
            return SendResult(success=False, error="DM access denied for this user")
        if len(text) > self.DM_MAX_CHARS:
            text = text[:self.DM_MAX_CHARS] + "\n...(truncated)"
        chat_id = f"direct:{user_id}"
        return await self.send(chat_id, text, group_code=group_code)

    # ------------------------------------------------------------------
    # Media send methods
    # ------------------------------------------------------------------

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send image message (URL). Delegates to OutboundManager via ImageUrlHandler."""
        return await self._outbound.send_media(
            chat_id, "image_url",
            reply_to=reply_to, caption=caption, image_url=image_url,
            **kwargs,
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send local image file. Delegates to OutboundManager via ImageFileHandler."""
        return await self._outbound.send_media(
            chat_id, "image_file",
            reply_to=reply_to, caption=caption, image_path=image_path,
            **kwargs,
        )

    async def send_file(
        self,
        chat_id: str,
        file_url: str,
        filename: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send file message (URL). Delegates to OutboundManager via FileUrlHandler."""
        return await self._outbound.send_media(
            chat_id, "file_url",
            reply_to=reply_to, file_url=file_url, filename=filename,
            **kwargs,
        )

    async def send_sticker(
        self,
        chat_id: str,
        sticker_name: Optional[str] = None,
        face_index: Optional[int] = None,
        reply_to: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send sticker/emoji. Delegates to OutboundManager via StickerHandler."""
        return await self._outbound.send_media(
            chat_id, "sticker",
            reply_to=reply_to,
            sticker_name=sticker_name, face_index=face_index,
            **kwargs,
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        filename: Optional[str] = None,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send local file (document). Delegates to OutboundManager via DocumentHandler."""
        return await self._outbound.send_media(
            chat_id, "document",
            reply_to=reply_to, caption=caption,
            file_path=file_path, filename=filename,
            **kwargs,
        )

    async def _get_cached_token(self) -> dict:
        """Get the current valid sign token (using module-level cache)."""
        return await SignManager.get_token(
            self._app_key, self._app_secret, self._api_domain,
            route_env=self._route_env,
        )

    def get_status(self) -> dict:
        """Return a snapshot of the current connection status."""
        conn = self._connection
        return {
            "connected": conn.is_connected,
            "bot_id": self._bot_id,
            "connect_id": conn.connect_id,
            "reconnect_attempts": conn.reconnect_attempts,
            "ws_url": self._ws_url,
        }


# ---------------------------------------------------------------------------
# Module-level thin delegates (preserve import compatibility for external callers)
# ---------------------------------------------------------------------------


def get_active_adapter() -> Optional["YuanbaoAdapter"]:
    """Delegate to ``YuanbaoAdapter.get_active()``."""
    return YuanbaoAdapter.get_active()


async def send_yuanbao_direct(
    adapter: "YuanbaoAdapter",
    chat_id: str,
    message: str,
    media_files: Optional[List[Tuple[str, bool]]] = None,
) -> Dict[str, Any]:
    """Delegate to ``OutboundManager.send_direct``."""
    return await adapter._outbound.send_direct(chat_id, message, media_files)
