"""Zulip gateway adapter.

Connects to any Zulip server (cloud or self-hosted) via the official
``zulip`` Python package.  Uses the long-polling event queue for
real-time message delivery and the REST API for sending.

Authentication uses the bot's email + API key + server URL — no OAuth
tokens required.

Environment variables:
    ZULIP_SITE_URL           Server URL (e.g. https://your-org.zulipchat.com)
    ZULIP_BOT_EMAIL          Bot's email address
    ZULIP_API_KEY            Bot's API key (from Zulip bot settings)
    ZULIP_ALLOWED_USERS      Comma-separated email addresses
    ZULIP_ALLOW_ALL_USERS    If "true", allow all Zulip users (skip allowlist)
    ZULIP_DEFAULT_STREAM     Default stream for outbound messages
    ZULIP_HOME_TOPIC         Default topic for cron/notification delivery
    ZULIP_HOME_CHANNEL       Home stream:topic for cron/notification delivery
    ZULIP_CERT_BUNDLE        Path to a CA bundle for self-hosted/self-signed TLS
    ZULIP_ALLOW_INSECURE     If "true", disable TLS verification (dev only)
    ZULIP_REQUIRE_MENTION    Require @mention in streams (default: "true")
    ZULIP_FREE_RESPONSE_STREAMS  Comma-separated stream names or IDs that
                             don't require @mention
    ZULIP_CONVERT_MATH       Rewrite common LaTeX delimiters to Zulip KaTeX
                             form before send (default: "true")
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    ExecApprovalPrompt,
    MessageEvent,
    MessageType,
    SendResult,
)
from .attachments import download_zulip_upload, extract_upload_paths

logger = logging.getLogger(__name__)

# Zulip's standard server limit is 10,000 Unicode code points. Self-hosted
# realms may configure a different limit; keep Hermes at the standard maximum
# so normal responses arrive as one message whenever the realm permits it.
MAX_MESSAGE_LENGTH = 10_000
_STANDALONE_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})
_STANDALONE_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"})


# Zulip KaTeX markup:
#   * Inline math:  $$O(n^2)$$
#   * Display math: fenced ```math ... ``` blocks
#
# Models (especially smaller ones) often emit traditional TeX delimiters
# that Zulip does NOT render:
#   * multi-line / whole-line $$...$$  (Zulip uses $$ only for inline)
#   * \[...\] display and \(...\) inline
#   * $...$ inline
# Rewrite those to Zulip's KaTeX forms before send.  Already-fenced code
# (including ```math) is left untouched.  Toggle with ZULIP_CONVERT_MATH
# or config.extra["convert_math"].
_CODEBLOCK_SPLIT_RE = re.compile(r"(```[\s\S]*?```)")
_DISPLAY_DOLLAR_MATH_RE = re.compile(
    r"(?ms)^[ \t]*\$\$\s*(?P<math>.*?)[ \t]*\$\$[ \t\r]*$"
)
_DISPLAY_BRACKET_MATH_RE = re.compile(
    r"(?ms)^[ \t]*\\\[(?P<math>.*?)\\\][ \t\r]*$"
)
_INLINE_PAREN_MATH_RE = re.compile(r"\\\((?P<math>.*?)\\\)")
# Single-dollar inline: require a space/BOL boundary and either a compact
# non-space body or a body with LaTeX-ish operators.  Pure currency-like
# amounts ($5, $5.00) are excluded to avoid false positives in prose.
_INLINE_DOLLAR_MATH_RE = re.compile(
    r"""
    (?:^|(?<=\s))                 # preceded by BOL or whitespace
    (?P<prefix>[^\w$\s]*)         # optional punctuation before the opener
    \$
    (?P<math>
        (?!\d+(?:\.\d+)?(?:\s|$)) # not a bare currency amount
        (?:
            [^\s$]+               # compact: $x$, $O(n)$, $n=1$
          | [^$\n]*[_^{}=\\<>+][^$\n]*  # or contains LaTeX-ish chars
        )
    )
    \$
    (?P<suffix>[^\w$\s]*)         # optional punctuation after the closer
    (?=\s|$)                      # followed by whitespace or EOL
    """,
    re.VERBOSE | re.MULTILINE,
)


def _process_noncode_parts(content: str, fn) -> str:
    """Apply *fn* to text outside fenced code blocks only."""
    if not content:
        return content
    out: list[str] = []
    for index, part in enumerate(_CODEBLOCK_SPLIT_RE.split(content)):
        if index % 2 == 1:
            out.append(part)
        else:
            out.append(fn(part))
    return "".join(out)


def _to_display_math(match: re.Match) -> str:
    body = match["math"].strip("\n")
    return f"```math\n{body}\n```"


def _to_inline_math(match: re.Match) -> str:
    body = match["math"].strip()
    prefix = match.groupdict().get("prefix") or ""
    suffix = match.groupdict().get("suffix") or ""
    return f"{prefix}$${body}$${suffix}"


def _normalize_zulip_math(content: str) -> str:
    r"""Rewrite non-Zulip math delimiters to Zulip KaTeX forms.

    Conversions (outside fenced code only):

    * whole-line ``$$...$$`` → fenced ``math`` block (display)
    * whole-line ``\[...\]`` → fenced ``math`` block (display)
    * ``\(...\)`` → ``$$...$$`` (inline)
    * ``$...$`` → ``$$...$$`` when the body looks like math (not currency)

    Mid-sentence single-line ``$$...$$`` (already valid Zulip inline math)
    is left unchanged because the display-dollar pattern only matches a
    full line.
    """
    if not content:
        return content
    # Cheap bail-out: nothing we rewrite can appear without one of these.
    if "$" not in content and r"\(" not in content and r"\[" not in content:
        return content

    content = _process_noncode_parts(
        content, lambda p: _DISPLAY_DOLLAR_MATH_RE.sub(_to_display_math, p)
    )
    content = _process_noncode_parts(
        content, lambda p: _DISPLAY_BRACKET_MATH_RE.sub(_to_display_math, p)
    )
    content = _process_noncode_parts(
        content, lambda p: _INLINE_PAREN_MATH_RE.sub(_to_inline_math, p)
    )
    content = _process_noncode_parts(
        content, lambda p: _INLINE_DOLLAR_MATH_RE.sub(_to_inline_math, p)
    )
    return content



# Inbound uploads arrive in message content as Markdown links targeting the
# realm's /user_uploads/ endpoint. All of them are materialized into Hermes's
# media cache; the core then routes images to vision and documents to its normal
# file/terminal path. Keep this compatibility helper for callers that only need
# to identify image links.
_UPLOAD_IMAGE_EXTENSIONS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

def _extract_upload_image_paths(content: str) -> list[str]:
    """Return unique ``/user_uploads/`` image paths from *content*, in order.

    Absolute URLs are normalized down to their ``/user_uploads/...`` path so
    the download always targets the configured site (never a foreign host
    smuggled into a markdown link).
    """
    paths: list[str] = []
    for path in extract_upload_paths(content):
        if Path(path).suffix.lower() not in _UPLOAD_IMAGE_EXTENSIONS:
            continue
        paths.append(path)
    return paths

# Event-queue reconnect parameters (exponential backoff).
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MAX_DELAY = 60.0
_RECONNECT_JITTER = 0.2

# Default per-stream cap for missed-message catch-up (see _run_missed_message_catchup).
_CATCHUP_DEFAULT_MAX_MESSAGES = 100


def _is_retryable_error(exc: Exception) -> bool:
    """Determine if a Zulip event queue error is worth retrying.

    Network errors, timeouts, and server errors (5xx) are retryable.
    Authentication failures (401/403) and other client errors (4xx)
    are not — the configuration or credentials need to be fixed first.

    Falls back to *retryable* for unrecognized error shapes so the
    event queue keeps trying on transient issues.
    """
    exc_name = type(exc).__name__

    # Network-level errors are always retryable.
    if any(keyword in exc_name for keyword in ("ConnectionError", "Timeout", "SSLError")):
        return True

    # Check for Zulip ``ClientError`` that carries an HTTP status.
    if hasattr(exc, "http_status"):
        status = getattr(exc, "http_status", 0)
        if status in (401, 403):
            return False
        if 400 <= status < 500:
            return False  # Client errors — user must fix config.

    return True


# ---------------------------------------------------------------------------
# Chat-ID helpers
#
# Zulip uses two distinct message types:
#   * Stream messages live in a stream and have a topic.
#   * Direct messages (DMs) are between exactly two users.
#
# We encode both into a single *chat_id* string that the gateway session
# layer can round-trip without understanding Zulip internals.
# ---------------------------------------------------------------------------

_DM_PREFIX = "dm:"
_GROUP_DM_PREFIX = "group_dm:"


def _build_zform_widget_content(heading: str, choices: list[dict[str, Any]]) -> str:
    """Serialize a Zulip ``zform`` choices widget for ``send_message``.

    Intended use:
        Zulip does not have Telegram-style hidden callback buttons for normal
        bot messages.  Its native button-like primitive is a message widget
        called ``zform``.  Each zform choice renders as a button in compatible
        Zulip clients, and clicking it sends a normal visible reply message
        containing that choice's ``reply`` string.

    How it works:
        The Zulip send-message API accepts a ``widget_content`` form field that
        must be a JSON string.  For zform choices, the JSON object has
        ``widget_type: "zform"`` and ``extra_data.type: "choices"``.  Hermes
        chooses reply strings that existing gateway text handlers already
        understand (for example ``/approve`` or a literal clarify answer), so
        zform remains a richer UI over the same audited text-command paths.

    All fields are normalized to strings because Zulip's validator expects
    string values for the choice labels and reply payloads.
    """
    normalized_choices = [
        {
            "type": str(choice.get("type", "multiple_choice")),
            "short_name": str(choice.get("short_name", "")),
            "long_name": str(choice.get("long_name", "")),
            "reply": str(choice.get("reply", "")),
        }
        for choice in choices
    ]
    return json.dumps(
        {
            "widget_type": "zform",
            "extra_data": {
                "type": "choices",
                "heading": str(heading),
                "choices": normalized_choices,
            },
        },
        ensure_ascii=False,
    )


def _format_approval_zform_heading(command: str, description: str = "") -> str:
    """Build a zform heading that states what the user is approving.

    Zulip clients foreground the zform widget over the message body (same
    pattern as the trivia-quiz bot, which puts the question in ``heading``).
    A generic heading like "Command approval required" leaves users staring at
    approve/deny buttons with no idea which command is blocked.
    """
    cmd = (command or "").strip()
    if not cmd:
        return "Command approval required"

    max_heading = 120

    if cmd.startswith("execute_code"):
        inner = [ln.strip() for ln in cmd.splitlines()[1:-1] if ln.strip()]
        preview = inner[0] if inner else ""
        prefix = "Approve execute_code: "
        budget = max_heading - len(prefix)
        if budget > 1 and len(preview) > budget:
            preview = preview[: budget - 1] + "…"
        return f"{prefix}{preview}" if preview else "Approve execute_code script"

    preview = cmd.splitlines()[0].strip() if "\n" in cmd else cmd
    prefix = "Approve: "
    budget = max_heading - len(prefix)
    if budget > 1 and len(preview) > budget:
        preview = preview[: budget - 1] + "…"
    return f"{prefix}{preview}"


# Shared exec-approval vocabulary (once/session/always/deny) → Zulip zform replies.
# Button clicks send these strings as normal messages; the gateway intercepts them
# the same way as typed `/approve`.
_ZULIP_EA_BUTTONS = {
    "once": ("Once", "Approve once", "/approve"),
    "session": ("Session", "Approve this pattern for the session", "/approve session"),
    "always": ("Always", "Approve this pattern permanently", "/approve always"),
    "deny": ("Deny", "Deny and cancel the command", "/deny"),
}


# Visible zform replies for /model. Clicks are normal Zulip messages, so the
# token is short and the adapter consumes it before the agent sees it.
_MODEL_PICKER_PREFIX = "hm-model:"
_MODEL_PICKER_PAGE = 8
_MODEL_PICKER_TTL_SECONDS = 600
_MODEL_PICKER_RE = re.compile(
    r"^hm-model:(?:"
    r"p:(?P<p_page>\d+):(?P<p_idx>\d+)"
    r"|m:(?P<m_page>\d+):(?P<m_idx>\d+)"
    r"|pp:(?P<pp>\d+)"
    r"|more:(?P<more>\d+)"
    r"|back"
    r"|cancel"
    r")$"
)


def _parse_model_picker_token(content: str) -> Optional[dict[str, Any]]:
    """Return a picker action dict, or None when *content* is ordinary text."""
    text = re.sub(r"@\*\*[^*]+\*\*", " ", content or "")
    text = re.sub(r"@\S+", " ", text)
    text = " ".join(text.split())
    match = _MODEL_PICKER_RE.match(text)
    if match is None:
        return None
    if match.group("p_page") is not None:
        return {"kind": "provider", "page": int(match.group("p_page")), "index": int(match.group("p_idx"))}
    if match.group("m_page") is not None:
        return {"kind": "model", "page": int(match.group("m_page")), "index": int(match.group("m_idx"))}
    if match.group("pp") is not None:
        return {"kind": "provider_page", "page": int(match.group("pp"))}
    if match.group("more") is not None:
        return {"kind": "model_page", "page": int(match.group("more"))}
    if text.endswith(":back"):
        return {"kind": "back"}
    return {"kind": "cancel"}


def _picker_page(items: list, page: int, size: int = _MODEL_PICKER_PAGE) -> tuple[list, int, int]:
    """Return ``(slice, clamped_page, total_pages)`` for a picker page."""
    total_pages = max(1, (len(items) + size - 1) // size)
    page = min(max(page, 0), total_pages - 1)
    start = page * size
    return items[start:start + size], page, total_pages


def _short_button_label(text: str, limit: int = 36) -> str:
    label = " ".join((text or "").split())
    if len(label) <= limit:
        return label or "?"
    return label[: limit - 1] + "…"


def _build_stream_chat_id(stream_id: int, topic: str) -> str:
    """Encode a stream message's origin as a stable chat ID.

    Format: ``"{stream_id}:{topic}"``
    """
    return f"{stream_id}:{topic}"


def _parse_stream_chat_id(chat_id: str) -> Optional[tuple[int, str]]:
    """Parse a canonical stream chat ID back into ``(stream_id, topic)``.

    Returns ``None`` if *chat_id* does not look like a canonical stream chat
    ID with a numeric stream ID prefix.
    """
    # Canonical stream chat IDs look like "123:some topic" — the part before
    # the first colon must be a plain integer.
    colon = chat_id.find(":")
    if colon < 1:
        return None
    stream_part = chat_id[:colon]
    if not stream_part.isdigit():
        return None
    topic = chat_id[colon + 1:] or "(no topic)"
    return (int(stream_part), topic)


def _parse_stream_name_topic(chat_id: str) -> Optional[tuple[str, str]]:
    """Parse a documented ``stream_name:topic`` target.

    This is intentionally separate from :func:`_parse_stream_chat_id` because
    the canonical internal format uses numeric stream IDs, while config/docs
    may use human-friendly stream names.
    """
    colon = chat_id.find(":")
    if colon < 1:
        return None
    stream_name = chat_id[:colon].strip()
    topic = chat_id[colon + 1:] or "(no topic)"
    if not stream_name or stream_name.isdigit():
        return None
    if chat_id.startswith(_DM_PREFIX) or chat_id.startswith(_GROUP_DM_PREFIX):
        return None
    if stream_name in {"dm", "group_dm"}:
        return None
    if "@" in stream_name:
        return None
    return stream_name, topic


def _build_dm_chat_id(sender_email: str) -> str:
    """Encode a DM origin as a stable chat ID.

    Format: ``"dm:{sender_email}"``
    """
    # Defensive: strip existing prefix so stale/cached IDs don't double-prefix.
    sender_email = sender_email.removeprefix(_DM_PREFIX)
    return f"{_DM_PREFIX}{sender_email}"


def _parse_dm_chat_id(chat_id: str) -> Optional[str]:
    """Parse a DM chat ID back into the sender email.

    Returns ``None`` if *chat_id* does not look like a DM chat ID.
    """
    if chat_id.startswith(_DM_PREFIX) and "@" in chat_id:
        return chat_id[len(_DM_PREFIX):]
    return None


def is_dm_chat_id(chat_id: str) -> bool:
    """Return True if *chat_id* represents a DM conversation."""
    return chat_id.startswith(_DM_PREFIX)


def _build_group_dm_chat_id(participant_emails: list) -> str:
    """Encode a group DM (3+ participants) as a stable chat ID.

    Sorts emails for deterministic round-tripping regardless of the order
    in which Zulip delivers the participant list.

    Format: ``"group_dm:email1@example.com,email2@example.com,..."``
    """
    sorted_emails = sorted(participant_emails)
    return f"{_GROUP_DM_PREFIX}{','.join(sorted_emails)}"


def _parse_group_dm_chat_id(chat_id: str) -> Optional[list]:
    """Parse a group DM chat ID back into a sorted list of emails.

    Returns ``None`` if *chat_id* does not look like a group DM chat ID.
    """
    if not chat_id.startswith(_GROUP_DM_PREFIX):
        return None
    emails_str = chat_id[len(_GROUP_DM_PREFIX):]
    if not emails_str:
        return None
    return emails_str.split(",")


def is_group_dm_chat_id(chat_id: str) -> bool:
    """Return True if *chat_id* represents a group DM conversation."""
    return chat_id.startswith(_GROUP_DM_PREFIX)


def _build_stream_typing_request(stream_id: int, topic: str, op: str) -> dict[str, Any]:
    """Build Zulip's channel typing payload using the modern stream_id + topic shape.

    Zulip's current typing API for streams/channels requires the numeric
    ``stream_id`` together with the ``topic``. The older ``{"to": [stream_name]}``
    form is only valid for direct-message recipients and is no longer reliable
    (or accepted) for channel typing notifications.
    """
    return {
        "stream_id": stream_id,
        "topic": topic,
        "type": "stream",
        "op": op,
    }


def _extract_dm_recipients(
    display_recipient: Any, bot_email: str, sender_email: str
) -> list:
    """Extract DM participant emails from ``display_recipient``.

    For 1:1 DMs, returns ``[other_user_email]``.
    For group DMs (3+ users), returns all emails except the bot's.
    Falls back to ``[sender_email]`` if the payload is malformed.
    """
    if isinstance(display_recipient, list):
        emails = [
            u.get("email", "")
            for u in display_recipient
            if isinstance(u, dict) and u.get("email") != bot_email
        ]
        if emails:
            return emails

    return [sender_email]


def _resolve_stream_name(
    message: dict[str, Any],
    stream_id: int,
    stream_name_cache: dict[int, str],
) -> str:
    """Get the stream name from cache or fall back to the message payload.

    Zulip's ``display_recipient`` for stream messages is either:
    - A string with the stream name (modern Zulip).
    - A dict with a ``name`` key (legacy Zulip).

    Falls back to ``str(stream_id)`` if nothing is available.
    """
    if stream_id in stream_name_cache:
        return stream_name_cache[stream_id]

    # Try display_recipient from the message payload.
    dr = message.get("display_recipient")
    if isinstance(dr, str) and dr:
        return dr
    if isinstance(dr, dict):
        name = dr.get("name", "")
        if name:
            return name

    return str(stream_id)


def _strip_bot_mention(
    content: str,
    mention_patterns: list[str],
) -> str:
    """Remove bot mention patterns from message content.

    Strips each pattern from the content (case-insensitive), then
    normalizes whitespace (collapses double spaces, strips edges).

    Zulip renders ``@**Full Name**`` and ``@email@example.com`` as
    clickable mentions.  We remove them so the agent doesn't see its
    own name as part of the user's message.
    """
    cleaned = content
    for pattern in mention_patterns:
        # Case-insensitive removal.
        cleaned = re.sub(
            re.escape(pattern), "", cleaned, count=1, flags=re.IGNORECASE
        )
    # Collapse any double spaces left by mention removal and strip edges.
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    return cleaned


def _format_context_block(context_lines: list) -> str:
    """Format fetched context messages as a readable block prepended to the
    user's message.  The block is separated from the user's current message
    by a ``---`` delimiter so the agent can clearly distinguish context
    from the question being asked.
    """
    if not context_lines:
        return ""
    header = "Recent messages in this topic:"
    body = "\n".join(context_lines)
    return f"{header}\n{body}\n---\n\n"


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------


def check_zulip_requirements(config: Optional[PlatformConfig] = None) -> bool:
    """Return True if the Zulip adapter can be used.

    Configured users get the same opt-in dependency behavior as other platform
    plugins: if the SDK is missing, try the allowlisted lazy install before
    reporting the platform unavailable. Unconfigured users do not import or
    install the Zulip SDK.
    """
    extra = config.extra if config else {}
    configured_key = (config.token or config.api_key) if config else ""
    api_key = configured_key or os.getenv("ZULIP_API_KEY", "")
    email = extra.get("bot_email") or os.getenv("ZULIP_BOT_EMAIL", "")
    site = extra.get("site_url") or os.getenv("ZULIP_SITE_URL", "")

    if not api_key:
        logger.debug("Zulip: ZULIP_API_KEY not set")
        return False
    if not email:
        logger.warning("Zulip: ZULIP_BOT_EMAIL not set")
        return False
    if not site:
        logger.warning("Zulip: ZULIP_SITE_URL not set")
        return False

    try:
        import zulip
        return True
    except ImportError:
        pass

    # Lazy-install the ``zulip`` pyproject extra via ``pm.extras`` — the same
    # seam Slack and Matrix use — instead of the removed tools.lazy_deps table.
    from pm.extras import ensure_and_bind

    def _import():
        import zulip
        return {}

    return ensure_and_bind("zulip", _import, globals())


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ZulipAdapter(BasePlatformAdapter):
    """Gateway adapter for Zulip (cloud or self-hosted)."""

    SUPPORTS_MESSAGE_EDITING = True
    splits_long_messages = True  # send() chunks via truncate_message(MAX_MESSAGE_LENGTH)

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("zulip"))

        self._site_url: str = (
            config.extra.get("site_url", "")
            or os.getenv("ZULIP_SITE_URL", "")
        ).rstrip("/")
        self._bot_email: str = (
            config.extra.get("bot_email", "")
            or os.getenv("ZULIP_BOT_EMAIL", "")
        )
        self._api_key: str = (
            config.token
            or config.api_key
            or os.getenv("ZULIP_API_KEY", "")
        )
        self._cert_bundle: str = os.getenv("ZULIP_CERT_BUNDLE", "")
        self._allow_insecure: bool = os.getenv(
            "ZULIP_ALLOW_INSECURE", "false"
        ).lower() in ("true", "1", "yes")
        self._streaming_edits_warning_logged = False
        # Rewrite common LaTeX delimiters to Zulip KaTeX before send.
        # config.extra["convert_math"] wins when present; else env
        # ZULIP_CONVERT_MATH (default on).
        if "convert_math" in config.extra:
            self._convert_math = bool(config.extra.get("convert_math"))
        else:
            self._convert_math = os.getenv(
                "ZULIP_CONVERT_MATH", "true"
            ).lower() in ("true", "1", "yes")

        # Mention gating configuration (follows Discord's pattern).
        self._require_mention: bool = os.getenv(
            "ZULIP_REQUIRE_MENTION", "true"
        ).lower() not in ("false", "0", "no")

        free_streams_raw = os.getenv("ZULIP_FREE_RESPONSE_STREAMS", "")
        self._free_response_streams: set = {
            s.strip().lower()
            for s in free_streams_raw.split(",")
            if s.strip()
        }

        # Historical context: when the bot is @mentioned in a stream, fetch
        # the last N messages from that stream+topic via Zulip's /messages API
        # and inject them as context before the user's message.  Survives
        # disconnects — the bot uses Zulip as the source of truth.
        try:
            self._context_depth: int = max(
                0,
                int(
                    config.extra.get("context_depth")
                    or os.getenv("ZULIP_CONTEXT_DEPTH", "0")
                ),
            )
        except (TypeError, ValueError):
            self._context_depth = 0

        # Missed-message catch-up (opt-in, default OFF).
        #
        # The Zulip events API only delivers events from queue registration
        # onward, so a message that arrives while the gateway is down is never
        # seen by the live queue. When enabled, this adapter back-fills each
        # known stream from a persisted watermark immediately before our
        # wrapper calls ``call_on_each_event``: on boot, and when that call
        # *returns* so the wrapper loop re-enters.
        #
        # SDK 0.9.1 handles ``BAD_EVENT_QUEUE_ID`` *inside* ``call_on_each_event``
        # by registering a replacement queue without returning here. That inner
        # re-register does not run this sweep. A message that arrives after the
        # history snapshot and before queue registration can also be missed.
        # Live events still go through ``_seen_events`` and mention gating.
        #
        # Default OFF on purpose: enabling it on a bot that has been offline for
        # a while replays the accumulated backlog (bounded by the per-stream
        # cap), which is usually surprising.  Opt in deliberately.
        self._catchup_enabled: bool = (
            str(config.extra.get("catchup_enabled", "")).lower()
            in ("true", "1", "yes")
            or os.getenv("ZULIP_CATCHUP", "false").lower()
            in ("true", "1", "yes")
        )
        # Per-stream cap on messages replayed per (re-)register — bounds the
        # backlog a long downtime can produce.
        try:
            self._catchup_max_messages: int = max(
                1,
                int(
                    config.extra.get("catchup_max_messages")
                    or os.getenv(
                        "ZULIP_CATCHUP_MAX_MESSAGES",
                        str(_CATCHUP_DEFAULT_MAX_MESSAGES),
                    )
                ),
            )
        except (TypeError, ValueError):
            self._catchup_max_messages = _CATCHUP_DEFAULT_MAX_MESSAGES

        # Background thread running the event queue. ``_event_generation``
        # fences callbacks so a leftover SDK poll cannot dispatch after
        # disconnect/reconnect (the SDK loop does not observe ``_closing``).
        self._event_thread: Optional[threading.Thread] = None
        self._event_generation: int = 0
        self._model_picker_state: dict[str, dict[str, Any]] = {}
        self._closing = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._event_futures: set[Any] = set()

        # Bot identity (resolved on connect)
        self._bot_user_id: int = -1
        self._bot_full_name: str = ""

        # Dedup cache: event_id → timestamp
        self._seen_events: dict[str, float] = {}
        self._SEEN_MAX = 2000
        self._SEEN_TTL = 300  # 5 minutes

        # Stream name → stream_id cache (populated on connect)
        self._stream_id_cache: dict[str, int] = {}
        # stream_id → stream_name reverse cache
        self._stream_name_cache: dict[int, str] = {}

        # Email.lower() → user_id cache for DM typing.
        #
        # CRITICAL Zulip API requirement (DM / direct-message typing):
        #   set_typing_status for type="direct" REQUIRES an array of *integer*
        #   user_ids in the "to" field.  Passing email strings (or any non-int)
        #   is rejected by the server (the client library may appear to accept
        #   it but the wire payload fails validation on the Zulip side).
        #
        # We populate this cache from every inbound private message's
        # "sender_id" (always present and numeric in Zulip events) and fall
        # back to on-demand lookup via client.get_user(email=...) for the
        # (rare) case of an outbound DM typing indicator before any inbound
        # traffic from that user has been seen in the current process.
        self._user_id_cache: dict[str, int] = {}

        # Graceful shutdown: event that wakes the event-queue thread
        # immediately when disconnect() is called, instead of waiting
        # for the full backoff sleep to elapse.
        self._shutdown_event = threading.Event()
        self._consecutive_failures = 0

        # Zulip client — created in connect(), used by the event-queue thread.
        # Send operations use _build_send_client() instead (thread safety).
        self._client: Any = None

    def _build_send_client(self) -> Any:
        """Create a fresh Zulip client for a send operation.

        The event-queue thread holds ``self._client`` for long-polling.
        Sharing a ``requests.Session`` across threads corrupts SSL state,
        so send operations get their own ephemeral client.
        """
        import zulip
        kwargs: dict[str, Any] = {
            "site": self._site_url,
            "email": self._bot_email,
            "api_key": self._api_key,
        }
        if self._cert_bundle:
            kwargs["cert_bundle"] = self._cert_bundle
        if self._allow_insecure:
            kwargs["insecure"] = True
        return zulip.Client(**kwargs)

    def _close_zulip_http_session(self, client: Any) -> None:
        """Abort an in-flight long-poll by closing the SDK's requests session."""
        session = getattr(client, "session", None) if client is not None else None
        if session is None:
            return
        try:
            session.close()
        except Exception:
            logger.debug("Zulip: error closing HTTP session", exc_info=True)

    def _dispatch_zulip_event(self, event: dict[str, Any], generation: int) -> None:
        """Drop events from a retired reader, even if ``_closing`` was cleared."""
        if not self._reader_is_current(generation):
            return
        self._on_zulip_event(event)

    def _reader_is_current(self, generation: int) -> bool:
        return not self._closing and generation == self._event_generation

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to Zulip, verify auth, and start the event queue.

        ``is_reconnect`` is forwarded by the gateway reconnect watcher after an
        outage. Zulip registers a fresh event queue on each connect; optional
        catch-up (when enabled) back-fills missed stream messages before going
        live.
        """
        if not self._site_url or not self._api_key or not self._bot_email:
            logger.error(
                "Zulip: missing configuration (site_url, api_key, or bot_email)"
            )
            return False

        # One bot per gateway (machine-global): Zulip delivers every message to
        # EVERY event queue the bot has open, so a second profile/process polling
        # the same bot fans each message into N sessions with N replies.
        if not self._acquire_platform_lock(
            "zulip-bot",
            f"{self._site_url}:{self._bot_email}",
            f"Zulip bot {self._bot_email} on {self._site_url}",
        ):
            return False

        import zulip

        leftover_client = self._client
        self._event_generation += 1
        self._close_zulip_http_session(leftover_client)

        # Create the synchronous Zulip client.
        client_kwargs: dict[str, Any] = {
            "site": self._site_url,
            "email": self._bot_email,
            "api_key": self._api_key,
        }
        if self._cert_bundle:
            client_kwargs["cert_bundle"] = self._cert_bundle
        if self._allow_insecure:
            client_kwargs["insecure"] = True

        self._client = zulip.Client(**client_kwargs)

        # Verify credentials by fetching the bot's own profile.
        try:
            result = self._client.get_profile()
        except Exception as exc:
            logger.error("Zulip: failed to authenticate — %s", exc)
            return False

        if result.get("result") != "success":
            msg = result.get("msg", "unknown error")
            logger.error(
                "Zulip: authentication failed — %s. "
                "Check ZULIP_API_KEY, ZULIP_BOT_EMAIL, and ZULIP_SITE_URL.",
                msg,
            )
            return False

        profile = result.get("profile") if isinstance(result.get("profile"), dict) else result
        self._bot_user_id = profile.get("user_id", -1)
        self._bot_full_name = profile.get("full_name", "")
        logger.info(
            "Zulip: authenticated as %s (user_id=%d) on %s",
            self._bot_email,
            self._bot_user_id,
            self._site_url,
        )

        # Populate stream-id cache early (helps typing indicators on first messages).
        self._refresh_stream_cache()
        logger.debug("Zulip: adapter fully connected and ready (stream cache has %d entries so far)", len(self._stream_id_cache))

        # Start the event queue in a background thread.
        self._loop = asyncio.get_running_loop()
        self._closing = False
        self._shutdown_event.clear()
        self._consecutive_failures = 0
        self._event_thread = threading.Thread(
            target=self._run_event_queue,
            name="zulip-event-queue",
            daemon=True,
        )
        self._event_thread.start()

        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Stop the event queue, cancel background tasks, and close the client."""
        self._closing = True
        self._model_picker_state.clear()
        self._event_generation += 1
        self._shutdown_event.set()  # Wake up the event thread immediately.
        self._close_zulip_http_session(self._client)
        with contextlib.suppress(Exception):
            self._release_platform_lock()

        thread = self._event_thread
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 10.0)
            if thread.is_alive():
                logger.warning(
                    "Zulip: event-queue thread still alive after join; "
                    "stale events are dropped by generation fence"
                )

        # Cancel any in-flight message-processing tasks that were
        # scheduled on the asyncio event loop.
        try:
            await self.cancel_background_tasks()
        except Exception:
            pass

        for future in list(self._event_futures):
            future.cancel()
        self._event_futures.clear()

        self._client = None
        self._loop = None

        # Clear caches to free memory and avoid stale data on reconnect.
        self._seen_events.clear()
        self._stream_id_cache.clear()
        self._stream_name_cache.clear()
        self._user_id_cache.clear()
        self._consecutive_failures = 0

        self._mark_disconnected()
        logger.info("Zulip: disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message (or multiple chunks) to a Zulip chat."""
        if not content:
            return SendResult(success=True)

        outbound_chat_id = self._metadata_adjusted_chat_id(chat_id, metadata)

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, MAX_MESSAGE_LENGTH)

        last_id = None
        for chunk in chunks:
            result = await asyncio.to_thread(
                self._do_send_message,
                outbound_chat_id,
                chunk,
                reply_to,
            )
            if result.success:
                last_id = result.message_id
            else:
                return result

        return SendResult(success=True, message_id=last_id)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        """Return chat name and type (dm/stream)."""
        # Try stream first.
        parsed = _parse_stream_chat_id(chat_id)
        if parsed:
            stream_id, topic = parsed
            stream_name = self._stream_name_cache.get(stream_id, chat_id)
            return {"name": f"#{stream_name} > {topic}", "type": "stream"}

        # Try DM.
        dm_email = _parse_dm_chat_id(chat_id)
        if dm_email:
            return {"name": dm_email, "type": "dm"}

        return {"name": chat_id, "type": "dm"}

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    async def send_typing(
        self, chat_id: str, metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Send a typing indicator to Zulip.

        For streams, the modern Zulip typing API requires a numeric ``stream_id``
        + ``topic`` (see :func:`_build_stream_typing_request`). For DMs it
        requires integer user IDs (never bare emails) in the ``to`` array.
        Both are resolved on-demand via caches + resolvers below.
        """
        if not self._client:
            logger.debug("Zulip: send_typing called but no client yet for chat_id=%r", chat_id)
            return

        outbound_chat_id = chat_id
        thread_id = metadata.get("thread_id") if metadata else None
        if thread_id and not _parse_stream_chat_id(chat_id):
            if not is_dm_chat_id(chat_id) and not is_group_dm_chat_id(chat_id):
                outbound_chat_id = f"{chat_id}:{thread_id}"

        request = self._build_typing_request(outbound_chat_id, op="start")
        if not request:
            logger.warning("Zulip: send_typing failed — could not resolve chat_id %r (no request built)", outbound_chat_id)
            return

        # Success-path logging is debug only; INFO would spam gateway.log on
        # every assistant turn because _keep_typing refreshes every ~2s.
        logger.debug("Zulip: sending typing indicator → chat_id=%r payload=%s", outbound_chat_id, request)

        try:
            send_client = self._build_send_client()
            result = await asyncio.to_thread(send_client.set_typing_status, request)
            if result.get("result") != "success":
                logger.debug("Zulip: send_typing API call failed — %s (payload was %s)", result.get("msg", "unknown error"), request)
            else:
                logger.debug("Zulip: send_typing SUCCESS for %r (payload=%s)", chat_id, request)
        except Exception as exc:
            logger.debug("Zulip: send_typing exception — %s (payload was %s)", exc, request)

    async def stop_typing(
        self, chat_id: str, metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Clear the typing indicator in Zulip (send 'op': 'stop').

        Thread metadata is applied with the same outbound_chat_id adjustment as
        send_typing. That matters when the gateway source is a named stream
        (``general``) and the actual Zulip topic lives in metadata
        (``thread_id``): the stop request must resolve to the same numeric
        ``stream_id`` + topic payload as the start request.
        """
        if not self._client:
            logger.debug("Zulip: stop_typing called but no client yet for chat_id=%r", chat_id)
            return

        outbound_chat_id = chat_id
        thread_id = metadata.get("thread_id") if metadata else None
        if thread_id and not _parse_stream_chat_id(chat_id):
            if not is_dm_chat_id(chat_id) and not is_group_dm_chat_id(chat_id):
                outbound_chat_id = f"{chat_id}:{thread_id}"

        request = self._build_typing_request(outbound_chat_id, op="stop")
        if not request:
            return

        # Debug level only — stop is called on every turn completion and must
        # not pollute INFO logs.
        logger.debug("Zulip: sending STOP typing → chat_id=%r payload=%s", outbound_chat_id, request)

        try:
            send_client = self._build_send_client()
            result = await asyncio.to_thread(send_client.set_typing_status, request)
            if result.get("result") != "success":
                logger.debug("Zulip: stop_typing failed — %s (payload=%s)", result.get("msg", "unknown error"), request)
            else:
                logger.debug("Zulip: stop_typing SUCCESS for %r", outbound_chat_id)
        except Exception as exc:
            logger.debug("Zulip: stop_typing exception — %s (payload=%s)", exc, request)

    async def _keep_typing(
        self,
        chat_id: str,
        interval: float = 2.0,
        metadata=None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Preserve Zulip topic metadata when the typing refresher stops."""
        try:
            await super()._keep_typing(
                chat_id,
                interval=interval,
                metadata=metadata,
                stop_event=stop_event,
            )
        finally:
            if metadata:
                try:
                    await self.stop_typing(chat_id, metadata=metadata)
                except Exception:
                    pass

    def warn_streaming_edits_enabled(self) -> None:
        """Warn once before Zulip edit-based streaming mutates a message."""
        if self._streaming_edits_warning_logged:
            return
        self._streaming_edits_warning_logged = True
        logger.warning(
            "Zulip streaming edits are enabled. Disable edit history in Zulip "
            "organization settings before using this, otherwise intermediate "
            "streamed content may be visible in edit history."
        )

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Edit an existing message.

        The gateway streaming adapter contract passes ``finalize=`` on every
        edit and may pass routing ``metadata`` for threaded platforms. Zulip's
        update API is the same call for intermediate and final frames, and the
        message ID already identifies the message to edit, so both keyword
        arguments are accepted for compatibility and intentionally ignored.
        """
        if not self._client or not message_id:
            return SendResult(success=False, error="Not supported")

        self.warn_streaming_edits_enabled()
        formatted = self.format_message(content)
        send_client = self._build_send_client()
        try:
            result = await asyncio.to_thread(
                send_client.update_message,
                {
                    "message_id": int(message_id),
                    "content": formatted,
                },
            )
            if result.get("result") == "success":
                return SendResult(success=True, message_id=message_id)
            else:
                return SendResult(
                    success=False,
                    error=result.get("msg", "update failed"),
                )
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    def format_message(self, content: str) -> str:
        """Zulip supports standard Markdown including code blocks, tables,
        LaTeX math, and image links.

        When ``_convert_math`` is enabled (default), common non-Zulip math
        delimiters are rewritten to Zulip KaTeX forms so smaller models that
        emit ``$...$`` / ``\\(...\\)`` / ``\\[...\\]`` / multi-line ``$$``
        still render.  Disable with ``ZULIP_CONVERT_MATH=false`` or
        ``config.extra["convert_math"]=False``.
        """
        if not self._convert_math:
            return content
        return _normalize_zulip_math(content)

    def _metadata_adjusted_chat_id(
        self,
        chat_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Apply gateway thread metadata to a Zulip outbound chat ID.

        Intended use:
            Gateway callers sometimes pass a stream name in ``chat_id`` and a
            topic in ``metadata['thread_id']``.  Zulip stream sends need both
            values in one target string before :meth:`_do_send_message` can
            turn them into the API request fields.  DMs and already-canonical
            ``stream_id:topic`` chat IDs are left untouched.
        """
        thread_id = metadata.get("thread_id") if metadata else None
        if thread_id and not _parse_stream_chat_id(chat_id) and not _parse_stream_name_topic(chat_id):
            if not is_dm_chat_id(chat_id) and not is_group_dm_chat_id(chat_id):
                return f"{chat_id}:{thread_id}"
        return chat_id

    async def _send_zform_choices(
        self,
        chat_id: str,
        content: str,
        heading: str,
        choices: list[dict[str, Any]],
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send a Zulip message with an attached zform choices widget.

        Intended use:
            High-level Hermes prompts that are naturally a set of choices:
            dangerous-command approvals, slash-command confirmations, and
            multiple-choice clarify questions.

        How it works:
            The text body remains readable in clients that ignore widgets.  The
            attached ``widget_content`` renders buttons in compatible Zulip
            clients.  Button clicks emit normal visible Zulip replies, so the
            choice ``reply`` values deliberately reuse Hermes' existing text
            protocol instead of introducing a second callback channel.

        The zform widget is attached to one Zulip message only.  Callers should
        keep prompt bodies concise; this helper truncates overlong Markdown to
        Zulip's normal outbound chunk size so the buttons stay with the prompt.
        """
        if not self._client:
            return SendResult(success=False, error="Not connected")

        outbound_chat_id = self._metadata_adjusted_chat_id(chat_id, metadata)
        formatted = self.format_message(content or heading)
        chunks = self.truncate_message(formatted, MAX_MESSAGE_LENGTH)
        prompt_text = chunks[0] if chunks else str(heading)
        widget_content = _build_zform_widget_content(heading=heading, choices=choices)
        return await asyncio.to_thread(
            self._do_send_message,
            outbound_chat_id,
            prompt_text,
            reply_to=reply_to,
            widget_content=widget_content,
        )

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        """Render a dangerous-command approval prompt as Zulip zform buttons.

        Intended use:
            Called by ``BasePlatformAdapter.send_exec_approval`` when the gateway
            needs a native button prompt. Overriding this hook (not
            ``send_exec_approval``) is what ``supports_exec_approval_buttons``
            looks at; without it the runner sends plain-text ``/approve`` steps.

        How it works:
            Zulip zform buttons do not deliver hidden callback payloads.  They
            send visible replies, so the zform choices are exactly the text
            commands the existing approval handler already accepts
            (``/approve``, optional ``/approve session`` / ``/approve always``,
            and ``/deny``).  The resolver remains session-scoped in
            ``tools.approval`` just like the plain-text fallback.

            The full command is sent as its own message first so long scripts
            stay readable.  The zform choices are then attached to a short
            follow-up that replies to that command message.
        """
        if prompt.smart_denied:
            heading_line = "⚠️ **Smart DENY — owner override for one operation:**"
            scope_note = (
                "\n\n**Smart DENY:** owner override applies to this one operation only."
            )
        else:
            heading_line = "⚠️ **Command Approval Required**"
            scope_note = ""

        # Full command in its own message so long scripts stay readable and are
        # not trimmed to keep a widget attached.  Chunked sends leave reply_to
        # pointing at the last chunk.
        command_body = (
            f"{heading_line}\n\n"
            f"```\n{prompt.command}\n```\n"
            f"Reason: {prompt.description}{scope_note}"
        )
        command_result = await self.send(
            chat_id=prompt.chat_id,
            content=command_body,
            metadata=prompt.metadata,
        )
        if not command_result.success:
            return command_result

        text_choices: list[str] = []
        choices: list[dict[str, Any]] = []
        for _label, choice, _style in prompt.actions:
            short_name, long_name, reply = _ZULIP_EA_BUTTONS[choice]
            text_choices.append(f"`{reply}`")
            choices.append(
                {
                    "short_name": short_name,
                    "long_name": long_name,
                    "reply": reply,
                }
            )

        if len(text_choices) == 2:
            choice_help = f"{text_choices[0]} or {text_choices[1]}"
        else:
            choice_help = ", ".join(text_choices[:-1]) + f", or {text_choices[-1]}"
        prompt_body = (
            "Use the buttons below, or reply with "
            f"{choice_help}."
        )
        heading = _format_approval_zform_heading(prompt.command, prompt.description)
        return await self._send_zform_choices(
            chat_id=prompt.chat_id,
            content=prompt_body,
            heading=heading,
            choices=choices,
            reply_to=command_result.message_id,
            metadata=prompt.metadata,
        )

    async def send_model_picker(
        self,
        chat_id: str,
        providers: list,
        current_model: str,
        current_provider: str,
        session_key: str,
        on_model_selected,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Two-step /model picker: provider buttons, then that provider's models.

        Zulip zform clicks are ordinary messages. Each button's reply is a short
        ``hm-model:`` token. :meth:`_consume_model_picker_reply` handles it and
        calls ``on_model_selected(chat_id, model_id, provider_slug)``.
        """
        if not self._client:
            return SendResult(success=False, error="Not connected")
        usable = [p for p in (providers or []) if p.get("slug")]
        if not usable:
            return SendResult(success=False, error="No providers")
        key = str(chat_id)
        self._model_picker_state[key] = {
            "providers": usable,
            "current_model": current_model or "",
            "current_provider": current_provider or "",
            "session_key": session_key,
            "on_model_selected": on_model_selected,
            "metadata": metadata,
            "selected_provider": None,
            "selected_provider_name": "",
            "model_list": [],
            "expires_at": time.time() + _MODEL_PICKER_TTL_SECONDS,
        }
        result = await self._send_provider_picker_page(key, 0)
        if not result.success:
            self._model_picker_state.pop(key, None)
        return result

    def _touch_model_picker(self, state: dict[str, Any]) -> None:
        state["expires_at"] = time.time() + _MODEL_PICKER_TTL_SECONDS

    async def _send_provider_picker_page(self, chat_id: str, page: int) -> SendResult:
        state = self._model_picker_state.get(chat_id)
        if not state:
            return SendResult(success=False, error="Picker expired")
        providers = state["providers"]
        shown, page, total_pages = _picker_page(providers, page)
        choices: list[dict[str, Any]] = []
        for index, provider in enumerate(shown):
            name = str(provider.get("name") or provider.get("slug") or "")
            count = provider.get("total_models", len(provider.get("models") or []))
            label = f"{name} ({count})"
            if provider.get("is_current"):
                label = f"\u2713 {label}"
            choices.append({
                "short_name": _short_button_label(label),
                "long_name": name,
                "reply": f"{_MODEL_PICKER_PREFIX}p:{page}:{index}",
            })
        if page + 1 < total_pages:
            choices.append({
                "short_name": "More",
                "long_name": "Next providers",
                "reply": f"{_MODEL_PICKER_PREFIX}pp:{page + 1}",
            })
        choices.append({
            "short_name": "Cancel",
            "long_name": "Cancel",
            "reply": f"{_MODEL_PICKER_PREFIX}cancel",
        })
        page_note = f" ({page + 1}/{total_pages})" if total_pages > 1 else ""
        current = state.get("current_model") or "unknown"
        return await self._send_zform_choices(
            chat_id=chat_id,
            content=(
                f"Current model: `{current}`{page_note}\n\n"
                "Tap a provider. The next message lists its models."
            ),
            heading="Choose a provider",
            choices=choices,
            metadata=state.get("metadata"),
        )

    async def _send_model_list_page(self, chat_id: str, page: int) -> SendResult:
        state = self._model_picker_state.get(chat_id)
        if not state:
            return SendResult(success=False, error="Picker expired")
        models = list(state.get("model_list") or [])
        shown, page, total_pages = _picker_page(models, page)
        choices: list[dict[str, Any]] = []
        for index, model_id in enumerate(shown):
            short = model_id.split("/")[-1] if "/" in model_id else model_id
            choices.append({
                "short_name": _short_button_label(short),
                "long_name": model_id,
                "reply": f"{_MODEL_PICKER_PREFIX}m:{page}:{index}",
            })
        if page + 1 < total_pages:
            choices.append({
                "short_name": "More",
                "long_name": "Next models",
                "reply": f"{_MODEL_PICKER_PREFIX}more:{page + 1}",
            })
        choices.append({
            "short_name": "Back",
            "long_name": "Back to providers",
            "reply": f"{_MODEL_PICKER_PREFIX}back",
        })
        choices.append({
            "short_name": "Cancel",
            "long_name": "Cancel",
            "reply": f"{_MODEL_PICKER_PREFIX}cancel",
        })
        name = state.get("selected_provider_name") or state.get("selected_provider") or "provider"
        provider = next(
            (p for p in state["providers"] if p.get("slug") == state.get("selected_provider")),
            None,
        )
        total = provider.get("total_models", len(models)) if provider else len(models)
        extra = ""
        if total > len(models):
            extra = f"\n\n{total - len(models)} more exist. Type `/model <name>` for one that is not listed."
        page_note = f" ({page + 1}/{total_pages})" if total_pages > 1 else ""
        return await self._send_zform_choices(
            chat_id=chat_id,
            content=f"Provider: **{name}**{page_note}\n\nTap a model.{extra}",
            heading=f"Models from {name}",
            choices=choices,
            metadata=state.get("metadata"),
        )

    async def _consume_model_picker_reply(
        self, chat_id: str, content: str, sender_email: str = ""
    ) -> bool:
        """Handle a zform click. True means the message must not reach the agent."""
        del sender_email  # clicks are chat-scoped; the token is not a hidden callback
        action = _parse_model_picker_token(content)
        if action is None:
            return False
        key = str(chat_id)
        state = self._model_picker_state.get(key)
        if state is None or time.time() > float(state.get("expires_at") or 0):
            self._model_picker_state.pop(key, None)
            await self.send(
                key,
                "That model picker expired. Run `/model` again.",
                metadata=(state or {}).get("metadata"),
            )
            return True
        self._touch_model_picker(state)
        kind = action["kind"]
        if kind == "cancel":
            self._model_picker_state.pop(key, None)
            await self.send(key, "Model switch cancelled.", metadata=state.get("metadata"))
            return True
        if kind == "back":
            state["selected_provider"] = None
            state["model_list"] = []
            await self._send_provider_picker_page(key, 0)
            return True
        if kind == "provider_page":
            await self._send_provider_picker_page(key, action["page"])
            return True
        if kind == "provider":
            shown, _page, _pages = _picker_page(state["providers"], action["page"])
            if action["index"] >= len(shown):
                await self.send(key, "That choice is no longer valid. Run `/model` again.", metadata=state.get("metadata"))
                self._model_picker_state.pop(key, None)
                return True
            provider = shown[action["index"]]
            models = [str(model_id) for model_id in (provider.get("models") or []) if model_id]
            state["selected_provider"] = str(provider.get("slug") or "")
            state["selected_provider_name"] = str(provider.get("name") or provider.get("slug") or "")
            state["model_list"] = models
            if not models:
                await self.send(
                    key,
                    f"No models are listed for {state['selected_provider_name']}. "
                    "Type `/model <name>` instead.",
                    metadata=state.get("metadata"),
                )
                return True
            await self._send_model_list_page(key, 0)
            return True
        if kind == "model_page":
            await self._send_model_list_page(key, action["page"])
            return True
        shown, _page, _pages = _picker_page(list(state.get("model_list") or []), action["page"])
        if action["index"] >= len(shown):
            await self.send(key, "That choice is no longer valid. Run `/model` again.", metadata=state.get("metadata"))
            self._model_picker_state.pop(key, None)
            return True
        model_id = shown[action["index"]]
        provider_slug = str(state.get("selected_provider") or "")
        callback = state.get("on_model_selected")
        metadata = state.get("metadata")
        self._model_picker_state.pop(key, None)
        try:
            result_text = await callback(key, model_id, provider_slug) if callback else "Model updated."
        except Exception as exc:
            logger.exception("Zulip model picker selection failed")
            result_text = f"Model switch failed: {exc}"
        await self.send(key, result_text or "Model updated.", metadata=metadata)
        return True
    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Render a slash-command confirmation prompt as Zulip zform buttons.

        Intended use:
            Called by the gateway's generic slash-confirm primitive for commands
            that need an explicit acknowledgement, such as ``/reload-mcp``.

        How it works:
            The zform button replies are the same text commands intercepted by
            ``GatewayRunner`` for non-button platforms: ``/approve`` resolves
            the prompt once, ``/always`` persists the confirmation policy, and
            ``/cancel`` denies it.  ``confirm_id`` and ``session_key`` are kept
            in the signature for adapter parity; Zulip's visible text replies
            resolve via the existing session-scoped slash-confirm state.
        """
        body = (
            f"**{title}**\n\n"
            f"{message}\n\n"
            "Use the buttons below, or reply with `/approve`, `/always`, or `/cancel`."
        )
        choices = [
            {"short_name": "Approve", "long_name": "Approve once", "reply": "/approve"},
            {"short_name": "Always", "long_name": "Always approve this action", "reply": "/always"},
            {"short_name": "Cancel", "long_name": "Cancel", "reply": "/cancel"},
        ]
        return await self._send_zform_choices(
            chat_id=chat_id,
            content=body,
            heading=title,
            choices=choices,
            metadata=metadata,
        )

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Render a clarify prompt using Zulip zform choices when possible.

        Intended use:
            Called by the gateway-side ``clarify`` tool callback while the agent
            thread is blocked waiting for the user's answer.

        How it works:
            For multiple-choice prompts, each zform button sends the literal
            choice text as a normal Zulip reply.  Before sending, the adapter
            marks the pending clarify entry as awaiting text so
            ``GatewayRunner._maybe_intercept_clarify_text`` captures either a
            button-emitted reply or a user-typed free-form answer.  For
            open-ended prompts, this method falls back to the base text path.
        """
        if not choices:
            return await super().send_clarify(
                chat_id=chat_id,
                question=question,
                choices=choices,
                clarify_id=clarify_id,
                session_key=session_key,
                metadata=metadata,
            )

        from tools.clarify_gateway import mark_awaiting_text
        from tools.clarify_tool import strip_recommended

        mark_awaiting_text(clarify_id)
        # "(Recommended)" is a text-UI affordance appended upstream to the first
        # choice.  On a button surface it is noise in the label and in the visible
        # click reply, so strip it at the presentation boundary.  Matching still
        # resolves either form: clarify_gateway._match_label() strips both sides
        # before comparing, and _clean_answer strips the locked answer.
        labels = [strip_recommended(str(choice)) for choice in choices]
        choice_rows = [
            {
                "short_name": str(index),
                "long_name": label,
                "reply": label,
            }
            for index, label in enumerate(labels, start=1)
        ]
        option_lines = "\n".join(
            f"  {index}. {label}"
            for index, label in enumerate(labels, start=1)
        )
        body = (
            f"❓ {question}\n\n"
            f"{option_lines}\n\n"
            "Use the buttons below, or type any other answer."
        )
        return await self._send_zform_choices(
            chat_id=chat_id,
            content=body,
            heading=str(question),
            choices=choice_rows,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Rich delivery: images, documents, video
    #
    # Zulip supports file uploads via ``POST /user_uploads`` which returns
    # a URI.  That URI is embedded in the message body using standard
    # Markdown image/link syntax:
    #
    #   * Images:  ``![alt](/user_uploads/...)``  →  rendered inline
    #   * Files:   ``[name](/user_uploads/...)``   →  rendered as link
    #
    # Voice messages have NO native representation in Zulip (no voice
    # bubbles).  ``send_voice`` intentionally falls back to the base
    # class, which sends the file path as text.
    # ------------------------------------------------------------------

    def _upload_file(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> Optional[str]:
        """Upload *file_bytes* to Zulip and return the public URI.

        Returns the URI string on success (e.g. ``"/user_uploads/1/..."``),
        or ``None`` on failure.  Logs a warning but never raises.
        """
        if not self._client:
            logger.warning("Zulip: upload_file called while not connected")
            return None

        try:
            # The Zulip client expects a file-like object.  ``upload_file``
            # passes it to ``requests.post(files=[...])`` which reads the
            # content and uses the ``.name`` attribute (if present) as the
            # uploaded filename.  We wrap in ``BytesIO`` and set ``.name``
            # so the server gets a proper filename.
            buf = io.BytesIO(file_bytes)
            buf.name = filename
            send_client = self._build_send_client()
            result = send_client.upload_file(buf)
            if result.get("result") == "success":
                uri = result.get("uri", "")
                if uri:
                    return uri
            logger.warning(
                "Zulip: upload_file failed — %s",
                result.get("msg", "unknown error"),
            )
        except Exception as exc:
            logger.error("Zulip: upload_file exception — %s", exc)
        return None

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Download an image URL, upload to Zulip, and send inline.

        Falls back to sending the URL as plain text if the download or
        upload fails.
        """
        import httpx

        try:
            async with httpx.AsyncClient(
                timeout=30.0, follow_redirects=True,
            ) as client:
                resp = await client.get(
                    image_url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (compatible; HermesAgent/1.0)"
                        ),
                        "Accept": "image/*,*/*;q=0.8",
                    },
                )
                resp.raise_for_status()
                file_bytes = resp.content
        except Exception as exc:
            logger.warning(
                "Zulip: failed to download image %s: %s", image_url, exc,
            )
            text = f"{caption}\n{image_url}" if caption else image_url
            return await self.send(chat_id, content=text, reply_to=reply_to, metadata=metadata)

        # Derive filename from URL path.
        url_path = image_url.rsplit("/", 1)[-1].split("?")[0]
        ext = Path(url_path).suffix.lower() or ".png"
        filename = f"image{ext}"

        uri = await asyncio.to_thread(
            self._upload_file, file_bytes, filename,
        )
        if not uri:
            # Upload failed — fall back to URL in text.
            text = f"{caption}\n{image_url}" if caption else image_url
            return await self.send(chat_id, content=text, reply_to=reply_to, metadata=metadata)

        alt = caption or "image"
        content = f"![{alt}]({uri})"
        return await self.send(chat_id, content=content, reply_to=reply_to, metadata=metadata)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Upload a local image file and send it inline."""
        metadata = kwargs.get("metadata")
        p = Path(image_path)
        if not p.exists():
            text = f"{caption or ''}\n(file not found: {image_path})".strip()
            return await self.send(chat_id, content=text, reply_to=reply_to, metadata=metadata)

        file_bytes = p.read_bytes()
        filename = p.name

        uri = await asyncio.to_thread(
            self._upload_file, file_bytes, filename,
        )
        if not uri:
            return SendResult(
                success=False,
                error="File upload failed",
            )

        alt = caption or "image"
        content = f"![{alt}]({uri})"
        return await self.send(chat_id, content=content, reply_to=reply_to, metadata=metadata)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Upload a local file and send it as a downloadable attachment.

        The file is presented as a Markdown link ``[filename](uri)`` in
        the message body, with *caption* as optional surrounding text.
        """
        metadata = kwargs.get("metadata")
        p = Path(file_path)
        if not p.exists():
            text = f"{caption or ''}\n(file not found: {file_path})".strip()
            return await self.send(chat_id, content=text, reply_to=reply_to, metadata=metadata)

        file_bytes = p.read_bytes()
        filename = file_name or p.name

        uri = await asyncio.to_thread(
            self._upload_file, file_bytes, filename,
        )
        if not uri:
            return SendResult(
                success=False,
                error="File upload failed",
            )

        # Format: optional caption + markdown link to uploaded file.
        link = f"[{filename}]({uri})"
        content = f"{caption}\n{link}" if caption else link
        return await self.send(chat_id, content=content, reply_to=reply_to, metadata=metadata)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Upload a video file and send it as a link.

        Zulip does not inline video playback.  The uploaded file is
        presented as a clickable Markdown link.  This is the best
        representation Zulip can provide for video content.
        """
        metadata = kwargs.get("metadata")
        p = Path(video_path)
        if not p.exists():
            text = f"{caption or ''}\n(file not found: {video_path})".strip()
            return await self.send(chat_id, content=text, reply_to=reply_to, metadata=metadata)

        file_bytes = p.read_bytes()
        filename = p.name

        uri = await asyncio.to_thread(
            self._upload_file, file_bytes, filename,
        )
        if not uri:
            return SendResult(
                success=False,
                error="Video upload failed",
            )

        link = f"[{filename}]({uri})"
        content = f"{caption}\n{link}" if caption else link
        return await self.send(chat_id, content=content, reply_to=reply_to, metadata=metadata)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ):
        """Send an audio file as a downloadable attachment.

        Zulip has no native voice message bubbles, so we upload the audio
        file and send it as a link (same as other file types).
        """
        return await self.send_document(
            chat_id=chat_id,
            file_path=audio_path,
            caption=caption,
            file_name=kwargs.get("file_name"),
            reply_to=reply_to,
            metadata=kwargs.get("metadata"),
        )

    # ------------------------------------------------------------------
    # Internal: sending
    # ------------------------------------------------------------------

    def _do_send_message(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        widget_content: Optional[str] = None,
    ) -> SendResult:
        """Build the correct request dict and call the Zulip API.

        This is synchronous because the Zulip client is not async.  Rich prompt
        helpers pass ``widget_content`` when they need Zulip to render a
        message widget such as zform; ordinary sends leave it unset so the
        request remains a normal text message.
        """
        if not self._client:
            return SendResult(success=False, error="Not connected")

        send_client = self._build_send_client()

        parsed = _parse_stream_chat_id(chat_id)
        if parsed:
            stream_id, topic = parsed
            request = {
                "type": "stream",
                "to": str(stream_id),
                "topic": topic,
                "content": content,
            }
        else:
            named_stream = _parse_stream_name_topic(chat_id)
            if named_stream:
                stream_name, topic = named_stream
                result = send_client.get_stream_id(stream_name)
                if result.get("result") != "success":
                    return SendResult(
                        success=False,
                        error=result.get("msg", f"Stream '{stream_name}' not found"),
                    )
                stream_id = result.get("stream_id")
                if stream_id is None:
                    return SendResult(success=False, error=f"Stream '{stream_name}' not found")
                request = {
                    "type": "stream",
                    "to": str(stream_id),
                    "topic": topic,
                    "content": content,
                }
            elif is_dm_chat_id(chat_id):
                email = _parse_dm_chat_id(chat_id)
                request = {
                    "type": "private",
                    "to": [email],
                    "content": content,
                }
            elif is_group_dm_chat_id(chat_id):
                emails = _parse_group_dm_chat_id(chat_id)
                if emails:
                    request = {
                        "type": "private",
                        "to": emails,
                        "content": content,
                    }
                else:
                    return SendResult(success=False, error="Invalid group DM chat ID")
            else:
                # Fallback: treat as DM to the email itself.
                request = {
                    "type": "private",
                    "to": [chat_id],
                    "content": content,
                }

        if widget_content is not None:
            request["widget_content"] = widget_content

        try:
            result = send_client.send_message(request)
            if result.get("result") == "success":
                msg_id = result.get("id")
                return SendResult(success=True, message_id=str(msg_id) if msg_id else None)
            else:
                return SendResult(
                    success=False,
                    error=result.get("msg", "send failed"),
                )
        except Exception as exc:
            logger.error("Zulip: send_message failed — %s", exc)
            return SendResult(success=False, error=str(exc))

    def _build_typing_request(
        self, chat_id: str, op: str = "start"
    ) -> Optional[dict[str, Any]]:
        """Return a typing request dict for ``set_typing_status``.

        Zulip typing API quirks (both paths now robust):
        * Streams/channels: MUST use ``{"stream_id": N, "topic": "...", "type": "stream"}``.
          The legacy ``{"to": [stream_name]}`` form is no longer reliable
          (community patch + on-demand resolver fixed this).
        * DMs (1:1 or group): MUST use ``{"to": [integer_user_id, ...], "type": "direct"}``.
          Emails (even as strings) are rejected by the server. See the long
          comment on ``_user_id_cache`` and the population site in
          ``_dispatch_inbound``.
        On-demand resolvers + caches make this work even on first message to
        a stream/user in the process lifetime.
        """
        parsed = _parse_stream_chat_id(chat_id)
        if parsed:
            stream_id, topic = parsed
            return _build_stream_typing_request(stream_id, topic, op)

        named_stream = _parse_stream_name_topic(chat_id)
        if named_stream:
            stream_name, topic = named_stream
            key = stream_name.lower()
            stream_id = self._stream_id_cache.get(key)
            if stream_id is None:
                stream_id = self._resolve_stream_id(stream_name)
            if stream_id is not None:
                return _build_stream_typing_request(stream_id, topic, op)

            # Only as a last-ditch fallback (should rarely happen now)
            logger.debug("Zulip: falling back to legacy stream name for typing of %r", chat_id)
            return {"to": [stream_name], "type": "stream", "op": op}

        dm_email = _parse_dm_chat_id(chat_id)
        if dm_email:
            key = dm_email.lower()
            user_id = self._user_id_cache.get(key)
            if user_id is None:
                user_id = self._resolve_user_id(dm_email)
            if user_id is not None:
                return {"to": [user_id], "type": "direct", "op": op}
            logger.warning("Zulip: could not resolve DM email %r to user ID for typing (no cache hit, resolution failed)", dm_email)
            # Do not fall back to email — Zulip rejects non-integer user IDs for direct typing
            return None
        return None

    def _resolve_stream_id(self, stream_name: str) -> Optional[int]:
        """Best-effort live lookup of stream name → ID using the Zulip client.

        Populates both caches on success. Used by the typing path so we don't
        depend on the background stream list having run yet (e.g. first typing
        indicator before any messages arrived in a stream).

        Follows the same cache + on-demand pattern as the user_id resolver
        for DM typing.
        """
        if not self._client or not stream_name:
            return None
        key = stream_name.lower()
        try:
            result = self._client.get_streams()
            for s in result.get("streams", []):
                if s.get("name", "").lower() == key:
                    sid = s["stream_id"]
                    self._stream_id_cache[key] = sid
                    self._stream_name_cache[sid] = s["name"]
                    logger.debug("Zulip: on-demand resolved stream %r -> id=%s for typing", stream_name, sid)
                    return sid
        except Exception as exc:
            logger.warning("Zulip: on-demand stream resolution failed for %r: %s", stream_name, exc)
        return None

    def _resolve_user_id(self, email: str) -> Optional[int]:
        """Best-effort lookup of user email → user_id for DM typing.

        Zulip's set_typing_status for direct messages (type="direct") *requires*
        integer user IDs in the "to" field — emails are rejected at the API
        level (this was the root cause of DM typing being completely silent).

        We primarily populate the cache from inbound "sender_id" in
        _dispatch_inbound (Zulip always sends numeric IDs in message events).
        This on-demand path (via the zulip client's get_user helper, which
        does a /users/{email} lookup) is the fallback for the first outbound
        typing indicator to a user we have never received a message from yet.
        """
        if not email or not self._client:
            return None
        key = email.lower()
        if key in self._user_id_cache:
            return self._user_id_cache[key]
        try:
            # Standard zulip client helper
            result = self._client.get_user(email=email)
            if result.get("result") == "success":
                user = result.get("user") or {}
                uid = user.get("user_id")
                if uid:
                    self._user_id_cache[key] = uid
                    logger.debug("Zulip: on-demand resolved user %r -> id=%s for typing", email, uid)
                    return uid
        except Exception as exc:
            logger.warning("Zulip: on-demand user resolution failed for %r: %s", email, exc)
        return None

    # ------------------------------------------------------------------
    # Internal: event queue
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal: missed-message catch-up (opt-in; see __init__)
    # ------------------------------------------------------------------

    def _catchup_watermark_path(self) -> Path:
        """Path to the persisted per-stream catch-up watermark file.

        Stored under HERMES_HOME so it survives restarts on the state volume;
        falls back to beside this module in dev/test environments.
        """
        try:
            from hermes_constants import get_hermes_home

            return get_hermes_home() / "zulip_catchup_watermarks.json"
        except Exception:
            return Path(__file__).parent / "zulip_catchup_watermarks.json"

    def _read_catchup_watermarks(self) -> dict[str, int]:
        """Load ``{stream_name: last_seen_msg_id}``; returns ``{}`` on any error."""
        try:
            data = json.loads(
                self._catchup_watermark_path().read_text(encoding="utf-8")
            )
            if isinstance(data, dict):
                return {
                    str(k): int(v)
                    for k, v in data.items()
                    if isinstance(v, (int, float)) and int(v) > 0
                }
        except Exception:
            pass
        return {}

    def _write_catchup_watermark(self, stream_name: str, msg_id: int) -> None:
        """Persist a stream's watermark monotonically (never moves backward)."""
        path = self._catchup_watermark_path()
        try:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    existing = {}
            except Exception:
                existing = {}
            if int(existing.get(stream_name, 0) or 0) >= msg_id:
                return
            existing[stream_name] = msg_id
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(existing), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:
            logger.warning(
                "Zulip: catch-up: failed to persist watermark for %r: %s",
                stream_name, exc,
            )

    @staticmethod
    def _stream_name_from_message(message: dict[str, Any]) -> str:
        """Lower-cased stream name from a stream message, else ``''``."""
        recipient = message.get("display_recipient")
        return recipient.lower() if isinstance(recipient, str) else ""

    def _advance_catchup_watermark(self, message: dict[str, Any]) -> None:
        """Move a stream's watermark forward as messages flow (live or replayed).

        Called from :meth:`_on_zulip_event` for every stream message so the next
        wrapper-level catch-up sweep resumes from a current position instead of
        re-fetching already-seen messages.  No-op when catch-up is disabled.
        SDK-internal queue replacement (``BAD_EVENT_QUEUE_ID``) does not run
        that sweep.
        """
        if not self._catchup_enabled:
            return
        if message.get("type") != "stream":
            return
        stream_name = self._stream_name_from_message(message)
        msg_id = int(message.get("id", 0) or 0)
        if stream_name and msg_id > 0:
            self._write_catchup_watermark(stream_name, msg_id)

    def _run_missed_message_catchup(self, generation: Optional[int] = None) -> None:
        """Back-fill messages that arrived while the wrapper queue was down.

        Runs synchronously in the event-queue thread immediately before our
        wrapper calls ``call_on_each_event``. It does **not** run when SDK 0.9.1
        replaces an expired queue inside that call. For each known stream:

        * **No stored watermark (first run):** record the newest message id as a
          baseline and back-fill nothing — a clean start never replays history.
        * **Stored watermark:** fetch up to ``catchup_max_messages`` messages
          after it and feed each through :meth:`_on_zulip_event` — the same path
          live events take, so ``_seen_events`` dedups any sweep/live overlap and
          mention-gating still applies.

        A message that arrives after the history snapshot and before queue
        registration can still be missed. Best-effort: any per-stream error is
        logged and skipped so a transient failure never blocks the queue from
        coming up.
        """
        live = self._event_generation if generation is None else generation
        if not self._reader_is_current(live):
            return
        if not self._client or not self._stream_id_cache:
            return
        if not self._loop or self._loop.is_closed():
            return

        watermarks = self._read_catchup_watermarks()
        send_client = self._build_send_client()

        for stream_name in sorted(self._stream_id_cache):
            if not self._reader_is_current(live):
                return
            watermark = watermarks.get(stream_name, 0)
            try:
                if watermark <= 0:
                    # First run for this stream — seed to newest, no back-fill.
                    result = send_client.get_messages({
                        "anchor": "newest",
                        "num_before": 1,
                        "num_after": 0,
                        "narrow": [["stream", stream_name]],
                        "apply_markdown": False,
                    })
                    if result.get("result") == "success":
                        msgs = result.get("messages", [])
                        newest = msgs[-1].get("id", 0) if msgs else 0
                        if newest > 0:
                            self._write_catchup_watermark(stream_name, newest)
                    continue

                result = send_client.get_messages({
                    "anchor": watermark + 1,
                    "num_before": 0,
                    "num_after": self._catchup_max_messages,
                    "narrow": [["stream", stream_name]],
                    "apply_markdown": False,
                })
            except Exception as exc:
                logger.warning(
                    "Zulip: catch-up: fetch failed for %r: %s", stream_name, exc
                )
                continue

            if result.get("result") != "success":
                continue

            replayed = 0
            for msg in result.get("messages", []):
                if not self._reader_is_current(live):
                    return
                if int(msg.get("id", 0) or 0) <= watermark:
                    continue  # anchor is inclusive — skip the watermark itself
                # Feed through the live event path: dedup, gating, dispatch, and
                # watermark advance all happen there, identical to a live event.
                self._on_zulip_event(
                    {"type": "message", "op": "add", "message": msg}
                )
                replayed += 1
            if replayed:
                logger.info(
                    "Zulip: catch-up: replayed %d missed message(s) on #%s",
                    replayed, stream_name,
                )

    def _run_event_queue(self) -> None:
        """Run the Zulip event queue in the current thread.

        Uses ``call_on_each_event`` which internally handles long-polling
        and basic reconnection.  Wraps with our own exponential backoff
        for the cases where the Zulip client's internal retry gives up.

        SDK 0.9.1 does not observe ``_closing``. Disconnect bumps
        ``_event_generation`` and closes the HTTP session so stale callbacks
        are dropped even if the SDK loop is still blocked in ``get_events``.

        The backoff sleep uses :pymeth:`threading.Event.wait` so that
        :meth:`disconnect` can wake the thread immediately instead of
        waiting for the full delay to elapse.
        """
        delay = _RECONNECT_BASE_DELAY
        self._consecutive_failures = 0
        generation = self._event_generation

        while not self._closing and generation == self._event_generation:
            # Opt-in catch-up before our wrapper (re-)enters the SDK loop.
            # Does not cover SDK-internal BAD_EVENT_QUEUE_ID re-register.
            if self._catchup_enabled:
                self._run_missed_message_catchup(generation)
                if not self._reader_is_current(generation):
                    return
            try:
                client = self._client
                if client is None:
                    return
                client.call_on_each_event(
                    lambda event, gen=generation: self._dispatch_zulip_event(event, gen),
                    event_types=["message"],
                    apply_markdown=False,
                )
                # ``call_on_each_event`` returned — server closed the
                # event queue stream or the client hit an internal limit.
                if not self._reader_is_current(generation):
                    return
                logger.info("Zulip: event queue stream ended — reconnecting")
                self._consecutive_failures = 0
                delay = _RECONNECT_BASE_DELAY
                continue
            except Exception as exc:
                if not self._reader_is_current(generation):
                    return

                self._consecutive_failures += 1
                retryable = _is_retryable_error(exc)

                if not retryable:
                    logger.error(
                        "Zulip: non-retryable error (attempt %d): %s — "
                        "stopping event queue",
                        self._consecutive_failures,
                        type(exc).__name__,
                    )
                    self._set_fatal_error(
                        "ZULIP_EVENT_QUEUE_FATAL",
                        f"Non-retryable error: {type(exc).__name__}: {exc}",
                        retryable=False,
                    )
                    return

                logger.warning(
                    "Zulip: event queue error (attempt %d): %s — "
                    "reconnecting in %.0fs",
                    self._consecutive_failures,
                    type(exc).__name__,
                    delay,
                )

            if not self._reader_is_current(generation):
                return

            # Exponential backoff with jitter.
            jitter = delay * _RECONNECT_JITTER * random.random()
            sleep_time = delay + jitter
            if self._consecutive_failures > 1:
                logger.info(
                    "Zulip: waiting %.1fs before reconnect attempt %d",
                    sleep_time,
                    self._consecutive_failures + 1,
                )
            if self._shutdown_event.wait(timeout=sleep_time):
                return  # Shutdown signal received during backoff.
            delay = min(delay * 2, _RECONNECT_MAX_DELAY)

    def _on_zulip_event(self, event: dict[str, Any]) -> None:
        """Callback invoked by ``call_on_each_event`` for each event.

        Runs in the event-queue thread.  Schedules the actual processing
        on the asyncio event loop via ``call_soon_threadsafe``.
        """
        if self._closing:
            return

        # Defense in depth: verify event shape.  The server-side filter
        # should only deliver "message" events, but validate anyway.
        event_type = event.get("type", "")
        if event_type != "message":
            logger.debug(
                "Zulip: ignoring non-message event (type=%s)",
                event_type,
            )
            return

        event_op = event.get("op", "add")
        if event_op != "add":
            # Edits/deletes come through as different event types or
            # ops — we only handle new-message creation.
            logger.debug(
                "Zulip: ignoring message event with op=%s",
                event_op,
            )
            return

        # Extract message payload.
        message = event.get("message")
        if not message or not isinstance(message, dict):
            return

        # Dedup by Zulip message ID.
        msg_id = str(message.get("id", ""))
        self._prune_seen()
        if msg_id and msg_id in self._seen_events:
            return
        if msg_id:
            self._seen_events[msg_id] = time.time()

        # Keep the catch-up watermark current as messages flow (no-op when
        # catch-up is disabled), so the next (re-)register resumes from here.
        self._advance_catchup_watermark(message)

        # Filter self-messages.
        sender_email = message.get("sender_email", "")
        sender_id = message.get("sender_id", -1)
        if sender_email == self._bot_email or sender_id == self._bot_user_id:
            return

        # Schedule async processing on the main event loop.
        msg_type_log = message.get("type", "unknown")
        logger.debug(
            "Zulip: inbound msg_id=%s sender=%s type=%s",
            msg_id,
            sender_email,
            msg_type_log,
        )
        if self._loop and not self._loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(
                self._dispatch_inbound(message, event), self._loop
            )
            self._event_futures.add(future)

            def _discard_done(done_future):
                self._event_futures.discard(done_future)
                if done_future.cancelled():
                    return
                try:
                    done_future.result()
                except Exception:
                    logger.exception("Zulip: inbound dispatch task failed")

            future.add_done_callback(_discard_done)

    async def _fetch_context(
        self, stream_name: str, topic: str
    ) -> list:
        """Fetch recent messages from a stream+topic via Zulip's /messages API.

        Returns a list of formatted context lines like ``["Alice: hello", ...]``,
        skipping the bot's own messages.  Runs the synchronous HTTP call in a
        thread executor so the event loop is never blocked.

        Returns an empty list on any failure — context is best-effort.
        """
        if self._context_depth <= 0:
            return []

        send_client = self._build_send_client()
        try:
            result = await asyncio.to_thread(
                send_client.get_messages,
                {
                    "anchor": "newest",
                    "num_before": self._context_depth,
                    "num_after": 0,
                    "narrow": [
                        ["stream", stream_name],
                        ["topic", topic],
                    ],
                    "apply_markdown": False,
                },
            )
        except Exception as exc:
            logger.warning("Zulip: context fetch failed — %s", exc)
            return []

        if result.get("result") != "success":
            logger.debug(
                "Zulip: context fetch error for #%s > %s — %s",
                stream_name, topic, result.get("msg", "unknown"),
            )
            return []

        messages = result.get("messages", [])
        context_lines = []
        for msg in reversed(messages):
            sender = msg.get("sender_full_name") or msg.get("sender_email", "?")
            content = (msg.get("content") or "").strip()
            # Skip the bot's own messages and empty content.
            if msg.get("sender_email") == self._bot_email:
                continue
            if not content:
                continue
            context_lines.append(f"{sender}: {content}")

        logger.debug(
            "Zulip: fetched %d context messages for #%s > %s (requested %d)",
            len(context_lines), stream_name, topic, self._context_depth,
        )
        return context_lines

    async def _fetch_inbound_uploads(
        self, content: str
    ) -> tuple[list[str], list[str]]:
        """Materialize all triggering-message uploads into the Hermes cache.

        Zulip sends files as Markdown links, not structured event attachments.
        The shared downloader obtains the temporary authenticated URL, caches
        images/audio/video/documents by MIME type, and returns a path the agent
        can access. A failed or oversized upload never drops the text message.
        """
        paths = extract_upload_paths(content)
        if not paths:
            return [], []

        local_paths: list[str] = []
        mime_types: list[str] = []
        for path in paths:
            try:
                downloaded = await download_zulip_upload(
                    site_url=self._site_url,
                    bot_email=self._bot_email,
                    api_key=self._api_key,
                    path=path,
                )
                local_paths.append(downloaded.path)
                mime_types.append(downloaded.mime_type)
            except Exception as exc:
                logger.warning("Zulip: failed to fetch upload %s — %s", path, exc)

        if local_paths:
            logger.info("Zulip: downloaded %d inbound attachment(s)", len(local_paths))
        return local_paths, mime_types

    async def _dispatch_inbound(self, message: dict[str, Any], raw_event: dict[str, Any]) -> None:
        """Process an inbound message on the asyncio event loop.

        For stream @mentions, fetches recent context from Zulip's /messages API
        and injects it before the user's message so the agent has full awareness.
        """

        # Determine message type and chat context.
        msg_type_name = message.get("type", "")  # "stream" or "private"
        content = message.get("content", "")
        sender_email = message.get("sender_email", "")
        sender_full_name = message.get("sender_full_name", "") or sender_email
        sender_id = message.get("sender_id", -1)
        msg_id = str(message.get("id", ""))

        # Reject whitespace-only content early (before type-specific logic).
        if not content or not content.strip():
            return

        if msg_type_name == "stream":
            stream_id = message.get("stream_id", -1)
            topic = message.get("subject") or "(no topic)"
            chat_id = _build_stream_chat_id(stream_id, topic)
            chat_type = "stream"
            chat_name = _resolve_stream_name(
                message, stream_id, self._stream_name_cache
            )
            chat_topic = topic
            user_id = sender_email
            user_name = sender_full_name

            # Picker clicks are plain messages and usually omit @mention.
            # Consume them before the mention gate so stream topics still work.
            if await self._consume_model_picker_reply(chat_id, content, sender_email):
                return

            # Check for @mention of the bot in stream messages.
            # DMs are always processed.
            mention_patterns = [
                f"@**{self._bot_full_name}**",
                f"@{self._bot_email}",
                # Zulip wildcard mentions that should wake the bot.
                "@**all**",
                "@**everyone**",
            ]

            # Determine if this stream requires a mention.
            require_mention = self._require_mention
            if require_mention and self._free_response_streams:
                # Check by stream name or stream ID.
                stream_name_lower = chat_name.lower()
                stream_id_str = str(stream_id)
                if (stream_name_lower in self._free_response_streams
                        or stream_id_str in self._free_response_streams):
                    require_mention = False

            if require_mention:
                has_mention = any(
                    pattern.lower() in content.lower()
                    for pattern in mention_patterns
                )
                if not has_mention:
                    logger.debug(
                        "Zulip: skipping stream message without @mention "
                        "(stream=%s, topic=%s)",
                        chat_name,
                        topic,
                    )
                    return

            # Fetch historical context from the same stream+topic via Zulip's
            # /messages API.  The bot gets full awareness of the conversation
            # even for messages that arrived while it was disconnected.
            if self._context_depth > 0:
                context = await self._fetch_context(chat_name, topic)
                if context:
                    content = _format_context_block(context) + content

            # Strip the bot mention from content so the agent sees
            # only the user's actual message (follows Slack/Discord pattern).
            bot_mention_only = [
                f"@**{self._bot_full_name}**",
                f"@{self._bot_email}",
            ]
            content = _strip_bot_mention(content, bot_mention_only)
        elif msg_type_name == "private":
            display_recipient = message.get("display_recipient")
            recipients = _extract_dm_recipients(
                display_recipient, self._bot_email, sender_email
            )

            if len(recipients) > 1:
                # Group DM (3+ original participants including bot).
                chat_id = _build_group_dm_chat_id(recipients)
                chat_type = "group"
                chat_name = ", ".join(recipients)
            else:
                # 1:1 DM.
                chat_id = _build_dm_chat_id(recipients[0] if recipients else sender_email)
                chat_type = "dm"
                chat_name = recipients[0] if recipients else sender_email

            chat_topic = None
            user_id = sender_email
            user_name = sender_full_name

            if await self._consume_model_picker_reply(chat_id, content, sender_email):
                return

            # Cache sender email -> user_id for DM typing.
            # Zulip typing for DMs requires the integer IDs (see _user_id_cache
            # declaration and _resolve_user_id for the full rationale).
            if sender_id > 0 and sender_email:
                self._user_id_cache[sender_email.lower()] = sender_id
        else:
            logger.debug("Zulip: ignoring message of type '%s'", msg_type_name)
            return

        # Determine message_type.
        mt = MessageType.TEXT
        if content.startswith("/") or content.startswith("!"):
            mt = MessageType.COMMAND

        # Reply-to detection (Zulip uses top-level reply metadata).
        reply_to_id = None
        # The Zulip event includes the message we're replying to in some
        # cases — but for now we handle outbound replies in send() via
        # the reply_to parameter.

        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=user_id,
            user_name=user_name,
            chat_topic=chat_topic,
        )

        # Download uploads from the TRIGGERING message. Parsed from raw message
        # content, not context-prepended text, so historical context stays as
        # text links until the agent explicitly requests an attachment tool.
        media_urls: list[str] = []
        media_types: list[str] = []
        raw_content = message.get("content") or ""
        # Do not fetch arbitrary files from users who will later fail the
        # gateway authorization gate. This mirrors adapters that treat
        # third-party context as untrusted and avoids spending bandwidth/cache
        # space before the message reaches the runner.
        sender_authorized = self._is_sender_authorized(
            user_id, chat_type, chat_id,
        )
        if sender_authorized is False:
            logger.debug("Zulip: skipping uploads from unauthorized sender %s", user_id)
        elif "/user_uploads/" in raw_content:
            media_urls, media_types = await self._fetch_inbound_uploads(
                raw_content
            )

        msg_event = MessageEvent(
            text=content,
            message_type=mt,
            source=source,
            raw_message=raw_event,
            message_id=msg_id,
            reply_to_message_id=reply_to_id,
            media_urls=media_urls,
            media_types=media_types,
        )

        # Schedule the handler coroutine on the event loop.
        asyncio.ensure_future(self.handle_message(msg_event))

    # ------------------------------------------------------------------
    # Internal: caches & helpers
    # ------------------------------------------------------------------

    def _refresh_stream_cache(self) -> None:
        """Fetch all streams and cache name ↔ ID mappings."""
        if not self._client:
            return
        try:
            result = self._client.get_streams()
            if result.get("result") == "success":
                for stream in result.get("streams", []):
                    sid = stream.get("stream_id")
                    name = stream.get("name", "")
                    if sid is not None and name:
                        self._stream_id_cache[name.lower()] = sid
                        self._stream_name_cache[sid] = name
                logger.info(
                    "Zulip: cached %d streams", len(self._stream_id_cache)
                )
        except Exception as exc:
            logger.warning("Zulip: failed to fetch streams — %s", exc)

    def _prune_seen(self) -> None:
        """Remove expired entries from the dedup cache."""
        if len(self._seen_events) < self._SEEN_MAX:
            return
        now = time.time()
        self._seen_events = {
            eid: ts
            for eid, ts in self._seen_events.items()
            if now - ts < self._SEEN_TTL
        }


def _zulip_configured(config: PlatformConfig) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(
        extra.get("site_url")
        and extra.get("bot_email")
        and (
            getattr(config, "token", None)
            or getattr(config, "api_key", None)
            or os.getenv("ZULIP_API_KEY", "").strip()
        )
    )


def _env_enablement() -> Optional[dict]:
    """Seed PlatformConfig.extra/home_channel from ZULIP_* env vars."""
    api_key = os.getenv("ZULIP_API_KEY", "").strip()
    bot_email = os.getenv("ZULIP_BOT_EMAIL", "").strip()
    site_url = os.getenv("ZULIP_SITE_URL", "").strip()
    default_stream = os.getenv("ZULIP_DEFAULT_STREAM", "").strip()
    home_topic = os.getenv("ZULIP_HOME_TOPIC", "").strip()
    home_channel = os.getenv("ZULIP_HOME_CHANNEL", "").strip()

    for legacy, value in (("ZULIP_DEFAULT_STREAM", default_stream), ("ZULIP_HOME_TOPIC", home_topic)):
        if value:
            logger.warning(
                "%s is deprecated and will be removed in a future release; "
                "set ZULIP_HOME_CHANNEL=<stream>:<topic> instead",
                legacy,
            )

    if not any([api_key, bot_email, site_url, default_stream, home_topic, home_channel]):
        return None

    if api_key and not bot_email:
        logger.warning("ZULIP_API_KEY set but ZULIP_BOT_EMAIL is missing")
    if api_key and not site_url:
        logger.warning("ZULIP_API_KEY set but ZULIP_SITE_URL is missing")

    seed: dict[str, Any] = {
        "site_url": site_url,
        "bot_email": bot_email,
    }
    if default_stream:
        seed["default_stream"] = default_stream
    if home_topic:
        seed["home_topic"] = home_topic
    catchup = os.getenv("ZULIP_CATCHUP", "").strip()
    if catchup:
        seed["catchup_enabled"] = catchup
    catchup_max = os.getenv("ZULIP_CATCHUP_MAX_MESSAGES", "").strip()
    if catchup_max:
        seed["catchup_max_messages"] = catchup_max

    home_chat_id = home_channel
    if not home_chat_id and default_stream and home_topic:
        home_chat_id = default_stream
    if home_topic and home_chat_id and ":" not in home_chat_id:
        home_chat_id = f"{home_chat_id}:{home_topic}"
    if home_chat_id:
        seed["home_channel"] = {
            "chat_id": home_chat_id,
            "name": os.getenv("ZULIP_HOME_CHANNEL_NAME", "Home"),
        }

    return seed


def _apply_yaml_config(yaml_cfg: dict, zulip_cfg: dict) -> Optional[dict]:
    """Translate config.yaml ``zulip:`` keys into env vars/extras.

    Env vars keep precedence. Returned values are merged into
    ``PlatformConfig.extra`` by the gateway config loader.
    """
    extra: dict[str, Any] = {}

    def _string_key(key: str, env_name: str) -> None:
        value = zulip_cfg.get(key)
        if value is None:
            return
        text = str(value).strip()
        if text and not os.getenv(env_name):
            os.environ[env_name] = text
        if text and key not in {"api_key", "token"}:
            extra[key] = text

    _string_key("site_url", "ZULIP_SITE_URL")
    _string_key("bot_email", "ZULIP_BOT_EMAIL")
    # Accept either key name in YAML; PlatformConfig.from_dict handles token
    # natively, while this env bridge lets the adapter and search tool continue
    # to work in env-driven paths.
    _string_key("api_key", "ZULIP_API_KEY")
    _string_key("token", "ZULIP_API_KEY")
    _string_key("default_stream", "ZULIP_DEFAULT_STREAM")
    _string_key("home_topic", "ZULIP_HOME_TOPIC")
    if extra.get("default_stream") or extra.get("home_topic"):
        logger.warning(
            "zulip: default_stream/home_topic are deprecated and will be removed in a "
            "future release; set home_channel: \"<stream>:<topic>\" instead"
        )
    _string_key("cert_bundle", "ZULIP_CERT_BUNDLE")
    _string_key("allow_insecure", "ZULIP_ALLOW_INSECURE")
    _string_key("require_mention", "ZULIP_REQUIRE_MENTION")
    _string_key("free_response_streams", "ZULIP_FREE_RESPONSE_STREAMS")
    _string_key("context_depth", "ZULIP_CONTEXT_DEPTH")
    _string_key("catchup_enabled", "ZULIP_CATCHUP")
    _string_key("catchup_max_messages", "ZULIP_CATCHUP_MAX_MESSAGES")

    return extra or None


def _iter_standalone_media(media_files) -> list[tuple[str, bool]]:
    """Normalize standalone media_files to (path, is_voice) pairs."""
    items: list[tuple[str, bool]] = []
    for item in media_files or []:
        if isinstance(item, (tuple, list)) and item:
            path = str(item[0] or "").strip()
            is_voice = bool(item[1]) if len(item) > 1 else False
        elif item:
            path = str(item).strip()
            is_voice = False
        else:
            continue
        if path:
            items.append((path, is_voice))
    return items


async def _standalone_send_zulip(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list] = None,
    force_document: bool = False,
) -> dict:
    """Send a Zulip message from out-of-process callers such as cron.

    Text uses ``adapter.send``. Attachments reuse the live adapter upload
    helpers (``send_image_file`` / ``send_video`` / ``send_document``),
    which talk to Zulip via a throwaway ``_build_send_client()`` — no
    event-queue connection required. A media failure is reported in
    ``warnings`` so the digest still lands.
    """
    if not check_zulip_requirements(pconfig):
        return {
            "error": (
                "Zulip SDK or credentials are unavailable. Configure "
                "ZULIP_SITE_URL, ZULIP_BOT_EMAIL, and ZULIP_API_KEY, or "
                "preinstall with: pip install 'hermes-agent[zulip]'"
            )
        }

    adapter = ZulipAdapter(pconfig)
    # ``send()`` treats ``_client is not None`` as the connected guard. Standalone
    # sends do not run the long-polling event queue, so mark the adapter ready
    # without sharing a live event client. Uploads use ``_build_send_client()``.
    adapter._client = object()
    metadata = {"thread_id": thread_id} if thread_id else None
    media_items = _iter_standalone_media(media_files)
    text = (message or "").strip()
    warnings: list[str] = []
    last_message_id = None
    delivered = False

    if text or not media_items:
        result = await adapter.send(chat_id, message, metadata=metadata)
        if not result.success:
            return {"error": result.error or "Zulip send failed"}
        last_message_id = result.message_id
        delivered = True

    for path, _is_voice in media_items:
        ext = Path(path).suffix.lower()
        try:
            if not force_document and ext in _STANDALONE_IMAGE_EXTS:
                result = await adapter.send_image_file(
                    chat_id, path, metadata=metadata,
                )
            elif not force_document and ext in _STANDALONE_VIDEO_EXTS:
                result = await adapter.send_video(
                    chat_id, path, metadata=metadata,
                )
            else:
                result = await adapter.send_document(
                    chat_id, path, metadata=metadata,
                )
        except Exception as exc:
            warning = f"Failed to send media {path}: {exc}"
            logger.error("Zulip standalone: %s", warning, exc_info=True)
            warnings.append(warning)
            continue
        if result.success:
            last_message_id = result.message_id or last_message_id
            delivered = True
        else:
            warnings.append(
                f"Failed to send media {path}: {result.error or 'upload failed'}"
            )

    if not delivered:
        error = "No deliverable text or media remained after processing"
        if warnings:
            return {"error": error, "warnings": warnings}
        return {"error": error}

    payload = {
        "success": True,
        "platform": "zulip",
        "chat_id": chat_id,
        "message_id": last_message_id,
    }
    if warnings:
        payload["warnings"] = warnings
    return payload


def interactive_setup() -> None:
    """Interactive ``hermes gateway setup`` flow for Zulip."""
    print()
    print("Zulip setup")
    print("-----------")
    print("Create a Zulip bot, then copy its site URL, bot email, and API key.")
    print()

    try:
        from hermes_cli.config import get_env_value, save_env_value
    except ImportError:
        print("hermes_cli.config not available; set ZULIP_* vars manually in ~/.hermes/.env")
        return

    def _prompt(var: str, label: str, *, secret: bool = False) -> None:
        existing = get_env_value(var) if callable(get_env_value) else None
        suffix = " [keep current]" if existing else ""
        try:
            if secret:
                from hermes_cli.secret_prompt import masked_secret_prompt
                value = masked_secret_prompt(f"{label}{suffix}: ")
            else:
                value = input(f"{label}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value:
            save_env_value(var, value)

    _prompt("ZULIP_SITE_URL", "Zulip site URL")
    _prompt("ZULIP_BOT_EMAIL", "Bot email")
    _prompt("ZULIP_API_KEY", "Bot API key", secret=True)
    _prompt("ZULIP_DEFAULT_STREAM", "Default stream (optional)")
    _prompt("ZULIP_HOME_TOPIC", "Home topic for cron delivery (optional)")
    _prompt("ZULIP_HOME_CHANNEL", "Home stream or stream:topic (optional)")
    _prompt("ZULIP_ALLOWED_USERS", "Allowed user emails (comma-separated; blank=skip)")
    print("Done. Run `hermes gateway status` to verify Zulip configuration.")


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="zulip",
        label="Zulip",
        adapter_factory=lambda cfg: ZulipAdapter(cfg),
        check_fn=check_zulip_requirements,
        validate_config=_zulip_configured,
        is_connected=_zulip_configured,
        required_env=["ZULIP_SITE_URL", "ZULIP_BOT_EMAIL", "ZULIP_API_KEY"],
        install_hint="pip install 'hermes-agent[zulip]'",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="ZULIP_ALLOWED_USERS",
        allow_all_env="ZULIP_ALLOW_ALL_USERS",
        cron_deliver_env_var="ZULIP_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send_zulip,
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="Z",
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Zulip. Zulip supports Markdown, code blocks, "
            "tables, LaTeX math (KaTeX), image links, streams, topics, direct "
            "messages, and group DMs. To send images or files, include "
            "MEDIA:<path> in your response (for example "
            "MEDIA:/absolute/path/to/image.png). For LaTeX: prefer $$y = x$$ on "
            "one line for inline math and a fenced math block for display:\n"
            "```math\nE = mc^2\n```\n"
            "(traditional $...$, \\(...\\), \\[...\\], and multi-line $$...$$ "
            "are also accepted and converted automatically). "
            "In streams, stay on the current topic unless the user explicitly "
            "asks to move elsewhere."
        ),
    )

    from .search_tool import register_zulip_search_tool
    from .attachments import register_zulip_attachment_tool
    from .topic_tool import register_zulip_topic_message_tool

    register_zulip_search_tool(ctx)
    register_zulip_attachment_tool(ctx)
    register_zulip_topic_message_tool(ctx)
