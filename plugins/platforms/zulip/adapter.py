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
"""

from __future__ import annotations

import asyncio
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


# Zulip KaTeX markup:
#   * Inline math:  $$O(n^2)$$
#   * Display math: fenced ```math ... ``` blocks
#
# Models often emit GitHub-style multi-line $$...$$ display math.  That is
# NOT display math in Zulip ($$ is only the inline delimiter), so we rewrite
# those blocks to Zulip math fences before send.
_DISPLAY_DOLLAR_MATH_RE = re.compile(
    r"(?ms)^[ \t]*\$\$[ \t]*\r?\n(.*?)\r?\n[ \t]*\$\$[ \t]*$",
)


def _normalize_zulip_math(content: str) -> str:
    """Rewrite multi-line ``$$...$$`` display blocks as Zulip ``math`` fences.

    Leaves single-line inline ``$$...$$`` alone, and does not touch content
    already inside fenced code blocks (including existing ``math`` fences).
    """
    if not content or "$$" not in content:
        return content

    parts = re.split(r"(```[\s\S]*?```)", content)
    out: List[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            # Already a fenced code/math block — leave untouched.
            out.append(part)
            continue

        def _to_math_fence(match: re.Match) -> str:
            body = match.group(1).rstrip("\n")
            return f"```math\n{body}\n```"

        out.append(_DISPLAY_DOLLAR_MATH_RE.sub(_to_math_fence, part))
    return "".join(out)



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

def _extract_upload_image_paths(content: str) -> List[str]:
    """Return unique ``/user_uploads/`` image paths from *content*, in order.

    Absolute URLs are normalized down to their ``/user_uploads/...`` path so
    the download always targets the configured site (never a foreign host
    smuggled into a markdown link).
    """
    paths: List[str] = []
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


def _build_zform_widget_content(heading: str, choices: List[Dict[str, Any]]) -> str:
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


def _build_stream_chat_id(stream_id: int, topic: str) -> str:
    """Encode a stream message's origin as a stable chat ID.

    Format: ``"{stream_id}:{topic}"``
    """
    return f"{stream_id}:{topic}"


def _parse_stream_chat_id(chat_id: str) -> Optional[Tuple[int, str]]:
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


def _parse_stream_name_topic(chat_id: str) -> Optional[Tuple[str, str]]:
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
    if sender_email.startswith(_DM_PREFIX):
        sender_email = sender_email[len(_DM_PREFIX):]
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


def _build_stream_typing_request(stream_id: int, topic: str, op: str) -> Dict[str, Any]:
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
    message: Dict[str, Any],
    stream_id: int,
    stream_name_cache: Dict[int, str],
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
    mention_patterns: List[str],
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
        import zulip  # noqa: F401
        return True
    except ImportError:
        pass

    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("platform.zulip", prompt=False)
    except Exception:
        return False

    try:
        import zulip  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ZulipAdapter(BasePlatformAdapter):
    """Gateway adapter for Zulip (cloud or self-hosted)."""

    SUPPORTS_MESSAGE_EDITING = True

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
        self._default_stream: str = (
            config.extra.get("default_stream", "")
            or os.getenv("ZULIP_DEFAULT_STREAM", "")
        )
        self._home_topic: str = (
            config.extra.get("home_topic", "")
            or os.getenv("ZULIP_HOME_TOPIC", "")
        )
        self._cert_bundle: str = os.getenv("ZULIP_CERT_BUNDLE", "")
        self._allow_insecure: bool = os.getenv(
            "ZULIP_ALLOW_INSECURE", "false"
        ).lower() in ("true", "1", "yes")
        self._streaming_edits_warning_logged = False

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
        # onward, so any message that arrives while the gateway is down
        # (process restart, BAD_EVENT_QUEUE_ID expiry, network drop) is never
        # seen by the bot.  When enabled, on every (re-)register the adapter
        # back-fills the gap for each known stream from a persisted per-stream
        # watermark and feeds the missed messages through the normal inbound
        # path — so dedup (``_seen_events``), mention-gating, and dispatch all
        # behave exactly as they do for live messages.
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

        # Background thread running the event queue.
        self._event_thread: Optional[threading.Thread] = None
        self._closing = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._event_futures: set[Any] = set()

        # Bot identity (resolved on connect)
        self._bot_user_id: int = -1
        self._bot_full_name: str = ""

        # Dedup cache: event_id → timestamp
        self._seen_events: Dict[str, float] = {}
        self._SEEN_MAX = 2000
        self._SEEN_TTL = 300  # 5 minutes

        # Stream name → stream_id cache (populated on connect)
        self._stream_id_cache: Dict[str, int] = {}
        # stream_id → stream_name reverse cache
        self._stream_name_cache: Dict[int, str] = {}

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
        self._user_id_cache: Dict[str, int] = {}

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
        kwargs: Dict[str, Any] = {
            "site": self._site_url,
            "email": self._bot_email,
            "api_key": self._api_key,
        }
        if self._cert_bundle:
            kwargs["cert_bundle"] = self._cert_bundle
        if self._allow_insecure:
            kwargs["insecure"] = True
        return zulip.Client(**kwargs)

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

        import zulip

        # Create the synchronous Zulip client.
        client_kwargs: Dict[str, Any] = {
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
        self._shutdown_event.set()  # Wake up the event thread immediately.

        # Wait for the event-queue thread to exit.
        if self._event_thread and self._event_thread.is_alive():
            self._event_thread.join(timeout=10.0)

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
        metadata: Optional[Dict[str, Any]] = None,
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

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
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
        self, chat_id: str, metadata: Optional[Dict[str, Any]] = None
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
        self, chat_id: str, metadata: Optional[Dict[str, Any]] = None
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
        metadata: Optional[Dict[str, Any]] = None,
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

        Multi-line ``$$...$$`` blocks (common model output for display math)
        are rewritten to Zulip's ``math`` fenced-code form so KaTeX renders
        them.  Inline ``$$...$$`` is left unchanged.
        """
        return _normalize_zulip_math(content)

    def _metadata_adjusted_chat_id(
        self,
        chat_id: str,
        metadata: Optional[Dict[str, Any]] = None,
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
        choices: List[Dict[str, Any]],
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
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

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
        allow_permanent: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        """Render a dangerous-command approval prompt as Zulip zform buttons.

        Intended use:
            Called by the gateway's approval notifier when a tool command is
            blocked waiting for user approval.

        How it works:
            Zulip zform buttons do not deliver hidden callback payloads.  They
            send visible replies, so the zform choices are exactly the text
            commands the existing approval handler already accepts
            (``/approve``, optional ``/approve session`` / ``/approve always``,
            and ``/deny``).  ``session_key`` is accepted for interface parity
            with other rich-button adapters; the resolver remains session-scoped
            in ``tools.approval`` just like the plain-text fallback.

            The full command is sent as its own message first so long scripts
            stay readable.  The zform choices are then attached to a short
            follow-up that replies to that command message.
        """
        del session_key  # Interface parity; resolution is session-scoped upstream.

        if smart_denied:
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
            f"```\n{command}\n```\n"
            f"Reason: {description}{scope_note}"
        )
        command_result = await self.send(
            chat_id=chat_id,
            content=command_body,
            metadata=metadata,
        )
        if not command_result.success:
            return command_result

        text_choices = ["`/approve`"]
        choices: List[Dict[str, Any]] = [
            {"short_name": "Once", "long_name": "Approve once", "reply": "/approve"},
        ]
        if not smart_denied:
            text_choices.append("`/approve session`")
            choices.append(
                {
                    "short_name": "Session",
                    "long_name": "Approve this pattern for the session",
                    "reply": "/approve session",
                }
            )
            if allow_permanent:
                text_choices.append("`/approve always`")
                choices.append(
                    {
                        "short_name": "Always",
                        "long_name": "Approve this pattern permanently",
                        "reply": "/approve always",
                    }
                )
        text_choices.append("`/deny`")
        choices.append(
            {"short_name": "Deny", "long_name": "Deny and cancel the command", "reply": "/deny"}
        )

        if len(text_choices) == 2:
            choice_help = f"{text_choices[0]} or {text_choices[1]}"
        else:
            choice_help = ", ".join(text_choices[:-1]) + f", or {text_choices[-1]}"
        prompt_body = (
            "Use the buttons below, or reply with "
            f"{choice_help}."
        )
        heading = _format_approval_zform_heading(command, description)
        return await self._send_zform_choices(
            chat_id=chat_id,
            content=prompt_body,
            heading=heading,
            choices=choices,
            reply_to=command_result.message_id,
            metadata=metadata,
        )

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None,
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
        metadata: Optional[Dict[str, Any]] = None,
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

        mark_awaiting_text(clarify_id)
        choice_rows = [
            {
                "short_name": str(index),
                "long_name": str(choice),
                "reply": str(choice),
            }
            for index, choice in enumerate(choices, start=1)
        ]
        option_lines = "\n".join(
            f"  {index}. {choice}"
            for index, choice in enumerate(choices, start=1)
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
        metadata: Optional[Dict[str, Any]] = None,
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
    ) -> Optional[Dict[str, Any]]:
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

    def _read_catchup_watermarks(self) -> Dict[str, int]:
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
    def _stream_name_from_message(message: Dict[str, Any]) -> str:
        """Lower-cased stream name from a stream message, else ``''``."""
        recipient = message.get("display_recipient")
        return recipient.lower() if isinstance(recipient, str) else ""

    def _advance_catchup_watermark(self, message: Dict[str, Any]) -> None:
        """Move a stream's watermark forward as messages flow (live or replayed).

        Called from :meth:`_on_zulip_event` for every stream message so the next
        (re-)register resumes from a current position instead of re-fetching
        already-seen messages.  No-op when catch-up is disabled.
        """
        if not self._catchup_enabled:
            return
        if message.get("type") != "stream":
            return
        stream_name = self._stream_name_from_message(message)
        msg_id = int(message.get("id", 0) or 0)
        if stream_name and msg_id > 0:
            self._write_catchup_watermark(stream_name, msg_id)

    def _run_missed_message_catchup(self) -> None:
        """Back-fill messages that arrived while the event queue was down.

        Runs synchronously in the event-queue thread immediately before the
        live queue (re-)registers.  For each known stream:

        * **No stored watermark (first run):** record the newest message id as a
          baseline and back-fill nothing — a clean start never replays history.
        * **Stored watermark:** fetch up to ``catchup_max_messages`` messages
          after it and feed each through :meth:`_on_zulip_event` — the same path
          live events take, so ``_seen_events`` dedups any sweep/live overlap and
          mention-gating still applies.

        Best-effort: any per-stream error is logged and skipped so a transient
        failure never blocks the queue from coming up.
        """
        if not self._client or not self._stream_id_cache:
            return
        if not self._loop or self._loop.is_closed():
            return

        watermarks = self._read_catchup_watermarks()
        send_client = self._build_send_client()

        for stream_name in sorted(self._stream_id_cache):
            if self._closing:
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
                if self._closing:
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

        The backoff sleep uses :pymeth:`threading.Event.wait` so that
        :meth:`disconnect` can wake the thread immediately instead of
        waiting for the full delay to elapse.
        """
        delay = _RECONNECT_BASE_DELAY
        self._consecutive_failures = 0

        while not self._closing:
            # Back-fill messages missed while the queue was down (opt-in) before
            # going live.  No-op unless catch-up is enabled; runs on every
            # (re-)register so both boot and mid-run queue expiry are covered.
            if self._catchup_enabled:
                self._run_missed_message_catchup()
                if self._closing:
                    return
            try:
                self._client.call_on_each_event(
                    self._on_zulip_event,
                    event_types=["message"],
                    apply_markdown=False,
                )
                # ``call_on_each_event`` returned — server closed the
                # event queue stream or the client hit an internal limit.
                if self._closing:
                    return
                logger.info("Zulip: event queue stream ended — reconnecting")
                self._consecutive_failures = 0
                delay = _RECONNECT_BASE_DELAY
                continue
            except Exception as exc:
                if self._closing:
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

            if self._closing:
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

    def _on_zulip_event(self, event: Dict[str, Any]) -> None:
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
    ) -> Tuple[List[str], List[str]]:
        """Materialize all triggering-message uploads into the Hermes cache.

        Zulip sends files as Markdown links, not structured event attachments.
        The shared downloader obtains the temporary authenticated URL, caches
        images/audio/video/documents by MIME type, and returns a path the agent
        can access. A failed or oversized upload never drops the text message.
        """
        paths = extract_upload_paths(content)
        if not paths:
            return [], []

        local_paths: List[str] = []
        mime_types: List[str] = []
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

    async def _dispatch_inbound(self, message: Dict[str, Any], raw_event: Dict[str, Any]) -> None:
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

        track_only = False  # May be set to True in stream block below

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
        media_urls: List[str] = []
        media_types: List[str] = []
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
    _string_key("cert_bundle", "ZULIP_CERT_BUNDLE")
    _string_key("allow_insecure", "ZULIP_ALLOW_INSECURE")
    _string_key("require_mention", "ZULIP_REQUIRE_MENTION")
    _string_key("free_response_streams", "ZULIP_FREE_RESPONSE_STREAMS")
    _string_key("context_depth", "ZULIP_CONTEXT_DEPTH")
    _string_key("catchup_enabled", "ZULIP_CATCHUP")
    _string_key("catchup_max_messages", "ZULIP_CATCHUP_MAX_MESSAGES")

    return extra or None


async def _standalone_send_zulip(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> dict:
    """Send a Zulip message from out-of-process callers such as cron."""
    if media_files:
        return {
            "error": (
                "Zulip standalone sending does not support media attachments "
                "yet; run the gateway adapter in-process or send text only."
            )
        }

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
    # without sharing a live event client.
    adapter._client = object()
    metadata = {"thread_id": thread_id} if thread_id else None
    result = await adapter.send(chat_id, message, metadata=metadata)
    if result.success:
        return {
            "success": True,
            "platform": "zulip",
            "chat_id": chat_id,
            "message_id": result.message_id,
        }
    return {"error": result.error or "Zulip send failed"}


def interactive_setup() -> None:
    """Interactive ``hermes gateway setup`` flow for Zulip."""
    print()
    print("Zulip setup")
    print("-----------")
    print("Create a Zulip bot, then copy its site URL, bot email, and API key.")
    print()

    try:
        from hermes_cli.config import get_env_var, set_env_var
    except ImportError:
        print("hermes_cli.config not available; set ZULIP_* vars manually in ~/.hermes/.env")
        return

    def _prompt(var: str, label: str, *, secret: bool = False) -> None:
        existing = get_env_var(var) if callable(get_env_var) else None
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
            set_env_var(var, value)

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
            "MEDIA:/absolute/path/to/image.png). For LaTeX: use $$y = x$$ on "
            "one line for inline math; for display equations prefer a fenced "
            "math block:\n```math\nE = mc^2\n```\n"
            "(multi-line $$...$$ is also accepted and converted automatically). "
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
