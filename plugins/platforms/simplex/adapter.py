"""SimpleX Chat platform adapter (Hermes plugin).

Connects to a simplex-chat daemon running in WebSocket mode.
Inbound messages arrive via a persistent WebSocket connection.
Outbound messages use the same WebSocket with JSON commands.

This adapter ships as a Hermes platform plugin under
``plugins/platforms/simplex/``. The Hermes plugin loader scans the
directory at startup, calls ``register(ctx)``, and the platform
becomes available to ``gateway/run.py`` and ``tools/send_message_tool``
through the registry — no edits to core files are required.

SimpleX chat daemon setup:
    simplex-chat -p 5225          # start daemon on port 5225
    # or via Docker:
    # docker run -p 5225:5225 simplexchat/simplex-chat-cli -p 5225

Required environment variables:
    SIMPLEX_WS_URL             WebSocket URL of the daemon
                               (default: ws://127.0.0.1:5225)

Optional environment variables:
    SIMPLEX_ALLOWED_USERS      Comma-separated allowlist. Each entry may be
                               either a numeric contactId (stable across
                               renames; visible via `/contacts` in the CLI)
                               or a contact display name (what the SimpleX
                               UI shows). Both forms are accepted.
    SIMPLEX_ALLOW_ALL_USERS    Set 'true' to allow all contacts
    SIMPLEX_AUTO_ACCEPT        Set 'false' to disable contact-request auto-accept
                               (default: 'true')
    SIMPLEX_GROUP_ALLOWED      Comma-separated group IDs to monitor, or '*'
                               for any group. Omit to disable groups entirely.
    SIMPLEX_HOME_CHANNEL       Default contact/group ID for cron delivery
    SIMPLEX_HOME_CHANNEL_NAME  Human label for the home channel
    HERMES_SIMPLEX_TEXT_BATCH_DELAY
                               Quiet-period seconds (default: 0.8) used to
                               concatenate rapid-fire inbound text messages
                               into a single MessageEvent — same pattern as
                               Telegram's text batching.

The ``websockets`` Python package is imported lazily — the plugin is
discoverable and ``hermes setup`` can describe it even when websockets is
not installed. ``check_requirements()`` returns False until the package
is present, so the gateway will not attempt to instantiate the adapter.
"""

import asyncio
import base64
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Lazy import: BasePlatformAdapter and friends live in the main repo.
# Imported at module top because they're stdlib-only inside Hermes — no
# external dependency that would block the plugin from loading.
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH = 8000  # SimpleX has no hard limit; chunk for sanity
WS_RETRY_DELAY_INITIAL = 2.0
WS_RETRY_DELAY_MAX = 60.0
HEALTH_CHECK_INTERVAL = 30.0
HEALTH_CHECK_STALE_THRESHOLD = 300.0

# Correlation ID prefix for requests we send so we can ignore our own echoes.
_CORR_PREFIX = "hermes-"

# Fallback tap-window for an approval prompt, used only when the core
# approval config cannot be read. The live value comes from
# ``tools.approval._get_approval_timeout()`` — the two timers are
# independent, and a prompt that outlives the agent-side wait would collect
# taps that resolve nothing, so the adapter follows the operator's
# ``approvals.timeout`` instead of hardcoding its own.
APPROVAL_PROMPT_TTL_FALLBACK_SECONDS = 300.0

# Correlated-wait budget for the anchoring ``/_send``. gateway/run.py gives
# ``send_exec_approval`` 15 seconds before it abandons the button lane and
# posts its own plain-text prompt, so waiting the full 15 here leaves no
# headroom and a slow daemon double-posts the approval.
ANCHORED_SEND_TIMEOUT_SECONDS = 10.0

# Tap targets the bot places on its own approval prompt, in seeding order:
# (emoji, choice, label).
#
# Exactly three, because simplex-chat holds at most three reactions per
# sender per item — measured against v7.0.0.11, where seeding a fourth comes
# back as ``commandError: "too many reactions"``. It is a count cap, not an
# emoji filter (removing one frees a slot), and it is per *sender*, so a user
# can always still add their own reaction to an item the bot has filled.
#
# Deny is first on purpose: whichever target is seeded last is the one a cap
# drops, and "I refuse" is the choice a security prompt must never lose.
#
# The emoji are NOT the Matrix adapter's ✅/🌀/♾️/❌ set. The simplex-chat
# daemon validates reaction emoji against a fixed list (``mrEmojiChar`` in
# ``src/Simplex/Chat/Protocol.hs`` accepts only 👍👎😀😂😢❤🚀✅ and rejects
# everything else), so 🌀/♾️/❌ would come back as command errors.
_APPROVAL_CHOICES: Tuple[Tuple[str, str, str], ...] = (
    ("👎", "deny", "deny"),
    ("✅", "once", "approve once"),
    ("🚀", "session", "approve for this session"),
)

_APPROVAL_REACTION_MAP: Dict[str, str] = {
    emoji: choice for emoji, choice, _label in _APPROVAL_CHOICES
}
# 👍 is not seeded but it is what people reach for first, and the gateway
# already reads a typed 👍 as approval.
_APPROVAL_REACTION_MAP["👍"] = "once"
# ❤ is approve-*always*, and it is deliberately typed-only: it writes a
# permanent, global, on-disk allowlist entry, which is the most consequential
# thing this prompt can do. A deliberate ``/approve always`` is the right cost
# for that tier, and it keeps the three tap slots for the tiers that are
# scoped to the moment. Still honoured inbound, so a user who places ❤ by hand
# gets what they asked for.
_APPROVAL_REACTION_MAP["❤"] = "always"

# Outcomes of asking the daemon to place or remove one of our reactions.
_REACTION_ACCEPTED = "accepted"
_REACTION_REJECTED = "rejected"
_REACTION_CAPPED = "capped"
_REACTION_NO_ANSWER = "no-answer"

# What the daemon says when a sender already holds its maximum reactions on
# an item. Matched on the message text rather than a decoded error shape
# because the error envelope varies between daemon versions, and mistaking a
# cap for a refusal would drop the whole reaction lane.
_REACTION_CAP_MARKER = "too many reactions"

_APPROVAL_ACK: Dict[str, str] = {
    "once": "✅ Approved — running this once.",
    "session": "✅ Approved for this session.",
    "always": "✅ Approved permanently (added to the allowlist).",
    "deny": "👎 Denied — the command will not run.",
}

# Unicode variation selectors. ❤ arrives as U+2764 U+FE0F from most clients
# and as bare U+2764 from others; the daemon's own validator only accepts the
# bare code point. Normalising on the way in means one map entry per reaction
# instead of one entry per spelling.
_VARIATION_SELECTORS = ("\ufe0f", "\ufe0e")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_reaction_emoji(emoji: str) -> str:
    """Strip Unicode variation selectors from a reaction emoji."""
    out = str(emoji or "")
    for selector in _VARIATION_SELECTORS:
        out = out.replace(selector, "")
    return out


def _is_reaction_cap_error(resp: dict) -> bool:
    """True when a ``chatCmdError`` is the daemon's per-sender reaction cap.

    "You already hold three reactions here" and "this daemon refuses your
    reactions" arrive as the same response type and mean opposite things —
    one is a full slot, the other is the feature being unavailable — so they
    must not share an outcome.
    """
    try:
        blob = json.dumps(resp.get("chatError") or resp, ensure_ascii=False)
    except (TypeError, ValueError):
        blob = str(resp)
    return _REACTION_CAP_MARKER in blob.lower()


def _approval_prompt_ttl() -> float:
    """Seconds a reaction prompt stays tappable — the operator's own value.

    Reads ``approvals.timeout`` through ``tools.approval``. Lazy import for
    the same reason the resolve call is lazy: the plugin must load without
    the agent package present.
    """
    try:
        from tools.approval import _get_approval_timeout

        return float(_get_approval_timeout())
    except Exception:
        return APPROVAL_PROMPT_TTL_FALLBACK_SECONDS


def _parse_comma_list(value: str) -> List[str]:
    """Split a comma-separated string into a stripped list."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _redact_id(contact_id: str) -> str:
    """Redact a contact/group ID for logging."""
    if not contact_id:
        return "<none>"
    s = str(contact_id)
    if len(s) <= 4:
        return s
    return s[:2] + "**" + s[-2:]


def _guess_extension(data: bytes) -> str:
    """Guess file extension from magic bytes."""
    if data[:4] == b"\x89PNG":
        return ".png"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:4] == b"GIF8":
        return ".gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:4] == b"%PDF":
        return ".pdf"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return ".mp4"
    if data[:4] == b"OggS":
        return ".ogg"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return ".mp3"
    return ".bin"


def _is_image_ext(ext: str) -> bool:
    return ext.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _is_audio_ext(ext: str) -> bool:
    return ext.lower() in {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".opus"}


class _SimplexCommandSendError(Exception):
    """The command never reached the daemon — the WebSocket write failed.

    Distinct from "no reply came back". A timeout leaves the command
    possibly delivered; this means it certainly was not, so a caller must
    not report the message it carried as sent.
    """


@dataclass
class _SimplexApprovalPrompt:
    """Tracks a pending SimpleX reaction-based exec approval prompt.

    In-memory only, like every other adapter's approval state: a gateway
    restart orphans pending prompts, and the agent-side wait in
    ``tools/approval.py`` is what eventually times them out.
    """

    session_key: str
    chat_id: str
    chat_ref: str
    item_id: str
    # Tiers this particular prompt offered. A reaction for a tier that was
    # never on the table is refused rather than silently upgraded — the
    # emoji→choice map is global, the offer is not.
    choices: frozenset = field(default_factory=frozenset)
    expires_at: float = 0.0
    resolved: bool = False
    # Emoji the bot actually placed, so cleanup removes only its own.
    seeded_emoji: List[str] = field(default_factory=list)
    # One "that reaction isn't an option" reply per prompt: a group member
    # reacting playfully must not turn the bot into a spam amplifier.
    feedback_sent: bool = False


# ---------------------------------------------------------------------------
# SimpleX Adapter
# ---------------------------------------------------------------------------

class SimplexAdapter(BasePlatformAdapter):
    """SimpleX Chat adapter using the simplex-chat daemon WebSocket API.

    Instantiated by the ``adapter_factory`` passed to
    ``ctx.register_platform()`` in :func:`register`.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig, **kwargs):
        platform = Platform("simplex")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}
        self.ws_url = extra.get("ws_url", "ws://127.0.0.1:5225").rstrip("/")

        # Contact-request auto-accept (on by default — matches the way most
        # bot deployments expect to behave). Read from env first, then fall
        # back to the value seeded by ``_env_enablement``.
        env_auto = os.getenv("SIMPLEX_AUTO_ACCEPT")
        if env_auto is not None:
            self.auto_accept = env_auto.strip().lower() not in {"0", "false", "no", ""}
        else:
            self.auto_accept = bool(extra.get("auto_accept", True))

        # Group allowlist. Without ``SIMPLEX_GROUP_ALLOWED``, group messages
        # are ignored entirely (safer default — a bot in a group otherwise
        # processes every member's traffic). Use ``*`` to accept any group.
        group_allowed_str = os.getenv("SIMPLEX_GROUP_ALLOWED", "") or extra.get(
            "group_allowed", ""
        )
        self.group_allow_from = set(_parse_comma_list(group_allowed_str))

        # Running state
        self._ws = None  # websockets connection
        self._ws_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_ws_activity = 0.0

        # Track sent correlation IDs to filter echoes
        self._pending_corr_ids: set = set()
        self._max_pending_corr = 200

        # File transfers awaiting rcvFileComplete (keyed by fileId). Populated
        # when a newChatItems event carries an unfinished rcvFileTransfer,
        # consumed when the file finishes downloading.
        self._pending_file_transfers: Dict[int, dict] = {}

        # Correlation tracking for ``_send_command``. Separate from
        # ``_pending_corr_ids`` (which is the upstream cosmetic echo filter)
        # because we actually await responses to commands we send.
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._corr_counter = 0

        # Text message batching — concatenate rapid-fire messages into one
        # event before dispatching, mirroring Telegram's batching.
        self._text_batch_delay = float(
            os.getenv("HERMES_SIMPLEX_TEXT_BATCH_DELAY", "0.8")
        )
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}

        # Reaction-based dangerous-command approvals. Prompts are correlated
        # by the daemon's chat-item id: the id of the message the bot posted
        # is what an inbound reaction event points back at.
        self._approval_prompts_by_item: Dict[str, _SimplexApprovalPrompt] = {}
        self._approval_prompt_by_session: Dict[str, str] = {}
        # Sessions that piled up more than one approval, and the moment the
        # pile-up can be assumed drained. Answering a prompt by typing is
        # invisible to this adapter, so once a session's queue holds more
        # than one entry the only safe assumption is that it still does,
        # until the whole approval window lapses.
        self._typed_only_sessions: Dict[str, float] = {}
        # Sessions with a ``send_exec_approval`` currently awaiting its
        # prompt send, plus a per-session entry counter. Two calls for one
        # session can interleave at that await: both would pass the
        # single-live-prompt check before either registers. The in-flight
        # count lets the later entrant see the earlier one; the entry
        # counter lets the earlier entrant see the later one when it
        # resumes. asyncio interleaves only at awaits, so plain dict
        # mutation here is race-free.
        self._approval_inflight: Dict[str, int] = {}
        self._approval_entry_gen: Dict[str, int] = {}
        # Tri-state feature detection for daemon reaction support: None until
        # the first /_reaction is answered, then True/False. Never assumed —
        # older daemons and future emoji-policy changes both show up as an
        # explicit command error, which downgrades this adapter to the typed
        # /approve flow instead of breaking approvals.
        self._reactions_supported: Optional[bool] = None
        self._background_tasks: Set[asyncio.Task] = set()

        logger.info(
            "SimpleX adapter initialized: url=%s auto_accept=%s groups=%s",
            self.ws_url,
            self.auto_accept,
            "enabled" if self.group_allow_from else "disabled",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to the simplex-chat daemon and start the WebSocket listener."""
        try:
            import websockets  # noqa: F401
        except ImportError:
            logger.error(
                "SimpleX: 'websockets' package not installed. "
                "Run: pip install websockets"
            )
            return False

        if not self.ws_url:
            logger.error("SimpleX: SIMPLEX_WS_URL is required")
            return False

        # Quick connectivity check — try to open and immediately close
        try:
            import websockets as _wsclient
            async with _wsclient.connect(self.ws_url, open_timeout=10):
                pass
        except Exception as e:
            logger.error("SimpleX: cannot reach daemon at %s: %s", self.ws_url, e)
            return False

        self._running = True
        self._last_ws_activity = time.time()
        self._ws_task = asyncio.create_task(self._ws_listener())
        self._health_task = asyncio.create_task(self._health_monitor())

        if hasattr(self, "_mark_connected"):
            self._mark_connected()
        logger.info("SimpleX: connected to %s", self.ws_url)
        return True

    async def disconnect(self) -> None:
        """Stop WebSocket listener and clean up."""
        self._running = False

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        # Cancel pending text-batch flush timers
        for task in list(self._pending_text_batch_tasks.values()):
            if not task.done():
                task.cancel()
        self._pending_text_batch_tasks.clear()
        self._pending_text_batches.clear()

        # Cancel pending command futures
        for fut in self._pending_responses.values():
            if not fut.done():
                fut.cancel()
        self._pending_responses.clear()

        # Cancel reaction seed/cleanup tasks and drop approval prompt state.
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
        self._background_tasks.clear()
        self._approval_prompts_by_item.clear()
        self._approval_prompt_by_session.clear()
        self._typed_only_sessions.clear()
        self._approval_inflight.clear()
        self._approval_entry_gen.clear()

        if hasattr(self, "_mark_disconnected"):
            self._mark_disconnected()
        logger.info("SimpleX: disconnected")

    # ------------------------------------------------------------------
    # WebSocket listener
    # ------------------------------------------------------------------

    async def _ws_listener(self) -> None:
        """Maintain a persistent WebSocket connection to the daemon."""
        import websockets as _wsclient
        from websockets.exceptions import ConnectionClosed

        backoff = WS_RETRY_DELAY_INITIAL

        while self._running:
            try:
                logger.debug("SimpleX WS: connecting to %s", self.ws_url)
                async with _wsclient.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                ) as ws:
                    self._ws = ws
                    backoff = WS_RETRY_DELAY_INITIAL
                    self._last_ws_activity = time.time()
                    # A fresh connection may be a fresh (or upgraded) daemon,
                    # so re-probe reaction support instead of carrying a
                    # single earlier command error forward for the life of
                    # the process.
                    self._reactions_supported = None
                    logger.info("SimpleX WS: connected")

                    async for raw in ws:
                        if not self._running:
                            break
                        self._last_ws_activity = time.time()
                        try:
                            msg = json.loads(raw)
                            await self._handle_event(msg)
                        except json.JSONDecodeError:
                            logger.debug("SimpleX WS: invalid JSON: %.100s", raw)
                        except Exception:
                            logger.exception("SimpleX WS: error handling event")

            except asyncio.CancelledError:
                break
            except ConnectionClosed as e:
                if self._running:
                    logger.warning(
                        "SimpleX WS: connection closed: %s (reconnecting in %.0fs)",
                        e, backoff,
                    )
            except Exception as e:
                if self._running:
                    logger.warning(
                        "SimpleX WS: unexpected error: %s (reconnecting in %.0fs)",
                        e, backoff,
                    )
            finally:
                self._ws = None

            if self._running:
                jitter = backoff * 0.2 * random.random()
                await asyncio.sleep(backoff + jitter)
                backoff = min(backoff * 2, WS_RETRY_DELAY_MAX)

    # ------------------------------------------------------------------
    # Health monitor
    # ------------------------------------------------------------------

    async def _health_monitor(self) -> None:
        """Observe WebSocket idleness without reconnecting healthy quiet links.

        simplex-chat can legitimately stay application-silent for long periods
        when no messages arrive. The websockets client already sends protocol
        pings (see _ws_listener ping_interval/ping_timeout), so treating lack of
        chat events as a stale connection causes needless reconnect churn.
        """
        while self._running:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            if not self._running:
                break
            elapsed = time.time() - self._last_ws_activity
            if elapsed > HEALTH_CHECK_STALE_THRESHOLD:
                logger.debug("SimpleX: WS application-idle for %.0fs", elapsed)

    # ------------------------------------------------------------------
    # Inbound event handling
    # ------------------------------------------------------------------

    async def _handle_event(self, event: dict) -> None:
        """Dispatch a daemon event to the appropriate handler."""
        # simplex-chat WebSocket messages are usually shaped as:
        #   {"corrId": "...", "resp": {"type": "newChatItems", ...}}
        # Older/examples may put the response fields at top-level. Normalize
        # both forms before dispatching, otherwise inbound chatItems are lost.
        nested = event.get("resp")
        resp = nested if isinstance(nested, dict) else event
        corr_id = event.get("corrId")

        # Handle correlated responses (replies to our own commands)
        if corr_id and corr_id in self._pending_responses:
            fut = self._pending_responses.pop(corr_id)
            if not fut.done():
                fut.set_result(resp)
            return

        # Cosmetic echo filter: prefixed corrIds are ours but didn't make it
        # into _pending_responses (e.g. fire-and-forget).
        if corr_id and isinstance(corr_id, str) and corr_id.startswith(_CORR_PREFIX):
            self._pending_corr_ids.discard(corr_id)
            return

        resp_type = resp.get("type") or event.get("type", "")

        # Auto-accept contact requests
        if resp_type == "contactRequest" and self.auto_accept:
            contact_req = resp.get("contactRequest", {}) or {}
            contact_req_id = contact_req.get("contactRequestId")
            if contact_req_id is not None:
                logger.info(
                    "SimpleX: auto-accepting contact request %s",
                    _redact_id(str(contact_req_id)),
                )
                await self._send_command(f"/accept {contact_req_id}")
            return

        # Early file-descriptor ready: simplex fires this before newChatItems
        # for some file types (especially large files and voice messages
        # transferred via XFTP). Send /freceive immediately so the download
        # starts; the chat item arrives in a subsequent newChatItems event.
        if resp_type == "rcvFileDescrReady":
            rcv_file = resp.get("rcvFileTransfer", {}) or {}
            file_id = rcv_file.get("fileId") if isinstance(rcv_file, dict) else None
            if file_id is not None:
                logger.debug(
                    "SimpleX: rcvFileDescrReady for fileId=%s — sending /freceive",
                    file_id,
                )
                await self._send_fire_and_forget(f"/freceive {file_id}")
            return

        # New messages — simplex-chat sends "newChatItems" with an array
        if resp_type == "newChatItems":
            chat_items = resp.get("chatItems", []) or []
            if not isinstance(chat_items, list):
                chat_items = [chat_items]
            for item in chat_items:
                try:
                    await self._handle_chat_item(item)
                except Exception:
                    logger.exception("SimpleX: error processing chat item")
            return

        # Singular variant — some daemon versions emit this
        if resp_type == "newChatItem":
            try:
                await self._handle_chat_item(resp)
            except Exception:
                logger.exception("SimpleX: error processing chat item")
            return

        # File transfer completion — deliver any deferred chat item
        if resp_type == "rcvFileComplete":
            chat_item = resp.get("chatItem", {}) or {}
            chat_item_data = chat_item.get("chatItem", {}) or {}
            file_info = chat_item_data.get("file", {}) or {}
            file_id = file_info.get("fileId") if isinstance(file_info, dict) else None
            if file_id is not None and file_id in self._pending_file_transfers:
                pending = self._pending_file_transfers.pop(file_id)
                file_source = file_info.get("fileSource", {}) or {}
                file_path = (
                    file_source.get("filePath")
                    if isinstance(file_source, dict)
                    else None
                )
                if file_path:
                    pending_item_data = pending.get("chatItem", {}) or {}
                    pending_item_data.setdefault("file", {})["fileSource"] = {
                        "filePath": file_path
                    }
                    pending["chatItem"] = pending_item_data
                    try:
                        await self._handle_chat_item(pending)
                    except Exception:
                        logger.exception(
                            "SimpleX: error processing deferred file message"
                        )
            return

        # A contact or group member reacted to one of our messages — this is
        # how tap-to-approve reaches the approval queue.
        if resp_type == "chatItemReaction":
            try:
                await self._handle_reaction_event(resp)
            except Exception:
                logger.exception("SimpleX: error processing reaction event")
            return

        if resp_type:
            logger.debug("SimpleX: unhandled event type: %s", resp_type)

    async def _handle_chat_item(self, chat_item: dict) -> None:
        """Process a single chat item from a newChatItems event."""
        chat_info = chat_item.get("chatInfo", {}) or {}
        chat_item_data = chat_item.get("chatItem", {}) or {}

        chat_type = chat_info.get("type", "")

        meta = chat_item_data.get("meta", {}) or {}
        content = chat_item_data.get("content", {}) or {}
        msg_content = content.get("msgContent", {}) or {}

        # Filter out our own messages
        item_direction = chat_item_data.get("chatDir", {}) or {}
        direction_type = (
            item_direction.get("type", "") if isinstance(item_direction, dict) else ""
        )
        if direction_type in ("directSnd", "groupSnd"):
            return

        # Only process received messages
        content_type = content.get("type", "") if isinstance(content, dict) else ""
        if content_type != "rcvMsgContent":
            return

        # Text content
        text = ""
        msg_type_str = (
            msg_content.get("type", "") if isinstance(msg_content, dict) else ""
        )
        if msg_type_str in ("text", "file", "image", "voice", "link", "video"):
            text = msg_content.get("text", "")

        if not text and msg_type_str not in ("image", "file", "voice"):
            return

        # Sender + chat IDs
        sender_id = ""
        sender_name = ""
        chat_id = ""
        is_group = False

        if chat_type == "direct":
            contact = chat_info.get("contact", {}) or {}
            sender_id = str(contact.get("contactId", ""))
            sender_name = contact.get("localDisplayName", "") or contact.get(
                "profile", {}
            ).get("displayName", "")
            chat_id = sender_id
        elif chat_type == "group":
            group_info = chat_info.get("groupInfo", {}) or {}
            group_id = str(group_info.get("groupId", ""))
            chat_id = f"group:{group_id}"
            is_group = True

            member = item_direction.get("groupMember", {}) or {}
            sender_id = str(member.get("memberId", ""))
            sender_name = member.get("localDisplayName", "") or member.get(
                "memberProfile", {}
            ).get("displayName", "")

            # Group allowlist
            if self.group_allow_from:
                if (
                    "*" not in self.group_allow_from
                    and group_id not in self.group_allow_from
                ):
                    logger.debug(
                        "SimpleX: group %s not in allowlist",
                        _redact_id(group_id),
                    )
                    return
            else:
                logger.debug(
                    "SimpleX: ignoring group message (no SIMPLEX_GROUP_ALLOWED)"
                )
                return
        else:
            logger.debug("SimpleX: unhandled chat type: %s", chat_type)
            return

        if not sender_id:
            logger.debug("SimpleX: ignoring message with no sender")
            return

        # File / image / voice attachment handling. File info is at
        # chatItem.chatItem.file (sibling of meta, content, chatDir).
        media_urls: List[str] = []
        media_types: List[str] = []
        file_info = chat_item_data.get("file")

        if file_info and isinstance(file_info, dict):
            file_source = file_info.get("fileSource", {}) or {}
            file_path = (
                file_source.get("filePath")
                if isinstance(file_source, dict)
                else None
            )
            file_name = file_info.get("fileName", "")
            file_id = file_info.get("fileId")

            ext = ""
            if file_path:
                ext = Path(file_path).suffix.lower()
            if not ext and file_name:
                ext = Path(file_name).suffix.lower()

            # Voice notes typically arrive before the file finishes
            # downloading. Defer the message until rcvFileComplete fires.
            if not file_path and _is_audio_ext(ext) and file_id is not None:
                logger.info(
                    "SimpleX: voice file %d not yet received, accepting transfer",
                    file_id,
                )
                self._pending_file_transfers[file_id] = chat_item
                # Fire-and-forget: simplex-chat does not return a corrId reply
                # for /freceive, so awaiting one would block the event loop.
                await self._send_fire_and_forget(f"/freceive {file_id}")
                return

            if file_path:
                ext = Path(file_path).suffix.lower() or (
                    Path(file_name).suffix.lower() if file_name else ""
                )
                if _is_image_ext(ext):
                    media_urls.append(file_path)
                    media_types.append(f"image/{ext.lstrip('.')}")
                elif _is_audio_ext(ext):
                    media_urls.append(file_path)
                    media_types.append(f"audio/{ext.lstrip('.')}")
                else:
                    media_urls.append(file_path)
                    media_types.append("application/octet-stream")

        # Source
        chat_name = sender_name
        if is_group:
            group_info = chat_info.get("groupInfo", {}) or {}
            chat_name = group_info.get("localDisplayName", "") or group_info.get(
                "groupProfile", {}
            ).get("displayName", chat_id)

        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type="group" if is_group else "dm",
            user_id=sender_id,
            user_name=sender_name or sender_id,
        )

        # Message type
        msg_type = MessageType.TEXT
        if media_types:
            if any(mt.startswith("audio/") for mt in media_types):
                msg_type = MessageType.VOICE
            elif any(mt.startswith("image/") for mt in media_types):
                msg_type = MessageType.PHOTO
            else:
                # Catch-all: non-image/non-audio files (tagged
                # application/octet-stream above) are documents so run.py's
                # document-context injection surfaces the file to the agent.
                msg_type = MessageType.DOCUMENT

        # Timestamp
        ts_str = meta.get("itemTs") or meta.get("createdAt", "")
        try:
            if ts_str:
                timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            else:
                timestamp = datetime.now(tz=timezone.utc)
        except (ValueError, AttributeError):
            timestamp = datetime.now(tz=timezone.utc)

        msg_event = MessageEvent(
            source=source,
            text=text or "",
            message_type=msg_type,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
            raw_message=chat_item,
        )

        logger.debug(
            "SimpleX: message from %s in %s: %s",
            _redact_id(sender_id),
            chat_id[:20],
            (text or "")[:50],
        )

        # Batch consecutive text messages so the agent sees one combined
        # message instead of dropping earlier ones when the user pastes
        # several lines in quick succession.
        if msg_type == MessageType.TEXT and text:
            self._enqueue_text_event(msg_event)
        else:
            await self.handle_message(msg_event)

    # ------------------------------------------------------------------
    # Text message batching
    # ------------------------------------------------------------------

    def _text_batch_key(self, event: MessageEvent) -> str:
        """Session-scoped key for text message batching."""
        return f"{event.source.platform.value}:{event.source.chat_id}"

    def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Buffer a text event and reset the flush timer."""
        key = self._text_batch_key(event)
        existing = self._pending_text_batches.get(key)
        if existing is None:
            self._pending_text_batches[key] = event
        else:
            if event.text:
                existing.text = (
                    f"{existing.text}\n{event.text}" if existing.text else event.text
                )
            if event.media_urls:
                existing.media_urls.extend(event.media_urls)
                existing.media_types.extend(event.media_types)

        prior_task = self._pending_text_batch_tasks.get(key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_text_batch_tasks[key] = asyncio.create_task(
            self._flush_text_batch(key)
        )

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch the aggregated text."""
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(self._text_batch_delay)
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            logger.info(
                "[SimpleX] Flushing text batch %s (%d chars)",
                key,
                len(event.text or ""),
            )
            await self.handle_message(event)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    # ------------------------------------------------------------------
    # Command interface
    # ------------------------------------------------------------------

    def _make_corr_id(self) -> str:
        """Mint a new correlation ID and remember it for echo-filtering.

        We add every minted id to ``_pending_corr_ids`` so the inbound
        event loop can drop the daemon's echo of our own commands without
        ever invoking ``_handle_chat_item``. The set is bounded — when
        it grows past ``_max_pending_corr``, the oldest entries are
        evicted in a single sweep.
        """
        self._corr_counter += 1
        corr_id = f"{_CORR_PREFIX}{self._corr_counter}-{int(time.time() * 1000)}"
        self._pending_corr_ids.add(corr_id)
        if len(self._pending_corr_ids) > self._max_pending_corr:
            overflow = len(self._pending_corr_ids) - self._max_pending_corr
            for _ in range(overflow):
                try:
                    self._pending_corr_ids.pop()
                except KeyError:
                    break
        return corr_id

    async def _send_ws(self, payload: dict) -> None:
        """Fire-and-forget JSON payload write.

        Drops cleanly when the WebSocket is missing or already closed; the
        caller never has to handle reconnection — the ``_ws_listener``
        loop does that out of band.
        """
        ws = self._ws
        if not ws:
            logger.debug("SimpleX: WS send dropped (not connected)")
            return
        try:
            await ws.send(json.dumps(payload))
        except Exception as e:
            logger.warning("SimpleX: WS send error: %s", e)

    async def _send_command(
        self,
        command: str,
        timeout: float = 30.0,
        *,
        raise_on_send_error: bool = False,
    ) -> Optional[dict]:
        """Send a command and await the correlated response.

        ``None`` means "no answer came back", which by default covers both a
        reply timeout (the daemon very likely got the command) and a write
        that raised (it certainly did not). Callers that have to tell those
        apart — anything reporting delivery to a user — pass
        ``raise_on_send_error=True`` and get :class:`_SimplexCommandSendError`
        for the write failure instead of an indistinguishable ``None``.
        """
        ws = self._ws
        if not ws:
            logger.warning("SimpleX: command sent but WebSocket not connected")
            if raise_on_send_error:
                raise _SimplexCommandSendError("WebSocket not connected")
            return None

        corr_id = self._make_corr_id()
        payload = json.dumps({"corrId": corr_id, "cmd": command})

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_responses[corr_id] = fut

        try:
            try:
                await ws.send(payload)
            except Exception as e:
                logger.warning(
                    "SimpleX: command not sent: %s — %s", command[:50], e
                )
                if raise_on_send_error:
                    raise _SimplexCommandSendError(str(e)) from e
                return None
            result = await asyncio.wait_for(fut, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning("SimpleX: command timed out: %s", command[:50])
            return None
        except _SimplexCommandSendError:
            raise
        except Exception as e:
            logger.warning("SimpleX: command failed: %s — %s", command[:50], e)
            return None
        finally:
            self._pending_responses.pop(corr_id, None)

    async def _send_fire_and_forget(self, command: str) -> None:
        """Send a command without waiting for a correlated response.

        Use this for commands the daemon never sends a corrId reply for,
        such as ``/freceive``. Awaiting a corr-id reply on those would
        stall the event loop for the full command timeout.
        """
        corr_id = self._make_corr_id()
        await self._send_ws({"corrId": corr_id, "cmd": command})

    # ------------------------------------------------------------------
    # Outbound — text
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message.

        If *content* contains ``MEDIA:<path>`` tags (embedded by TTS / audio
        tools to signal file attachments), they are stripped from the text
        body and sent as native voice notes or documents.

        Groups use the structured ``/_send #<id> json [...]`` form
        because the bracket chat-command syntax (``#[<id>] text``) is
        parsed by the daemon as a display-name lookup, which silently
        drops when the group's display name isn't the literal ID. DMs
        use the simple ``@<id> text`` form which has always worked in
        production.

        The call is fire-and-forget at the WebSocket level: the daemon
        doesn't always return a corrId reply for chat commands, and
        waiting for one would serialise all outbound traffic behind a
        30-second timeout.
        """
        _voice_exts = {".ogg", ".mp3", ".wav", ".m4a", ".opus"}
        media_paths = re.findall(r"MEDIA:(\S+)", content)
        if media_paths:
            content = re.sub(r"MEDIA:\S+", "", content).strip()

        if content:
            corr_id = self._make_corr_id()
            # Structured form: addresses by ID, and json.dumps escapes
            # newlines + special chars correctly.  The bare @id text
            # syntax is unreliable for DMs — the daemon silently drops
            # messages when it cannot resolve the display name.
            composed = json.dumps(
                [{"msgContent": {"type": "text", "text": content}}]
            )
            if chat_id.startswith("group:"):
                cmd_str = f"/_send #{chat_id[6:]} json {composed}"
            else:
                cmd_str = f"/_send @{chat_id} json {composed}"

            await self._send_ws({"corrId": corr_id, "cmd": cmd_str})

        for path in media_paths:
            is_voice = os.path.splitext(path)[1].lower() in _voice_exts
            if is_voice:
                media_result = await self.send_voice(chat_id, path)
            else:
                media_result = await self.send_document(chat_id, path)
            if not media_result.success:
                return media_result

        return SendResult(success=True)

    # ------------------------------------------------------------------
    # Channel directory enumeration
    # ------------------------------------------------------------------

    async def list_channels(self) -> Optional[List[Dict[str, Any]]]:
        """Enumerate contacts and allowed groups for the channel directory.

        Called by ``gateway.channel_directory.build_channel_directory()``
        every refresh cycle. Uses the daemon's ``/contacts`` and ``/groups``
        commands over the live WebSocket. Returns ``None`` (not ``[]``) when
        the WebSocket is down so the directory falls back to session-history
        discovery instead of wiping previously known targets.

        Entry ``id`` values match the send-target formats the adapter
        accepts: bare contact display name for DMs (``simplex:<name>``) and
        ``group:<groupId>`` for groups (``simplex:group:<id>``).
        """
        if not self._ws:
            return None

        channels: List[Dict[str, Any]] = []

        resp = await self._send_command("/contacts", timeout=10.0)
        if resp is None:
            # Daemon unresponsive — keep whatever the directory already has.
            return None
        for contact in resp.get("contacts") or []:
            if not isinstance(contact, dict):
                continue
            contact_id = contact.get("contactId")
            name = (
                contact.get("localDisplayName", "")
                or (contact.get("profile", {}) or {}).get("displayName", "")
            )
            if contact_id is None and not name:
                continue
            channels.append({
                # Display name is what the DM send path (``@<name>``)
                # actually addresses; fall back to the numeric contactId.
                "id": str(name or contact_id),
                "name": str(name or contact_id),
                "type": "dm",
            })

        resp = await self._send_command("/groups", timeout=10.0)
        if resp is not None:
            for group in resp.get("groups") or []:
                # The daemon returns each group as either a groupInfo dict
                # or a [groupInfo, groupSummary] pair depending on version.
                if isinstance(group, list) and group:
                    group = group[0]
                if not isinstance(group, dict):
                    continue
                group_id = group.get("groupId")
                if group_id is None:
                    continue
                name = (
                    group.get("localDisplayName", "")
                    or (group.get("groupProfile", {}) or {}).get("displayName", "")
                    or str(group_id)
                )
                channels.append({
                    "id": f"group:{group_id}",
                    "name": str(name),
                    "type": "group",
                })

        return channels

    # ------------------------------------------------------------------
    # Reaction-based exec approvals
    # ------------------------------------------------------------------

    def _spawn_background(self, coro) -> None:
        """Run *coro* detached, holding a strong reference until it finishes."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    @staticmethod
    def _chat_ref(chat_id: str) -> Optional[str]:
        """Build a daemon ``ChatRef`` (``@<contactId>`` / ``#<groupId>``).

        Returns ``None`` for a direct chat addressed by display name. The
        ``/_send`` and ``/_reaction`` command forms take numeric ids, while
        ``list_channels`` deliberately emits display names for DMs because
        the plain ``@<name> text`` send form addresses by name. Callers
        degrade to the typed ``/approve`` flow rather than guess an id.
        """
        if not chat_id:
            return None
        if chat_id.startswith("group:"):
            group_id = chat_id[6:]
            return f"#{group_id}" if group_id.isdigit() else None
        return f"@{chat_id}" if chat_id.isdigit() else None

    @staticmethod
    def _is_group_chat(chat_id: str) -> bool:
        """True for a group chat id (``group:<id>``), False for a DM."""
        return str(chat_id or "").startswith("group:")

    async def _send_anchored_text(self, chat_ref: str, text: str) -> Tuple[Optional[str], bool]:
        """Send text via ``/_send`` and return ``(item_id, delivered)``.

        Unlike :meth:`send`, this waits for the correlated ``newChatItems``
        reply, because a reaction has to be anchored to the daemon's
        ``itemId`` for the message it decorates.

        ``delivered`` is False when the send was never attempted (no live
        WebSocket), when the write itself raised (the connection died between
        the check and the write), or when the daemon answered with an explicit
        error — all three mean the user will see nothing, so the caller must
        fail and let ``gateway/run.py`` post its own plain-text prompt. A
        *timeout* is different: the message is possibly delivered, so the
        caller reports success and skips the reactions rather than
        double-posting.
        """
        if not self._ws:
            logger.warning(
                "SimpleX: approval prompt not sent — WebSocket not connected"
            )
            return None, False
        composed = json.dumps([{"msgContent": {"type": "text", "text": text}}])
        try:
            resp = await self._send_command(
                f"/_send {chat_ref} json {composed}",
                timeout=ANCHORED_SEND_TIMEOUT_SECONDS,
                raise_on_send_error=True,
            )
        except _SimplexCommandSendError:
            # The write failed, so nothing was delivered. Reporting this as
            # sent would leave the user with no prompt at all for a command
            # still queued in core.
            logger.warning(
                "SimpleX: approval prompt not sent — WebSocket write failed"
            )
            return None, False
        if resp is None:
            return None, True
        if resp.get("type") == "chatCmdError":
            logger.warning("SimpleX: daemon rejected /_send for %s", chat_ref)
            return None, False
        for item in resp.get("chatItems") or []:
            if not isinstance(item, dict):
                continue
            meta = (item.get("chatItem") or {}).get("meta") or {}
            item_id = meta.get("itemId")
            if item_id is not None:
                return str(item_id), True
        return None, True

    async def _set_reaction(
        self,
        chat_ref: str,
        item_id: str,
        emoji: str,
        *,
        add: bool = True,
    ) -> str:
        """Add or remove one of the bot's own reactions on a chat item.

        Returns one of :data:`_REACTION_ACCEPTED`, :data:`_REACTION_CAPPED`
        (the sender already holds the daemon's maximum on this item),
        :data:`_REACTION_REJECTED` (an explicit command error — the
        feature-detection signal) or :data:`_REACTION_NO_ANSWER` (nothing
        came back: inconclusive, and deliberately not read as "reactions
        unsupported").
        """
        reaction = json.dumps({"type": "emoji", "emoji": emoji}, ensure_ascii=False)
        toggle = "on" if add else "off"
        resp = await self._send_command(
            f"/_reaction {chat_ref} {item_id} {toggle} {reaction}", timeout=10.0
        )
        if resp is None:
            return _REACTION_NO_ANSWER
        if resp.get("type") == "chatCmdError":
            if _is_reaction_cap_error(resp):
                return _REACTION_CAPPED
            logger.info(
                "SimpleX: daemon rejected reaction %s on item %s", emoji, item_id
            )
            return _REACTION_REJECTED
        return _REACTION_ACCEPTED

    @staticmethod
    def _approval_tiers(
        allow_permanent: bool,
        allow_session: bool,
        smart_denied: bool,
    ) -> frozenset:
        """The approval tiers this particular request actually offers.

        A smart-DENY owner override is one operation only — ``tools/approval``
        collapses any wider choice back to a single run — so the prompt must
        not offer session or permanent scope for it.
        """
        if smart_denied or not allow_session:
            return frozenset({"once", "deny"})
        if not allow_permanent:
            return frozenset({"once", "session", "deny"})
        return frozenset({"once", "session", "always", "deny"})

    @staticmethod
    def _seed_plan(tiers: frozenset) -> Tuple[Tuple[str, str, str], ...]:
        """Tap targets to place, in order, for an approval offering *tiers*.

        Always a subset of the three seedable choices: ``always`` has no tap
        target at all, so an approval that offers it still seeds three.
        """
        return tuple(c for c in _APPROVAL_CHOICES if c[1] in tiers)

    def _format_simplex_exec_approval(
        self,
        command: str,
        description: str,
        smart_denied: bool,
        tiers: frozenset,
    ) -> str:
        """Compose the approval prompt: shared core plus the typed commands.

        *tiers* is the full set this approval offers; the typed instructions
        list every one of them on every path, because typing is the lane that
        always works. The reaction legend is not here — it is sent afterwards
        from the taps that were actually placed (see
        :meth:`_seed_approval_reactions`), so the prompt cannot advertise a
        tap the user never received.
        """
        prefix = self.typed_command_prefix
        scope = ""
        if not smart_denied:
            if "session" in tiers:
                scope += (
                    f"Reply `{prefix}approve session` to approve this pattern "
                    "for the session, "
                )
            if "always" in tiers:
                scope += f"`{prefix}approve always` to approve permanently, "
        return (
            f"{self._format_exec_approval(command, description, smart_denied)}\n\n"
            f"{scope}Reply `{prefix}approve` to execute once, or "
            f"`{prefix}deny` to cancel."
        )

    @staticmethod
    def _format_tap_legend(landed: List[str]) -> str:
        """Explain the taps that are actually on the message, and only those."""
        labels = {emoji: label for emoji, _choice, label in _APPROVAL_CHOICES}
        legend = "\n".join(f"{emoji} = {labels[emoji]}" for emoji in landed)
        return "Or tap a reaction on the message above:\n" + legend

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
        allow_permanent: bool = True,
        allow_session: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        """Send an exec approval prompt, with a tap lane when it is unambiguous.

        Mirrors the Matrix adapter: one ordinary chat message carrying the
        shared approval text, with the bot pre-seeding the decision emoji so
        answering is a tap instead of a retyped command. The typed
        ``/approve`` / ``/deny`` instructions stay in the message on every
        path, so every fallback below still leaves a working flow.

        The tap lane is offered only when a tap can mean exactly one thing:
        a direct chat with exactly one live approval. Groups and concurrent
        approvals fall back to the typed lane — see the guards below.

        Every exit that does not end with a registered prompt holds the
        session to the typed lane for one approval window, because
        ``tools/approval`` has already queued the approval by the time this
        runs: an unregistered prompt is a pending command the single-prompt
        guard cannot see.

        *command* arrives already credential-redacted from ``gateway/run.py``
        — do not re-redact it and do not log it.
        """
        tiers = self._approval_tiers(allow_permanent, allow_session, smart_denied)

        # Housekeeping first, and before any early return: prompts nobody
        # ever answered would otherwise accumulate for the life of the
        # gateway and make the single-pending check below lie.
        self._sweep_expired_prompts()

        # resolve_gateway_approval(session_key, choice) is FIFO per session
        # (upstream #64001) — a tap cannot target a specific queue entry, so
        # a second live prompt in one session would let a tap on either
        # message answer the older command. Withdraw the older prompt and
        # send this one typed-only rather than offer an ambiguous tap.
        #
        # The session then stays typed-only for a full approval window: a
        # third prompt would find an empty prompt map while two unanswered
        # commands are still queued in core, and a tap on it would run the
        # oldest of them.
        now = time.monotonic()
        if self._retire_live_prompt_for_session(session_key) or (
            self._typed_only_sessions.get(session_key, 0.0) > now
        ):
            queued = True
        else:
            queued = False
        if self._approval_inflight.get(session_key, 0) > 0:
            # A prompt send for this session is already in flight — the
            # single-live-prompt check above cannot see it, because that
            # call has not registered yet. Two live tap prompts must never
            # exist, so the later entrant takes the typed lane and holds
            # the session there.
            self._mark_session_typed_only(session_key)
            queued = True
        self._approval_inflight[session_key] = (
            self._approval_inflight.get(session_key, 0) + 1
        )
        entry_gen = self._approval_entry_gen.get(session_key, 0) + 1
        self._approval_entry_gen[session_key] = entry_gen

        # Anything short of a registered prompt below leaves an approval
        # pending in core with no tappable message of its own, so the
        # session is held to the typed lane by the ``finally``. That covers
        # the WebSocket being down, the write raising, the daemon rejecting
        # or not answering the send, a group or display-name-addressed chat,
        # and an exception escaping into gateway/run.py's text fallback.
        registered = False
        try:
            chat_ref = self._chat_ref(chat_id)
            # v1 keeps the reaction lane to direct chats. In a group, any
            # member can react, and the only identity the daemon hands us for
            # a group reactor is one this adapter cannot yet tie to a verified
            # operator. Groups get the typed lane, which the gateway
            # authorizes properly.
            reaction_lane = (
                bool(chat_ref) and not self._is_group_chat(chat_id) and not queued
            )
            # Seeding is a separate capability from the inbound lane: a daemon
            # that refuses to let the bot place a reaction has not said
            # anything about reactions a *user* places.
            seed = reaction_lane and self._reactions_supported is not False

            text = self._format_simplex_exec_approval(
                command, description, smart_denied, tiers
            )
            if not reaction_lane:
                return await self._send_approval_message(
                    chat_id, text, metadata=metadata
                )

            item_id, delivered = await self._send_anchored_text(str(chat_ref), text)
            if not delivered:
                # Let gateway/run.py fall back to its own plain-text prompt.
                return SendResult(
                    success=False,
                    error="SimpleX daemon rejected the approval prompt",
                )
            if not item_id:
                # The prompt is out but unanchored; its typed instructions
                # work, and its reaction legend is a dud — nothing correlates
                # a tap on a message whose item id we never learned.
                return SendResult(success=True)

            prompt = _SimplexApprovalPrompt(
                session_key=session_key,
                chat_id=chat_id,
                chat_ref=str(chat_ref),
                item_id=item_id,
                choices=tiers,
                expires_at=time.monotonic() + _approval_prompt_ttl(),
            )
            self._approval_prompts_by_item[item_id] = prompt
            self._approval_prompt_by_session[session_key] = item_id
            registered = True

            if self._approval_entry_gen.get(session_key, 0) != entry_gen:
                # Another approval for this session entered while the prompt
                # send above was in flight. That entrant saw this call and
                # went typed-only — but this prompt was composed against a
                # queue that has since grown, so a tap on it could resolve a
                # different command than the one it shows. Withdraw it and
                # hold the session to the typed lane.
                self._retire_live_prompt_for_session(session_key)
                self._mark_session_typed_only(session_key)
                return SendResult(success=True, message_id=item_id)

            if seed:
                # Seed detached: every /_reaction waits on a correlated daemon
                # reply, and gateway/run.py only allows send_exec_approval 15
                # seconds before it abandons the button lane. Prompt state is
                # already registered, so a reaction that beats the seeding
                # still correlates. The legend follows from the same task,
                # once there is a placed-taps list to describe.
                self._spawn_background(
                    self._seed_approval_reactions(prompt, self._seed_plan(tiers))
                )
            return SendResult(success=True, message_id=item_id)
        finally:
            if not registered:
                self._mark_session_typed_only(session_key)
            remaining = self._approval_inflight.get(session_key, 0) - 1
            if remaining > 0:
                self._approval_inflight[session_key] = remaining
            else:
                # Last one out drops both entries: every overlap check
                # snapshots at entry and compares while in flight, so the
                # counter has no meaning once nobody is.
                self._approval_inflight.pop(session_key, None)
                self._approval_entry_gen.pop(session_key, None)

    async def _send_approval_message(
        self,
        chat_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an approval-flow message via the structured ``/_send`` form.

        Every notice this feature emits — the prompt itself on the typed
        path, the tap legend, the acknowledgement, superseded, expired,
        no-longer-pending and unusable-reaction replies — goes out through
        here rather than through :meth:`send`.

        :meth:`send` composes a DM as ``@<chat_id> <text>``, and the daemon
        parses ``@x`` as a *display-name* lookup: on any contact whose
        display name is not literally its numeric id it answers
        ``contactNotFound`` and nothing reaches the chat, while the send
        still reports success. (Verified against simplex-chat v7.0.0.11:
        ``@3 hi`` → ``chatCmdError/contactNotFound``, the structured form →
        ``newChatItems``. The media paths already use the structured form;
        only DM text is affected. That is a separate bug in ``send`` and is
        not fixed here.) A user who taps ✅ and sees no reply reads the tap
        as broken, so the approval flow must not depend on that branch.

        Chats with no numeric ``ChatRef`` — DMs addressed by display name —
        fall back to :meth:`send`, which is the form that addresses them
        correctly.
        """
        chat_ref = self._chat_ref(chat_id)
        if not chat_ref:
            return await self.send(chat_id, text, metadata=metadata)
        composed = json.dumps([{"msgContent": {"type": "text", "text": text}}])
        await self._send_fire_and_forget(f"/_send {chat_ref} json {composed}")
        return SendResult(success=True)

    def _mark_session_typed_only(self, session_key: str) -> None:
        """Refuse the tap lane for *session_key* for one approval window.

        ``tools/approval`` queues an approval entry *before* it notifies the
        adapter, so any ``send_exec_approval`` call that ends without a
        registered prompt still leaves a command pending in core — one the
        single-live-prompt guard cannot see and the user may never have been
        shown. A later prompt in the same session would then look
        unambiguous, and a tap on it would resolve the FIFO head: the older,
        unseen command. Holding the session typed-only until every entry that
        could still be queued has timed out is the same principle the
        pile-up latch applies, extended to the paths where nothing was sent.

        Never shortens an existing window.
        """
        deadline = time.monotonic() + _approval_prompt_ttl()
        if self._typed_only_sessions.get(session_key, 0.0) < deadline:
            self._typed_only_sessions[session_key] = deadline

    def _sweep_expired_prompts(self) -> None:
        """Retire every prompt whose tap window has closed."""
        now = time.monotonic()
        for prompt in list(self._approval_prompts_by_item.values()):
            if now > prompt.expires_at:
                self._retire_prompt(prompt)
        for session_key, deadline in list(self._typed_only_sessions.items()):
            if now > deadline:
                self._typed_only_sessions.pop(session_key, None)

    def _retire_live_prompt_for_session(self, session_key: str) -> bool:
        """Withdraw the session's live prompt, if any. True when one was live.

        Tells the user what happened: the message they are looking at now has
        no working tap targets, and the reason is not obvious from the chat.
        """
        item_id = self._approval_prompt_by_session.get(session_key)
        live = self._approval_prompts_by_item.get(item_id) if item_id else None
        if live is None or live.resolved:
            return False
        prefix = self.typed_command_prefix
        self._retire_prompt(
            live,
            notice=(
                "A newer approval request superseded this one. Reactions on "
                f"the message above no longer do anything — reply "
                f"`{prefix}approve` or `{prefix}deny` to answer."
            ),
        )
        return True

    async def _seed_approval_reactions(
        self,
        prompt: _SimplexApprovalPrompt,
        plan: Tuple[Tuple[str, str, str], ...],
    ) -> None:
        """Place the tap targets, then explain the ones that landed.

        The legend is written from *landed* rather than from *plan*, so the
        message can never advertise a tap that is not on it. That is why the
        prompt goes out without a legend and this task sends one.
        """
        landed: List[str] = []
        for index, (emoji, _choice, _label) in enumerate(plan):
            if prompt.resolved:
                return
            outcome = await self._set_reaction(
                prompt.chat_ref, prompt.item_id, emoji, add=True
            )
            if outcome == _REACTION_ACCEPTED:
                self._reactions_supported = True
                if prompt.resolved:
                    # Cleanup ran while this seed was in flight: take our own
                    # late reaction back off instead of stranding it on a
                    # resolved prompt as a live-looking tap target.
                    await self._set_reaction(
                        prompt.chat_ref, prompt.item_id, emoji, add=False
                    )
                    return
                prompt.seeded_emoji.append(emoji)
                landed.append(emoji)
            elif outcome == _REACTION_CAPPED:
                # Not a refusal: the daemon takes our reactions, we have just
                # run out of slots on this item (three per sender). Stop here
                # — every later target would hit the same wall — and let the
                # legend describe what is really on the message. Deny is
                # seeded first precisely so it is never the casualty.
                logger.warning(
                    "SimpleX: reaction slots full on item %s after %d of %d "
                    "tap targets — %s and anything after it were not placed; "
                    "the legend will list only %s",
                    prompt.item_id,
                    len(landed),
                    len(plan),
                    emoji,
                    "".join(landed) or "nothing",
                )
                break
            elif outcome == _REACTION_REJECTED and index == 0:
                # The first emoji is 👎, which the daemon's own validator
                # allows, so rejecting it means reactions are refused
                # wholesale. A rejection further down the list is about that
                # one emoji and must not disable the whole lane.
                await self._downgrade_to_typed_approvals(prompt)
                return
            elif outcome == _REACTION_REJECTED:
                logger.info(
                    "SimpleX: daemon refused approval emoji %s — "
                    "the remaining tap targets still stand",
                    emoji,
                )
        if landed and not prompt.resolved:
            await self._send_approval_message(
                prompt.chat_id, self._format_tap_legend(landed)
            )

    async def _downgrade_to_typed_approvals(
        self, prompt: _SimplexApprovalPrompt
    ) -> None:
        """Record that this daemon refuses to let the bot place reactions.

        This disables *seeding* and the legend, not the inbound lane: a user
        who places one of the approval emoji themselves still resolves the
        prompt. Says so once per downgrade.
        """
        if self._reactions_supported is False:
            return
        self._reactions_supported = False
        logger.warning(
            "SimpleX: daemon does not accept reactions from this bot — "
            "prompts will not be pre-seeded; typed %sapprove still works",
            self.typed_command_prefix,
        )
        prefix = self.typed_command_prefix
        await self._send_approval_message(
            prompt.chat_id,
            "This SimpleX daemon does not let me place the approval "
            f"reactions. Reply `{prefix}approve` or `{prefix}deny` instead.",
        )

    async def _clear_seeded_reactions(self, prompt: _SimplexApprovalPrompt) -> None:
        """Remove the bot's own seeds, leaving the user's reaction in place.

        SimpleX reactions are toggles keyed by emoji rather than addressable
        events, so cleanup is the same command with ``off`` — there is no
        redaction step and nothing to schedule around delivery lag.

        Emoji come off the list as they are toggled, not from a snapshot: the
        seeder may still be appending while this runs.
        """
        while prompt.seeded_emoji:
            emoji = prompt.seeded_emoji.pop(0)
            await self._set_reaction(
                prompt.chat_ref, prompt.item_id, emoji, add=False
            )

    @staticmethod
    def _event_chat_id(wrapper: dict) -> str:
        """Chat the reaction event happened in, in adapter chat-id form.

        Matrix compares the reaction's room against the prompt's room before
        it authorizes anything (``plugins/platforms/matrix/adapter.py``); this
        is the SimpleX equivalent, and it is what turns "a DM reactor is the
        DM owner" from an assumption into a checked invariant.
        """
        chat_info = wrapper.get("chatInfo") or {}
        if not isinstance(chat_info, dict):
            return ""
        info_type = chat_info.get("type")
        if info_type == "direct":
            contact = chat_info.get("contact") or {}
            if isinstance(contact, dict):
                return str(contact.get("contactId", "") or "")
            return ""
        if info_type == "group":
            group = chat_info.get("groupInfo") or {}
            if isinstance(group, dict) and group.get("groupId") is not None:
                return f"group:{group.get('groupId')}"
        return ""

    @staticmethod
    def _reaction_sender(wrapper: dict, chat_reaction: dict) -> Tuple[str, str]:
        """Return ``(user_id, display_name)`` for whoever placed a reaction."""
        chat_dir = chat_reaction.get("chatDir") or {}
        member = chat_dir.get("groupMember") or {} if isinstance(chat_dir, dict) else {}
        if isinstance(member, dict) and member:
            return (
                str(member.get("memberId", "") or ""),
                str(member.get("localDisplayName", "") or ""),
            )
        contact = (wrapper.get("chatInfo") or {}).get("contact") or {}
        if not isinstance(contact, dict):
            return "", ""
        return (
            str(contact.get("contactId", "") or ""),
            str(contact.get("localDisplayName", "") or ""),
        )

    def _reactor_authorized(
        self,
        prompt: _SimplexApprovalPrompt,
        user_id: str,
        chat_reaction: dict,
    ) -> bool:
        """Fail-closed check that this reactor may answer this prompt.

        The reaction lane is direct-chat only, and that is the whole of the
        authorization model: the only party who can react in a DM is the
        contact that owns it, and that contact is the one whose message
        raised the approval — so identity is the chat id. Nothing is read
        from an allowlist env var, and no allow-all flag can widen it;
        ``SIMPLEX_ALLOW_ALL_USERS`` governs who may *talk* to the bot, never
        who may approve a dangerous command. DM-paired users
        (``hermes pairing approve simplex``) keep working with no extra
        configuration, and the typed ``/approve`` path — fully authorized by
        the gateway — remains the way in for everyone else.
        """
        if self._is_group_chat(prompt.chat_id):
            return False
        chat_dir = chat_reaction.get("chatDir") or {}
        if isinstance(chat_dir, dict) and chat_dir.get("groupMember"):
            # A group member reacting on something registered as a DM: the
            # identity namespaces do not match, so refuse rather than guess.
            return False
        return bool(user_id) and user_id == prompt.chat_id

    async def _reaction_feedback(
        self, prompt: _SimplexApprovalPrompt, text: str
    ) -> None:
        """Reply once per prompt about an unusable reaction."""
        if prompt.feedback_sent:
            return
        prompt.feedback_sent = True
        await self._send_approval_message(prompt.chat_id, text)

    async def _expire_approval_prompt(self, prompt: _SimplexApprovalPrompt) -> None:
        """Retire a prompt whose tap window closed."""
        self._retire_prompt(prompt)
        await self._send_approval_message(
            prompt.chat_id,
            "This approval prompt has expired. Run the command again if you "
            "still want to approve it.",
        )

    def _retire_prompt(
        self, prompt: _SimplexApprovalPrompt, *, notice: Optional[str] = None
    ) -> None:
        """Retire a prompt: no live tap targets, no correlation state left.

        The single retirement path. Marking ``resolved`` is what stops a
        later tap and what tells an in-flight seeder to take its own late
        reaction back off; clearing the seeds is what stops the message from
        *looking* answerable. Skipping either one leaves a message with four
        live-looking buttons that resolve something else.
        """
        prompt.resolved = True
        self._approval_prompts_by_item.pop(prompt.item_id, None)
        if self._approval_prompt_by_session.get(prompt.session_key) == prompt.item_id:
            self._approval_prompt_by_session.pop(prompt.session_key, None)
        if prompt.seeded_emoji:
            self._spawn_background(self._clear_seeded_reactions(prompt))
        if notice:
            self._spawn_background(
                self._send_approval_message(prompt.chat_id, notice)
            )

    async def _handle_reaction_event(self, resp: dict) -> None:
        """Resolve a pending exec approval from a ``chatItemReaction`` event.

        The daemon reports reactions as
        ``{"type": "chatItemReaction", "added": bool, "reaction": ACIReaction}``
        where ``ACIReaction`` is ``{"chatInfo": ..., "chatReaction": {...}}``.
        ``chatReaction.chatDir`` describes who reacted, and
        ``chatReaction.chatItem`` is the message they reacted to.
        """
        if resp.get("added") is not True:
            return  # un-reacting never answers a prompt

        wrapper = resp.get("reaction") or {}
        if not isinstance(wrapper, dict):
            return
        chat_reaction = wrapper.get("chatReaction") or {}
        if not isinstance(chat_reaction, dict):
            return

        chat_dir = chat_reaction.get("chatDir") or {}
        if isinstance(chat_dir, dict) and chat_dir.get("type") in (
            "directSnd",
            "groupSnd",
        ):
            return  # our own seed echoing back

        chat_item = chat_reaction.get("chatItem") or {}
        meta = chat_item.get("meta") or {} if isinstance(chat_item, dict) else {}
        item_id = str(meta.get("itemId") or "")
        if not item_id:
            return

        prompt = self._approval_prompts_by_item.get(item_id)
        if prompt is None or prompt.resolved:
            return

        # Currency: this must still be the session's one live prompt. The
        # guard in send_exec_approval already retires anything older, so this
        # should be unreachable — it stays because it is the invariant the
        # whole design rests on, and it costs one dict lookup.
        if self._approval_prompt_by_session.get(prompt.session_key) != prompt.item_id:
            self._retire_prompt(prompt)
            await self._send_approval_message(
                prompt.chat_id,
                "That approval is no longer pending — a newer request "
                "replaced it. The command did not run.",
            )
            return

        # Same chat the prompt was posted in, or nothing doing.
        if self._event_chat_id(wrapper) != prompt.chat_id:
            logger.info(
                "SimpleX: ignoring approval reaction from a different chat"
            )
            return

        user_id, _display_name = self._reaction_sender(wrapper, chat_reaction)
        if not self._reactor_authorized(prompt, user_id, chat_reaction):
            # Logged, not answered: an unauthorized reactor would otherwise
            # get the bot to post on demand.
            logger.info(
                "SimpleX: ignoring approval reaction from unauthorized user %s",
                _redact_id(user_id),
            )
            return

        # Expiry after authorization: an unauthorized party must not be able
        # to make the bot post the expiry notice on demand.
        if time.monotonic() > prompt.expires_at:
            await self._expire_approval_prompt(prompt)
            return

        msg_reaction = chat_reaction.get("reaction") or {}
        emoji = _normalize_reaction_emoji(
            msg_reaction.get("emoji", "") if isinstance(msg_reaction, dict) else ""
        )
        choice = _APPROVAL_REACTION_MAP.get(emoji)
        if not choice or choice not in prompt.choices:
            # Either not an approval emoji at all, or a tier this prompt
            # deliberately did not offer — approve-always on a smart-denied
            # command, say. Never silently widen the offer.
            await self._reaction_feedback(
                prompt, "That reaction is not one of the approval options."
            )
            return

        try:
            from tools.approval import resolve_gateway_approval

            count = resolve_gateway_approval(prompt.session_key, choice)
        except Exception as exc:
            logger.error(
                "Failed to resolve gateway approval from SimpleX reaction: %s", exc
            )
            return

        self._retire_prompt(prompt)
        if not count:
            # Nothing was pending: a typed /approve, an interrupt, or the
            # agent-side timeout already drained the queue. Say so — silently
            # doing nothing reads as a broken tap.
            await self._send_approval_message(
                prompt.chat_id,
                "That approval is no longer pending — it was already answered "
                "or it timed out. The command did not run.",
            )
            return

        logger.info(
            "SimpleX reaction resolved %d approval(s) for session %s (choice=%s)",
            count,
            prompt.session_key,
            choice,
        )
        await self._send_approval_message(prompt.chat_id, _APPROVAL_ACK[choice])

    # ------------------------------------------------------------------
    # Outbound — media
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_image(file_path: str) -> tuple[str, str]:
        """Ensure *file_path* is a PNG and return ``(png_path, thumb_data_uri)``.

        SimpleX clients can't display WebP and a few other formats inline.
        This converts to PNG when needed and generates a small JPEG thumbnail
        for the ``image`` field in the ``/_send`` payload so the chat shows
        an inline preview. Uses Pillow when available, falls back to
        ImageMagick ``convert``.
        """
        import subprocess
        import tempfile

        p = Path(file_path)
        png_path = file_path
        thumb_uri = ""

        try:
            from PIL import Image

            img = Image.open(file_path)
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                png_path = str(p.with_suffix(".png"))
                img.save(png_path, "PNG")
            thumb = img.copy()
            thumb.thumbnail((128, 128))
            import io

            buf = io.BytesIO()
            thumb.save(buf, "JPEG", quality=70)
            thumb_uri = (
                "data:image/jpg;base64,"
                + base64.b64encode(buf.getvalue()).decode()
            )
        except ImportError:
            try:
                if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    png_path = str(p.with_suffix(".png"))
                    subprocess.run(
                        ["convert", file_path, png_path],
                        check=True,
                        capture_output=True,
                        timeout=30,
                    )
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name
                subprocess.run(
                    [
                        "convert",
                        file_path,
                        "-resize",
                        "128x128",
                        "-quality",
                        "70",
                        tmp_path,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                with open(tmp_path, "rb") as f:
                    thumb_uri = (
                        "data:image/jpg;base64," + base64.b64encode(f.read()).decode()
                    )
                os.remove(tmp_path)
            except (FileNotFoundError, subprocess.SubprocessError) as exc:
                logger.warning("SimpleX: image conversion unavailable: %s", exc)

        return png_path, thumb_uri

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an image. Supports ``file://`` URLs and ``http(s)://`` URLs."""
        from urllib.parse import unquote

        if image_url.startswith("file://"):
            file_path = unquote(image_url[7:])
        else:
            try:
                from gateway.platforms.base import cache_image_from_url

                file_path = await cache_image_from_url(image_url)
            except Exception as e:
                logger.warning("SimpleX: failed to download image: %s", e)
                return SendResult(success=False, error=str(e))

        if not file_path or not Path(file_path).exists():
            return SendResult(success=False, error="Image file not found")

        png_path, thumb_uri = self._prepare_image(file_path)

        # /_send addresses by numeric ID; /f only accepts display names which
        # breaks for group IDs.
        composed = json.dumps(
            [
                {
                    "filePath": png_path,
                    "msgContent": {
                        "type": "image",
                        "image": thumb_uri,
                        "text": caption or "",
                    },
                }
            ]
        )

        if chat_id.startswith("group:"):
            group_id = chat_id[6:]
            command = f"/_send #{group_id} json {composed}"
        else:
            command = f"/_send @{chat_id} json {composed}"

        result = await self._send_command(command)
        if result is not None:
            return SendResult(success=True)
        return SendResult(success=False, error="Failed to send image")

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file via SimpleX."""
        return await self.send_image(
            chat_id, f"file://{image_path}", caption=caption, **kwargs
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a video file via SimpleX (as a file attachment)."""
        return await self.send_document(chat_id, video_path, caption=caption)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a document/file attachment."""
        if not Path(file_path).exists():
            return SendResult(success=False, error="File not found")

        composed = json.dumps(
            [
                {
                    "filePath": file_path,
                    "msgContent": {"type": "file", "text": caption or ""},
                }
            ]
        )

        if chat_id.startswith("group:"):
            group_id = chat_id[6:]
            command = f"/_send #{group_id} json {composed}"
        else:
            command = f"/_send @{chat_id} json {composed}"

        result = await self._send_command(command)
        if result is not None:
            return SendResult(success=True)
        return SendResult(success=False, error="Failed to send document")

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        duration: int = 0,
        **kwargs,
    ) -> SendResult:
        """Send an audio file as a SimpleX voice note (plays inline).

        SimpleX distinguishes a generic file attachment (``type: "file"``)
        from an inline voice note (``type: "voice"``). ``/f`` would deliver
        a downloadable file; the structured ``/_send`` form with
        ``msgContent.type == "voice"`` produces the voice-note player.
        """
        if not Path(audio_path).exists():
            return SendResult(success=False, error="Voice file not found")

        composed = json.dumps(
            [
                {
                    "msgContent": {
                        "type": "voice",
                        "text": caption or "",
                        "duration": duration,
                    },
                    "fileSource": {"filePath": audio_path},
                }
            ]
        )

        if chat_id.startswith("group:"):
            group_id = chat_id[6:]
            command = f"/_send #{group_id} json {composed}"
        else:
            command = f"/_send @{chat_id} json {composed}"

        result = await self._send_command(command)
        if result is not None:
            return SendResult(success=True)
        return SendResult(success=False, error="Failed to send voice message")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """SimpleX has no typing-indicator API — no-op."""

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic chat info."""
        if chat_id.startswith("group:"):
            return {"chat_id": chat_id, "type": "group", "name": chat_id[6:]}
        return {"chat_id": chat_id, "type": "dm", "name": chat_id}


# ---------------------------------------------------------------------------
# Plugin entry-point hooks
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """Plugin gate: require SIMPLEX_WS_URL AND the websockets package.

    Returning False keeps the platform out of ``get_connected_platforms()``
    so the gateway never instantiates the adapter when the dependency is
    missing or no daemon URL is configured.
    """
    if not os.getenv("SIMPLEX_WS_URL"):
        return False
    try:
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


def validate_config(config) -> bool:
    """Validate that the platform config has enough info to connect."""
    extra = getattr(config, "extra", {}) or {}
    ws_url = os.getenv("SIMPLEX_WS_URL") or extra.get("ws_url", "")
    return bool(ws_url)


def is_connected(config) -> bool:
    """Check whether SimpleX is configured (env or config.yaml)."""
    extra = getattr(config, "extra", {}) or {}
    ws_url = os.getenv("SIMPLEX_WS_URL") or extra.get("ws_url", "")
    return bool(ws_url)


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Called by the platform registry's env-enablement hook BEFORE adapter
    construction, so ``gateway status`` and ``get_connected_platforms()``
    reflect env-only configuration without instantiating the WebSocket
    client. Returns ``None`` when SimpleX isn't minimally configured.

    The special ``home_channel`` key is handled by the core hook — it
    becomes a proper ``HomeChannel`` dataclass on the ``PlatformConfig``
    rather than being merged into ``extra``.
    """
    ws_url = os.getenv("SIMPLEX_WS_URL", "").strip()
    if not ws_url:
        return None
    seed: dict = {"ws_url": ws_url}

    auto_accept = os.getenv("SIMPLEX_AUTO_ACCEPT", "").strip().lower()
    if auto_accept:
        seed["auto_accept"] = auto_accept not in {"0", "false", "no"}

    group_allowed = os.getenv("SIMPLEX_GROUP_ALLOWED", "").strip()
    if group_allowed:
        seed["group_allowed"] = group_allowed

    home = os.getenv("SIMPLEX_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("SIMPLEX_HOME_CHANNEL_NAME", "").strip() or home,
        }
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
    """Open an ephemeral WebSocket to the daemon, send, and close.

    Used by ``tools/send_message_tool._send_via_adapter`` when the gateway
    runner is not in this process (e.g. ``hermes cron`` running as a
    separate process from ``hermes gateway``). Without this hook,
    ``deliver=simplex`` cron jobs fail with "No live adapter for platform".

    ``thread_id`` and ``force_document`` are accepted for signature parity
    with other plugins but are not meaningful here. ``media_files`` is
    accepted but only the text body is delivered — SimpleX file transfers
    require the daemon's filesystem-backed flow, which an ephemeral
    connection cannot drive safely.
    """
    try:
        import websockets as _wsclient
    except ImportError:
        return {"error": "websockets not installed. Run: pip install websockets"}

    extra = getattr(pconfig, "extra", {}) or {}
    ws_url = os.getenv("SIMPLEX_WS_URL") or extra.get(
        "ws_url", "ws://127.0.0.1:5225"
    )
    if not ws_url:
        return {"error": "SimpleX standalone send: SIMPLEX_WS_URL is required"}

    try:
        composed = json.dumps(
            [{"msgContent": {"type": "text", "text": message}}]
        )
        if chat_id.startswith("group:"):
            group_id = chat_id[6:]
            cmd_str = f"/_send #{group_id} json {composed}"
        else:
            cmd_str = f"/_send @{chat_id} json {composed}"

        payload = {
            "corrId": f"{_CORR_PREFIX}snd-{int(time.time() * 1000)}",
            "cmd": cmd_str,
        }

        async with _wsclient.connect(
            ws_url, open_timeout=10, close_timeout=5
        ) as ws:
            await ws.send(json.dumps(payload))
            # Give the daemon a moment to process the command before closing.
            await asyncio.sleep(0.5)

        return {"success": True, "platform": "simplex", "chat_id": chat_id}
    except Exception as e:
        return {"error": f"SimpleX send failed: {e}"}


def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes setup gateway`` → SimpleX.

    Prompts for the WebSocket URL and the optional allowlist / groups /
    auto-accept / home channel. Writes to ``~/.hermes/.env`` via
    ``hermes_cli.config``.
    """
    print()
    print("SimpleX Chat setup")
    print("------------------")
    print("Requirements:")
    print("  1. simplex-chat daemon running (e.g. `simplex-chat -p 5225`).")
    print("  2. Python package `websockets` installed (`pip install websockets`).")
    print()

    try:
        from hermes_cli.config import get_env_value, save_env_value
    except ImportError:
        print(
            "hermes_cli.config not available; set SIMPLEX_* vars manually in "
            "~/.hermes/.env"
        )
        return

    def _prompt(var: str, prompt: str, *, secret: bool = False) -> None:
        existing = get_env_value(var) if callable(get_env_value) else None
        suffix = " [keep current]" if existing else ""
        try:
            if secret:
                from hermes_cli.secret_prompt import masked_secret_prompt
                value = masked_secret_prompt(f"{prompt}{suffix}: ")
            else:
                value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value:
            save_env_value(var, value)

    _prompt("SIMPLEX_WS_URL", "Daemon WebSocket URL (default ws://127.0.0.1:5225)")
    _prompt("SIMPLEX_ALLOWED_USERS", "Allowed contactIds or display names (comma-separated; blank=skip)")
    _prompt(
        "SIMPLEX_GROUP_ALLOWED",
        "Allowed group IDs (comma-separated, or '*' for any; blank=disable groups)",
    )
    _prompt(
        "SIMPLEX_AUTO_ACCEPT",
        "Auto-accept incoming contact requests? (true/false, default true)",
    )
    _prompt("SIMPLEX_HOME_CHANNEL", "Home channel contact/group ID (or empty)")
    print(
        "Done. Make sure the simplex-chat daemon is running before starting "
        "the gateway."
    )


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="simplex",
        label="SimpleX Chat",
        adapter_factory=lambda cfg: SimplexAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["SIMPLEX_WS_URL"],
        install_hint=(
            "pip install websockets   # SimpleX adapter requires the "
            "websockets package"
        ),
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="SIMPLEX_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="SIMPLEX_ALLOWED_USERS",
        allow_all_env="SIMPLEX_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🔒",
        # SimpleX uses opaque contact IDs only — no phone numbers or email
        # addresses to redact.
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via SimpleX Chat, a private decentralised "
            "messenger. Contacts are identified by opaque internal IDs, "
            "not phone numbers or usernames. SimpleX supports standard "
            "markdown formatting. There is no typing indicator and no "
            "hard message length limit, but keep responses conversational. "
            "You can attach native images, voice notes, and arbitrary "
            "files; the adapter handles MEDIA:<path> tags by sending them "
            "as inline voice notes (audio extensions) or documents."
        ),
    )
