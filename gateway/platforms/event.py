"""Inbound message event types shared by every gateway platform adapter.

A leaf module: adapters, helpers and the runner import it, so it must not import from
gateway.platforms.*.
"""

import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from gateway.session import SessionSource



def _is_command_boundary_char(ch: str) -> bool:
    """Invisible padding that messaging clients wrap around pasted text.

    Covers whitespace/newlines plus Unicode ``Cc`` (control) and ``Cf``
    (format) — the latter is where the characters that actually show up in
    chat transports live: U+2060 WORD JOINER, U+200B ZWSP, U+200C/D
    ZWNJ/ZWJ, U+FEFF BOM, U+200E/F LRM/RLM.
    """
    return ch.isspace() or unicodedata.category(ch) in {"Cc", "Cf"}


def _strip_command_boundary_chars(text: str) -> str:
    """Trim invisible boundary padding from both ends of ``text``.

    Only the outer edges are touched: interior characters are left alone so
    a mid-word joiner (``/ne⁠w``) never gets normalized into a
    different command identity.
    """
    start = 0
    end = len(text)
    while start < end and _is_command_boundary_char(text[start]):
        start += 1
    while end > start and _is_command_boundary_char(text[end - 1]):
        end -= 1
    return text[start:end]


def _lstrip_command_boundary_chars(text: str) -> str:
    """Trim leading invisible padding only.

    Used where the tail is argument payload. Per the contract established in
    0a53663ef8 ("only the command delimiter is nonsemantic"), trailing
    whitespace inside arguments is authored input this layer must not
    consume — every handler that wants it gone calls ``.strip()`` itself.
    """
    start = 0
    while start < len(text) and _is_command_boundary_char(text[start]):
        start += 1
    return text[start:]


def _rstrip_invisible_chars(text: str) -> str:
    """Drop trailing Cc/Cf characters while preserving trailing whitespace.

    Client-injected padding wrapped around a whole message lands after the
    arguments, so it still has to go. Whitespace does not: see
    ``_lstrip_command_boundary_chars`` for why that distinction exists.
    """
    end = len(text)
    while end > 0 and unicodedata.category(text[end - 1]) in {"Cc", "Cf"}:
        end -= 1
    return text[:end]


def looks_like_slash_command(text: str) -> bool:
    """Whether ``text`` reads as a slash command once padding is ignored.

    Single source of truth for the inbound-classification side, shared with
    ``MessageEvent.is_command()``. Adapters that tag ``MessageType.COMMAND``
    must use this rather than a bare ``startswith("/")``: a plain prefix test
    routes ``⁠/deny`` into the text-batching path, where the debounce
    window merges it with whatever the user types next.

    This answers "does it look like a command", not "may it run as one" —
    authorization stays with ``allow_gateway_control`` and the slash-access
    checks in gateway.run.
    """
    return _strip_command_boundary_chars(text or "").startswith("/")


class MessageType(Enum):
    """Types of incoming messages."""
    TEXT = "text"
    LOCATION = "location"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE = "voice"
    DOCUMENT = "document"
    STICKER = "sticker"
    COMMAND = "command"  # /command style


class ProcessingOutcome(Enum):
    """Result classification for message-processing lifecycle hooks."""
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass
class MessageEvent:
    """Incoming message from a platform — the normalized shape all adapters produce."""
    text: str
    message_type: MessageType = MessageType.TEXT
    # Author, mirrored from ``source`` for per-message prompt builders; None for non-IM sources.
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    # None only in isolated unit tests; production always sets it. Typing it Optional
    # exposes ~60 unguarded ``.source.<attr>`` reads, so that is a separate change.
    source: SessionSource = None
    raw_message: Any = None
    message_id: Optional[str] = None
    # Delivery-ledger identity for the final send, when it differs from ``message_id``. A queued
    # (/queue) chain answers the LAST message of the chain, so its final send has to be ledgered
    # under that message's id. Keyed on the opening event's id instead, two chained turns carrying
    # the same text collide on one obligation id and the earlier turn's row is overwritten (a
    # refused first reply then reads as delivered). Reply routing is unaffected: the reply anchor
    # still comes from this event.
    ledger_message_id: Optional[str] = None
    # Platform update id (Telegram ``update_id``): ``/restart`` records it so the new gateway
    # advances past it even if PTB's shutdown ACK times out.
    platform_update_id: Optional[int] = None
    # Media attachments: local file paths (for vision tool access)
    media_urls: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    # Per-attachment text-inlining contract; None = legacy "text/* already inlined into ``text``".
    media_text_inlined: List[Optional[bool]] = field(default_factory=list)
    reply_to_message_id: Optional[str] = None
    reply_to_text: Optional[str] = None  # Text of the replied-to message (for context injection)
    reply_to_author_id: Optional[str] = None
    reply_to_author_name: Optional[str] = None
    reply_to_is_own_message: bool = False  # True when the user replied to this bot/assistant's message
    # Structured interactive-prompt reply (relay only): {prompt_id, option_id, label?,
    # prompt_message_id?}; routed to the approval/slash-confirm/clarify resolvers BEFORE dispatch.
    prompt_response: Optional[Dict[str, Any]] = None
    # Auto-loaded skill(s) for topic/channel bindings; a single name or ordered list.
    auto_skill: Optional[str | list[str]] = None
    # Per-channel ephemeral system prompt; applied at API call time, never persisted to transcript.
    channel_prompt: Optional[str] = None
    # History-backfilled channel context (missed under require_mention); kept out of ``text`` so
    # run.py's sender-prefix logic sees only the trigger message.
    channel_context: Optional[str] = None
    # Set for synthetic events (e.g. background-process notifications) that must bypass user authorization.
    internal: bool = False
    # Free-form per-event metadata (e.g. ``whatsapp_from_owner=True``); plugins must ``.get()``.
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    # May this event resolve gateway commands / control prompts? Proactive plugin events set False
    # so untrusted payload text stays conversational. Kept last for positional compat.
    allow_gateway_control: bool = True

    # Process-local admission receipt, never routing metadata or execution acknowledgement.
    _gateway_accepted: bool = field(default=False, init=False, repr=False, compare=False)

    def is_command(self) -> bool:
        """Check if this is a command message (e.g., /new, /reset)."""
        return self.allow_gateway_control and looks_like_slash_command(self.text)

    def get_command(self) -> Optional[str]:
        """Extract command name if this is a command message."""
        if not self.is_command():
            return None
        token = _strip_command_boundary_chars(self.text or "").split(maxsplit=1)[0]
        # Re-trim the command token: when a client wraps just the command in
        # invisible padding ("⁠/deny⁠ reason"), the trailing joiner
        # sits inside the token and would leave the name unresolvable.
        raw = _strip_command_boundary_chars(token[1:]).lower().split("@", 1)[0]
        # Reject file paths: valid command names never contain /
        return None if "/" in raw else raw

    def get_command_args(self) -> str:
        """Get the arguments after a command."""
        if not self.is_command():
            return self.text
        # Leading-only trim here: the tail is argument payload, whose trailing
        # whitespace is authored input this layer must not consume.
        parts = _lstrip_command_boundary_chars(self.text or "").split(maxsplit=1)
        args = _rstrip_invisible_chars(parts[1]) if len(parts) > 1 else ""
        # iOS auto-corrects -- to — (em dash) and - to – (en dash)
        return args.replace("\u2014\u2014", "--").replace("\u2014", "--").replace("\u2013", "-")
