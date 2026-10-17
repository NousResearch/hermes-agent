"""Inbound message event types shared by every gateway platform adapter.

A leaf module: adapters, helpers and the runner import it, so it must not import from
gateway.platforms.*.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from gateway.session import SessionSource

# Desktop attachment reference tags prepended by buildContextText before the
# user's visible text (e.g. "@image:/tmp/foo.png\n\n/moa ask something").
# Strip these when detecting slash commands so a media-ref prefix does not hide
# a slash token from MessageEvent.is_command() / get_command().
# The pattern matches to end-of-line (not just whitespace-bounded) to handle
# Windows paths that may contain spaces (e.g. "C:\Users\John Doe\image.png").
_ATTACHMENT_REF_RE = re.compile(r"^(?:@(?:image|file|url):[^\n]+\n?)+", re.IGNORECASE)


def parse_slash_command(text: Optional[str]) -> Optional[str]:
    """Return the slash-command name ``text`` carries, or ``None`` when it carries none.

    ``/new`` is a command; ``/tmp/x is broken`` and ``/root/notes.md done`` are not: a valid
    command name never contains ``/``. This is the single source of truth for "is this text a
    slash command" — ``MessageEvent.is_command``/``get_command`` and every platform adapter's
    inbound classification must route through it.

    An adapter that classifies on a bare ``startswith("/")`` tags a file path
    ``MessageType.COMMAND`` and the gateway then refuses the message as ``invalid_params``
    before any handler sees it, so the turn is lost with only a refusal notice. (Desktop
    attachment refs are stripped first, so ``@file:<path>\\n\\n/new`` still finds the command.)
    """
    cleaned = _ATTACHMENT_REF_RE.sub("", (text or "").lstrip()).lstrip()
    if not cleaned.startswith("/"):
        return None
    raw = cleaned.split(maxsplit=1)[0][1:].lower().split("@", 1)[0]
    # "": a lone "/" or "/@bot" carries no command name either.
    return raw if raw and "/" not in raw else None


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
    # Reply anchor for the final send when the answer is to a DIFFERENT message than the one that
    # opened the turn: a successful busy redirect turns the running turn onto the redirecting
    # message, so its reply must quote that message (#115001). ``_reply_anchor_for_event``
    # honours this over ``message_id``; None = derive from the event as usual.
    reply_anchor_override: Optional[str] = None
    # Platform update id (Telegram ``update_id``): ``/restart`` records it so the new gateway
    # advances past it even if PTB's shutdown ACK times out.
    platform_update_id: Optional[int] = None
    # Media attachments: local file paths (for vision tool access)
    media_urls: list[str] = field(default_factory=list)
    media_types: list[str] = field(default_factory=list)
    # Per-attachment text-inlining contract; None = legacy "text/* already inlined into ``text``".
    media_text_inlined: list[Optional[bool]] = field(default_factory=list)
    reply_to_message_id: Optional[str] = None
    reply_to_text: Optional[str] = None  # Text of the replied-to message (for context injection)
    reply_to_author_id: Optional[str] = None
    reply_to_author_name: Optional[str] = None
    reply_to_is_own_message: bool = False  # True when the user replied to this bot/assistant's message
    # Structured interactive-prompt reply (relay only): {prompt_id, option_id, label?,
    # prompt_message_id?}; routed to the approval/slash-confirm/clarify resolvers BEFORE dispatch.
    prompt_response: Optional[dict[str, Any]] = None
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
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    # May this event resolve gateway commands / control prompts? Proactive plugin events set False
    # so untrusted payload text stays conversational. New fields append after it (positional compat).
    allow_gateway_control: bool = True
    # Was this message addressed to this bot? False lets a bare silence marker stand (the adapter
    # knows the message was meant for someone else); None means unknown and keeps the visible
    # fallback, like True.
    reply_expected: Optional[bool] = None
    # Set for synthetic turns that carry no channel inputs of their own (goal continuation,
    # heartbeat due-prompts, /goal resume) so the turn reuses the session's pinned channel
    # prompt/parent instead of overwriting it with this event's own None (#126109). Independent
    # of ``internal``: these events still go through user authorization. Appended after
    # ``reply_expected`` for positional compat (see the field-order rule above).
    inherit_channel_pin: bool = False

    # Process-local admission receipt, never routing metadata or execution acknowledgement.
    _gateway_accepted: bool = field(default=False, init=False, repr=False, compare=False)
    # Run-owned final presentation snapshot; never deserialized from ingress metadata.
    _notification_reply_muted: Optional[bool] = field(default=None, init=False, repr=False, compare=False)

    def absorb_reply_expected(self, other: "MessageEvent") -> None:
        """One turn now answers *other* too: an addressed message wins, then an unknown one."""
        if self.reply_expected is not True and other.reply_expected is not False:
            self.reply_expected = other.reply_expected

    def _command_text(self) -> str:
        """Return the message text with leading Desktop attachment refs stripped.

        Desktop's buildContextText prepends ``@image:<path>``, ``@file:<path>``,
        or ``@url:<url>`` tags before the user's visible text.  Stripping these
        lets is_command / get_command / get_command_args work correctly even
        when the payload is prefixed with one or more media refs.
        """
        return _ATTACHMENT_REF_RE.sub("", (self.text or "").lstrip()).lstrip()

    def _command_name(self) -> Optional[str]:
        """The command name this message carries, or ``None`` when it carries none.

        ``allow_gateway_control`` gates it: a platform that opted out of gateway control has no
        commands, whatever the text looks like.
        """
        if not self.allow_gateway_control:
            return None
        return parse_slash_command(self._command_text())

    def is_command(self) -> bool:
        """Check if this is a command message (e.g., /new, /reset).

        Shares its verdict with :meth:`get_command`: a message whose first token is a file path
        is text, not a command, so the two can never disagree — disagreement is what made
        ``/root/notes.md done`` a "command" with no name to dispatch, which the gateway refused
        as ``invalid_params``.
        """
        return self._command_name() is not None

    def get_command(self) -> Optional[str]:
        """Extract command name if this is a command message."""
        return self._command_name()

    def get_command_args(self) -> str:
        """Get the arguments after a command."""
        if not self.is_command():
            return self.text
        parts = self._command_text().lstrip().split(maxsplit=1)
        args = parts[1] if len(parts) > 1 else ""
        # iOS auto-corrects -- to — (em dash) and - to – (en dash)
        return args.replace("\u2014\u2014", "--").replace("\u2014", "--").replace("\u2013", "-")
