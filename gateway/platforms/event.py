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

    # Captured before batching mutates text/media; input links, not output copies.
    input_members: List[Dict[str, Any]] = field(default_factory=list)
    terminal_event: Optional["MessageEvent"] = field(default=None, repr=False, compare=False)
    # Derived at canonical ingress, never trusted from platform metadata.
    ingress_provenance: List[Dict[str, Any]] = field(default_factory=list, repr=False, compare=False)

    def original_inputs(self) -> List[Dict[str, Any]]:
        if self.input_members:
            return self.input_members
        return [{
            "inbound_id": self.metadata.get("_hermes_durable_inbound_id", ""),
            "message_id": self.message_id, "platform_update_id": self.platform_update_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "reply_to_message_id": self.reply_to_message_id, "internal": self.internal,
            "reply_to_text": self.reply_to_text, "text": self.text,
            "media_urls": list(self.media_urls), "media_types": list(self.media_types),
            "media_text_inlined": list(self.media_text_inlined),
            "message_type": self.message_type.value,
            "reply_to_author_id": self.reply_to_author_id,
            "reply_to_author_name": self.reply_to_author_name,
            "reply_to_is_own_message": self.reply_to_is_own_message,
        }]

    def retain_inputs(self, members: List[Dict[str, Any]]) -> None:
        """Rebuild a mixed replay batch from only the untransferred originals.

        Keep this event object: queued/recursive callers hold its identity for
        the terminal transfer. Never retain a replied-to anchor from a removed
        member, or enrich an already-answered attachment a second time.
        """
        if not members:
            raise ValueError("cannot rebuild an empty input batch")
        first = members[0]
        self.input_members = list(members)
        self.text = "\n\n".join(m["text"] for m in members if m.get("text"))
        self.media_urls = [url for m in members for url in m.get("media_urls", [])]
        self.media_types = [kind for m in members for kind in m.get("media_types", [])]
        self.media_text_inlined = []
        for member in members:
            flags = list(member.get("media_text_inlined", []))
            self.media_text_inlined.extend(flags + [None] * (len(member.get("media_urls", [])) - len(flags)))
        self.message_type = MessageType(first.get("message_type", "text"))
        if any(m.get("message_type") == "photo" for m in members):
            self.message_type = MessageType.PHOTO
        for name in ("message_id", "platform_update_id", "reply_to_message_id", "reply_to_text",
                     "reply_to_author_id", "reply_to_author_name"):
            setattr(self, name, first.get(name))
        self.reply_to_is_own_message = first.get("reply_to_is_own_message", False)
        self.internal = first.get("internal", False)
        if first.get("timestamp"):
            self.timestamp = datetime.fromisoformat(first["timestamp"])
        self.metadata = {**self.metadata, "_hermes_durable_inbound_id": first["inbound_id"]}
        self.metadata.pop("_hermes_durable_inbound_path", None)
        self.ledger_message_id = None
        for name in ("_gateway_pending_stt_text", "_gateway_pending_stt_transcripts"):
            if hasattr(self, name):
                delattr(self, name)

    def absorb_input_identity(self, other: "MessageEvent") -> None:
        members = self.original_inputs() + other.original_inputs()
        self.input_members = list({
            (m["inbound_id"] or (m["message_id"], m["platform_update_id"], m["timestamp"])): m
            for m in members
        }.values())

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

    def is_command(self) -> bool:
        """Check if this is a command message (e.g., /new, /reset)."""
        return self.allow_gateway_control and self._command_text().startswith("/")

    def get_command(self) -> Optional[str]:
        """Extract command name if this is a command message."""
        if not self.is_command():
            return None
        raw = self._command_text().split(maxsplit=1)[0][1:].lower().split("@", 1)[0]
        # Reject file paths: valid command names never contain /
        return None if "/" in raw else raw

    def get_command_args(self) -> str:
        """Get the arguments after a command."""
        if not self.is_command():
            return self.text
        parts = self._command_text().lstrip().split(maxsplit=1)
        args = parts[1] if len(parts) > 1 else ""
        # iOS auto-corrects -- to — (em dash) and - to – (en dash)
        return args.replace("\u2014\u2014", "--").replace("\u2014", "--").replace("\u2013", "-")
