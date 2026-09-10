"""Inbound message event types shared by every gateway platform adapter.

A leaf module: adapters, helpers and the runner import it, so it must not import from
gateway.platforms.*.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from gateway.session import SessionSource


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

    def is_command(self) -> bool:
        """Check if this is a command message (e.g., /new, /reset)."""
        return self.allow_gateway_control and (self.text or "").lstrip().startswith("/")

    def get_command(self) -> Optional[str]:
        """Extract command name if this is a command message."""
        if not self.is_command():
            return None
        raw = (self.text or "").lstrip().split(maxsplit=1)[0][1:].lower().split("@", 1)[0]
        # Reject file paths: valid command names never contain /
        return None if "/" in raw else raw

    def get_command_args(self) -> str:
        """Get the arguments after a command."""
        if not self.is_command():
            return self.text
        parts = (self.text or "").lstrip().split(maxsplit=1)
        args = parts[1] if len(parts) > 1 else ""
        # iOS auto-corrects -- to — (em dash) and - to – (en dash)
        return args.replace("\u2014\u2014", "--").replace("\u2014", "--").replace("\u2013", "-")
