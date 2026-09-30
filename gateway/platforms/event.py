"""Inbound message event types shared by every gateway platform adapter.

A leaf module: adapters, helpers and the runner import it, so it must not import from
gateway.platforms.*.
"""

import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from gateway.session import SessionSource, neutralize_untrusted_inline_text

# Desktop attachment reference tags prepended by buildContextText before the
# user's visible text (e.g. "@image:/tmp/foo.png\n\n/moa ask something").
# Strip these when detecting slash commands so a media-ref prefix does not hide
# a slash token from MessageEvent.is_command() / get_command().
# The pattern matches to end-of-line (not just whitespace-bounded) to handle
# Windows paths that may contain spaces (e.g. "C:\Users\John Doe\image.png").
_ATTACHMENT_REF_RE = re.compile(r"^(?:@(?:image|file|url):[^\n]+\n?)+", re.IGNORECASE)

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.inbound_context import InboundContextSnapshot, PreparedInboundMessage


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


@dataclass(frozen=True)
class QuotedMediaDependency:
    """A media attachment whose content depends on another platform event."""
    room_id: str
    event_id: str
    media_index: int
    content_id: str


@dataclass(frozen=True)
class TurnContextUpdate:
    """What ``BasePlatformAdapter.prepare_turn_context`` reports for one turn.

    ``note`` is prepended to the user message. ``channel_state`` is saved with the user transcript
    row, so the change is acknowledged only when the turn that reported it is saved. It is ``None``
    when the adapter could not read the chat state, and the saved state then stays unchanged.
    """
    note: Optional[str]
    channel_state: Optional[Dict[str, Any]]


class _ProcessingPhase(Enum):
    PENDING = "pending"
    DEFERRED = "deferred"
    RUNNING = "running"
    ABSORBED = "absorbed"
    COMPLETED = "completed"


@dataclass
class _ProcessingInput:
    event: "MessageEvent"
    text: Optional[str]


@dataclass
class _ProcessingCompletion:
    adapter: "BasePlatformAdapter"
    event: "MessageEvent"


@dataclass
class _ProcessingState:
    """Processing lifecycle of one input, shared by every copy of its event.

    Invariant: once an input's start has been reported (``start_notified``), it completes exactly
    once, with the outcome of the turn that consumed it, or CANCELLED when it is discarded before a
    turn runs it. Every hand-off below preserves this: an input absorbed by the running turn
    completes with that turn; an input parked for a later turn (deferred or handed to a copy) is
    ``awaiting_start`` and completes when that turn runs, or through ``discard()`` when it is dropped.
    """

    phase: _ProcessingPhase = _ProcessingPhase.PENDING
    outcome: Optional[ProcessingOutcome] = None
    receipt_message_id: Optional[str] = None
    consumed_receipt_message_id: Optional[str] = None
    receipt_inputs: List[_ProcessingInput] = field(default_factory=list)
    pending_completion: Optional[_ProcessingCompletion] = None
    absorbed: List[_ProcessingCompletion] = field(default_factory=list)
    start_notified: bool = False
    awaiting_start: bool = False
    # The event copy that runs this input's lifecycle; completions through other copies are ignored.
    owner: Optional["MessageEvent"] = field(default=None, repr=False)

    def defer(self) -> None:
        """Park the input for a later turn."""
        self.phase = _ProcessingPhase.DEFERRED
        if self.start_notified:
            self.awaiting_start = True

    def defer_unstarted(self) -> None:
        """Defer an input that has not started. The adapter completes a started input when its
        handler returns."""
        if self.phase is _ProcessingPhase.PENDING:
            self.phase = _ProcessingPhase.DEFERRED

    def absorb(self, adapter: "BasePlatformAdapter", event: "MessageEvent") -> bool:
        """Complete a started input with this turn instead of when its own handler returns."""
        state = event._processing_state
        if state is self or self.phase is not _ProcessingPhase.RUNNING or state.phase is not _ProcessingPhase.RUNNING:
            return False
        state.phase = _ProcessingPhase.ABSORBED
        self.absorbed.append(_ProcessingCompletion(adapter, event))
        return True

    def attach(self, completion: _ProcessingCompletion) -> None:
        """Complete *completion*'s started input with this input's turn."""
        completion.event._processing_state.phase = _ProcessingPhase.ABSORBED
        self.absorbed.append(completion)

    def release(self, event: "MessageEvent") -> Optional[_ProcessingCompletion]:
        """Detach an absorbed input so another turn completes it. Returns its completion."""
        released = None
        kept = []
        for completion in self.absorbed:
            if completion.event._processing_state is event._processing_state:
                released = completion
            else:
                kept.append(completion)
        self.absorbed = kept
        return released

    def hand_over(self, copy: "MessageEvent") -> None:
        """Let the turn that runs *copy* own this started input. Its original handler may still be
        running, for example while it sends a /steer acknowledgement, and must not complete it."""
        self.owner = copy
        self.awaiting_start = True

    def discard(self) -> bool:
        """Prepare a started input that was parked for a turn that will not run it to complete.
        Returns False when there is nothing to complete: the input never started, its turn has
        started, or it has completed."""
        if not (self.start_notified and self.awaiting_start):
            return False
        self.phase = _ProcessingPhase.RUNNING
        self.awaiting_start = False
        return True

    @property
    def has_unrun_attached(self) -> bool:
        return bool(self.absorbed) and self.phase in {_ProcessingPhase.PENDING, _ProcessingPhase.DEFERRED}

    def take_attached_if_unrun(self) -> List[_ProcessingCompletion]:
        """The inputs attached to an event that never ran; they complete when the event is dropped."""
        return self.take_absorbed() if self.has_unrun_attached else []

    def take_absorbed(self) -> List[_ProcessingCompletion]:
        absorbed, self.absorbed = self.absorbed, []
        for completion in absorbed:
            completion.event._processing_state.phase = _ProcessingPhase.RUNNING
        return absorbed

    def take_pending_inputs(self, pending_text: str) -> List["MessageEvent"]:
        """Remove the turn's steered inputs and return, in arrival order, those that the model left
        unconsumed in *pending_text*."""
        pending_indices: set[int] = set()
        remaining = pending_text
        # A later correction can quote an earlier input's complete payload.
        for index in range(len(self.receipt_inputs) - 1, -1, -1):
            text = self.receipt_inputs[index].text
            if text is not None and text in remaining:
                pending_indices.add(index)
                remaining = remaining.replace(text, "", 1)
        pending_inputs = []
        for index, incoming in enumerate(self.receipt_inputs):
            if index in pending_indices:
                pending_inputs.append(incoming.event)
                continue
            self.consumed_receipt_message_id = incoming.event.receipt_message_id
        self.receipt_inputs.clear()
        self.receipt_message_id = self.consumed_receipt_message_id
        return pending_inputs

    def start(self) -> bool:
        """Begin processing the input. Returns False when its start was already reported, so the
        turn that runs a parked or handed-over input does not report a second start."""
        first = not self.start_notified
        self.phase = _ProcessingPhase.RUNNING
        self.outcome = None
        self.consumed_receipt_message_id = self.receipt_message_id
        self.receipt_inputs.clear()
        self.start_notified = True
        self.awaiting_start = False
        return first

    def complete(self) -> bool:
        if self.phase in {_ProcessingPhase.DEFERRED, _ProcessingPhase.ABSORBED, _ProcessingPhase.COMPLETED}:
            return False
        self.phase = _ProcessingPhase.COMPLETED
        self.start_notified = self.awaiting_start = False
        self.owner = None
        return True

    def complete_inline(self) -> bool:
        if self.phase is not _ProcessingPhase.PENDING:
            return False
        return self.complete()


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
    # dataclasses.replace shares this mutable state with the adapter's original event.
    _processing_state: _ProcessingState = field(default_factory=_ProcessingState, repr=False, compare=False)
    # Deliver this external event as a new turn when its session is busy.
    defer_until_idle: bool = False
    # Snapshot from ``BasePlatformAdapter.prepare_turn_context``. The user transcript row saves it,
    # and the saved snapshot is the baseline for the adapter's next comparison.
    channel_state: Optional[Dict[str, Any]] = None
    # Whether the quoted author passed the adapter's authorisation check; None when the adapter
    # did not check. The reply pointer identifies the author only when this is set.
    reply_to_author_authorized: Optional[bool] = None
    # IDs of other events represented in this turn.
    merged_message_ids: List[str] = field(default_factory=list)

    # Process-local admission receipt, never routing metadata or execution acknowledgement.
    if TYPE_CHECKING:
        _gateway_pending_stt_echoed_paths: set[str] = field(
            default_factory=set, init=False, repr=False, compare=False
        )

    _gateway_accepted: bool = field(default=False, init=False, repr=False, compare=False)
    # Run-owned final presentation snapshot; never deserialized from ingress metadata.
    _notification_reply_muted: Optional[bool] = field(default=None, init=False, repr=False, compare=False)
    _prepared_inbound: Optional["PreparedInboundMessage"] = field(default=None, init=False, repr=False, compare=False)
    _quoted_media_dependencies: tuple[QuotedMediaDependency, ...] = field(
        default=(), kw_only=True, repr=False, compare=False,
    )

    _inbound_context_dependencies: tuple["InboundContextSnapshot", ...] = field(
        default=(), kw_only=True, repr=False, compare=False,
    )

    _ingress_order: Optional[int] = field(default=None, init=False, repr=False, compare=False)

    # Process-local: the events merged into this one, in arrival order, each paired with the
    # function that merged it (None for the first). Empty until something is merged in.
    # ``withdraw_pending_message`` replays the list without a withdrawn message.
    _merged_parts: List[
        Tuple[
            "MessageEvent", Optional[Callable[["MessageEvent", "MessageEvent"], None]]
        ]
    ] = field(default_factory=list, init=False, repr=False, compare=False)

    def absorb_reply_expected(self, other: MessageEvent) -> None:
        """One turn now answers *other* too: an addressed message wins, then an unknown one."""
        if self.reply_expected is not True and other.reply_expected is not False:
            self.reply_expected = other.reply_expected

    def absorb_message_ids(self, other: "MessageEvent") -> None:
        self.merged_message_ids.extend(
            message_id for message_id in (other.message_id, *other.merged_message_ids) if message_id
        )

    def absorb_turn_input(self, other: "MessageEvent", *, input_text: Optional[str] = None) -> None:
        self.absorb_reply_expected(other)
        receipt_id = other.receipt_message_id
        if receipt_id:
            self._processing_state.receipt_message_id = receipt_id
            if self._processing_state is not other._processing_state:
                self._processing_state.receipt_inputs.append(_ProcessingInput(other, input_text))

    @property
    def receipt_message_id(self) -> Optional[str]:
        return self._processing_state.receipt_message_id or self.message_id

    def _command_text(self) -> str:
        """Return the message text with leading Desktop attachment refs stripped.

        Desktop's buildContextText prepends ``@image:<path>``, ``@file:<path>``,
        or ``@url:<url>`` tags before the user's visible text.  Stripping these
        lets is_command / get_command / get_command_args work correctly even
        when the payload is prefixed with one or more media refs.
        """
        return _ATTACHMENT_REF_RE.sub("", (self.text or "").lstrip()).lstrip()

    def _replies_to_message(self) -> bool:
        return bool(self.reply_to_message_id or self.reply_to_text)

    def reply_context(self) -> tuple:
        return (self.reply_to_message_id, self.reply_to_text, self.reply_to_author_id,
                self.reply_to_author_name, bool(self.reply_to_is_own_message), self.reply_to_author_authorized)

    def reply_context_conflicts(self, other: "MessageEvent") -> bool:
        """True when both events reply to a message and their reply contexts differ. A merged
        event has room for only one reply context."""
        return (self._replies_to_message() and other._replies_to_message()
                and self.reply_context() != other.reply_context())

    def absorb_reply_context(self, other: "MessageEvent") -> None:
        """One turn now answers *other* too: take its reply context if this event has none."""
        if self._replies_to_message() or not other._replies_to_message():
            return
        (self.reply_to_message_id, self.reply_to_text, self.reply_to_author_id,
         self.reply_to_author_name, self.reply_to_is_own_message,
         self.reply_to_author_authorized) = other.reply_context()
        self.absorb_context_dependencies(other)

    def absorb_context_dependencies(self, other: "MessageEvent") -> None:
        self._inbound_context_dependencies += tuple(
            dependency for dependency in other._inbound_context_dependencies
            if all(dependency is not existing for existing in self._inbound_context_dependencies)
        )

    def absorb_media(self, other: "MessageEvent") -> None:
        """Append attachments with their inline flags and quoted-event dependencies."""
        self.absorb_context_dependencies(other)
        offset = len(self.media_urls)
        self.media_text_inlined = [
            *self.media_text_inlined,
            *([None] * (offset - len(self.media_text_inlined))),
            *other.media_text_inlined,
            *([None] * (len(other.media_urls) - len(other.media_text_inlined))),
        ]
        self.media_urls.extend(other.media_urls)
        self.media_types.extend(other.media_types)
        self._quoted_media_dependencies += tuple(
            replace(dependency, media_index=dependency.media_index + offset)
            for dependency in other._quoted_media_dependencies
        )

    def authored_media(self) -> "MessageEvent":
        """Return attachments that do not depend on a quoted event."""
        quoted = {
            dependency.media_index for dependency in self._quoted_media_dependencies
        }
        indices = [
            index for index in range(len(self.media_urls)) if index not in quoted
        ]
        return replace(
            self,
            media_urls=[self.media_urls[index] for index in indices],
            media_types=[
                self.media_types[index]
                for index in indices
                if index < len(self.media_types)
            ],
            media_text_inlined=[
                self.media_text_inlined[index]
                if index < len(self.media_text_inlined)
                else None
                for index in indices
            ],
            _quoted_media_dependencies=(),
        )

    def add_channel_context(self, block: str) -> None:
        """Append *block* to ``channel_context``, after a blank line."""
        block = block.strip()
        if not block:
            return
        self.channel_context = f"{self.channel_context.rstrip()}\n\n{block}" if self.channel_context else block


    def text_with_channel_context(self, text: str) -> str:
        """*text* after ``channel_context`` and a ``[New message]`` marker, as the model sees it."""
        return f"{self.channel_context}\n\n[New message]\n{text}" if self.channel_context else text


    def absorb_channel_context(self, other: "MessageEvent") -> None:
        """Keep *other*'s ``channel_context`` when *other* is merged into this event."""
        if other.channel_context and other.channel_context != self.channel_context:
            self.add_channel_context(other.channel_context)


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


def attributed_context(label: str, text: str, author: Optional[str] = None) -> str:
    """A ``channel_context`` block for text that someone other than the sender wrote, headed
    ``[<label>]`` or ``[<label> from <author>]``."""
    author = neutralize_untrusted_inline_text(author) if author else ""
    header = f"[{label} from {author}]" if author else f"[{label}]"
    return f"{header}\n{text.strip()}"
