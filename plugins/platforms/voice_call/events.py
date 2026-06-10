"""Core domain types for the voice_call platform.

Every carrier webhook payload — regardless of provider — is parsed into the
``NormalizedEvent`` shape so the call manager never sees provider-specific
wire formats. Call lifecycle state lives in ``CallRecord``, persisted as
JSONL by ``store.py``.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class CallState(str, Enum):
    """Lifecycle states for a call.

    Non-terminal flow: initiated → ringing → answered → active, with
    speaking ⇄ listening oscillation during a conversation. Terminal states
    are absorbing.
    """

    INITIATED = "initiated"
    RINGING = "ringing"
    ANSWERED = "answered"
    ACTIVE = "active"
    SPEAKING = "speaking"
    LISTENING = "listening"
    # Terminal
    COMPLETED = "completed"
    HANGUP_USER = "hangup-user"
    HANGUP_BOT = "hangup-bot"
    TIMEOUT = "timeout"
    ERROR = "error"
    FAILED = "failed"
    NO_ANSWER = "no-answer"
    BUSY = "busy"
    VOICEMAIL = "voicemail"


TERMINAL_STATES = frozenset(
    {
        CallState.COMPLETED,
        CallState.HANGUP_USER,
        CallState.HANGUP_BOT,
        CallState.TIMEOUT,
        CallState.ERROR,
        CallState.FAILED,
        CallState.NO_ANSWER,
        CallState.BUSY,
        CallState.VOICEMAIL,
    }
)

# Forward progression order for non-terminal states. Backwards transitions
# are rejected except for the speaking ⇄ listening oscillation.
_STATE_ORDER = {
    CallState.INITIATED: 0,
    CallState.RINGING: 1,
    CallState.ANSWERED: 2,
    CallState.ACTIVE: 3,
    CallState.SPEAKING: 4,
    CallState.LISTENING: 4,
}


def is_terminal(state: CallState) -> bool:
    return state in TERMINAL_STATES


def is_valid_transition(current: CallState, new: CallState) -> bool:
    """Return True when ``current → new`` is a legal state transition.

    Carrier webhooks can arrive out of order, so illegal transitions are a
    normal occurrence — callers log and drop them rather than raising.
    """
    if current == new:
        return False
    if current in TERMINAL_STATES:
        return False
    if new in TERMINAL_STATES:
        return True
    # speaking ⇄ listening may oscillate
    if {current, new} == {CallState.SPEAKING, CallState.LISTENING}:
        return True
    return _STATE_ORDER[new] > _STATE_ORDER[current]


class EventType(str, Enum):
    """Provider-agnostic webhook event types (the normalization target)."""

    CALL_INITIATED = "call.initiated"
    CALL_RINGING = "call.ringing"
    CALL_ANSWERED = "call.answered"
    CALL_ACTIVE = "call.active"
    CALL_SPEAKING = "call.speaking"  # outbound TTS started/finished
    CALL_SPEECH = "call.speech"      # inbound transcript (text, is_final)
    CALL_SILENCE = "call.silence"
    CALL_DTMF = "call.dtmf"
    CALL_ENDED = "call.ended"
    CALL_ERROR = "call.error"


# EventType → the CallState it implies (when it implies one at all).
EVENT_STATE_MAP = {
    EventType.CALL_INITIATED: CallState.INITIATED,
    EventType.CALL_RINGING: CallState.RINGING,
    EventType.CALL_ANSWERED: CallState.ANSWERED,
    EventType.CALL_ACTIVE: CallState.ACTIVE,
}


@dataclass
class NormalizedEvent:
    """A single provider webhook event, normalized.

    ``provider_call_id`` is the carrier's call identifier; ``call_id`` is
    ours (recovered from ``client_state``/metadata when the provider echoes
    it back). ``dedupe_key`` feeds replay/duplicate suppression.
    """

    type: EventType
    provider: str = ""
    provider_call_id: Optional[str] = None
    call_id: Optional[str] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    direction: Optional[Literal["inbound", "outbound"]] = None
    # call.speech
    text: Optional[str] = None
    is_final: bool = True
    # call.dtmf
    digits: Optional[str] = None
    # call.ended / call.error
    reason: Optional[str] = None
    retryable: bool = False
    # call.silence
    duration_ms: Optional[int] = None
    dedupe_key: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptEntry:
    timestamp: float
    speaker: Literal["bot", "user"]
    text: str
    is_final: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "speaker": self.speaker,
            "text": self.text,
            "is_final": self.is_final,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptEntry":
        return cls(
            timestamp=float(data.get("timestamp", 0.0)),
            speaker="bot" if data.get("speaker") == "bot" else "user",
            text=str(data.get("text", "")),
            is_final=bool(data.get("is_final", True)),
        )


def new_call_id() -> str:
    return f"vc-{uuid.uuid4().hex[:20]}"


@dataclass
class CallRecord:
    """Full state of one call. Persisted as one JSONL line per change."""

    call_id: str
    provider: str
    direction: Literal["inbound", "outbound"]
    state: CallState = CallState.INITIATED
    provider_call_id: Optional[str] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    session_key: Optional[str] = None
    mode: Literal["notify", "conversation"] = "conversation"
    started_at: float = field(default_factory=time.time)
    answered_at: Optional[float] = None
    ended_at: Optional[float] = None
    end_reason: Optional[str] = None
    transcript: List[TranscriptEntry] = field(default_factory=list)
    processed_event_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def peer_number(self) -> Optional[str]:
        """The remote party's number (the caller for inbound, callee for outbound)."""
        return self.from_number if self.direction == "inbound" else self.to_number

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "provider": self.provider,
            "direction": self.direction,
            "state": self.state.value,
            "provider_call_id": self.provider_call_id,
            "from_number": self.from_number,
            "to_number": self.to_number,
            "session_key": self.session_key,
            "mode": self.mode,
            "started_at": self.started_at,
            "answered_at": self.answered_at,
            "ended_at": self.ended_at,
            "end_reason": self.end_reason,
            "transcript": [t.to_dict() for t in self.transcript],
            "processed_event_ids": list(self.processed_event_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CallRecord":
        try:
            state = CallState(data.get("state", "initiated"))
        except ValueError:
            state = CallState.ERROR
        return cls(
            call_id=str(data.get("call_id", "")) or new_call_id(),
            provider=str(data.get("provider", "")),
            direction="inbound" if data.get("direction") == "inbound" else "outbound",
            state=state,
            provider_call_id=data.get("provider_call_id"),
            from_number=data.get("from_number"),
            to_number=data.get("to_number"),
            session_key=data.get("session_key"),
            mode="notify" if data.get("mode") == "notify" else "conversation",
            started_at=float(data.get("started_at", 0.0)),
            answered_at=data.get("answered_at"),
            ended_at=data.get("ended_at"),
            end_reason=data.get("end_reason"),
            transcript=[
                TranscriptEntry.from_dict(t)
                for t in data.get("transcript", [])
                if isinstance(t, dict)
            ],
            processed_event_ids=[str(e) for e in data.get("processed_event_ids", [])],
            metadata=dict(data.get("metadata") or {}),
        )
