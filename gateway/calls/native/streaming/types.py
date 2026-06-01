from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from gateway.calls.native.voice_turn import VoiceDebugTracePolicy


class TranscriptKind(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"


class TurnEventKind(str, Enum):
    USER_SPEECH_STARTED = "user_speech_started"
    USER_SPEECH_STOPPED = "user_speech_stopped"
    ENDPOINT_DETECTED = "endpoint_detected"
    POSSIBLE_BACKCHANNEL = "possible_backchannel"


class BrainEventKind(str, Enum):
    PARTIAL_TEXT = "partial_text"
    TOOL_STATUS = "tool_status"
    FINAL_TEXT = "final_text"
    ERROR = "error"


class TtsEventKind(str, Enum):
    AUDIO = "audio"
    MARK = "mark"
    DONE = "done"
    CANCELLED = "cancelled"


class InterruptionAction(str, Enum):
    INTERRUPT = "interrupt"
    IGNORE = "ignore"
    RESUME = "resume"
    WAIT = "wait"


class TurnEndReason(str, Enum):
    COMPLETED = "completed"
    BARGED_IN = "barged_in"
    FALSE_INTERRUPTION = "false_interruption"
    BRAIN_ERROR = "brain_error"


@dataclass(frozen=True)
class MediaFormat:
    sample_rate: int
    channels: int = 1
    frame_ms: int = 20


@dataclass(frozen=True)
class AudioFrame:
    pcm16: bytes
    media: MediaFormat
    timestamp_ms: int
    seq: int

    @property
    def duration_ms(self) -> int:
        samples = len(self.pcm16) / 2 / self.media.channels
        return int(samples / self.media.sample_rate * 1000)


@dataclass(frozen=True)
class WordTiming:
    word: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0


@dataclass(frozen=True)
class TranscriptEvent:
    call_id: str
    kind: TranscriptKind
    text: str
    start_ms: int = 0
    end_ms: int = 0
    stability: float = 1.0
    words: tuple[WordTiming, ...] = ()
    provider: str = ""


@dataclass(frozen=True)
class TurnEvent:
    call_id: str
    kind: TurnEventKind
    at_ms: int
    speech_duration_ms: int = 0
    vad_confidence: float = 0.0
    endpoint_confidence: float = 0.0
    source: str = ""


@dataclass(frozen=True)
class BrainEvent:
    call_id: str
    kind: BrainEventKind
    text: str = ""
    tool_name: str = ""
    error_code: str = ""

    @property
    def is_final(self) -> bool:
        return self.kind is BrainEventKind.FINAL_TEXT


@dataclass(frozen=True)
class PlaybackMark:
    call_id: str
    char_offset: int
    text_so_far: str
    at_ms: int
    boundary: str = "word"


@dataclass(frozen=True)
class TtsAudioEvent:
    call_id: str
    kind: TtsEventKind
    frame: "AudioFrame | None" = None
    mark: "PlaybackMark | None" = None
    span_text: str = ""
    span_start_char: int = 0
    span_end_char: int = 0


@dataclass(frozen=True)
class InterruptionParams:
    min_speech_ms: int = 350
    min_words: int = 2
    backchannel_max_ms: int = 600
    false_interruption_timeout_ms: int = 2000


@dataclass(frozen=True)
class EndpointParams:
    vad_confidence: float = 0.7
    start_secs: float = 0.2
    stop_secs: float = 0.2
    endpoint_threshold: float = 0.5
    max_delay_ms: int = 3000


@dataclass(frozen=True)
class InterruptionSignal:
    call_id: str
    at_ms: int
    assistant_speaking: bool
    turn_event: "TurnEvent | None"
    latest_partial: "TranscriptEvent | None"
    playhead: "PlaybackMark | None"
    params: InterruptionParams
    ms_since_speech_start: int = 0
    ms_since_assistant_silent_partial: int = 0


@dataclass(frozen=True)
class InterruptionDecision:
    action: InterruptionAction
    reason: str
    at_ms: int


@dataclass(frozen=True)
class FlushResult:
    dropped_frames: int
    dropped_ms: int
    last_sent_mark: "PlaybackMark | None"


@dataclass(frozen=True)
class StreamingCallContext:
    call_id: str
    contact_id: str
    session_id: str
    media: MediaFormat
    interruption: InterruptionParams = field(default_factory=InterruptionParams)
    endpoint: EndpointParams = field(default_factory=EndpointParams)
    debug: VoiceDebugTracePolicy = field(default_factory=VoiceDebugTracePolicy)


@dataclass(frozen=True)
class CallTurnRecord:
    call_id: str
    turn_index: int
    user_transcript: str
    assistant_heard_text: str
    assistant_abandoned_text: str = ""
    interrupted: bool = False
    ended_reason: TurnEndReason = TurnEndReason.COMPLETED
