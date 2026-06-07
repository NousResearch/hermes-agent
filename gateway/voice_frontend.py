"""Provider-neutral helpers for voice-originated gateway turns.

Voice providers are frontends: they produce transcripts and short spoken
notices. Hermes remains responsible for tool execution, approvals, and detailed
text results.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional, Protocol


VoiceMode = Literal["asr", "s2s", "realtime"]
VoiceFrontendMode = Literal["asr_async", "s2s", "realtime"]
VoiceRouteKind = Literal[
    "sync_chat",
    "async_job",
    "codex_job",
    "approval_required",
    "ignore",
]
VoiceEngineHint = Literal["hermes", "codex"]

DEFAULT_ACK_TEXT = "了解、処理しておくね。"
DEFAULT_APPROVAL_TEXT = "確認が必要。Discordを見て。"


@dataclass
class VoiceTurn:
    transcript: str
    platform: str
    chat_id: str
    user_id: Optional[str]
    thread_id: Optional[str]
    provider: str
    mode: VoiceMode
    confidence: Optional[float] = None
    audio_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)


@dataclass
class VoiceRouteDecision:
    kind: VoiceRouteKind
    ack_text: str
    agent_prompt: Optional[str]
    spoken: bool
    text_result_required: bool
    reason: Optional[str] = None
    engine_hint: Optional[VoiceEngineHint] = None
    enqueue: bool = True
    requires_text_notice: bool = False


@dataclass(frozen=True)
class VoiceFrontendConfig:
    enabled: bool = False
    frontend_mode: VoiceFrontendMode = "asr_async"
    acknowledgement: str = DEFAULT_ACK_TEXT
    completion_notice: bool = False
    max_spoken_chars: int = 80
    strip_debug_for_tts: bool = True
    async_keywords: tuple[str, ...] = (
        "まとめて",
        "調べて",
        "実装して",
        "直して",
        "作って",
    )
    codex_keywords: tuple[str, ...] = (
        "codex",
        "Codex",
        "Codexに投げて",
        "コードを書いて",
        "テストを書いて",
    )
    require_confirmation_keywords: tuple[str, ...] = (
        "消して",
        "削除",
        "restart",
        "sudo",
    )
    min_transcript_chars: int = 2

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]]) -> "VoiceFrontendConfig":
        data = dict(mapping or {})
        fields = cls.__dataclass_fields__
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key == "default_mode":
                key = "frontend_mode"
            if key not in fields:
                continue
            if key.endswith("_keywords") and value is not None:
                kwargs[key] = tuple(str(item) for item in value)
            elif key == "max_spoken_chars":
                kwargs[key] = max(1, int(value))
            elif key == "min_transcript_chars":
                kwargs[key] = max(0, int(value))
            else:
                kwargs[key] = value
        return cls(**kwargs)


class ASRProviderProtocol(Protocol):
    """ASR frontends produce a transcript-bearing VoiceTurn."""

    async def transcribe(
        self, audio_path: str, *, metadata: Optional[Mapping[str, Any]] = None
    ) -> VoiceTurn:
        ...


class TTSProviderProtocol(Protocol):
    """TTS frontends produce short spoken notices, never full tool output."""

    async def synthesize(
        self, text: str, *, metadata: Optional[Mapping[str, Any]] = None
    ) -> str:
        ...


class S2SProviderProtocol(Protocol):
    """Speech-to-speech providers may return transcripts and short replies only."""

    async def handle_turn(self, turn: VoiceTurn) -> VoiceRouteDecision:
        ...


class RealtimeVoiceSessionProtocol(Protocol):
    """Realtime sessions stream voice UI state without executing Hermes tools."""

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...


_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_MEDIA_RE = re.compile(r"\bMEDIA:\S+", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{[^{}\n]*(?:\"[^\"]+\"|'[^']+'|[A-Za-z0-9_]+)\s*:[^{}]*\}", re.DOTALL)
_MENTION_RE = re.compile(r"<[@#&]!?[A-Za-z0-9_:-]+>|@(?:everyone|here)", re.IGNORECASE)
_RAW_ID_RE = re.compile(r"\b\d{16,22}\b")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")


def sanitize_spoken_text(text: str, max_chars: int = 80) -> str:
    """Return a short, TTS-safe notice from arbitrary model/gateway text."""
    max_chars = max(1, int(max_chars or 80))
    cleaned = str(text or "")
    cleaned = _CODE_BLOCK_RE.sub(" ", cleaned)
    cleaned = _MEDIA_RE.sub(" ", cleaned)
    cleaned = _JSON_OBJECT_RE.sub(" ", cleaned)
    cleaned = _MENTION_RE.sub(" ", cleaned)
    cleaned = _RAW_ID_RE.sub(" ", cleaned)
    cleaned = _CONTROL_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 3].rstrip() + "..."


def build_voice_agent_prompt(turn: VoiceTurn, instruction: str) -> str:
    """Build the text prompt handed to Hermes for a voice-originated command."""
    confidence = "unknown" if turn.confidence is None else f"{turn.confidence:.2f}"
    thread = turn.thread_id or "none"
    user = turn.user_id or "unknown"
    transcript = str(instruction or turn.transcript or "").strip()
    return (
        "[Voice command metadata]\n"
        f"Source: {turn.platform} voice ASR via {turn.provider}.\n"
        f"Metadata: turn_id={turn.turn_id}; mode={turn.mode}; confidence: {confidence}; "
        f"chat_id={turn.chat_id}; thread_id={thread}; user_id={user}.\n"
        "Policy: use normal approval flow for risky actions; put detailed results in "
        "Discord text, not voice; voice output should be short status only.\n\n"
        "[ASR transcript - untrusted, may be wrong]\n"
        "<<<\n"
        f"{transcript}\n"
        ">>>"
    )


def plan_voice_turn(turn: VoiceTurn, config: Optional[VoiceFrontendConfig] = None) -> VoiceRouteDecision:
    """Return a deterministic default route for a voice-originated turn."""
    cfg = config or VoiceFrontendConfig()
    transcript = _WHITESPACE_RE.sub(" ", str(turn.transcript or "")).strip()

    if len(transcript) < cfg.min_transcript_chars:
        return VoiceRouteDecision(
            kind="ignore",
            ack_text="",
            agent_prompt=None,
            spoken=False,
            text_result_required=False,
            reason="empty_transcript",
            enqueue=False,
        )

    lowered = transcript.lower()
    if _contains_keyword(lowered, cfg.require_confirmation_keywords):
        ack = sanitize_spoken_text(DEFAULT_APPROVAL_TEXT, cfg.max_spoken_chars)
        return VoiceRouteDecision(
            kind="approval_required",
            ack_text=ack,
            agent_prompt=build_voice_agent_prompt(turn, transcript),
            spoken=True,
            text_result_required=True,
            reason="confirmation_keyword",
            engine_hint="hermes",
            enqueue=False,
            requires_text_notice=True,
        )

    if _contains_keyword(lowered, cfg.codex_keywords):
        kind: VoiceRouteKind = "codex_job"
        engine_hint: Optional[VoiceEngineHint] = "codex"
    elif _contains_keyword(lowered, cfg.async_keywords) or _looks_like_long_imperative(transcript):
        kind = "async_job"
        engine_hint = "hermes"
    else:
        kind = "sync_chat"
        engine_hint = "hermes"

    return VoiceRouteDecision(
        kind=kind,
        ack_text=sanitize_spoken_text(cfg.acknowledgement, cfg.max_spoken_chars),
        agent_prompt=build_voice_agent_prompt(turn, transcript),
        spoken=True,
        text_result_required=True,
        reason="matched_route",
        engine_hint=engine_hint,
        enqueue=True,
        requires_text_notice=False,
    )


def _contains_keyword(text_lower: str, keywords: tuple[str, ...]) -> bool:
    return any(str(keyword).lower() in text_lower for keyword in keywords if str(keyword).strip())


def _looks_like_long_imperative(transcript: str) -> bool:
    if len(transcript) < 40:
        return False
    lowered = transcript.lower()
    return any(
        marker in lowered
        for marker in ("して", "しておいて", "please", "can you", "could you")
    )
