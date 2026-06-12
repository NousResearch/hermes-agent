"""Realtime voice session abstraction.

Hermes has no first-class realtime voice provider registry, so the
voice_call plugin carries its own (modeled on ``plugins/google_meet/
realtime`` but asyncio-native): a :class:`RealtimeVoiceSession` is one live
speech-to-speech connection — caller PCM in, model PCM out, plus
transcripts, barge-in signals, and tool calls.
"""

import abc
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict

AGENT_CONSULT_TOOL = {
    "name": "agent_consult",
    "description": (
        "Ask the Hermes agent a question you cannot answer from the "
        "conversation alone — anything needing the user's data, memory, "
        "files, or up-to-date information. Tell the caller you are "
        "checking, then call this."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question, with any context the agent needs.",
            }
        },
        "required": ["question"],
    },
}

END_CALL_TOOL = {
    "name": "end_call",
    "description": (
        "Hang up the phone call. Use when the caller says goodbye, asks you "
        "to hang up, or the conversation is clearly finished. Say a brief "
        "goodbye FIRST, then call this — the line closes a moment after."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Short reason (e.g. 'caller said goodbye').",
            }
        },
        "required": [],
    },
}

DEFAULT_INSTRUCTIONS = (
    "You are Hermes, a helpful assistant speaking on a live phone call. "
    "Reply briefly and naturally, in a conversational spoken style. Use the "
    "agent_consult tool when you need the user's data or fresh information. "
    "Never read secrets, tokens, or credentials aloud."
)

# Always appended to the session instructions (even user-configured ones):
# how to sound while a consult is running. Without this the model
# improvises negatively ("I don't have the result yet").
WAITING_ETIQUETTE = (
    " While a lookup is in progress and the caller says something, respond "
    "briefly, warmly, and positively — for example 'Still checking, just a "
    "couple more seconds' or 'Almost there'. Never say you don't have the "
    "result, never say it failed or is unavailable while it is still "
    "running, and never offer to get back to them later — the answer "
    "arrives in this call. Vary the phrasing; don't repeat the same filler "
    "twice in a row. When the result arrives, deliver it directly without "
    "re-announcing that you were checking. When the caller says goodbye, "
    "asks you to hang up, or the conversation has clearly ended, say a "
    "brief goodbye and then call the end_call tool — do not leave the line "
    "open."
)


@dataclass
class RealtimeEvent:
    """One event from a realtime session."""

    type: str          # audio | transcript | speech_started | response_done
                       # | tool_call | error | closed
    audio: bytes = b""             # PCM16 at output_sample_rate (audio)
    text: str = ""                 # transcript text / error detail
    role: str = ""                 # transcript speaker: "user" | "assistant"
    is_final: bool = True
    tool_call_id: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)


class RealtimeVoiceSession(abc.ABC):
    """One live speech-to-speech connection to a realtime voice model."""

    name: str = ""
    input_sample_rate: int = 24000   # PCM16 rate the model expects in
    output_sample_rate: int = 24000  # PCM16 rate the model produces
    # "pcm16": bridge transcodes carrier µ-law ⇄ PCM16 at the rates above.
    # "ulaw": the model speaks G.711 µ-law @ 8 kHz natively (OpenAI
    # audio/pcmu) and the bridge passes carrier frames straight through.
    audio_wire_format: str = "pcm16"
    # True while the model has a response in flight (set from
    # response.created/turn events). The bridge uses this to decide whether
    # caller speech is barge-in (interrupt) or just a normal turn, and to
    # defer tool results until the current utterance finishes.
    response_active: bool = False

    @abc.abstractmethod
    async def connect(self) -> None:
        """Open the websocket and configure the session."""

    @abc.abstractmethod
    async def send_audio(self, pcm16: bytes) -> None:
        """Stream caller audio (PCM16 @ input_sample_rate) to the model."""

    @abc.abstractmethod
    def events(self) -> AsyncIterator[RealtimeEvent]:
        """Yield model events until the session closes."""

    @abc.abstractmethod
    async def send_tool_result(self, tool_call_id: str, result: str) -> None:
        """Return a tool result so the model can speak the answer."""

    @abc.abstractmethod
    async def inject_text(self, text: str) -> None:
        """Ask the model to say something (e.g. the inbound greeting)."""

    @abc.abstractmethod
    async def cancel_response(self) -> None:
        """Stop the in-flight model response (barge-in)."""

    @abc.abstractmethod
    async def close(self) -> None: ...
