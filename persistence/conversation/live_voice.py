"""Trusted GPT-Live transcript fragments and their canonical publication selection."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VoiceFragment:
    event_id: str
    role: Literal['user', 'assistant']
    text: str
    start_ms: float | None
    end_ms: float | None


@dataclass(frozen=True)
class VoiceTurn:
    event_ids: tuple[str, ...]
    partial: bool = False
