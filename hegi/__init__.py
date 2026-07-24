"""HEGI research meeting automation."""

from .models import (
    ActionItem,
    MeetingEpisode,
    MeetingMinutes,
    MemoryEvaluation,
    SourceMessage,
)

__all__ = [
    "ActionItem",
    "MeetingEpisode",
    "MeetingMinutes",
    "MemoryEvaluation",
    "SourceMessage",
]

__version__ = "2.0.2"
