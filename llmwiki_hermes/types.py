"""Shared enums and type aliases."""

from enum import StrEnum


class NoteKind(StrEnum):
    SOURCE = "source"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"


class MemoryType(StrEnum):
    AUTO = "auto"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
