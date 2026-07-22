"""Per-turn provenance and automatic-memory retention policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


TURN_MEMORY_DISPOSITION_KEY = "_turn_memory_disposition"


class TurnMemoryDisposition(str, Enum):
    """Automatic-memory retention policy for one user turn."""

    RETAIN = "retain"
    DO_NOT_RETAIN = "do_not_retain"


@dataclass(frozen=True)
class TurnProvenance:
    """Semantic origin of a user turn at the shared memory boundary."""

    kind: str
    memory_disposition: TurnMemoryDisposition = TurnMemoryDisposition.RETAIN


NORMAL_USER_TURN = TurnProvenance(kind="user")
ASYNC_DELEGATION_COMPLETION_TURN = TurnProvenance(
    kind="async_delegation_completion",
    memory_disposition=TurnMemoryDisposition.DO_NOT_RETAIN,
)


def normalize_turn_provenance(value: TurnProvenance | None) -> TurnProvenance:
    """Return the effective provenance for a turn."""

    return value if isinstance(value, TurnProvenance) else NORMAL_USER_TURN


def should_retain_turn_memory(value: TurnProvenance | None) -> bool:
    """Return whether automatic memory retention is enabled for the turn."""

    return (
        normalize_turn_provenance(value).memory_disposition
        == TurnMemoryDisposition.RETAIN
    )


def stamp_turn_provenance(message: dict[str, Any], value: TurnProvenance | None) -> dict[str, Any]:
    """Stamp the internal per-turn memory marker onto a message dict."""

    disposition = normalize_turn_provenance(value).memory_disposition
    if disposition == TurnMemoryDisposition.RETAIN:
        message.pop(TURN_MEMORY_DISPOSITION_KEY, None)
        return message
    message[TURN_MEMORY_DISPOSITION_KEY] = disposition.value
    return message


def get_message_turn_memory_disposition(message: dict[str, Any] | None) -> TurnMemoryDisposition:
    """Return a message's retention policy, defaulting legacy rows to retain."""

    raw = (message or {}).get(TURN_MEMORY_DISPOSITION_KEY)
    if raw == TurnMemoryDisposition.DO_NOT_RETAIN.value:
        return TurnMemoryDisposition.DO_NOT_RETAIN
    return TurnMemoryDisposition.RETAIN
