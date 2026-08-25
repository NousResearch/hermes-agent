"""Shared, fail-closed identity and lifecycle rules for message Tapbacks.

A Tapback operation carries only exact routing identifiers.  It deliberately has
no message-text, participant-list, chat-alias, or fallback fields: callers must
resolve one authoritative chat and one target message before constructing it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any


class TapbackValidationError(ValueError):
    """A Tapback cannot be represented as one unambiguous operation."""


class TapbackType(StrEnum):
    """Native iMessage Tapbacks supported by BlueBubbles."""

    LOVE = "love"
    LIKE = "like"
    DISLIKE = "dislike"
    LAUGH = "laugh"
    EMPHASIZE = "emphasize"
    QUESTION = "question"


class TapbackAction(StrEnum):
    """Whether the sender adds or removes the named Tapback."""

    ADD = "added"
    REMOVE = "removed"


class TapbackDirection(StrEnum):
    """The transport direction of the operation."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"


class TapbackStatus(StrEnum):
    """Processing lifecycle for an immutable Tapback operation."""

    RECEIVED = "received"
    VALIDATED = "validated"
    PENDING = "pending"
    PROCESSING = "processing"
    APPLIED = "applied"
    REJECTED = "rejected"
    FAILED = "failed"


_ALLOWED_STATUS_TRANSITIONS: dict[TapbackStatus, frozenset[TapbackStatus]] = {
    TapbackStatus.RECEIVED: frozenset(
        {TapbackStatus.VALIDATED, TapbackStatus.REJECTED}
    ),
    TapbackStatus.VALIDATED: frozenset(
        {TapbackStatus.PENDING, TapbackStatus.REJECTED}
    ),
    TapbackStatus.PENDING: frozenset(
        {TapbackStatus.PROCESSING, TapbackStatus.REJECTED}
    ),
    TapbackStatus.PROCESSING: frozenset(
        {TapbackStatus.APPLIED, TapbackStatus.FAILED}
    ),
    # A failed provider or handler attempt is retryable with the same identity.
    TapbackStatus.FAILED: frozenset(
        {TapbackStatus.PENDING, TapbackStatus.REJECTED}
    ),
    TapbackStatus.APPLIED: frozenset(),
    TapbackStatus.REJECTED: frozenset(),
}


def _require_exact_identifier(name: str, value: Any) -> str:
    """Accept one nonempty scalar identifier and reject ambiguous targets."""
    if not isinstance(value, str):
        raise TapbackValidationError(f"{name} must be one exact string identifier")
    normalized = value.strip()
    if not normalized:
        raise TapbackValidationError(f"{name} is required")
    if "\x00" in normalized:
        raise TapbackValidationError(f"{name} contains an invalid separator")
    return normalized


@dataclass(frozen=True, slots=True)
class TapbackOperation:
    """One exact inbound or outbound Tapback operation.

    ``deduplication_key`` identifies a full source event, including add/remove.
    ``state_key`` identifies the current reaction slot, intentionally excluding
    action and source-event identity.  A state ledger can therefore suppress an
    adjacent replay while still accepting add -> remove -> add, even when a
    provider reuses the same source event GUID for those updates.
    """

    platform: str
    chat_id: str
    target_message_id: str
    sender_id: str
    reaction: TapbackType
    action: TapbackAction
    direction: TapbackDirection
    source_event_id: str
    part_index: int = 0
    status: TapbackStatus = TapbackStatus.RECEIVED

    def __post_init__(self) -> None:
        for field_name in (
            "platform",
            "chat_id",
            "target_message_id",
            "sender_id",
            "source_event_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_exact_identifier(field_name, getattr(self, field_name)),
            )
        if not isinstance(self.reaction, TapbackType):
            raise TapbackValidationError("reaction must be a supported TapbackType")
        if not isinstance(self.action, TapbackAction):
            raise TapbackValidationError("action must be a TapbackAction")
        if not isinstance(self.direction, TapbackDirection):
            raise TapbackValidationError("direction must be a TapbackDirection")
        if not isinstance(self.status, TapbackStatus):
            raise TapbackValidationError("status must be a TapbackStatus")
        if (
            isinstance(self.part_index, bool)
            or not isinstance(self.part_index, int)
            or self.part_index < 0
        ):
            raise TapbackValidationError(
                "part_index must be a non-negative integer"
            )

    @property
    def state_key(self) -> tuple[str, str, str, int, str, str]:
        """Return the exact reaction slot, isolated by chat and sender."""
        return (
            self.platform,
            self.chat_id,
            self.target_message_id,
            self.part_index,
            self.sender_id,
            self.direction.value,
        )

    @property
    def deduplication_key(self) -> str:
        """Return a deterministic, transport-safe identity for this event."""
        canonical = json.dumps(
            {
                "action": self.action.value,
                "chat_id": self.chat_id,
                "direction": self.direction.value,
                "part_index": self.part_index,
                "platform": self.platform,
                "reaction": self.reaction.value,
                "sender_id": self.sender_id,
                "source_event_id": self.source_event_id,
                "target_message_id": self.target_message_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def transition_to(self, status: TapbackStatus) -> TapbackOperation:
        """Return the operation at an allowed next processing status."""
        if not isinstance(status, TapbackStatus):
            raise ValueError("Tapback status transitions require TapbackStatus")
        if status not in _ALLOWED_STATUS_TRANSITIONS[self.status]:
            raise ValueError(
                f"Tapback cannot transition from {self.status.value} to {status.value}"
            )
        return replace(self, status=status)

    def to_platform_payload(self) -> dict[str, Any]:
        """Serialize the exact, text-free reaction boundary payload."""
        return {
            "action": self.action.value,
            "reaction": self.reaction.value,
            "message_id": self.target_message_id,
            "reaction_message_id": self.source_event_id,
            "part_index": self.part_index,
            "chat_id": self.chat_id,
            "user_id": self.sender_id,
            "direction": self.direction.value,
            "status": self.status.value,
            "deduplication_key": self.deduplication_key,
        }
