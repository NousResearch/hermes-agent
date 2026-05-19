"""ContextOps Epistemic State Engine substrate models.

These models intentionally keep ContextOps concepts separate from generic
conversation memory: threads are evidence-anchored trajectories, context packs
are restore/avoid contracts, and state deltas are epistemic changes with either
evidence or an explicit low-confidence marker.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _non_empty(values: list[str], field_name: str) -> list[str]:
    cleaned = [value.strip() for value in values if value and value.strip()]
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


class ContextOpsModel(BaseModel):
    """Base model with strict-ish assignment and compact JSON-friendly output."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Event(ContextOpsModel):
    """An observed local/offline event that can support later epistemic state."""

    id: str
    source: str
    text: str
    refs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "source", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class Thread(ContextOpsModel):
    """A specific evidence-anchored thread; not a broad topic bucket."""

    id: str
    anchor_event_ids: list[str]
    stance: str
    heat: float = Field(default=0.0, ge=0.0, le=1.0)
    status: Literal["open", "resolved", "dormant"] = "open"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _reject_topic_only_ids(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("thread id must not be empty")
        if not value.startswith("thread:") or value.count(":") < 2:
            raise ValueError(
                "thread id appears topic-only; use a specific evidence-anchored id "
                "such as 'thread:<source>:<scope>:<anchor>'"
            )
        return value

    @field_validator("anchor_event_ids")
    @classmethod
    def _anchor_events_required(cls, value: list[str]) -> list[str]:
        return _non_empty(value, "anchor_event_ids")

    @field_validator("stance")
    @classmethod
    def _stance_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("stance must not be empty")
        return value


class Tension(ContextOpsModel):
    """An unresolved pressure or contradiction inside a specific thread."""

    id: str
    thread_id: str
    description: str
    evidence_refs: list[str] = Field(default_factory=list)
    status: Literal["open", "resolved"] = "open"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "thread_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class ContextPack(ContextOpsModel):
    """A compact restore/avoid contract, not a transcript or generic summary."""

    id: str
    thread_ids: list[str]
    restore: list[str]
    avoid: list[str]
    event_ids: list[str] = Field(default_factory=list)
    tension_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("id must not be empty")
        return value

    @field_validator("thread_ids")
    @classmethod
    def _thread_ids_required(cls, value: list[str]) -> list[str]:
        return _non_empty(value, "thread_ids")

    @field_validator("restore")
    @classmethod
    def _restore_required(cls, value: list[str]) -> list[str]:
        return _non_empty(value, "restore")

    @field_validator("avoid")
    @classmethod
    def _avoid_required(cls, value: list[str]) -> list[str]:
        return _non_empty(value, "avoid")


class StateDelta(ContextOpsModel):
    """An epistemic state change with evidence or explicit uncertainty."""

    id: str
    kind: str
    description: str
    evidence_refs: list[str] = Field(default_factory=list)
    low_confidence: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "kind", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @model_validator(mode="after")
    def _evidence_or_low_confidence(self) -> "StateDelta":
        if not self.evidence_refs and not self.low_confidence:
            raise ValueError("state delta requires evidence refs unless low_confidence is true")
        return self
