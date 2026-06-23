"""Typed contracts for Hermes knowledge routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class KnowledgeDestination(StrEnum):
    STATIC_KNOWLEDGE = "static_knowledge"
    DYNAMIC_MEMORY = "dynamic_memory"
    NONE = "none"
    REVIEW = "review"


class RouteAction(StrEnum):
    WRITE_DOCUMENT = "write_document"
    RETAIN_MEMORY = "retain_memory"
    SKIP = "skip"
    DRY_RUN_ONLY = "dry_run_only"


class DuplicatePolicy(StrEnum):
    UPDATE_EXISTING = "update_existing"
    CREATE_NEW = "create_new"
    SKIP_EXISTING = "skip_existing"
    REVIEW = "review"


def _normalize_content_type(value: str) -> str:
    return (value or "").strip().lower().replace("-", "_")


def normalize_tags(tags: tuple[str, ...] | list[str] | set[str] | str | None) -> tuple[str, ...]:
    if tags is None:
        raw: list[str] = []
    elif isinstance(tags, str):
        raw = [part.strip() for part in tags.split(",")]
    else:
        raw = [str(part).strip() for part in tags]
    return tuple(dict.fromkeys(part for part in raw if part))


@dataclass(frozen=True)
class KnowledgeWriteRequest:
    content_type: str
    content: str
    title: str = ""
    context: str = ""
    tags: tuple[str, ...] | list[str] | set[str] | str | None = field(default_factory=tuple)
    dry_run: bool = False
    destination_override: KnowledgeDestination | str | None = None
    idempotency_key: str = ""
    update_mode: str = ""
    duplicate_policy: DuplicatePolicy | str = DuplicatePolicy.UPDATE_EXISTING
    metadata: Mapping[str, Any] = field(default_factory=dict)
    notebook: str = ""
    path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_type", _normalize_content_type(self.content_type))
        object.__setattr__(self, "tags", normalize_tags(self.tags))
        if isinstance(self.duplicate_policy, str):
            object.__setattr__(self, "duplicate_policy", DuplicatePolicy(self.duplicate_policy))
        if isinstance(self.destination_override, str) and self.destination_override:
            object.__setattr__(self, "destination_override", KnowledgeDestination(self.destination_override))


@dataclass(frozen=True)
class RouteDecision:
    destination: KnowledgeDestination
    action: RouteAction
    reason: str
    requires_title: bool = False
    snapshot: bool = False
    content_type: str = ""
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["destination"] = str(self.destination)
        data["action"] = str(self.action)
        return data


@dataclass(frozen=True)
class KnowledgeWriteResult:
    success: bool
    destination: KnowledgeDestination | str
    action: str
    written: bool = False
    dry_run: bool = False
    id: str = ""
    path: str = ""
    backend: str = ""
    decision: RouteDecision | dict[str, Any] | None = None
    existing: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    result: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["destination"] = str(self.destination)
        if isinstance(self.decision, RouteDecision):
            data["decision"] = self.decision.to_dict()
        return data
