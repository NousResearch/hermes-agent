"""Adapter contracts for knowledge routing backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from knowledge.types import KnowledgeWriteRequest


@dataclass(frozen=True)
class ExistingKnowledge:
    id: str
    title: str = ""
    path: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WriteResult:
    success: bool
    backend: str
    action: str
    id: str = ""
    path: str = ""
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KnowledgeAdapter(Protocol):
    @property
    def name(self) -> str: ...

    def search_existing(self, request: KnowledgeWriteRequest) -> list[ExistingKnowledge]:
        return []

    def write(self, request: KnowledgeWriteRequest) -> WriteResult: ...

    def update(self, existing: ExistingKnowledge, request: KnowledgeWriteRequest) -> WriteResult:
        return self.write(request)
