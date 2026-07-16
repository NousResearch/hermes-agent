"""Strategic/technical memory routing through MemoryManager."""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent.beta.specialists import SpecialistRegistry, default_specialist_registry


class MemoryScope(StrEnum):
    STRATEGIC = "strategic"
    TECHNICAL = "technical"


class MemoryRoute(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope: MemoryScope
    category: str
    specialist_id: str | None = None

    @property
    def tag(self) -> str:
        return "beta:strategic" if self.scope == MemoryScope.STRATEGIC else f"beta:specialist:{self.specialist_id}"


_STRATEGIC_CATEGORIES = frozenset(
    {"preference", "goal", "decision", "priority", "operating_rule", "team_structure"}
)
_STRATEGIC_PATTERN = re.compile(
    r"\b(prefer|preferencia|preference|objetivo|goal|decisao|decision|prioridade|priority|regra|rule|equipe|team|chefe|chief)\w*\b",
    re.IGNORECASE,
)


def _terms(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return set(re.findall(r"[a-z0-9]+", normalized))


def classify_memory(
    content: str,
    *,
    category: str | None = None,
    specialist_id: str | None = None,
    registry: SpecialistRegistry | None = None,
) -> MemoryRoute:
    """Route durable knowledge to strategic or specialist-owned memory."""
    if category in _STRATEGIC_CATEGORIES or (category is None and _STRATEGIC_PATTERN.search(content)):
        return MemoryRoute(scope=MemoryScope.STRATEGIC, category=category or "decision")

    registry = registry or default_specialist_registry()
    if specialist_id is None:
        content_terms = _terms(content)
        ranked = sorted(
            (
                (
                    len(content_terms.intersection(_terms(" ".join((*specialist.capabilities, *specialist.keywords))))),
                    specialist.id,
                )
                for specialist in registry.enabled()
            ),
            reverse=True,
        )
        specialist_id = ranked[0][1] if ranked and ranked[0][0] else "memory"
    if registry.get(specialist_id) is None:
        raise ValueError(f"unknown specialist memory scope: {specialist_id}")
    return MemoryRoute(
        scope=MemoryScope.TECHNICAL,
        category=category or "technical_fact",
        specialist_id=specialist_id,
    )


class ScopedMemory:
    """Policy boundary over an existing MemoryManager instance."""

    def __init__(self, manager: Any, registry: SpecialistRegistry | None = None):
        self.manager = manager
        self.registry = registry or default_specialist_registry()

    def retain(
        self,
        content: str,
        *,
        actor: str = "beta",
        category: str | None = None,
        specialist_id: str | None = None,
    ) -> Any:
        route = classify_memory(
            content,
            category=category,
            specialist_id=specialist_id,
            registry=self.registry,
        )
        if actor != "beta":
            specialist = self.registry.get(actor)
            if specialist is None or specialist.memory_access != "read_write":
                raise PermissionError(f"specialist {actor} cannot write memory")
            if route.scope == MemoryScope.STRATEGIC:
                raise PermissionError("specialists cannot write Beta strategic memory")
            if route.specialist_id != actor:
                raise PermissionError("specialists can write only their own technical memory")

        metadata = {
            "beta_memory_scope": route.scope.value,
            "beta_memory_category": route.category,
            "beta_specialist_id": route.specialist_id or "",
        }
        if self.manager.has_tool("hindsight_retain"):
            return self.manager.handle_tool_call(
                "hindsight_retain",
                {
                    "content": content,
                    "context": f"Beta {route.scope.value} memory",
                    "tags": ["beta-memory", route.tag, f"category:{route.category}"],
                },
            )
        self.manager.on_memory_write("add", "memory", content, metadata=metadata)
        return {"success": True, "scope": route.tag}

    def recall(
        self,
        query: str,
        *,
        actor: str = "beta",
        specialist_id: str | None = None,
    ) -> Any:
        if actor != "beta":
            specialist = self.registry.get(actor)
            if specialist is None or specialist.memory_access == "none":
                raise PermissionError(f"specialist {actor} cannot read memory")
            if specialist_id not in {None, actor}:
                raise PermissionError("specialists can read only their own technical memory")
            specialist_id = actor
        route = (
            MemoryRoute(scope=MemoryScope.TECHNICAL, category="technical_fact", specialist_id=specialist_id)
            if specialist_id
            else MemoryRoute(scope=MemoryScope.STRATEGIC, category="decision")
        )
        if self.manager.has_tool("hindsight_recall"):
            return self.manager.handle_tool_call(
                "hindsight_recall",
                {"query": query, "tags": [route.tag], "tags_match": "all_strict"},
            )
        return self.manager.prefetch_all(f"[{route.tag}] {query}")

