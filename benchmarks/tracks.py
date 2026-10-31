"""Track and capability metadata for benchmark categories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass(frozen=True)
class CategorySpec:
    track: str
    required_capabilities: List[str] = field(default_factory=list)
    in_core_score: bool = False
    description: str = ""


CATEGORY_SPECS = {
    "semantic_recall": CategorySpec("core", [], True, "Basic semantic retrieval quality"),
    "contradictions": CategorySpec("core", [], True, "Currentness and contradiction handling"),
    "cross_reference": CategorySpec("core", [], True, "Combining multiple relevant facts"),
    "importance_filtering": CategorySpec("core", [], True, "Signal over noise discrimination"),
    "adversarial": CategorySpec("core", [], True, "Resilience to malicious memory content"),
    "scale": CategorySpec("core", [], True, "Retrieval quality at larger memory counts"),
    "conversation_memory": CategorySpec("core", [], True, "Multi-turn conversational recall"),
    "topic_shift_recall": CategorySpec("core", [], True, "Recovering the right context after a topic pivot"),
    "deduplication": CategorySpec("core", [], True, "Duplicate handling and retrieval cleanliness"),
    "capacity_stress": CategorySpec("core", [], False, "Stress retrieval under larger fact loads"),
    "temporal_decay": CategorySpec("temporal", ["time_simulation"], False, "Time-sensitive ranking and forgetting"),
    "consolidation": CategorySpec("temporal", ["consolidation", "time_simulation", "access_rehearsal"], False, "Memory consolidation cycles"),
    "compression": CategorySpec("temporal", ["consolidation", "time_simulation"], False, "Compression and value preservation"),
    "typed_decay": CategorySpec("temporal", ["typed_facts", "time_simulation"], False, "Decay differences by fact type"),
    "scopes": CategorySpec("structured", ["scopes"], False, "Scope isolation and global access"),
    "supersession": CategorySpec("structured", ["supersession"], False, "Automatic replacement of outdated structured facts"),
    "scope_lifecycle": CategorySpec("structured", ["scopes", "consolidation", "time_simulation"], False, "Scope lifecycle changes over time"),
    "notation_parsing": CategorySpec("structured", ["typed_facts"], False, "Structured notation ingestion and recall"),
    "integration": CategorySpec("core", [], False, "End-to-end memory lifecycle behavior"),
    "compression_survival": CategorySpec("lifecycle", [], False, "Recall after simulated context compression"),
    "delegation_memory": CategorySpec("lifecycle", [], False, "Recall outcomes from delegated work"),
    "qlearning": CategorySpec("core", [], False, "Feedback-driven retrieval improvement"),
}


def get_category_spec(category: str) -> CategorySpec:
    return CATEGORY_SPECS.get(category, CategorySpec(track="uncategorized"))


def categories_for_track(track: str) -> list[str]:
    return [name for name, spec in CATEGORY_SPECS.items() if spec.track == track]


def core_categories() -> list[str]:
    return [name for name, spec in CATEGORY_SPECS.items() if spec.in_core_score]


def missing_capabilities(capabilities, category: str) -> list[str]:
    spec = get_category_spec(category)
    return [cap for cap in spec.required_capabilities if not getattr(capabilities, cap, False)]


def backend_supports_category(capabilities, category: str) -> bool:
    return not missing_capabilities(capabilities, category)


def executed_tracks(categories: Iterable[str]) -> list[str]:
    seen = []
    for category in categories:
        track = get_category_spec(category).track
        if track not in seen:
            seen.append(track)
    return seen