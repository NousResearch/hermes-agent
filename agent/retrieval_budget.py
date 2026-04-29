from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


ALLOWED_ESCALATION_OUTCOMES = {"empty", "timed_out", "too_broad", "invalid_scope"}


@dataclass
class RetrievalBudget:
    max_retrieval_calls: int
    max_broad_search_calls: int
    max_subtree_expansions: int
    max_total_retrieval_seconds: float
    recommended_sequence: list[str] = field(default_factory=list)
    allow_broad_search: bool = False
    retrieval_calls: int = 0
    broad_search_calls: int = 0
    subtree_expansions: int = 0
    total_retrieval_seconds: float = 0.0
    attempted_stages: list[str] = field(default_factory=list)
    last_outcome_by_stage: Dict[str, str] = field(default_factory=dict)

    def can_attempt(self, stage: str) -> tuple[bool, Optional[str]]:
        if self.retrieval_calls >= self.max_retrieval_calls:
            return False, "max_retrieval_calls"
        if self.total_retrieval_seconds >= self.max_total_retrieval_seconds:
            return False, "max_total_retrieval_seconds"
        if stage == "broad_search":
            if not self.allow_broad_search:
                return False, "broad_search_disabled"
            if self.broad_search_calls >= self.max_broad_search_calls:
                return False, "max_broad_search_calls"
        if self.recommended_sequence and stage not in self.recommended_sequence:
            if stage != "broad_search":
                return False, "stage_not_planned"
        if stage == "known_subtree" and self.subtree_expansions >= self.max_subtree_expansions:
            return False, "max_subtree_expansions"
        if self.recommended_sequence and stage in self.recommended_sequence:
            idx = self.recommended_sequence.index(stage)
            if idx > 0:
                previous_stage = self.recommended_sequence[idx - 1]
                if previous_stage not in self.last_outcome_by_stage:
                    return False, "stage_order"
                previous_outcome = self.last_outcome_by_stage.get(previous_stage)
                if previous_outcome == "success":
                    return False, "previous_stage_succeeded"
                if previous_outcome not in ALLOWED_ESCALATION_OUTCOMES:
                    return False, "stage_order"
        return True, None

    def record_attempt(self, stage: str, tool: str, *, seconds: float, outcome: str) -> None:
        self.retrieval_calls += 1
        self.total_retrieval_seconds += max(0.0, float(seconds))
        self.last_outcome_by_stage[stage] = outcome
        self.attempted_stages.append(stage)
        if stage == "broad_search":
            self.broad_search_calls += 1
        if stage == "known_subtree":
            self.subtree_expansions += 1

    def remaining_calls(self) -> int:
        return max(0, self.max_retrieval_calls - self.retrieval_calls)

    def as_dict(self) -> dict:
        return {
            "max_retrieval_calls": self.max_retrieval_calls,
            "max_broad_search_calls": self.max_broad_search_calls,
            "max_subtree_expansions": self.max_subtree_expansions,
            "max_total_retrieval_seconds": self.max_total_retrieval_seconds,
            "recommended_sequence": list(self.recommended_sequence),
            "allow_broad_search": self.allow_broad_search,
            "retrieval_calls": self.retrieval_calls,
            "broad_search_calls": self.broad_search_calls,
            "subtree_expansions": self.subtree_expansions,
            "total_retrieval_seconds": self.total_retrieval_seconds,
            "last_outcome_by_stage": dict(self.last_outcome_by_stage),
        }
