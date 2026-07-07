"""Report-only context retention evaluation for compression golden tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from agent.model_metadata import estimate_tokens_rough


@dataclass(frozen=True)
class CriticalFact:
    """A fact that must survive compaction, optionally via a recoverable ref."""

    id: str
    text: str
    source_ref: str | None = None


@dataclass(frozen=True)
class ContextGoldenTranscript:
    """Minimal golden fixture for report-only retention checks."""

    name: str
    raw_context: str
    compacted_context: str
    required_facts: list[CriticalFact]
    refs: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextRetentionResult:
    name: str
    raw_tokens: int
    compressed_tokens: int
    critical_fact_recall: float
    ref_recoverability: float
    missing_fact_ids: list[str]
    unresolved_ref_ids: list[str]
    passed: bool


def _fact_survives(fact: CriticalFact, compacted_context: str, refs: Mapping[str, str]) -> bool:
    if fact.text and fact.text in compacted_context:
        return True
    if fact.source_ref and fact.source_ref in refs:
        ref_value = str(refs.get(fact.source_ref) or "")
        if fact.text and fact.text in ref_value:
            return True
        if fact.source_ref in compacted_context:
            return True
    return False


def evaluate_context_retention(golden: ContextGoldenTranscript) -> ContextRetentionResult:
    """Evaluate whether a compacted context preserved required facts and refs.

    This is deterministic and model-free by design: P0 needs CI-safe fixtures
    before any runtime compression policy changes.
    """
    required = list(golden.required_facts or [])
    missing = [
        fact.id
        for fact in required
        if not _fact_survives(fact, golden.compacted_context, golden.refs)
    ]
    if required:
        critical_fact_recall = (len(required) - len(missing)) / len(required)
    else:
        critical_fact_recall = 1.0

    required_ref_ids = [fact.source_ref for fact in required if fact.source_ref]
    unresolved_ref_ids = [ref_id for ref_id in required_ref_ids if ref_id not in golden.refs]
    if required_ref_ids:
        ref_recoverability = (len(required_ref_ids) - len(unresolved_ref_ids)) / len(required_ref_ids)
    else:
        ref_recoverability = 1.0

    raw_tokens = estimate_tokens_rough(golden.raw_context or "")
    compressed_tokens = estimate_tokens_rough(golden.compacted_context or "")
    passed = critical_fact_recall == 1.0 and ref_recoverability == 1.0

    return ContextRetentionResult(
        name=golden.name,
        raw_tokens=raw_tokens,
        compressed_tokens=compressed_tokens,
        critical_fact_recall=critical_fact_recall,
        ref_recoverability=ref_recoverability,
        missing_fact_ids=missing,
        unresolved_ref_ids=unresolved_ref_ids,
        passed=passed,
    )
