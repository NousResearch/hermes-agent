"""Evidence-aware consolidation for Beta specialist results."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field

from agent.beta.delegation import SpecialistResult
from agent.beta.risk import RiskLevel, classify_risk
from agent.beta.router import RoutingDecision


class ConsolidatedResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    understanding: str
    agents_activated: tuple[str, ...]
    result: str
    evidence: tuple[str, ...]
    facts: tuple[str, ...]
    hypotheses: tuple[str, ...]
    probable_cause: str | None = None
    confidence: float = Field(ge=0, le=1)
    risk: RiskLevel
    recommendation: tuple[str, ...]
    authorization_required: bool
    next_step: str
    contradictions: tuple[str, ...] = ()
    partial_failures: tuple[str, ...] = ()
    qa_required: bool = False
    qa_performed: bool = False


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return " ".join("".join(char for char in text if not unicodedata.combining(char)).split())


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen = set()
    result = []
    for value in values:
        key = _normalize(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return tuple(result)


_HIGH = frozenset({"high", "alta", "alto", "elevated", "pico", "saturated", "saturado"})
_NORMAL = frozenset({"normal", "healthy", "saudavel", "stable", "estavel", "low", "baixa", "baixo"})
_CLAIM_NOISE = _HIGH | _NORMAL | frozenset({"is", "esta", "estao", "usage", "uso", "the", "a", "o", "de"})


def _polarity(fact: str) -> str | None:
    words = set(re.findall(r"[a-z0-9]+", _normalize(fact)))
    if words.intersection(_HIGH):
        return "high"
    if words.intersection(_NORMAL):
        return "normal"
    return None


def _subject(fact: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalize(fact))) - _CLAIM_NOISE


def _contradictions(results: tuple[SpecialistResult, ...]) -> tuple[str, ...]:
    claims = [
        (result.specialist_id, fact, _polarity(fact), _subject(fact))
        for result in results
        if result.status == "completed"
        for fact in result.facts
    ]
    conflicts = []
    for index, (left_agent, left, left_polarity, left_subject) in enumerate(claims):
        if left_polarity is None:
            continue
        for right_agent, right, right_polarity, right_subject in claims[index + 1 :]:
            if left_agent == right_agent or right_polarity is None or left_polarity == right_polarity:
                continue
            if left_subject.intersection(right_subject):
                conflicts.append(f"{left_agent}: {left} <> {right_agent}: {right}")
    return _unique(conflicts)


def _probable_cause(results: tuple[SpecialistResult, ...], has_evidence: bool) -> str | None:
    if not has_evidence:
        return None
    hypotheses = [
        (hypothesis, result.confidence)
        for result in results
        if result.status == "completed"
        for hypothesis in result.hypotheses
    ]
    if not hypotheses:
        return None
    counts = Counter(_normalize(hypothesis) for hypothesis, _confidence in hypotheses)
    return max(hypotheses, key=lambda item: (counts[_normalize(item[0])], item[1]))[0]


def consolidate_results(
    request: str,
    decision: RoutingDecision,
    specialist_results: Iterable[SpecialistResult],
    *,
    qa_validator: Callable[[tuple[SpecialistResult, ...], tuple[str, ...]], SpecialistResult] | None = None,
) -> ConsolidatedResponse:
    """Consolidate facts and evidence without promoting hypotheses to facts."""
    results = tuple(specialist_results)
    completed = tuple(result for result in results if result.status == "completed")
    contradictions = _contradictions(completed)
    recommendations = _unique(action for result in completed for action in result.recommended_actions)
    risk = RiskLevel(decision.initial_risk)
    if any(classify_risk(action) == RiskLevel.HIGH for action in recommendations):
        risk = RiskLevel.HIGH
    qa_required = bool(contradictions) or risk == RiskLevel.HIGH
    qa_performed = False
    if qa_required and qa_validator is not None:
        qa_result = qa_validator(results, contradictions)
        results += (qa_result,)
        if qa_result.status == "completed":
            completed += (qa_result,)
            qa_performed = True

    evidence = _unique(item for result in completed for item in result.evidence)
    facts = _unique(item for result in completed for item in result.facts)
    hypotheses = _unique(item for result in completed for item in result.hypotheses)
    partial_failures = tuple(
        f"{result.specialist_id}: {'; '.join(result.errors) or result.status}"
        for result in results
        if result.status != "completed"
    )
    base_confidence = (
        sum(result.confidence for result in completed) / len(completed)
        if completed
        else 0.0
    )
    confidence = max(0.0, min(1.0, base_confidence - 0.2 * len(contradictions) - 0.1 * len(partial_failures)))
    if not evidence and not facts:
        confidence = min(confidence, 0.49)
    probable_cause = _probable_cause(completed, bool(evidence or facts))
    summaries = _unique(result.summary for result in completed if result.summary)
    result_text = " ".join(summaries) or "No validated specialist result was available."
    if not evidence and not facts:
        result_text += " Evidence is insufficient for a categorical conclusion."

    authorization_required = risk == RiskLevel.HIGH or any(
        result.authorization_required for result in results
    )
    if qa_required and not qa_performed:
        next_step = "Run QA validation before acting."
    elif authorization_required:
        next_step = "Request explicit approval for the exact high-risk operation."
    elif partial_failures:
        next_step = "Review partial failures and collect missing evidence."
    else:
        next_step = "Review the recommendation with the Chief."

    return ConsolidatedResponse(
        understanding=request,
        agents_activated=_unique(result.specialist_id for result in results),
        result=result_text,
        evidence=evidence,
        facts=facts,
        hypotheses=hypotheses,
        probable_cause=probable_cause,
        confidence=confidence,
        risk=risk,
        recommendation=recommendations,
        authorization_required=authorization_required,
        next_step=next_step,
        contradictions=contradictions,
        partial_failures=partial_failures,
        qa_required=qa_required,
        qa_performed=qa_performed,
    )

