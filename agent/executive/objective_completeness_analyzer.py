"""Objective Completeness Analyzer canary.

This module is intentionally narrow and pure: it analyzes a human objective
and reports whether enough information exists to plan safely. It does not
build strategies, contracts, tasks, goals, schedules, or worker runs.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .classifier import classify_objective
from .normalizer import tokenize


_REQUIRED_OUTPUT_KEYS = (
    "objective_fingerprint",
    "normalized_objective",
    "objective_type",
    "confidence",
    "ambiguity_score",
    "known_information",
    "missing_information",
    "contradictions",
    "recommended_questions",
    "ready_for_strategy",
)

_READONLY_TERMS = {
    "readonly", "read-only", "solo lectura", "sólo lectura", "report-only",
    "analysis only", "análisis únicamente",
}

_MUTATION_TERMS = {
    "implement": "requested implementation",
    "implementar": "requested implementation",
    "create task": "requested task creation",
    "crear kanban": "requested task-board creation",
    "kanban": "requested task-board operation",
    "worker": "requested worker operation",
    "workers": "requested worker operation",
    "push": "requested git publish",
    "merge": "requested merge",
    "deploy": "requested deployment",
    "run": "requested execution",
    "ejecutar": "requested execution",
}

_FORBIDDEN_PATTERNS = {
    "no strategy builder": "no_strategy_builder",
    "no genera estrategia": "no_strategy_builder",
    "no execution contract": "no_execution_contract",
    "no crea contrato": "no_execution_contract",
    "no kanban": "no_kanban",
    "no goal": "no_goal",
    "no crea goal": "no_goal",
    "no worker": "no_workers",
    "no workers": "no_workers",
    "no notebook": "no_remote_notebook",
    "no red": "no_network",
    "no network": "no_network",
    "no push": "no_push",
    "no merge": "no_merge",
    "no pr": "no_pr",
    "no knowledge store write": "no_knowledge_store_write",
    "no note store write": "no_note_store_write",
}

_TARGET_HINTS = (
    "executive runtime", "objective completeness analyzer",
    "repo", "path", "/home/", "agent/executive", "analysis.json",
)

_SUCCESS_HINTS = (
    "pass", "validación", "validaciones", "validation", "tests", "compile",
    "hashes", "rollback", "schema", "entregables", "deliverables",
)

_SCOPE_HINTS = (
    "no ", "only", "únicamente", "unicamente", "solo", "sólo", "scope",
    "restricciones", "restrictions", "forbidden",
)


@dataclass(frozen=True)
class MissingInformation:
    kind: str
    severity: str
    recoverable: bool
    suggested_question: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "recoverable": self.recoverable,
            "suggested_question": self.suggested_question,
        }


@dataclass(frozen=True)
class ObjectiveCompletenessAnalysis:
    objective_fingerprint: str
    normalized_objective: str
    objective_type: str
    confidence: float
    ambiguity_score: float
    known_information: dict[str, Any]
    missing_information: tuple[MissingInformation, ...]
    contradictions: tuple[str, ...]
    recommended_questions: tuple[str, ...]
    ready_for_strategy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_fingerprint": self.objective_fingerprint,
            "normalized_objective": self.normalized_objective,
            "objective_type": self.objective_type,
            "confidence": round(float(self.confidence), 4),
            "ambiguity_score": round(float(self.ambiguity_score), 4),
            "known_information": self.known_information,
            "missing_information": [m.to_dict() for m in self.missing_information],
            "contradictions": list(self.contradictions),
            "recommended_questions": list(self.recommended_questions),
            "ready_for_strategy": bool(self.ready_for_strategy),
        }


def _normalize_text(objective_text: str) -> str:
    text = " ".join(objective_text.strip().split())
    return text[:10_000]


def _fingerprint(normalized: str, user_id: str, constraints: tuple[str, ...]) -> str:
    payload = json.dumps(
        {
            "normalized_objective": normalized.lower(),
            "user_id": user_id,
            "constraints": sorted(constraints),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_deliverables(text: str) -> list[str]:
    names = re.findall(r"\b[a-zA-Z0-9_./-]+\.(?:py|json|md|txt)\b", text)
    seen: dict[str, None] = {}
    for name in names:
        seen[name.rstrip(".,;:")] = None
    if "analysis.json" in text.lower():
        seen.setdefault("analysis.json", None)
    return list(seen)


def _extract_forbidden_actions(lower_text: str) -> list[str]:
    found: dict[str, None] = {}
    for pattern, action in _FORBIDDEN_PATTERNS.items():
        if pattern in lower_text:
            found[action] = None
    return list(found)


def _has_any(lower_text: str, hints: tuple[str, ...] | set[str]) -> bool:
    return any(h in lower_text for h in hints)


def _detect_contradictions(lower_text: str) -> tuple[str, ...]:
    has_readonly = _has_any(lower_text, _READONLY_TERMS)
    if not has_readonly:
        return ()
    conflicts: list[str] = []
    for term, meaning in _MUTATION_TERMS.items():
        if term in lower_text:
            # Mentioning a forbidden action with an immediately preceding "no" is a constraint,
            # not a contradiction.
            if f"no {term}" in lower_text:
                continue
            conflicts.append(f"readonly scope conflicts with {meaning}: {term}")
    return tuple(dict.fromkeys(conflicts))


def _build_missing(
    *,
    has_target: bool,
    has_success: bool,
    has_scope: bool,
    has_verification: bool,
    contradictions: tuple[str, ...],
) -> tuple[MissingInformation, ...]:
    missing: list[MissingInformation] = []
    if not has_target:
        missing.append(MissingInformation(
            kind="missing_target",
            severity="critical",
            recoverable=False,
            suggested_question="¿Cuál es el sistema, repositorio, ruta o fuente objetivo?",
        ))
    if not has_success:
        missing.append(MissingInformation(
            kind="missing_success_criteria",
            severity="critical",
            recoverable=False,
            suggested_question="¿Qué evidencia observable define que el objetivo quedó completo?",
        ))
    if not has_scope:
        missing.append(MissingInformation(
            kind="missing_scope_boundary",
            severity="high",
            recoverable=False,
            suggested_question="¿Qué acciones están explícitamente dentro y fuera de alcance?",
        ))
    if not has_verification:
        missing.append(MissingInformation(
            kind="missing_verification_path",
            severity="medium",
            recoverable=True,
            suggested_question="¿Qué validaciones deben ejecutarse antes de considerar listo el análisis?",
        ))
    if contradictions:
        missing.append(MissingInformation(
            kind="conflicting_scope",
            severity="critical",
            recoverable=False,
            suggested_question="¿Debe prevalecer el modo readonly o la acción mutante solicitada?",
        ))
    return tuple(missing)


def analyze_objective(
    objective_text: str,
    *,
    user_id: str = "executive-runtime-canary",
    explicit_constraints: list[str] | tuple[str, ...] | None = None,
) -> ObjectiveCompletenessAnalysis:
    """Analyze objective completeness and return a serializable result.

    The function is deterministic for equal text, user_id, and explicit constraints.
    It performs heuristic intake only and intentionally avoids runtime discovery.
    """
    if not objective_text or not objective_text.strip():
        raise ValueError("objective_text must be non-empty")
    if not user_id:
        raise ValueError("user_id must be non-empty")

    normalized = _normalize_text(objective_text)
    lower_text = normalized.lower()
    tokens = tokenize(normalized)
    classified = classify_objective(tokens)
    objective_type = classified.goal_class.value
    if any(t in lower_text for t in ("implementar", "implementation", "implementación")):
        objective_type = "BUILD"

    deliverables = _extract_deliverables(normalized)
    forbidden_actions = _extract_forbidden_actions(lower_text)
    constraints = tuple(dict.fromkeys(tuple(explicit_constraints or ()) + tuple(forbidden_actions)))
    contradictions = _detect_contradictions(lower_text)

    has_target = _has_any(lower_text, _TARGET_HINTS) or bool(re.search(r"\b\w+\.py\b", lower_text))
    has_success = _has_any(lower_text, _SUCCESS_HINTS) or bool(deliverables)
    has_scope = _has_any(lower_text, _SCOPE_HINTS) or bool(forbidden_actions)
    has_verification = any(h in lower_text for h in ("compile", "test", "tests", "schema", "hash", "rollback", "valid"))

    missing = _build_missing(
        has_target=has_target,
        has_success=has_success,
        has_scope=has_scope,
        has_verification=has_verification,
        contradictions=contradictions,
    )

    critical_missing = any(m.severity == "critical" for m in missing)
    ambiguity_score = min(1.0, 0.12 + 0.18 * len(missing) + 0.22 * len(contradictions))
    if not missing:
        ambiguity_score = 0.08
    confidence = max(0.0, min(1.0, 0.92 - ambiguity_score * 0.55))
    ready_for_strategy = not critical_missing and not contradictions and ambiguity_score < 0.5

    known_information = {
        "intent": classified.rationale,
        "target_detected": has_target,
        "success_criteria_detected": has_success,
        "scope_boundary_detected": has_scope,
        "verification_path_detected": has_verification,
        "deliverables": deliverables,
        "forbidden_actions": forbidden_actions,
        "tokens": tokens[:50],
    }

    questions = tuple(m.suggested_question for m in missing)

    return ObjectiveCompletenessAnalysis(
        objective_fingerprint=_fingerprint(normalized, user_id, constraints),
        normalized_objective=normalized,
        objective_type=objective_type,
        confidence=confidence,
        ambiguity_score=ambiguity_score,
        known_information=known_information,
        missing_information=missing,
        contradictions=contradictions,
        recommended_questions=questions,
        ready_for_strategy=ready_for_strategy,
    )


def write_analysis_json(
    analysis: ObjectiveCompletenessAnalysis,
    output_path: str | Path,
) -> Path:
    """Write exactly one analysis JSON file to the caller-provided path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "MissingInformation",
    "ObjectiveCompletenessAnalysis",
    "analyze_objective",
    "write_analysis_json",
]
