"""Deterministic intent and specialist routing for Beta."""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from agent.beta.specialists import Specialist, SpecialistRegistry, default_specialist_registry


class Intent(StrEnum):
    CONVERSATION = "conversation"
    INFORMATION = "information"
    DIAGNOSIS = "diagnosis"
    PLANNING = "planning"
    TECHNICAL_EXECUTION = "technical_execution"
    PRODUCTION_CHANGE = "production_change"
    AUDIT = "audit"
    MEMORY = "memory"


class RoutingDecision(BaseModel):
    """Structured result consumed by the Beta orchestrator."""

    model_config = ConfigDict(frozen=True)

    intent: Intent
    specialists: tuple[str, ...] = ()
    rationale: str
    initial_risk: str
    delegation_needed: bool
    parallelizable: bool
    confidence: float = Field(ge=0, le=1)


_INTENT_PATTERNS = (
    (Intent.CONVERSATION, r"^(oi|ola|bom dia|boa tarde|boa noite|hi|hello|hey)\b"),
    (Intent.MEMORY, r"\b(lembre|memorize|recorde|remember|memorize|recall)\b"),
    (
        Intent.PRODUCTION_CHANGE,
        r"\b(producao|production|deploy|reinici|restart|firewall|permiss|exclu|delete|drop)\w*\b",
    ),
    (Intent.DIAGNOSIS, r"\b(por que|porque|why|diagnos|investig|lento|slow|erro|error|falha|issue)\w*\b"),
    (Intent.PLANNING, r"\b(planej|plano|plan|strategy|estrateg)\w*\b"),
    (Intent.AUDIT, r"\b(audit|revise|review|avalie|evaluate|contrato|contract)\w*\b"),
    (Intent.TECHNICAL_EXECUTION, r"\b(implemente|implement|corrija|fix|execute|run|configure|crie|create)\w*\b"),
)

_CONCEPT_ALIASES = {
    "contract": {"contrato", "contract"},
    "database": {"banco", "database", "db", "postgres", "postgresql", "mysql", "sql"},
    "postgresql": {"postgres", "postgresql"},
    "performance": {"lento", "lenta", "slow", "performance", "desempenho", "latencia", "latency"},
    "monitoring": {"monitor", "monitoring", "metrica", "metric", "latencia", "latency", "alerta", "alert", "lento", "lenta", "slow"},
    "infrastructure": {"infra", "host", "servidor", "server", "cpu", "disco", "disk", "latencia", "latency", "lento", "lenta", "slow"},
    "security": {"seguranca", "security", "vulnerabilidade", "vulnerability", "firewall", "ataque", "attack"},
    "deployment": {"deploy", "deployment", "pipeline", "ci", "cd", "container", "kubernetes"},
    "memory": {"memoria", "memory", "lembre", "remember", "decisao", "decision", "preferencia", "preference"},
    "audit": {"audit", "auditoria", "revise", "review", "valide", "validate", "evidencia", "evidence"},
}

_RISK_BY_INTENT = {
    Intent.PRODUCTION_CHANGE: "high",
    Intent.TECHNICAL_EXECUTION: "medium",
}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", _normalize(text)))
    concepts = {
        concept
        for concept, aliases in _CONCEPT_ALIASES.items()
        if words.intersection(aliases)
    }
    return words | concepts


def _classify(text: str) -> Intent:
    normalized = _normalize(text).strip()
    for intent, pattern in _INTENT_PATTERNS:
        if re.search(pattern, normalized):
            return intent
    return Intent.INFORMATION


def _specialist_terms(specialist: Specialist) -> set[str]:
    return _terms(" ".join((*specialist.capabilities, *specialist.keywords)))


def route_request(
    request: str,
    registry: SpecialistRegistry | None = None,
) -> RoutingDecision:
    """Classify a request and select specialists from manifest capabilities."""
    registry = registry or default_specialist_registry()
    intent = _classify(request)
    request_terms = _terms(request)
    ranked = sorted(
        (
            (len(request_terms.intersection(_specialist_terms(specialist))), specialist)
            for specialist in registry.enabled()
        ),
        key=lambda item: (-item[0], item[1].id),
    )
    selected = tuple(specialist.id for score, specialist in ranked if score >= 2)

    delegation_needed = bool(selected) and intent != Intent.CONVERSATION
    matched = ", ".join(selected) if selected else "no matching specialist"
    top_score = ranked[0][0] if ranked else 0
    confidence = 0.98 if intent == Intent.CONVERSATION else min(0.95, 0.6 + top_score * 0.08)
    return RoutingDecision(
        intent=intent,
        specialists=selected,
        rationale=f"Manifest capability matches: {matched}.",
        initial_risk=_RISK_BY_INTENT.get(intent, "low"),
        delegation_needed=delegation_needed,
        parallelizable=len(selected) > 1,
        confidence=confidence,
    )
