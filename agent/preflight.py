"""Advisory preflight risk classification for Hermes requests.

This module is intentionally pure and source-only. It classifies a natural
language request before any tool use so future orchestration layers can choose a
more cautious workflow. It is not a security boundary: it never blocks, mutates
state, reads config, talks to the gateway, or executes commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping
import re


class PreflightRiskLevel(str, Enum):
    """Advisory risk level for a user request."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PreflightRecommendation(str, Enum):
    """Suggested orchestration posture for the request."""

    OBSERVE = "observe"
    TDD = "tdd"
    CLAUDE_REVIEW = "claude_review"
    EXPLICIT_APPROVAL = "explicit_approval"


@dataclass(frozen=True)
class PreflightSignal:
    """One advisory risk signal found in a request."""

    name: str
    weight: int


@dataclass(frozen=True)
class PreflightReport:
    """Deterministic advisory preflight classification result."""

    level: PreflightRiskLevel
    score: int
    signals: tuple[PreflightSignal, ...]
    recommendations: tuple[PreflightRecommendation, ...]
    rationale: str


@dataclass(frozen=True)
class _Rule:
    name: str
    weight: int
    pattern: re.Pattern[str]


_RULES = (
    _Rule(
        "destructive_command",
        3,
        re.compile(
            r"\brm\s+-rf\b|\bdrop\s+(?:table|database)\b|\bdelete\s+from\b|"
            r"\btruncate\b|\breset\s+--hard\b|\bforce\s+push\b",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "sensitive_config_or_secret",
        3,
        re.compile(
            r"(?:^|[/\s])\.env(?:\b|\.)|\bcredentials(?:\.json)?\b|\bsecrets?\b|"
            r"\bid_rsa\b|\bapi[_-]?key\b|\btoken\b|\bsecret[_-]?token\b",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "runtime_process_control",
        3,
        re.compile(
            r"\b(?:restart|reload)\b[^\n]*(?:gateway|hermes|slack)|"
            r"\b(?:kill(?:all)?|pkill)\b[^\n]*(?:hermes|gateway|python)|"
            r"\b(?:systemctl|launchctl)\b",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "network_bootstrap",
        2,
        re.compile(r"\b(?:curl|wget)\b[^\n|]*(?:\|\s*(?:sh|bash))", re.IGNORECASE),
    ),
    _Rule(
        "schema_change",
        2,
        re.compile(r"\b(?:alter\s+table|migrate|migration|schema\s+change)\b", re.IGNORECASE),
    ),
    _Rule(
        "credential_rotation",
        3,
        re.compile(r"\b(?:rotate|rotation|regenerate|reissue)\b[^\n]*(?:credential|secret|token|api[_-]?key|signing\s+secret)", re.IGNORECASE),
    ),
    _Rule(
        "deployment_or_production",
        2,
        re.compile(r"\b(?:deploy|release|rollout|promote)\b[^\n]*(?:prod|production|live)|\bproduction\s+(?:deploy|release|rollout)\b", re.IGNORECASE),
    ),
    _Rule(
        "dependency_change",
        2,
        re.compile(r"\b(?:upgrade|update|bump|install)\b[^\n]*(?:dependenc|package|pip|npm|pnpm|yarn|brew|requirements|pyproject)", re.IGNORECASE),
    ),
    _Rule(
        "external_side_effect",
        2,
        re.compile(r"\b(?:slack\s+send|email|post\s+to|publish)\b", re.IGNORECASE),
    ),
    _Rule(
        "mass_scope",
        3,
        re.compile(r"\b(?:all\s+users|every\s+(?:file|python\s+file)|all\s+files|entire\s+repository|whole\s+repo|--all)\b|\*\*/\*", re.IGNORECASE),
    ),
    _Rule(
        "hot_path_edit",
        1,
        re.compile(r"\b(?:run_agent\.py|cli\.py|gateway/)\b", re.IGNORECASE),
    ),
)
_READ_ONLY_RE = re.compile(r"\b(?:read|show|explain|find|list|search|grep)\b", re.IGNORECASE)
_EDIT_INTENT_RE = re.compile(r"\b(?:add|edit|modify|refactor|rename|implement|fix|change)\b", re.IGNORECASE)


def classify_request(
    request: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> PreflightReport:
    """Classify a request into an advisory risk report.

    ``context`` is accepted for forward compatibility with future runtime hooks,
    but this v1 classifier is deterministic and does not inspect external state.
    """

    del context
    text = request.casefold() if isinstance(request, str) else ""
    signals = _collect_signals(text)
    score = sum(signal.weight for signal in signals)
    level = _classify_level(score, signals)
    recommendations = _recommendations_for(level, signals)
    rationale = _build_rationale(level, signals)
    return PreflightReport(
        level=level,
        score=score,
        signals=signals,
        recommendations=recommendations,
        rationale=rationale,
    )


def format_preflight_summary(report: PreflightReport) -> str:
    """Return a compact redacted summary that never echoes the request text."""

    signal_names = ",".join(signal.name for signal in report.signals) or "none"
    recommendations = ",".join(recommendation.value for recommendation in report.recommendations)
    return (
        "preflight risk summary: "
        f"level={report.level.value} "
        f"score={report.score} "
        f"signals={signal_names} "
        f"recommendations={recommendations}"
    )


def _collect_signals(text: str) -> tuple[PreflightSignal, ...]:
    signals: list[PreflightSignal] = []
    seen: set[str] = set()
    for rule in _RULES:
        if rule.name in seen or not rule.pattern.search(text):
            continue
        if rule.name == "hot_path_edit" and not _EDIT_INTENT_RE.search(text):
            continue
        seen.add(rule.name)
        signals.append(PreflightSignal(name=rule.name, weight=rule.weight))

    if not signals and _EDIT_INTENT_RE.search(text) and not _READ_ONLY_RE.search(text):
        signals.append(PreflightSignal(name="code_change_intent", weight=1))
    return tuple(signals)


def _classify_level(
    score: int,
    signals: tuple[PreflightSignal, ...],
) -> PreflightRiskLevel:
    if any(signal.weight >= 3 for signal in signals) or score >= 3:
        return PreflightRiskLevel.HIGH
    if score > 0:
        return PreflightRiskLevel.MEDIUM
    return PreflightRiskLevel.LOW


def _recommendations_for(
    level: PreflightRiskLevel,
    signals: tuple[PreflightSignal, ...],
) -> tuple[PreflightRecommendation, ...]:
    if level == PreflightRiskLevel.LOW:
        return (PreflightRecommendation.OBSERVE,)

    recommendations: list[PreflightRecommendation]
    if level == PreflightRiskLevel.MEDIUM:
        recommendations = [PreflightRecommendation.TDD, PreflightRecommendation.CLAUDE_REVIEW]
    else:
        recommendations = [
            PreflightRecommendation.EXPLICIT_APPROVAL,
            PreflightRecommendation.CLAUDE_REVIEW,
        ]
        if any(signal.name == "schema_change" for signal in signals):
            recommendations.append(PreflightRecommendation.TDD)
    return tuple(recommendations)


def _build_rationale(
    level: PreflightRiskLevel,
    signals: tuple[PreflightSignal, ...],
) -> str:
    if not signals:
        return f"{level.value} risk: no advisory risk signals detected"
    names = ", ".join(signal.name for signal in signals)
    return f"{level.value} risk: detected {names}"
