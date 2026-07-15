"""Deterministic advisory risk assessment for newly-created Kanban tasks.

The classifier deliberately recognizes independent delivery dimensions rather
than assigning points for body length or generic keyword frequency.  It is
pure: callers can recompute the same versioned assessment for presentation
without reading configuration or board state.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal, Mapping


RiskLevel = Literal["low", "medium", "high"]
Recommendation = Literal[
    "dispatch", "consider_decomposition", "decompose_before_dispatch"
]

_DIMENSION_ORDER = (
    "unknown_design",
    "subsystem_breadth",
    "data_ops_risk",
    "external_gate",
    "delivery_lifecycle",
    "acceptance_density",
)

_DESIGN_RE = re.compile(
    r"\b(?:investigat(?:e|ion)|diagnos(?:e|is)|trace|decid(?:e|ing)|design|define|"
    r"specify|determine|discover|research)\b",
    re.IGNORECASE,
)
_IMPLEMENT_RE = re.compile(
    r"\b(?:implement|build|add|fix|repair|change|update|wire|create|harden|ship)\b",
    re.IGNORECASE,
)
_DATA_OPS_RE = re.compile(
    r"\b(?:migrat(?:e|ion)|repair|backfill|live\s+(?:data|database|mutation)|"
    r"production\s+(?:data|database|mutation)|backup|restore|rollout|rollback|"
    r"deploy(?:ment|ing)?)\b",
    re.IGNORECASE,
)
_EXTERNAL_GATE_RE = re.compile(
    r"\b(?:credentials?|third[- ]party\s+auth|external\s+auth|app\s*store|"
    r"testflight|device\s+(?:test|validation)|screenshots?|manual(?:ly)?\s+validat(?:e|ion)|"
    r"external\s+(?:system|deployment)|unavailable\s+system)\b",
    re.IGNORECASE,
)
_EXPLICIT_LONG_RE = re.compile(
    r"\b(?:exceed|longer\s+than|more\s+than|over)\s+15\s*(?:minutes?|mins?|m)\b",
    re.IGNORECASE,
)
_CHECKABLE_ITEM_RE = re.compile(r"(?m)^\s*(?:[-*+]\s+(?:\[[ xX]\]\s*)?|\d+[.)]\s+)\S")
_SECTION_RE = re.compile(r"^\s*(?:#{1,6}\s*)?([A-Za-z][A-Za-z /_-]{1,50}):?\s*$")
_SOURCE_ROOT_RE = re.compile(
    r"(?<![\w.-])([A-Za-z][A-Za-z0-9_.-]*)/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]*"
)

_SUBSYSTEM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("api_server", re.compile(r"\b(?:api|server|endpoint|sse|websocket)\b", re.I)),
    (
        "persistence",
        re.compile(
            r"\b(?:database|sqlite|sql|storage|journal|persistence|schema)\b", re.I
        ),
    ),
    (
        "cli_tool",
        re.compile(r"\b(?:cli|command(?:-line)?|model\s+tool|tooling)\b", re.I),
    ),
    (
        "dashboard_client",
        re.compile(r"\b(?:dashboard|frontend|client|react|swiftui|ui)\b", re.I),
    ),
    (
        "gateway_worker",
        re.compile(r"\b(?:gateway|dispatcher|worker|queue|scheduler)\b", re.I),
    ),
    (
        "generated_contract",
        re.compile(r"\b(?:openapi|generated\s+contract|codegen)\b", re.I),
    ),
    (
        "runtime_ops",
        re.compile(
            r"\b(?:git\s+worktree|worktrees?|deployment|runtime\s+operations?)\b", re.I
        ),
    ),
)

_LIFECYCLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pr", re.compile(r"\b(?:pull request|pr)\b", re.I)),
    ("ci", re.compile(r"\bci\b|continuous integration", re.I)),
    ("merge", re.compile(r"\bmerg(?:e|ed|ing)\b", re.I)),
    ("deploy", re.compile(r"\bdeploy(?:ment|ed|ing)?\b", re.I)),
    (
        "runtime_validation",
        re.compile(
            r"\b(?:post[- ]merge|runtime|production|live)\s+validat(?:e|ion)\b", re.I
        ),
    ),
)


@dataclass(frozen=True)
class TaskRiskAssessment:
    """A bounded, serializable advisory assessment."""

    version: int
    level: RiskLevel
    dimensions: tuple[str, ...]
    evidence: tuple[str, ...]
    recommendation: Recommendation
    checkpoint_required: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "level": self.level,
            "dimensions": list(self.dimensions),
            "evidence": list(self.evidence),
            "recommendation": self.recommendation,
            "checkpoint_required": self.checkpoint_required,
        }

    def event_payload(self) -> dict[str, Any]:
        """Return the deliberately smaller, redacted audit-event payload."""

        return {
            "version": self.version,
            "level": self.level,
            "dimensions": list(self.dimensions),
            "recommendation": self.recommendation,
            "checkpoint_required": self.checkpoint_required,
        }

    def message(self) -> str:
        dimensions = ", ".join(self.dimensions)
        return (
            "Task was created. Non-blocking preflight advice: "
            f"{self.recommendation.replace('_', ' ')} "
            f"(risk dimensions: {dimensions})."
        )


def _source_roots(text: str) -> tuple[str, ...]:
    roots = {match.group(1).casefold() for match in _SOURCE_ROOT_RE.finditer(text)}
    return tuple(sorted(roots))


def _checkable_item_count(body: str) -> int:
    """Count only checklist-like items in acceptance/required sections."""
    active = False
    count = 0
    for line in body.splitlines():
        section = _SECTION_RE.match(line)
        if section:
            name = section.group(1).strip().lower()
            active = name.startswith("acceptance") or name.startswith("required")
            continue
        if active and _CHECKABLE_ITEM_RE.match(line):
            count += 1
    return count


def _contract_item_count(task_contract: Mapping[str, Any] | None) -> int:
    if not task_contract:
        return 0
    count = 0
    for key in ("acceptance_criteria", "required_items", "verification"):
        value = task_contract.get(key)
        if isinstance(value, (list, tuple)):
            count += len(value)
    return count


def assess_task_risk(
    *,
    title: str,
    body: str | None,
    goal_mode: bool,
    max_runtime_seconds: int | None,
    task_contract: Mapping[str, Any] | None = None,
) -> TaskRiskAssessment:
    """Classify independent task dimensions with conservative thresholds.

    ``goal_mode`` is accepted as part of the stable public API, but cannot add
    or remove risk.  Body length likewise has no scoring effect.
    """

    del goal_mode
    text = f"{title}\n{body or ''}"
    roots = _source_roots(text)
    subsystem_families = tuple(
        name for name, pattern in _SUBSYSTEM_PATTERNS if pattern.search(text)
    )
    lifecycle_stages = tuple(
        name for name, pattern in _LIFECYCLE_PATTERNS if pattern.search(text)
    )
    checkable_items = _checkable_item_count(body or "") + _contract_item_count(
        task_contract
    )

    present: set[str] = set()
    evidence: list[str] = []

    if _DESIGN_RE.search(text) and _IMPLEMENT_RE.search(text):
        present.add("unknown_design")
        evidence.append("unknown_design:design_and_implementation")
    if len(subsystem_families) >= 3 or len(roots) >= 3:
        present.add("subsystem_breadth")
        evidence.append(
            f"subsystem_breadth:families={len(subsystem_families)},roots={len(roots)}"
        )
    if _DATA_OPS_RE.search(text):
        present.add("data_ops_risk")
        evidence.append("data_ops_risk:operation_required")
    if _EXTERNAL_GATE_RE.search(text):
        present.add("external_gate")
        evidence.append("external_gate:external_or_manual_gate")
    if _IMPLEMENT_RE.search(text) and len(lifecycle_stages) >= 3:
        present.add("delivery_lifecycle")
        evidence.append(f"delivery_lifecycle:stages={len(lifecycle_stages)}")
    if checkable_items >= 7 or (
        checkable_items >= 5 and (len(roots) >= 3 or len(subsystem_families) >= 3)
    ):
        present.add("acceptance_density")
        evidence.append(f"acceptance_density:items={checkable_items}")

    dimensions = tuple(name for name in _DIMENSION_ORDER if name in present)
    dimension_count = len(dimensions)
    if dimension_count >= 3:
        level: RiskLevel = "high"
        recommendation: Recommendation = "decompose_before_dispatch"
    elif dimension_count == 2:
        level = "medium"
        recommendation = "consider_decomposition"
    else:
        level = "low"
        recommendation = "dispatch"

    long_run_hint = (
        max_runtime_seconds is not None and int(max_runtime_seconds) >= 1800
    ) or bool(_EXPLICIT_LONG_RE.search(text))
    if long_run_hint:
        evidence.append("long_run_hint:checkpoint_required")

    return TaskRiskAssessment(
        version=1,
        level=level,
        dimensions=dimensions,
        evidence=tuple(evidence),
        recommendation=recommendation,
        checkpoint_required=level != "low" or long_run_hint,
    )
