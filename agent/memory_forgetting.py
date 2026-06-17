"""Non-destructive forgetting and decay audit for semantic memory records.

This module is deliberately deterministic and side-effect free. It does not
remove, rewrite, suppress, or promote memory entries; it reports review findings
from typed sidecar metadata so callers can make memory cleanup explicit and
reversible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import time
from typing import Any, Literal, Mapping, Sequence, Tuple

ForgettingAction = Literal[
    "keep",
    "review",
    "compress",
    "replace",
    "remove",
    "promote_to_skill",
]
ForgettingSeverity = Literal["info", "low", "medium", "high"]
ForgettingTarget = Literal["memory", "user", "all"]

STALE_AFTER_SECONDS = 180 * 24 * 60 * 60
OVERLONG_TEXT_CHARS = 280

_STALE_MARKER_RE = re.compile(
    r"\b(used to|formerly|no longer|old|previously)\b",
    re.IGNORECASE,
)

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2, "info": 3}
_ACTION_RANK = {
    "promote_to_skill": 0,
    "remove": 1,
    "replace": 2,
    "compress": 3,
    "review": 4,
    "keep": 5,
}


@dataclass(frozen=True)
class ForgettingFinding:
    record_id: str
    target: Literal["memory", "user"]
    text: str
    suggested_action: ForgettingAction
    reason: str
    severity: ForgettingSeverity
    confidence: float
    salience: float
    signals: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["signals"] = list(self.signals)
        return data


@dataclass(frozen=True)
class ForgettingAuditReport:
    target: ForgettingTarget
    generated_at: int
    findings: Tuple[ForgettingFinding, ...]
    summary: Mapping[str, int]

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "generated_at": self.generated_at,
            "findings": [finding.to_dict() for finding in self.findings],
            "summary": dict(self.summary),
        }


def _clamp(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _record_text(record: Mapping[str, Any]) -> str:
    return " ".join(str(record.get("text") or "").strip().split())


def _record_id(record: Mapping[str, Any], target: str, index: int) -> str:
    value = record.get("id")
    if isinstance(value, str) and value:
        return value
    return f"{target}:{index}"


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _is_old(record: Mapping[str, Any], now: int) -> bool:
    try:
        updated_at = int(record.get("updated_at") or record.get("created_at") or 0)
    except (TypeError, ValueError):
        updated_at = 0
    return bool(updated_at and now - updated_at >= STALE_AFTER_SECONDS)


def _make_finding(
    record: Mapping[str, Any],
    *,
    target: Literal["memory", "user"],
    index: int,
    suggested_action: ForgettingAction,
    reason: str,
    severity: ForgettingSeverity,
    signals: Sequence[str],
) -> ForgettingFinding:
    return ForgettingFinding(
        record_id=_record_id(record, target, index),
        target=target,
        text=_record_text(record),
        suggested_action=suggested_action,
        reason=reason,
        severity=severity,
        confidence=_clamp(record.get("confidence"), 0.7),
        salience=_clamp(record.get("salience"), 0.5),
        signals=tuple(signals),
    )


def _finding_sort_key(finding: ForgettingFinding) -> tuple:
    return (
        _SEVERITY_RANK[finding.severity],
        _ACTION_RANK[finding.suggested_action],
        -finding.salience,
        -finding.confidence,
        finding.record_id,
    )


def _summarize(findings: Sequence[ForgettingFinding], total_records: int) -> dict[str, int]:
    summary: dict[str, int] = {
        "total_records": total_records,
        "total_findings": len(findings),
    }
    for finding in findings:
        summary[f"severity:{finding.severity}"] = summary.get(f"severity:{finding.severity}", 0) + 1
        summary[f"action:{finding.suggested_action}"] = summary.get(
            f"action:{finding.suggested_action}", 0
        ) + 1
        for signal in finding.signals:
            summary[f"signal:{signal}"] = summary.get(f"signal:{signal}", 0) + 1
    return summary


def audit_memory_records(
    records: Sequence[Mapping[str, Any]],
    *,
    target: Literal["memory", "user"],
    now: int | None = None,
) -> ForgettingAuditReport:
    """Audit typed semantic records and return non-destructive findings.

    The report suggests actions only. It never mutates records and never decides
    that a memory should be hidden from prompt rendering.
    """
    if target not in {"memory", "user"}:
        raise ValueError("target must be 'memory' or 'user'")

    generated_at = int(time.time()) if now is None else int(now)
    findings: list[ForgettingFinding] = []
    duplicate_buckets: dict[str, list[int]] = {}

    for index, record in enumerate(records):
        text = _record_text(record)
        if not text:
            continue
        duplicate_buckets.setdefault(_normalized_text(text), []).append(index)

        kind = str(record.get("kind") or "")
        consolidation_action = str(record.get("consolidation_action") or "")
        salience = _clamp(record.get("salience"), 0.5)
        confidence = _clamp(record.get("confidence"), 0.7)
        old = _is_old(record, generated_at)

        if kind == "procedural_candidate" or consolidation_action == "procedural_skill_candidate":
            findings.append(
                _make_finding(
                    record,
                    target=target,
                    index=index,
                    suggested_action="promote_to_skill",
                    reason="Reusable workflow belongs in procedural memory, not semantic memory.",
                    severity="medium" if salience >= 0.6 else "low",
                    signals=("procedural_candidate",),
                )
            )
            continue

        if kind == "episodic_note" or consolidation_action in {"episodic_only", "working_memory_only"}:
            remove_candidate = confidence >= 0.7 and salience <= 0.3
            findings.append(
                _make_finding(
                    record,
                    target=target,
                    index=index,
                    suggested_action="remove" if remove_candidate else "review",
                    reason="Task progress belongs in session history, not durable semantic memory.",
                    severity="medium" if remove_candidate else "low",
                    signals=("episodic_or_task_local",),
                )
            )
            continue

        if confidence < 0.5:
            findings.append(
                _make_finding(
                    record,
                    target=target,
                    index=index,
                    suggested_action="review",
                    reason="Fact is weakly supported and should be reviewed before continued injection.",
                    severity="medium" if old and salience < 0.5 else "low",
                    signals=("low_confidence",),
                )
            )

        if salience < 0.25:
            findings.append(
                _make_finding(
                    record,
                    target=target,
                    index=index,
                    suggested_action="remove" if old else "review",
                    reason="Low-salience memory may not justify scarce prompt-injected memory budget.",
                    severity="medium" if old else "low",
                    signals=("low_salience",),
                )
            )

        if old and salience < 0.5:
            findings.append(
                _make_finding(
                    record,
                    target=target,
                    index=index,
                    suggested_action="review",
                    reason="Old low-salience memory should be checked before continuing to inject it.",
                    severity="low",
                    signals=("old_low_salience",),
                )
            )

        if len(text) > OVERLONG_TEXT_CHARS:
            findings.append(
                _make_finding(
                    record,
                    target=target,
                    index=index,
                    suggested_action="compress",
                    reason="Semantic memory should stay compact and declarative.",
                    severity="low",
                    signals=("overlong",),
                )
            )

        if _STALE_MARKER_RE.search(text):
            findings.append(
                _make_finding(
                    record,
                    target=target,
                    index=index,
                    suggested_action="review",
                    reason="Entry contains stale/conflict marker language that needs human or tool verification.",
                    severity="low",
                    signals=("stale_marker",),
                )
            )

    for indexes in duplicate_buckets.values():
        if len(indexes) <= 1:
            continue
        for index in indexes[1:]:
            findings.append(
                _make_finding(
                    records[index],
                    target=target,
                    index=index,
                    suggested_action="replace",
                    reason="Exact duplicate memory wastes prompt budget and should be merged or removed.",
                    severity="medium",
                    signals=("exact_duplicate",),
                )
            )

    ordered = tuple(sorted(findings, key=_finding_sort_key))
    return ForgettingAuditReport(
        target=target,
        generated_at=generated_at,
        findings=ordered,
        summary=_summarize(ordered, len(records)),
    )


def audit_memory_record_sets(
    record_sets: Mapping[Literal["memory", "user"], Sequence[Mapping[str, Any]]],
    *,
    now: int | None = None,
) -> ForgettingAuditReport:
    """Audit memory and user record sets together without mutating either."""
    generated_at = int(time.time()) if now is None else int(now)
    all_findings: list[ForgettingFinding] = []
    total_records = 0
    for target in ("memory", "user"):
        records = record_sets.get(target, ())
        total_records += len(records)
        all_findings.extend(
            audit_memory_records(records, target=target, now=generated_at).findings
        )
    ordered = tuple(sorted(all_findings, key=_finding_sort_key))
    return ForgettingAuditReport(
        target="all",
        generated_at=generated_at,
        findings=ordered,
        summary=_summarize(ordered, total_records),
    )


def finding_to_dict(finding: ForgettingFinding) -> dict:
    return finding.to_dict()


def report_to_dict(report: ForgettingAuditReport) -> dict:
    return report.to_dict()


__all__ = [
    "ForgettingAction",
    "ForgettingAuditReport",
    "ForgettingFinding",
    "ForgettingSeverity",
    "ForgettingTarget",
    "OVERLONG_TEXT_CHARS",
    "STALE_AFTER_SECONDS",
    "audit_memory_record_sets",
    "audit_memory_records",
    "finding_to_dict",
    "report_to_dict",
]
