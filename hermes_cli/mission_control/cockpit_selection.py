"""Side-effect-free Mission Control cockpit selection policy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

MAX_DECIDE_NOW = 3

GUARDRAILS = {
    "external_send": False,
    "auto_dispatch": False,
    "cron_created": False,
    "kanban_launch": False,
}

HANDOFF_KINDS = {"handoff", "kanban_started", "output_ready"}
AUDIT_KINDS = {"log", "duplicate", "metric", "audit"}
DONE_STATUSES = {"done", "archived", "discarded", "duplicate"}
URGENT_STATUSES = {"failed_after_retry", "approval_required", "blocked", "review_required"}
RISK_SCORE = {"stop": 40, "critical": 35, "high": 30, "medium": 15, "low": 5, "none": 0}
BENEFIT_SCORE = {"high": 12, "medium": 8, "low": 3, "none": 0}
EFFORT_SCORE = {"small": 6, "medium": 3, "large": 0}
EVIDENCE_SCORE = {"verified": 8, "strong": 6, "medium": 3, "weak": 0, "missing": -10, "none": -10}
KIND_SCORE = {"release_gate": 22, "failure": 30, "regression": 14}


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _priority(item: Mapping[str, Any]) -> int:
    kind = _text(item.get("kind"))
    status = _text(item.get("status"))
    score = 0
    score += RISK_SCORE.get(_text(item.get("risk")), 0)
    score += BENEFIT_SCORE.get(_text(item.get("benefit")), 0)
    score += EFFORT_SCORE.get(_text(item.get("effort")), 0)
    score += EVIDENCE_SCORE.get(_text(item.get("evidence")), 0)
    if status in URGENT_STATUSES:
        score += 20
    score += KIND_SCORE.get(kind, 0)
    if _bool(item.get("decision_required")):
        score += 12
    if _bool(item.get("external_effect")):
        score += 10
    return score


def _normalize(item: Mapping[str, Any], lane: str) -> dict[str, Any]:
    normalized = dict(item)
    normalized["lane"] = lane
    normalized["priority_score"] = _priority(item)
    if lane == "decide_now":
        normalized.setdefault(
            "ui_copy",
            "Preview-only: nessun task Kanban, worker, cron o invio esterno viene creato.",
        )
        normalized.setdefault("max_options", 3)
    return normalized


def select_cockpit_lanes(signals: Iterable[Mapping[str, Any]], *, max_decide_now: int = MAX_DECIDE_NOW) -> dict[str, Any]:
    """Select low-noise Mission Control lanes from raw signals.

    The selector is deliberately deterministic and pure. Handoff-like records
    stay in the handoff lane even when completed; routine logs, duplicates and
    completed non-handoff records stay in deep audit rather than the main view.
    """

    decide_candidates: list[dict[str, Any]] = []
    team_pulse: list[dict[str, Any]] = []
    handoff: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for raw in signals:
        kind = _text(raw.get("kind"))
        status = _text(raw.get("status"))
        if kind in HANDOFF_KINDS:
            handoff.append(_normalize(raw, "handoff_collapsed"))
            continue
        if kind in AUDIT_KINDS or status in DONE_STATUSES:
            audit.append(_normalize(raw, "deep_audit_collapsed"))
            continue
        if status in URGENT_STATUSES or kind in {"release_gate", "failure", "regression"} or _bool(raw.get("decision_required")):
            decide_candidates.append(_normalize(raw, "decide_now"))
            continue
        team_pulse.append(_normalize(raw, "team_pulse"))

    decide_candidates.sort(key=lambda item: (-int(item["priority_score"]), str(item.get("id") or item.get("title") or "")))
    decide_now = decide_candidates[:max_decide_now]
    overflow = decide_candidates[max_decide_now:]
    team_pulse.extend({**item, "lane": "team_pulse"} for item in overflow)

    return {
        "ok": True,
        "max_decide_now": max_decide_now,
        "input_count": len(decide_candidates) + len(team_pulse) + len(handoff) + len(audit),
        "decide_now": decide_now,
        "team_pulse": team_pulse,
        "handoff_collapsed": handoff,
        "deep_audit_collapsed": audit,
        "guardrails": dict(GUARDRAILS),
    }
