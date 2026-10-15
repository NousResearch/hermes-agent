"""
summarizer.py — Hermes output-pipeline submodule C§1.8

Phase 2 §4.2 reference: «Executive Summarizer condenses a ResultPackage from
an upstream system (OpenClaw, state-engine, or MCP tool) plus optional Company
KPIs into an ExecutiveSummary suitable for the executive front door. The summary
is kept to ≤5 bullet points, always includes a recommended next action, and
flags any SLA breaches or open approvals.»

Event emitted: hermes.summary.emitted
Wire-up: task C§1.9 (EventEmitter instance injected by turn_handler)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from agent.modules.event_emitter import EventEmitter


# ---------------------------------------------------------------------------
# Input types (stubs; C§1.9 replaces with @agrv/mission-contract imports)
# ---------------------------------------------------------------------------

class ResultPackage(BaseModel):
    """
    Upstream result bundle. Aligned with @agrv/mission-contract.ResultPackage
    (Phase 2 §6). Wire-up task C§1.9 replaces this stub.
    """
    mission_id: str
    success: bool
    outputs: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    duration_ms: Optional[int] = None
    sla_breach: bool = False
    pending_approvals: list[str] = Field(default_factory=list)


class CompanyKPIs(BaseModel):
    """
    Optional company-wide KPI snapshot. Injected by Hermes context layer;
    absent when the Company Model hasn't been loaded yet (Phase B§2).
    """
    kv: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

class ExecutiveSummary(BaseModel):
    """
    Executive summary per @agrv/mission-contract.ExecutiveSummarySchema.

    bullets:          ≤5 concise outcome statements.
    next_action:      single recommended next step for the executive.
    sla_breach:       True when a deadline was missed.
    pending_approvals: list of approval tokens awaiting decision.
    generated_at:     ISO-8601 UTC timestamp.
    kpi_deltas:       changed KPI values this mission touched (empty if no KPIs).
    """
    mission_id: str
    success: bool
    bullets: list[str]
    next_action: str
    sla_breach: bool
    pending_approvals: list[str]
    generated_at: str
    kpi_deltas: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Module-level emitter (injected by turn_handler)
# ---------------------------------------------------------------------------

_emitter: Optional[EventEmitter] = None


def set_emitter(emitter: EventEmitter) -> None:
    """Inject the shared event emitter.

    Called by turn_handler.run_turn() before processing.
    """
    global _emitter
    _emitter = emitter


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

_MAX_BULLETS = 5


def summarize(
    result: ResultPackage,
    kpis: Optional[CompanyKPIs] = None,
) -> ExecutiveSummary:
    """
    Produce an ExecutiveSummary from a ResultPackage and optional KPI snapshot.

    Summary construction rules (Phase 2 §4.2 — Executive Summarizer):
    1. Outcome bullet: success/failure statement with mission_id.
    2. Error bullets: up to 2 top errors when success=False.
    3. SLA bullet: added when sla_breach=True.
    4. Approval bullet: added when pending_approvals is non-empty.
    5. Duration bullet: added when duration_ms is present.
    Bullets are capped at _MAX_BULLETS (5); lower-priority bullets are dropped.

    next_action is derived from the result state (see _derive_next_action).

    Emits: hermes.summary.emitted (stdout JSON line)

    Ref: Phase 2 §4.2 — Executive Summarizer.
    """
    bullets: list[str] = []

    # 1. Outcome
    outcome = "Mission completed successfully." if result.success else "Mission failed."
    bullets.append(f"{outcome} (id: {result.mission_id})")

    # 2. Errors (max 2)
    for err in result.errors[:2]:
        if len(bullets) >= _MAX_BULLETS:
            break
        bullets.append(f"Error: {err}")

    # 3. SLA breach
    if result.sla_breach and len(bullets) < _MAX_BULLETS:
        bullets.append("SLA breach detected — deadline was not met.")

    # 4. Pending approvals
    if result.pending_approvals and len(bullets) < _MAX_BULLETS:
        count = len(result.pending_approvals)
        bullets.append(f"{count} approval(s) pending: {', '.join(result.pending_approvals[:3])}")

    # 5. Duration
    if result.duration_ms is not None and len(bullets) < _MAX_BULLETS:
        secs = result.duration_ms / 1000
        bullets.append(f"Completed in {secs:.1f}s.")

    # KPI deltas — attach any KPI values relevant to this mission's outputs
    kpi_deltas: dict[str, Any] = {}
    if kpis:
        for key in result.outputs:
            if key in kpis.kv:
                kpi_deltas[key] = kpis.kv[key]

    summary = ExecutiveSummary(
        mission_id=result.mission_id,
        success=result.success,
        bullets=bullets,
        next_action=_derive_next_action(result),
        sla_breach=result.sla_breach,
        pending_approvals=result.pending_approvals,
        generated_at=datetime.now(timezone.utc).isoformat(),
        kpi_deltas=kpi_deltas,
    )

    if _emitter is not None:
        _emitter.emit("hermes.summary.emitted", {
            "mission_id": summary.mission_id,
            "success": summary.success,
            "sla_breach": summary.sla_breach,
            "pending_approval_count": len(summary.pending_approvals),
            "bullet_count": len(summary.bullets),
        })

    return summary


def _derive_next_action(result: ResultPackage) -> str:
    """Derive the recommended next action from the result state."""
    if result.pending_approvals:
        return "Review and action pending approvals."
    if result.sla_breach:
        return "Investigate SLA breach; escalate if recurring."
    if not result.success:
        return "Review errors and retry or escalate the mission."
    return "No action required — mission closed successfully."
