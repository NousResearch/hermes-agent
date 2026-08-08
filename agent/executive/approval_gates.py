"""Phase 4A Approval Gates — 8-layer consolidated evaluator.

Centralizes the 4 Phase 1+2+3 approval layers (default, STRATEGIC,
HIGH_RISK, cross-session) plus 4 new Phase 4A-specific layers
(Kanban_create, Worker_spawn, External_call, Token_expiry) into a
single ``evaluate_approval_gates(...)`` entry point.

The output is an ``ApprovalGateResult`` — a frozen dataclass that
records:

* ``approved`` (bool) — True when all applicable gates pass.
* ``layer_results`` (tuple[dict, ...]) — per-layer pass/fail trace.
* ``approval_request`` (``ApprovalRequest``) — built only when all
  applicable gates pass; ``None`` otherwise.

Raises ``BridgeApprovalError`` on the first failing gate. Layer 1
failure subsumes Layer 2. Layer 3 failure subsumes Layers 5-7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .goalmanager_bridge import BridgeApprovalError
from .types import (
    ApprovalRequest,
    PolicyDecision,
    RiskLevel,
    compute_request_fingerprint,
    now_iso8601,
)


# ──────────────────────────────────────────────────────────────────────
# ApprovalGateResult dataclass
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ApprovalGateResult:
    """Result of ``evaluate_approval_gates(...)``.

    ``approved`` is True iff every applicable layer passed. When
    ``approved`` is True, ``approval_request`` is set; otherwise it
    is ``None``.

    ``layer_results`` is a tuple of dicts (one per layer that fired),
    each shaped::

        {
            "layer": <int>,
            "name": <str>,
            "passed": <bool>,
            "reason": <str>,
        }
    """

    approved: bool
    layer_results: tuple[dict, ...] = ()
    approval_request: Optional[ApprovalRequest] = None
    failure_layer: Optional[int] = None
    failure_reason: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _parse_isoformat(value: str) -> datetime:
    """Parse an ISO 8601 string into an aware datetime (UTC fallback)."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _default_reason(decision: PolicyDecision) -> str:
    return f"{decision.risk_level.name} policy decision for objective {decision.objective_id}"


# ──────────────────────────────────────────────────────────────────────
# 8-layer evaluator
# ──────────────────────────────────────────────────────────────────────

def evaluate_approval_gates(
    policy_decision: PolicyDecision,
    *,
    approver_id: Optional[str] = None,
    approval_token: Optional[str] = None,
    kanban_approver_id: Optional[str] = None,
    worker_approver_id: Optional[str] = None,
    external_approver_id: Optional[str] = None,
    cross_session: bool = False,
    session_id: Optional[str] = None,
    expiry: Optional[str] = None,
    renewal: bool = False,
    approval_reason: str = "",
    scope: Optional[tuple[str, ...]] = None,
) -> ApprovalGateResult:
    """Run the 8 approval layers and return an ``ApprovalGateResult``.

    Layers (in evaluation order):

    1. default — ``approver_id`` required when ``approval_required``.
    2. STRATEGIC — ``approver_id`` required when any approval_req has
       a STRATEGIC gate.
    3. HIGH_RISK — ``approval_token`` required at R4+.
    4. cross-session — ``cross_session=True`` required when
       ``session_id`` is supplied.
    5. Kanban_create — ``kanban_approver_id`` required at R4.
    6. Worker_spawn — ``worker_approver_id`` required at R5.
    7. External_call — ``external_approver_id`` required at R6.
    8. Token_expiry — ``renewal=True`` required when ``expiry`` is
       in the past.

    Does NOT mutate state_meta. Does NOT spawn workers, run Kanban,
    or call any provider. Raises ``BridgeApprovalError`` on the
    first failing gate (subsumption rules apply).
    """
    if not isinstance(policy_decision, PolicyDecision):
        raise BridgeApprovalError(
            f"approval evaluate: policy_decision must be a PolicyDecision (got {type(policy_decision).__name__})"
        )

    layer_results: list[dict] = []

    # ── Layer 1: default ──────────────────────────────────────────
    if policy_decision.approval_required:
        if not approver_id:
            layer_results.append({
                "layer": 1,
                "name": "default",
                "passed": False,
                "reason": "approver_id required when approval_required=True",
            })
            raise BridgeApprovalError(
                "Layer 1: approver_id required when approval_required=True"
            )
        layer_results.append({
            "layer": 1,
            "name": "default",
            "passed": True,
            "reason": "approver_id present",
        })
    else:
        layer_results.append({
            "layer": 1,
            "name": "default",
            "passed": True,
            "reason": "approval not required (skipped)",
        })

    # ── Layer 2: STRATEGIC ─────────────────────────────────────────
    has_strategic = any(
        "STRATEGIC" in str(ar.get("gate", "")).upper()
        for ar in policy_decision.approval_requirements
    )
    if has_strategic:
        if not approver_id:
            layer_results.append({
                "layer": 2,
                "name": "STRATEGIC",
                "passed": False,
                "reason": "STRATEGIC requires approver_id",
            })
            raise BridgeApprovalError(
                "Layer 2: STRATEGIC requires approver_id"
            )
        layer_results.append({
            "layer": 2,
            "name": "STRATEGIC",
            "passed": True,
            "reason": "approver_id present",
        })
    else:
        layer_results.append({
            "layer": 2,
            "name": "STRATEGIC",
            "passed": True,
            "reason": "no STRATEGIC gate (skipped)",
        })

    # ── Layer 3: HIGH_RISK ────────────────────────────────────────
    if int(policy_decision.risk_level) >= int(RiskLevel.R4):
        if not approval_token:
            layer_results.append({
                "layer": 3,
                "name": "HIGH_RISK",
                "passed": False,
                "reason": "HIGH_RISK (R4+) requires approval_token",
            })
            raise BridgeApprovalError(
                "Layer 3: HIGH_RISK (R4+) requires approval_token"
            )
        layer_results.append({
            "layer": 3,
            "name": "HIGH_RISK",
            "passed": True,
            "reason": "approval_token present",
        })
    else:
        layer_results.append({
            "layer": 3,
            "name": "HIGH_RISK",
            "passed": True,
            "reason": "below R4 (skipped)",
        })

    # ── Layer 4: cross-session ────────────────────────────────────
    if session_id is not None:
        if not cross_session:
            layer_results.append({
                "layer": 4,
                "name": "cross-session",
                "passed": False,
                "reason": "cross_session=True required when session_id supplied",
            })
            raise BridgeApprovalError(
                "Layer 4: cross-session requires cross_session=True when a session_id is supplied"
            )
        layer_results.append({
            "layer": 4,
            "name": "cross-session",
            "passed": True,
            "reason": "cross_session=True",
        })
    else:
        layer_results.append({
            "layer": 4,
            "name": "cross-session",
            "passed": True,
            "reason": "no session_id (skipped)",
        })

    # ── Layer 5: Kanban_create (R4) ──────────────────────────────
    if int(policy_decision.risk_level) == int(RiskLevel.R4):
        if not kanban_approver_id:
            layer_results.append({
                "layer": 5,
                "name": "Kanban_create",
                "passed": False,
                "reason": "R4 (Kanban apply) requires kanban_approver_id",
            })
            raise BridgeApprovalError(
                "Layer 5: R4 (Kanban apply) requires kanban_approver_id"
            )
        layer_results.append({
            "layer": 5,
            "name": "Kanban_create",
            "passed": True,
            "reason": "kanban_approver_id present",
        })
    else:
        layer_results.append({
            "layer": 5,
            "name": "Kanban_create",
            "passed": True,
            "reason": "not R4 (skipped)",
        })

    # ── Layer 6: Worker_spawn (R5) ───────────────────────────────
    if int(policy_decision.risk_level) == int(RiskLevel.R5):
        if not worker_approver_id:
            layer_results.append({
                "layer": 6,
                "name": "Worker_spawn",
                "passed": False,
                "reason": "R5 (workers) requires worker_approver_id",
            })
            raise BridgeApprovalError(
                "Layer 6: R5 (workers) requires worker_approver_id"
            )
        layer_results.append({
            "layer": 6,
            "name": "Worker_spawn",
            "passed": True,
            "reason": "worker_approver_id present",
        })
    else:
        layer_results.append({
            "layer": 6,
            "name": "Worker_spawn",
            "passed": True,
            "reason": "not R5 (skipped)",
        })

    # ── Layer 7: External_call (R6) ──────────────────────────────
    if int(policy_decision.risk_level) == int(RiskLevel.R6):
        if not external_approver_id:
            layer_results.append({
                "layer": 7,
                "name": "External_call",
                "passed": False,
                "reason": "R6 (external) requires external_approver_id",
            })
            raise BridgeApprovalError(
                "Layer 7: R6 (external) requires external_approver_id"
            )
        layer_results.append({
            "layer": 7,
            "name": "External_call",
            "passed": True,
            "reason": "external_approver_id present",
        })
    else:
        layer_results.append({
            "layer": 7,
            "name": "External_call",
            "passed": True,
            "reason": "not R6 (skipped)",
        })

    # ── Layer 8: Token_expiry ─────────────────────────────────────
    if expiry:
        try:
            expiry_dt = _parse_isoformat(expiry)
        except ValueError as exc:
            layer_results.append({
                "layer": 8,
                "name": "Token_expiry",
                "passed": False,
                "reason": f"invalid expiry format: {exc}",
            })
            raise BridgeApprovalError(
                f"Layer 8: invalid expiry format: {exc}"
            ) from exc
        now = datetime.now(timezone.utc)
        if now > expiry_dt and not renewal:
            layer_results.append({
                "layer": 8,
                "name": "Token_expiry",
                "passed": False,
                "reason": "token expired; renewal=True required",
            })
            raise BridgeApprovalError(
                "Layer 8: token expired; renewal=True required"
            )
        layer_results.append({
            "layer": 8,
            "name": "Token_expiry",
            "passed": True,
            "reason": "renewal accepted" if renewal else "expiry not yet reached",
        })
    else:
        layer_results.append({
            "layer": 8,
            "name": "Token_expiry",
            "passed": True,
            "reason": "no expiry set (skipped)",
        })

    # ── All layers pass; build ApprovalRequest ────────────────────
    final_scope = (
        tuple(scope) if scope else tuple(policy_decision.allowed_actions)
    )

    fingerprint = compute_request_fingerprint(
        objective_id=policy_decision.objective_id,
        risk_level=policy_decision.risk_level,
        approver_id=approver_id,
        approval_token=approval_token,
        kanban_approver_id=kanban_approver_id,
        worker_approver_id=worker_approver_id,
        external_approver_id=external_approver_id,
        scope=final_scope,
    )

    request = ApprovalRequest(
        objective_id=policy_decision.objective_id,
        risk_level=policy_decision.risk_level,
        approver_id=approver_id,
        approval_token=approval_token,
        kanban_approver_id=kanban_approver_id,
        worker_approver_id=worker_approver_id,
        external_approver_id=external_approver_id,
        approval_reason=approval_reason or _default_reason(policy_decision),
        scope=final_scope,
        expiry=expiry,
        created_at=now_iso8601(),
        request_fingerprint=fingerprint,
        policy_decision_fingerprint=policy_decision.decision_fingerprint,
    )

    return ApprovalGateResult(
        approved=True,
        layer_results=tuple(layer_results),
        approval_request=request,
        failure_layer=None,
        failure_reason=None,
    )


__all__ = [
    "ApprovalGateEvaluator",
    "ApprovalGateResult",
    "evaluate_approval_gates",
]


# ──────────────────────────────────────────────────────────────────────
# High-level facade: ApprovalGateEvaluator
# ──────────────────────────────────────────────────────────────────────

class ApprovalGateEvaluator:
    """Thin object-oriented facade for ``evaluate_approval_gates``.

    Wraps the pure module-level function with a class API so that
    callers can hold an instance and reuse it across many objectives.
    Does NOT mutate state_meta.
    """

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        policy_decision: PolicyDecision,
        *,
        approver_id: Optional[str] = None,
        approval_token: Optional[str] = None,
        kanban_approver_id: Optional[str] = None,
        worker_approver_id: Optional[str] = None,
        external_approver_id: Optional[str] = None,
        cross_session: bool = False,
        session_id: Optional[str] = None,
        expiry: Optional[str] = None,
        renewal: bool = False,
        approval_reason: str = "",
        scope: Optional[tuple[str, ...]] = None,
    ) -> ApprovalGateResult:
        return evaluate_approval_gates(
            policy_decision,
            approver_id=approver_id,
            approval_token=approval_token,
            kanban_approver_id=kanban_approver_id,
            worker_approver_id=worker_approver_id,
            external_approver_id=external_approver_id,
            cross_session=cross_session,
            session_id=session_id,
            expiry=expiry,
            renewal=renewal,
            approval_reason=approval_reason,
            scope=scope,
        )