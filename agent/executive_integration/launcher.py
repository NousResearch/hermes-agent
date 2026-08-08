"""ExecutiveLauncher — prepares ExecutiveLaunchRequest.

Default behavior does NOT execute Executive v2. It prepares a structured
request for operator approval and tracks its lifecycle (PENDING, APPROVED,
REJECTED, LAUNCHED, FAILED, CANCELLED).

When the explicit launch-edge canary flags are enabled, ``launch()`` may
create an ObjectiveEngine contract preview and must stop at
EXECUTION_PREVIEW_READY / CONTRACT_DRAFT. It must not start runtime work.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from .types import (
    ExecutiveLaunchRequest,
    LaunchStatus,
    ObjectiveGatewayDecision,
    RouteKind,
    LAUNCH_FINGERPRINT_KEYS,
    _now_iso8601,
    new_request_id,
)
from .objective_gateway import _flags_enabled


# ──────────────────────────────────────────────────────────────────────
# Expected phases (constant, deterministic)
# ──────────────────────────────────────────────────────────────────────


ALL_EXPECTED_PHASES: Tuple[str, ...] = (
    "phase1_classify",
    "phase2_link",
    "phase3_plan",
    "phase4a_policy",
    "phase4b_kanban_apply",
    "phase5_worker_dispatch",
    "phase6_success_evaluator",
    "phase7_recovery",
)


# ──────────────────────────────────────────────────────────────────────
# ExecutiveLauncher
# ──────────────────────────────────────────────────────────────────────


class ExecutiveLauncher:
    """Pure facade. Prepares ExecutiveLaunchRequests but does NOT execute.

    Cardinal rules:
      * No LLM. No provider. No network. No subprocess.
      * No DB write. No commit.
    """

    SCHEMA_VERSION = "eil.v1"

    def __init__(
        self,
        *,
        gateway: Any = None,
        policy_engine: Any = None,
    ) -> None:
        self._gateway = gateway
        self._policy_engine = policy_engine
        self._store: Dict[str, ExecutiveLaunchRequest] = {}
        self._launch_edge_results: Dict[str, Dict[str, Any]] = {}

    # ── public ────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return _flags_enabled()["integration_enabled"]

    def autolaunch_enabled(self) -> bool:
        return _flags_enabled()["autolaunch_enabled"]

    def launch_edge_enabled(self) -> bool:
        """Return True only when all dry launch-edge canary flags are on."""
        return _launch_edge_flags_enabled()["edge_ready"]

    def prepare(
        self,
        user_message: str,
        *,
        gateway_decision: Optional[ObjectiveGatewayDecision] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutiveLaunchRequest:
        """Build an ExecutiveLaunchRequest. Does NOT launch.

        The request is stored in-memory with status=PENDING.
        """
        if gateway_decision is None and self._gateway is not None:
            gateway_decision = self._gateway.route(user_message, context=context)

        flags = _flags_enabled()
        if not flags["integration_enabled"]:
            request_id = new_request_id()
            return ExecutiveLaunchRequest(
                request_id=request_id,
                objective_text=user_message,
                objective_id=None,
                expected_phases=(),
                estimated_complexity="unknown",
                risk_level="R0",
                risk_rationale="EIL disabled (HERMES_EXECUTIVE_INTEGRATION_ENABLED=0).",
                requires_human_approval=False,
                keywords_matched=(),
                intent_routing_strategy="",
                gateway_decision_fingerprint="",
                user_summary="EIL is disabled.",
                approval_request_id=None,
                status=LaunchStatus.FAILED,
                created_at=_now_iso8601(),
                created_by="ExecutiveLauncher",
            )

        if gateway_decision is None or gateway_decision.route_kind != RouteKind.EXECUTIVE:
            request_id = new_request_id()
            return ExecutiveLaunchRequest(
                request_id=request_id,
                objective_text=user_message,
                objective_id=None,
                expected_phases=(),
                estimated_complexity="unknown",
                risk_level="R0",
                risk_rationale="Not an EXECUTIVE route; no phases to launch.",
                requires_human_approval=False,
                keywords_matched=(),
                intent_routing_strategy="",
                gateway_decision_fingerprint="",
                user_summary="No Executive phases needed.",
                approval_request_id=None,
                status=LaunchStatus.PENDING,
                created_at=_now_iso8601(),
                created_by="ExecutiveLauncher",
            )

        # Build a real ExecutiveLaunchRequest.
        keywords = gateway_decision.matched_keywords
        intent_strategy = gateway_decision.intent_routing_strategy or ""
        complexity = _estimate_complexity(user_message, keywords)
        risk_level, risk_rationale = _estimate_risk(user_message, complexity)
        approval_required = risk_level in ("R5", "R6")

        request = ExecutiveLaunchRequest(
            request_id=new_request_id(),
            objective_text=user_message,
            objective_id=None,
            expected_phases=ALL_EXPECTED_PHASES,
            estimated_complexity=complexity,
            risk_level=risk_level,
            risk_rationale=risk_rationale,
            requires_human_approval=approval_required,
            keywords_matched=keywords,
            intent_routing_strategy=intent_strategy,
            gateway_decision_fingerprint=gateway_decision.fingerprint,
            user_summary=_summarize(user_message, keywords, risk_level),
            approval_request_id=None,
            status=LaunchStatus.PENDING,
            created_at=_now_iso8601(),
            created_by="ExecutiveLauncher",
        )
        self._store[request.request_id] = request
        return request

    def approve(
        self, request_id: str, *, approver_id: str
    ) -> ExecutiveLaunchRequest:
        """Mark a pending request as APPROVED. Returns the updated request."""
        existing = self._require(request_id)
        if existing.status != LaunchStatus.PENDING:
            return existing
        approval_id = f"approval-{approver_id}-{existing.request_id}"
        approved = ExecutiveLaunchRequest(
            request_id=existing.request_id,
            objective_text=existing.objective_text,
            objective_id=existing.objective_id,
            expected_phases=existing.expected_phases,
            estimated_complexity=existing.estimated_complexity,
            risk_level=existing.risk_level,
            risk_rationale=existing.risk_rationale,
            requires_human_approval=existing.requires_human_approval,
            keywords_matched=existing.keywords_matched,
            intent_routing_strategy=existing.intent_routing_strategy,
            gateway_decision_fingerprint=existing.gateway_decision_fingerprint,
            user_summary=existing.user_summary,
            approval_request_id=approval_id,
            status=LaunchStatus.APPROVED,
            created_at=existing.created_at,
            created_by=existing.created_by,
        )
        self._store[request_id] = approved
        return approved

    def reject(
        self, request_id: str, *, reason: str
    ) -> ExecutiveLaunchRequest:
        """Mark a pending request as REJECTED. Returns the updated request."""
        existing = self._require(request_id)
        if existing.status != LaunchStatus.PENDING:
            return existing
        rejected = ExecutiveLaunchRequest(
            request_id=existing.request_id,
            objective_text=existing.objective_text,
            objective_id=existing.objective_id,
            expected_phases=existing.expected_phases,
            estimated_complexity=existing.estimated_complexity,
            risk_level=existing.risk_level,
            risk_rationale=f"Rejected: {reason}",
            requires_human_approval=existing.requires_human_approval,
            keywords_matched=existing.keywords_matched,
            intent_routing_strategy=existing.intent_routing_strategy,
            gateway_decision_fingerprint=existing.gateway_decision_fingerprint,
            user_summary=existing.user_summary,
            approval_request_id=None,
            status=LaunchStatus.REJECTED,
            created_at=existing.created_at,
            created_by=existing.created_by,
        )
        self._store[request_id] = rejected
        return rejected

    def launch(
        self,
        request_id: str,
        *,
        objective_engine_factory: Any = None,
        persist_to_state_meta: bool = False,
        stop_after_contract: bool = True,
    ) -> ExecutiveLaunchRequest:
        """Launch the approved request.

        With legacy flags only, preserve the historical behavior: return
        LAUNCHED when autolaunch is enabled and FAILED when disabled.

        With explicit launch-edge canary flags, run only the ObjectiveEngine
        preview pipeline: submit -> normalize -> classify -> discover ->
        generate_contract, then stop at CONTRACT_DRAFT and expose
        EXECUTION_PREVIEW_READY. No runtime, workers, Kanban, providers,
        subprocesses, network, or default persistence are reached here.
        """
        existing = self._require(request_id)
        if existing.status != LaunchStatus.APPROVED:
            return ExecutiveLaunchRequest(
                request_id=existing.request_id,
                objective_text=existing.objective_text,
                objective_id=existing.objective_id,
                expected_phases=existing.expected_phases,
                estimated_complexity=existing.estimated_complexity,
                risk_level=existing.risk_level,
                risk_rationale=existing.risk_rationale,
                requires_human_approval=existing.requires_human_approval,
                keywords_matched=existing.keywords_matched,
                intent_routing_strategy=existing.intent_routing_strategy,
                gateway_decision_fingerprint=existing.gateway_decision_fingerprint,
                user_summary=existing.user_summary,
                approval_request_id=existing.approval_request_id,
                status=LaunchStatus.FAILED,
                created_at=existing.created_at,
                created_by=existing.created_by,
            )

        if not self.autolaunch_enabled():
            failed = ExecutiveLaunchRequest(
                request_id=existing.request_id,
                objective_text=existing.objective_text,
                objective_id=existing.objective_id,
                expected_phases=existing.expected_phases,
                estimated_complexity=existing.estimated_complexity,
                risk_level=existing.risk_level,
                risk_rationale="Autolaunch disabled (HERMES_EXECUTIVE_AUTOLAUNCH_ENABLED=0).",
                requires_human_approval=existing.requires_human_approval,
                keywords_matched=existing.keywords_matched,
                intent_routing_strategy=existing.intent_routing_strategy,
                gateway_decision_fingerprint=existing.gateway_decision_fingerprint,
                user_summary=existing.user_summary,
                approval_request_id=existing.approval_request_id,
                status=LaunchStatus.FAILED,
                created_at=existing.created_at,
                created_by=existing.created_by,
            )
            self._store[request_id] = failed
            return failed

        edge_flags = _launch_edge_flags_enabled()
        if edge_flags["edge_ready"]:
            return self._launch_preview_edge(
                existing,
                objective_engine_factory=objective_engine_factory,
                persist_to_state_meta=persist_to_state_meta,
                stop_after_contract=stop_after_contract,
            )

        # Autolaunch enabled, launch edge disabled — preserve legacy status.
        launched = ExecutiveLaunchRequest(
            request_id=existing.request_id,
            objective_text=existing.objective_text,
            objective_id=existing.objective_id,
            expected_phases=existing.expected_phases,
            estimated_complexity=existing.estimated_complexity,
            risk_level=existing.risk_level,
            risk_rationale=existing.risk_rationale,
            requires_human_approval=existing.requires_human_approval,
            keywords_matched=existing.keywords_matched,
            intent_routing_strategy=existing.intent_routing_strategy,
            gateway_decision_fingerprint=existing.gateway_decision_fingerprint,
            user_summary=existing.user_summary,
            approval_request_id=existing.approval_request_id,
            status=LaunchStatus.LAUNCHED,
            created_at=existing.created_at,
            created_by=existing.created_by,
        )
        self._store[request_id] = launched
        return launched

    def cancel(self, request_id: str) -> ExecutiveLaunchRequest:
        """Cancel a PENDING or APPROVED request. Returns the updated request."""
        existing = self._require(request_id)
        if existing.status not in (LaunchStatus.PENDING, LaunchStatus.APPROVED):
            return existing
        cancelled = ExecutiveLaunchRequest(
            request_id=existing.request_id,
            objective_text=existing.objective_text,
            objective_id=existing.objective_id,
            expected_phases=existing.expected_phases,
            estimated_complexity=existing.estimated_complexity,
            risk_level=existing.risk_level,
            risk_rationale=existing.risk_rationale,
            requires_human_approval=existing.requires_human_approval,
            keywords_matched=existing.keywords_matched,
            intent_routing_strategy=existing.intent_routing_strategy,
            gateway_decision_fingerprint=existing.gateway_decision_fingerprint,
            user_summary=existing.user_summary,
            approval_request_id=existing.approval_request_id,
            status=LaunchStatus.CANCELLED,
            created_at=existing.created_at,
            created_by=existing.created_by,
        )
        self._store[request_id] = cancelled
        return cancelled

    def get(self, request_id: str) -> Optional[ExecutiveLaunchRequest]:
        return self._store.get(request_id)

    def get_launch_edge_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Return the in-memory preview result for a launch-edge canary run."""
        result = self._launch_edge_results.get(request_id)
        return dict(result) if result is not None else None

    # ── private ───────────────────────────────────────────────

    def _require(self, request_id: str) -> ExecutiveLaunchRequest:
        existing = self._store.get(request_id)
        if existing is None:
            raise KeyError(f"unknown request_id: {request_id}")
        return existing

    def _launch_preview_edge(
        self,
        existing: ExecutiveLaunchRequest,
        *,
        objective_engine_factory: Any = None,
        persist_to_state_meta: bool = False,
        stop_after_contract: bool = True,
    ) -> ExecutiveLaunchRequest:
        """Run the controlled EIL -> ObjectiveEngine canary edge.

        The raw ``objective_text`` from prepare() is passed intact into
        ObjectiveEngine. The gateway fingerprint remains only an identifier.
        """
        flags = _launch_edge_flags_enabled()
        if not stop_after_contract or not flags["stop_after_contract"]:
            failed = _copy_request(
                existing,
                status=LaunchStatus.FAILED,
                risk_rationale="EXECUTION_BLOCKED_BY_POLICY: stop_after_contract is required.",
            )
            self._store[existing.request_id] = failed
            return failed
        if persist_to_state_meta or flags["persist_state"]:
            failed = _copy_request(
                existing,
                status=LaunchStatus.FAILED,
                risk_rationale="EXECUTION_BLOCKED_BY_DEFAULT_OFF: persistence is not enabled for the dry canary.",
            )
            self._store[existing.request_id] = failed
            return failed

        try:
            if objective_engine_factory is None:
                from agent.executive.objective_engine import ObjectiveEngine

                objective_engine_factory = lambda: ObjectiveEngine(
                    user_id="executive-launch-edge",
                    enabled=True,
                )
            engine = objective_engine_factory()
            objective_id = engine.run_pipeline(
                existing.objective_text,
                constraints=[
                    "launch_edge_canary=true",
                    "persist_to_state_meta=false",
                    "stop_after_contract=true",
                ],
                persist_to_state_meta=False,
            )
            state = engine.get_state(objective_id)
            stop_state = getattr(state.state, "value", str(state.state))
            if stop_state != "CONTRACT_DRAFT" or not state.contract:
                raise RuntimeError(f"unexpected launch-edge stop state: {stop_state}")

            result = {
                "result": "EXECUTION_PREVIEW_READY",
                "objective_id": objective_id,
                "request_id": existing.request_id,
                "raw_user_message": existing.objective_text,
                "raw_user_message_intact": state.objective_text == existing.objective_text,
                "gateway_decision_fingerprint": existing.gateway_decision_fingerprint,
                "fingerprint_present": bool(existing.gateway_decision_fingerprint),
                "submit_state": "OBJECTIVE_DRAFT",
                "stop_state": stop_state,
                "contract_draft_generated": bool(state.contract),
                "persisted": False,
                "runtime_launch_allowed": False,
                "forbidden_effects": 0,
                "dry_runs": {
                    "capability_discovery": True,
                    "goal_discovery": True,
                    "knowledge_discovery": True,
                },
            }
            self._launch_edge_results[existing.request_id] = result
            preview_ready = _copy_request(
                existing,
                objective_id=objective_id,
                status=LaunchStatus.EXECUTION_PREVIEW_READY,
                risk_rationale=f"EXECUTION_PREVIEW_READY: stopped at {stop_state}; no runtime launched.",
            )
            self._store[existing.request_id] = preview_ready
            return preview_ready
        except Exception as exc:
            failed = _copy_request(
                existing,
                status=LaunchStatus.FAILED,
                risk_rationale=f"EXECUTION_BLOCKED_BY_POLICY: {exc}",
            )
            self._store[existing.request_id] = failed
            return failed


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _estimate_complexity(message: str, keywords: Tuple[str, ...]) -> str:
    """Estimate complexity: low | medium | high."""
    n_words = len(message.split())
    n_keywords = len(keywords)
    if n_words < 8 and n_keywords == 0:
        return "low"
    if n_words < 20 and n_keywords <= 1:
        return "medium"
    return "high"


def _estimate_risk(message: str, complexity: str) -> Tuple[str, str]:
    """Estimate risk level R0 | R3 | R4 | R5 | R6 + rationale."""
    msg_lc = message.lower()
    if any(kw in msg_lc for kw in ("deploy", "production", "external", "publish", "release")):
        return "R6", "R6 risk: external/production keyword detected (requires human approval)."
    if complexity == "high":
        return "R5", "R5 risk: high-complexity request (requires human approval)."
    if complexity == "medium":
        return "R4", "R4 risk: medium-complexity request (Kanban approval)."
    return "R3", "R3 risk: low-complexity request."



def _launch_edge_flags_enabled() -> Dict[str, bool]:
    """Evaluate the explicit launch-edge canary flags.

    The edge is default-off. Legacy autolaunch alone is intentionally not
    enough to construct ObjectiveEngine or create a contract preview.
    """
    flags = _flags_enabled()
    edge_enabled = os.environ.get("HERMES_EXECUTIVE_LAUNCH_EDGE_ENABLED", "0") == "1"
    create_contract = os.environ.get("HERMES_EXECUTIVE_LAUNCH_EDGE_CREATE_CONTRACT", "0") == "1"
    persist_state = os.environ.get("HERMES_EXECUTIVE_LAUNCH_EDGE_PERSIST_STATE", "0") == "1"
    stop_after_contract = os.environ.get("HERMES_EXECUTIVE_LAUNCH_EDGE_STOP_AFTER_CONTRACT", "1") == "1"
    forbid_external_effects = os.environ.get("HERMES_EXECUTIVE_FORBID_EXTERNAL_EFFECTS", "1") == "1"
    return {
        "integration_enabled": flags["integration_enabled"],
        "gateway_enabled": flags["gateway_enabled"],
        "autolaunch_enabled": flags["autolaunch_enabled"],
        "edge_enabled": edge_enabled,
        "create_contract": create_contract,
        "persist_state": persist_state,
        "stop_after_contract": stop_after_contract,
        "forbid_external_effects": forbid_external_effects,
        "edge_ready": (
            flags["integration_enabled"]
            and flags["gateway_enabled"]
            and flags["autolaunch_enabled"]
            and edge_enabled
            and create_contract
            and stop_after_contract
            and forbid_external_effects
            and not persist_state
        ),
    }


def _copy_request(
    existing: ExecutiveLaunchRequest,
    *,
    status: LaunchStatus,
    objective_id: Optional[str] = None,
    risk_rationale: Optional[str] = None,
) -> ExecutiveLaunchRequest:
    """Copy an immutable ExecutiveLaunchRequest with controlled updates."""
    return ExecutiveLaunchRequest(
        request_id=existing.request_id,
        objective_text=existing.objective_text,
        objective_id=objective_id if objective_id is not None else existing.objective_id,
        expected_phases=existing.expected_phases,
        estimated_complexity=existing.estimated_complexity,
        risk_level=existing.risk_level,
        risk_rationale=risk_rationale if risk_rationale is not None else existing.risk_rationale,
        requires_human_approval=existing.requires_human_approval,
        keywords_matched=existing.keywords_matched,
        intent_routing_strategy=existing.intent_routing_strategy,
        gateway_decision_fingerprint=existing.gateway_decision_fingerprint,
        user_summary=existing.user_summary,
        approval_request_id=existing.approval_request_id,
        status=status,
        created_at=existing.created_at,
        created_by=existing.created_by,
    )

def _summarize(message: str, keywords: Tuple[str, ...], risk_level: str) -> str:
    """Build a one-line user summary."""
    keyword_list = list(keywords) if keywords else ["(no executive keywords)"]
    return (
        f"Your request was classified as EXECUTIVE (risk {risk_level}). "
        f"Matched keywords: {keyword_list}. "
        f"Please review and approve before launch."
    )


__all__ = [
    "ExecutiveLauncher",
    "ALL_EXPECTED_PHASES",
    "LAUNCH_FINGERPRINT_KEYS",
    "_launch_edge_flags_enabled",
]
