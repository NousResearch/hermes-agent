"""Deterministic dispatch manifest lookup for Hermes production orders.

Slice 1 only: resolve the next profile task from a reconstructed production
order without invoking any profile, mutating workflow state, or writing back
to Kanban.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import sqlite3
from typing import Any, Iterable

from .production_order_db import (
    ProductionOrder,
    STATE_OWNERS,
    StageEntry,
    WORKFLOW_SPEC_SOURCE,
    list_production_orders,
)


class DispatchManifestError(ValueError):
    """Raised when a production order cannot be mapped to a dispatch manifest."""


@dataclass(frozen=True)
class DispatchManifest:
    production_order_id: str
    current_state: str
    current_owner_profile: str
    target_profile: str
    target_child_card_id: str
    task_type: str
    required_input_packet: str
    expected_result_packet: str
    bridge_function: str
    manual_fallback: dict[str, Any]
    stop_conditions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable copy of the manifest."""
        data = asdict(self)
        data["stop_conditions"] = list(self.stop_conditions)
        return data


def build_dispatch_manifest(
    conn: sqlite3.Connection,
    production_order_id: str,
) -> DispatchManifest:
    """Load a production order from SQLite and build its dispatch manifest."""
    po = _load_production_order(conn, production_order_id)
    _validate_child_graph(conn, po)
    return dispatch_manifest_for_order(po)


def dispatch_manifest_for_order(po: ProductionOrder) -> DispatchManifest:
    """Build a deterministic manifest from a reconstructed production order."""
    _validate_reconstructed_order(po)

    state = po.current_state
    if state == "PRODUCTION_ORDER_CREATED":
        return _orchestrator_triage_manifest(po, task_type="orchestrator_triage")
    if state == "ORCHESTRATOR_TRIAGE":
        if _has_default_rejection_provenance(po.stage_history):
            return _orchestrator_default_rejection_classification_manifest(po)
        return _orchestrator_triage_manifest(po, task_type="orchestrator_triage")
    if state == "ARCHITECT_SPEC":
        return _architect_spec_manifest(po)
    if state in {"ARCHITECT_READY_FOR_DEV", "DEV_IMPLEMENTING"}:
        return _dev_build_manifest(po, state)
    if state in {"DEV_COMPLETE", "AUDIT_REVIEW"}:
        return _audit_review_manifest(po, state)
    if state == "AUDIT_REJECTED":
        return _orchestrator_rework_manifest(po)
    if state in {"AUDIT_PASSED", "ARCHITECT_RECONCILE"}:
        return _architect_reconcile_manifest(po, state)
    if state in {"ARCHITECT_ACCEPTED", "DEFAULT_FINAL_REVIEW"}:
        return _default_final_review_manifest(po, state)
    if state == "DEFAULT_REJECTED":
        return _default_rejection_triage_manifest(po)
    if state == "DEV_REWORK":
        return _dev_rework_manifest(po)

    raise DispatchManifestError(
        f"Unsupported dispatch state {state!r} for production order "
        f"{po.production_order_id!r}. Supported states: "
        f"{_supported_dispatch_states_text()}"
    )


def _load_production_order(
    conn: sqlite3.Connection,
    production_order_id: str,
) -> ProductionOrder:
    matches = [
        order for order in list_production_orders(conn)
        if order.production_order_id == production_order_id
    ]
    if not matches:
        raise DispatchManifestError(
            f"production order {production_order_id!r} not found"
        )
    return matches[0]


def _validate_reconstructed_order(po: ProductionOrder) -> None:
    if not po.production_order_id:
        raise DispatchManifestError("production_order_id is required")
    expected_owner = STATE_OWNERS.get(po.current_state)
    if expected_owner is None:
        raise DispatchManifestError(
            f"No owner is defined for state {po.current_state!r}; the dispatch "
            "manifest layer only supports workflow states with an assigned owner"
        )
    if po.current_owner_profile != expected_owner:
        raise DispatchManifestError(
            f"Production order {po.production_order_id!r} is owned by "
            f"{po.current_owner_profile!r}; expected {expected_owner!r} for "
            f"state {po.current_state!r}"
        )
    if not po.parent_kanban_card_id:
        raise DispatchManifestError(
            f"Production order {po.production_order_id!r} is missing its parent Kanban card"
        )
    if len(po.child_kanban_card_ids) != 6:
        raise DispatchManifestError(
            f"Production order {po.production_order_id!r} must have exactly 6 child cards; "
            f"found {len(po.child_kanban_card_ids)}"
        )
    if len(set(po.child_kanban_card_ids)) != 6:
        raise DispatchManifestError(
            f"Production order {po.production_order_id!r} has duplicate child card IDs"
        )


def _validate_child_graph(conn: sqlite3.Connection, po: ProductionOrder) -> None:
    placeholders = ",".join(["?"] * len(po.child_kanban_card_ids))
    rows = conn.execute(
        f"SELECT id FROM tasks WHERE id IN ({placeholders})",
        tuple(po.child_kanban_card_ids),
    ).fetchall()
    if len(rows) != 6:
        raise DispatchManifestError(
            f"Production order {po.production_order_id!r} references missing child card(s)"
        )


def _has_default_rejection_provenance(stage_history: Iterable[StageEntry]) -> bool:
    return any(
        entry.from_state == "DEFAULT_REJECTED" and entry.to_state == "ORCHESTRATOR_TRIAGE"
        for entry in stage_history
    )


def _child_card_id(po: ProductionOrder, index: int) -> str:
    try:
        return po.child_kanban_card_ids[index]
    except IndexError as exc:  # pragma: no cover - guarded by graph validation
        raise DispatchManifestError(
            f"Production order {po.production_order_id!r} does not have child card index {index + 1}"
        ) from exc


def _manual_fallback(
    *,
    po: ProductionOrder,
    target_profile: str,
    target_child_card_id: str,
    task_type: str,
    required_input_packet: str,
    expected_result_packet: str,
    bridge_function: str,
    stop_conditions: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "enabled": True,
        "target_profile": target_profile,
        "target_child_card_id": target_child_card_id,
        "task_type": task_type,
        "source_truth": WORKFLOW_SPEC_SOURCE,
        "required_input_packet": required_input_packet,
        "expected_result_packet": expected_result_packet,
        "bridge_function": bridge_function,
        "task_prompt_template": None,
        "stop_conditions": list(stop_conditions),
        "notes": (
            "Manual fallback is metadata only in Slice 1; the bridge does not "
            "generate a copy/paste task prompt yet."
        ),
        "production_order_id": po.production_order_id,
        "current_state": po.current_state,
    }


def _make_manifest(
    po: ProductionOrder,
    *,
    target_profile: str,
    target_child_index: int,
    task_type: str,
    required_input_packet: str,
    expected_result_packet: str,
    bridge_function: str,
    stop_conditions: tuple[str, ...],
) -> DispatchManifest:
    target_child_card_id = _child_card_id(po, target_child_index)
    return DispatchManifest(
        production_order_id=po.production_order_id,
        current_state=po.current_state,
        current_owner_profile=po.current_owner_profile,
        target_profile=target_profile,
        target_child_card_id=target_child_card_id,
        task_type=task_type,
        required_input_packet=required_input_packet,
        expected_result_packet=expected_result_packet,
        bridge_function=bridge_function,
        manual_fallback=_manual_fallback(
            po=po,
            target_profile=target_profile,
            target_child_card_id=target_child_card_id,
            task_type=task_type,
            required_input_packet=required_input_packet,
            expected_result_packet=expected_result_packet,
            bridge_function=bridge_function,
            stop_conditions=stop_conditions,
        ),
        stop_conditions=stop_conditions,
    )


def _orchestrator_triage_manifest(
    po: ProductionOrder,
    *,
    task_type: str,
) -> DispatchManifest:
    return _make_manifest(
        po,
        target_profile="orchestrator_os",
        target_child_index=0,
        task_type=task_type,
        required_input_packet="orchestrator_handoff_packet",
        expected_result_packet="architect_handoff_packet",
        bridge_function="run_orchestrator_triage_bridge",
        stop_conditions=(
            "Do not invoke a profile in Slice 1; return the deterministic manifest only.",
            "The production order must keep exactly six linked child cards.",
            "Pause if the workflow state is unsupported or the owner mismatches the state.",
        ),
    )


def _orchestrator_default_rejection_classification_manifest(
    po: ProductionOrder,
) -> DispatchManifest:
    return _make_manifest(
        po,
        target_profile="orchestrator_os",
        target_child_index=0,
        task_type="orchestrator_default_rejection_classification",
        required_input_packet="default_rejection_handoff_packet",
        expected_result_packet="orchestrator_classification_packet",
        bridge_function="run_orchestrator_classification_bridge",
        stop_conditions=(
            "Default rejection provenance must be present in stage history.",
            "Route targets that would require SPEC_REWORK remain deferred in this slice.",
            "Do not invoke a profile in Slice 1; return the deterministic manifest only.",
        ),
    )


def _architect_spec_manifest(po: ProductionOrder) -> DispatchManifest:
    return _make_manifest(
        po,
        target_profile="architect_os",
        target_child_index=1,
        task_type="architect_spec",
        required_input_packet="architect_handoff_packet",
        expected_result_packet="architect_spec_packet",
        bridge_function="run_architect_spec_bridge",
        stop_conditions=(
            "The frozen ArchitectOS handoff must be present on the second child card.",
            "Do not expand scope beyond the approved brief.",
            "Do not invoke a profile in Slice 1; return the deterministic manifest only.",
        ),
    )


def _dev_build_manifest(po: ProductionOrder, state: str) -> DispatchManifest:
    return _make_manifest(
        po,
        target_profile="dev_os",
        target_child_index=2,
        task_type="dev_build",
        required_input_packet="devos_handoff_packet",
        expected_result_packet="devos_build_packet",
        bridge_function="run_devos_complete_bridge",
        stop_conditions=(
            f"Current state {state!r} must still point at the frozen DevOS handoff.",
            "Implementation evidence must stay within the approved brief and spec.",
            "Do not invoke a profile in Slice 1; return the deterministic manifest only.",
        ),
    )


def _audit_review_manifest(po: ProductionOrder, state: str) -> DispatchManifest:
    return _make_manifest(
        po,
        target_profile="audit_os",
        target_child_index=3,
        task_type="audit_review",
        required_input_packet="auditos_handoff_packet",
        expected_result_packet="auditos_review_packet",
        bridge_function="run_auditos_review_complete_bridge",
        stop_conditions=(
            f"Current state {state!r} must still point at the frozen AuditOS handoff.",
            "AuditOS must return a validated review packet or the workflow must pause.",
            "Do not invoke a profile in Slice 1; return the deterministic manifest only.",
        ),
    )


def _dev_rework_manifest(po: ProductionOrder) -> DispatchManifest:
    return _make_manifest(
        po,
        target_profile="dev_os",
        target_child_index=2,
        task_type="dev_rework",
        required_input_packet="devos_rework_handoff_packet",
        expected_result_packet="devos_build_packet",
        bridge_function="run_devos_rework_complete_bridge",
        stop_conditions=(
            "The rework handoff must originate from the rejection loop.",
            "The correction must not expand beyond the approved brief.",
            "Do not invoke a profile in Slice 1; return the deterministic manifest only.",
        ),
    )

def _orchestrator_rework_manifest(po: ProductionOrder) -> DispatchManifest:
    return _make_manifest(
        po,
        target_profile="orchestrator_os",
        target_child_index=2,
        task_type="orchestrator_rework",
        required_input_packet="auditos_rejection_packet",
        expected_result_packet="devos_rework_handoff_packet",
        bridge_function="run_orchestrator_rework_bridge",
        stop_conditions=(
            "AuditOS rejection provenance must be present on the originating child card.",
            "The rework route must freeze the DevOS handoff before any profile is resumed.",
            "Do not invoke a profile in Slice 1; return the deterministic manifest only.",
        ),
    )


def _architect_reconcile_manifest(po: ProductionOrder, state: str) -> DispatchManifest:
    return _make_manifest(
        po,
        target_profile="architect_os",
        target_child_index=4,
        task_type="architect_reconcile",
        required_input_packet="architect_reconcile_handoff_packet",
        expected_result_packet="architect_reconcile_packet",
        bridge_function="run_architect_reconcile_bridge",
        stop_conditions=(
            f"Current state {state!r} must still point at the frozen ArchitectOS reconcile handoff.",
            "Do not widen scope or rewrite approved implementation intent.",
            "Do not invoke a profile in Slice 1; return the deterministic manifest only.",
        ),
    )


def _default_final_review_manifest(po: ProductionOrder, state: str) -> DispatchManifest:
    return _make_manifest(
        po,
        target_profile="default",
        target_child_index=5,
        task_type="default_final_review",
        required_input_packet="default_final_review_handoff_packet",
        expected_result_packet="default_final_review_packet",
        bridge_function="run_default_final_review_bridge",
        stop_conditions=(
            f"Current state {state!r} must still point at the frozen final-review handoff.",
            "Default Hermes must not approve or reject from free-text alone.",
            "Do not invoke a profile in Slice 1; return the deterministic manifest only.",
        ),
    )


def _default_rejection_triage_manifest(po: ProductionOrder) -> DispatchManifest:
    return _make_manifest(
        po,
        target_profile="orchestrator_os",
        target_child_index=0,
        task_type="default_rejection_triage",
        required_input_packet="default_rejection_packet",
        expected_result_packet="default_rejection_handoff_packet",
        bridge_function="run_orchestrator_default_rejection_triage_bridge",
        stop_conditions=(
            "The final-review rejection packet must be present and validated.",
            "This route is only valid after DEFAULT_FINAL_REVIEW rejection.",
            "Do not invoke a profile in Slice 1; return the deterministic manifest only.",
        ),
    )


def _supported_dispatch_states_text() -> str:
    return ", ".join(
        [
            "PRODUCTION_ORDER_CREATED",
            "ORCHESTRATOR_TRIAGE",
            "ARCHITECT_SPEC",
            "ARCHITECT_READY_FOR_DEV",
            "DEV_IMPLEMENTING",
            "DEV_COMPLETE",
            "AUDIT_REVIEW",
            "AUDIT_REJECTED",
            "AUDIT_PASSED",
            "ARCHITECT_RECONCILE",
            "ARCHITECT_ACCEPTED",
            "DEFAULT_FINAL_REVIEW",
            "DEFAULT_REJECTED",
            "DEV_REWORK",
        ]
    )

