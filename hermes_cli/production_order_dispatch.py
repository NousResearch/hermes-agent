"""Deterministic dispatch manifest lookup for Hermes production orders.

Slice 1 only: resolve the next profile task from a reconstructed production
order without invoking any profile, mutating workflow state, or writing back
to Kanban.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import sqlite3
from typing import Any, Iterable

from .production_order_db import (
    ARCHITECT_HANDOFF_TEMPLATE,
    CHILD_CARD_DEFS,
    ProductionOrder,
    REQUIRED_ARCHITECT_RECONCILE_PACKET_FIELDS,
    REQUIRED_ARCHITECT_SPEC_PACKET_FIELDS,
    REQUIRED_AUDITOS_REVIEW_PACKET_FIELDS,
    REQUIRED_DEFAULT_FINAL_REVIEW_PACKET_FIELDS,
    REQUIRED_DEVOS_BUILD_PACKET_FIELDS,
    STATE_OWNERS,
    StageEntry,
    WORKFLOW_SPEC_SOURCE,
    _parse_source_brief,
    get_brief_value,
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


@dataclass(frozen=True)
class ProfileTaskEnvelope:
    production_order_id: str
    parent_kanban_card_id: str
    child_kanban_card_id: str
    target_profile: str
    source_state: str
    expected_next_state: str
    objective: str
    source_truth: tuple[str, ...]
    frozen_brief: str
    input_packet: dict[str, Any]
    expected_output_packet: dict[str, Any]
    acceptance_criteria: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    approval_boundaries: tuple[str, ...]
    allowed_files_or_scope: str | None = None
    repo_or_workspace: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable copy of the envelope."""
        data = asdict(self)
        data["source_truth"] = list(self.source_truth)
        data["acceptance_criteria"] = list(self.acceptance_criteria)
        data["stop_conditions"] = list(self.stop_conditions)
        data["approval_boundaries"] = list(self.approval_boundaries)
        return data


@dataclass(frozen=True)
class ManualFallbackHandoff:
    production_order_id: str
    source_state: str
    expected_next_state: str
    target_profile: str
    target_child_card_id: str
    source_truth: tuple[str, ...]
    required_input_packet: dict[str, Any]
    expected_result_packet: dict[str, Any]
    stop_conditions: tuple[str, ...]
    approval_boundaries: tuple[str, ...]
    bridge_function: str
    result_return_action: str
    copy_paste_prompt: str
    repo_or_workspace: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable copy of the manual fallback handoff."""
        data = asdict(self)
        data["source_truth"] = list(self.source_truth)
        data["stop_conditions"] = list(self.stop_conditions)
        data["approval_boundaries"] = list(self.approval_boundaries)
        return data


_ENVELOPE_RESULT_FIELD_MAP: dict[str, tuple[str, ...]] = {
    "architect_handoff_packet": tuple(
        sorted({"production_order_id", *ARCHITECT_HANDOFF_TEMPLATE.keys()})
    ),
    "architect_spec_packet": tuple(sorted(REQUIRED_ARCHITECT_SPEC_PACKET_FIELDS)),
    "devos_build_packet": tuple(sorted(REQUIRED_DEVOS_BUILD_PACKET_FIELDS)),
    "auditos_review_packet": tuple(sorted(REQUIRED_AUDITOS_REVIEW_PACKET_FIELDS)),
    "architect_reconcile_packet": tuple(sorted(REQUIRED_ARCHITECT_RECONCILE_PACKET_FIELDS)),
    "default_final_review_packet": tuple(sorted(REQUIRED_DEFAULT_FINAL_REVIEW_PACKET_FIELDS)),
}


_MANUAL_FALLBACK_RESULT_ACTIONS: dict[str, tuple[str, str]] = {
    "run_architect_spec_bridge": ("architect_packet", "architect_spec_packet"),
    "run_devos_complete_bridge": ("devos_packet", "devos_build_packet"),
    "run_architect_reconcile_bridge": (
        "reconcile_packet",
        "architect_reconcile_packet",
    ),
}


_SUPPORTED_ENVELOPE_ROUTES: dict[tuple[str, str], str] = {
    ("PRODUCTION_ORDER_CREATED", "orchestrator_triage"): "ORCHESTRATOR_TRIAGE",
    ("ORCHESTRATOR_TRIAGE", "orchestrator_triage"): "ARCHITECT_SPEC",
    ("ARCHITECT_SPEC", "architect_spec"): "ARCHITECT_READY_FOR_DEV",
    ("ARCHITECT_READY_FOR_DEV", "dev_build"): "DEV_COMPLETE",
    ("DEV_IMPLEMENTING", "dev_build"): "DEV_COMPLETE",
    ("AUDIT_PASSED", "architect_reconcile"): "ARCHITECT_ACCEPTED",
    ("ARCHITECT_RECONCILE", "architect_reconcile"): "ARCHITECT_ACCEPTED",
}


def build_dispatch_manifest(
    conn: sqlite3.Connection,
    production_order_id: str,
) -> DispatchManifest:
    """Load a production order from SQLite and build its dispatch manifest."""
    po = _load_production_order(conn, production_order_id)
    _validate_child_graph(conn, po)
    return dispatch_manifest_for_order(po)


def build_profile_task_envelope(
    conn: sqlite3.Connection,
    production_order_id: str,
) -> ProfileTaskEnvelope:
    """Load a production order from SQLite and build a task envelope."""
    po = _load_production_order(conn, production_order_id)
    _validate_child_graph(conn, po)
    manifest = dispatch_manifest_for_order(po)
    return profile_task_envelope_for_order(po, manifest)


def build_manual_fallback_handoff(
    conn: sqlite3.Connection,
    production_order_id: str,
) -> ManualFallbackHandoff:
    """Load a production order from SQLite and build its manual fallback handoff."""
    envelope = build_profile_task_envelope(conn, production_order_id)
    return manual_fallback_handoff_for_envelope(envelope)


def manual_fallback_handoff_for_envelope(
    envelope: ProfileTaskEnvelope,
) -> ManualFallbackHandoff:
    """Build a deterministic manual fallback handoff from a task envelope."""
    _validate_profile_task_envelope(envelope)
    _validate_manual_fallback_envelope(envelope)

    bridge_function = str(envelope.expected_output_packet["bridge_function"]).strip()
    result_return_action = _manual_fallback_result_return_action(envelope, bridge_function)
    handoff = ManualFallbackHandoff(
        production_order_id=envelope.production_order_id,
        source_state=envelope.source_state,
        expected_next_state=envelope.expected_next_state,
        target_profile=envelope.target_profile,
        target_child_card_id=envelope.child_kanban_card_id,
        source_truth=envelope.source_truth,
        required_input_packet=dict(envelope.input_packet),
        expected_result_packet=dict(envelope.expected_output_packet),
        stop_conditions=envelope.stop_conditions,
        approval_boundaries=envelope.approval_boundaries,
        bridge_function=bridge_function,
        result_return_action=result_return_action,
        copy_paste_prompt=_manual_fallback_prompt(envelope, result_return_action),
        repo_or_workspace=envelope.repo_or_workspace,
    )
    _validate_manual_fallback_handoff(handoff)
    return handoff


def profile_task_envelope_for_order(
    po: ProductionOrder,
    manifest: DispatchManifest,
) -> ProfileTaskEnvelope:
    """Build a deterministic profile task envelope from a production order."""
    _validate_reconstructed_order(po)
    if manifest.production_order_id != po.production_order_id:
        raise DispatchManifestError(
            "Dispatch manifest production_order_id does not match the reconstructed production order"
        )
    if manifest.current_state != po.current_state:
        raise DispatchManifestError(
            "Dispatch manifest current_state does not match the reconstructed production order"
        )

    expected_next_state = _expected_next_state_for_manifest(manifest)
    brief = _parse_source_brief(po.source_brief)
    objective = str(get_brief_value(brief, "objective", po.title)).strip()
    acceptance_criteria = _normalize_text_list(
        get_brief_value(brief, "acceptance criteria", ()),
        fallback=("Acceptance criteria are frozen in the production-order brief.",),
    )
    stop_conditions = _merge_text_lists(
        _normalize_text_list(get_brief_value(brief, "stop conditions", ())),
        manifest.stop_conditions,
    )
    approval_boundaries = _normalize_text_list(
        po.approval_boundaries or get_brief_value(brief, "approval boundaries", ()),
        fallback=(
            "Pause before publishing, spending, destructive changes, permission widening, or scope expansion.",
        ),
    )
    repo_or_workspace = str(
        po.repo_or_workspace or get_brief_value(brief, "target repo or workspace", "")
    ).strip() or None
    allowed_files_or_scope = str(get_brief_value(brief, "scope", "")).strip() or None
    child_card_meta = _child_card_metadata(po, manifest.target_child_card_id)

    envelope = ProfileTaskEnvelope(
        production_order_id=po.production_order_id,
        parent_kanban_card_id=po.parent_kanban_card_id,
        child_kanban_card_id=manifest.target_child_card_id,
        target_profile=manifest.target_profile,
        source_state=po.current_state,
        expected_next_state=expected_next_state,
        objective=objective,
        source_truth=(WORKFLOW_SPEC_SOURCE,),
        frozen_brief=po.source_brief,
        input_packet={
            "packet_type": manifest.required_input_packet,
            "production_order_id": po.production_order_id,
            "source_state": po.current_state,
            "target_profile": manifest.target_profile,
            "parent_kanban_card_id": po.parent_kanban_card_id,
            "child_kanban_card_id": manifest.target_child_card_id,
            "child_card_title": child_card_meta["title"],
            "repo_or_workspace": repo_or_workspace,
            "brief_context": {
                "objective": objective,
                "scope": allowed_files_or_scope,
                "out_of_scope": _brief_text_or_none(brief, "out of scope"),
                "constraints": _brief_text_or_none(brief, "constraints"),
                "expected_output": _brief_text_or_none(brief, "expected output"),
            },
        },
        expected_output_packet={
            "packet_type": manifest.expected_result_packet,
            "production_order_id": po.production_order_id,
            "owner_profile": manifest.target_profile,
            "source_state": po.current_state,
            "expected_next_state": expected_next_state,
            "bridge_function": manifest.bridge_function,
            "required_fields": list(
                _ENVELOPE_RESULT_FIELD_MAP.get(manifest.expected_result_packet, ())
            ),
        },
        acceptance_criteria=acceptance_criteria,
        stop_conditions=stop_conditions,
        approval_boundaries=approval_boundaries,
        allowed_files_or_scope=allowed_files_or_scope,
        repo_or_workspace=repo_or_workspace,
    )
    _validate_profile_task_envelope(envelope)
    return envelope


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


def _expected_next_state_for_manifest(manifest: DispatchManifest) -> str:
    expected_next_state = _SUPPORTED_ENVELOPE_ROUTES.get(
        (manifest.current_state, manifest.task_type)
    )
    if expected_next_state is None:
        raise DispatchManifestError(
            f"State {manifest.current_state!r} with task type {manifest.task_type!r} "
            "does not yet support envelope generation"
        )
    return expected_next_state


def _normalize_text_list(
    value: Any,
    *,
    fallback: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if value is None:
        return fallback
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else fallback
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        normalized: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return tuple(normalized) if normalized else fallback
    text = str(value).strip()
    return (text,) if text else fallback


def _merge_text_lists(*groups: Iterable[str]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                merged.append(text)
    return tuple(merged)


def _brief_text_or_none(brief: dict[str, Any], canonical_key: str) -> str | None:
    value = get_brief_value(brief, canonical_key, "")
    text = str(value).strip()
    return text or None


def _child_card_metadata(po: ProductionOrder, child_card_id: str) -> dict[str, Any]:
    for index, (order, title, owner_profile, default_status) in enumerate(CHILD_CARD_DEFS):
        if po.child_kanban_card_ids[index] == child_card_id:
            return {
                "order": order,
                "title": title,
                "owner_profile": owner_profile,
                "default_status": default_status,
            }
    raise DispatchManifestError(
        f"Production order {po.production_order_id!r} target child card {child_card_id!r} "
        "is not part of the reconstructed six-card graph"
    )


def _validate_profile_task_envelope(envelope: ProfileTaskEnvelope) -> None:
    required_text_fields = {
        "production_order_id": envelope.production_order_id,
        "parent_kanban_card_id": envelope.parent_kanban_card_id,
        "child_kanban_card_id": envelope.child_kanban_card_id,
        "target_profile": envelope.target_profile,
        "source_state": envelope.source_state,
        "expected_next_state": envelope.expected_next_state,
        "objective": envelope.objective,
        "frozen_brief": envelope.frozen_brief,
    }
    for field_name, value in required_text_fields.items():
        if not str(value).strip():
            raise DispatchManifestError(f"Profile task envelope field {field_name!r} is required")
    if not envelope.source_truth:
        raise DispatchManifestError("Profile task envelope field 'source_truth' is required")
    if WORKFLOW_SPEC_SOURCE not in envelope.source_truth:
        raise DispatchManifestError(
            "Profile task envelope must include the workflow spec in source_truth"
        )
    if envelope.input_packet.get("packet_type") is None:
        raise DispatchManifestError("Profile task envelope input_packet must include packet_type")
    if envelope.expected_output_packet.get("packet_type") is None:
        raise DispatchManifestError(
            "Profile task envelope expected_output_packet must include packet_type"
        )
    if not envelope.acceptance_criteria:
        raise DispatchManifestError(
            "Profile task envelope field 'acceptance_criteria' is required"
        )
    if not envelope.stop_conditions:
        raise DispatchManifestError("Profile task envelope field 'stop_conditions' is required")
    if not envelope.approval_boundaries:
        raise DispatchManifestError(
            "Profile task envelope field 'approval_boundaries' is required"
        )


def _validate_manual_fallback_envelope(envelope: ProfileTaskEnvelope) -> None:
    packet_type = str(envelope.expected_output_packet.get("packet_type", "")).strip()
    if not packet_type:
        raise DispatchManifestError(
            "Manual fallback requires expected_output_packet.packet_type"
        )
    required_fields = envelope.expected_output_packet.get("required_fields")
    if not isinstance(required_fields, list) or not required_fields:
        raise DispatchManifestError(
            "Manual fallback requires expected_output_packet.required_fields"
        )
    bridge_function = str(envelope.expected_output_packet.get("bridge_function", "")).strip()
    if not bridge_function:
        raise DispatchManifestError(
            "Manual fallback requires expected_output_packet.bridge_function"
        )
    if bridge_function not in _MANUAL_FALLBACK_RESULT_ACTIONS:
        raise DispatchManifestError(
            f"Manual fallback does not yet support result-return action for bridge {bridge_function!r}"
        )
    expected_packet_type = _MANUAL_FALLBACK_RESULT_ACTIONS[bridge_function][1]
    if packet_type != expected_packet_type:
        raise DispatchManifestError(
            "Manual fallback expected_output_packet packet_type does not match the bridge action"
        )


def _manual_fallback_result_return_action(
    envelope: ProfileTaskEnvelope,
    bridge_function: str,
) -> str:
    packet_arg_name, _ = _MANUAL_FALLBACK_RESULT_ACTIONS[bridge_function]
    return (
        f"Call {bridge_function}(conn, production_order_id={envelope.production_order_id!r}, "
        f"{packet_arg_name}=<returned_result_packet>)"
    )


def _manual_fallback_prompt(
    envelope: ProfileTaskEnvelope,
    result_return_action: str,
) -> str:
    brief_context = envelope.input_packet.get("brief_context", {})
    scope = envelope.allowed_files_or_scope or brief_context.get("scope") or "From the frozen brief."
    out_of_scope = brief_context.get("out_of_scope") or "Not specified in the frozen brief."
    constraints = brief_context.get("constraints") or "None specified beyond the frozen brief."
    expected_output = brief_context.get("expected_output") or "Return the required result packet only."
    expected_result_packet = _manual_fallback_expected_result_packet_example(envelope)
    sections = [
        f"Target profile: {envelope.target_profile}",
        f"Objective: {envelope.objective}",
        f"Production order ID: {envelope.production_order_id}",
        f"Source state: {envelope.source_state}",
        f"Expected next state: {envelope.expected_next_state}",
        f"Target child card ID: {envelope.child_kanban_card_id}",
    ]
    if envelope.repo_or_workspace:
        sections.append(f"Repo or workspace: {envelope.repo_or_workspace}")
    sections.extend(
        [
            "Frozen brief context:",
            f"- Scope: {scope}",
            f"- Out of scope: {out_of_scope}",
            f"- Constraints: {constraints}",
            f"- Expected output: {expected_output}",
            "Source truth:",
            *[f"- {item}" for item in envelope.source_truth],
            "Required input packet:",
            json.dumps(envelope.input_packet, indent=2, sort_keys=True),
            "Expected result packet requirements:",
            json.dumps(envelope.expected_output_packet, indent=2, sort_keys=True),
            "Return exactly one structured result packet JSON object matching this shape:",
            json.dumps(expected_result_packet, indent=2, sort_keys=True),
            "Acceptance criteria:",
            *[f"- {item}" for item in envelope.acceptance_criteria],
            "Stop conditions:",
            *[f"- {item}" for item in envelope.stop_conditions],
            "Approval boundaries:",
            *[f"- {item}" for item in envelope.approval_boundaries],
            "Guardrails:",
            "- Do not execute external, destructive, publishing, spending, permission-widening, or secret-requesting actions without explicit approval.",
            "- Do not change Hermes production-workflow state directly.",
            "- Do not invoke another profile.",
            "- If required source truth is missing or contracts conflict, stop and return a blocked result packet instead of guessing.",
            "After result return:",
            f"- {result_return_action}",
        ]
    )
    return "\n".join(str(section) for section in sections)


def _manual_fallback_expected_result_packet_example(
    envelope: ProfileTaskEnvelope,
) -> dict[str, Any]:
    required_fields = [
        str(field).strip()
        for field in envelope.expected_output_packet.get("required_fields", [])
        if str(field).strip()
    ]
    packet: dict[str, Any] = {"packet_type": envelope.expected_output_packet["packet_type"]}
    for field_name in required_fields:
        if field_name == "production_order_id":
            packet[field_name] = envelope.production_order_id
        elif field_name == "owner_profile":
            packet[field_name] = envelope.target_profile
        elif field_name == "source_state":
            packet[field_name] = envelope.source_state
        elif field_name == "next_state":
            packet[field_name] = envelope.expected_next_state
        elif field_name == "stage" and envelope.expected_output_packet["packet_type"] == "architect_spec_packet":
            packet[field_name] = "architect_spec"
        else:
            packet[field_name] = f"<{field_name}>"
    return packet


def _validate_manual_fallback_handoff(handoff: ManualFallbackHandoff) -> None:
    required_text_fields = {
        "production_order_id": handoff.production_order_id,
        "source_state": handoff.source_state,
        "expected_next_state": handoff.expected_next_state,
        "target_profile": handoff.target_profile,
        "target_child_card_id": handoff.target_child_card_id,
        "bridge_function": handoff.bridge_function,
        "result_return_action": handoff.result_return_action,
        "copy_paste_prompt": handoff.copy_paste_prompt,
    }
    for field_name, value in required_text_fields.items():
        if not str(value).strip():
            raise DispatchManifestError(f"Manual fallback handoff field {field_name!r} is required")
    if not handoff.source_truth:
        raise DispatchManifestError("Manual fallback handoff field 'source_truth' is required")
    if not handoff.stop_conditions:
        raise DispatchManifestError("Manual fallback handoff field 'stop_conditions' is required")
    if not handoff.approval_boundaries:
        raise DispatchManifestError(
            "Manual fallback handoff field 'approval_boundaries' is required"
        )
    if handoff.required_input_packet.get("packet_type") is None:
        raise DispatchManifestError(
            "Manual fallback handoff required_input_packet must include packet_type"
        )
    if handoff.expected_result_packet.get("packet_type") is None:
        raise DispatchManifestError(
            "Manual fallback handoff expected_result_packet must include packet_type"
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

