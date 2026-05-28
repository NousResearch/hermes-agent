"""Deterministic dispatch manifest lookup for Hermes production orders.

Slice 1 only: resolve the next profile task from a reconstructed production
order without invoking any profile, mutating workflow state, or writing back
to Kanban.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
import sqlite3
from typing import Any, Iterable

from .production_order_db import (
    ARCHITECT_HANDOFF_TEMPLATE,
    CHILD_CARD_DEFS,
    DEFAULT_REJECTION_HANDOFF_TEMPLATE,
    ORCHESTRATOR_CLASSIFICATION_HANDOFF_TEMPLATE,
    ProductionOrder,
    create_auditos_handoff,
    create_architect_reconcile_handoff,
    create_architect_rework_handoff,
    create_default_final_review_handoff,
    create_devos_rework_handoff,
    create_devos_handoff,
    REQUIRED_ARCHITECT_RECONCILE_PACKET_FIELDS,
    REQUIRED_ARCHITECT_SPEC_PACKET_FIELDS,
    REQUIRED_AUDITOS_REVIEW_PACKET_FIELDS,
    REQUIRED_DEFAULT_FINAL_REVIEW_PACKET_FIELDS,
    REQUIRED_ORCHESTRATOR_CLASSIFICATION_PACKET_FIELDS,
    REQUIRED_DEVOS_BUILD_PACKET_FIELDS,
    STATE_OWNERS,
    StageEntry,
    WORKFLOW_SPEC_SOURCE,
    _parse_source_brief,
    freeze_handoff_on_card,
    freeze_result_on_card,
    get_brief_value,
    list_production_orders,
    log_workflow_event,
    transition_state,
    validate_auditos_review_packet,
    validate_architect_reconcile_packet,
    validate_architect_spec_packet,
    validate_default_final_review_packet,
    validate_devos_build_packet,
)


class DispatchManifestError(ValueError):
    """Raised when a production order cannot be mapped to a dispatch manifest."""


ALLOWED_DISPATCH_EVENT_TYPES: tuple[str, ...] = (
    "dispatch_planned",
    "dispatch_started",
    "dispatch_handoff_created",
    "dispatch_completed",
    "dispatch_failed",
    "dispatch_blocked",
    "packet_validated",
    "packet_rejected",
)


def _validate_dispatch_event_type(event_type: str) -> str:
    normalized = str(event_type).strip()
    if normalized not in ALLOWED_DISPATCH_EVENT_TYPES:
        raise DispatchManifestError(
            f"Unsupported dispatch event type {event_type!r}. Allowed types: "
            f"{', '.join(ALLOWED_DISPATCH_EVENT_TYPES)}"
        )
    return normalized


@dataclass(frozen=True)
class DispatchManifest:
    production_order_id: str
    current_state: str
    current_owner_profile: str
    dispatch_attempt: int
    idempotency_key: str
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
    dispatch_attempt: int
    idempotency_key: str
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
    dispatch_attempt: int
    idempotency_key: str
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


@dataclass(frozen=True)
class ResultPacketIngestion:
    accepted: bool
    production_order_id: str
    dispatch_attempt: int
    idempotency_key: str
    source_state: str
    owner_profile: str
    target_profile: str
    child_kanban_card_id: str
    packet_id: str | None
    bridge_function: str
    runtime_action: str
    error: str | None
    next_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DispatchExecutionResult:
    executed: bool
    fallback_required: bool
    production_order_id: str
    dispatch_attempt: int
    idempotency_key: str
    source_state: str
    target_profile: str
    target_child_card_id: str
    task_type: str
    result_packet: dict[str, Any] | None
    artifact_reference: str | None
    manual_fallback: dict[str, Any] | None
    error: str | None
    next_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReworkRouteDecision:
    production_order_id: str
    source_state: str
    route_decision: str
    target_profile: str
    target_child_card_id: str | None
    task_type: str
    explanation: str
    stop_condition: str | None
    idempotency_key: str
    applied: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ENVELOPE_RESULT_FIELD_MAP: dict[str, tuple[str, ...]] = {
    "architect_handoff_packet": tuple(
        sorted(
            {
                "production_order_id",
                "context",
                "scope",
                "out_of_scope",
                "inputs",
                *ARCHITECT_HANDOFF_TEMPLATE.keys(),
            }
        )
    ),
    "default_rejection_handoff_packet": tuple(
        sorted(
            {
                "production_order_id",
                "objective",
                "scope",
                "out_of_scope",
                "inputs",
                "acceptance_criteria",
                "stop_conditions",
                *DEFAULT_REJECTION_HANDOFF_TEMPLATE.keys(),
            }
        )
    ),
    "devos_rework_handoff_packet": tuple(
        sorted(
            {
                "production_order_id",
                "objective",
                "scope",
                "out_of_scope",
                "inputs",
                "acceptance_criteria",
                "stop_conditions",
                "approval_boundaries",
                "artifact_references",
                "default_rejection_reason",
                "correction_request",
                "classification",
                "route_target",
                "route_reason",
                "next_handoff_target",
                *ARCHITECT_HANDOFF_TEMPLATE.keys(),
            }
        )
    ),
    "architect_spec_packet": tuple(sorted(REQUIRED_ARCHITECT_SPEC_PACKET_FIELDS)),
    "devos_build_packet": tuple(sorted(REQUIRED_DEVOS_BUILD_PACKET_FIELDS)),
    "auditos_review_packet": tuple(sorted(REQUIRED_AUDITOS_REVIEW_PACKET_FIELDS)),
    "orchestrator_classification_packet": tuple(
        sorted(REQUIRED_ORCHESTRATOR_CLASSIFICATION_PACKET_FIELDS)
    ),
    "architect_reconcile_packet": tuple(sorted(REQUIRED_ARCHITECT_RECONCILE_PACKET_FIELDS)),
    "default_final_review_packet": tuple(sorted(REQUIRED_DEFAULT_FINAL_REVIEW_PACKET_FIELDS)),
}


_MANUAL_FALLBACK_RESULT_ACTIONS: dict[str, tuple[str, str]] = {
    "run_orchestrator_triage_bridge": ("handoff_packet", "architect_handoff_packet"),
    "run_orchestrator_classification_bridge": (
        "classification_packet",
        "orchestrator_classification_packet",
    ),
    "run_orchestrator_rework_bridge": (
        "rejection_packet",
        "devos_rework_handoff_packet",
    ),
    "run_orchestrator_default_rejection_triage_bridge": (
        "rejection_packet",
        "default_rejection_handoff_packet",
    ),
    "run_architect_spec_bridge": ("architect_packet", "architect_spec_packet"),
    "run_devos_complete_bridge": ("devos_packet", "devos_build_packet"),
    "run_auditos_review_complete_bridge": ("review_packet", "auditos_review_packet"),
    "run_architect_reconcile_bridge": (
        "reconcile_packet",
        "architect_reconcile_packet",
    ),
    "run_default_final_review_bridge": ("final_packet", "default_final_review_packet"),
}


_SUPPORTED_ENVELOPE_ROUTES: dict[tuple[str, str], str] = {
    ("PRODUCTION_ORDER_CREATED", "orchestrator_triage"): "ORCHESTRATOR_TRIAGE",
    ("ORCHESTRATOR_TRIAGE", "orchestrator_triage"): "ARCHITECT_SPEC",
    (
        "ORCHESTRATOR_TRIAGE",
        "orchestrator_default_rejection_classification",
    ): "ORCHESTRATOR_TRIAGE",
    ("ARCHITECT_SPEC", "architect_spec"): "ARCHITECT_READY_FOR_DEV",
    ("ARCHITECT_READY_FOR_DEV", "dev_build"): "DEV_COMPLETE",
    ("DEV_IMPLEMENTING", "dev_build"): "DEV_COMPLETE",
    ("DEV_COMPLETE", "audit_review"): "AUDIT_PASSED",
    ("AUDIT_REVIEW", "audit_review"): "AUDIT_PASSED",
    ("AUDIT_PASSED", "architect_reconcile"): "ARCHITECT_ACCEPTED",
    ("ARCHITECT_RECONCILE", "architect_reconcile"): "ARCHITECT_ACCEPTED",
    ("ARCHITECT_ACCEPTED", "default_final_review"): "DONE",
    ("DEFAULT_FINAL_REVIEW", "default_final_review"): "DONE",
}


_RESULT_PACKET_SECTION_RE = re.compile(
    r"--- RESULT PACKET ---\n(.*?)\n--- END RESULT PACKET ---",
    re.DOTALL,
)

_EVENT_HASH_RE = re.compile(r"\bpayload_hash=([0-9a-f]{64})\b")


def _event_payload_hash(result: str | None) -> str | None:
    if not result:
        return None
    match = _EVENT_HASH_RE.search(str(result))
    if not match:
        return None
    return match.group(1)


def _dispatch_attempt_for_state(po: ProductionOrder) -> int:
    attempts = sum(1 for entry in po.stage_history if entry.to_state == po.current_state)
    return max(1, attempts)



def _dispatch_idempotency_key(
    *,
    production_order_id: str,
    source_state: str,
    target_profile: str,
    target_child_card_id: str,
    task_type: str,
    dispatch_attempt: int,
) -> str:
    return (
        f"dispatch:{production_order_id}:{source_state}:{target_profile}:"
        f"{target_child_card_id}:{task_type}:attempt-{dispatch_attempt}"
    )



def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)



def _payload_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()



def _dispatch_slot_key(
    *,
    production_order_id: str,
    source_state: str,
    dispatch_attempt: int,
) -> str:
    return f"slot:{production_order_id}:{source_state}:attempt-{dispatch_attempt}"


def _load_task_body(conn: sqlite3.Connection, card_id: str) -> str:
    row = conn.execute("SELECT body FROM tasks WHERE id = ?", (card_id,)).fetchone()
    if row is None:
        raise DispatchManifestError(f"Kanban card {card_id!r} not found")
    return str(row[0] or "")


def _frozen_result_packets(conn: sqlite3.Connection, card_id: str) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    body = _load_task_body(conn, card_id)
    for match in _RESULT_PACKET_SECTION_RE.finditer(body):
        raw_packet = match.group(1).strip()
        if not raw_packet:
            continue
        try:
            packet = json.loads(raw_packet)
        except json.JSONDecodeError:
            continue
        if isinstance(packet, dict):
            packets.append(packet)
    return packets


def _find_dispatch_event(
    conn: sqlite3.Connection,
    *,
    production_order_id: str,
    event_type: str,
    from_state: str,
    target_profile: str,
    kanban_card_id: str,
    packet_id: str | None = None,
) -> dict[str, Any] | None:
    for event in list_dispatch_events(conn, production_order_id):
        if event["event_type"] != event_type:
            continue
        if event["from_state"] != from_state:
            continue
        if event["target_profile"] != target_profile:
            continue
        if event["kanban_card_id"] != kanban_card_id:
            continue
        if packet_id is not None and event["packet_id"] != packet_id:
            continue
        return event
    return None


def _ensure_dispatch_planned_event(
    conn: sqlite3.Connection,
    manifest: DispatchManifest,
    *,
    envelope: ProfileTaskEnvelope | None = None,
) -> dict[str, Any]:
    payload = envelope.to_dict() if envelope is not None else manifest.to_dict()
    payload_hash = _payload_hash(payload)
    manifest_hash = _payload_hash(manifest.to_dict())
    existing = _find_dispatch_event(
        conn,
        production_order_id=manifest.production_order_id,
        event_type="dispatch_planned",
        from_state=manifest.current_state,
        target_profile=manifest.target_profile,
        kanban_card_id=manifest.target_child_card_id,
        packet_id=manifest.idempotency_key,
    )
    if existing is not None:
        existing_hash = _event_payload_hash(existing.get("result"))
        if existing_hash and existing_hash not in {payload_hash, manifest_hash}:
            raise DispatchManifestError(
                "Conflicting duplicate dispatch_planned attempt for "
                f"{manifest.idempotency_key!r}: payload hash mismatch"
            )
        return existing

    to_state = _SUPPORTED_ENVELOPE_ROUTES.get((manifest.current_state, manifest.task_type))
    log_dispatch_event(
        conn,
        production_order_id=manifest.production_order_id,
        event_type="dispatch_planned",
        from_state=manifest.current_state,
        to_state=to_state,
        owner_profile=manifest.current_owner_profile,
        target_profile=manifest.target_profile,
        kanban_card_id=manifest.target_child_card_id,
        packet_id=manifest.idempotency_key,
        result=f"dispatch_planned payload_hash={manifest_hash}",
        next_action="manual_dispatch_ready",
    )
    created = _find_dispatch_event(
        conn,
        production_order_id=manifest.production_order_id,
        event_type="dispatch_planned",
        from_state=manifest.current_state,
        target_profile=manifest.target_profile,
        kanban_card_id=manifest.target_child_card_id,
        packet_id=manifest.idempotency_key,
    )
    assert created is not None
    return created


def _ensure_dispatch_handoff_event(
    conn: sqlite3.Connection,
    handoff: ManualFallbackHandoff,
) -> dict[str, Any]:
    payload_hash = _payload_hash(handoff.to_dict())
    existing = _find_dispatch_event(
        conn,
        production_order_id=handoff.production_order_id,
        event_type="dispatch_handoff_created",
        from_state=handoff.source_state,
        target_profile=handoff.target_profile,
        kanban_card_id=handoff.target_child_card_id,
        packet_id=handoff.idempotency_key,
    )
    if existing is not None:
        existing_hash = _event_payload_hash(existing.get("result"))
        if existing_hash and existing_hash != payload_hash:
            raise DispatchManifestError(
                "Conflicting duplicate dispatch_handoff_created attempt for "
                f"{handoff.idempotency_key!r}: payload hash mismatch"
            )
        return existing

    log_dispatch_event(
        conn,
        production_order_id=handoff.production_order_id,
        event_type="dispatch_handoff_created",
        from_state=handoff.source_state,
        to_state=handoff.expected_next_state,
        owner_profile=handoff.target_profile,
        target_profile=handoff.target_profile,
        kanban_card_id=handoff.target_child_card_id,
        packet_id=handoff.idempotency_key,
        result=f"manual_fallback_created payload_hash={payload_hash}",
        next_action="copy_prompt_to_profile",
    )
    created = _find_dispatch_event(
        conn,
        production_order_id=handoff.production_order_id,
        event_type="dispatch_handoff_created",
        from_state=handoff.source_state,
        target_profile=handoff.target_profile,
        kanban_card_id=handoff.target_child_card_id,
        packet_id=handoff.idempotency_key,
    )
    assert created is not None
    return created


def _ensure_dispatch_started_event(
    conn: sqlite3.Connection,
    envelope: ProfileTaskEnvelope,
    *,
    result: str,
    next_action: str,
) -> dict[str, Any]:
    payload_hash = _payload_hash(
        {
            "idempotency_key": envelope.idempotency_key,
            "result": result,
            "next_action": next_action,
        }
    )
    existing = _find_dispatch_event(
        conn,
        production_order_id=envelope.production_order_id,
        event_type="dispatch_started",
        from_state=envelope.source_state,
        target_profile=envelope.target_profile,
        kanban_card_id=envelope.child_kanban_card_id,
        packet_id=envelope.idempotency_key,
    )
    if existing is not None:
        existing_hash = _event_payload_hash(existing.get("result"))
        if existing_hash and existing_hash != payload_hash:
            raise DispatchManifestError(
                "Conflicting duplicate dispatch_started attempt for "
                f"{envelope.idempotency_key!r}: payload hash mismatch"
            )
        return existing

    log_dispatch_event(
        conn,
        production_order_id=envelope.production_order_id,
        event_type="dispatch_started",
        from_state=envelope.source_state,
        to_state=envelope.expected_next_state,
        owner_profile=envelope.target_profile,
        target_profile=envelope.target_profile,
        kanban_card_id=envelope.child_kanban_card_id,
        packet_id=envelope.idempotency_key,
        result=f"{result} payload_hash={payload_hash}",
        next_action=next_action,
    )
    created = _find_dispatch_event(
        conn,
        production_order_id=envelope.production_order_id,
        event_type="dispatch_started",
        from_state=envelope.source_state,
        target_profile=envelope.target_profile,
        kanban_card_id=envelope.child_kanban_card_id,
        packet_id=envelope.idempotency_key,
    )
    assert created is not None
    return created


def _accepted_result_conflict(
    conn: sqlite3.Connection,
    envelope: ProfileTaskEnvelope,
    validated_packet: dict[str, Any],
) -> dict[str, Any] | None:
    packet_hash = _payload_hash(validated_packet)
    packet_id = _packet_id(validated_packet)
    for frozen in _frozen_result_packets(conn, envelope.child_kanban_card_id):
        if frozen.get("production_order_id") != envelope.production_order_id:
            continue
        if frozen.get("owner_profile") != envelope.target_profile:
            continue
        if frozen.get("source_state") != envelope.source_state:
            continue
        if _payload_hash(frozen) == packet_hash:
            return frozen
        frozen_packet_id = _packet_id(frozen)
        if packet_id and frozen_packet_id and frozen_packet_id != packet_id:
            raise DispatchManifestError(
                "Conflicting accepted result packet already exists for dispatch route "
                f"{envelope.idempotency_key!r}"
            )
        raise DispatchManifestError(
            "Conflicting accepted result packet payload already exists for dispatch route "
            f"{envelope.idempotency_key!r}"
        )
    return None


def build_dispatch_manifest(
    conn: sqlite3.Connection,
    production_order_id: str,
) -> DispatchManifest:
    """Load a production order from SQLite and build its dispatch manifest."""
    po = _load_production_order(conn, production_order_id)
    _validate_child_graph(conn, po)
    manifest = dispatch_manifest_for_order(po)
    _ensure_dispatch_planned_event(conn, manifest)
    return manifest


def build_profile_task_envelope(
    conn: sqlite3.Connection,
    production_order_id: str,
) -> ProfileTaskEnvelope:
    """Load a production order from SQLite and build a task envelope."""
    po = _load_production_order(conn, production_order_id)
    _validate_child_graph(conn, po)
    manifest = dispatch_manifest_for_order(po)
    envelope = profile_task_envelope_for_order(po, manifest)
    _ensure_dispatch_planned_event(conn, manifest, envelope=envelope)
    return envelope


def build_manual_fallback_handoff(
    conn: sqlite3.Connection,
    production_order_id: str,
) -> ManualFallbackHandoff:
    """Load a production order from SQLite and build its manual fallback handoff."""
    envelope = build_profile_task_envelope(conn, production_order_id)
    handoff = manual_fallback_handoff_for_envelope(envelope)
    _ensure_dispatch_handoff_event(conn, handoff)
    return handoff


def execute_profile_dispatch(
    conn: sqlite3.Connection,
    production_order_id: str,
    *,
    source_state: str | None = None,
    target_profile: str | None = None,
    target_child_card_id: str | None = None,
    task_type: str | None = None,
    dispatch_attempt: int | None = None,
) -> dict[str, Any]:
    """Run one bounded dispatch attempt for the active task envelope.

    This entrypoint deliberately does not start the Kanban daemon or spawn a
    background worker. The current production workflow has no synchronous,
    result-returning profile invocation primitive, so the safe executor mode is
    to validate the dispatch identity, log that execution began, and return the
    typed manual fallback handoff for the same envelope.
    """
    envelope = build_profile_task_envelope(conn, production_order_id)
    _assert_executor_identity(
        envelope,
        source_state=source_state,
        target_profile=target_profile,
        target_child_card_id=target_child_card_id,
        task_type=task_type,
        dispatch_attempt=dispatch_attempt,
    )
    handoff = manual_fallback_handoff_for_envelope(envelope)
    _ensure_dispatch_started_event(
        conn,
        envelope,
        result="manual_fallback_required",
        next_action="copy_prompt_to_profile",
    )
    _ensure_dispatch_handoff_event(conn, handoff)
    return DispatchExecutionResult(
        executed=False,
        fallback_required=True,
        production_order_id=production_order_id,
        dispatch_attempt=envelope.dispatch_attempt,
        idempotency_key=envelope.idempotency_key,
        source_state=envelope.source_state,
        target_profile=envelope.target_profile,
        target_child_card_id=envelope.child_kanban_card_id,
        task_type=task_type or _task_type_from_idempotency_key(envelope.idempotency_key),
        result_packet=None,
        artifact_reference=None,
        manual_fallback=handoff.to_dict(),
        error="No safe synchronous profile invocation mechanism is available for production-order dispatch.",
        next_action="manual_fallback_required",
    ).to_dict()


def ingest_profile_result_packet(
    conn: sqlite3.Connection,
    production_order_id: str,
    result_packet: Any,
) -> dict[str, Any]:
    """Validate and freeze a returned profile result packet without advancing state."""
    envelope = build_profile_task_envelope(conn, production_order_id)
    bridge_function, runtime_action = _resolve_ingestion_runtime_action(envelope)
    packet_id = _packet_id(result_packet)

    try:
        validated_packet = validate_profile_result_packet(envelope, result_packet)
    except ValueError as exc:
        error = str(exc)
        log_dispatch_event(
            conn,
            production_order_id=production_order_id,
            event_type="packet_rejected",
            from_state=envelope.source_state,
            to_state=envelope.expected_next_state,
            owner_profile=envelope.target_profile,
            target_profile=envelope.target_profile,
            kanban_card_id=envelope.child_kanban_card_id,
            packet_id=packet_id,
            result="rejected",
            error=error,
            next_action="manual_review_rejected_packet",
        )
        return ResultPacketIngestion(
            accepted=False,
            production_order_id=production_order_id,
            dispatch_attempt=envelope.dispatch_attempt,
            idempotency_key=envelope.idempotency_key,
            source_state=envelope.source_state,
            owner_profile=envelope.target_profile,
            target_profile=envelope.target_profile,
            child_kanban_card_id=envelope.child_kanban_card_id,
            packet_id=packet_id,
            bridge_function=bridge_function,
            runtime_action=runtime_action,
            error=error,
            next_action="manual_review_rejected_packet",
        ).to_dict()

    existing_packet = _accepted_result_conflict(conn, envelope, validated_packet)
    packet_id = _packet_id(validated_packet)
    if existing_packet is None:
        freeze_result_on_card(conn, envelope.child_kanban_card_id, validated_packet)
        log_dispatch_event(
            conn,
            production_order_id=production_order_id,
            event_type="packet_validated",
            from_state=envelope.source_state,
            to_state=envelope.expected_next_state,
            owner_profile=envelope.target_profile,
            target_profile=envelope.target_profile,
            kanban_card_id=envelope.child_kanban_card_id,
            packet_id=packet_id,
            result="accepted",
            next_action=runtime_action,
        )
    return ResultPacketIngestion(
        accepted=True,
        production_order_id=production_order_id,
        dispatch_attempt=envelope.dispatch_attempt,
        idempotency_key=envelope.idempotency_key,
        source_state=envelope.source_state,
        owner_profile=envelope.target_profile,
        target_profile=envelope.target_profile,
        child_kanban_card_id=envelope.child_kanban_card_id,
        packet_id=packet_id,
        bridge_function=bridge_function,
        runtime_action=runtime_action,
        error=None,
        next_action=runtime_action,
    ).to_dict()


def validate_profile_result_packet(
    envelope: ProfileTaskEnvelope,
    result_packet: Any,
) -> dict[str, Any]:
    """Validate a returned result packet against the active profile envelope."""
    bridge_function, _ = _resolve_ingestion_runtime_action(envelope)
    expected_packet = envelope.expected_output_packet
    expected_packet_type = str(expected_packet.get("packet_type", "")).strip()
    required_fields = tuple(expected_packet.get("required_fields") or ())

    if not isinstance(result_packet, dict):
        raise ValueError("result packet must be a JSON object")
    if not result_packet.get("production_order_id"):
        raise ValueError("result packet missing required field: production_order_id")
    if result_packet["production_order_id"] != envelope.production_order_id:
        raise ValueError("result packet production_order_id does not match the active production order")
    if result_packet.get("owner_profile") != envelope.target_profile:
        raise ValueError("result packet owner_profile does not match the active envelope target_profile")
    if result_packet.get("source_state") != envelope.source_state:
        raise ValueError("result packet source_state does not match the active envelope source_state")

    missing = sorted(
        field for field in required_fields
        if field not in result_packet or result_packet[field] in (None, "", [], {})
    )
    if missing:
        raise ValueError(
            "result packet missing required field(s): " + ", ".join(missing)
        )

    _validate_expected_result_packet_type(expected_packet_type, result_packet)
    _validate_approval_boundaries(result_packet)
    _validate_no_direct_workflow_mutation(result_packet)

    if bridge_function == "run_architect_spec_bridge":
        validate_architect_spec_packet(
            result_packet,
            expected_production_order_id=envelope.production_order_id,
        )
        if result_packet.get("next_state") != envelope.expected_next_state:
            raise ValueError("result packet next_state is incompatible with the active envelope expected_next_state")
    elif bridge_function == "run_devos_complete_bridge":
        validate_devos_build_packet(
            result_packet,
            expected_production_order_id=envelope.production_order_id,
            expected_source_state=envelope.source_state,
            expected_next_handoff_target="audit_os",
        )
    elif bridge_function == "run_auditos_review_complete_bridge":
        validate_auditos_review_packet(
            result_packet,
            expected_production_order_id=envelope.production_order_id,
            expected_source_state=envelope.source_state,
        )
    elif bridge_function == "run_architect_reconcile_bridge":
        validate_architect_reconcile_packet(
            result_packet,
            expected_production_order_id=envelope.production_order_id,
            expected_source_state=envelope.source_state,
        )
    elif bridge_function == "run_default_final_review_bridge":
        validate_default_final_review_packet(
            result_packet,
            expected_production_order_id=envelope.production_order_id,
            expected_source_state=envelope.source_state,
        )
    else:  # pragma: no cover - guarded by _resolve_ingestion_runtime_action
        raise DispatchManifestError(
            f"No result packet validator is implemented for bridge function {bridge_function!r}"
        )

    return dict(result_packet)


def apply_accepted_result_action(
    conn: sqlite3.Connection,
    production_order_id: str,
    *,
    result_packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the runtime action for an already accepted result packet."""
    po = _load_production_order(conn, production_order_id)
    if result_packet is not None and result_packet.get("source_state") != po.current_state:
        return _already_applied_or_conflicting_supplied_packet(conn, po, result_packet)
    envelope = build_profile_task_envelope(conn, production_order_id)
    packet = result_packet or _latest_validated_packet(conn, envelope)
    if packet is None:
        raise DispatchManifestError(
            "Cannot apply result action before a result packet has been accepted"
        )
    packet = validate_profile_result_packet(envelope, packet)
    _require_packet_validated_event(conn, envelope, packet)
    bridge_function, runtime_action = _resolve_ingestion_runtime_action(envelope)

    current = _load_production_order(conn, production_order_id)
    if current.current_state != envelope.source_state:
        return _already_applied_result(current, envelope, packet, runtime_action)

    if bridge_function == "run_architect_spec_bridge":
        applied = _apply_architect_spec_action(conn, current, packet)
    elif bridge_function == "run_devos_complete_bridge":
        applied = _apply_devos_complete_action(conn, current, packet)
    elif bridge_function == "run_auditos_review_complete_bridge":
        applied = _apply_audit_pass_action(conn, current, packet)
    elif bridge_function == "run_architect_reconcile_bridge":
        applied = _apply_architect_reconcile_action(conn, current, packet)
    elif bridge_function == "run_default_final_review_bridge":
        applied = _apply_default_final_review_action(conn, current, packet)
    else:  # pragma: no cover - guarded by _resolve_ingestion_runtime_action
        raise DispatchManifestError(
            f"Result action application is not implemented for {bridge_function!r}"
        )

    log_dispatch_event(
        conn,
        production_order_id=production_order_id,
        event_type="dispatch_completed",
        from_state=envelope.source_state,
        to_state=applied.current_state,
        owner_profile=envelope.target_profile,
        target_profile=envelope.target_profile,
        kanban_card_id=envelope.child_kanban_card_id,
        packet_id=_packet_id(packet),
        result="result_action_applied",
        next_action=runtime_action,
    )
    return {
        "applied": True,
        "production_order_id": production_order_id,
        "from_state": envelope.source_state,
        "to_state": applied.current_state,
        "bridge_function": bridge_function,
        "runtime_action": runtime_action,
        "next_action": _next_dispatch_action_for_state(applied.current_state),
    }


def route_production_order_rework(
    conn: sqlite3.Connection,
    production_order_id: str,
    *,
    rejection_source: str,
    rejection_packet: dict[str, Any],
    affected_child_card_id: str | None = None,
    reason_category: str | None = None,
) -> dict[str, Any]:
    """Classify and apply a bounded OrchestratorOS rework route."""
    po = _load_production_order(conn, production_order_id)
    source_state = str(rejection_source or po.current_state).strip()
    if source_state not in {"AUDIT_REJECTED", "DEFAULT_REJECTED"}:
        raise DispatchManifestError(
            "Rework routing only supports AUDIT_REJECTED or DEFAULT_REJECTED sources"
        )
    if po.current_state != source_state:
        prior = _find_rework_route_event(conn, production_order_id, source_state)
        if prior is not None:
            requested_route, _ = _classify_rework_route(
                source_state,
                rejection_packet,
                reason_category=reason_category,
            )
            existing_route = _event_route_decision(prior)
            if requested_route != existing_route:
                raise DispatchManifestError(
                    f"Requested rework route {requested_route!r} conflicts with existing route {existing_route!r}"
                )
            return _rework_decision_from_event(prior, po, applied=False)
        raise DispatchManifestError(
            f"Rework source {source_state!r} conflicts with current state {po.current_state!r}"
        )

    route, explanation = _classify_rework_route(
        source_state,
        rejection_packet,
        reason_category=reason_category,
    )
    idempotency_key = _rework_route_key(
        production_order_id=production_order_id,
        source_state=source_state,
        route_decision=route,
        rejection_packet=rejection_packet,
    )
    existing = _find_rework_route_event(conn, production_order_id, source_state)
    if existing is not None:
        existing_route = _event_route_decision(existing)
        if existing_route != route:
            raise DispatchManifestError(
                f"Conflicting rework route {route!r}; existing route is {existing_route!r}"
            )
        return _rework_decision_from_event(existing, po, applied=False)

    if route == "BLOCKED_NEEDS_JARREN":
        transition_state(
            conn,
            po,
            "BLOCKED_NEEDS_JARREN",
            "orchestrator_os" if source_state == "DEFAULT_REJECTED" else "orchestrator_os",
            result=explanation,
            next_action="ask_jarren_for_approval_or_clarification",
            card_id=po.parent_kanban_card_id,
            event_type="blocked",
        )
        _sync_child_current_state(conn, po)
        event_type = "dispatch_blocked"
        target_profile = "default"
        target_card_id = po.parent_kanban_card_id
        task_type = "blocked_needs_jarren"
        stop_condition = explanation
    elif route == "DEV_REWORK":
        target_profile = "dev_os"
        target_card_id = po.child_kanban_card_ids[2]
        task_type = "dev_rework"
        freeze_handoff_on_card(
            conn,
            target_card_id,
            create_devos_rework_handoff(
                po,
                rejection_packet,
                source_state=source_state,
                route_reason=explanation,
                route_target="DEV_REWORK",
                next_handoff_target="dev_os",
            ),
        )
        transition_state(
            conn,
            po,
            "DEV_REWORK",
            "orchestrator_os",
            result=explanation,
            next_action="dispatch_dev_rework",
            card_id=po.parent_kanban_card_id,
            event_type="retry_started",
        )
        _sync_child_current_state(conn, po)
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (target_card_id,))
        event_type = "dispatch_completed"
        stop_condition = None
    elif route == "SPEC_REWORK":
        target_profile = "architect_os"
        target_card_id = po.child_kanban_card_ids[1]
        task_type = "spec_rework"
        classification_packet = _classification_packet_for_rework(
            production_order_id,
            source_state,
            rejection_packet,
            route,
            explanation,
        )
        freeze_handoff_on_card(
            conn,
            target_card_id,
            create_architect_rework_handoff(po, classification_packet),
        )
        transition_state(
            conn,
            po,
            "SPEC_REWORK",
            "orchestrator_os",
            result=explanation,
            next_action="dispatch_spec_rework",
            card_id=po.parent_kanban_card_id,
            event_type="retry_started",
        )
        _sync_child_current_state(conn, po)
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (target_card_id,))
        event_type = "dispatch_completed"
        stop_condition = None
    else:
        raise DispatchManifestError(f"Unsupported rework route {route!r}")

    log_dispatch_event(
        conn,
        production_order_id=production_order_id,
        event_type=event_type,
        from_state=source_state,
        to_state=route,
        owner_profile="orchestrator_os",
        target_profile=target_profile,
        kanban_card_id=target_card_id,
        packet_id=idempotency_key,
        result=f"rework_route={route}; {explanation}",
        next_action=task_type,
    )
    return ReworkRouteDecision(
        production_order_id=production_order_id,
        source_state=source_state,
        route_decision=route,
        target_profile=target_profile,
        target_child_card_id=target_card_id,
        task_type=task_type,
        explanation=explanation,
        stop_condition=stop_condition,
        idempotency_key=idempotency_key,
        applied=True,
    ).to_dict()


def _resolve_ingestion_runtime_action(
    envelope: ProfileTaskEnvelope,
) -> tuple[str, str]:
    bridge_function = str(envelope.expected_output_packet.get("bridge_function", "")).strip()
    if not bridge_function:
        raise DispatchManifestError("Profile task envelope is missing expected_output_packet.bridge_function")
    try:
        arg_name, expected_packet_type = _MANUAL_FALLBACK_RESULT_ACTIONS[bridge_function]
    except KeyError as exc:
        raise DispatchManifestError(
            f"Result packet ingestion is not safely implemented for bridge function {bridge_function!r}"
        ) from exc

    envelope_packet_type = str(envelope.expected_output_packet.get("packet_type", "")).strip()
    if envelope_packet_type != expected_packet_type:
        raise DispatchManifestError(
            f"Envelope expected_result_packet {envelope_packet_type!r} cannot be mapped to bridge function {bridge_function!r}"
        )

    runtime_action = (
        f"{bridge_function}(conn, production_order_id={envelope.production_order_id!r}, "
        f"{arg_name}=<validated_result_packet>)"
    )
    return bridge_function, runtime_action


def _validate_expected_result_packet_type(
    expected_packet_type: str,
    result_packet: dict[str, Any],
) -> None:
    packet_type = str(result_packet.get("packet_type", "")).strip()
    if packet_type:
        if packet_type != expected_packet_type:
            raise ValueError(
                f"result packet packet_type must be {expected_packet_type!r}"
            )
        return

    result_type = str(result_packet.get("result_type", "")).strip().lower()
    stage = str(result_packet.get("stage", "")).strip().lower()
    compatibility = {
        "architect_spec_packet": {"packet_type": "architect_spec_packet", "stage": "architect_spec"},
        "devos_build_packet": {"packet_type": "devos_build_packet", "result_type": "build_complete"},
        "auditos_review_packet": {"packet_type": "auditos_review_packet", "verdict": "PASS"},
        "architect_reconcile_packet": {"packet_type": "architect_reconcile_packet", "result_type": "accepted", "reconcile_result": "accepted"},
        "default_final_review_packet": {"packet_type": "default_final_review_packet", "final_status": "DONE"},
    }
    expected = compatibility.get(expected_packet_type)
    if expected is None:
        raise DispatchManifestError(
            f"Expected result packet type {expected_packet_type!r} is too vague to validate safely"
        )
    if expected_packet_type == "architect_spec_packet" and stage != expected["stage"]:
        raise ValueError("result packet stage is not compatible with expected architect_spec_packet")
    if expected_packet_type == "devos_build_packet" and "build" not in result_type:
        raise ValueError("result packet result_type is not compatible with expected devos_build_packet")
    if expected_packet_type == "auditos_review_packet":
        verdict = str(result_packet.get("verdict", "")).strip().upper()
        review_result = str(result_packet.get("review_result", "")).strip().lower()
        if verdict != "PASS" or not any(token in review_result for token in ("pass", "passed")):
            raise ValueError(
                "result packet verdict/review_result is not compatible with expected auditos_review_packet"
            )
    if expected_packet_type == "architect_reconcile_packet":
        reconcile_result = str(result_packet.get("reconcile_result", "")).strip().lower()
        if reconcile_result not in {"accept", "accepted", "aligned", "passed", "pass"}:
            raise ValueError(
                "result packet reconcile_result is not compatible with expected architect_reconcile_packet"
            )
    if expected_packet_type == "default_final_review_packet":
        final_status = str(result_packet.get("final_status", "")).strip().upper()
        final_review_result = str(result_packet.get("final_review_result", "")).strip().lower()
        if final_status != "DONE" or not any(token in final_review_result for token in ("accept", "accepted")):
            raise ValueError(
                "result packet final_status/final_review_result is not compatible with expected default_final_review_packet"
            )


def _validate_approval_boundaries(result_packet: dict[str, Any]) -> None:
    text = json.dumps(result_packet, sort_keys=True, default=str).lower()
    approval_required_tokens = (
        "publish",
        "sending",
        "send to",
        "spent $",
        "delete database",
        "drop table",
        "permission widen",
        "request credential",
        "request secret",
        "external api key",
        "destructive change",
    )
    if any(token in text for token in approval_required_tokens):
        raise ValueError("result packet implies external or destructive action without approval")


def _validate_no_direct_workflow_mutation(result_packet: dict[str, Any]) -> None:
    forbidden_keys = {
        "current_owner_profile",
        "workflow_state",
        "transition_state",
        "transition_to_state",
        "mutate_workflow_state",
        "set_task_status",
        "set_current_state",
    }
    found = sorted(key for key in forbidden_keys if key in result_packet)
    if found:
        raise ValueError(
            "result packet attempts to mutate workflow state directly via field(s): "
            + ", ".join(found)
        )


def _packet_id(result_packet: Any) -> str | None:
    if not isinstance(result_packet, dict):
        return None
    value = result_packet.get("packet_id")
    if value in (None, ""):
        return None
    return str(value)


def _task_type_from_idempotency_key(idempotency_key: str) -> str:
    parts = idempotency_key.split(":")
    return parts[5] if len(parts) >= 7 else ""


def _assert_executor_identity(
    envelope: ProfileTaskEnvelope,
    *,
    source_state: str | None,
    target_profile: str | None,
    target_child_card_id: str | None,
    task_type: str | None,
    dispatch_attempt: int | None,
) -> None:
    expected = {
        "source_state": envelope.source_state,
        "target_profile": envelope.target_profile,
        "target_child_card_id": envelope.child_kanban_card_id,
        "task_type": _task_type_from_idempotency_key(envelope.idempotency_key),
        "dispatch_attempt": envelope.dispatch_attempt,
    }
    provided = {
        "source_state": source_state,
        "target_profile": target_profile,
        "target_child_card_id": target_child_card_id,
        "task_type": task_type,
        "dispatch_attempt": dispatch_attempt,
    }
    for key, value in provided.items():
        if value is None:
            continue
        if value != expected[key]:
            raise DispatchManifestError(
                f"Dispatch executor {key} {value!r} does not match active envelope {expected[key]!r}"
            )


def _latest_validated_packet(
    conn: sqlite3.Connection,
    envelope: ProfileTaskEnvelope,
) -> dict[str, Any] | None:
    packet_id = None
    found_event = False
    for event in reversed(list_dispatch_events(conn, envelope.production_order_id)):
        if event["event_type"] != "packet_validated":
            continue
        if event["from_state"] != envelope.source_state:
            continue
        if event["target_profile"] != envelope.target_profile:
            continue
        if event["kanban_card_id"] != envelope.child_kanban_card_id:
            continue
        packet_id = event["packet_id"]
        found_event = True
        break
    if not found_event:
        return None
    for packet in reversed(_frozen_result_packets(conn, envelope.child_kanban_card_id)):
        if packet_id is not None and _packet_id(packet) == packet_id:
            return packet
        if packet_id is None and (
            packet.get("production_order_id") == envelope.production_order_id
            and packet.get("owner_profile") == envelope.target_profile
            and packet.get("source_state") == envelope.source_state
        ):
            return packet
    return None


def _require_packet_validated_event(
    conn: sqlite3.Connection,
    envelope: ProfileTaskEnvelope,
    packet: dict[str, Any],
) -> None:
    packet_id = _packet_id(packet)
    event = _find_dispatch_event(
        conn,
        production_order_id=envelope.production_order_id,
        event_type="packet_validated",
        from_state=envelope.source_state,
        target_profile=envelope.target_profile,
        kanban_card_id=envelope.child_kanban_card_id,
        packet_id=packet_id,
    )
    if event is None:
        raise DispatchManifestError(
            "Cannot apply result action before the result packet is accepted"
        )


def _already_applied_result(
    po: ProductionOrder,
    envelope: ProfileTaskEnvelope,
    packet: dict[str, Any],
    runtime_action: str,
) -> dict[str, Any]:
    if po.current_state == envelope.expected_next_state or _state_has_passed(
        po, envelope.source_state, envelope.expected_next_state
    ):
        return {
            "applied": False,
            "production_order_id": envelope.production_order_id,
            "from_state": envelope.source_state,
            "to_state": po.current_state,
            "bridge_function": envelope.expected_output_packet["bridge_function"],
            "runtime_action": runtime_action,
            "packet_id": _packet_id(packet),
            "next_action": "already_applied",
        }
    raise DispatchManifestError(
        f"Accepted result packet for {envelope.source_state!r} cannot be applied while "
        f"production order is in conflicting state {po.current_state!r}"
    )


def _already_applied_or_conflicting_supplied_packet(
    conn: sqlite3.Connection,
    po: ProductionOrder,
    packet: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(packet, dict):
        raise DispatchManifestError("result_packet must be a JSON object")
    packet_source_state = str(packet.get("source_state", "")).strip()
    packet_owner = str(packet.get("owner_profile", "")).strip()
    packet_id = _packet_id(packet)
    for event in list_dispatch_events(conn, po.production_order_id):
        if event["event_type"] != "packet_validated":
            continue
        if event["from_state"] != packet_source_state:
            continue
        if event["target_profile"] != packet_owner:
            continue
        if packet_id is not None and event["packet_id"] != packet_id:
            continue
        if packet_id is None and event["packet_id"] is not None:
            continue
        expected_next_state = event["to_state"]
        if expected_next_state and (
            po.current_state == expected_next_state
            or _state_has_passed(po, packet_source_state, expected_next_state)
        ):
            return {
                "applied": False,
                "production_order_id": po.production_order_id,
                "from_state": packet_source_state,
                "to_state": po.current_state,
                "bridge_function": "",
                "runtime_action": "",
                "packet_id": packet_id,
                "next_action": "already_applied",
            }
    raise DispatchManifestError(
        f"Accepted result packet source_state {packet_source_state!r} conflicts with "
        f"current production order state {po.current_state!r}"
    )


def _state_has_passed(po: ProductionOrder, from_state: str, expected_next_state: str) -> bool:
    seen_from = False
    for entry in po.stage_history:
        if entry.from_state == from_state:
            seen_from = True
        if seen_from and entry.to_state == expected_next_state:
            return True
    return False


def _sync_child_current_state(conn: sqlite3.Connection, po: ProductionOrder) -> None:
    for cid in po.child_kanban_card_ids:
        conn.execute(
            "UPDATE tasks SET current_state = ? WHERE id = ?",
            (po.current_state, cid),
        )


def _apply_architect_spec_action(
    conn: sqlite3.Connection,
    po: ProductionOrder,
    packet: dict[str, Any],
) -> ProductionOrder:
    devos_card_id = po.child_kanban_card_ids[2]
    freeze_handoff_on_card(conn, devos_card_id, create_devos_handoff(po, packet))
    transition_state(
        conn,
        po,
        "ARCHITECT_READY_FOR_DEV",
        "architect_os",
        result="accepted ArchitectOS result applied; DevOS handoff attached",
        next_action="dispatch_dev_os",
        card_id=po.parent_kanban_card_id,
        event_type="architect_spec_completed",
    )
    log_workflow_event(
        conn,
        po.production_order_id,
        "handoff_created",
        from_state="ARCHITECT_SPEC",
        to_state="ARCHITECT_READY_FOR_DEV",
        owner_profile="architect_os",
        kanban_card_id=devos_card_id,
        result="DevOS handoff packet attached",
        next_action="dispatch_dev_os",
    )
    _sync_child_current_state(conn, po)
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (devos_card_id,))
    return po


def _apply_devos_complete_action(
    conn: sqlite3.Connection,
    po: ProductionOrder,
    packet: dict[str, Any],
) -> ProductionOrder:
    auditos_card_id = po.child_kanban_card_ids[3]
    from_state = po.current_state
    audit_handoff = create_auditos_handoff(
        po,
        packet,
        from_state=from_state if from_state == "DEV_REWORK" else "DEV_IMPLEMENTING",
        to_state="DEV_COMPLETE",
    )
    if po.current_state == "ARCHITECT_READY_FOR_DEV":
        transition_state(
            conn,
            po,
            "DEV_IMPLEMENTING",
            "dev_os",
            result="dev implementation started",
            next_action="complete_dev_build",
            card_id=po.parent_kanban_card_id,
            event_type="dev_build_started",
        )
    transition_state(
        conn,
        po,
        "DEV_COMPLETE",
        "dev_os",
        result="accepted DevOS result applied; AuditOS handoff attached",
        next_action="dispatch_audit_os",
        card_id=po.parent_kanban_card_id,
        event_type="dev_build_completed",
    )
    freeze_handoff_on_card(conn, auditos_card_id, audit_handoff)
    log_workflow_event(
        conn,
        po.production_order_id,
        "handoff_created",
        from_state=from_state,
        to_state="DEV_COMPLETE",
        owner_profile="dev_os",
        kanban_card_id=auditos_card_id,
        result="AuditOS handoff packet attached",
        next_action="dispatch_audit_os",
    )
    _sync_child_current_state(conn, po)
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (auditos_card_id,))
    return po


def _apply_audit_pass_action(
    conn: sqlite3.Connection,
    po: ProductionOrder,
    packet: dict[str, Any],
) -> ProductionOrder:
    reconcile_card_id = po.child_kanban_card_ids[4]
    reconcile_handoff = create_architect_reconcile_handoff(po, packet)
    if po.current_state == "DEV_COMPLETE":
        transition_state(
            conn,
            po,
            "AUDIT_REVIEW",
            "audit_os",
            result="audit review started",
            next_action="complete_audit_review",
            card_id=po.parent_kanban_card_id,
            event_type="audit_review_started",
        )
    transition_state(
        conn,
        po,
        "AUDIT_PASSED",
        "audit_os",
        result="accepted AuditOS result applied; ArchitectOS reconcile handoff attached",
        next_action="dispatch_architect_reconcile",
        card_id=po.parent_kanban_card_id,
        event_type="audit_review_completed",
    )
    freeze_handoff_on_card(conn, reconcile_card_id, reconcile_handoff)
    log_workflow_event(
        conn,
        po.production_order_id,
        "handoff_created",
        from_state="AUDIT_REVIEW",
        to_state="AUDIT_PASSED",
        owner_profile="audit_os",
        kanban_card_id=reconcile_card_id,
        result="ArchitectOS reconcile handoff packet attached",
        next_action="dispatch_architect_reconcile",
    )
    _sync_child_current_state(conn, po)
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (reconcile_card_id,))
    return po


def _apply_architect_reconcile_action(
    conn: sqlite3.Connection,
    po: ProductionOrder,
    packet: dict[str, Any],
) -> ProductionOrder:
    final_card_id = po.child_kanban_card_ids[5]
    final_handoff = create_default_final_review_handoff(po, packet)
    if po.current_state == "AUDIT_PASSED":
        transition_state(
            conn,
            po,
            "ARCHITECT_RECONCILE",
            "architect_os",
            result="architect reconciliation started",
            next_action="complete_architect_reconcile",
            card_id=po.parent_kanban_card_id,
            event_type="architect_reconcile_started",
        )
    transition_state(
        conn,
        po,
        "ARCHITECT_ACCEPTED",
        "architect_os",
        result="accepted ArchitectOS reconcile result applied; final review handoff attached",
        next_action="dispatch_default_final_review",
        card_id=po.parent_kanban_card_id,
        event_type="architect_reconcile_completed",
    )
    freeze_handoff_on_card(conn, final_card_id, final_handoff)
    log_workflow_event(
        conn,
        po.production_order_id,
        "handoff_created",
        from_state="ARCHITECT_RECONCILE",
        to_state="ARCHITECT_ACCEPTED",
        owner_profile="architect_os",
        kanban_card_id=final_card_id,
        result="Default final review handoff packet attached",
        next_action="dispatch_default_final_review",
    )
    _sync_child_current_state(conn, po)
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (final_card_id,))
    return po


def _apply_default_final_review_action(
    conn: sqlite3.Connection,
    po: ProductionOrder,
    packet: dict[str, Any],
) -> ProductionOrder:
    if po.current_state == "ARCHITECT_ACCEPTED":
        transition_state(
            conn,
            po,
            "DEFAULT_FINAL_REVIEW",
            "default",
            result="default final review started",
            next_action="complete_default_final_review",
            card_id=po.parent_kanban_card_id,
            event_type="default_final_review_started",
        )
    transition_state(
        conn,
        po,
        "DONE",
        "default",
        result="accepted Default final review result applied",
        next_action=packet["next_action"],
        card_id=po.parent_kanban_card_id,
        event_type="default_final_review_completed",
    )
    log_workflow_event(
        conn,
        po.production_order_id,
        "workflow_completed",
        owner_profile="default",
        kanban_card_id=po.parent_kanban_card_id,
        result=str(packet["final_status"]),
        next_action=packet["next_action"],
    )
    po.final_status = str(packet["final_status"])
    _sync_child_current_state(conn, po)
    conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (po.child_kanban_card_ids[5],))
    return po


def _next_dispatch_action_for_state(state: str) -> str:
    return {
        "ARCHITECT_READY_FOR_DEV": "prepare_dev_dispatch",
        "DEV_COMPLETE": "prepare_audit_dispatch",
        "AUDIT_PASSED": "prepare_architect_reconcile_dispatch",
        "ARCHITECT_ACCEPTED": "prepare_default_final_review_dispatch",
        "DONE": "report_done_to_jarren",
    }.get(state, "inspect_production_order")


def _classify_rework_route(
    source_state: str,
    packet: dict[str, Any],
    *,
    reason_category: str | None,
) -> tuple[str, str]:
    structured_fields = [
        reason_category,
        packet.get("classification"),
        packet.get("rework_owner"),
        packet.get("rejection_category"),
        packet.get("failure_type"),
        packet.get("recommended_route"),
        packet.get("route_target"),
    ]
    fallback_text_fields = [
        packet.get("summary"),
        packet.get("rejection_reason"),
        packet.get("original_brief_mismatch"),
        packet.get("correction_request"),
        packet.get("risks_or_notes"),
    ]
    structured_text = json.dumps(structured_fields, sort_keys=True, default=str).lower()
    fallback_text = json.dumps(fallback_text_fields, sort_keys=True, default=str).lower()
    approval_tokens = (
        "approval",
        "needs_approval",
        "scope change",
        "scope_change",
        "destructive",
        "spending",
        "publish",
        "permission",
        "secret",
        "credential",
        "new repo",
        "external",
    )
    spec_tokens = (
        "spec",
        "design",
        "architecture",
        "architect",
        "brief mismatch",
        "brief/spec",
        "requirement ambiguity",
        "ambiguous requirement",
    )
    dev_tokens = (
        "implementation",
        "test failure",
        "tests failed",
        "bug",
        "regression",
        "output mismatch",
        "final output mismatch",
        "implemented behavior",
    )
    audit_tokens = ("audit miss", "audit_miss", "audit gap")
    if any(token in structured_text for token in approval_tokens) or any(
        token in fallback_text for token in approval_tokens
    ):
        return "BLOCKED_NEEDS_JARREN", "Rejection crosses an approval boundary or requires Jarren input."
    if any(token in structured_text for token in spec_tokens):
        return "SPEC_REWORK", "Rejection is classified as a spec/design mismatch for ArchitectOS."
    if any(token in structured_text for token in dev_tokens):
        return "DEV_REWORK", "Rejection is classified as an implementation/test mismatch for DevOS."
    if source_state == "DEFAULT_REJECTED" and any(token in structured_text for token in audit_tokens):
        return "BLOCKED_NEEDS_JARREN", "Default rejection points at an audit gap, but this slice does not add an AuditOS re-route."
    if any(token in fallback_text for token in spec_tokens):
        return "SPEC_REWORK", "Rejection narrative indicates a spec/design mismatch for ArchitectOS."
    if any(token in fallback_text for token in dev_tokens):
        return "DEV_REWORK", "Rejection narrative indicates an implementation/test mismatch for DevOS."
    if source_state == "DEFAULT_REJECTED" and any(token in fallback_text for token in audit_tokens):
        return "BLOCKED_NEEDS_JARREN", "Default rejection points at an audit gap, but this slice does not add an AuditOS re-route."
    return "BLOCKED_NEEDS_JARREN", "Rework route is ambiguous and must not be guessed."


def _rework_route_key(
    *,
    production_order_id: str,
    source_state: str,
    route_decision: str,
    rejection_packet: dict[str, Any],
) -> str:
    return (
        f"rework:{production_order_id}:{source_state}:{route_decision}:"
        f"{_payload_hash(rejection_packet)}"
    )


def _classification_packet_for_rework(
    production_order_id: str,
    source_state: str,
    packet: dict[str, Any],
    route: str,
    explanation: str,
) -> dict[str, Any]:
    return {
        "production_order_id": production_order_id,
        "owner_profile": "orchestrator_os",
        "source_state": source_state,
        "default_rejection_reason": packet.get("rejection_reason", packet.get("summary", "")),
        "classification": "spec_or_design_mismatch",
        "route_target": route,
        "route_reason": explanation,
        "next_handoff_target": "architect_os",
        "correction_request": packet.get("correction_request", packet.get("evidence", [])),
    }


def _find_rework_route_event(
    conn: sqlite3.Connection,
    production_order_id: str,
    source_state: str,
) -> dict[str, Any] | None:
    for event in reversed(list_dispatch_events(conn, production_order_id)):
        if event["from_state"] != source_state:
            continue
        if event["event_type"] in {"dispatch_completed", "dispatch_blocked"}:
            route = _event_route_decision(event)
            if route in {"DEV_REWORK", "SPEC_REWORK", "BLOCKED_NEEDS_JARREN"}:
                return event
    return None


def _event_route_decision(event: dict[str, Any]) -> str:
    result = str(event.get("result") or "")
    match = re.search(r"rework_route=([A-Z_]+)", result)
    if match:
        return match.group(1)
    return str(event.get("to_state") or "")


def _rework_decision_from_event(
    event: dict[str, Any],
    po: ProductionOrder,
    *,
    applied: bool,
) -> dict[str, Any]:
    route = _event_route_decision(event)
    return ReworkRouteDecision(
        production_order_id=po.production_order_id,
        source_state=str(event["from_state"]),
        route_decision=route,
        target_profile=str(event["target_profile"]),
        target_child_card_id=event["kanban_card_id"],
        task_type=str(event["next_action"] or ""),
        explanation=str(event["result"] or ""),
        stop_condition=str(event["error"] or "") or None,
        idempotency_key=str(event["packet_id"] or ""),
        applied=applied,
    ).to_dict()


def log_dispatch_event(
    conn: sqlite3.Connection,
    *,
    production_order_id: str,
    event_type: str,
    from_state: str | None,
    owner_profile: str,
    target_profile: str,
    kanban_card_id: str,
    to_state: str | None = None,
    packet_id: str | None = None,
    result: str | None = None,
    error: str | None = None,
    next_action: str | None = None,
) -> int:
    """Write a dispatch-only lifecycle event to the shared event ledger."""
    event_type = _validate_dispatch_event_type(event_type)
    return log_workflow_event(
        conn,
        production_order_id,
        event_type,
        from_state=from_state,
        to_state=to_state,
        owner_profile=owner_profile,
        target_profile=target_profile,
        kanban_card_id=kanban_card_id,
        packet_id=packet_id,
        result=result,
        error=error,
        next_action=next_action,
    )


def list_dispatch_events(
    conn: sqlite3.Connection,
    production_order_id: str,
) -> list[dict[str, Any]]:
    """Return dispatch-only events for a production order in insertion order."""
    rows = conn.execute(
        """
        SELECT id, timestamp, production_order_id, event_type, from_state, to_state,
               owner_profile, target_profile, kanban_card_id, packet_id,
               result, error, next_action
        FROM production_order_events
        WHERE production_order_id = ? AND event_type IN ({placeholders})
        ORDER BY id
        """.format(placeholders=",".join("?" for _ in ALLOWED_DISPATCH_EVENT_TYPES)),
        (production_order_id, *ALLOWED_DISPATCH_EVENT_TYPES),
    ).fetchall()
    return [dispatch_event_to_dict(row) for row in rows]


def dispatch_event_to_dict(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    """Normalize a dispatch-event row into the public ledger shape."""
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "production_order_id": row["production_order_id"],
        "event_type": row["event_type"],
        "from_state": row["from_state"],
        "to_state": row["to_state"],
        "owner_profile": row["owner_profile"],
        "target_profile": row["target_profile"],
        "kanban_card_id": row["kanban_card_id"],
        "packet_id": row["packet_id"],
        "result": row["result"],
        "error": row["error"],
        "next_action": row["next_action"],
    }


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
        dispatch_attempt=envelope.dispatch_attempt,
        idempotency_key=envelope.idempotency_key,
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
        dispatch_attempt=manifest.dispatch_attempt,
        idempotency_key=manifest.idempotency_key,
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
    dispatch_attempt = _dispatch_attempt_for_state(po)
    idempotency_key = _dispatch_idempotency_key(
        production_order_id=po.production_order_id,
        source_state=po.current_state,
        target_profile=target_profile,
        target_child_card_id=target_child_card_id,
        task_type=task_type,
        dispatch_attempt=dispatch_attempt,
    )
    return DispatchManifest(
        production_order_id=po.production_order_id,
        current_state=po.current_state,
        current_owner_profile=po.current_owner_profile,
        dispatch_attempt=dispatch_attempt,
        idempotency_key=idempotency_key,
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
