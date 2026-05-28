"""Tests for Slice 4 — Hermes Production Workflow Runtime Bridge.

Tests the ``production_order_db`` module against an in-memory Kanban DB
fixture.  Covers:

- ProductionOrder dataclass schema
- production_order_id generation (format, uniqueness)
- Parent Kanban card creation with workflow_template_id
- Six child cards with correct owner profiles and linking
- State transition validation (valid + invalid + ownership)
- Trigger phrase detection (accept + reject)
- Brief validation (accept + reject)
- Event logging to production_order_events table
- Handoff packet creation and freezing on OrchestratorOS card
- Full bridge end-to-end smoke test
- Backward/compatibility: pre-existing tasks have NULL new columns
- Negative: empty brief, invalid transitions, wrong profile
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.production_order_db import (
    CHILD_CARD_DEFS,
    PRODUCTION_ORDER_FIELD,
    STATE_FIELD,
    WORKFLOW_INITIAL_STATE,
    WORKFLOW_SPEC_SOURCE,
    WORKFLOW_TEMPLATE_ID,
    ProductionOrder,
    StageEntry,
    StateTransitionError,
    _base36_random,
    create_architect_handoff,
    create_auditos_handoff,
    create_devos_handoff,
    create_orchestrator_handoff,
    create_production_kanban_graph,
    create_production_order,
    detect_approval_phrase,
    freeze_handoff_on_card,
    freeze_result_on_card,
    generate_production_order_id,
    list_production_orders,
    log_workflow_event,
    run_architect_spec_bridge,
    run_architect_reconcile_bridge,
    run_auditos_review_complete_bridge,
    run_auditos_review_reject_bridge,
    run_default_final_review_bridge,
    run_default_final_review_reject_bridge,
    run_devos_complete_bridge,
    run_devos_rework_complete_bridge,
    run_full_bridge,
    run_orchestrator_rework_bridge,
    run_orchestrator_classification_bridge,
    run_orchestrator_default_rejection_triage_bridge,
    run_orchestrator_triage_bridge,
    transition_state,
    validate_auditos_rejection_packet,
    validate_auditos_review_packet,
    validate_architect_spec_packet,
    validate_architect_reconcile_packet,
    validate_brief,
    validate_default_final_review_packet,
    validate_default_rejection_packet,
    validate_devos_build_packet,
    validate_orchestrator_classification_packet,
    validate_state_transition,
)
from hermes_cli.production_order_dispatch import (
    ALLOWED_DISPATCH_EVENT_TYPES,
    DispatchManifestError,
    build_dispatch_manifest,
    build_manual_fallback_handoff,
    build_profile_task_envelope,
    dispatch_event_to_dict,
    dispatch_manifest_for_order,
    execute_profile_dispatch,
    apply_accepted_result_action,
    ingest_profile_result_packet,
    list_dispatch_events,
    log_dispatch_event,
    manual_fallback_handoff_for_envelope,
    profile_task_envelope_for_order,
    route_production_order_rework,
    validate_profile_result_packet,
)
from hermes_cli.kanban import _cmd_production_order


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Set up a temp Kanban DB for testing."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    yield
    # cleanup handled by tmp_path


@pytest.fixture
def conn(kanban_home):
    """Provide a Kanban DB connection."""
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def sample_brief() -> dict:
    """A valid approved brief for testing."""
    return {
        "title": "Test authentication feature",
        "objective": "Add JWT authentication to the Relay demo",
        "target repo or workspace": "relay-go-app",
        "scope": "Implement /login and /register endpoints",
        "out of scope": "OAuth, password reset",
        "acceptance criteria": "All tests pass, protected routes reject unauthenticated requests",
        "stop conditions": "Secret management requires external service",
        "approval boundaries": "No spending, no publishing",
        "constraints": "Use existing Go module",
        "expected output": "Working auth endpoints with tests",
    }


@pytest.fixture
def snake_case_brief(sample_brief) -> dict:
    """Equivalent valid brief using supported snake_case aliases."""
    return {
        "title": sample_brief["title"],
        "objective": sample_brief["objective"],
        "target_repo_or_workspace": sample_brief["target repo or workspace"],
        "scope": sample_brief["scope"],
        "out_of_scope": sample_brief["out of scope"],
        "acceptance_criteria": sample_brief["acceptance criteria"],
        "stop_conditions": sample_brief["stop conditions"],
        "approval_boundaries": sample_brief["approval boundaries"],
        "constraints": sample_brief["constraints"],
        "expected_output": sample_brief["expected output"],
    }


def architect_spec_packet(production_order_id: str) -> dict:
    """Minimal ArchitectOS result/spec packet for Slice 6 tests."""
    return {
        "production_order_id": production_order_id,
        "packet_type": "architect_spec_packet",
        "stage": "architect_spec",
        "owner_profile": "architect_os",
        "source_state": "ARCHITECT_SPEC",
        "objective": "Specify the bounded Slice 6 handoff bridge.",
        "source_truth": [WORKFLOW_SPEC_SOURCE],
        "scope": ["Attach a DevOS handoff packet and advance the PO state."],
        "out_of_scope": ["DevOS implementation execution", "Slice 7"],
        "acceptance_criteria": [
            "Production order transitions to ARCHITECT_READY_FOR_DEV.",
            "Current owner becomes dev_os.",
        ],
        "devos_task": "Prepare for implementation from the approved spec; do not execute Slice 7.",
        "files_or_areas_allowed": [
            "hermes_cli/production_order_db.py",
            "hermes_cli/kanban.py",
            "tests/hermes_cli/test_production_order_bridge.py",
        ],
        "stop_conditions": [
            "Production order is not in ARCHITECT_SPEC.",
            "Current owner is not architect_os.",
        ],
        "approval_boundaries": ["Do not trigger DevOS execution in Slice 6."],
        "artifact_references": ["architect-spec.json"],
        "next_state": "ARCHITECT_READY_FOR_DEV",
    }


def devos_build_packet(production_order_id: str) -> dict:
    """Minimal DevOS build/result packet for Slice 7 tests."""
    return {
        "production_order_id": production_order_id,
        "packet_type": "devos_build_packet",
        "owner_profile": "dev_os",
        "source_state": "ARCHITECT_READY_FOR_DEV",
        "result_type": "build_complete",
        "summary": "Implemented the approved Slice 7 bridge and preserved graph semantics.",
        "files_changed": [
            "hermes_cli/production_order_db.py",
            "hermes_cli/kanban.py",
            "tests/hermes_cli/test_production_order_bridge.py",
        ],
        "tests_run": [
            "pytest tests/hermes_cli/test_production_order_bridge.py -q",
        ],
        "test_status": "green",
        "limitations_or_notes": ["AuditOS should verify smoke evidence against the existing board."],
        "next_handoff_target": "audit_os",
    }


def devos_rework_packet(production_order_id: str) -> dict:
    """Minimal DevOS rework/result packet for failure-loop tests."""
    return {
        "production_order_id": production_order_id,
        "owner_profile": "dev_os",
        "source_state": "DEV_REWORK",
        "result_type": "rework_complete",
        "summary": "Applied the AuditOS correction request and preserved approved scope.",
        "files_changed": [
            "hermes_cli/production_order_db.py",
            "tests/hermes_cli/test_production_order_bridge.py",
        ],
        "tests_run": [
            "pytest tests/hermes_cli/test_production_order_bridge.py -q",
        ],
        "test_status": "green",
        "limitations_or_notes": ["Rework evidence is ready for AuditOS review."],
        "next_handoff_target": "audit_os",
    }



def create_architect_spec_order(conn, sample_brief) -> ProductionOrder:
    """Advance a fresh production order to the ArchitectOS spec stage."""
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )
    return run_orchestrator_triage_bridge(conn, production_order_id=po.production_order_id)



def create_ready_for_dev_order(conn, sample_brief) -> ProductionOrder:
    """Advance a fresh production order through Slice 6 for Slice 7 tests."""
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )
    po = run_orchestrator_triage_bridge(conn, production_order_id=po.production_order_id)
    return run_architect_spec_bridge(
        conn,
        production_order_id=po.production_order_id,
        architect_packet=architect_spec_packet(po.production_order_id),
    )


def audit_review_packet(production_order_id: str, source_state: str = "DEV_COMPLETE") -> dict:
    """Minimal AuditOS review/result packet for happy-path tests."""
    return {
        "production_order_id": production_order_id,
        "owner_profile": "audit_os",
        "source_state": source_state,
        "review_result": "passed",
        "summary": "Audit verified the implementation and evidence.",
        "evidence": ["tests passed", "changed files reviewed"],
        "tests_reviewed": ["pytest tests/hermes_cli/test_production_order_bridge.py -q"],
        "verdict": "PASS",
        "risks_or_notes": ["No blocking risks found."],
        "next_handoff_target": "architect_os",
    }


def audit_rejection_packet(
    production_order_id: str,
    source_state: str = "DEV_COMPLETE",
) -> dict:
    """Minimal AuditOS rejection/correction packet for failure-loop tests."""
    return {
        "production_order_id": production_order_id,
        "owner_profile": "audit_os",
        "source_state": source_state,
        "review_result": "rejected",
        "summary": "Audit found issues that require DevOS rework.",
        "evidence": ["audit review notes", "failing edge-case coverage"],
        "tests_reviewed": ["pytest tests/hermes_cli/test_production_order_bridge.py -q"],
        "verdict": "REJECT",
        "risks_or_notes": ["Implementation needs correction before re-audit."],
        "next_handoff_target": "orchestrator_os",
        "correction_request": [
            "Address the audit findings and preserve the approved scope.",
            "Return updated evidence to AuditOS after rework.",
        ],
    }


def architect_reconcile_packet(
    production_order_id: str,
    source_state: str = "AUDIT_PASSED",
) -> dict:
    """Minimal ArchitectOS reconciliation packet for happy-path tests."""
    return {
        "production_order_id": production_order_id,
        "packet_type": "architect_reconcile_packet",
        "owner_profile": "architect_os",
        "source_state": source_state,
        "reconcile_result": "accepted",
        "summary": "Implementation remains aligned with the approved architecture.",
        "architecture_alignment": "aligned",
        "drift_assessment": "No architecture drift.",
        "spec_patch_needed": False,
        "risks_or_notes": ["No rework needed."],
        "next_handoff_target": "default",
    }


def final_review_packet(
    production_order_id: str,
    source_state: str = "ARCHITECT_ACCEPTED",
) -> dict:
    """Minimal Default Hermes final-review packet for happy-path tests."""
    return {
        "production_order_id": production_order_id,
        "owner_profile": "default",
        "source_state": source_state,
        "final_review_result": "accepted",
        "summary": "Final review confirms the approved brief is complete.",
        "original_brief_alignment": "Matches the approved brief.",
        "artifacts_reviewed": ["DevOS result", "AuditOS result", "ArchitectOS reconcile result"],
        "evidence_summary": "All stage evidence is present and accepted.",
        "final_status": "DONE",
        "next_action": "report_done_to_jarren",
    }


def default_rejection_packet(
    production_order_id: str,
    source_state: str = "DEFAULT_FINAL_REVIEW",
) -> dict:
    """Minimal Default Hermes rejection packet for failure-loop tests."""
    return {
        "production_order_id": production_order_id,
        "owner_profile": "default",
        "source_state": source_state,
        "review_result": "rejected",
        "summary": "Final review found the output does not match the approved brief.",
        "original_brief_mismatch": "The delivered output misses the requested final behavior.",
        "rejection_reason": "The work does not satisfy the approved brief.",
        "evidence": ["final review notes", "brief comparison mismatch"],
        "recommended_route": "orchestrator_triage",
        "next_handoff_target": "orchestrator_os",
    }


def orchestrator_classification_packet(
    production_order_id: str,
    classification: str,
    *,
    source_state: str = "ORCHESTRATOR_TRIAGE",
) -> dict:
    """Minimal OrchestratorOS classification packet for default rejection routing."""
    route_target = "DEV_REWORK" if classification == "implementation_mismatch" else "SPEC_REWORK"
    next_handoff_target = "dev_os" if route_target == "DEV_REWORK" else "architect_os"
    route_reason = (
        "Default rejection indicates an implementation mismatch that DevOS must correct."
        if route_target == "DEV_REWORK"
        else "Default rejection indicates a spec or design mismatch that ArchitectOS must correct."
    )
    return {
        "production_order_id": production_order_id,
        "owner_profile": "orchestrator_os",
        "source_state": source_state,
        "default_rejection_reason": "The delivered output does not match the approved brief.",
        "classification": classification,
        "route_target": route_target,
        "route_reason": route_reason,
        "next_handoff_target": next_handoff_target,
        "correction_request": ["Correct the mismatch and preserve the approved scope."],
    }


def create_dev_complete_order(conn, sample_brief) -> ProductionOrder:
    """Advance a fresh production order through Slice 7."""
    po = create_ready_for_dev_order(conn, sample_brief)
    return run_devos_complete_bridge(
        conn,
        production_order_id=po.production_order_id,
        devos_packet=devos_build_packet(po.production_order_id),
    )


def create_audit_passed_order(conn, sample_brief) -> ProductionOrder:
    """Advance a fresh production order through AuditOS happy path."""
    po = create_dev_complete_order(conn, sample_brief)
    return run_auditos_review_complete_bridge(
        conn,
        production_order_id=po.production_order_id,
        review_packet=audit_review_packet(po.production_order_id),
    )


def create_architect_accepted_order(conn, sample_brief) -> ProductionOrder:
    """Advance a fresh production order through ArchitectOS reconciliation."""
    po = create_audit_passed_order(conn, sample_brief)
    return run_architect_reconcile_bridge(
        conn,
        production_order_id=po.production_order_id,
        reconcile_packet=architect_reconcile_packet(po.production_order_id),
    )


def create_default_final_review_order(conn, sample_brief) -> ProductionOrder:
    """Advance a fresh production order to the Default final review stage."""
    po = create_architect_accepted_order(conn, sample_brief)
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
    for cid in po.child_kanban_card_ids:
        conn.execute(
            "UPDATE tasks SET current_state = ? WHERE id = ?",
            (po.current_state, cid),
        )
    return po

def _assert_six_card_graph_preserved(conn, po, original_child_ids):
    refreshed = [
        order for order in list_production_orders(conn)
        if order.production_order_id == po.production_order_id
    ][0]
    assert refreshed.parent_kanban_card_id == po.parent_kanban_card_id
    assert refreshed.child_kanban_card_ids == original_child_ids
    assert len(refreshed.child_kanban_card_ids) == 6

def _po_event_types(conn, production_order_id):
    rows = conn.execute(
        "SELECT event_type FROM production_order_events WHERE production_order_id = ? ORDER BY id",
        (production_order_id,),
    ).fetchall()
    return [row["event_type"] for row in rows]

# ---------------------------------------------------------------------------
# Production Order ID Generation
# ---------------------------------------------------------------------------



def test_production_order_id_format():
    """ID format: PO-YYYYMMDD-XXXX where XXXX is base36."""
    po_id = generate_production_order_id()
    assert re.match(r"^PO-\d{8}-[0-9a-z]{4}$", po_id), f"Bad format: {po_id}"


def test_production_order_id_collision():
    """Two calls produce different IDs (extremely unlikely collision)."""
    ids = {generate_production_order_id() for _ in range(100)}
    assert len(ids) > 1, "All generated IDs are identical (extremely unlikely)"


def test_production_order_id_unique_within_date():
    """IDs generated on the same date are different."""
    ids = {generate_production_order_id() for _ in range(50)}
    assert len(ids) == 50, "Expected 50 unique IDs"


# ---------------------------------------------------------------------------
# ProductionOrder Dataclass
# ---------------------------------------------------------------------------


def test_production_order_schema():
    """All ProductionOrder fields populated correctly."""
    po = ProductionOrder(
        production_order_id="PO-20260525-A3XF",
        title="Test order",
        source_brief="The brief",
        approved_by="Jarren",
        approved_at=1234567890,
        priority_lane="Hermes OS",
        repo_or_workspace="hermes-agent",
        current_state="PRODUCTION_ORDER_CREATED",
        current_owner_profile="orchestrator_os",
        stage_history=[StageEntry("BRIEF_DRAFTED", "ACTION_APPROVED", "hermes", 1234567890)],
        parent_kanban_card_id="abc123",
        child_kanban_card_ids=["child1", "child2"],
    )
    assert po.production_order_id == "PO-20260525-A3XF"
    assert po.title == "Test order"
    assert po.current_state == "PRODUCTION_ORDER_CREATED"
    assert po.current_owner_profile == "orchestrator_os"
    assert len(po.stage_history) == 1
    assert po.parent_kanban_card_id == "abc123"
    assert po.child_kanban_card_ids == ["child1", "child2"]
    assert po.final_status is None


# ---------------------------------------------------------------------------
# Production Order Creation
# ---------------------------------------------------------------------------


def test_create_production_order(conn, sample_brief):
    """Creates a production order with parent card and correct fields."""
    brief_text = json.dumps(sample_brief, indent=2)
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=brief_text,
        approved_by="Jarren",
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )

    # Check production_order_id format
    assert re.match(r"^PO-\d{8}-[0-9a-z]{4}$", po.production_order_id)

    # Check parent card exists
    parent = kb.get_task(conn, po.parent_kanban_card_id)
    assert parent is not None
    assert parent.title == f"Production Order: {sample_brief['title']}"
    assert parent.body == brief_text
    assert parent.production_order_id == po.production_order_id
    assert parent.current_state == WORKFLOW_INITIAL_STATE
    assert parent.workflow_template_id == WORKFLOW_TEMPLATE_ID


def test_parent_kanban_card_created(conn, sample_brief):
    """Parent card exists with correct metadata."""
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )
    parent = kb.get_task(conn, po.parent_kanban_card_id)
    assert parent is not None
    assert parent.workflow_template_id == WORKFLOW_TEMPLATE_ID
    assert parent.production_order_id == po.production_order_id
    assert parent.current_state == WORKFLOW_INITIAL_STATE


def test_production_order_idempotency(conn, sample_brief):
    """Same idempotency_key returns existing order, no duplicate."""
    brief_text = json.dumps(sample_brief, indent=2)
    key = "test-dup-key"
    po1 = create_production_order(
        conn, title=sample_brief["title"],
        source_brief=brief_text,
        idempotency_key=key,
    )
    po2 = create_production_order(
        conn, title=sample_brief["title"],
        source_brief=brief_text,
        idempotency_key=key,
    )
    assert po1.production_order_id == po2.production_order_id
    assert po1.parent_kanban_card_id == po2.parent_kanban_card_id


# ---------------------------------------------------------------------------
# Kanban Graph Creation
# ---------------------------------------------------------------------------


def test_six_child_cards_created(conn, sample_brief):
    """Exactly 6 child cards exist."""
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )
    child_ids = create_production_kanban_graph(conn, po)
    assert len(child_ids) == 6, f"Expected 6 child cards, got {len(child_ids)}"


def test_child_card_owner_profiles(conn, sample_brief):
    """Each child card has the correct assignee."""
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )
    child_ids = create_production_kanban_graph(conn, po)

    for child_id, (order, expected_title, expected_owner, expected_status) in zip(
        child_ids, CHILD_CARD_DEFS
    ):
        task = kb.get_task(conn, child_id)
        assert task is not None, f"Child card {child_id} not found"
        assert task.assignee == expected_owner, (
            f"Card {order}: expected assignee {expected_owner!r}, got {task.assignee!r}"
        )


def test_child_card_linked_to_parent(conn, sample_brief):
    """task_links has 6 parent->child entries."""
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )
    child_ids = create_production_kanban_graph(conn, po)

    links = conn.execute(
        "SELECT COUNT(*) AS n FROM task_links WHERE parent_id = ?",
        (po.parent_kanban_card_id,),
    ).fetchone()["n"]
    assert links == 6, f"Expected 6 parent->child links, got {links}"


def test_child_card_initial_status(conn, sample_brief):
    """OrchestratorOS card is ready, others are todo."""
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )
    child_ids = create_production_kanban_graph(conn, po)

    for child_id, (order, title, owner, expected_status) in zip(
        child_ids, CHILD_CARD_DEFS
    ):
        task = kb.get_task(conn, child_id)
        assert task is not None
        assert task.status == expected_status, (
            f"Card {title}: expected status {expected_status!r}, got {task.status!r}"
        )


def test_workflow_template_id_set(conn, sample_brief):
    """All 7 cards have workflow_template_id = hermes-production-workflow-v1."""
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )
    child_ids = create_production_kanban_graph(conn, po)

    all_ids = [po.parent_kanban_card_id] + child_ids
    for card_id in all_ids:
        task = kb.get_task(conn, card_id)
        assert task is not None
        assert task.workflow_template_id == WORKFLOW_TEMPLATE_ID, (
            f"Card {card_id}: expected template ID {WORKFLOW_TEMPLATE_ID}"
        )


# ---------------------------------------------------------------------------
# State Transition Validation
# ---------------------------------------------------------------------------


def test_valid_state_transitions():
    """Wired production workflow transitions pass validation."""
    # BRIEF_DRAFTED -> ACTION_APPROVED
    assert validate_state_transition("BRIEF_DRAFTED", "ACTION_APPROVED", "hermes")
    # ACTION_APPROVED -> PRODUCTION_ORDER_CREATED
    assert validate_state_transition("ACTION_APPROVED", "PRODUCTION_ORDER_CREATED", "hermes")
    # PRODUCTION_ORDER_CREATED -> ORCHESTRATOR_TRIAGE
    assert validate_state_transition("PRODUCTION_ORDER_CREATED", "ORCHESTRATOR_TRIAGE", "orchestrator_os")
    # ORCHESTRATOR_TRIAGE -> ARCHITECT_SPEC
    assert validate_state_transition("ORCHESTRATOR_TRIAGE", "ARCHITECT_SPEC", "orchestrator_os")
    # ARCHITECT_SPEC -> ARCHITECT_READY_FOR_DEV
    assert validate_state_transition("ARCHITECT_SPEC", "ARCHITECT_READY_FOR_DEV", "architect_os")
    assert validate_state_transition("ARCHITECT_READY_FOR_DEV", "DEV_IMPLEMENTING", "dev_os")
    assert validate_state_transition("DEV_IMPLEMENTING", "DEV_COMPLETE", "dev_os")
    assert validate_state_transition("DEV_COMPLETE", "AUDIT_REVIEW", "audit_os")
    assert validate_state_transition("AUDIT_REVIEW", "AUDIT_PASSED", "audit_os")
    assert validate_state_transition("AUDIT_REVIEW", "AUDIT_REJECTED", "audit_os")
    assert validate_state_transition("AUDIT_REJECTED", "DEV_REWORK", "orchestrator_os")
    assert validate_state_transition("DEV_REWORK", "DEV_COMPLETE", "dev_os")
    assert validate_state_transition("AUDIT_PASSED", "ARCHITECT_RECONCILE", "architect_os")
    assert validate_state_transition("ARCHITECT_RECONCILE", "ARCHITECT_ACCEPTED", "architect_os")
    assert validate_state_transition("ARCHITECT_ACCEPTED", "DEFAULT_FINAL_REVIEW", "default")
    assert validate_state_transition("DEFAULT_FINAL_REVIEW", "DONE", "default")
    assert validate_state_transition("DEFAULT_FINAL_REVIEW", "DEFAULT_REJECTED", "default")
    assert validate_state_transition("DEFAULT_REJECTED", "ORCHESTRATOR_TRIAGE", "orchestrator_os")
    assert validate_state_transition("ORCHESTRATOR_TRIAGE", "DEV_REWORK", "orchestrator_os")
    assert validate_state_transition("ORCHESTRATOR_TRIAGE", "SPEC_REWORK", "orchestrator_os")


def test_invalid_state_transition_rejected():
    """Invalid transition raises StateTransitionError."""
    # ARCHITECT_READY_FOR_DEV -> DEV_COMPLETE (skip DEV_IMPLEMENTING) - not wired
    with pytest.raises(StateTransitionError):
        validate_state_transition("ARCHITECT_READY_FOR_DEV", "DEV_COMPLETE", "dev_os")

    # DEV_IMPLEMENTING -> AUDIT_REVIEW (skip DEV_COMPLETE) - not wired in Slice 4
    with pytest.raises(StateTransitionError):
        validate_state_transition("DEV_IMPLEMENTING", "AUDIT_REVIEW", "dev_os")

    # Unknown from_state
    with pytest.raises(StateTransitionError):
        validate_state_transition("UNKNOWN_STATE", "SOME_STATE", "hermes")


def test_state_ownership_enforced():
    """Wrong profile for state raises StateTransitionError."""
    # dev_os cannot transition PRODUCTION_ORDER_CREATED (owned by orchestrator_os)
    with pytest.raises(StateTransitionError):
        validate_state_transition("PRODUCTION_ORDER_CREATED", "ORCHESTRATOR_TRIAGE", "dev_os")

    # audit_os cannot advance out of AUDIT_REJECTED
    with pytest.raises(StateTransitionError):
        validate_state_transition("AUDIT_REJECTED", "DEV_REWORK", "audit_os")

    # default cannot advance out of DEFAULT_REJECTED
    with pytest.raises(StateTransitionError):
        validate_state_transition("DEFAULT_REJECTED", "ORCHESTRATOR_TRIAGE", "default")


def test_auditos_rejection_packet_validation():
    packet = audit_rejection_packet("PO-20260525-REJECT")

    validated = validate_auditos_rejection_packet(
        packet,
        expected_production_order_id="PO-20260525-REJECT",
        expected_source_state="DEV_COMPLETE",
    )
    assert validated["next_handoff_target"] == "orchestrator_os"
    assert validated["review_result"] == "rejected"

    wrong_owner = dict(packet)
    wrong_owner["owner_profile"] = "dev_os"
    with pytest.raises(ValueError, match="owner_profile"):
        validate_auditos_rejection_packet(
            wrong_owner,
            expected_production_order_id="PO-20260525-REJECT",
            expected_source_state="DEV_COMPLETE",
        )

def test_default_rejection_packet_validation():
    packet = default_rejection_packet("PO-20260525-DEFAULT-REJECT")

    validated = validate_default_rejection_packet(
        packet,
        expected_production_order_id="PO-20260525-DEFAULT-REJECT",
        expected_source_state="DEFAULT_FINAL_REVIEW",
    )
    assert validated["recommended_route"] == "orchestrator_triage"
    assert validated["next_handoff_target"] == "orchestrator_os"
    assert validated["review_result"] == "rejected"

    wrong_target = dict(packet)
    wrong_target["next_handoff_target"] = "dev_os"
    with pytest.raises(ValueError, match="next_handoff_target"):
        validate_default_rejection_packet(
            wrong_target,
            expected_production_order_id="PO-20260525-DEFAULT-REJECT",
            expected_source_state="DEFAULT_FINAL_REVIEW",
        )


def test_orchestrator_classification_packet_validation():
    dev_packet = orchestrator_classification_packet(
        "PO-20260525-CLASSIFY",
        "implementation_mismatch",
    )
    validated_dev = validate_orchestrator_classification_packet(
        dev_packet,
        expected_production_order_id="PO-20260525-CLASSIFY",
        expected_source_state="ORCHESTRATOR_TRIAGE",
    )
    assert validated_dev["route_target"] == "DEV_REWORK"
    assert validated_dev["next_handoff_target"] == "dev_os"

    spec_packet = orchestrator_classification_packet(
        "PO-20260525-CLASSIFY-SPEC",
        "spec_or_design_mismatch",
    )
    validated_spec = validate_orchestrator_classification_packet(
        spec_packet,
        expected_production_order_id="PO-20260525-CLASSIFY-SPEC",
        expected_source_state="ORCHESTRATOR_TRIAGE",
    )
    assert validated_spec["route_target"] == "SPEC_REWORK"
    assert validated_spec["next_handoff_target"] == "architect_os"

    wrong_route = dict(dev_packet)
    wrong_route["route_target"] = "SPEC_REWORK"
    with pytest.raises(ValueError, match="route_target"):
        validate_orchestrator_classification_packet(
            wrong_route,
            expected_production_order_id="PO-20260525-CLASSIFY",
            expected_source_state="ORCHESTRATOR_TRIAGE",
        )


def test_invalid_state_transition_empty():
    """Empty from_state or to_state raises ValueError."""
    with pytest.raises(ValueError, match="from_state is required"):
        validate_state_transition("", "ACTION_APPROVED", "hermes")
    with pytest.raises(ValueError, match="to_state is required"):
        validate_state_transition("BRIEF_DRAFTED", "", "hermes")
    with pytest.raises(ValueError, match="calling_profile is required"):
        validate_state_transition("BRIEF_DRAFTED", "ACTION_APPROVED", "")


# ---------------------------------------------------------------------------
# State Transition Execution
# ---------------------------------------------------------------------------


def test_transition_state_execution(conn, sample_brief):
    """Transition updates card state, PO object, and logs event."""
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )

    new_state = transition_state(
        conn, po, "ORCHESTRATOR_TRIAGE", "orchestrator_os",
        next_action="dispatch_orchestrator",
    )
    assert new_state == "ORCHESTRATOR_TRIAGE"
    assert po.current_state == "ORCHESTRATOR_TRIAGE"
    assert len(po.stage_history) == 3  # 2 from creation + 1 from transition

    # Parent card state updated
    parent = kb.get_task(conn, po.parent_kanban_card_id)
    assert parent.current_state == "ORCHESTRATOR_TRIAGE"


def test_transition_state_rejects_invalid(conn, sample_brief):
    """Invalid transition via transition_state raises and doesn't mutate."""
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )
    with pytest.raises(StateTransitionError):
        transition_state(conn, po, "DEV_IMPLEMENTING", "dev_os")

    # State should NOT have changed
    assert po.current_state == WORKFLOW_INITIAL_STATE


# ---------------------------------------------------------------------------
# Trigger Phrase Detection
# ---------------------------------------------------------------------------


def test_trigger_phrase_detection():
    """Recognizes all 4 approval phrases."""
    assert detect_approval_phrase("Action approved for this brief.") == "approved"
    assert detect_approval_phrase("Approved. Execute this brief.") == "approved"
    assert detect_approval_phrase("Send this into the production workflow.") == "approved"
    assert detect_approval_phrase("Proceed with this approved scope.") == "approved"


def test_non_trigger_phrases_rejected():
    """looks good, go ahead, approved (bare) return None."""
    assert detect_approval_phrase("looks good") is None
    assert detect_approval_phrase("go ahead") is None
    assert detect_approval_phrase("sounds right") is None
    assert detect_approval_phrase("approved") is None
    assert detect_approval_phrase("") is None
    assert detect_approval_phrase("this is not an approval") is None


def test_trigger_phrase_case_insensitive():
    """Detection is case-insensitive."""
    assert detect_approval_phrase("ACTION APPROVED FOR THIS BRIEF.") == "approved"
    assert detect_approval_phrase("action Approved for this brief!!") == "approved"


# ---------------------------------------------------------------------------
# Brief Validation
# ---------------------------------------------------------------------------


def test_approval_brief_validated(conn, sample_brief):
    """Valid brief passes validation."""
    missing = validate_brief(sample_brief)
    assert missing == [], f"Unexpected missing fields: {missing}"


def test_approval_brief_snake_case_aliases_validated(snake_case_brief):
    """Snake_case aliases satisfy required brief fields."""
    missing = validate_brief(snake_case_brief)
    assert missing == [], f"Unexpected missing fields: {missing}"


def test_cli_create_with_human_label_brief_keys_succeeds(kanban_home, tmp_path, sample_brief, capsys):
    """CLI create accepts canonical human-label brief keys."""
    brief_path = tmp_path / "human-brief.json"
    brief_path.write_text(json.dumps(sample_brief), encoding="utf-8")

    rc = _cmd_production_order(argparse.Namespace(
        po_action="create",
        brief_path=str(brief_path),
        board=None,
        idempotency_key=None,
        json=True,
    ))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["production_order_id"].startswith("PO-")
    assert len(payload["child_card_ids"]) == 6


def test_cli_create_with_snake_case_brief_keys_succeeds(kanban_home, tmp_path, snake_case_brief, capsys):
    """CLI create accepts snake_case brief aliases."""
    brief_path = tmp_path / "snake-brief.json"
    brief_path.write_text(json.dumps(snake_case_brief), encoding="utf-8")

    rc = _cmd_production_order(argparse.Namespace(
        po_action="create",
        brief_path=str(brief_path),
        board=None,
        idempotency_key=None,
        json=True,
    ))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["production_order_id"].startswith("PO-")
    assert len(payload["child_card_ids"]) == 6


def test_missing_brief_validation():
    """Empty brief raises validation with all fields missing."""
    missing = validate_brief({})
    # All required fields should be missing
    assert len(missing) == 9, f"Expected 9 missing fields, got {len(missing)}: {missing}"


def test_partial_brief_rejected():
    """Brief missing 'scope' is rejected."""
    brief = {
        "objective": "Do something",
        "target repo or workspace": "repo",
        # missing: scope
        "out of scope": "N/A",
        "acceptance criteria": "Works",
        "stop conditions": "None",
        "approval boundaries": "None",
        "constraints": "None",
        "expected output": "Something",
    }
    missing = validate_brief(brief)
    assert "scope" in missing


def test_scope_expansion_detected():
    """Validate brief recognizes missing fields."""
    brief = {
        "objective": "Task",
        "target repo or workspace": "repo",
        "scope": "In scope",
        "out of scope": "Out of scope",
        "acceptance criteria": "Criteria",
        "stop conditions": "None",
        "approval boundaries": "None",
        "constraints": "None",
        "expected output": "Output",
    }
    missing = validate_brief(brief)
    assert missing == []


# ---------------------------------------------------------------------------
# Event Logging
# ---------------------------------------------------------------------------


def test_event_log_creation(conn, sample_brief):
    """Events written with correct production_order_id and event_type."""
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )
    child_ids = create_production_kanban_graph(conn, po)

    events = conn.execute(
        "SELECT * FROM production_order_events "
        "WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()

    assert len(events) >= 2
    assert events[0]["event_type"] == "brief_approved"
    assert events[0]["production_order_id"] == po.production_order_id
    assert events[1]["event_type"] == "production_order_created"

    # Check kanban_graph_created event exists
    graph_events = [e for e in events if e["event_type"] == "kanban_graph_created"]
    assert len(graph_events) == 1


def test_event_log_schema_enforced(conn):
    """Event log with missing production_order_id raises."""
    with pytest.raises(ValueError, match="production_order_id is required"):
        log_workflow_event(conn, "", "state_transitioned")
    with pytest.raises(ValueError, match="event_type is required"):
        log_workflow_event(conn, "PO-1", "")


def test_production_order_events_table_exists(conn):
    """production_order_events table is created."""
    tables = [
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    assert "production_order_events" in tables


def test_dispatch_event_logging_and_ordering(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    target_card_id = po.child_kanban_card_ids[2]

    first_id = log_dispatch_event(
        conn,
        production_order_id=po.production_order_id,
        event_type="dispatch_planned",
        from_state=po.current_state,
        to_state="DEV_COMPLETE",
        owner_profile=po.current_owner_profile,
        target_profile="dev_os",
        kanban_card_id=target_card_id,
        result="envelope_created",
        next_action="await_manual_dispatch",
    )
    second_id = log_dispatch_event(
        conn,
        production_order_id=po.production_order_id,
        event_type="dispatch_started",
        from_state=po.current_state,
        owner_profile=po.current_owner_profile,
        target_profile="dev_os",
        kanban_card_id=target_card_id,
        packet_id="pkt-dev-1",
        result="manual_handoff_started",
        next_action="await_profile_result",
    )

    events = list_dispatch_events(conn, po.production_order_id)

    assert [event["event_type"] for event in events[-2:]] == [
        "dispatch_planned",
        "dispatch_started",
    ]
    assert events[-2]["id"] == first_id
    assert events[-1]["id"] == second_id
    assert events[-1]["packet_id"] == "pkt-dev-1"
    assert events[-1]["target_profile"] == "dev_os"


def test_dispatch_event_payload_includes_required_fields(conn, sample_brief):
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )
    target_card_id = po.child_kanban_card_ids[0]

    event_id = log_dispatch_event(
        conn,
        production_order_id=po.production_order_id,
        event_type="dispatch_handoff_created",
        from_state=po.current_state,
        to_state="ARCHITECT_SPEC",
        owner_profile=po.current_owner_profile,
        target_profile="orchestrator_os",
        kanban_card_id=target_card_id,
        packet_id="handoff-123",
        result="manual_fallback_created",
        error=None,
        next_action="copy_prompt_to_profile",
    )
    row = conn.execute(
        "SELECT * FROM production_order_events WHERE id = ?",
        (event_id,),
    ).fetchone()

    payload = dispatch_event_to_dict(row)
    assert set(payload) == {
        "id",
        "timestamp",
        "production_order_id",
        "event_type",
        "from_state",
        "to_state",
        "owner_profile",
        "target_profile",
        "kanban_card_id",
        "packet_id",
        "result",
        "error",
        "next_action",
    }
    assert payload["production_order_id"] == po.production_order_id
    assert payload["from_state"] == po.current_state
    assert payload["to_state"] == "ARCHITECT_SPEC"
    assert payload["owner_profile"] == "orchestrator_os"
    assert payload["target_profile"] == "orchestrator_os"
    assert payload["kanban_card_id"] == target_card_id
    assert payload["packet_id"] == "handoff-123"
    assert payload["result"] == "manual_fallback_created"
    assert payload["error"] is None
    assert payload["next_action"] == "copy_prompt_to_profile"
    assert isinstance(payload["timestamp"], int)


def test_invalid_dispatch_event_type_is_rejected(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)

    with pytest.raises(DispatchManifestError, match="Unsupported dispatch event type"):
        log_dispatch_event(
            conn,
            production_order_id=po.production_order_id,
            event_type="handoff_created",
            from_state=po.current_state,
            owner_profile=po.current_owner_profile,
            target_profile="dev_os",
            kanban_card_id=po.child_kanban_card_ids[2],
        )


def test_dispatch_event_logging_is_observability_only(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    before_reconstructed = [
        order for order in list_production_orders(conn)
        if order.production_order_id == po.production_order_id
    ][0]
    before_state = before_reconstructed.current_state
    before_owner = before_reconstructed.current_owner_profile
    before_history = list(before_reconstructed.stage_history)
    before_task_rows = conn.execute(
        "SELECT id, body, current_state, status FROM tasks WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    before_event_types = _po_event_types(conn, po.production_order_id)

    log_dispatch_event(
        conn,
        production_order_id=po.production_order_id,
        event_type="dispatch_blocked",
        from_state=po.current_state,
        owner_profile=po.current_owner_profile,
        target_profile="dev_os",
        kanban_card_id=po.child_kanban_card_ids[2],
        error="waiting_for_manual_profile_execution",
        next_action="jarren_or_runtime_resume_dispatch",
    )

    refreshed = [
        order for order in list_production_orders(conn)
        if order.production_order_id == po.production_order_id
    ][0]
    after_task_rows = conn.execute(
        "SELECT id, body, current_state, status FROM tasks WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    after_event_types = _po_event_types(conn, po.production_order_id)

    assert refreshed.current_state == before_state
    assert refreshed.current_owner_profile == before_owner
    assert refreshed.stage_history == before_history
    assert after_task_rows == before_task_rows
    assert after_event_types[:-1] == before_event_types
    assert after_event_types[-1] == "dispatch_blocked"
    assert "state_transitioned" in before_event_types
    assert "handoff_created" in before_event_types


def test_production_order_events_table_supports_dispatch_columns(conn):
    cols = {
        row["name"] for row in conn.execute("PRAGMA table_info(production_order_events)").fetchall()
    }
    assert "target_profile" in cols
    assert "packet_id" in cols


# ---------------------------------------------------------------------------
# Handoff Packet
# ---------------------------------------------------------------------------


def test_handoff_packet_created(conn, sample_brief):
    """OrchestratorOS handoff packet contains all required fields."""
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )
    child_ids = create_production_kanban_graph(conn, po)

    handoff = create_orchestrator_handoff(
        po, scope=sample_brief["scope"], out_of_scope=sample_brief["out of scope"]
    )
    required_fields = [
        "production_order_id", "from_profile", "to_profile",
        "current_state", "requested_next_state", "objective",
        "context", "source_truth", "scope", "out_of_scope",
        "inputs", "expected_output", "acceptance_criteria",
        "stop_conditions", "approval_required_before", "evidence_required",
    ]
    for field in required_fields:
        assert field in handoff, f"Missing handoff field: {field}"
        assert handoff[field], f"Empty handoff field: {field}"

    assert handoff["from_profile"] == "hermes"
    assert handoff["to_profile"] == "orchestrator_os"
    assert handoff["current_state"] == WORKFLOW_INITIAL_STATE
    assert handoff["requested_next_state"] == "ORCHESTRATOR_TRIAGE"


def test_handoff_packet_frozen(conn, sample_brief):
    """OrchestratorOS card body contains valid handoff packet."""
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )
    child_ids = create_production_kanban_graph(conn, po)
    handoff = create_orchestrator_handoff(po)
    freeze_handoff_on_card(conn, child_ids[0], handoff)

    card = kb.get_task(conn, child_ids[0])
    assert card is not None
    assert card.body is not None
    assert "--- HANDOFF PACKET ---" in card.body
    assert po.production_order_id in card.body
    assert "orchestrator_os" in card.body


# ---------------------------------------------------------------------------
# Full Bridge End-to-End
# ---------------------------------------------------------------------------


def test_full_bridge_smoke(conn, sample_brief):
    """Full bridge creates PO, graph, handoff, transitions to ORCHESTRATOR_TRIAGE."""
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )

    # Check PO created
    assert re.match(r"^PO-\d{8}-[0-9a-z]{4}$", po.production_order_id)
    assert po.current_state == "ORCHESTRATOR_TRIAGE"
    assert po.current_owner_profile == "orchestrator_os"

    # Check parent card
    parent = kb.get_task(conn, po.parent_kanban_card_id)
    assert parent is not None
    assert parent.current_state == "ORCHESTRATOR_TRIAGE"

    # Check child cards
    assert len(po.child_kanban_card_ids) == 6

    # Check OrchestratorOS card has handoff packet
    os_card = kb.get_task(conn, po.child_kanban_card_ids[0])
    assert os_card is not None
    assert os_card.body is not None
    assert "--- HANDOFF PACKET ---" in os_card.body

    # Check events exist
    events = conn.execute(
        "SELECT event_type FROM production_order_events "
        "WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    event_types = [e["event_type"] for e in events]
    assert "brief_approved" in event_types
    assert "production_order_created" in event_types
    assert "kanban_graph_created" in event_types
    assert "handoff_created" in event_types
    assert "state_transitioned" in event_types


def test_orchestrator_triage_bridge_moves_existing_order_to_architect_spec(conn, sample_brief):
    """Slice 5 advances an existing PO without duplicating the Kanban graph."""
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )
    original_child_ids = list(po.child_kanban_card_ids)

    triaged = run_orchestrator_triage_bridge(
        conn,
        production_order_id=po.production_order_id,
    )

    assert triaged.current_state == "ARCHITECT_SPEC"
    assert triaged.current_owner_profile == "architect_os"
    assert triaged.child_kanban_card_ids == original_child_ids
    assert len(triaged.child_kanban_card_ids) == 6
    assert triaged.repo_or_workspace == sample_brief["target repo or workspace"]
    assert any(
        s.from_state == "ORCHESTRATOR_TRIAGE" and s.to_state == "ARCHITECT_SPEC"
        for s in triaged.stage_history
    )

    link_count = conn.execute(
        "SELECT COUNT(*) AS n FROM task_links WHERE parent_id = ?",
        (triaged.parent_kanban_card_id,),
    ).fetchone()["n"]
    assert link_count == 6

    architect_card = kb.get_task(conn, original_child_ids[1])
    assert architect_card is not None
    assert architect_card.status == "ready"
    assert architect_card.body is not None
    assert "--- HANDOFF PACKET ---" in architect_card.body
    assert '"to_profile": "architect_os"' in architect_card.body

    events = conn.execute(
        "SELECT event_type, from_state, to_state, owner_profile FROM production_order_events "
        "WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    assert [event["event_type"] for event in events] == [
        "brief_approved",
        "production_order_created",
        "kanban_graph_created",
        "state_transitioned",
        "handoff_created",
        "orchestrator_triage_completed",
        "handoff_created",
    ]
    assert events[-2]["from_state"] == "ORCHESTRATOR_TRIAGE"
    assert events[-2]["to_state"] == "ARCHITECT_SPEC"
    assert events[-2]["owner_profile"] == "orchestrator_os"


def test_orchestrator_triage_bridge_rejects_missing_graph(conn, sample_brief):
    """Slice 5 requires the already-created 6-card Kanban graph."""
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )
    transition_state(conn, po, "ORCHESTRATOR_TRIAGE", "orchestrator_os")

    with pytest.raises(ValueError, match="must have 6 child cards"):
        run_orchestrator_triage_bridge(
            conn,
            production_order_id=po.production_order_id,
        )


def test_architect_handoff_uses_frozen_brief_metadata(conn, sample_brief):
    """ArchitectOS handoff packet is derived from the frozen PO brief."""
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )

    handoff = create_architect_handoff(po)

    assert handoff["from_profile"] == "orchestrator_os"
    assert handoff["to_profile"] == "architect_os"
    assert handoff["current_state"] == "ORCHESTRATOR_TRIAGE"
    assert handoff["requested_next_state"] == "ARCHITECT_SPEC"
    assert handoff["scope"] == sample_brief["scope"]
    assert handoff["out_of_scope"] == sample_brief["out of scope"]
    assert sample_brief["target repo or workspace"] in handoff["inputs"]


def test_validate_architect_spec_packet_accepts_minimal_packet():
    """Slice 6 packet accepts explicit canonical workflow source truth."""
    packet = architect_spec_packet("PO-20260525-test")

    validated = validate_architect_spec_packet(
        packet,
        expected_production_order_id="PO-20260525-test",
    )

    assert validated is packet


def test_validate_architect_spec_packet_rejects_missing_required_field():
    """Slice 6 requires a complete ArchitectOS packet; runtime cannot invent it."""
    packet = architect_spec_packet("PO-20260525-test")
    packet.pop("devos_task")

    with pytest.raises(ValueError, match="devos_task"):
        validate_architect_spec_packet(
            packet,
            expected_production_order_id="PO-20260525-test",
        )


def test_validate_architect_spec_packet_rejects_wrong_source_truth():
    """ArchitectOS packet must cite the canonical workspace workflow spec."""
    packet = architect_spec_packet("PO-20260525-test")
    packet["source_truth"] = [
        "specs/architecture/workflows/hermes-production-workflow-v1.md"
    ]

    with pytest.raises(ValueError, match="canonical workflow spec"):
        validate_architect_spec_packet(
            packet,
            expected_production_order_id="PO-20260525-test",
        )


def test_create_devos_handoff_from_architect_packet(conn, sample_brief):
    """DevOS handoff packet preserves ArchitectOS packet boundaries."""
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )
    packet = architect_spec_packet(po.production_order_id)

    handoff = create_devos_handoff(po, packet)

    assert handoff["production_order_id"] == po.production_order_id
    assert handoff["current_state"] == "ARCHITECT_SPEC"
    assert handoff["requested_next_state"] == "ARCHITECT_READY_FOR_DEV"
    assert handoff["from_profile"] == "architect_os"
    assert handoff["to_profile"] == "dev_os"
    assert handoff["devos_task"] == packet["devos_task"]
    assert handoff["allowed_files_or_areas"] == packet["files_or_areas_allowed"]
    assert handoff["approval_boundaries"] == packet["approval_boundaries"]
    assert handoff["artifact_references"] == packet["artifact_references"]


def test_architect_spec_bridge_moves_existing_order_to_ready_for_dev(conn, sample_brief):
    """Slice 6 advances an existing ARCHITECT_SPEC PO and preserves its graph."""
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )
    po = run_orchestrator_triage_bridge(conn, production_order_id=po.production_order_id)
    original_parent_id = po.parent_kanban_card_id
    original_child_ids = list(po.child_kanban_card_ids)

    completed = run_architect_spec_bridge(
        conn,
        production_order_id=po.production_order_id,
        architect_packet=architect_spec_packet(po.production_order_id),
    )

    assert completed.current_state == "ARCHITECT_READY_FOR_DEV"
    assert completed.current_owner_profile == "dev_os"
    assert completed.parent_kanban_card_id == original_parent_id
    assert completed.child_kanban_card_ids == original_child_ids
    assert len(completed.child_kanban_card_ids) == 6
    assert completed.repo_or_workspace == sample_brief["target repo or workspace"]
    assert any(
        s.from_state == "ARCHITECT_SPEC" and s.to_state == "ARCHITECT_READY_FOR_DEV"
        for s in completed.stage_history
    )

    link_count = conn.execute(
        "SELECT COUNT(*) AS n FROM task_links WHERE parent_id = ?",
        (completed.parent_kanban_card_id,),
    ).fetchone()["n"]
    assert link_count == 6

    devos_card = kb.get_task(conn, original_child_ids[2])
    assert devos_card is not None
    assert devos_card.status == "ready"
    assert devos_card.body is not None
    assert "--- HANDOFF PACKET ---" in devos_card.body
    assert '\"to_profile\": \"dev_os\"' in devos_card.body
    assert "Prepare for implementation from the approved spec" in devos_card.body

    events = conn.execute(
        "SELECT event_type, from_state, to_state, owner_profile FROM production_order_events "
        "WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    assert [event["event_type"] for event in events] == [
        "brief_approved",
        "production_order_created",
        "kanban_graph_created",
        "state_transitioned",
        "handoff_created",
        "orchestrator_triage_completed",
        "handoff_created",
        "architect_spec_completed",
        "handoff_created",
    ]
    assert events[-2]["from_state"] == "ARCHITECT_SPEC"
    assert events[-2]["to_state"] == "ARCHITECT_READY_FOR_DEV"
    assert events[-2]["owner_profile"] == "architect_os"


def test_architect_spec_bridge_rejects_wrong_state(conn, sample_brief):
    """Slice 6 refuses to skip OrchestratorOS triage / ArchitectOS spec ownership."""
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )

    with pytest.raises(StateTransitionError, match="expected 'ARCHITECT_SPEC'"):
        run_architect_spec_bridge(
            conn,
            production_order_id=po.production_order_id,
            architect_packet=architect_spec_packet(po.production_order_id),
        )


def test_cli_architect_complete_json(capsys, conn, tmp_path, sample_brief):
    """CLI architect-complete consumes a JSON spec packet and emits strict JSON."""
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )
    po = run_orchestrator_triage_bridge(conn, production_order_id=po.production_order_id)
    packet_path = tmp_path / "architect-packet.json"
    packet_path.write_text(json.dumps(architect_spec_packet(po.production_order_id)), encoding="utf-8")

    rc = _cmd_production_order(argparse.Namespace(
        po_action="architect-complete",
        production_order_id=po.production_order_id,
        board=None,
        spec_file=str(packet_path),
        json=True,
    ))

    captured = capsys.readouterr()
    assert rc == 0
    parsed = json.loads(captured.out)
    assert parsed["production_order_id"] == po.production_order_id
    assert parsed["current_state"] == "ARCHITECT_READY_FOR_DEV"
    assert parsed["current_owner_profile"] == "dev_os"
    assert parsed["child_card_ids"] == po.child_kanban_card_ids
    assert "Slice 6" not in captured.out


def test_validate_devos_build_packet_accepts_minimal_packet():
    packet = devos_build_packet("PO-20260525-test")

    validated = validate_devos_build_packet(
        packet,
        expected_production_order_id="PO-20260525-test",
    )

    assert validated is packet


def test_validate_devos_build_packet_accepts_stage_result_and_implementation_artifacts():
    packet = devos_build_packet("PO-20260525-test")
    packet.pop("result_type")
    packet.pop("files_changed")
    packet["stage_result"] = "build_complete"
    packet["implementation_artifacts"] = ["runtime-bridge.patch"]

    validated = validate_devos_build_packet(
        packet,
        expected_production_order_id="PO-20260525-test",
    )

    assert validated is packet



def test_dispatch_manifest_covers_supported_states_and_routes(conn, sample_brief):
    def expect_manifest(manifest, *, state, owner, profile, card_id, task_type, required_input_packet, expected_result_packet, bridge_function):
        assert manifest.current_state == state
        assert manifest.current_owner_profile == owner
        assert manifest.target_profile == profile
        assert manifest.target_child_card_id == card_id
        assert manifest.task_type == task_type
        assert manifest.required_input_packet == required_input_packet
        assert manifest.expected_result_packet == expected_result_packet
        assert manifest.bridge_function == bridge_function
        assert manifest.stop_conditions
        assert manifest.manual_fallback["enabled"] is True
        assert manifest.manual_fallback["task_prompt_template"] is None
        assert manifest.manual_fallback["target_profile"] == profile
        assert manifest.manual_fallback["target_child_card_id"] == card_id
        assert manifest.manual_fallback["task_type"] == task_type
        assert manifest.manual_fallback["required_input_packet"] == required_input_packet
        assert manifest.manual_fallback["expected_result_packet"] == expected_result_packet
        assert manifest.manual_fallback["bridge_function"] == bridge_function
        assert manifest.manual_fallback["stop_conditions"] == list(manifest.stop_conditions)
        assert manifest.manual_fallback["source_truth"] == WORKFLOW_SPEC_SOURCE
        assert manifest.manual_fallback["current_state"] == state
        assert manifest.to_dict()["stop_conditions"] == list(manifest.stop_conditions)

    initial = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )
    expect_manifest(
        dispatch_manifest_for_order(initial),
        state="ORCHESTRATOR_TRIAGE",
        owner="orchestrator_os",
        profile="orchestrator_os",
        card_id=initial.child_kanban_card_ids[0],
        task_type="orchestrator_triage",
        required_input_packet="orchestrator_handoff_packet",
        expected_result_packet="architect_handoff_packet",
        bridge_function="run_orchestrator_triage_bridge",
    )

    architect_spec = run_orchestrator_triage_bridge(
        conn,
        production_order_id=initial.production_order_id,
    )
    expect_manifest(
        dispatch_manifest_for_order(architect_spec),
        state="ARCHITECT_SPEC",
        owner="architect_os",
        profile="architect_os",
        card_id=architect_spec.child_kanban_card_ids[1],
        task_type="architect_spec",
        required_input_packet="architect_handoff_packet",
        expected_result_packet="architect_spec_packet",
        bridge_function="run_architect_spec_bridge",
    )

    ready_for_dev = create_ready_for_dev_order(conn, sample_brief)
    expect_manifest(
        dispatch_manifest_for_order(ready_for_dev),
        state="ARCHITECT_READY_FOR_DEV",
        owner="dev_os",
        profile="dev_os",
        card_id=ready_for_dev.child_kanban_card_ids[2],
        task_type="dev_build",
        required_input_packet="devos_handoff_packet",
        expected_result_packet="devos_build_packet",
        bridge_function="run_devos_complete_bridge",
    )

    dev_implementing = replace(
        ready_for_dev,
        current_state="DEV_IMPLEMENTING",
        current_owner_profile="dev_os",
    )
    expect_manifest(
        dispatch_manifest_for_order(dev_implementing),
        state="DEV_IMPLEMENTING",
        owner="dev_os",
        profile="dev_os",
        card_id=dev_implementing.child_kanban_card_ids[2],
        task_type="dev_build",
        required_input_packet="devos_handoff_packet",
        expected_result_packet="devos_build_packet",
        bridge_function="run_devos_complete_bridge",
    )

    dev_complete = create_dev_complete_order(conn, sample_brief)
    expect_manifest(
        dispatch_manifest_for_order(dev_complete),
        state="DEV_COMPLETE",
        owner="audit_os",
        profile="audit_os",
        card_id=dev_complete.child_kanban_card_ids[3],
        task_type="audit_review",
        required_input_packet="auditos_handoff_packet",
        expected_result_packet="auditos_review_packet",
        bridge_function="run_auditos_review_complete_bridge",
    )

    audit_review = replace(
        dev_complete,
        current_state="AUDIT_REVIEW",
        current_owner_profile="audit_os",
    )
    expect_manifest(
        dispatch_manifest_for_order(audit_review),
        state="AUDIT_REVIEW",
        owner="audit_os",
        profile="audit_os",
        card_id=audit_review.child_kanban_card_ids[3],
        task_type="audit_review",
        required_input_packet="auditos_handoff_packet",
        expected_result_packet="auditos_review_packet",
        bridge_function="run_auditos_review_complete_bridge",
    )

    audit_passed = create_audit_passed_order(conn, sample_brief)
    expect_manifest(
        dispatch_manifest_for_order(audit_passed),
        state="AUDIT_PASSED",
        owner="architect_os",
        profile="architect_os",
        card_id=audit_passed.child_kanban_card_ids[4],
        task_type="architect_reconcile",
        required_input_packet="architect_reconcile_handoff_packet",
        expected_result_packet="architect_reconcile_packet",
        bridge_function="run_architect_reconcile_bridge",
    )

    architect_reconcile = replace(
        audit_passed,
        current_state="ARCHITECT_RECONCILE",
        current_owner_profile="architect_os",
    )
    expect_manifest(
        dispatch_manifest_for_order(architect_reconcile),
        state="ARCHITECT_RECONCILE",
        owner="architect_os",
        profile="architect_os",
        card_id=architect_reconcile.child_kanban_card_ids[4],
        task_type="architect_reconcile",
        required_input_packet="architect_reconcile_handoff_packet",
        expected_result_packet="architect_reconcile_packet",
        bridge_function="run_architect_reconcile_bridge",
    )

    architect_accepted = create_architect_accepted_order(conn, sample_brief)
    expect_manifest(
        dispatch_manifest_for_order(architect_accepted),
        state="ARCHITECT_ACCEPTED",
        owner="default",
        profile="default",
        card_id=architect_accepted.child_kanban_card_ids[5],
        task_type="default_final_review",
        required_input_packet="default_final_review_handoff_packet",
        expected_result_packet="default_final_review_packet",
        bridge_function="run_default_final_review_bridge",
    )

    default_final_review = create_default_final_review_order(conn, sample_brief)
    expect_manifest(
        dispatch_manifest_for_order(default_final_review),
        state="DEFAULT_FINAL_REVIEW",
        owner="default",
        profile="default",
        card_id=default_final_review.child_kanban_card_ids[5],
        task_type="default_final_review",
        required_input_packet="default_final_review_handoff_packet",
        expected_result_packet="default_final_review_packet",
        bridge_function="run_default_final_review_bridge",
    )

    default_rejected = run_default_final_review_reject_bridge(
        conn,
        production_order_id=default_final_review.production_order_id,
        rejection_packet=default_rejection_packet(default_final_review.production_order_id),
    )
    expect_manifest(
        dispatch_manifest_for_order(default_rejected),
        state="DEFAULT_REJECTED",
        owner="orchestrator_os",
        profile="orchestrator_os",
        card_id=default_rejected.child_kanban_card_ids[0],
        task_type="default_rejection_triage",
        required_input_packet="default_rejection_packet",
        expected_result_packet="default_rejection_handoff_packet",
        bridge_function="run_orchestrator_default_rejection_triage_bridge",
    )

    default_triage = run_orchestrator_default_rejection_triage_bridge(
        conn,
        production_order_id=default_rejected.production_order_id,
        rejection_packet=default_rejection_packet(
            default_rejected.production_order_id,
            source_state="DEFAULT_REJECTED",
        ),
    )
    expect_manifest(
        dispatch_manifest_for_order(default_triage),
        state="ORCHESTRATOR_TRIAGE",
        owner="orchestrator_os",
        profile="orchestrator_os",
        card_id=default_triage.child_kanban_card_ids[0],
        task_type="orchestrator_default_rejection_classification",
        required_input_packet="default_rejection_handoff_packet",
        expected_result_packet="orchestrator_classification_packet",
        bridge_function="run_orchestrator_classification_bridge",
    )

    audit_rejected_source = create_dev_complete_order(conn, sample_brief)
    audit_rejected = run_auditos_review_reject_bridge(
        conn,
        production_order_id=audit_rejected_source.production_order_id,
        rejection_packet=audit_rejection_packet(audit_rejected_source.production_order_id),
    )
    expect_manifest(
        dispatch_manifest_for_order(audit_rejected),
        state="AUDIT_REJECTED",
        owner="orchestrator_os",
        profile="orchestrator_os",
        card_id=audit_rejected.child_kanban_card_ids[2],
        task_type="orchestrator_rework",
        required_input_packet="auditos_rejection_packet",
        expected_result_packet="devos_rework_handoff_packet",
        bridge_function="run_orchestrator_rework_bridge",
    )

    dev_rework = run_orchestrator_rework_bridge(
        conn,
        production_order_id=audit_rejected.production_order_id,
        rejection_packet=audit_rejection_packet(audit_rejected.production_order_id, "AUDIT_REJECTED"),
    )
    expect_manifest(
        dispatch_manifest_for_order(dev_rework),
        state="DEV_REWORK",
        owner="dev_os",
        profile="dev_os",
        card_id=dev_rework.child_kanban_card_ids[2],
        task_type="dev_rework",
        required_input_packet="devos_rework_handoff_packet",
        expected_result_packet="devos_build_packet",
        bridge_function="run_devos_rework_complete_bridge",
    )


def test_build_dispatch_manifest_logs_once_without_mutating_cards_and_rejects_unsupported_states(conn, sample_brief):
    po = create_default_final_review_order(conn, sample_brief)
    before_total_changes = conn.total_changes
    before_tasks = conn.execute(
        "SELECT id, current_state, status FROM tasks WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    before_events = conn.execute(
        "SELECT COUNT(*) AS n FROM production_order_events WHERE production_order_id = ?",
        (po.production_order_id,),
    ).fetchone()["n"]

    manifest = build_dispatch_manifest(conn, po.production_order_id)

    after_total_changes = conn.total_changes
    after_tasks = conn.execute(
        "SELECT id, current_state, status FROM tasks WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    after_events = conn.execute(
        "SELECT COUNT(*) AS n FROM production_order_events WHERE production_order_id = ?",
        (po.production_order_id,),
    ).fetchone()["n"]

    assert after_total_changes == before_total_changes + 1
    assert after_tasks == before_tasks
    assert after_events == before_events + 1
    assert manifest.production_order_id == po.production_order_id
    assert manifest.dispatch_attempt >= 1
    assert manifest.idempotency_key == (
        f"dispatch:{po.production_order_id}:{po.current_state}:{manifest.target_profile}:"
        f"{manifest.target_child_card_id}:{manifest.task_type}:attempt-{manifest.dispatch_attempt}"
    )
    assert manifest.manual_fallback["task_prompt_template"] is None
    assert manifest.to_dict()["stop_conditions"] == list(manifest.stop_conditions)

    repeat_manifest = build_dispatch_manifest(conn, po.production_order_id)
    repeated_events = conn.execute(
        "SELECT COUNT(*) AS n FROM production_order_events WHERE production_order_id = ?",
        (po.production_order_id,),
    ).fetchone()["n"]
    assert repeat_manifest == manifest
    assert repeated_events == after_events

    with pytest.raises(DispatchManifestError, match="SPEC_REWORK"):
        dispatch_manifest_for_order(
            replace(
                po,
                current_state="SPEC_REWORK",
                current_owner_profile="architect_os",
            )
        )


def test_build_profile_task_envelope_happy_path_fields_and_json_serializable(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)

    envelope = build_profile_task_envelope(conn, po.production_order_id)
    serialized = envelope.to_dict()

    assert serialized["production_order_id"] == po.production_order_id
    assert serialized["parent_kanban_card_id"] == po.parent_kanban_card_id
    assert serialized["child_kanban_card_id"] == po.child_kanban_card_ids[2]
    assert serialized["target_profile"] == "dev_os"
    assert serialized["source_state"] == "ARCHITECT_READY_FOR_DEV"
    assert serialized["expected_next_state"] == "DEV_COMPLETE"
    assert serialized["objective"] == sample_brief["objective"]
    assert WORKFLOW_SPEC_SOURCE in serialized["source_truth"]
    assert serialized["frozen_brief"] == po.source_brief
    assert serialized["input_packet"]["packet_type"] == "devos_handoff_packet"
    assert serialized["input_packet"]["repo_or_workspace"] == sample_brief["target repo or workspace"]
    assert serialized["expected_output_packet"]["packet_type"] == "devos_build_packet"
    assert serialized["acceptance_criteria"] == [sample_brief["acceptance criteria"]]
    assert sample_brief["stop conditions"] in serialized["stop_conditions"]
    assert sample_brief["approval boundaries"] in serialized["approval_boundaries"]
    assert serialized["allowed_files_or_scope"] == sample_brief["scope"]
    assert serialized["repo_or_workspace"] == sample_brief["target repo or workspace"]
    json.dumps(serialized)


def test_build_profile_task_envelope_rejects_deferred_spec_rework_state(conn, sample_brief):
    ready_for_dev = create_ready_for_dev_order(conn, sample_brief)

    with pytest.raises(DispatchManifestError, match="SPEC_REWORK"):
        dispatch_manifest_for_order(
            replace(
                ready_for_dev,
                current_state="SPEC_REWORK",
                current_owner_profile="architect_os",
            )
        )


def test_build_profile_task_envelope_is_idempotent_and_does_not_mutate_cards(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    before_total_changes = conn.total_changes
    before_tasks = conn.execute(
        "SELECT id, current_state, status, body FROM tasks WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    before_events = conn.execute(
        "SELECT COUNT(*) AS n FROM production_order_events WHERE production_order_id = ?",
        (po.production_order_id,),
    ).fetchone()["n"]

    envelope = build_profile_task_envelope(conn, po.production_order_id)

    after_total_changes = conn.total_changes
    after_tasks = conn.execute(
        "SELECT id, current_state, status, body FROM tasks WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    after_events = conn.execute(
        "SELECT COUNT(*) AS n FROM production_order_events WHERE production_order_id = ?",
        (po.production_order_id,),
    ).fetchone()["n"]

    assert after_total_changes == before_total_changes + 1
    assert after_tasks == before_tasks
    assert after_events == before_events + 1
    assert envelope.production_order_id == po.production_order_id

    repeat_envelope = build_profile_task_envelope(conn, po.production_order_id)
    repeated_events = conn.execute(
        "SELECT COUNT(*) AS n FROM production_order_events WHERE production_order_id = ?",
        (po.production_order_id,),
    ).fetchone()["n"]
    assert repeat_envelope == envelope
    assert repeated_events == after_events


@pytest.mark.parametrize(
    ("factory", "expected_profile", "expected_bridge", "expected_packet", "expected_next_state"),
    [
        (
            create_architect_spec_order,
            "architect_os",
            "run_architect_spec_bridge",
            "architect_spec_packet",
            "ARCHITECT_READY_FOR_DEV",
        ),
        (
            create_ready_for_dev_order,
            "dev_os",
            "run_devos_complete_bridge",
            "devos_build_packet",
            "DEV_COMPLETE",
        ),
        (
            create_audit_passed_order,
            "architect_os",
            "run_architect_reconcile_bridge",
            "architect_reconcile_packet",
            "ARCHITECT_ACCEPTED",
        ),
        (
            create_dev_complete_order,
            "audit_os",
            "run_auditos_review_complete_bridge",
            "auditos_review_packet",
            "AUDIT_PASSED",
        ),
        (
            create_architect_accepted_order,
            "default",
            "run_default_final_review_bridge",
            "default_final_review_packet",
            "DONE",
        ),
    ],
)
def test_build_manual_fallback_handoff_fields_prompt_and_json(
    conn,
    sample_brief,
    factory,
    expected_profile,
    expected_bridge,
    expected_packet,
    expected_next_state,
):
    po = factory(conn, sample_brief)

    handoff = build_manual_fallback_handoff(conn, po.production_order_id)
    serialized = handoff.to_dict()
    prompt = serialized["copy_paste_prompt"]

    assert serialized["production_order_id"] == po.production_order_id
    assert serialized["source_state"] == po.current_state
    assert serialized["expected_next_state"] == expected_next_state
    assert serialized["target_profile"] == expected_profile
    assert serialized["target_child_card_id"] in po.child_kanban_card_ids
    assert WORKFLOW_SPEC_SOURCE in serialized["source_truth"]
    assert serialized["required_input_packet"]["packet_type"]
    assert serialized["expected_result_packet"]["packet_type"] == expected_packet
    assert serialized["expected_result_packet"]["required_fields"]
    assert serialized["bridge_function"] == expected_bridge
    assert serialized["result_return_action"].startswith(f"Call {expected_bridge}(")
    assert serialized["repo_or_workspace"] == sample_brief["target repo or workspace"]
    assert f"Target profile: {expected_profile}" in prompt
    assert f"Objective: {sample_brief['objective']}" in prompt
    assert f"Production order ID: {po.production_order_id}" in prompt
    assert f"Source state: {po.current_state}" in prompt
    assert f"Expected next state: {expected_next_state}" in prompt
    assert f"Target child card ID: {serialized['target_child_card_id']}" in prompt
    assert f"Repo or workspace: {sample_brief['target repo or workspace']}" in prompt
    assert f"- Scope: {sample_brief['scope']}" in prompt
    assert f"- Out of scope: {sample_brief['out of scope']}" in prompt
    assert "Required input packet:" in prompt
    assert '"packet_type":' in prompt
    assert "Expected result packet requirements:" in prompt
    assert f'"packet_type": "{expected_packet}"' in prompt
    assert "Return exactly one structured result packet JSON object matching this shape:" in prompt
    assert '"production_order_id":' in prompt
    assert '"owner_profile":' in prompt or expected_packet == "architect_spec_packet"
    assert "Acceptance criteria:" in prompt
    assert sample_brief["acceptance criteria"] in prompt
    assert "Stop conditions:" in prompt
    assert sample_brief["stop conditions"] in prompt
    assert "Approval boundaries:" in prompt
    assert sample_brief["approval boundaries"] in prompt
    assert "Do not execute external, destructive, publishing, spending, permission-widening, or secret-requesting actions without explicit approval." in prompt
    assert "Do not change Hermes production-workflow state directly." in prompt
    assert "Do not invoke another profile." in prompt
    assert serialized["result_return_action"] in prompt
    json.dumps(serialized)


def test_build_manual_fallback_handoff_supports_orchestrator_triage(conn, sample_brief):
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )

    handoff = build_manual_fallback_handoff(conn, po.production_order_id)

    assert handoff.target_profile == "orchestrator_os"
    assert handoff.bridge_function == "run_orchestrator_triage_bridge"
    assert handoff.expected_next_state == "ARCHITECT_SPEC"



def test_build_manual_fallback_handoff_is_idempotent_and_does_not_mutate_cards(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    before_total_changes = conn.total_changes
    before_tasks = conn.execute(
        "SELECT id, current_state, status, body FROM tasks WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    before_events = conn.execute(
        "SELECT COUNT(*) AS n FROM production_order_events WHERE production_order_id = ?",
        (po.production_order_id,),
    ).fetchone()["n"]

    handoff = build_manual_fallback_handoff(conn, po.production_order_id)

    after_total_changes = conn.total_changes
    after_tasks = conn.execute(
        "SELECT id, current_state, status, body FROM tasks WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    after_events = conn.execute(
        "SELECT COUNT(*) AS n FROM production_order_events WHERE production_order_id = ?",
        (po.production_order_id,),
    ).fetchone()["n"]

    assert after_total_changes == before_total_changes + 2
    assert after_tasks == before_tasks
    assert after_events == before_events + 2
    assert handoff.production_order_id == po.production_order_id

    repeat_handoff = build_manual_fallback_handoff(conn, po.production_order_id)
    repeated_events = conn.execute(
        "SELECT COUNT(*) AS n FROM production_order_events WHERE production_order_id = ?",
        (po.production_order_id,),
    ).fetchone()["n"]
    assert repeat_handoff == handoff
    assert repeated_events == after_events


@pytest.mark.parametrize(
    ("factory", "packet_factory", "expected_bridge", "expected_profile", "card_index"),
    [
        (
            create_architect_spec_order,
            architect_spec_packet,
            "run_architect_spec_bridge",
            "architect_os",
            1,
        ),
        (
            create_ready_for_dev_order,
            devos_build_packet,
            "run_devos_complete_bridge",
            "dev_os",
            2,
        ),
        (
            create_audit_passed_order,
            architect_reconcile_packet,
            "run_architect_reconcile_bridge",
            "architect_os",
            4,
        ),
    ],
)
def test_ingest_profile_result_packet_accepts_freezes_and_logs_dispatch_event(
    conn,
    sample_brief,
    factory,
    packet_factory,
    expected_bridge,
    expected_profile,
    card_index,
):
    po = factory(conn, sample_brief)
    before = [order for order in list_production_orders(conn) if order.production_order_id == po.production_order_id][0]
    before_event_rows = conn.execute(
        "SELECT event_type, from_state, to_state, owner_profile FROM production_order_events WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    before_task_rows = conn.execute(
        "SELECT id, status, current_state FROM tasks WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()

    packet = packet_factory(po.production_order_id)
    packet["packet_id"] = f"pkt-{card_index}"
    result = ingest_profile_result_packet(conn, po.production_order_id, packet)

    after = [order for order in list_production_orders(conn) if order.production_order_id == po.production_order_id][0]
    target_card = kb.get_task(conn, po.child_kanban_card_ids[card_index])
    assert target_card is not None
    assert target_card.body is not None
    assert "--- RESULT PACKET ---" in target_card.body
    assert f'"packet_id": "{packet["packet_id"]}"' in target_card.body

    assert result == json.loads(json.dumps(result))
    assert result["accepted"] is True
    assert result["production_order_id"] == po.production_order_id
    assert result["source_state"] == before.current_state
    assert result["owner_profile"] == expected_profile
    assert result["target_profile"] == expected_profile
    assert result["child_kanban_card_id"] == po.child_kanban_card_ids[card_index]
    assert result["packet_id"] == packet["packet_id"]
    assert result["bridge_function"] == expected_bridge
    assert result["runtime_action"].startswith(f"{expected_bridge}(")
    assert result["next_action"] == result["runtime_action"]
    assert result["error"] is None

    dispatch_events = list_dispatch_events(conn, production_order_id=po.production_order_id)
    assert dispatch_events[-1]["event_type"] == "packet_validated"
    assert dispatch_events[-1]["result"] == "accepted"
    assert dispatch_events[-1]["packet_id"] == packet["packet_id"]
    assert dispatch_events[-1]["next_action"] == result["runtime_action"]

    after_event_rows = conn.execute(
        "SELECT event_type, from_state, to_state, owner_profile FROM production_order_events WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    assert len(after_event_rows) == len(before_event_rows) + 2
    assert after_event_rows[-1]["event_type"] == "packet_validated"
    assert after.current_state == before.current_state
    assert after.current_owner_profile == before.current_owner_profile
    assert after.stage_history == before.stage_history

    task_rows = conn.execute(
        "SELECT id, status, current_state FROM tasks WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    assert [(row["id"], row["status"], row["current_state"]) for row in task_rows] == [
        (row["id"], row["status"], row["current_state"]) for row in before_task_rows
    ]


def test_ingest_profile_result_packet_rejects_free_text_and_logs_without_freezing(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    target_card = kb.get_task(conn, po.child_kanban_card_ids[2])
    assert target_card is not None
    before_body = target_card.body or ""

    result = ingest_profile_result_packet(
        conn,
        po.production_order_id,
        "looks good ship it",
    )

    refreshed_card = kb.get_task(conn, po.child_kanban_card_ids[2])
    assert refreshed_card is not None
    assert refreshed_card.body == before_body
    assert result["accepted"] is False
    assert result["error"] == "result packet must be a JSON object"
    assert result["bridge_function"] == "run_devos_complete_bridge"
    assert result["runtime_action"].startswith("run_devos_complete_bridge(")
    assert result["next_action"] == "manual_review_rejected_packet"

    dispatch_events = list_dispatch_events(conn, production_order_id=po.production_order_id)
    assert dispatch_events[-1]["event_type"] == "packet_rejected"
    assert dispatch_events[-1]["result"] == "rejected"
    assert dispatch_events[-1]["error"] == "result packet must be a JSON object"


def test_ingest_profile_result_packet_rejects_mismatched_or_unsafe_packet_without_state_transition(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    envelope = build_profile_task_envelope(conn, po.production_order_id)
    packet = devos_build_packet(po.production_order_id)
    packet["packet_id"] = "pkt-reject"
    packet["current_owner_profile"] = "audit_os"

    with pytest.raises(ValueError, match="mutate workflow state directly"):
        validate_profile_result_packet(envelope, packet)

    packet.pop("current_owner_profile")
    packet["production_order_id"] = "PO-wrong"
    before = [order for order in list_production_orders(conn) if order.production_order_id == po.production_order_id][0]
    result = ingest_profile_result_packet(conn, po.production_order_id, packet)
    after = [order for order in list_production_orders(conn) if order.production_order_id == po.production_order_id][0]

    assert result["accepted"] is False
    assert "does not match the active production order" in result["error"]
    assert after.current_state == before.current_state
    assert after.current_owner_profile == before.current_owner_profile
    assert after.stage_history == before.stage_history

    target_card = kb.get_task(conn, po.child_kanban_card_ids[2])
    assert target_card is not None
    assert "pkt-reject" not in (target_card.body or "")


def test_ingest_profile_result_packet_does_not_call_bridge_function_or_pollute_stage_history(conn, sample_brief):
    po = create_architect_spec_order(conn, sample_brief)
    before = [order for order in list_production_orders(conn) if order.production_order_id == po.production_order_id][0]
    before_event_types = [entry["event_type"] for entry in conn.execute(
        "SELECT event_type FROM production_order_events WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()]

    result = ingest_profile_result_packet(
        conn,
        po.production_order_id,
        architect_spec_packet(po.production_order_id),
    )
    after = [order for order in list_production_orders(conn) if order.production_order_id == po.production_order_id][0]
    after_event_types = [entry["event_type"] for entry in conn.execute(
        "SELECT event_type FROM production_order_events WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()]

    assert result["accepted"] is True
    assert after.current_state == before.current_state == "ARCHITECT_SPEC"
    assert after.current_owner_profile == before.current_owner_profile == "architect_os"
    assert after.stage_history == before.stage_history
    assert after_event_types[-1] == "packet_validated"
    assert "architect_spec_completed" not in after_event_types[len(before_event_types):]

    devos_card = kb.get_task(conn, po.child_kanban_card_ids[2])
    assert devos_card is not None
    assert "--- HANDOFF PACKET ---" not in (devos_card.body or "")


def test_manual_fallback_dispatch_events_can_be_logged_without_mutating_cards(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    manifest = build_dispatch_manifest(conn, po.production_order_id)
    envelope = build_profile_task_envelope(conn, po.production_order_id)
    handoff = build_manual_fallback_handoff(conn, po.production_order_id)
    before_card = kb.get_task(conn, manifest.target_child_card_id)
    assert before_card is not None

    log_dispatch_event(
        conn,
        production_order_id=po.production_order_id,
        event_type="dispatch_planned",
        from_state=po.current_state,
        to_state=envelope.expected_next_state,
        owner_profile=po.current_owner_profile,
        target_profile=manifest.target_profile,
        kanban_card_id=manifest.target_child_card_id,
        result="manifest_envelope_manual_fallback_created",
        next_action="manual_dispatch_ready",
    )
    log_dispatch_event(
        conn,
        production_order_id=po.production_order_id,
        event_type="dispatch_handoff_created",
        from_state=po.current_state,
        to_state=envelope.expected_next_state,
        owner_profile=po.current_owner_profile,
        target_profile=handoff.target_profile,
        kanban_card_id=handoff.target_child_card_id,
        packet_id=f"manual-fallback:{po.production_order_id}",
        result="manual_fallback_handoff_created",
        next_action="copy_paste_prompt_to_target_profile",
    )

    events = list_dispatch_events(conn, po.production_order_id)
    after_card = kb.get_task(conn, manifest.target_child_card_id)
    assert after_card is not None

    assert {event["event_type"] for event in events} <= set(ALLOWED_DISPATCH_EVENT_TYPES)
    assert events[-2]["event_type"] == "dispatch_planned"
    assert events[-2]["to_state"] == envelope.expected_next_state
    assert events[-1]["event_type"] == "dispatch_handoff_created"
    assert events[-1]["target_profile"] == handoff.target_profile
    assert events[-1]["kanban_card_id"] == handoff.target_child_card_id
    assert events[-1]["packet_id"] == f"manual-fallback:{po.production_order_id}"
    assert after_card.body == before_card.body
    assert after_card.status == before_card.status
    assert after_card.current_state == before_card.current_state


def test_profile_task_envelope_conflicting_duplicate_fails_loudly(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    envelope = build_profile_task_envelope(conn, po.production_order_id)
    conn.execute(
        """
        UPDATE production_order_events
        SET result = ?
        WHERE production_order_id = ? AND event_type = 'dispatch_planned' AND packet_id = ?
        """,
        (
            "dispatch_planned payload_hash=" + ("0" * 64),
            po.production_order_id,
            envelope.idempotency_key,
        ),
    )
    conn.commit()

    with pytest.raises(DispatchManifestError, match="payload hash mismatch"):
        build_profile_task_envelope(conn, po.production_order_id)


def test_manual_fallback_conflicting_duplicate_fails_loudly(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    handoff = build_manual_fallback_handoff(conn, po.production_order_id)
    conn.execute(
        """
        UPDATE production_order_events
        SET result = ?
        WHERE production_order_id = ? AND event_type = 'dispatch_handoff_created' AND packet_id = ?
        """,
        (
            "manual_fallback_created payload_hash=" + ("1" * 64),
            po.production_order_id,
            handoff.idempotency_key,
        ),
    )
    conn.commit()

    with pytest.raises(DispatchManifestError, match="payload hash mismatch"):
        build_manual_fallback_handoff(conn, po.production_order_id)


def test_ingest_profile_result_packet_is_idempotent_for_same_packet(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    packet = devos_build_packet(po.production_order_id)
    packet["packet_id"] = "pkt-idempotent"

    first = ingest_profile_result_packet(conn, po.production_order_id, packet)
    events_after_first = list_dispatch_events(conn, po.production_order_id)
    card_after_first = kb.get_task(conn, po.child_kanban_card_ids[2])
    assert card_after_first is not None

    second = ingest_profile_result_packet(conn, po.production_order_id, dict(packet))
    events_after_second = list_dispatch_events(conn, po.production_order_id)
    card_after_second = kb.get_task(conn, po.child_kanban_card_ids[2])
    assert card_after_second is not None

    assert second == first
    assert len(events_after_second) == len(events_after_first)
    assert card_after_second.body == card_after_first.body
    assert card_after_second.body is not None
    assert card_after_second.body.count("--- RESULT PACKET ---") == 1


def test_ingest_profile_result_packet_rejects_conflicting_accepted_duplicate(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    packet = devos_build_packet(po.production_order_id)
    packet["packet_id"] = "pkt-original"
    ingest_profile_result_packet(conn, po.production_order_id, packet)

    conflicting = devos_build_packet(po.production_order_id)
    conflicting["packet_id"] = "pkt-conflict"
    conflicting["summary"] = "Different accepted payload"

    with pytest.raises(DispatchManifestError, match="Conflicting accepted result packet"):
        ingest_profile_result_packet(conn, po.production_order_id, conflicting)


def test_dispatch_lifecycle_restart_safety_reuses_existing_records(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    manifest = build_dispatch_manifest(conn, po.production_order_id)
    envelope = build_profile_task_envelope(conn, po.production_order_id)
    handoff = build_manual_fallback_handoff(conn, po.production_order_id)
    before_events = list_dispatch_events(conn, po.production_order_id)

    restarted = kb.connect()
    try:
        manifest_after_restart = build_dispatch_manifest(restarted, po.production_order_id)
        envelope_after_restart = build_profile_task_envelope(restarted, po.production_order_id)
        handoff_after_restart = build_manual_fallback_handoff(restarted, po.production_order_id)
        after_events = list_dispatch_events(restarted, po.production_order_id)
    finally:
        restarted.close()

    assert manifest_after_restart == manifest
    assert envelope_after_restart == envelope
    assert handoff_after_restart == handoff
    assert after_events == before_events



def test_manual_fallback_handoff_rejects_missing_expected_result_packet_details(conn, sample_brief):
    envelope = build_profile_task_envelope(conn, create_ready_for_dev_order(conn, sample_brief).production_order_id)

    with pytest.raises(DispatchManifestError, match="required_fields"):
        manual_fallback_handoff_for_envelope(
            replace(
                envelope,
                expected_output_packet={
                    **envelope.expected_output_packet,
                    "required_fields": [],
                },
            )
        )

    with pytest.raises(DispatchManifestError, match="bridge_function"):
        manual_fallback_handoff_for_envelope(
            replace(
                envelope,
                expected_output_packet={
                    **envelope.expected_output_packet,
                    "bridge_function": "",
                },
            )
        )



def test_build_profile_task_envelope_fails_loudly_for_missing_parent_child_and_graph_errors(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)

    with pytest.raises(DispatchManifestError, match="missing its parent Kanban card"):
        profile_task_envelope_for_order(
            replace(po, parent_kanban_card_id=""),
            dispatch_manifest_for_order(po),
        )

    with pytest.raises(DispatchManifestError, match="exactly 6 child cards"):
        profile_task_envelope_for_order(
            replace(po, child_kanban_card_ids=po.child_kanban_card_ids[:5]),
            dispatch_manifest_for_order(po),
        )

    with pytest.raises(DispatchManifestError, match="duplicate child card IDs"):
        profile_task_envelope_for_order(
            replace(
                po,
                child_kanban_card_ids=[po.child_kanban_card_ids[0]] * 6,
            ),
            dispatch_manifest_for_order(po),
        )

    missing_child_id = po.child_kanban_card_ids[-1]
    conn.execute("DELETE FROM task_links WHERE child_id = ?", (missing_child_id,))
    conn.execute("DELETE FROM tasks WHERE id = ?", (missing_child_id,))
    conn.commit()

    with pytest.raises(DispatchManifestError, match="references missing child card"):
        build_profile_task_envelope(conn, po.production_order_id)


@pytest.mark.parametrize(

    ("mutator", "expected_message"),
    [
        (lambda packet: packet.pop("summary"), "summary"),
        (lambda packet: packet.__setitem__("production_order_id", "PO-wrong"), "production_order_id"),
        (lambda packet: packet.__setitem__("owner_profile", "architect_os"), "owner_profile"),
        (lambda packet: packet.__setitem__("source_state", "ARCHITECT_SPEC"), "source_state"),
        (lambda packet: packet.__setitem__("test_status", "failed"), "pass/green/success"),
        (lambda packet: packet.__setitem__("next_handoff_target", "dev_os"), "next_handoff_target"),
    ],
)
def test_validate_devos_build_packet_rejects_invalid_fields(mutator, expected_message):
    packet = devos_build_packet("PO-20260525-test")
    mutator(packet)

    with pytest.raises(ValueError, match=expected_message):
        validate_devos_build_packet(
            packet,
            expected_production_order_id="PO-20260525-test",
        )


def test_freeze_result_on_card_appends_packet(conn):
    task_id = kb.create_task(conn, title="DevOS card", body="existing")
    result_packet = devos_build_packet("PO-20260525-test")

    freeze_result_on_card(conn, task_id, result_packet)

    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.body is not None
    assert "existing" in task.body
    assert "--- RESULT PACKET ---" in task.body
    assert '"owner_profile": "dev_os"' in task.body


def test_create_auditos_handoff_contains_expected_fields(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)

    handoff = create_auditos_handoff(po, devos_build_packet(po.production_order_id))

    assert handoff["production_order_id"] == po.production_order_id
    assert handoff["from_profile"] == "dev_os"
    assert handoff["to_profile"] == "audit_os"
    assert handoff["from_state"] == "DEV_IMPLEMENTING"
    assert handoff["to_state"] == "DEV_COMPLETE"
    assert handoff["devos_summary"]
    assert handoff["implementation_artifacts"]
    assert handoff["tests_and_evidence"]["tests_run"]
    assert handoff["tests_and_evidence"]["test_status"] == "green"
    assert handoff["acceptance_criteria"]
    assert handoff["stop_conditions"]
    assert handoff["reference_workflow_spec"] == WORKFLOW_SPEC_SOURCE


def test_devos_complete_bridge_moves_existing_order_to_dev_complete(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    original_parent_id = po.parent_kanban_card_id
    original_child_ids = list(po.child_kanban_card_ids)

    completed = run_devos_complete_bridge(
        conn,
        production_order_id=po.production_order_id,
        devos_packet=devos_build_packet(po.production_order_id),
    )

    assert completed.current_state == "DEV_COMPLETE"
    assert completed.current_owner_profile == "audit_os"
    assert completed.parent_kanban_card_id == original_parent_id
    assert completed.child_kanban_card_ids == original_child_ids
    assert len(completed.child_kanban_card_ids) == 6
    assert any(
        s.from_state == "ARCHITECT_READY_FOR_DEV" and s.to_state == "DEV_IMPLEMENTING"
        for s in completed.stage_history
    )
    assert any(
        s.from_state == "DEV_IMPLEMENTING" and s.to_state == "DEV_COMPLETE"
        for s in completed.stage_history
    )

    link_count = conn.execute(
        "SELECT COUNT(*) AS n FROM task_links WHERE parent_id = ?",
        (completed.parent_kanban_card_id,),
    ).fetchone()["n"]
    assert link_count == 6

    devos_card = kb.get_task(conn, original_child_ids[2])
    audit_card = kb.get_task(conn, original_child_ids[3])
    assert devos_card is not None
    assert audit_card is not None
    assert devos_card.body is not None
    assert audit_card.body is not None
    assert "--- RESULT PACKET ---" in devos_card.body
    assert '"owner_profile": "dev_os"' in devos_card.body
    assert "--- HANDOFF PACKET ---" in audit_card.body
    assert '\"to_profile\": \"audit_os\"' in audit_card.body
    assert audit_card.status == "ready"

    events = conn.execute(
        "SELECT event_type, from_state, to_state, owner_profile FROM production_order_events "
        "WHERE production_order_id = ? ORDER BY id",
        (po.production_order_id,),
    ).fetchall()
    assert [event["event_type"] for event in events][-3:] == [
        "dev_build_started",
        "dev_build_completed",
        "handoff_created",
    ]
    assert events[-3]["from_state"] == "ARCHITECT_READY_FOR_DEV"
    assert events[-3]["to_state"] == "DEV_IMPLEMENTING"
    assert events[-2]["from_state"] == "DEV_IMPLEMENTING"
    assert events[-2]["to_state"] == "DEV_COMPLETE"
    assert events[-1]["from_state"] == "DEV_IMPLEMENTING"
    assert events[-1]["to_state"] == "DEV_COMPLETE"
    assert events[-3]["owner_profile"] == "dev_os"
    assert events[-2]["owner_profile"] == "dev_os"
    assert events[-1]["owner_profile"] == "dev_os"


def test_audit_rejection_rework_loop_preserves_history_and_reconstructs(conn, sample_brief):
    po = create_dev_complete_order(conn, sample_brief)
    original_parent_id = po.parent_kanban_card_id
    original_child_ids = list(po.child_kanban_card_ids)
    rejection_packet = audit_rejection_packet(po.production_order_id)

    rejected = run_auditos_review_reject_bridge(
        conn,
        production_order_id=po.production_order_id,
        rejection_packet=rejection_packet,
    )
    assert rejected.current_state == "AUDIT_REJECTED"
    assert rejected.current_owner_profile == "orchestrator_os"

    audit_card = kb.get_task(conn, original_child_ids[3])
    assert audit_card is not None
    assert audit_card.body is not None
    assert "--- RESULT PACKET ---" in audit_card.body
    assert '"review_result": "rejected"' in audit_card.body
    assert '"correction_request"' in audit_card.body

    routed = run_orchestrator_rework_bridge(
        conn,
        production_order_id=po.production_order_id,
        rejection_packet=rejection_packet,
    )
    assert routed.current_state == "DEV_REWORK"
    assert routed.current_owner_profile == "dev_os"

    devos_card = kb.get_task(conn, original_child_ids[2])
    assert devos_card is not None
    assert devos_card.body is not None
    assert devos_card.body.count("--- HANDOFF PACKET ---") >= 2
    assert '"current_state": "AUDIT_REJECTED"' in devos_card.body

    reworked = run_devos_rework_complete_bridge(
        conn,
        production_order_id=po.production_order_id,
        devos_packet=devos_rework_packet(po.production_order_id),
    )
    assert reworked.current_state == "DEV_COMPLETE"
    assert reworked.current_owner_profile == "audit_os"

    audit_after_rework = kb.get_task(conn, original_child_ids[3])
    assert audit_after_rework is not None
    assert audit_after_rework.body is not None
    assert audit_after_rework.body.count("--- RESULT PACKET ---") >= 1
    assert audit_after_rework.body.count("--- HANDOFF PACKET ---") >= 2
    devos_after_rework = kb.get_task(conn, original_child_ids[2])
    assert devos_after_rework is not None
    assert devos_after_rework.body is not None
    assert '"source_state": "DEV_REWORK"' in devos_after_rework.body

    passed = run_auditos_review_complete_bridge(
        conn,
        production_order_id=po.production_order_id,
        review_packet=audit_review_packet(po.production_order_id, source_state="DEV_COMPLETE"),
    )
    assert passed.current_state == "AUDIT_PASSED"
    assert passed.current_owner_profile == "architect_os"
    assert passed.parent_kanban_card_id == original_parent_id
    assert passed.child_kanban_card_ids == original_child_ids
    assert passed.repo_or_workspace == sample_brief["target repo or workspace"]
    _assert_six_card_graph_preserved(conn, passed, original_child_ids)

    reconstructed = [
        order for order in list_production_orders(conn)
        if order.production_order_id == po.production_order_id
    ][0]
    assert reconstructed.parent_kanban_card_id == original_parent_id
    assert reconstructed.child_kanban_card_ids == original_child_ids
    assert reconstructed.repo_or_workspace == sample_brief["target repo or workspace"]
    assert any(entry.to_state == "AUDIT_REJECTED" for entry in reconstructed.stage_history)
    assert any(entry.to_state == "DEV_REWORK" for entry in reconstructed.stage_history)
    assert any(entry.to_state == "DEV_COMPLETE" and entry.from_state == "DEV_REWORK" for entry in reconstructed.stage_history)
    assert any(entry.to_state == "AUDIT_REVIEW" and entry.from_state == "DEV_COMPLETE" for entry in reconstructed.stage_history)
    assert any(entry.to_state == "AUDIT_PASSED" for entry in reconstructed.stage_history)

    event_types = _po_event_types(conn, po.production_order_id)
    assert "state_transitioned" in event_types
    assert "stage_rejected" in event_types
    assert "retry_started" in event_types
    assert "handoff_created" in event_types
    assert "stage_completed" in event_types


@pytest.mark.parametrize(
    ("classification", "expected_state", "expected_owner", "target_card_index", "expected_target_token"),
    [
        ("implementation_mismatch", "DEV_REWORK", "dev_os", 2, '"to_profile": "dev_os"'),
        ("spec_or_design_mismatch", "SPEC_REWORK", "architect_os", 1, '"to_profile": "architect_os"'),
    ],
)
def test_orchestrator_default_rejection_classification_bridge(
    conn,
    sample_brief,
    classification,
    expected_state,
    expected_owner,
    target_card_index,
    expected_target_token,
):
    po = create_default_final_review_order(conn, sample_brief)
    default_packet = default_rejection_packet(po.production_order_id)

    rejected = run_default_final_review_reject_bridge(
        conn,
        production_order_id=po.production_order_id,
        rejection_packet=default_packet,
    )
    assert rejected.current_state == "DEFAULT_REJECTED"
    assert rejected.current_owner_profile == "orchestrator_os"

    triaged = run_orchestrator_default_rejection_triage_bridge(
        conn,
        production_order_id=po.production_order_id,
        rejection_packet=default_packet,
    )
    assert triaged.current_state == "ORCHESTRATOR_TRIAGE"
    assert triaged.current_owner_profile == "orchestrator_os"

    routed = run_orchestrator_classification_bridge(
        conn,
        production_order_id=po.production_order_id,
        classification_packet=orchestrator_classification_packet(
            po.production_order_id,
            classification,
        ),
    )
    assert routed.current_state == expected_state
    assert routed.current_owner_profile == expected_owner

    event_types = _po_event_types(conn, po.production_order_id)
    assert event_types[-3:] == [
        "retry_started",
        "state_transitioned",
        "handoff_created",
    ]


def test_architect_reconcile_bridge_moves_existing_order_to_architect_accepted(conn, sample_brief):
    po = create_audit_passed_order(conn, sample_brief)
    original_child_ids = list(po.child_kanban_card_ids)

    completed = run_architect_reconcile_bridge(
        conn,
        production_order_id=po.production_order_id,
        reconcile_packet=architect_reconcile_packet(po.production_order_id),
    )

    assert completed.current_state == "ARCHITECT_ACCEPTED"
    assert completed.current_owner_profile == "default"
    _assert_six_card_graph_preserved(conn, completed, original_child_ids)

    reconcile_card = kb.get_task(conn, original_child_ids[4])
    final_card = kb.get_task(conn, original_child_ids[5])
    assert reconcile_card is not None
    assert final_card is not None
    assert reconcile_card.body is not None
    assert final_card.body is not None
    assert "--- RESULT PACKET ---" in reconcile_card.body
    assert '"owner_profile": "architect_os"' in reconcile_card.body
    assert "--- HANDOFF PACKET ---" in final_card.body
    assert '"to_profile": "default"' in final_card.body
    assert final_card.status == "ready"

    event_types = _po_event_types(conn, po.production_order_id)
    assert event_types[-3:] == [
        "architect_reconcile_started",
        "architect_reconcile_completed",
        "handoff_created",
    ]


def test_default_final_review_bridge_moves_existing_order_to_done(conn, sample_brief):
    po = create_architect_accepted_order(conn, sample_brief)
    original_child_ids = list(po.child_kanban_card_ids)

    completed = run_default_final_review_bridge(
        conn,
        production_order_id=po.production_order_id,
        final_packet=final_review_packet(po.production_order_id),
    )

    assert completed.current_state == "DONE"
    assert completed.current_owner_profile == "default"
    assert completed.final_status == "DONE"
    _assert_six_card_graph_preserved(conn, completed, original_child_ids)

    final_card = kb.get_task(conn, original_child_ids[5])
    assert final_card is not None
    assert final_card.body is not None
    assert "--- RESULT PACKET ---" in final_card.body
    assert '"owner_profile": "default"' in final_card.body
    assert final_card.status == "done"

    event_types = _po_event_types(conn, po.production_order_id)
    assert event_types[-3:] == [
        "default_final_review_started",
        "default_final_review_completed",
        "workflow_completed",
    ]


def test_default_rejection_bridge_routes_to_orchestrator_triage(conn, sample_brief):
    po = create_default_final_review_order(conn, sample_brief)
    original_child_ids = list(po.child_kanban_card_ids)

    rejected = run_default_final_review_reject_bridge(
        conn,
        production_order_id=po.production_order_id,
        rejection_packet=default_rejection_packet(po.production_order_id),
    )
    assert rejected.current_state == "DEFAULT_REJECTED"
    assert rejected.current_owner_profile == "orchestrator_os"
    _assert_six_card_graph_preserved(conn, rejected, original_child_ids)

    final_card = kb.get_task(conn, original_child_ids[5])
    assert final_card is not None
    assert final_card.body is not None
    assert final_card.body.count("--- HANDOFF PACKET ---") == 1
    assert final_card.body.count("--- RESULT PACKET ---") == 1
    assert '"owner_profile": "default"' in final_card.body
    assert '"review_result": "rejected"' in final_card.body

    routed = run_orchestrator_default_rejection_triage_bridge(
        conn,
        production_order_id=po.production_order_id,
        rejection_packet=default_rejection_packet(po.production_order_id),
    )
    assert routed.current_state == "ORCHESTRATOR_TRIAGE"
    assert routed.current_owner_profile == "orchestrator_os"
    _assert_six_card_graph_preserved(conn, routed, original_child_ids)

    orchestrator_card = kb.get_task(conn, original_child_ids[0])
    assert orchestrator_card is not None
    assert orchestrator_card.body is not None
    assert orchestrator_card.body.count("--- HANDOFF PACKET ---") == 2
    assert '"to_profile": "orchestrator_os"' in orchestrator_card.body
    assert '"requested_next_state": "ORCHESTRATOR_TRIAGE"' in orchestrator_card.body

    event_types = _po_event_types(conn, po.production_order_id)
    assert "default_final_review_started" in event_types
    assert "state_transitioned" in event_types
    assert "stage_rejected" in event_types
    assert "handoff_created" in event_types
    assert event_types[-2:] == ["state_transitioned", "handoff_created"]


def test_full_happy_path_chain_from_dev_complete_to_done(conn, sample_brief):
    po = create_dev_complete_order(conn, sample_brief)
    original_parent_id = po.parent_kanban_card_id
    original_child_ids = list(po.child_kanban_card_ids)

    po = run_auditos_review_complete_bridge(
        conn,
        production_order_id=po.production_order_id,
        review_packet=audit_review_packet(po.production_order_id),
    )
    po = run_architect_reconcile_bridge(
        conn,
        production_order_id=po.production_order_id,
        reconcile_packet=architect_reconcile_packet(po.production_order_id),
    )
    po = run_default_final_review_bridge(
        conn,
        production_order_id=po.production_order_id,
        final_packet=final_review_packet(po.production_order_id),
    )

    assert po.current_state == "DONE"
    assert po.current_owner_profile == "default"
    assert po.parent_kanban_card_id == original_parent_id
    _assert_six_card_graph_preserved(conn, po, original_child_ids)
    assert po.repo_or_workspace == sample_brief["target repo or workspace"]
    assert any(s.to_state == "AUDIT_REVIEW" for s in po.stage_history)
    assert any(s.to_state == "AUDIT_PASSED" for s in po.stage_history)
    assert any(s.to_state == "ARCHITECT_RECONCILE" for s in po.stage_history)
    assert any(s.to_state == "ARCHITECT_ACCEPTED" for s in po.stage_history)
    assert any(s.to_state == "DEFAULT_FINAL_REVIEW" for s in po.stage_history)
    assert any(s.to_state == "DONE" for s in po.stage_history)

    event_types = _po_event_types(conn, po.production_order_id)
    assert "audit_review_completed" in event_types
    assert "architect_reconcile_completed" in event_types
    assert "default_final_review_completed" in event_types
    assert "workflow_completed" in event_types
    assert event_types.count("handoff_created") >= 5


@pytest.mark.parametrize(
    ("po_action", "file_attr", "expected"),
    [
        ("audit-complete", "review_file", "--review-file is required"),
        ("architect-reconcile", "reconcile_file", "--reconcile-file is required"),
        ("final-review", "final_file", "--final-file is required"),
    ],
)
def test_remaining_bridge_cli_commands_require_packet_files(capsys, po_action, file_attr, expected):
    args = {
        "po_action": po_action,
        "production_order_id": "PO-20260525-test",
        "board": None,
        "json": False,
        "review_file": None,
        "reconcile_file": None,
        "final_file": None,
    }
    args[file_attr] = None

    rc = _cmd_production_order(argparse.Namespace(**args))

    captured = capsys.readouterr()
    assert rc == 1
    assert expected in captured.err


@pytest.mark.parametrize(
    ("bridge_name", "po_factory", "packet_factory", "runner", "mutator", "expected_message"),
    [
        ("audit", create_dev_complete_order, audit_review_packet, run_auditos_review_complete_bridge, lambda p: p.pop("summary"), "summary"),
        ("audit", create_dev_complete_order, audit_review_packet, run_auditos_review_complete_bridge, lambda p: p.__setitem__("production_order_id", "PO-wrong"), "production_order_id"),
        ("audit", create_dev_complete_order, audit_review_packet, run_auditos_review_complete_bridge, lambda p: p.__setitem__("owner_profile", "dev_os"), "owner_profile"),
        ("audit", create_dev_complete_order, audit_review_packet, run_auditos_review_complete_bridge, lambda p: p.__setitem__("source_state", "ARCHITECT_READY_FOR_DEV"), "source_state"),
        ("audit", create_dev_complete_order, audit_review_packet, run_auditos_review_complete_bridge, lambda p: p.__setitem__("verdict", "FAIL"), "happy-path"),
        ("reconcile", create_audit_passed_order, architect_reconcile_packet, run_architect_reconcile_bridge, lambda p: p.pop("summary"), "summary"),
        ("reconcile", create_audit_passed_order, architect_reconcile_packet, run_architect_reconcile_bridge, lambda p: p.__setitem__("production_order_id", "PO-wrong"), "production_order_id"),
        ("reconcile", create_audit_passed_order, architect_reconcile_packet, run_architect_reconcile_bridge, lambda p: p.__setitem__("owner_profile", "audit_os"), "owner_profile"),
        ("reconcile", create_audit_passed_order, architect_reconcile_packet, run_architect_reconcile_bridge, lambda p: p.__setitem__("source_state", "AUDIT_REVIEW"), "source_state"),
        ("reconcile", create_audit_passed_order, architect_reconcile_packet, run_architect_reconcile_bridge, lambda p: p.__setitem__("reconcile_result", "rejected"), "happy-path"),
        ("final", create_architect_accepted_order, final_review_packet, run_default_final_review_bridge, lambda p: p.pop("summary"), "summary"),
        ("final", create_architect_accepted_order, final_review_packet, run_default_final_review_bridge, lambda p: p.__setitem__("production_order_id", "PO-wrong"), "production_order_id"),
        ("final", create_architect_accepted_order, final_review_packet, run_default_final_review_bridge, lambda p: p.__setitem__("owner_profile", "audit_os"), "owner_profile"),
        ("final", create_architect_accepted_order, final_review_packet, run_default_final_review_bridge, lambda p: p.__setitem__("source_state", "ARCHITECT_RECONCILE"), "source_state"),
        ("final", create_architect_accepted_order, final_review_packet, run_default_final_review_bridge, lambda p: p.__setitem__("final_status", "rejected"), "happy-path"),
        ("default_reject", create_default_final_review_order, default_rejection_packet, run_default_final_review_reject_bridge, lambda p: p.pop("summary"), "summary"),
        ("default_reject", create_default_final_review_order, default_rejection_packet, run_default_final_review_reject_bridge, lambda p: p.__setitem__("owner_profile", "hermes"), "owner_profile"),
    ],
)
def test_remaining_bridge_packet_failures_do_not_mutate_state(
    conn,
    sample_brief,
    bridge_name,
    po_factory,
    packet_factory,
    runner,
    mutator,
    expected_message,
):
    po = po_factory(conn, sample_brief)
    original_state = po.current_state
    original_owner = po.current_owner_profile
    original_event_count = conn.execute(
        "SELECT COUNT(*) AS n FROM production_order_events WHERE production_order_id = ?",
        (po.production_order_id,),
    ).fetchone()["n"]
    original_bodies = {
        cid: kb.get_task(conn, cid).body
        for cid in po.child_kanban_card_ids
        if kb.get_task(conn, cid) is not None
    }
    packet = packet_factory(po.production_order_id)
    mutator(packet)

    with pytest.raises(ValueError, match=expected_message):
        if bridge_name == "audit":
            runner(conn, production_order_id=po.production_order_id, review_packet=packet)
        elif bridge_name == "reconcile":
            runner(conn, production_order_id=po.production_order_id, reconcile_packet=packet)
        elif bridge_name == "default_reject":
            runner(conn, production_order_id=po.production_order_id, rejection_packet=packet)
        else:
            runner(conn, production_order_id=po.production_order_id, final_packet=packet)

    refreshed = [
        order for order in list_production_orders(conn)
        if order.production_order_id == po.production_order_id
    ][0]
    refreshed_event_count = conn.execute(
        "SELECT COUNT(*) AS n FROM production_order_events WHERE production_order_id = ?",
        (po.production_order_id,),
    ).fetchone()["n"]
    assert refreshed.current_state == original_state
    assert refreshed.current_owner_profile == original_owner
    assert refreshed.child_kanban_card_ids == po.child_kanban_card_ids
    assert refreshed_event_count == original_event_count
    for cid, body in original_bodies.items():
        task = kb.get_task(conn, cid)
        assert task is not None
        assert task.body == body


def test_remaining_bridge_cli_json_outputs(capsys, conn, tmp_path, sample_brief):
    po = create_dev_complete_order(conn, sample_brief)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit_review_packet(po.production_order_id)), encoding="utf-8")

    rc = _cmd_production_order(argparse.Namespace(
        po_action="audit-complete",
        production_order_id=po.production_order_id,
        board=None,
        review_file=str(audit_path),
        json=True,
    ))
    captured = capsys.readouterr()
    assert rc == 0
    parsed = json.loads(captured.out)
    assert parsed["current_state"] == "AUDIT_PASSED"
    assert parsed["current_owner_profile"] == "architect_os"

    reconcile_path = tmp_path / "reconcile.json"
    reconcile_path.write_text(
        json.dumps(architect_reconcile_packet(po.production_order_id)),
        encoding="utf-8",
    )
    rc = _cmd_production_order(argparse.Namespace(
        po_action="architect-reconcile",
        production_order_id=po.production_order_id,
        board=None,
        reconcile_file=str(reconcile_path),
        json=True,
    ))
    captured = capsys.readouterr()
    assert rc == 0
    parsed = json.loads(captured.out)
    assert parsed["current_state"] == "ARCHITECT_ACCEPTED"
    assert parsed["current_owner_profile"] == "default"

    final_path = tmp_path / "final.json"
    final_path.write_text(json.dumps(final_review_packet(po.production_order_id)), encoding="utf-8")
    rc = _cmd_production_order(argparse.Namespace(
        po_action="final-review",
        production_order_id=po.production_order_id,
        board=None,
        final_file=str(final_path),
        json=True,
    ))
    captured = capsys.readouterr()
    assert rc == 0
    parsed = json.loads(captured.out)
    assert parsed["current_state"] == "DONE"
    assert parsed["current_owner_profile"] == "default"


def test_completed_workflow_show_events_json_is_strict_json(capsys, conn, sample_brief):
    po = create_architect_accepted_order(conn, sample_brief)
    po = run_default_final_review_bridge(
        conn,
        production_order_id=po.production_order_id,
        final_packet=final_review_packet(po.production_order_id),
    )

    rc = _cmd_production_order(argparse.Namespace(
        po_action="show",
        production_order_id=po.production_order_id,
        board=None,
        events=True,
        json=True,
    ))

    captured = capsys.readouterr()
    assert rc == 0
    parsed = json.loads(captured.out)
    assert captured.err == ""
    assert parsed["current_state"] == "DONE"
    assert parsed["current_owner_profile"] == "default"
    event_types = [event["event_type"] for event in parsed["events"]]
    assert "audit_review_completed" in event_types
    assert "architect_reconcile_completed" in event_types
    assert "default_final_review_completed" in event_types
    assert "workflow_completed" in event_types
    assert "\n  Events (" not in captured.out


def test_completed_workflow_show_events_text_remains_human_readable(capsys, conn, sample_brief):
    po = create_architect_accepted_order(conn, sample_brief)
    po = run_default_final_review_bridge(
        conn,
        production_order_id=po.production_order_id,
        final_packet=final_review_packet(po.production_order_id),
    )

    rc = _cmd_production_order(argparse.Namespace(
        po_action="show",
        production_order_id=po.production_order_id,
        board=None,
        events=True,
        json=False,
    ))

    captured = capsys.readouterr()
    assert rc == 0
    assert "Events (" in captured.out
    assert "audit_review_completed" in captured.out
    assert "architect_reconcile_completed" in captured.out
    assert "default_final_review_completed" in captured.out
    assert "workflow_completed" in captured.out


# ---------------------------------------------------------------------------
# List / Status
# ---------------------------------------------------------------------------


def test_list_production_orders(conn, sample_brief):
    """List returns created production orders."""
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )
    create_production_kanban_graph(conn, po)

    orders = list_production_orders(conn)
    found = [o for o in orders if o.production_order_id == po.production_order_id]
    assert len(found) == 1
    assert found[0].title == sample_brief["title"]


def test_reconstructed_order_includes_children_repo_and_stage_history(conn, sample_brief):
    """List/show reconstruction returns a trustworthy runtime view after create."""
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )

    reconstructed = [
        o for o in list_production_orders(conn)
        if o.production_order_id == po.production_order_id
    ][0]

    assert reconstructed.child_kanban_card_ids == po.child_kanban_card_ids
    assert len(reconstructed.child_kanban_card_ids) == 6
    assert reconstructed.repo_or_workspace == sample_brief["target repo or workspace"]
    assert reconstructed.stage_history
    assert any(s.to_state == "ORCHESTRATOR_TRIAGE" for s in reconstructed.stage_history)


def test_reconstructed_order_uses_snake_case_repo_alias(conn, snake_case_brief):
    """Reconstruction populates repo/workspace from snake_case brief metadata."""
    po = run_full_bridge(
        conn,
        title=snake_case_brief["title"],
        source_brief=json.dumps(snake_case_brief),
        priority_lane="Hermes OS",
        repo_or_workspace=snake_case_brief["target_repo_or_workspace"],
    )

    reconstructed = [
        o for o in list_production_orders(conn)
        if o.production_order_id == po.production_order_id
    ][0]

    assert reconstructed.repo_or_workspace == snake_case_brief["target_repo_or_workspace"]


def test_production_order_show_events_json_is_strict_json(capsys, conn, sample_brief):
    """``show --events --json`` emits one strict JSON object with events embedded."""
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )

    rc = _cmd_production_order(argparse.Namespace(
        po_action="show",
        production_order_id=po.production_order_id,
        board=None,
        events=True,
        json=True,
    ))

    captured = capsys.readouterr()
    assert rc == 0
    parsed = json.loads(captured.out)
    assert parsed["production_order_id"] == po.production_order_id
    assert "\n  Events (" not in captured.out
    assert "events" in parsed
    assert [event["event_type"] for event in parsed["events"]] == [
        "brief_approved",
        "production_order_created",
        "kanban_graph_created",
        "state_transitioned",
        "handoff_created",
    ]


def test_production_order_show_events_text_still_prints_block(capsys, conn, sample_brief):
    """``show --events`` keeps human-readable event output outside JSON mode."""
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )

    rc = _cmd_production_order(argparse.Namespace(
        po_action="show",
        production_order_id=po.production_order_id,
        board=None,
        events=True,
        json=False,
    ))

    captured = capsys.readouterr()
    assert rc == 0
    assert "\n  Events (5):" in captured.out
    assert "brief_approved" in captured.out
    assert "handoff_created" in captured.out


# ---------------------------------------------------------------------------
# Negative Tests
# ---------------------------------------------------------------------------


def test_kanban_card_without_parent_exists_independently(conn):
    """A regular Kanban task can exist without a production order."""
    task_id = kb.create_task(conn, title="Regular task")
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.production_order_id is None
    assert task.current_state is None


def test_new_columns_null_on_existing_tasks(conn):
    """Existing (non-PO) tasks have NULL for production_order_id and current_state."""
    task_id = kb.create_task(conn, title="Regular task")

    row = conn.execute(
        "SELECT production_order_id, current_state FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert row["production_order_id"] is None
    assert row["current_state"] is None


def test_empty_brief_rejected(conn):
    """Empty/empty brief fields are caught by validate_brief."""
    brief = {}
    missing = validate_brief(brief)
    assert len(missing) == 9


def test_production_order_duplicate_id_is_unique():
    """Two consecutive calls produce different IDs."""
    a = generate_production_order_id()
    b = generate_production_order_id()
    assert a != b


def test_approval_boundary_none(conn, sample_brief):
    """Bridge does NOT enter BLOCKED_NEEDS_JARREN for internal operations."""
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )
    assert po.current_state != "BLOCKED_NEEDS_JARREN"
    assert po.blockers == []


def test_handoff_packet_missing_field(conn, sample_brief):
    """Handoff packet always has all required fields (template is complete)."""
    po = create_production_order(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
    )
    handoff = create_orchestrator_handoff(po)
    required = [
        "production_order_id", "from_profile", "to_profile",
        "current_state", "requested_next_state", "objective",
        "source_truth", "stop_conditions", "approval_required_before",
        "evidence_required",
    ]
    for field in required:
        assert field in handoff, f"Missing required field: {field}"
        assert handoff[field], f"Empty required field: {field}"


# ---------------------------------------------------------------------------
# Full Bridge with Fixture Files
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_path() -> Path:
    """Path to test fixture directory."""
    return Path(__file__).parent.parent / "fixtures" / "production-workflow"


def test_full_bridge_with_good_fixture(conn, fixture_path):
    """accepted-good-brief.json fixture creates a full production order."""
    path = fixture_path / "accepted-good-brief.json"
    assert path.exists(), f"Fixture not found: {path}"

    brief = json.loads(path.read_text())
    po = run_full_bridge(
        conn,
        title=brief.get("title", brief.get("objective", "Untitled")),
        source_brief=json.dumps(brief, indent=2),
        priority_lane=brief.get("priority_lane", "Hermes OS"),
        repo_or_workspace=brief.get("target repo or workspace", ""),
    )

    assert po.production_order_id.startswith("PO-")
    assert po.current_state == "ORCHESTRATOR_TRIAGE"
    assert len(po.child_kanban_card_ids) == 6


def test_vague_brief_blocked(conn, fixture_path):
    """rejected-vague-brief.json has missing 'scope' field.

    Note: validate_brief catches the missing field BEFORE any bridge
    side effects. This test verifies that the bridge would block.
    """
    path = fixture_path / "rejected-vague-brief.json"
    assert path.exists()

    brief = json.loads(path.read_text())
    missing = validate_brief(brief)
    assert "scope" in missing or "out of scope" in missing


def test_ambiguous_approval_detected():
    """Non-explicit approval phrase is rejected by detect_approval_phrase."""
    # A message like "approved" (bare) is not sufficient
    assert detect_approval_phrase("approved") is None
    assert detect_approval_phrase("looks good") is None


def test_skill_only_request_no_side_effects(conn, fixture_path):
    """skill-only-request.json: the bridge does NOT create side effects
    unless explicitly routed through a production order.

    Note: In Slice 4, the bridge is called intentionally (approved brief
    + explicit bridge invocation). This test verifies that even with a
    valid brief, no production order is created unless the user invokes
    the bridge.
    """
    path = fixture_path / "skill-only-request.json"
    assert path.exists()

    brief = json.loads(path.read_text())
    # Valid brief - would pass validation
    missing = validate_brief(brief)
    assert missing == [], f"Fixture should be valid but has missing: {missing}"

    # But if we DON'T call create_production_order, no PO exists
    orders = list_production_orders(conn)
    matching = [o for o in orders if o.title == brief.get("objective")]
    assert len(matching) == 0


def test_dashboard_ignores_null_columns(conn):
    """Dashboard helper includes new fields as null for non-PO tasks."""
    task_id = kb.create_task(conn, title="Non-PO task")

    # The _task_to_dict in kanban.py includes production_order_id and
    # current_state, both None for non-PO tasks
    task = kb.get_task(conn, task_id)
    assert task.production_order_id is None
    assert task.current_state is None

    # Kanban db list_tasks also works
    tasks = kb.list_tasks(conn)
    assert any(t.id == task_id for t in tasks)
    non_po = [t for t in tasks if t.id == task_id]
    assert len(non_po) == 1
    assert non_po[0].production_order_id is None


def test_dispatch_executor_returns_manual_fallback_without_invocation(conn, sample_brief):
    po = create_architect_spec_order(conn, sample_brief)

    result = execute_profile_dispatch(conn, po.production_order_id)

    assert result["executed"] is False
    assert result["fallback_required"] is True
    assert result["target_profile"] == "architect_os"
    assert result["manual_fallback"]["target_profile"] == "architect_os"
    assert result["next_action"] == "manual_fallback_required"
    refreshed = list_production_orders(conn)[0]
    assert refreshed.current_state == "ARCHITECT_SPEC"


def test_dispatch_executor_repeat_reuses_fallback_and_events(conn, sample_brief):
    po = create_architect_spec_order(conn, sample_brief)

    first = execute_profile_dispatch(conn, po.production_order_id)
    second = execute_profile_dispatch(conn, po.production_order_id)

    assert second["idempotency_key"] == first["idempotency_key"]
    assert second["manual_fallback"] == first["manual_fallback"]
    events = list_dispatch_events(conn, po.production_order_id)
    assert [e["event_type"] for e in events].count("dispatch_started") == 1
    assert [e["event_type"] for e in events].count("dispatch_handoff_created") == 1


def test_dispatch_executor_refuses_unknown_identity(conn, sample_brief):
    po = create_architect_spec_order(conn, sample_brief)

    with pytest.raises(DispatchManifestError, match="target_profile"):
        execute_profile_dispatch(
            conn,
            po.production_order_id,
            target_profile="unknown_profile",
        )


def test_dispatch_executor_accepts_correct_task_type(conn, sample_brief):
    po = create_architect_spec_order(conn, sample_brief)
    envelope = build_profile_task_envelope(conn, po.production_order_id)

    result = execute_profile_dispatch(
        conn,
        po.production_order_id,
        task_type="architect_spec",
    )

    assert result["task_type"] == "architect_spec"
    assert envelope.idempotency_key.split(":")[5] == "architect_spec"


def test_dispatch_executor_rejects_child_card_id_as_task_type(conn, sample_brief):
    po = create_architect_spec_order(conn, sample_brief)

    with pytest.raises(DispatchManifestError, match="task_type"):
        execute_profile_dispatch(
            conn,
            po.production_order_id,
            task_type=po.child_kanban_card_ids[1],
        )


def test_dispatch_executor_orchestrator_triage_returns_manual_fallback(conn, sample_brief):
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )

    result = execute_profile_dispatch(
        conn,
        po.production_order_id,
        task_type="orchestrator_triage",
    )

    assert result["fallback_required"] is True
    assert result["target_profile"] == "orchestrator_os"
    assert result["manual_fallback"]["bridge_function"] == "run_orchestrator_triage_bridge"
    assert result["manual_fallback"]["expected_result_packet"]["packet_type"] == "architect_handoff_packet"


def test_dispatch_executor_orchestrator_default_rejection_classification_returns_manual_fallback(
    conn,
    sample_brief,
):
    rejected = _default_rejected_order(conn, sample_brief)
    triaged = run_orchestrator_default_rejection_triage_bridge(
        conn,
        production_order_id=rejected.production_order_id,
        rejection_packet=default_rejection_packet(
            rejected.production_order_id,
            source_state="DEFAULT_REJECTED",
        ),
    )

    result = execute_profile_dispatch(
        conn,
        triaged.production_order_id,
        task_type="orchestrator_default_rejection_classification",
    )

    assert result["fallback_required"] is True
    assert result["target_profile"] == "orchestrator_os"
    assert result["manual_fallback"]["bridge_function"] == "run_orchestrator_classification_bridge"
    assert result["manual_fallback"]["expected_result_packet"]["packet_type"] == "orchestrator_classification_packet"


def test_dispatch_executor_orchestrator_repeat_is_idempotent(conn, sample_brief):
    po = run_full_bridge(
        conn,
        title=sample_brief["title"],
        source_brief=json.dumps(sample_brief),
        priority_lane="Relay",
        repo_or_workspace=sample_brief["target repo or workspace"],
    )

    first = execute_profile_dispatch(
        conn,
        po.production_order_id,
        task_type="orchestrator_triage",
    )
    second = execute_profile_dispatch(
        conn,
        po.production_order_id,
        task_type="orchestrator_triage",
    )

    assert second["idempotency_key"] == first["idempotency_key"]
    assert second["manual_fallback"] == first["manual_fallback"]
    events = list_dispatch_events(conn, po.production_order_id)
    assert [e["event_type"] for e in events].count("dispatch_started") == 1
    assert [e["event_type"] for e in events].count("dispatch_handoff_created") == 1


def _ingest_and_apply(conn, production_order_id, packet):
    ingestion = ingest_profile_result_packet(conn, production_order_id, packet)
    assert ingestion["accepted"] is True
    return apply_accepted_result_action(conn, production_order_id, result_packet=packet)


def test_apply_accepted_architect_result_prepares_dev_stage(conn, sample_brief):
    po = create_architect_spec_order(conn, sample_brief)
    packet = architect_spec_packet(po.production_order_id)

    result = _ingest_and_apply(conn, po.production_order_id, packet)

    assert result["applied"] is True
    assert result["to_state"] == "ARCHITECT_READY_FOR_DEV"
    refreshed = list_production_orders(conn)[0]
    assert refreshed.current_owner_profile == "dev_os"
    assert kb.get_task(conn, refreshed.child_kanban_card_ids[2]).status == "ready"


def test_apply_accepted_dev_result_prepares_audit_stage(conn, sample_brief):
    po = create_ready_for_dev_order(conn, sample_brief)
    packet = devos_build_packet(po.production_order_id)

    result = _ingest_and_apply(conn, po.production_order_id, packet)

    assert result["to_state"] == "DEV_COMPLETE"
    refreshed = list_production_orders(conn)[0]
    assert refreshed.current_owner_profile == "audit_os"
    assert kb.get_task(conn, refreshed.child_kanban_card_ids[3]).status == "ready"


def test_apply_accepted_audit_pass_prepares_architect_reconcile(conn, sample_brief):
    po = create_dev_complete_order(conn, sample_brief)
    packet = audit_review_packet(po.production_order_id)

    result = _ingest_and_apply(conn, po.production_order_id, packet)

    assert result["to_state"] == "AUDIT_PASSED"
    refreshed = list_production_orders(conn)[0]
    assert refreshed.current_owner_profile == "architect_os"
    assert kb.get_task(conn, refreshed.child_kanban_card_ids[4]).status == "ready"


def test_apply_accepted_architect_reconcile_prepares_default_review(conn, sample_brief):
    po = create_audit_passed_order(conn, sample_brief)
    packet = architect_reconcile_packet(po.production_order_id)

    result = _ingest_and_apply(conn, po.production_order_id, packet)

    assert result["to_state"] == "ARCHITECT_ACCEPTED"
    refreshed = list_production_orders(conn)[0]
    assert refreshed.current_owner_profile == "default"
    assert kb.get_task(conn, refreshed.child_kanban_card_ids[5]).status == "ready"


def test_apply_accepted_default_pass_completes_done(conn, sample_brief):
    po = create_architect_accepted_order(conn, sample_brief)
    packet = final_review_packet(po.production_order_id)

    result = _ingest_and_apply(conn, po.production_order_id, packet)

    assert result["to_state"] == "DONE"
    refreshed = list_production_orders(conn)[0]
    assert refreshed.current_state == "DONE"
    assert kb.get_task(conn, refreshed.child_kanban_card_ids[5]).status == "done"


def test_apply_accepted_result_is_idempotent(conn, sample_brief):
    po = create_architect_spec_order(conn, sample_brief)
    packet = architect_spec_packet(po.production_order_id)

    first = _ingest_and_apply(conn, po.production_order_id, packet)
    second = apply_accepted_result_action(conn, po.production_order_id, result_packet=packet)

    assert first["applied"] is True
    assert second["applied"] is False
    events = _po_event_types(conn, po.production_order_id)
    assert events.count("architect_spec_completed") == 1


def test_apply_conflicting_action_fails(conn, sample_brief):
    po = create_architect_spec_order(conn, sample_brief)
    packet = architect_spec_packet(po.production_order_id)
    _ingest_and_apply(conn, po.production_order_id, packet)
    other = dict(packet)
    other["packet_id"] = "different"
    other["summary"] = "different accepted packet"

    with pytest.raises(DispatchManifestError, match="conflicting state|source_state"):
        apply_accepted_result_action(conn, po.production_order_id, result_packet=other)


def test_unaccepted_packet_cannot_be_applied(conn, sample_brief):
    po = create_architect_spec_order(conn, sample_brief)

    with pytest.raises(DispatchManifestError, match="before.*accepted"):
        apply_accepted_result_action(
            conn,
            po.production_order_id,
            result_packet=architect_spec_packet(po.production_order_id),
        )


def _audit_rejected_order(conn, sample_brief):
    po = create_dev_complete_order(conn, sample_brief)
    packet = audit_rejection_packet(po.production_order_id)
    return run_auditos_review_reject_bridge(
        conn,
        production_order_id=po.production_order_id,
        rejection_packet=packet,
    )


def _default_rejected_order(conn, sample_brief):
    po = create_default_final_review_order(conn, sample_brief)
    packet = default_rejection_packet(po.production_order_id)
    return run_default_final_review_reject_bridge(
        conn,
        production_order_id=po.production_order_id,
        rejection_packet=packet,
    )


def test_rework_router_audit_implementation_failure_to_dev(conn, sample_brief):
    po = _audit_rejected_order(conn, sample_brief)
    packet = audit_rejection_packet(po.production_order_id)
    packet["failure_type"] = "implementation test failure"

    result = route_production_order_rework(
        conn,
        po.production_order_id,
        rejection_source="AUDIT_REJECTED",
        rejection_packet=packet,
    )

    assert result["route_decision"] == "DEV_REWORK"
    assert result["target_profile"] == "dev_os"
    assert list_production_orders(conn)[0].current_state == "DEV_REWORK"


def test_rework_router_audit_spec_failure_to_architect(conn, sample_brief):
    po = _audit_rejected_order(conn, sample_brief)
    packet = audit_rejection_packet(po.production_order_id)
    packet["failure_type"] = "spec design ambiguity"

    result = route_production_order_rework(
        conn,
        po.production_order_id,
        rejection_source="AUDIT_REJECTED",
        rejection_packet=packet,
    )

    assert result["route_decision"] == "SPEC_REWORK"
    assert result["target_profile"] == "architect_os"
    assert list_production_orders(conn)[0].current_state == "SPEC_REWORK"


def test_rework_router_audit_approval_boundary_blocks(conn, sample_brief):
    po = _audit_rejected_order(conn, sample_brief)
    packet = audit_rejection_packet(po.production_order_id)
    packet["failure_type"] = "needs approval for secret credential"

    result = route_production_order_rework(
        conn,
        po.production_order_id,
        rejection_source="AUDIT_REJECTED",
        rejection_packet=packet,
    )

    assert result["route_decision"] == "BLOCKED_NEEDS_JARREN"
    assert result["stop_condition"]
    assert list_production_orders(conn)[0].current_state == "BLOCKED_NEEDS_JARREN"


def test_rework_router_default_implementation_mismatch_to_dev(conn, sample_brief):
    po = _default_rejected_order(conn, sample_brief)
    packet = default_rejection_packet(po.production_order_id)
    packet["rejection_category"] = "implementation output mismatch"

    result = route_production_order_rework(
        conn,
        po.production_order_id,
        rejection_source="DEFAULT_REJECTED",
        rejection_packet=packet,
    )

    assert result["route_decision"] == "DEV_REWORK"
    assert result["target_profile"] == "dev_os"


def test_rework_router_default_brief_spec_mismatch_to_architect(conn, sample_brief):
    po = _default_rejected_order(conn, sample_brief)
    packet = default_rejection_packet(po.production_order_id)
    packet["rejection_category"] = "brief/spec mismatch"

    result = route_production_order_rework(
        conn,
        po.production_order_id,
        rejection_source="DEFAULT_REJECTED",
        rejection_packet=packet,
    )

    assert result["route_decision"] == "SPEC_REWORK"
    assert result["target_profile"] == "architect_os"


def test_rework_router_ambiguous_rejection_blocks(conn, sample_brief):
    po = _audit_rejected_order(conn, sample_brief)
    packet = audit_rejection_packet(po.production_order_id)
    packet["summary"] = "Rejected for unclear reasons."
    packet["correction_request"] = ["Needs another look."]
    packet["risks_or_notes"] = ["Unclear."]

    result = route_production_order_rework(
        conn,
        po.production_order_id,
        rejection_source="AUDIT_REJECTED",
        rejection_packet=packet,
    )

    assert result["route_decision"] == "BLOCKED_NEEDS_JARREN"


def test_rework_router_repeat_is_idempotent(conn, sample_brief):
    po = _audit_rejected_order(conn, sample_brief)
    packet = audit_rejection_packet(po.production_order_id)
    packet["failure_type"] = "implementation bug"

    first = route_production_order_rework(
        conn,
        po.production_order_id,
        rejection_source="AUDIT_REJECTED",
        rejection_packet=packet,
    )
    second = route_production_order_rework(
        conn,
        po.production_order_id,
        rejection_source="AUDIT_REJECTED",
        rejection_packet=packet,
    )

    assert first["route_decision"] == second["route_decision"] == "DEV_REWORK"
    assert second["applied"] is False


def test_rework_router_conflicting_repeat_fails(conn, sample_brief):
    po = _audit_rejected_order(conn, sample_brief)
    packet = audit_rejection_packet(po.production_order_id)
    packet["failure_type"] = "implementation bug"
    route_production_order_rework(
        conn,
        po.production_order_id,
        rejection_source="AUDIT_REJECTED",
        rejection_packet=packet,
    )
    other = audit_rejection_packet(po.production_order_id)
    other["failure_type"] = "spec design ambiguity"

    with pytest.raises(DispatchManifestError, match="conflicts"):
        route_production_order_rework(
            conn,
            po.production_order_id,
            rejection_source="AUDIT_REJECTED",
            rejection_packet=other,
        )


def test_base36_random():
    """base36 random string has correct length and character set."""
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    for length in [1, 4, 8]:
        result = _base36_random(length)
        assert len(result) == length
        assert all(c in chars for c in result)
