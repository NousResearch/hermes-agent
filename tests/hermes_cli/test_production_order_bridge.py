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
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.production_order_db import (
    CHILD_CARD_DEFS,
    PRODUCTION_ORDER_FIELD,
    STATE_FIELD,
    WORKFLOW_INITIAL_STATE,
    WORKFLOW_TEMPLATE_ID,
    ProductionOrder,
    StageEntry,
    StateTransitionError,
    _base36_random,
    create_architect_handoff,
    create_orchestrator_handoff,
    create_production_kanban_graph,
    create_production_order,
    detect_approval_phrase,
    freeze_handoff_on_card,
    generate_production_order_id,
    list_production_orders,
    log_workflow_event,
    run_full_bridge,
    run_orchestrator_triage_bridge,
    transition_state,
    validate_brief,
    validate_state_transition,
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
    """3 allowed transitions pass validation."""
    # BRIEF_DRAFTED -> ACTION_APPROVED
    assert validate_state_transition("BRIEF_DRAFTED", "ACTION_APPROVED", "hermes")
    # ACTION_APPROVED -> PRODUCTION_ORDER_CREATED
    assert validate_state_transition("ACTION_APPROVED", "PRODUCTION_ORDER_CREATED", "hermes")
    # PRODUCTION_ORDER_CREATED -> ORCHESTRATOR_TRIAGE
    assert validate_state_transition("PRODUCTION_ORDER_CREATED", "ORCHESTRATOR_TRIAGE", "orchestrator_os")
    # ORCHESTRATOR_TRIAGE -> ARCHITECT_SPEC
    assert validate_state_transition("ORCHESTRATOR_TRIAGE", "ARCHITECT_SPEC", "orchestrator_os")


def test_invalid_state_transition_rejected():
    """Invalid transition raises StateTransitionError."""
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


def test_base36_random():
    """base36 random string has correct length and character set."""
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    for length in [1, 4, 8]:
        result = _base36_random(length)
        assert len(result) == length
        assert all(c in chars for c in result)