"""Native dynamic Kanban workflow aggregation contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def workflow_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "workflow.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    kbc._INITIALIZED_PATHS.discard(str(path.resolve()))
    conn = kbc.connect(path)
    try:
        yield conn
    finally:
        conn.close()
        kbc._INITIALIZED_PATHS.discard(str(path.resolve()))


def test_workflow_create_is_idempotent_and_designates_acceptance(workflow_db):
    acceptance = kb.create_task(
        workflow_db, title="accept", assignee="orchestrator", tenant="tenant-a",
        session_id="provenance-only",
    )
    actor = kb.KanbanActorContext(
        principal_id="svc:orchestrator", profile_name="orchestrator",
        board_identity="board:test", tenant="tenant-a",
        capabilities=frozenset({"workflow.manage"}), source_kind="orchestrator",
    )

    first = kb.create_workflow(
        workflow_db, workflow_id="wf_test", name="release", tenant="tenant-a",
        designated_acceptance_task_id=acceptance, actor=actor, mutation_id="mut-create",
    )
    retry = kb.create_workflow(
        workflow_db, workflow_id="wf_test", name="release", tenant="tenant-a",
        designated_acceptance_task_id=acceptance, actor=actor, mutation_id="mut-create",
    )

    assert retry == first
    assert first["workflow"]["state"] == "ACTIVE"
    assert first["workflow"]["version"] == 1
    assert first["generation"]["designated_acceptance_task_id"] == acceptance
    assert first["members"] == [{
        "task_id": acceptance, "stage_key": "acceptance", "stage_role": "acceptance",
        "required": True,
    }]
    assert workflow_db.execute(
        "SELECT COUNT(*) FROM kanban_workflow_events WHERE workflow_id='wf_test'"
    ).fetchone()[0] == 1
    assert workflow_db.execute(
        "SELECT COUNT(*) FROM kanban_workflow_mutations WHERE workflow_id='wf_test'"
    ).fetchone()[0] == 1

    with pytest.raises(kb.WorkflowConflictError, match="mutation_id"):
        kb.create_workflow(
            workflow_db, workflow_id="wf_test", name="different request", tenant="tenant-a",
            designated_acceptance_task_id=acceptance, actor=actor, mutation_id="mut-create",
        )


def _admin_actor() -> kb.KanbanActorContext:
    return kb.KanbanActorContext(
        principal_id="svc:orchestrator", profile_name="orchestrator",
        board_identity="board:test", tenant="tenant-a",
        capabilities=frozenset({"workflow.read", "workflow.manage", "workflow.admin"}),
        source_kind="orchestrator",
    )


def test_member_enrollment_is_explicit_and_rejects_cross_tenant_links(workflow_db):
    acceptance = kb.create_task(workflow_db, title="accept", tenant="tenant-a")
    implementation = kb.create_task(workflow_db, title="implementation", tenant="tenant-a")
    foreign = kb.create_task(workflow_db, title="foreign", tenant="tenant-b")
    actor = _admin_actor()
    created = kb.create_workflow(
        workflow_db, workflow_id="wf_members", name="release", tenant="tenant-a",
        designated_acceptance_task_id=acceptance, actor=actor, mutation_id="create",
    )
    result = kb.add_workflow_member(
        workflow_db, workflow_id="wf_members", task_id=implementation,
        stage_key="implementation", stage_role="implementation", required=True,
        actor=actor, mutation_id="add", expected_version=created["workflow"]["version"],
    )

    assert {member["task_id"] for member in result["members"]} == {acceptance, implementation}
    assert workflow_db.execute("SELECT COUNT(*) FROM task_links").fetchone()[0] == 0
    with pytest.raises(kb.WorkflowIntegrityError, match="matching non-null tenant"):
        kb.link_tasks(workflow_db, implementation, foreign)


def test_terminal_generation_is_immutable_and_reopen_creates_a_new_generation(workflow_db):
    acceptance = kb.create_task(workflow_db, title="accept", tenant="tenant-a")
    replacement = kb.create_task(workflow_db, title="reaccept", tenant="tenant-a")
    remediation = kb.create_task(workflow_db, title="remediate", tenant="tenant-a")
    reverification = kb.create_task(workflow_db, title="reverify", tenant="tenant-a")
    actor = _admin_actor()
    created = kb.create_workflow(
        workflow_db, workflow_id="wf_reopen", name="release", tenant="tenant-a",
        designated_acceptance_task_id=acceptance, actor=actor, mutation_id="create",
    )
    cancelled = kb.cancel_workflow(
        workflow_db, workflow_id="wf_reopen", actor=actor, mutation_id="cancel",
        expected_version=created["workflow"]["version"], reason="new evidence",
    )
    with pytest.raises(kb.WorkflowConflictError, match="terminal"):
        kb.add_workflow_member(
            workflow_db, workflow_id="wf_reopen", task_id=replacement,
            stage_key="retry", stage_role="reverification", required=True, actor=actor,
            mutation_id="late-add", expected_version=cancelled["workflow"]["version"],
        )
    reopened = kb.reopen_workflow(
        workflow_db, workflow_id="wf_reopen", designated_acceptance_task_id=replacement,
        members=[
            {"task_id": replacement, "stage_key": "acceptance", "stage_role": "acceptance", "required": True},
            {"task_id": remediation, "stage_key": "remediation", "stage_role": "remediation", "required": True},
            {"task_id": reverification, "stage_key": "reverification", "stage_role": "reverification", "required": True},
        ], actor=actor, mutation_id="reopen",
        expected_version=cancelled["workflow"]["version"], reason="retry acceptance",
    )
    assert reopened["workflow"]["state"] == "ACTIVE"
    assert reopened["workflow"]["active_generation"] == 2
    assert workflow_db.execute("SELECT generation_state FROM kanban_workflow_generations WHERE workflow_id='wf_reopen' AND generation=1").fetchone()[0] == "CANCELLED"


def test_workflow_delivery_completion_only_acknowledges_its_claim(workflow_db):
    """An older send cannot clear a newer failed delivery's retry state."""
    acceptance = kb.create_task(workflow_db, title="accept", tenant="tenant-a")
    replacement = kb.create_task(workflow_db, title="replacement", tenant="tenant-a")
    remediation = kb.create_task(workflow_db, title="remediate", tenant="tenant-a")
    reverification = kb.create_task(workflow_db, title="reverify", tenant="tenant-a")
    actor = _admin_actor()
    created = kb.create_workflow(
        workflow_db, workflow_id="wf_delivery_claims", name="release", tenant="tenant-a",
        designated_acceptance_task_id=acceptance, actor=actor, mutation_id="create",
    )
    workflow_db.execute(
        "INSERT INTO kanban_workflow_subscriptions "
        "(workflow_id,role,platform,chat_id,notifier_profile,target_states,tenant,created_at,last_event_id) "
        "VALUES (?,'origin','telegram','chat','default','[\"CANCELLED\"]','tenant-a',0,?)",
        ("wf_delivery_claims", created["workflow"]["last_event_id"]),
    )
    cancelled = kb.cancel_workflow(
        workflow_db, workflow_id="wf_delivery_claims", actor=actor, mutation_id="cancel-one",
        expected_version=created["workflow"]["version"], reason="first cancellation",
    )
    reopened = kb.reopen_workflow(
        workflow_db, workflow_id="wf_delivery_claims", designated_acceptance_task_id=replacement,
        members=[
            {"task_id": replacement, "stage_key": "acceptance",
             "stage_role": "acceptance", "required": True},
            {"task_id": remediation, "stage_key": "remediation",
             "stage_role": "remediation", "required": True},
            {"task_id": reverification, "stage_key": "reverification",
             "stage_role": "reverification", "required": True},
        ], actor=actor, mutation_id="reopen",
        expected_version=cancelled["workflow"]["version"], reason="retry",
    )
    kb.cancel_workflow(
        workflow_db, workflow_id="wf_delivery_claims", actor=actor, mutation_id="cancel-two",
        expected_version=reopened["workflow"]["version"], reason="second cancellation",
    )

    old_first, first_cursor, first_events = kb.claim_workflow_events_for_subscription(
        workflow_db, workflow_id="wf_delivery_claims",
    )
    old_second, second_cursor, second_events = kb.claim_workflow_events_for_subscription(
        workflow_db, workflow_id="wf_delivery_claims",
    )
    assert first_events and second_events
    assert old_second == first_cursor
    assert second_cursor > first_cursor > old_first

    failed = kb.fail_workflow_delivery(
        workflow_db, workflow_id="wf_delivery_claims", claimed_cursor=second_cursor,
        old_cursor=old_second, error_class="AdapterError", retry_delay_seconds=0,
    )
    failed_state = {
        key: failed[key]
        for key in ("last_event_id", "retry_count", "next_attempt_at", "dead_lettered_at", "last_error_class")
    }
    assert failed_state["retry_count"] == 1
    assert failed_state["next_attempt_at"] is not None

    assert kb.complete_workflow_delivery(
        workflow_db, workflow_id="wf_delivery_claims", claimed_cursor=first_cursor,
    ) == failed
    assert dict(workflow_db.execute(
        "SELECT last_event_id,retry_count,next_attempt_at,dead_lettered_at,last_error_class "
        "FROM kanban_workflow_subscriptions WHERE workflow_id='wf_delivery_claims'"
    ).fetchone()) == failed_state

    retry_old_cursor, retry_cursor, retry_events = kb.claim_workflow_events_for_subscription(
        workflow_db, workflow_id="wf_delivery_claims",
    )
    assert retry_old_cursor == old_second
    assert retry_cursor == second_cursor
    assert retry_events
    completed = kb.complete_workflow_delivery(
        workflow_db, workflow_id="wf_delivery_claims", claimed_cursor=retry_cursor,
    )
    assert completed["last_event_id"] == retry_cursor
    assert completed["retry_count"] == 0
    assert completed["next_attempt_at"] is None
    assert completed["dead_lettered_at"] is None
    assert completed["last_error_class"] is None

    assert kb.complete_workflow_delivery(
        workflow_db, workflow_id="wf_delivery_claims", claimed_cursor=old_first,
    ) == completed
