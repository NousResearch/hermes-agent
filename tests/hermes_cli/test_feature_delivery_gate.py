from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.feature_delivery import (
    ACCEPTANCE_FINAL_MARKER,
    FEATURE_DELIVERY_WORKFLOW,
    AcceptanceReport,
    DeliveryCommitContext,
    DeveloperReport,
    FeatureDeliveryState,
    TaskContract,
    compute_contract_hash,
    evaluate_delivery_gate,
    invalidate_downstream_evidence_on_new_developer_commit,
    validate_delivery_commit_integrity,
)


SHA_A = "a" * 40
SHA_B = "b" * 40


def make_contract(**changes) -> TaskContract:
    data = {
        "task_id": "task-1",
        "title": "Feature delivery",
        "objective": "Gate delivery",
        "repository": "hermes-agent",
        "base_commit": SHA_A,
        "branch": "feature/feature-delivery-v1",
        "acceptance_criteria": [
            {"id": "AC-1", "requirement": "Tests pass"},
            {"id": "AC-2", "requirement": "Gate passes"},
        ],
        "constraints": ["No runner"],
        "required_tests": ["feature tests"],
        "required_evidence": ["tests", "diff-check"],
        "out_of_scope": ["deploy"],
        "delivery_gate": "acceptance_agent",
    }
    data.update(changes)
    return TaskContract.model_validate(data)


def make_acceptance(**changes) -> AcceptanceReport:
    data = {
        "task_id": "task-1",
        "agent": "acceptance",
        "accepted_commit": SHA_A,
        "status": "ACCEPT",
        "criteria": [
            {"id": "AC-1", "met": True, "evidence": "tests"},
            {"id": "AC-2", "met": True, "evidence": "review"},
        ],
        "blocking_issues": [],
        "evidence": ["tests", "diff-check"],
        "final_marker": ACCEPTANCE_FINAL_MARKER,
    }
    data.update(changes)
    return AcceptanceReport.model_validate(data)


def make_context(**changes) -> DeliveryCommitContext:
    data = {
        "developer_commit": SHA_A,
        "tested_commit": SHA_A,
        "accepted_commit": SHA_A,
        "branch_head": SHA_A,
    }
    data.update(changes)
    return DeliveryCommitContext.model_validate(data)


def gate(**changes):
    contract = changes.pop("contract", make_contract())
    report = changes.pop("acceptance_report", make_acceptance())
    context = changes.pop("commit_context", make_context())
    defaults = {
        "workflow_template_id": FEATURE_DELIVERY_WORKFLOW,
        "current_state": FeatureDeliveryState.ACCEPTANCE,
        "expected_contract_hash": compute_contract_hash(contract),
    }
    defaults.update(changes)
    return evaluate_delivery_gate(contract, report, context, **defaults)


def test_delivery_gate_allows_fully_accepted_commit():
    result = gate()
    assert result.allowed
    assert result.reasons == ()


def test_delivery_gate_rejects_missing_acceptance_criterion():
    report = make_acceptance(criteria=[
        {"id": "AC-1", "met": True, "evidence": "tests"},
    ])
    result = gate(acceptance_report=report)
    assert not result.allowed
    assert any("missing acceptance criteria" in reason for reason in result.reasons)


def test_delivery_gate_rejects_unknown_acceptance_criterion():
    report = make_acceptance(criteria=[
        {"id": "AC-1", "met": True, "evidence": "tests"},
        {"id": "AC-2", "met": True, "evidence": "review"},
        {"id": "AC-X", "met": True, "evidence": "unknown"},
    ])
    result = gate(acceptance_report=report)
    assert not result.allowed
    assert any("unknown acceptance criteria" in reason for reason in result.reasons)


def test_delivery_gate_rejects_unmet_acceptance_criterion():
    report = make_acceptance(criteria=[
        {"id": "AC-1", "met": False, "evidence": "failed"},
        {"id": "AC-2", "met": True, "evidence": "review"},
    ])
    result = gate(acceptance_report=report)
    assert not result.allowed
    assert any("unmet acceptance criteria" in reason for reason in result.reasons)


def test_delivery_gate_rejects_missing_required_evidence():
    result = gate(acceptance_report=make_acceptance(evidence=["tests"]))
    assert not result.allowed
    assert any("diff-check" in reason for reason in result.reasons)


def test_stage_evidence_can_cover_required_evidence():
    report = make_acceptance(evidence=["tests"])
    assert gate(acceptance_report=report, stage_evidence=["diff-check"]).allowed


def test_delivery_gate_rejects_contract_hash_mismatch():
    result = gate(expected_contract_hash="0" * 64)
    assert not result.allowed
    assert "contract hash changed" in result.reasons


def test_delivery_gate_rejects_wrong_workflow():
    result = gate(workflow_template_id="ordinary")
    assert not result.allowed
    assert any("workflow" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "state",
    [
        FeatureDeliveryState.DEVELOPING,
        FeatureDeliveryState.READY_FOR_TEST,
        FeatureDeliveryState.TESTING,
        FeatureDeliveryState.TEST_FAILED,
        FeatureDeliveryState.TEST_PASSED,
        FeatureDeliveryState.REJECTED,
    ],
)
def test_delivery_gate_rejects_non_acceptance_state(state):
    result = gate(current_state=state)
    assert not result.allowed
    assert any("current state" in reason for reason in result.reasons)


def test_delivery_gate_rejects_developer_report_even_with_accept_text():
    report = DeveloperReport(
        task_id="task-1",
        agent="developer",
        status="READY_FOR_TEST",
        commit=SHA_A,
        implementation_summary="FINAL: ACCEPT",
    )
    result = gate(acceptance_report=report)
    assert not result.allowed
    assert any("acceptance report" in reason for reason in result.reasons)


def test_delivery_gate_rejects_mismatched_task():
    result = gate(acceptance_report=make_acceptance(task_id="task-2"))
    assert not result.allowed
    assert any("task" in reason for reason in result.reasons)


def test_delivery_gate_rejects_rejection_report():
    report = make_acceptance(status="REJECT", final_marker=None)
    result = gate(acceptance_report=report)
    assert not result.allowed
    assert any("status" in reason for reason in result.reasons)


def test_commit_integrity_accepts_one_commit():
    assert validate_delivery_commit_integrity(make_context()).allowed


def test_commit_integrity_rejects_tested_mismatch():
    result = validate_delivery_commit_integrity(make_context(tested_commit=SHA_B))
    assert not result.allowed
    assert any("tested commit" in reason for reason in result.reasons)


def test_commit_integrity_rejects_accepted_mismatch():
    result = validate_delivery_commit_integrity(make_context(accepted_commit=SHA_B))
    assert not result.allowed
    assert any("accepted commit" in reason for reason in result.reasons)


def test_commit_integrity_rejects_branch_head_mismatch():
    result = validate_delivery_commit_integrity(make_context(branch_head=SHA_B))
    assert not result.allowed
    assert any("branch HEAD" in reason for reason in result.reasons)


def test_commit_integrity_rejects_missing_tested_commit():
    result = validate_delivery_commit_integrity(make_context(tested_commit=None))
    assert not result.allowed
    assert "tested commit is missing" in result.reasons


def test_commit_integrity_rejects_missing_accepted_commit():
    result = validate_delivery_commit_integrity(make_context(accepted_commit=None))
    assert not result.allowed
    assert "accepted commit is missing" in result.reasons


def test_delivery_gate_rejects_report_commit_mismatch():
    report = make_acceptance(accepted_commit=SHA_B)
    result = gate(acceptance_report=report)
    assert not result.allowed
    assert any("report commit" in reason for reason in result.reasons)


def test_new_developer_commit_clears_tested_commit():
    updated = invalidate_downstream_evidence_on_new_developer_commit(make_context(), SHA_B)
    assert updated.developer_commit == SHA_B
    assert updated.tested_commit is None


def test_new_developer_commit_clears_accepted_commit():
    updated = invalidate_downstream_evidence_on_new_developer_commit(make_context(), SHA_B)
    assert updated.accepted_commit is None
    assert updated.branch_head == SHA_B


def test_unchanged_developer_commit_preserves_evidence():
    context = make_context()
    assert invalidate_downstream_evidence_on_new_developer_commit(context, SHA_A) is context


@pytest.fixture
def workflow_task(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = home / "kanban.db"
    conn = kb.connect(db_path)
    task_id = kb.create_task(conn, title="feature delivery task")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workflow_template_id = ?, current_step_key = ? WHERE id = ?",
            (FEATURE_DELIVERY_WORKFLOW, "ACCEPTANCE", task_id),
        )
    try:
        yield conn, task_id
    finally:
        conn.close()


def test_workflow_cas_updates_expected_state(workflow_task):
    conn, task_id = workflow_task
    original_status = kb.get_task(conn, task_id).status
    assert kb.transition_workflow_step_cas(
        conn,
        task_id=task_id,
        workflow_template_id=FEATURE_DELIVERY_WORKFLOW,
        expected_step="ACCEPTANCE",
        new_step="DELIVERED",
    )
    updated = kb.get_task(conn, task_id)
    assert updated.current_step_key == "DELIVERED"
    assert updated.status == original_status


def test_workflow_cas_rejects_stale_expected_state(workflow_task):
    conn, task_id = workflow_task
    assert kb.transition_workflow_step_cas(
        conn,
        task_id=task_id,
        workflow_template_id=FEATURE_DELIVERY_WORKFLOW,
        expected_step="ACCEPTANCE",
        new_step="DELIVERED",
    )
    assert not kb.transition_workflow_step_cas(
        conn,
        task_id=task_id,
        workflow_template_id=FEATURE_DELIVERY_WORKFLOW,
        expected_step="ACCEPTANCE",
        new_step="DELIVERED",
    )


def test_workflow_cas_writes_transition_event_once(workflow_task):
    conn, task_id = workflow_task
    assert kb.transition_workflow_step_cas(
        conn,
        task_id=task_id,
        workflow_template_id=FEATURE_DELIVERY_WORKFLOW,
        expected_step="ACCEPTANCE",
        new_step="DELIVERED",
        event_payload={"contract_hash": "hash", "from_step": "forged"},
    )
    assert not kb.transition_workflow_step_cas(
        conn,
        task_id=task_id,
        workflow_template_id=FEATURE_DELIVERY_WORKFLOW,
        expected_step="ACCEPTANCE",
        new_step="DELIVERED",
    )
    events = [event for event in kb.list_events(conn, task_id) if event.kind == "workflow_step_transitioned"]
    assert len(events) == 1
    assert events[0].payload == {
        "contract_hash": "hash",
        "workflow_template_id": FEATURE_DELIVERY_WORKFLOW,
        "from_step": "ACCEPTANCE",
        "to_step": "DELIVERED",
    }


def test_workflow_cas_rolls_back_step_when_event_write_fails(workflow_task, monkeypatch):
    conn, task_id = workflow_task

    def fail_event(*args, **kwargs):
        raise RuntimeError("event write failed")

    monkeypatch.setattr(kb, "_append_event", fail_event)
    with pytest.raises(RuntimeError, match="event write failed"):
        kb.transition_workflow_step_cas(
            conn,
            task_id=task_id,
            workflow_template_id=FEATURE_DELIVERY_WORKFLOW,
            expected_step="ACCEPTANCE",
            new_step="DELIVERED",
        )
    assert kb.get_task(conn, task_id).current_step_key == "ACCEPTANCE"


def test_workflow_cas_leaves_ordinary_kanban_task_unchanged(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    conn = kb.connect(home / "ordinary.db")
    try:
        task_id = kb.create_task(conn, title="ordinary task")
        before = kb.get_task(conn, task_id)
        assert not kb.transition_workflow_step_cas(
            conn,
            task_id=task_id,
            workflow_template_id=FEATURE_DELIVERY_WORKFLOW,
            expected_step="NEW",
            new_step="CONTRACT_READY",
        )
        after = kb.get_task(conn, task_id)
        assert after.status == before.status
        assert after.workflow_template_id is None
        assert after.current_step_key is None
        assert not any(
            event.kind == "workflow_step_transitioned"
            for event in kb.list_events(conn, task_id)
        )
    finally:
        conn.close()
