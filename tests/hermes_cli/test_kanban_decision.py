"""Behavior contracts for the single Kanban next-action decision."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_decision import (
    auto_decompose_allowed,
    claim_allowed,
    decide,
    record_decision,
)


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


def _task(conn, title, *, status="ready", eligible=True, assignee="worker"):
    tid = kb.create_task(
        conn, title=title, body="Acceptance: one bounded action.", assignee=assignee,
        workspace_kind="scratch", initial_status="running",
    )
    conn.execute("UPDATE tasks SET status=?,dispatch_eligible=? WHERE id=?",
                 (status, int(eligible), tid))
    return tid


def _contract(conn, tid, **extra):
    payload = {
        "version": 1, "kind": "implementation", "local_only": True,
        "require_review": True, "require_commands": True,
        "criteria": ["authoritative_scope_satisfied"],
        "qualified_for_dispatch": True,
        "scope": "bounded fixture", "scope_sha256": "fixture",
        **extra,
    }
    kb._append_event(conn, tid, "completion_requirements", payload)
    return payload


def _accepted_output(conn, tid):
    from hermes_cli.kanban_completion_evidence import contract_record
    from hermes_cli.kanban_dependency_output import certify_output
    artifact = Path(conn.execute('PRAGMA database_list').fetchone()[2]).parent / (tid + '-output.md')
    artifact.write_text('A bounded retained producer output.\n')
    aid = kb.add_attachment(conn, tid, filename=artifact.name, stored_path=str(artifact))
    cid, contract = contract_record(conn, tid)
    return certify_output(conn, tid, contract_event_id=cid, artifact_attachment_id=aid, criterion_id=contract['criteria'][0],
                          evidence={'reference': 'qualified retained review', 'assertions': {'output_verified': True, 'applicability_verified': True, 'authority_verified': True}})


def test_t_6a127187_waiting_authority_is_never_claimed_or_decomposed(board):
    tid = _task(board, "pruning live A/B", status="ready")
    _contract(board, tid, selected_next_action={
        "phase": "WAITING_FOR_AUTHORITY", "type": "live_experiment",
        "action": "Grant the live pruning A/B window", "resolver": "Adrian",
        "resume_condition": "operator grant is recorded",
    })
    task = kb.get_task(board, tid)
    decision = decide(board, task)
    assert decision["workflow_stage"] == "WAITING"
    assert decision["dispatchable"] is False
    assert decision["auto_decompose_allowed"] is False
    assert claim_allowed(board, task, "ready")[0] is False


def test_t_6a127187_authority_grant_atomically_releases_existing_card(board):
    tid = _task(board, "pruning live A/B", status="ready")
    contract = _contract(board, tid, selected_next_action={
        "phase": "WAITING_FOR_AUTHORITY", "type": "live_experiment",
        "action": "Grant the live pruning A/B window", "resolver": "Adrian",
        "resume_condition": "operator grant is recorded",
    })
    assert kb.recompute_ready(board) == 1
    waiting = kb.get_task(board, tid)
    assert (waiting.status, waiting.dispatch_eligible) == ("todo", True)
    assert decide(board, waiting)["workflow_stage"] == "WAITING"

    released_contract = dict(contract)
    released_contract["selected_next_action"] = {
        "phase": "READY", "type": "worker_action",
        "action": "Run the authorised live pruning A/B",
    }
    kb._append_event(board, tid, "completion_requirements", released_contract)
    kb._append_event(board, tid, "operator_authority_granted", {
        "authority": "live pruning A/B window",
    })
    assert kb.recompute_ready(board) == 1
    released = kb.get_task(board, tid)
    assert (released.status, released.dispatch_eligible) == ("ready", True)
    assert decide(board, released)["workflow_stage"] == "READY"


def test_legacy_todo_disabled_wait_is_normalized_then_released(board):
    tid = _task(board, "legacy disabled authority wait", status="todo", eligible=False)
    contract = _contract(board, tid, selected_next_action={
        "phase": "WAITING_FOR_AUTHORITY", "type": "live_experiment",
        "action": "Grant the live experiment window", "resolver": "Adrian",
        "resume_condition": "operator grant is recorded",
    })
    assert decide(board, kb.get_task(board, tid))["workflow_stage"] == "WAITING"
    assert kb.recompute_ready(board) == 1
    normalized = kb.get_task(board, tid)
    assert (normalized.status, normalized.dispatch_eligible) == ("todo", True)

    released_contract = dict(contract)
    released_contract["selected_next_action"] = {
        "phase": "READY", "type": "worker_action", "action": "Run the experiment",
    }
    kb._append_event(board, tid, "completion_requirements", released_contract)
    kb._append_event(board, tid, "operator_authority_granted", {"authority": "window"})
    assert kb.recompute_ready(board) == 1
    released = kb.get_task(board, tid)
    assert (released.status, released.dispatch_eligible) == ("ready", True)


def test_authority_wait_semantics_outrank_execution_health(board, monkeypatch):
    import hermes_cli.kanban_decision as decision_module

    tid = _task(board, "authority-gated action with old infra fault", status="ready")
    _contract(board, tid, selected_next_action={
        "phase": "WAITING_FOR_AUTHORITY", "type": "live_experiment",
        "action": "Grant authority", "resolver": "Adrian",
        "resume_condition": "authority is recorded",
    })
    now = int(time.time())
    board.execute(
        "INSERT INTO task_runs(task_id,profile,status,started_at,ended_at,outcome,error,metadata) "
        "VALUES(?,?,'crashed',?,?, 'crashed',?,?)",
        (tid, "worker", now - 2, now - 1,
         "no dependency environment is committed for this install",
         json.dumps({"infrastructure": True, "environment_fingerprint": "same"})),
    )
    monkeypatch.setattr(decision_module, "environment_fingerprint", lambda: {
        "identity": "same", "healthy": False,
    })
    decision = decide(board, kb.get_task(board, tid))
    assert decision["workflow_stage"] == "WAITING"
    assert decision["owner"] == "Adrian"
    assert decision["next_action"] == "Grant authority"
    assert decision["resume_condition"] == "authority is recorded"
    assert decision["execution_health"]["state"] == "INFRASTRUCTURE_FAULT"


def test_explicit_stop_outranks_unfinished_dependency(board):
    parent = _task(board, "unfinished prerequisite", status="todo")
    stopped = _task(board, "operator stopped programme", status="blocked", eligible=False)
    board.execute("UPDATE tasks SET block_kind='explicit_stop' WHERE id=?", (stopped,))
    _contract(board, parent)
    _contract(board, stopped, selected_next_action={
        "action": "run the programme", "type": "worker_action",
    })
    kb.link_tasks(board, parent, stopped)
    decision = decide(board, kb.get_task(board, stopped))
    assert decision["workflow_stage"] == "HELD"
    assert decision["block_kind"] == "explicit_stop"
    assert decision["dispatchable"] is False
    assert decision["resume_condition"] == "operator explicitly revokes the stop"


def test_substantive_operator_decision_is_not_receipt_administration(board):
    tid = _task(board, "choose a private connection boundary", status="blocked", eligible=False)
    _contract(board, tid, selected_next_action={
        "action": "configure the private connection", "type": "operator_action",
    })
    kb._append_event(board, tid, "operator_decision_pending", {
        "reason": "Private connection authority and secure endpoint are undecided",
        "resolver": "Adrian",
        "next_action": "Choose the private connection boundary",
        "resume_condition": "operator records the authorised boundary",
    })
    decision = decide(board, kb.get_task(board, tid))
    assert decision["workflow_stage"] == "OPERATOR"
    assert decision["owner"] == "Adrian"
    assert decision["dispatchable"] is False
    assert decision.get("administrative_pending") is not True

    kb._append_event(board, tid, "operator_decision_resolved", {"decision": "approved"})
    assert decide(board, kb.get_task(board, tid))["workflow_stage"] == "OPERATOR"
    from hermes_cli.kanban_wait_migration import migrate
    from hermes_cli.kanban_wait_contract import facts
    from hermes_cli.kanban_wait_transitions import resolve_gate
    migrate(board, [tid])
    gate = facts(board, tid)["remaining_gates"][0]
    released = resolve_gate(board, tid, gate["id"], revision=gate["revision"], declaration=gate["declaration"],
                            evidence={"sha256": "a"*64, "reference": "retained authorised choice", "assertions": {"decision_authorised": True, "selected_continuation_authorised": True}}, decision={"selection": "approved"})
    assert released["workflow_stage"] == "READY"
    assert released["dispatchable"] is True


def test_t_e5d24de5_review_flag_and_claim_guard_cannot_disagree(board):
    tid = _task(board, "dashboard review", status="review", eligible=False)
    _contract(board, tid)
    task = kb.get_task(board, tid)
    decision = decide(board, task)
    assert decision["workflow_stage"] == "WORKFLOW_FAULT"
    assert decision["dispatchable"] is False
    assert any("claim rejects" in item for item in decision["invariant_violations"])
    assert claim_allowed(board, task, "review")[0] is False


def test_t_ebcf430c_infrastructure_fault_does_not_change_review_semantics(board):
    import hermes_cli.kanban_decision as decision_module

    original = decision_module.environment_fingerprint
    decision_module.environment_fingerprint = lambda: {
        "identity": "broken-env", "healthy": False,
    }
    tid = _task(board, "independent MoA review", status="review")
    _contract(board, tid, kind="review", require_review=False, require_commands=False)
    now = int(time.time())
    board.execute(
        "INSERT INTO task_runs(task_id,profile,status,started_at,ended_at,outcome,error) "
        "VALUES(?,?,'crashed',?,?, 'spawn_failed',?)",
        (tid, "reviewer", now - 2, now - 1,
         "no dependency environment is committed for this install"),
    )
    try:
        decision = decide(board, kb.get_task(board, tid))
    finally:
        decision_module.environment_fingerprint = original
    assert decision["workflow_stage"] == "REVIEW"
    assert decision["execution_health"]["state"] == "INFRASTRUCTURE_FAULT"
    assert decision["dispatchable"] is False
    assert decision["resume_condition"] == "execution environment fingerprint changes"
    assert decision["evaluation_mode"] == "automatically_observed"
    assert decision["state_if_satisfied_now"]["state"] == "REVIEW"
    assert decision["state_if_satisfied_now"]["dispatchable"]
    assert decision["remaining_gates"][0]["evaluator"] == "execution_environment_changed"


def test_t_ebcf430c_environment_repair_releases_same_review(board, monkeypatch):
    import hermes_cli.kanban_decision as decision_module

    tid = _task(board, "independent MoA review after repair", status="review")
    _contract(board, tid, kind="review", require_review=False, require_commands=False)
    now = int(time.time())
    board.execute(
        "INSERT INTO task_runs(task_id,profile,status,started_at,ended_at,outcome,error,metadata) "
        "VALUES(?,?,'crashed',?,?, 'crashed',?,?)",
        (tid, "reviewer", now - 2, now - 1,
         "no dependency environment is committed for this install",
         json.dumps({"infrastructure": True, "environment_fingerprint": "old"})),
    )
    monkeypatch.setattr(decision_module, "environment_fingerprint", lambda: {
        "identity": "repaired", "healthy": True,
    })
    decision = decide(board, kb.get_task(board, tid))
    assert decision["workflow_stage"] == "REVIEW"
    assert decision["execution_health"]["state"] == "HEALTHY"
    assert decision["dispatchable"] is True


def test_infrastructure_fault_recompute_does_not_self_seal_ready(board, monkeypatch):
    import hermes_cli.kanban_decision as decision_module

    tid = _task(board, "ready action during infrastructure fault", status="ready")
    _contract(board, tid, selected_next_action={
        "phase": "READY", "type": "worker_action", "action": "Run the action",
    })
    now = int(time.time())
    board.execute(
        "INSERT INTO task_runs(task_id,profile,status,started_at,ended_at,outcome,error,metadata) "
        "VALUES(?,?,'crashed',?,?, 'crashed',?,?)",
        (tid, "worker", now - 2, now - 1,
         "no dependency environment is committed for this install",
         json.dumps({"infrastructure": True, "environment_fingerprint": "broken"})),
    )
    monkeypatch.setattr(decision_module, "environment_fingerprint", lambda: {
        "identity": "broken", "healthy": False,
    })
    fault = decide(board, kb.get_task(board, tid))
    assert fault["workflow_stage"] == "READY"
    assert fault["dispatchable"] is False
    assert kb.recompute_ready(board) == 0
    assert kb.get_task(board, tid).dispatch_eligible is True

    monkeypatch.setattr(decision_module, "environment_fingerprint", lambda: {
        "identity": "repaired", "healthy": True,
    })
    recovered = decide(board, kb.get_task(board, tid))
    assert recovered["workflow_stage"] == "READY"
    assert recovered["dispatchable"] is True
    assert kb.recompute_ready(board) == 0
    assert kb.get_task(board, tid).dispatch_eligible is True


def test_t_8ff8ec92_specific_policy_output_releases_experiment_without_parent_done(board):
    parent = _task(board, "produce shared policy", status="ready")
    child = _task(board, "run experiment", status="todo")
    _contract(board, parent)
    _contract(board, child, selected_next_action={
        "action": "run the experiment", "type": "worker_action",
    })
    kb.link_tasks(board, parent, child)
    from hermes_cli.kanban_dependency_output import declare, accept
    output = _accepted_output(board, parent)
    declare(board, child, parent, 'policy-v1', evidence_identity=output["evidence_identity"], requirement="accepted producer output")
    declaration_id = board.execute("SELECT max(id) FROM task_events WHERE task_id=? AND kind='dependency_requirement'", (child,)).fetchone()[0]
    assert not decide(board, kb.get_task(board, child))["dispatchable"]
    output_id = output["producer_evidence_event_id"]
    accept(board, child, 'policy-v1', declaration_event_id=declaration_id, producer_evidence_event_id=output_id)
    released = decide(board, kb.get_task(board, child), status_override="ready")
    assert released["workflow_stage"] == "READY"
    assert released["dispatchable"] is True
    assert kb.get_task(board, parent).status not in {"done", "archived"}


def test_t_e07ae2d6_review_approval_closes_mechanically_without_receipt_ceremony(board):
    tid = _task(board, "MoA route status implementation", status="review")
    _contract(board, tid, kind="review", require_review=False, require_commands=False)
    assert kb.complete_task(board, tid, result="APPROVED", force=True) is True
    task = kb.get_task(board, tid)
    assert task.status == "done"
    assert decide(board, task)["workflow_stage"] == "DONE"


def test_t_1d2e2334_upstream_completion_recomputes_each_dependant(board):
    parent = _task(board, "package provisioning", status="ready")
    ready_child = _task(board, "consumer with action", status="todo")
    waiting_child = _task(board, "consumer waiting on window", status="todo")
    _contract(
        board, parent, kind="research", require_review=False, require_commands=False,
        accepted_completed_actions=["existing environments falsify missing-package premise"],
        remaining_required_actions=[],
    )
    _contract(board, ready_child, selected_next_action={"action": "use packages", "type": "worker_action"})
    _contract(
        board, waiting_child,
        selected_next_action={"action": "run live experiment", "type": "worker_action"},
        external_conditions=[{
            "condition_id": "window", "requirement": "exclusive window granted",
            "resolver": "operator", "resume_condition": "window grant is recorded",
            "satisfied": False,
        }],
    )
    kb.link_tasks(board, parent, ready_child)
    kb.link_tasks(board, parent, waiting_child)
    assert kb.recompute_ready(board) >= 1
    assert kb.get_task(board, parent).status == "done"
    assert kb.get_task(board, ready_child).status == "ready"
    assert kb.get_task(board, waiting_child).status == "todo"
    assert decide(board, kb.get_task(board, waiting_child))["workflow_stage"] == "WAITING"


def test_t_29ca9c21_authorised_missing_checkout_is_provisioned_directly(tmp_path):
    from hermes_cli.kanban_preparation import provision_authorised_workspace

    origin = tmp_path / "origin"
    origin.mkdir()
    subprocess.run(["git", "init", "-q", str(origin)], check=True)
    subprocess.run(["git", "-C", str(origin), "config", "user.email", "fixture@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(origin), "config", "user.name", "Fixture"], check=True)
    (origin / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(origin), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(origin), "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(origin), "rev-parse", "HEAD"], text=True,
    ).strip()
    destination = tmp_path / "qualified-checkout"
    ok, reason = provision_authorised_workspace({
        "provision_missing_workspace": True,
        "isolated_workspace": True,
        "canonical_remote": str(origin),
        "source_revision": revision,
        "workspace_path": str(destination),
    })
    assert (ok, reason) == (True, "")
    assert subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True,
    ).strip() == revision


def test_revised_dependency_invalidates_stale_satisfaction(board):
    parent = _task(board, "artifact producer", status="ready")
    child = _task(board, "artifact consumer", status="todo")
    _contract(board, parent)
    _contract(board, child, selected_next_action={"action": "consume artifact", "type": "worker_action"})
    kb.link_tasks(board, parent, child)
    from hermes_cli.kanban_dependency_output import declare, accept
    output = _accepted_output(board, parent)
    declare(board, child, parent, 'artifact', evidence_identity=output["evidence_identity"], requirement="accepted producer output")
    declaration_id = board.execute("SELECT max(id) FROM task_events WHERE task_id=? AND kind='dependency_requirement'", (child,)).fetchone()[0]
    assert not decide(board, kb.get_task(board, child))["dispatchable"]
    output_id = output["producer_evidence_event_id"]
    accept(board, child, 'artifact', declaration_event_id=declaration_id, producer_evidence_event_id=output_id)
    assert decide(board, kb.get_task(board, child), status_override="ready")["dispatchable"] is True
    kb._append_event(board, child, "dependency_requirement", {
        "parent_id": parent, "requirement_id": "artifact", "evidence_identity": "sha256:v2",
    })
    revised = decide(board, kb.get_task(board, child), status_override="ready")
    assert revised["workflow_stage"] == "WAITING"
    assert revised["dispatchable"] is False


def test_unsatisfied_external_condition_gates_selected_worker_action(board):
    tid = _task(board, "NotebookLM retrieval", status="ready")
    _contract(
        board, tid,
        selected_next_action={"action": "retrieve NotebookLM sources", "type": "worker_action"},
        external_conditions=[{
            "condition_id": "google-auth", "type": "external_authentication",
            "requirement": "Google authentication is valid", "resolver": "Adrian",
            "resume_condition": "nlm login --check succeeds", "satisfied": False,
        }],
    )
    decision = decide(board, kb.get_task(board, tid))
    assert decision["workflow_stage"] == "WAITING"
    assert decision["dispatchable"] is False


def test_completed_scope_without_selected_action_never_manufactures_work(board):
    tid = _task(board, "accepted bounded work", status="ready")
    _contract(
        board, tid, accepted_completed_actions=["bounded scope accepted"],
        remaining_required_actions=[],
    )
    decision = decide(board, kb.get_task(board, tid))
    assert decision["workflow_stage"] == "DONE"
    assert decision["dispatchable"] is False
    assert decision["next_action"] is None


def test_completed_scope_ignores_stale_selected_action(board):
    """Accepted terminal truth outranks obsolete action metadata."""
    tid = _task(board, "accepted work with stale action", status="ready")
    _contract(
        board, tid,
        accepted_completed_actions=["bounded scope accepted"],
        remaining_required_actions=[],
        selected_next_action={"action": "obsolete action", "type": "worker_action"},
    )
    decision = decide(board, kb.get_task(board, tid))
    assert decision["workflow_stage"] == "DONE"
    assert decision["dispatchable"] is False
    assert decision["next_action"] is None


def test_completed_truth_closes_stale_blocked_state(board):
    tid = _task(board, "accepted work left blocked", status="ready")
    _contract(board, tid)
    assert kb.block_task(
        board, tid, reason="operator input was previously needed", kind="needs_input",
    ) is True
    _contract(
        board, tid,
        accepted_completed_actions=["bounded scope accepted"],
        remaining_required_actions=[],
        selected_next_action={"action": "obsolete action", "type": "worker_action"},
    )
    decision = decide(board, kb.get_task(board, tid))
    assert decision["workflow_stage"] == "DONE"
    assert decision["dispatchable"] is False
    assert kb.recompute_ready(board) == 1
    assert kb.get_task(board, tid).status == "done"


def test_accepted_completion_never_overrides_explicit_stop(board):
    tid = _task(board, "accepted work under explicit stop", status="ready")
    _contract(board, tid)
    assert kb.block_task(
        board, tid, reason="operator prohibited further transition", kind="needs_input",
    ) is True
    # Explicit stop is an operator/migration state, not a worker-selectable
    # block_task kind. Preserve the real blocked event that supplies the
    # ordinary semantic resume override, then apply the authority state.
    board.execute(
        "UPDATE tasks SET block_kind='explicit_stop',dispatch_eligible=0 WHERE id=?", (tid,),
    )
    _contract(
        board, tid,
        accepted_completed_actions=["bounded scope accepted"],
        remaining_required_actions=[],
    )
    decision = decide(board, kb.get_task(board, tid))
    assert decision["workflow_stage"] == "HELD"
    assert decision["block_kind"] == "explicit_stop"
    assert kb.recompute_ready(board) == 0
    stopped = kb.get_task(board, tid)
    assert (stopped.status, stopped.block_kind, stopped.dispatch_eligible) == (
        "blocked", "explicit_stop", False,
    )


def test_recompute_preserves_native_scheduled_wait(board):
    """A future wake remains Scheduled rather than becoming held Todo."""
    tid = _task(board, "wait for maintenance window", status="ready", eligible=True)
    _contract(
        board, tid,
        selected_next_action={"action": "run maintenance", "type": "worker_action"},
        external_conditions=[{
            "condition_id": "window", "requirement": "maintenance window opens",
            "resolver": "scheduler", "resume_condition": "scheduled time arrives",
            "satisfied": False,
        }],
    )
    kb.schedule_task(board, tid, reason="maintenance window opens", scheduled_at=time.time()+600)
    before = decide(board, kb.get_task(board, tid))
    assert before["workflow_stage"] == "WAITING"
    assert kb.recompute_ready(board) == 0
    task = kb.get_task(board, tid)
    assert (task.status, task.dispatch_eligible) == ("scheduled", True)
    assert decide(board, task)["workflow_stage"] == "WAITING"


def test_unchanged_no_work_assessment_is_idempotent(board):
    tid = _task(board, "known future wait", status="triage")
    _contract(board, tid, remaining_required_actions=[])
    task = kb.get_task(board, tid)
    allowed, decision = auto_decompose_allowed(board, task)
    assert allowed is False
    with kbc.write_txn(board):
        assert record_decision(board, tid, decision, "no_worker_executable_action") is True
    allowed_again, same = auto_decompose_allowed(board, kb.get_task(board, tid))
    assert allowed_again is False
    assert same["decision_fingerprint"] == decision["decision_fingerprint"]
    with kbc.write_txn(board):
        assert record_decision(board, tid, same, "no_worker_executable_action") is False
    count = board.execute(
        "SELECT count(*) FROM task_events WHERE task_id=? AND kind='control_decision_recorded'",
        (tid,),
    ).fetchone()[0]
    assert count == 1


def test_done_with_remaining_actions_is_an_impossible_state(board):
    tid = _task(board, "bad terminal", status="done")
    _contract(board, tid, remaining_required_actions=["still do work"])
    decision = decide(board, kb.get_task(board, tid))
    assert decision["workflow_stage"] == "DONE"
    assert "DONE && remaining_required_actions" in decision["invariant_violations"]


def test_waiting_resume_condition_satisfaction_atomically_enables_existing_action(board, monkeypatch):
    """NotebookLM regression: successful auth cannot leave Ready + disabled."""
    from hermes_cli.kanban_resume import satisfy_resume_condition

    tid = _task(board, "retrieve NotebookLM sources", status="blocked", eligible=False)
    _contract(board, tid, kind="research", require_review=False, require_commands=True,
              selected_next_action={
                  "phase": "WAITING_FOR_AUTHORITY", "type": "external_authentication",
                  "action": "Authenticate Google profile default", "resolver": "Adrian",
                  "resume_condition": "nlm login --check succeeds",
                  "condition_id": "google-auth", "condition_probe": "notebooklm_auth",
                  "resume_next_action": "retrieve/export supplied NotebookLM sources",
              })
    before = decide(board, kb.get_task(board, tid))
    assert before["workflow_stage"] == "WAITING"

    import hermes_cli.kanban_resume as resume_module
    monkeypatch.setitem(resume_module._PROBES, "notebooklm_auth", lambda: {"probe": "notebooklm_auth", "exit_code": 0, "profile": "default"})
    after = satisfy_resume_condition(
        board, tid, condition="nlm login --check succeeds",
        evidence={"exit_code": 0, "profile": "default", "notebooks_visible": 479},
        next_action="retrieve/export supplied NotebookLM sources",
        expected_status="blocked", expected_decision_fingerprint=before["decision_fingerprint"],
        condition_id="google-auth",
    )
    task = kb.get_task(board, tid)
    assert (task.status, task.dispatch_eligible, task.block_kind) == ("ready", True, None)
    assert after["workflow_stage"] == "READY"
    assert after["dispatchable"] is True
    assert after["next_action"] == "retrieve/export supplied NotebookLM sources"


def test_dispatch_observer_notices_notebooklm_login_without_model_turn(board):
    from hermes_cli.kanban_resume import observe_resume_conditions

    tid = _task(board, "retrieve NotebookLM sources", status="blocked", eligible=False)
    _contract(board, tid, kind="research", require_review=False,
              selected_next_action={
                  "phase": "WAITING_FOR_AUTHORITY", "type": "external_authentication",
                  "action": "Authenticate Google profile default", "resolver": "Adrian",
                  "resume_condition": "nlm login --check succeeds",
                  "condition_probe": "notebooklm_auth", "condition_id": "google-auth",
                  "resume_next_action": "retrieve/export supplied NotebookLM sources",
              })
    resumed = observe_resume_conditions(
        board,
        probes={"notebooklm_auth": lambda: {"probe": "notebooklm_auth", "exit_code": 0, "profile": "default"}},
    )
    assert resumed == [tid]
    task = kb.get_task(board, tid)
    assert (task.status, task.dispatch_eligible) == ("ready", True)


def test_resume_rolls_back_if_no_worker_action_can_become_ready(board):
    from hermes_cli.kanban_resume import satisfy_resume_condition

    parent = _task(board, "still required", status="ready")
    child = _task(board, "waiting child", status="blocked", eligible=False)
    _contract(board, parent)
    _contract(board, child, kind="research", require_review=False, require_commands=False)
    kb.link_tasks(board, parent, child)
    with pytest.raises(ValueError, match="No matching native condition"):
        # A free-text observation cannot choose or satisfy this card's real gate.
        satisfy_resume_condition(
            board, child, condition="authentication valid", evidence={"exit_code": 0},
            next_action="retrieve sources", expected_status="blocked",
        )
    task = kb.get_task(board, child)
    assert (task.status, task.dispatch_eligible) == ("blocked", False)


def test_scope_contract_intake_triage_card_is_auto_specifiable(board):
    """Intake regression (t_9c2c5733): the dashboard materialises a deterministic
    scope contract at creation (control_plane_pre_promotion) whose single
    remaining requirement is the plain scope STRING, equal to
    selected_next_action.action. That shape must be auto-decomposable, or every
    freshly-intaked card sits in triage forever (0/179 dict requirements live)."""
    tid = _task(board, "Automatically triage new issues and move them to Todo",
                status="triage", eligible=True)
    _contract(
        board, tid,
        kind="research", require_review=False, require_commands=False,
        criteria=["explicit_completion_result_recorded"],
        remaining_required_actions=["Automatically triage new issues and move them to Todo"],
        selected_next_action={
            "phase": "READY", "type": "worker_action",
            "action": "Automatically triage new issues and move them to Todo",
            "required_authority": None,
        },
    )
    allowed, decision = auto_decompose_allowed(board, kb.get_task(board, tid))
    assert allowed is True, decision
    assert decision["workflow_stage"] == "TRIAGE"


def test_wide_reach_intake_card_with_children_is_not_auto_specifiable(board):
    """The string-scope allowance must not fan out a card already decomposed."""
    tid = _task(board, "parent with children", status="triage", eligible=True)
    _contract(board, tid, remaining_required_actions=["parent with children"])
    child = _task(board, "child", status="ready")
    kb.link_tasks(board, tid, child)
    allowed, _ = auto_decompose_allowed(board, kb.get_task(board, tid))
    assert allowed is False
