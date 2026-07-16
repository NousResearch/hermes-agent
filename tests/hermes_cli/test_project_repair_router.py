"""Focused behavior tests for HOF-013's bounded project repair router."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hermes_cli.project_failure_envelope import ProjectFailureEnvelope
from hermes_cli.project_finalizer import ProjectEvaluation
from hermes_cli.project_repair_router import (
    NO_ACTION,
    REPAIR_ALREADY_EXISTS,
    REPAIR_CREATED,
    BUDGET_EXHAUSTED,
    MALFORMED,
    STALE_SNAPSHOT,
    UNSUPPORTED_FAILURE,
    AtomicRepairRegistration,
    ProjectIdentity,
    ProjectRepairRequest,
    ProjectVersionToken,
    RepairMembership,
    route_project_repair,
)


class FakeAtomicRegistrar:
    """A restart-shareable stand-in for the atomic task/member boundary."""

    def __init__(self, durable=None, *, force_stale: bool = False):
        self.durable = durable if durable is not None else {}
        self.force_stale = force_stale
        self.calls = []

    def __call__(self, action, expected_token):
        self.calls.append((action, expected_token))
        if self.force_stale:
            return AtomicRepairRegistration("stale_snapshot")
        existing = self.durable.get(action.repair_identity)
        if existing is not None:
            return AtomicRepairRegistration("already_exists", repair_task_id=existing)
        task_id = f"repair-{len(self.durable) + 1}"
        self.durable[action.repair_identity] = task_id
        return AtomicRepairRegistration("created", repair_task_id=task_id)


def _project(*, generation: int = 2) -> ProjectIdentity:
    return ProjectIdentity(
        project_id="HERMES-ORCH-FINISH-001",
        board_id="board-a",
        root_task_id="root-a",
        generation=generation,
    )


def _evaluation(
    *,
    state: str = "REPAIRABLE",
    snapshot_version: str = "sha256:snapshot-a",
    repair_generation: int = 0,
    repair_budget: int = 2,
) -> ProjectEvaluation:
    return ProjectEvaluation(
        board_id="board-a",
        root_task_id="root-a",
        generation=2,
        snapshot_version=snapshot_version,
        evaluation_state=state,
        terminal_outcome=None,
        required_task_ids=("failed-task",),
        optional_task_ids=(),
        unfinished_task_ids=(),
        successful_task_ids=("failed-task",),
        blocked_task_ids=(),
        failed_task_ids=(),
        checker_task_id="failed-task",
        checker_verdict="FAIL_REPAIRABLE",
        repair_generation=repair_generation,
        repair_budget=repair_budget,
        repair_eligible=state == "REPAIRABLE",
        finalization_eligible=False,
        blocker=None,
        failure_reason="checker_fail_repairable",
        evidence_references=("task:failed-task", "run:failed-task:17"),
    )


def _failure(
    *,
    run_id: int | None = 17,
    fingerprint: str = "a" * 64,
    failure_class: str = "artifact_failure",
) -> ProjectFailureEnvelope:
    return ProjectFailureEnvelope(
        id=8,
        board_id="board-a",
        root_task_id="root-a",
        generation=2,
        task_id="failed-task",
        run_id=run_id,
        provider=None,
        model=None,
        failure_class=failure_class,
        status_code=None,
        retry_after=None,
        redacted_error="redacted checker failure",
        error_fingerprint=fingerprint,
        created_at=100,
    )


def _request(**overrides) -> ProjectRepairRequest:
    values = {
        "project": _project(),
        "evaluation": _evaluation(),
        "failed_task_id": "failed-task",
        "failed_run_id": 17,
        "failure_envelope": _failure(),
        "project_repair_budget": 2,
        "task_retry_limit": 1,
        "existing_repairs": (),
        "worker_profile": "builder-grok",
        "allowed_worker_profiles": ("builder-grok",),
        "task_contract": {
            "version": 1,
            "scope": "repair only",
            "allowed_files": ["app.py"],
            "allow_child_creation": False,
        },
        "notification_route_identities": ("subscription:project-owner",),
        "version_token": ProjectVersionToken(
            snapshot_version="sha256:snapshot-a",
            project_version=4,
            lock_token="repair-router-lock",
        ),
    }
    values.update(overrides)
    return ProjectRepairRequest(**values)


def _membership(action, *, repair_task_id: str = "repair-existing") -> RepairMembership:
    return RepairMembership(
        project=action.project,
        repair_identity=action.repair_identity,
        repair_task_id=repair_task_id,
        failed_task_id=action.failed_task_id,
        failed_run_id=action.failed_run_id,
        failure_fingerprint=action.failure_fingerprint,
        repair_index=action.repair_index,
    )


def test_one_eligible_failure_creates_one_bounded_repair_action():
    registrar = FakeAtomicRegistrar()

    result = route_project_repair(_request(), register_repair=registrar)

    assert result.outcome == REPAIR_CREATED
    assert result.reason == "eligible_repair_registered"
    assert result.repair_task_id == "repair-1"
    assert result.repair_identity.startswith("repair:sha256:")
    assert len(registrar.calls) == 1
    action, token = registrar.calls[0]
    assert action.repair_identity == result.repair_identity
    assert action.idempotency_key == result.repair_identity
    assert action.failed_task_id == "failed-task"
    assert action.failed_run_id == 17
    assert action.repair_index == 1
    assert action.task_retry_index == 1
    assert action.worker_profile == "builder-grok"
    assert action.membership_kind == "repair"
    assert action.required is True
    assert token == _request().version_token


def test_existing_repair_membership_wins_over_budget_and_never_mutates():
    first_registrar = FakeAtomicRegistrar()
    first = route_project_repair(_request(), register_repair=first_registrar)
    action = first_registrar.calls[0][0]
    request = _request(
        existing_repairs=(_membership(action),),
        project_repair_budget=0,
        task_retry_limit=0,
    )
    replay_registrar = FakeAtomicRegistrar()

    replay = route_project_repair(request, register_repair=replay_registrar)

    assert replay.outcome == REPAIR_ALREADY_EXISTS
    assert replay.reason == "repair_identity_already_registered"
    assert replay.repair_identity == first.repair_identity
    assert replay.repair_task_id == "repair-existing"
    assert replay_registrar.calls == []


def test_restart_replay_is_idempotent_through_atomic_registration_boundary():
    durable = {}
    before_restart = route_project_repair(
        _request(), register_repair=FakeAtomicRegistrar(durable)
    )
    after_restart = route_project_repair(
        _request(), register_repair=FakeAtomicRegistrar(durable)
    )

    assert before_restart.outcome == REPAIR_CREATED
    assert after_restart.outcome == REPAIR_ALREADY_EXISTS
    assert after_restart.repair_identity == before_restart.repair_identity
    assert after_restart.repair_task_id == before_restart.repair_task_id
    assert len(durable) == 1


def test_exhausted_project_repair_budget_refuses_before_registration():
    registrar = FakeAtomicRegistrar()
    request = _request(
        evaluation=_evaluation(repair_generation=2, repair_budget=2),
        project_repair_budget=2,
    )

    result = route_project_repair(request, register_repair=registrar)

    assert result.outcome == BUDGET_EXHAUSTED
    assert result.reason == "project_repair_budget_exhausted"
    assert registrar.calls == []


def test_exhausted_failed_task_retry_limit_refuses_before_registration():
    seed_registrar = FakeAtomicRegistrar()
    route_project_repair(_request(), register_repair=seed_registrar)
    prior = _membership(seed_registrar.calls[0][0])
    next_evaluation = _evaluation(repair_generation=1)
    request = _request(
        evaluation=next_evaluation,
        existing_repairs=(prior,),
        failed_run_id=18,
        failure_envelope=_failure(run_id=18, fingerprint="b" * 64),
        version_token=replace(
            _request().version_token,
            snapshot_version=next_evaluation.snapshot_version,
        ),
    )
    registrar = FakeAtomicRegistrar()

    result = route_project_repair(request, register_repair=registrar)

    assert result.outcome == BUDGET_EXHAUSTED
    assert result.reason == "failed_task_retry_limit_exhausted"
    assert registrar.calls == []


def test_unsupported_failure_class_returns_redaction_safe_reason():
    registrar = FakeAtomicRegistrar()
    envelope = replace(
        _failure(failure_class="provider_auth"),
        redacted_error="secret-token-must-not-escape",
    )

    result = route_project_repair(
        _request(failure_envelope=envelope), register_repair=registrar
    )

    assert result.outcome == UNSUPPORTED_FAILURE
    assert result.reason == "failure_class_not_repairable"
    assert "secret-token-must-not-escape" not in result.reason
    assert registrar.calls == []


def test_stale_local_snapshot_and_atomic_cas_refuse_without_created_repair():
    local_registrar = FakeAtomicRegistrar()
    local = route_project_repair(
        _request(
            version_token=ProjectVersionToken(
                snapshot_version="sha256:older",
                project_version=4,
                lock_token="repair-router-lock",
            )
        ),
        register_repair=local_registrar,
    )
    cas_registrar = FakeAtomicRegistrar(force_stale=True)
    cas = route_project_repair(_request(), register_repair=cas_registrar)

    assert local.outcome == STALE_SNAPSHOT
    assert local.reason == "evaluation_snapshot_version_mismatch"
    assert local_registrar.calls == []
    assert cas.outcome == STALE_SNAPSHOT
    assert cas.reason == "atomic_registration_rejected_stale_token"
    assert cas.repair_task_id is None


def test_worker_profile_must_be_explicitly_permitted():
    registrar = FakeAtomicRegistrar()

    result = route_project_repair(
        _request(worker_profile="generalist"), register_repair=registrar
    )

    assert result.outcome == MALFORMED
    assert result.reason == "worker_profile_not_permitted"
    assert registrar.calls == []


def test_registration_inherits_contract_notification_identity_and_membership():
    contract = {
        "version": 1,
        "scope": "exact inherited scope",
        "allowed_files": ["one.py", "two.py"],
        "allow_child_creation": False,
    }
    routes = ("subscription:owner", "subscription:review-thread")
    registrar = FakeAtomicRegistrar()

    result = route_project_repair(
        _request(task_contract=contract, notification_route_identities=routes),
        register_repair=registrar,
    )

    assert result.outcome == REPAIR_CREATED
    action = registrar.calls[0][0]
    assert action.task_contract == contract
    assert action.task_contract is not contract
    assert action.notification_route_identities == routes
    assert action.project == _project()
    assert action.membership_kind == "repair"
    assert action.required is True


def test_unrelated_failure_occurrences_and_project_generations_stay_separate():
    first_registrar = FakeAtomicRegistrar()
    first = route_project_repair(_request(), register_repair=first_registrar)

    run_registrar = FakeAtomicRegistrar()
    different_run = route_project_repair(
        _request(
            failed_run_id=18,
            failure_envelope=_failure(run_id=18, fingerprint="b" * 64),
        ),
        register_repair=run_registrar,
    )

    next_project = _project(generation=3)
    next_evaluation = replace(_evaluation(), generation=3)
    next_envelope = replace(_failure(), generation=3)
    generation_registrar = FakeAtomicRegistrar()
    different_generation = route_project_repair(
        _request(
            project=next_project,
            evaluation=next_evaluation,
            failure_envelope=next_envelope,
        ),
        register_repair=generation_registrar,
    )

    assert first.outcome == REPAIR_CREATED
    assert different_run.outcome == REPAIR_CREATED
    assert different_generation.outcome == REPAIR_CREATED
    assert len(
        {
            first.repair_identity,
            different_run.repair_identity,
            different_generation.repair_identity,
        }
    ) == 3


def test_fingerprint_is_the_stable_occurrence_when_run_id_is_absent():
    registrar = FakeAtomicRegistrar()
    request = _request(
        failed_run_id=None,
        failure_envelope=_failure(run_id=None, fingerprint="c" * 64),
    )

    first = route_project_repair(request, register_repair=registrar)
    replay = route_project_repair(request, register_repair=registrar)

    assert first.outcome == REPAIR_CREATED
    assert replay.outcome == REPAIR_ALREADY_EXISTS
    assert replay.repair_identity == first.repair_identity


@pytest.mark.parametrize(
    "state",
    ["WAITING", "COMPLETE_ELIGIBLE", "BLOCKED", "FAILED", "MALFORMED"],
)
def test_non_repairable_and_terminal_evaluation_states_are_no_action(state):
    registrar = FakeAtomicRegistrar()

    result = route_project_repair(
        _request(evaluation=_evaluation(state=state)), register_repair=registrar
    )

    assert result.outcome == NO_ACTION
    assert result.reason == "evaluation_state_not_repairable"
    assert result.repair_identity is None
    assert registrar.calls == []
