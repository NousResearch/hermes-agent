from __future__ import annotations

import subprocess
import time

import pytest

from hermes_cli.autonomous_coordinator import (
    AutonomousTaskState,
    AutonomousTaskStateStore,
    AutonomousTaskStatus,
    ConcurrentCoordinatorUpdate,
    CoordinatorEvent,
    CoordinatorRunner,
    VesselMindRepositoryLock,
)


class ScriptedActions:
    def __init__(self, script):
        self.script = {state: list(events) for state, events in script.items()}
        self.calls = []

    def action(self, state, operation_id):
        self.calls.append((state.state, operation_id))
        event = self.script[state.state].pop(0)
        if event is None:
            return None
        return CoordinatorEvent(operation_id=operation_id, **event)

    def mapping(self):
        return {state: self.action for state in self.script}


def event(status, **values):
    return {"status": status, **values}


def runner(tmp_path, script, **kwargs):
    actions = ScriptedActions(script)
    instance = CoordinatorRunner(
        AutonomousTaskStateStore(tmp_path / "coordinator.db"),
        actions.mapping(),
        **kwargs,
    )
    return instance, actions


def submit(instance, project="Hermes", repository="repo"):
    return instance.submit(
        task_id="VM-001",
        project=project,
        repository=repository,
        target_branch="dev",
    )


def full_script():
    return {
        AutonomousTaskStatus.DEVELOPING: [event("PASS", commit="a" * 40)],
        AutonomousTaskStatus.TESTING: [event("FAIL"), event("PASS")],
        AutonomousTaskStatus.REPAIRING: [event("PASS", commit="b" * 40)],
        AutonomousTaskStatus.ACCEPTANCE_PRE_DEPLOY: [event("PASS")],
        AutonomousTaskStatus.DELIVERING: [
            event("PASS", pr_number=17, ci_status="SUCCESS")
        ],
        AutonomousTaskStatus.DEPLOYING: [event("PASS", deployment_status="SUCCESS")],
        AutonomousTaskStatus.ACCEPTANCE_RUNTIME: [event("PASS")],
    }


def test_one_owner_task_runs_full_repair_delivery_and_runtime_flow(tmp_path):
    instance, actions = runner(tmp_path, full_script())

    result = submit(instance)

    assert result.state == AutonomousTaskStatus.DONE
    assert result.next_action == "DELIVERED_AND_VERIFIED"
    assert result.repair_loops == 1
    assert result.developer_commit == "b" * 40
    assert result.tester_verdict == "PASS"
    assert result.acceptance_verdict == "PASS"
    assert result.pr_number == 17
    assert result.ci_status == "SUCCESS"
    assert result.deployment_status == "SUCCESS"
    assert result.runtime_acceptance == "PASS"
    assert [call[0] for call in actions.calls] == [
        AutonomousTaskStatus.DEVELOPING,
        AutonomousTaskStatus.TESTING,
        AutonomousTaskStatus.REPAIRING,
        AutonomousTaskStatus.TESTING,
        AutonomousTaskStatus.ACCEPTANCE_PRE_DEPLOY,
        AutonomousTaskStatus.DELIVERING,
        AutonomousTaskStatus.DEPLOYING,
        AutonomousTaskStatus.ACCEPTANCE_RUNTIME,
    ]


@pytest.mark.parametrize(
    ("waiting_state", "completion", "expected_next"),
    [
        (
            AutonomousTaskStatus.DEVELOPING,
            event("PASS", commit="a" * 40),
            AutonomousTaskStatus.TESTING,
        ),
        (AutonomousTaskStatus.TESTING, event("FAIL"), AutonomousTaskStatus.REPAIRING),
        (
            AutonomousTaskStatus.DELIVERING,
            event("PASS", ci_status="SUCCESS"),
            AutonomousTaskStatus.DEPLOYING,
        ),
        (
            AutonomousTaskStatus.DEPLOYING,
            event("PASS", deployment_status="SUCCESS"),
            AutonomousTaskStatus.ACCEPTANCE_RUNTIME,
        ),
        (
            AutonomousTaskStatus.ACCEPTANCE_RUNTIME,
            event("PASS"),
            AutonomousTaskStatus.DONE,
        ),
    ],
)
def test_restart_resumes_from_persisted_waiting_state(
    tmp_path, waiting_state, completion, expected_next
):
    path = tmp_path / "coordinator.db"
    store = AutonomousTaskStateStore(path)
    state = store.create(
        AutonomousTaskState(
            task_id="VM-restart",
            project="Hermes",
            repository="repo",
            target_branch="dev",
            state=waiting_state,
            operation_id="same-operation",
            current_agent={
                AutonomousTaskStatus.DEVELOPING: "developer",
                AutonomousTaskStatus.TESTING: "tester",
                AutonomousTaskStatus.ACCEPTANCE_RUNTIME: "acceptance",
            }.get(waiting_state),
            next_action="waiting",
        )
    )
    actions = ScriptedActions({expected_next: [None]})
    restarted = CoordinatorRunner(store, actions.mapping())

    result = restarted.handle_event(
        state.task_id,
        CoordinatorEvent(operation_id="same-operation", **completion),
    )

    assert result.state == expected_next


def test_gateway_restart_resume_all_redispatches_same_operation_id(tmp_path):
    path = tmp_path / "coordinator.db"
    store = AutonomousTaskStateStore(path)
    store.create(
        AutonomousTaskState(
            task_id="VM-restart-all",
            project="Hermes",
            repository="repo",
            target_branch="dev",
            state=AutonomousTaskStatus.DEVELOPING,
            operation_id="durable-operation",
            current_agent="developer",
            next_action="Wait for Developer completion",
        )
    )
    actions = ScriptedActions(full_script())
    restarted = CoordinatorRunner(store, actions.mapping())

    result = restarted.resume_all()[0]

    assert result.state == AutonomousTaskStatus.DONE
    assert actions.calls[0] == (
        AutonomousTaskStatus.DEVELOPING,
        "durable-operation",
    )


def test_running_ci_event_checkpoints_without_redispatch(tmp_path):
    path = tmp_path / "coordinator.db"
    store = AutonomousTaskStateStore(path)
    store.create(
        AutonomousTaskState(
            task_id="VM-ci",
            project="Hermes",
            repository="repo",
            target_branch="dev",
            state=AutonomousTaskStatus.DELIVERING,
            operation_id="ci-op",
            next_action="Wait for CI",
        )
    )
    calls = []
    instance = CoordinatorRunner(
        store,
        {AutonomousTaskStatus.DELIVERING: lambda *_: calls.append(True)},
    )

    result = instance.handle_event(
        "VM-ci",
        CoordinatorEvent(
            operation_id="ci-op", status="RUNNING", pr_number=9, ci_status="RUNNING"
        ),
    )

    assert result.pr_number == 9
    assert result.ci_status == "RUNNING"
    assert calls == []


def test_ci_failure_reenters_repair_without_owner_message(tmp_path):
    script = full_script()
    script[AutonomousTaskStatus.DELIVERING] = [
        event("FAIL"),
        event("PASS", ci_status="SUCCESS"),
    ]
    script[AutonomousTaskStatus.REPAIRING].append(event("PASS", commit="c" * 40))
    script[AutonomousTaskStatus.TESTING].append(event("PASS"))
    script[AutonomousTaskStatus.ACCEPTANCE_PRE_DEPLOY].append(event("PASS"))
    instance, _ = runner(tmp_path, script)

    assert submit(instance).state == AutonomousTaskStatus.DONE


def test_deployment_failure_retries_internally(tmp_path):
    script = full_script()
    script[AutonomousTaskStatus.DEPLOYING] = [
        event("FAIL", error="temporary deploy failure"),
        event("PASS", deployment_status="SUCCESS"),
    ]
    instance, actions = runner(tmp_path, script)

    result = submit(instance)

    assert result.state == AutonomousTaskStatus.DONE
    assert result.deployment_retries == 1
    assert [state for state, _ in actions.calls].count(
        AutonomousTaskStatus.DEPLOYING
    ) == 2


def test_action_exception_is_repaired_without_owner_interaction(tmp_path):
    script = full_script()

    def crash_once(state, operation_id):
        if not getattr(crash_once, "called", False):
            crash_once.called = True
            raise RuntimeError("worker crashed")
        return CoordinatorEvent(
            operation_id=operation_id, status="PASS", commit="b" * 40
        )

    actions = ScriptedActions(script)
    mapping = actions.mapping()
    mapping[AutonomousTaskStatus.DEVELOPING] = crash_once
    mapping[AutonomousTaskStatus.REPAIRING] = crash_once
    instance = CoordinatorRunner(
        AutonomousTaskStateStore(tmp_path / "coordinator.db"), mapping
    )

    result = submit(instance)

    assert result.state == AutonomousTaskStatus.DONE
    assert result.repair_loops == 2


def test_runtime_rejection_runs_developer_test_and_acceptance_again(tmp_path):
    script = full_script()
    script[AutonomousTaskStatus.ACCEPTANCE_RUNTIME] = [event("FAIL"), event("PASS")]
    script[AutonomousTaskStatus.REPAIRING].append(event("PASS", commit="c" * 40))
    script[AutonomousTaskStatus.TESTING].append(event("PASS"))
    script[AutonomousTaskStatus.ACCEPTANCE_PRE_DEPLOY].append(event("PASS"))
    script[AutonomousTaskStatus.DELIVERING].append(event("PASS", ci_status="SUCCESS"))
    script[AutonomousTaskStatus.DEPLOYING].append(
        event("PASS", deployment_status="SUCCESS")
    )
    instance, _ = runner(tmp_path, script)

    result = submit(instance)

    assert result.state == AutonomousTaskStatus.DONE
    assert result.acceptance_loops == 1
    assert result.repair_loops == 2


def test_event_callback_reinvokes_runner_and_stale_event_is_ignored(tmp_path):
    script = full_script()
    script[AutonomousTaskStatus.DEVELOPING] = [None]
    instance, _ = runner(tmp_path, script)
    waiting = submit(instance)

    stale = instance.handle_event(
        waiting.task_id,
        CoordinatorEvent(operation_id="old", status="PASS", commit="f" * 40),
    )
    assert stale.state == AutonomousTaskStatus.DEVELOPING

    result = instance.handle_event(
        waiting.task_id,
        CoordinatorEvent(
            operation_id=waiting.operation_id, status="PASS", commit="a" * 40
        ),
    )
    assert result.state == AutonomousTaskStatus.DONE


def test_async_delegation_completion_invokes_runner_not_only_footer(tmp_path):
    from tools import async_delegation as delegation

    script = full_script()
    script[AutonomousTaskStatus.DEVELOPING] = [None]
    instance, _ = runner(tmp_path, script)
    waiting = submit(instance)

    dispatch = delegation.dispatch_async_delegation(
        goal="developer",
        context=None,
        toolsets=None,
        role="leaf",
        model="test",
        session_key="",
        runner=lambda: {
            "status": "completed",
            "summary": "developer complete",
            "coordinator_event": {
                "operation_id": waiting.operation_id,
                "status": "PASS",
                "commit": "a" * 40,
            },
        },
        completion_callback=instance.completion_callback(waiting.task_id),
    )
    assert dispatch["status"] == "dispatched"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if instance.store.load(waiting.task_id).state == AutonomousTaskStatus.DONE:
            break
        time.sleep(0.02)

    assert instance.store.load(waiting.task_id).state == AutonomousTaskStatus.DONE


def test_store_compare_and_swap_rejects_stale_writer(tmp_path):
    store = AutonomousTaskStateStore(tmp_path / "coordinator.db")
    original = store.create(
        AutonomousTaskState(
            task_id="VM-cas", project="Hermes", repository="repo", target_branch="dev"
        )
    )
    store.save(original.model_copy(update={"next_action": "one"}))

    with pytest.raises(ConcurrentCoordinatorUpdate):
        store.save(original.model_copy(update={"next_action": "two"}))


def test_store_persists_required_recovery_fields(tmp_path):
    store = AutonomousTaskStateStore(tmp_path / "coordinator.db")
    saved = store.create(
        AutonomousTaskState(
            task_id="VM-fields",
            project="VesselMind",
            repository="repo",
            target_branch="dev",
        )
    )

    assert {
        "task_id",
        "project",
        "repository",
        "target_branch",
        "state",
        "current_agent",
        "developer_commit",
        "tester_verdict",
        "acceptance_verdict",
        "pr_number",
        "ci_status",
        "deployment_status",
        "runtime_acceptance",
        "repair_loops",
        "acceptance_loops",
        "last_error",
        "next_action",
    } <= saved.model_dump().keys()


def test_compact_owner_progress_contains_only_required_fields(tmp_path):
    instance, _ = runner(tmp_path, {AutonomousTaskStatus.DEVELOPING: [None]})
    result = submit(instance)

    assert result.render().splitlines() == [
        "TASK: VM-001",
        "STATE: DEVELOPING",
        "REPAIR_LOOPS: 0",
        "NEXT_ACTION: Wait for Developer completion",
        "OWNER_ACTION_REQUIRED: NO",
    ]


def git(path, *args):
    result = subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_vesselmind_lock_recovers_when_approved_remote_is_not_origin(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    git(repo, "remote", "add", "origin", "https://github.com/wrong/repository.git")
    git(
        repo,
        "remote",
        "add",
        "jason",
        "git@github.com:JasonCheungCN/ceramic-ai-designer-h5.git",
    )

    assert VesselMindRepositoryLock().verify(repo) == "jason"


def test_vesselmind_lock_blocks_unknown_lineage_before_delivery(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    git(repo, "remote", "add", "origin", "https://github.com/wrong/repository.git")
    store = AutonomousTaskStateStore(tmp_path / "coordinator.db")
    store.create(
        AutonomousTaskState(
            task_id="VM-lock",
            project="VesselMind",
            repository=str(repo),
            target_branch="dev",
            state=AutonomousTaskStatus.DELIVERING,
            next_action="Deliver",
        )
    )
    instance = CoordinatorRunner(
        store,
        {AutonomousTaskStatus.DELIVERING: lambda *_: pytest.fail("must not deliver")},
    )

    result = instance.run("VM-lock")

    assert result.state == AutonomousTaskStatus.BLOCKED
    assert "BLOCKED_BY_REPOSITORY_LINEAGE" in result.last_error
