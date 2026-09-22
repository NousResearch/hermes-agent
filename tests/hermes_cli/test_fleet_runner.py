from types import SimpleNamespace

from hermes_cli.fleet_protocol import FleetTask, RunnerCapability, TaskRequirement
from hermes_cli.fleet_runner import FleetRunner
from hermes_cli.fleet_store import TaskClaim, TaskRecord


def _capability() -> RunnerCapability:
    return RunnerCapability(
        node_id="windows",
        profile="coding-expert",
        models=("gpt-5",),
        tools=("git",),
        projects=("LunaBot",),
        platform="win32",
    )


def _task() -> FleetTask:
    return FleetTask(
        task_id="global-task-1",
        title="Build LunaBot",
        body="Run the build",
        requirement=TaskRequirement(models=("gpt-5",), tools=("git",), project="LunaBot"),
        idempotency_key="kanban:global-task-1",
    )


class _Coordinator:
    def __init__(self):
        self.capability = _capability()
        self.task = _task()
        self.claims = [
            TaskClaim("global-task-1", "claim-1", "windows", "coding-expert", 200.0, 1),
            TaskClaim("global-task-1", "claim-2", "windows", "coding-expert", 300.0, 2),
        ]
        self.registered = []
        self.completed = []
        self.failed = []

    def register_runner(self, capability, **_kwargs):
        self.registered.append(capability)
        return True

    def heartbeat_runner(self, capability, **_kwargs):
        return True

    def claim_task(self, _capability, **_kwargs):
        return self.claims.pop(0) if self.claims else None

    def list_tasks(self):
        return [
            SimpleNamespace(
                task_id=self.task.task_id,
                title=self.task.title,
                body=self.task.body,
                requirement=self.task.requirement,
                idempotency_key=self.task.idempotency_key,
            )
        ]

    def complete_task(self, task_id, claim_id, **kwargs):
        self.completed.append((task_id, claim_id, kwargs))
        return True

    def fail_task(self, task_id, claim_id, **kwargs):
        self.failed.append((task_id, claim_id, kwargs))
        return True


def test_runner_uses_windows_executable_as_an_argument_vector_and_reports_completion():
    coordinator = _Coordinator()
    calls = []

    def execute(argv):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="done", stderr="")

    runner = FleetRunner(
        "windows",
        coordinator,
        r"C:\Users\idrat\AppData\Local\hermes\bin\hermes.exe",
        profiles=["coding-expert"],
        projects=["LunaBot"],
        capabilities=[coordinator.capability],
        liveness_check=lambda: True,
        executor=execute,
    )

    assert runner.run_once(now=100.0) is True
    assert calls == [[
        r"C:\Users\idrat\AppData\Local\hermes\bin\hermes.exe",
        "-p",
        "coding-expert",
        "-z",
        "Run the build",
    ]]
    assert coordinator.completed[0][0:2] == ("global-task-1", "claim-1")
    assert coordinator.failed == []


def test_runner_does_not_register_or_claim_when_desktop_is_not_live():
    coordinator = _Coordinator()
    runner = FleetRunner(
        "windows",
        coordinator,
        "hermes",
        profiles=["coding-expert"],
        projects=["LunaBot"],
        capabilities=[coordinator.capability],
        liveness_check=lambda: False,
    )

    assert runner.register() is False
    assert runner.run_once(now=100.0) is False
    assert coordinator.registered == []


def test_duplicate_delivery_is_idempotent_and_stop_is_graceful():
    coordinator = _Coordinator()
    executions = []
    runner = FleetRunner(
        "windows",
        coordinator,
        "hermes",
        profiles=["coding-expert"],
        projects=["LunaBot"],
        capabilities=[coordinator.capability],
        liveness_check=lambda: True,
        executor=lambda argv: executions.append(argv) or SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )

    assert runner.run_once(now=100.0) is True
    assert runner.run_once(now=101.0) is True
    runner.stop()

    assert len(executions) == 1
    assert [item[1] for item in coordinator.completed] == ["claim-1", "claim-2"]
    assert runner.stopped is True
