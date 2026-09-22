from hermes_cli.fleet_protocol import FleetTask, RunnerCapability, TaskRequirement
from hermes_cli.fleet_scheduler import RunnerStatus, eligible_runners, select_runner


def _task(*, models=("gpt-5",), tools=("git",), project="LunaBot"):
    return FleetTask(
        task_id="task-1",
        title="Build",
        body="Build it",
        requirement=TaskRequirement(models=models, tools=tools, project=project),
        idempotency_key="task-1",
    )


def _runner(node_id, *, models=("gpt-5",), tools=("git",), projects=("LunaBot",), load=0, expires=200):
    return RunnerStatus(
        capability=RunnerCapability(
            node_id=node_id,
            profile="coding-expert",
            models=models,
            tools=tools,
            projects=projects,
            platform="darwin",
        ),
        last_seen=100.0,
        expires_at=expires,
        active_load=load,
    )


def test_eligible_runners_require_exact_model_and_all_capabilities():
    runners = [
        _runner("good"),
        _runner("wrong-model", models=("gpt-4",)),
        _runner("wrong-tool", tools=("terminal",)),
        _runner("wrong-project", projects=("Hermes Agent",)),
    ]

    assert [item.capability.node_id for item in eligible_runners(_task(), runners, now=150.0)] == ["good"]


def test_expired_heartbeat_is_never_eligible():
    assert eligible_runners(_task(), [_runner("offline", expires=150.0)], now=150.0) == []


def test_selection_prefers_least_loaded_runner_then_stable_identity():
    runners = [_runner("zulu", load=1), _runner("alpha", load=1), _runner("busy", load=3)]

    selected = select_runner(_task(), runners, now=150.0)

    assert selected is not None
    assert selected.capability.node_id == "alpha"


def test_selection_returns_none_when_no_runner_matches():
    assert select_runner(_task(models=("claude-sonnet",)), [_runner("mac")], now=150.0) is None
