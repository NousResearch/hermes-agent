from hermes_cli.fleet_protocol import FleetTask, RunnerCapability, TaskRequirement
from hermes_cli.fleet_store import FleetStore


def _runner(node_id: str = "mac") -> RunnerCapability:
    return RunnerCapability(
        node_id=node_id,
        profile="coding-expert",
        models=("gpt-5", "local:7b"),
        tools=("git", "terminal"),
        projects=("LunaBot",),
        platform="darwin",
    )


def _task(task_id: str, *, model: str = "gpt-5") -> FleetTask:
    return FleetTask(
        task_id=task_id,
        title=f"Task {task_id}",
        body="Do the work",
        requirement=TaskRequirement(
            models=(model,), tools=("git",), project="LunaBot", workspace_kind="worktree"
        ),
        idempotency_key=f"kanban:{task_id}",
    )


def test_store_creates_wal_schema_and_idempotently_submits(tmp_path):
    store = FleetStore(tmp_path / "fleet.db")

    first = store.submit_task(_task("task-1"), now=100.0)
    second = store.submit_task(_task("task-1"), now=101.0)

    assert first.task_id == second.task_id == "task-1"
    assert len(store.list_tasks()) == 1
    assert store.journal_mode == "wal"


def test_claim_is_capability_aware_and_contains_lease_metadata(tmp_path):
    store = FleetStore(tmp_path / "fleet.db")
    store.register_runner(_runner(), now=100.0, ttl=30.0)
    store.submit_task(_task("task-1"), now=100.0)
    store.submit_task(_task("task-2", model="unavailable"), now=100.0)

    claim = store.claim_task(_runner(), now=101.0, lease_seconds=20.0)

    assert claim is not None
    assert claim.task_id == "task-1"
    assert claim.node_id == "mac"
    assert claim.runner_profile == "coding-expert"
    assert claim.lease_expires_at == 121.0
    assert claim.attempt == 1
    assert store.claim_task(_runner("windows"), now=101.0, lease_seconds=20.0) is None


def test_lease_renewal_and_completion_require_the_current_claim(tmp_path):
    store = FleetStore(tmp_path / "fleet.db")
    store.register_runner(_runner(), now=100.0, ttl=30.0)
    store.submit_task(_task("task-1"), now=100.0)
    claim = store.claim_task(_runner(), now=101.0, lease_seconds=10.0)
    assert claim is not None

    renewed = store.renew_claim("task-1", claim.claim_id, now=105.0, lease_seconds=30.0)

    assert renewed is not None
    assert renewed.lease_expires_at == 135.0
    assert store.complete_task("task-1", "wrong-claim", result="ignored") is False
    assert store.complete_task("task-1", claim.claim_id, result="done", now=106.0) is True
    assert store.complete_task("task-1", claim.claim_id, result="done", now=107.0) is True
    assert store.list_tasks()[0].status == "completed"


def test_expired_claim_is_reclaimed_with_a_new_attempt(tmp_path):
    store = FleetStore(tmp_path / "fleet.db")
    store.register_runner(_runner(), now=100.0, ttl=30.0)
    store.register_runner(_runner("windows"), now=100.0, ttl=30.0)
    store.submit_task(_task("task-1"), now=100.0)
    first = store.claim_task(_runner(), now=101.0, lease_seconds=5.0)
    assert first is not None

    assert store.reclaim_expired(now=107.0) == 1
    second = store.claim_task(_runner("windows"), now=108.0, lease_seconds=5.0)

    assert second is not None
    assert second.claim_id != first.claim_id
    assert second.attempt == 2


def test_competing_claims_cannot_both_commit_the_same_task(tmp_path):
    store = FleetStore(tmp_path / "fleet.db")
    store.register_runner(_runner(), now=100.0, ttl=30.0)
    store.register_runner(_runner("windows"), now=100.0, ttl=30.0)
    store.submit_task(_task("task-1"), now=100.0)

    first = store.claim_task(_runner(), now=101.0, lease_seconds=20.0)
    second = store.claim_task(_runner("windows"), now=101.0, lease_seconds=20.0)

    assert first is not None
    assert second is None
