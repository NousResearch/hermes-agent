from contextlib import contextmanager
import time

import pytest

from hermes_cli.fleet_client import FleetClient, FleetTransportError
from hermes_cli.fleet_protocol import FleetTask, RunnerCapability, TaskRequirement
from hermes_cli.fleet_server import FleetServer
from hermes_cli.fleet_store import FleetStore


def _runner(node_id: str) -> RunnerCapability:
    return RunnerCapability(
        node_id=node_id,
        profile="coding-expert",
        models=("gpt-5",),
        tools=("git",),
        projects=("LunaBot",),
        platform="darwin",
    )


def _task(task_id: str = "task-1") -> FleetTask:
    return FleetTask(
        task_id=task_id,
        title="Build",
        body="Build it",
        requirement=TaskRequirement(models=("gpt-5",), tools=("git",), project="LunaBot"),
        idempotency_key=f"kanban:{task_id}",
    )


@contextmanager
def _server(tmp_path):
    store = FleetStore(tmp_path / "fleet.db")
    server = FleetServer(store, token="test-token")
    server.start()
    try:
        yield store, server, FleetClient(server.url, token="test-token")
    finally:
        server.close()


def test_all_endpoints_require_the_machine_bearer_key(tmp_path):
    with _server(tmp_path) as (_store, server, _client):
        client = FleetClient(server.url, token="wrong")
        with pytest.raises(FleetTransportError, match="401"):
            client.health()


def test_task_submission_is_idempotent_over_http(tmp_path):
    with _server(tmp_path) as (_store, coordinator, client):
        first = client.submit_task(_task())
        second = client.submit_task(_task())

        assert first.task_id == second.task_id == "task-1"
        assert len(client.list_tasks()) == 1


def test_claim_cas_and_completion_authorization_are_enforced(tmp_path):
    with _server(tmp_path) as (_store, coordinator, client):
        mac = _runner("mac")
        windows = _runner("windows")
        client.register_runner(mac, now=100.0, ttl=30.0)
        client.register_runner(windows, now=100.0, ttl=30.0)
        client.submit_task(_task(), now=100.0)

        first = client.claim_task(mac, now=101.0, lease_seconds=20.0)
        second = client.claim_task(windows, now=101.0, lease_seconds=20.0)

        assert first is not None
        assert second is None
        assert client.complete_task("task-1", "not-the-claim", result="bad") is False
        assert client.complete_task("task-1", first.claim_id, result="done", now=102.0) is True


def test_expired_heartbeat_is_not_claimable_over_http(tmp_path):
    with _server(tmp_path) as (_store, coordinator, client):
        mac = _runner("mac")
        client.register_runner(mac, now=100.0, ttl=1.0)
        client.submit_task(_task(), now=100.0)

        assert client.claim_task(mac, now=102.0, lease_seconds=20.0) is None


def test_runner_status_keeps_expired_rows_visible_for_desktop(tmp_path):
    with _server(tmp_path) as (_store, coordinator, client):
        mac = _runner("mac")
        now = time.time()
        client.register_runner(mac, now=now, ttl=1.0)

        online = client.list_runners()
        assert online[0]["node_id"] == "mac"
        assert online[0]["capability"]["models"] == ["gpt-5"]
        assert online[0]["online"] is True

        expired = _store.list_runners(now=now + 2.0)
        assert expired[0]["online"] is False
        assert expired[0]["profile"] == "coding-expert"
