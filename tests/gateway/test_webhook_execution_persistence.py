"""Durable actual-task execution registry regressions."""

import asyncio

import pytest

from gateway.platforms.webhook_executions import WebhookExecutionRegistry


@pytest.mark.asyncio
async def test_registry_completes_only_when_bound_real_task_finishes(tmp_path):
    registry = WebhookExecutionRegistry(tmp_path / "executions.json")
    accepted = registry.accept(
        profile="default", route="route", provider="github",
        delivery_id="delivery", session_key="webhook:route:delivery",
    )
    gate = asyncio.Event()

    async def actual_agent_run():
        await gate.wait()

    task = asyncio.create_task(actual_agent_run())
    registry.bind(accepted["execution_id"], task)
    await asyncio.sleep(0)
    assert registry.public(accepted["execution_id"])["state"] == "running"
    gate.set()
    await task
    await asyncio.sleep(0)
    assert registry.public(accepted["execution_id"])["state"] == "completed"


@pytest.mark.asyncio
async def test_cancel_is_cancelling_until_task_observes_cancellation(tmp_path):
    registry = WebhookExecutionRegistry(tmp_path / "executions.json")
    accepted = registry.accept(
        profile="default", route="route", provider="github",
        delivery_id="delivery", session_key="session",
    )

    async def blocked():
        await asyncio.Event().wait()

    task = asyncio.create_task(blocked())
    registry.bind(accepted["execution_id"], task)
    assert registry.request_cancel(accepted["execution_id"]) == "cancelling"
    assert registry.public(accepted["execution_id"])["state"] == "cancelling"
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert registry.public(accepted["execution_id"])["state"] == "cancelled"


@pytest.mark.asyncio
async def test_exact_ids_bind_concurrent_deliveries_sharing_one_session(tmp_path):
    registry = WebhookExecutionRegistry(tmp_path / "executions.json")
    first = registry.accept(
        profile="default", route="route", provider="github",
        delivery_id="one", session_key="durable-session",
    )
    second = registry.accept(
        profile="default", route="route", provider="github",
        delivery_id="two", session_key="durable-session",
    )
    gate_one = asyncio.Event()
    gate_two = asyncio.Event()
    task_one = asyncio.create_task(gate_one.wait())
    task_two = asyncio.create_task(gate_two.wait())

    assert registry.bind(second["execution_id"], task_two)
    assert registry.bind(first["execution_id"], task_one)
    assert registry.public(first["execution_id"])["state"] == "running"
    assert registry.public(second["execution_id"])["state"] == "running"

    gate_two.set()
    await task_two
    await asyncio.sleep(0)
    assert registry.public(second["execution_id"])["state"] == "completed"
    assert registry.public(first["execution_id"])["state"] == "running"
    gate_one.set()
    await task_one


def test_dispatcher_failure_only_finishes_an_unbound_execution(tmp_path):
    registry = WebhookExecutionRegistry(tmp_path / "executions.json")
    accepted = registry.accept(
        profile="default", route="route", provider="github",
        delivery_id="delivery", session_key="session",
    )
    assert registry.finish_if_unbound(
        accepted["execution_id"], "failed", "dispatcher failed"
    )
    record = registry.public(accepted["execution_id"])
    assert record["state"] == "failed"
    assert record["error"] == "dispatcher failed"


@pytest.mark.asyncio
async def test_prebind_cancel_cancels_the_real_task_when_it_arrives(tmp_path):
    registry = WebhookExecutionRegistry(tmp_path / "executions.json")
    accepted = registry.accept(
        profile="default", route="route", provider="github",
        delivery_id="delivery", session_key="session",
    )
    assert registry.request_cancel(accepted["execution_id"]) == "cancelling"

    async def blocked():
        await asyncio.Event().wait()

    task = asyncio.create_task(blocked())
    assert registry.bind(accepted["execution_id"], task)
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert registry.public(accepted["execution_id"])["state"] == "cancelled"


def test_restart_reconciles_nonterminal_records_as_interrupted(tmp_path):
    path = tmp_path / "executions.json"
    first = WebhookExecutionRegistry(path)
    accepted = first.accept(
        profile="default", route="route", provider="github",
        delivery_id="delivery", session_key="session",
    )
    second = WebhookExecutionRegistry(path)
    assert second.public(accepted["execution_id"])["state"] == "interrupted"


def test_authorization_is_bound_to_profile_and_route(tmp_path):
    registry = WebhookExecutionRegistry(tmp_path / "executions.json")
    accepted = registry.accept(
        profile="alpha", route="route-a", provider="github",
        delivery_id="delivery", session_key="session",
    )
    execution_id = accepted["execution_id"]
    token = accepted["access_token"]
    assert registry.authorize_scoped(
        execution_id, token, profile="alpha", route="route-a"
    )
    assert not registry.authorize_scoped(
        execution_id, token, profile="beta", route="route-a"
    )
    assert not registry.authorize_scoped(
        execution_id, token, profile="alpha", route="route-b"
    )


def test_failure_text_is_bounded(tmp_path):
    registry = WebhookExecutionRegistry(tmp_path / "executions.json")
    accepted = registry.accept(
        profile="default", route="route", provider="github",
        delivery_id="delivery", session_key="session",
    )
    registry.finish(accepted["execution_id"], "failed", "x" * 5000)
    assert len(registry.public(accepted["execution_id"])["error"]) <= 1024


def test_status_token_is_stored_only_as_hash(tmp_path):
    path = tmp_path / "executions.json"
    registry = WebhookExecutionRegistry(path)
    accepted = registry.accept(
        profile="default", route="route", provider="github",
        delivery_id="delivery", session_key="session",
    )
    assert registry.authorize(accepted["execution_id"], accepted["access_token"])
    assert accepted["access_token"] not in path.read_text(encoding="utf-8")


def test_read_only_registry_never_rewrites_live_ledger(tmp_path):
    path = tmp_path / "executions.json"
    writer = WebhookExecutionRegistry(path)
    accepted = writer.accept(
        profile="worker", route="route", provider="github",
        delivery_id="delivery", session_key="session",
    )
    before = path.read_bytes()
    reader = WebhookExecutionRegistry(
        path, reconcile_restart=False, read_only=True
    )
    assert reader.public(accepted["execution_id"])["state"] == "accepted"
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError, match="read-only"):
        reader.request_cancel(accepted["execution_id"])


def test_corrupt_writer_quarantines_exact_ledger_bytes(tmp_path):
    path = tmp_path / "executions.json"
    path.write_bytes(b"{not-json")
    WebhookExecutionRegistry(path)
    quarantined = list(tmp_path.glob("executions.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"{not-json"
    assert not path.exists()
