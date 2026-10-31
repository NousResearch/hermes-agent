"""Message owners remain visible through delivery and publish idle on completion."""

import asyncio
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


@pytest.mark.asyncio
@pytest.mark.parametrize("secondary", [False, True], ids=["primary", "secondary"])
async def test_final_delivery_publishes_idle_status(monkeypatch, secondary):
    runner, adapter = make_restart_runner()
    # Every production creation path binds the owner through this factory.
    monkeypatch.setattr(runner, "_instantiate_adapter", lambda *_: adapter)
    assert runner._create_adapter(Platform.TELEGRAM, adapter.config) is adapter
    if secondary:
        runner._profile_adapters = {"beta": runner.adapters}
        runner.adapters = {}
    record = {}
    monkeypatch.setattr(gateway_run, "_write_runtime_status_quiet", lambda **kw: record.update(kw))
    event = MessageEvent(text="work", source=make_restart_source())
    key = build_session_key(event.source)

    async def handler(event):
        runner._running_agents[key] = SimpleNamespace()
        runner._persist_active_agents()
        runner._release_running_agent_state(key)
        return "completed"

    adapter.set_message_handler(handler)
    assert adapter._start_session_processing(event, key)
    await asyncio.wait_for(adapter._session_tasks[key], timeout=5)

    assert adapter.sent == ["completed"]
    assert runner._active_work_count() == 0
    assert record["active_agents"] == 0


@pytest.mark.asyncio
async def test_cancelled_owner_also_publishes_idle_status(monkeypatch):
    runner, adapter = make_restart_runner()
    adapter.gateway_runner = runner
    record = {}
    monkeypatch.setattr(gateway_run, "_write_runtime_status_quiet", lambda **kw: record.update(kw))
    task = asyncio.create_task(asyncio.Event().wait())
    adapter._track_session_task("agent:main:telegram:dm:cancelled", task)
    runner._persist_active_agents()
    assert record["active_agents"] == 1

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert runner._active_work_count() == 0
    assert record["active_agents"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("overlap", [False, True], ids=["disjoint", "overlapping"])
async def test_wedged_runner_never_hides_another_sessions_delivery(monkeypatch, overlap):
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "10")
    runner, adapter = make_restart_runner()
    runner._running_agents["agent:main:telegram:dm:handoff"] = SimpleNamespace(
        get_activity_summary=lambda: {"seconds_since_activity": 100})
    key = "agent:beta:telegram:dm:delivery"
    if overlap:
        runner._running_agents[key] = SimpleNamespace()
    # A shared adapter may appear in several profile maps; it is still one owner.
    runner._profile_adapters = {"beta": runner.adapters.copy()}
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    adapter._track_session_task(key, task)
    try:
        assert runner._wedged_chat_agent_count() == 1
        assert runner._active_work_count() == 2
        assert runner._awaitable_work_count() == 1
    finally:
        release.set()
        await task
