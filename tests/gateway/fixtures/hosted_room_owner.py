"""A real canonical room owner for turn tests: authority, admission FIFO, driver and room log.

Turns run through the authority's own drain with ``runner._handle_message`` as the agent; the
background room worker is never started, so each test drives the driver steps it needs.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from gateway import hosted_room_driver as tasks

MEMBERS = (dict(member_id="writer", profile="default", handle="writer"),
           dict(member_id="reviewer", profile="reviewer", handle="reviewer"))


@asynccontextmanager
async def owner(tmp_path, monkeypatch, *, members=MEMBERS):
    """A default-profile owner of room ``room``; ``reviewer`` is a second seated profile."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from hermes_state_runtime import begin_runtime_epoch
    from tui_gateway.hosted_room_driver import HostedRoomRuntime
    import hermes_state

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    reviewer = tmp_path / "profiles" / "reviewer"
    reviewer.mkdir(parents=True)
    config = {"model": {"default": "fixture"}, "platform_toolsets": {"cli": []}}
    monkeypatch.setattr(run, "_load_gateway_config", lambda: config)
    monkeypatch.setattr(run, "_resolve_gateway_model", lambda _: "fixture")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             config=GatewayConfig(), _cached_agent_for=lambda _: None)
    adapter = lambda source: runner.adapters.get(source.platform)
    runner._adapter_for_source = runner._intake_adapter_for = runner._delivery_adapter_for = adapter
    runner._adapters_for_profile = lambda profile: runner.adapters if profile in (None, "", "default") else {}
    epoch = begin_runtime_epoch(store._db, instance_id="test")
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id="test", db=store._db, epoch=epoch)
    runner.session_authority = authority
    store._local_authority_epoch = epoch
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)

    def forbidden_start(*args, **kwargs):
        raise AssertionError("the background room worker is outside these tests")
    monkeypatch.setattr(HostedRoomRuntime, "start", forbidden_start)
    service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
    monkeypatch.setattr(service, "profile_homes", lambda: {"default": Path(tmp_path), "reviewer": reviewer})
    authority.hosted_room_service = service
    service.authorize_room("alice", "room", create=True)
    service.create_room(room_id="room", name="Files", members=[dict(member) for member in members])
    try:
        yield authority, service, runner
    finally:
        running = [live.task for live in authority.sessions.values() if live.task and not live.task.done()]
        if running:
            await asyncio.wait_for(asyncio.gather(*running), timeout=5)
        store._db.close()


async def admit_turn(service, *, event_id="request", text="@writer Write the report", thread_id="thread"):
    """Send one user message and start the writer's queued task under the room worker's lease."""
    service.send(room_id="room", event_id=event_id, payload=dict(thread_id=thread_id, text=text))
    task = tasks.list_tasks(service.db_path, room_id="room", status="queued")[0]
    binding = service.bindings()[0]
    lease = service.runtime._ensure_lease(binding)
    attempt = tasks.start_task(service.db_path, task["identity"], lease, expected_cancel_generation=0,
                               clock=time.time)
    return task, binding, attempt


async def run_turn(authority, service, *, event_id="request", text="@writer Write the report",
                   thread_id="thread", publish=True):
    """Drive one real hosted turn: submit, the authority's FIFO drain, then the terminal callback."""
    task, binding, attempt = await admit_turn(service, event_id=event_id, text=text, thread_id=thread_id)
    if not publish:
        service.runtime.publish_terminal = lambda *args: None
    rpc = service._resolve_member_transport(binding, task)
    coords = dict(profile="default", source="bot_room")
    sid = (await asyncio.to_thread(rpc.create, **coords, title="Group: room"))["session_id"]
    done, failures = asyncio.Event(), []

    def terminal(receipt):
        try:
            service.runtime._on_terminal(binding, attempt, receipt)
        except Exception as exc:  # pragma: no cover - surfaced below
            failures.append(exc)
        finally:
            done.set()

    receipt = await asyncio.to_thread(
        rpc.submit, **coords, session_id=sid, prompt=task["payload"]["prompt"], task=task["identity"],
        execution_generation=attempt.execution_generation, on_terminal=terminal)
    await authority._drain(rpc.ref)
    await asyncio.wait_for(done.wait(), timeout=5)
    if failures:
        raise failures[0]
    return SimpleNamespace(rpc=rpc, receipt=receipt, task=task, binding=binding, attempt=attempt, session_id=sid)


def events(service, kind=None):
    return [event for event in service._events("room") if kind is None or event["kind"] == kind]
