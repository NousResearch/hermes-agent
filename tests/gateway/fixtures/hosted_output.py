"""A real canonical owner for Group Chat file tests: authority, FIFO, driver and room log."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

from gateway import hosted_room_driver as tasks


@asynccontextmanager
async def owner(tmp_path, monkeypatch, *, members=None):
    """Default-profile owner whose hosted turns run through the real admission FIFO."""
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
    (reviewer / "profile.yaml").write_text("name: reviewer\n")
    config = {"model": {"default": "fixture"}, "platform_toolsets": {"cli": []},
              "hosted_rooms": {"profiles": {"reviewer": str(reviewer)}}}
    monkeypatch.setattr(run, "_load_gateway_config", lambda: config)
    monkeypatch.setattr(run, "_resolve_gateway_model", lambda _: "fixture")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             config=GatewayConfig(), _cached_agent_for=lambda _: None)
    adapter = lambda source: runner.adapters.get(source.platform)  # noqa: E731
    runner._adapter_for_source = runner._intake_adapter_for = runner._delivery_adapter_for = adapter
    epoch = begin_runtime_epoch(store._db, instance_id="test")
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id="test", db=store._db, epoch=epoch)
    runner.session_authority = authority
    store._local_authority_epoch = epoch
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)

    def forbidden_start(*args, **kwargs):
        raise AssertionError("the background room worker is outside these tests")
    monkeypatch.setattr(HostedRoomRuntime, "start", forbidden_start)
    service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
    authority.hosted_room_service = service
    service.authorize_room("alice", "room", create=True)
    service.create_room(room_id="room", name="Files", members=members or [
        dict(member_id="writer", profile="default", handle="writer"),
        dict(member_id="reviewer", profile="reviewer", handle="reviewer")])
    try:
        yield authority, service, runner
    finally:
        running = [live.task for live in authority.sessions.values() if live.task and not live.task.done()]
        if running:
            await asyncio.wait_for(asyncio.gather(*running), timeout=5)
        store._db.close()


async def admit_turn(service, *, event_id="request", text="@writer Write the report", thread_id="thread"):
    """Send one user message and start the writer's queued task under a driver lease."""
    service.send(room_id="room", event_id=event_id, payload=dict(thread_id=thread_id, text=text))
    task = tasks.list_tasks(service.db_path, room_id="room", status="queued")[0]
    binding = service.bindings()[0]
    lease = service.runtime._ensure_lease(binding)  # the room worker's own lease, as in production
    attempt = tasks.start_task(service.db_path, task["identity"], lease, expected_cancel_generation=0,
                               clock=time.time)
    return task, binding, attempt


async def run_turn(authority, service, *, event_id="request", text="@writer Write the report",
                   thread_id="thread", publish=True, before_drain=None):
    """Drive one real hosted turn: submit, the bounded FIFO drain, then the terminal callback."""
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
    if before_drain is not None:
        before_drain(task)
    await authority._drain(rpc.ref)
    await asyncio.wait_for(done.wait(), timeout=5)
    if failures:
        raise failures[0]
    return SimpleNamespace(rpc=rpc, receipt=receipt, task=task, binding=binding, attempt=attempt, session_id=sid)


def events(service, kind=None):
    return [e for e in service._events("room") if kind is None or e["kind"] == kind]


def obligations(service):
    from gateway.session_hosted_output_publication import OBLIGATIONS, obligations_exist
    with service.authority.db._read_ctx() as conn:
        if not obligations_exist(conn):
            return []
        return [dict(row) for row in conn.execute(f"SELECT * FROM {OBLIGATIONS} ORDER BY created_at")]


def held_threads(service, room_id="room"):
    """Threads Policy leaves out because a reply there still waits on its files."""
    with service.authority.db._read_ctx() as conn:
        return service._held_output_threads(conn, room_id)


def outbox_rows(service):
    with service.authority.db._read_ctx() as conn:
        from gateway.hosted_room_artifacts import output_store_exists
        if not output_store_exists(conn):
            return []
        return [dict(row) for row in conn.execute("SELECT * FROM hosted_room_output_artifacts")]


def write_file(tmp_path, name="report.txt", data=b"explicit result bytes\n"):
    path = tmp_path / "cache" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    return path


def share(path, **kwargs):
    from tools import hosted_room_artifact  # noqa: F401 - registers the tool like discovery does
    from tools.registry import registry
    return json.loads(registry.dispatch("share_group_file", {"path": str(path), **kwargs}, task_id="default"))
