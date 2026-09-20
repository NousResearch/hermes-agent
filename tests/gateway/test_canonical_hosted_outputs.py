"""Owner-local output through the real FIFO, tool registry and publication path."""

import asyncio
from contextlib import asynccontextmanager
import json
import time
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as tasks, hosted_rooms as rooms
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authority import SessionAuthority
from gateway.session_hosted_service import CanonicalHostedRoomService
from gateway.session_results import admission_result
from hermes_state_runtime import list_session_admissions
from tools.registry import registry


@asynccontextmanager
async def owner(tmp_path, monkeypatch):
    from gateway import run
    import hermes_state
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    reviewer = tmp_path / "profiles" / "reviewer"
    reviewer.mkdir(parents=True)
    config = {"model": {"default": "fixture"}, "platform_toolsets": {"cli": []},
              "hosted_rooms": {"profiles": {"reviewer": str(reviewer)}}}
    monkeypatch.setattr(run, "_load_gateway_config", lambda: config)
    monkeypatch.setattr(run, "_resolve_gateway_model", lambda _: "fixture")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             config=GatewayConfig(), _cached_agent_for=lambda _: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    from gateway.hosted_room_input_custody import initialize_input_custody
    from hermes_state_runtime import begin_runtime_epoch
    initialize_input_custody(store._db)
    epoch = begin_runtime_epoch(store._db, instance_id="test")
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id="test", db=store._db, epoch=epoch)
    runner.session_authority = authority
    store._local_authority_epoch = epoch
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)
    from tui_gateway.hosted_room_driver import HostedRoomRuntime
    def forbidden_start(*args, **kwargs):
        raise AssertionError("runtime startup is outside this inert test")
    monkeypatch.setattr(HostedRoomRuntime, "start", forbidden_start)
    service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
    authority.hosted_room_service = service
    service.authorize_room("alice", "room", create=True)
    service.create_room(room_id="room", name="Files", members=[
        dict(member_id="writer", profile="default", handle="writer"),
        dict(member_id="reviewer", profile="reviewer", handle="reviewer")])
    try:
        yield authority, service, runner
    finally:
        running = [live.task for live in authority.sessions.values() if live.task and not live.task.done()]
        if running:
            await asyncio.wait_for(asyncio.gather(*running), timeout=5)
        store._db.close()


async def execute_group_turn(authority, service, *, event_id="request", before_admission=None, defer_publication=False, input_manifest=None, thread_id="thread"):
    service.send(room_id="room", event_id=event_id, payload=dict(thread_id=thread_id, text="@writer Write the report",
        **({"attachments": input_manifest} if input_manifest else {})))
    task = tasks.list_tasks(service.db_path, room_id="room", status="queued")[0]
    if before_admission is not None:
        before_admission(task)
    if defer_publication:
        service.runtime.publish_terminal = lambda *args: None
    binding = service.bindings()[0]
    lease = tasks.acquire_lease(service.db_path, room_id="room", gateway_id=binding.gateway_id,
                               authority_epoch=binding.authority_epoch, process_generation="driver", ttl_seconds=60, clock=time.time)
    attempt = tasks.start_task(service.db_path, task["identity"], lease, expected_cancel_generation=0, clock=time.time)
    rpc = service._resolve_member_transport(binding, task)
    coords = dict(profile="default", source="bot_room")
    sid = (await asyncio.to_thread(rpc.create, **coords, title="Group: room"))["session_id"]
    done = asyncio.Event()
    failures = []
    def terminal(receipt):
        try:
            service.runtime._on_terminal(binding, attempt, receipt)
        except Exception as exc:
            failures.append(exc)
        finally:
            done.set()
    request = dict(**coords, session_id=sid, prompt=task["payload"]["prompt"], task=task["identity"],
                   execution_generation=attempt.execution_generation, on_terminal=terminal,
                   **({"attachments": task["payload"]["attachments"]} if task["payload"].get("attachments") else {}))
    receipt = await asyncio.to_thread(rpc.submit, **request)
    # Drive the real bounded drain directly; no scheduled runtime or monitor.
    await authority._drain(rpc.ref)
    await asyncio.wait_for(done.wait(), timeout=5)
    if failures:
        raise failures[0]
    return rpc, request, receipt, task, binding


@pytest.mark.asyncio
async def test_canonical_tool_output_is_retained_published_and_replayed_once(tmp_path, monkeypatch):
    from tools import hosted_room_artifact  # actual registry discovery owner
    from gateway.session_hosted_output import current_output_binding
    from gateway.session_group_files import dispatch_group_files
    from gateway.session_contract import Principal
    from model_tools import get_tool_definitions
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        output = tmp_path / "cache" / "report.txt"
        output.parent.mkdir(exist_ok=True)
        output.write_bytes(b"explicit result bytes")
        executions, tool_results, bindings, contexts = [], [], [], []
        async def handle(event):
            executions.append(event.message_id)
            bindings.append(current_output_binding())
            from contextvars import copy_context
            contexts.append(copy_context())
            definitions = get_tool_definitions(enabled_toolsets=["bot_room"])
            assert any(t["function"]["name"] == "share_group_file" for t in definitions)
            value = await asyncio.to_thread(registry.dispatch, "share_group_file", {"path": str(output)}, task_id="default")
            result = json.loads(value)
            tool_results.append(result)
            assert result.get("ok") is True, result
            repeated = json.loads(await asyncio.to_thread(registry.dispatch, "share_group_file", {"path": str(output)}))
            assert repeated["artifact_id"] == result["artifact_id"]
            return "@reviewer Shared report."
        runner._handle_message = handle
        rpc, request, receipt, task, binding = await execute_group_turn(authority, service)
        stored = tasks.get_task(service.db_path, task["identity"])
        assert stored["status"] == "settled", stored
        assert stored["result"]["artifacts"]["items"][0]["sha256"] == tool_results[0]["sha256"]
        saved = admission_result(authority.db, receipt["admission_id"])
        assert saved["result"]["artifact_scope"] == stored["result"]["artifact_scope"]
        actor = Principal("alice", str(tmp_path), frozenset({"session:read"}), "viewer")
        page = dispatch_group_files(service, actor, "groups.attachment.list", {"room_id": "room"})
        item, = page["items"]
        following = tasks.list_tasks(service.db_path, room_id="room", status="queued")
        assert following[0]["payload"]["target_member_id"] == "reviewer"
        assert following[0]["payload"]["attachments"][0]["attachment_id"] == item["attachment_id"]
        assert following[0]["payload"]["attachments"][0]["event_id"] == item["event_id"]
        result = dispatch_group_files(service, actor, "groups.attachment.download",
                                      dict(room_id="room", event_id=item["event_id"], attachment_id=item["attachment_id"]))
        import base64
        assert base64.b64decode(result["data_base64"]) == output.read_bytes()
        history = await asyncio.to_thread(rpc.history, profile="default", session_id=rpc.ref.session_id, source="bot_room")
        assert history[-1]["artifacts"] == stored["result"]["artifacts"]
        service.prepare_room(binding)
        assert len(executions) == 1
        assert len(list_session_admissions(authority.db, session_id=rpc.ref.session_id, pending_only=False)) == 1
        assert len([e for e in service._events("room") if e["kind"] == "message.member"]) == 1
        assert current_output_binding() is None and bindings[0].active is False
        assert json.loads(contexts[0].run(registry.dispatch, "share_group_file", {"path": str(output)}))["ok"] is False
        assert json.loads(registry.dispatch("share_group_file", {"path": str(output)})).get("ok") is False


@pytest.mark.asyncio
async def test_output_publication_fences_the_actual_attempt_in_append_transaction(tmp_path, monkeypatch):
    from tools import hosted_room_artifact
    from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactScope
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        output = tmp_path / "cache" / "report.txt"
        output.parent.mkdir(exist_ok=True)
        output.write_bytes(b"owned output")
        async def handle(event):
            from gateway.session_hosted_output import current_output_binding
            assert current_output_binding() is not None
            result = json.loads(await asyncio.to_thread(registry.dispatch, "share_group_file", {"path": str(output)}))
            assert result.get("ok"), result
            return "Shared report."
        runner._handle_message = handle
        _, _, _, task, _ = await execute_group_turn(authority, service)
        stored = tasks.get_task(service.db_path, task["identity"])
        scope = RoomArtifactScope.from_mapping(stored["result"]["artifact_scope"])
        event = next(e for e in service._events("room") if e["kind"] == "message.member")
        expected = dict(scope=scope.as_mapping(), manifest=stored["result"]["artifacts"],
                        cancel_generation=stored["cancel_generation"] + 1)
        with pytest.raises(RoomArtifactError, match="attempt changed"):
            rooms.append_event(service.db_path, room_id="room", event_id="late-output", kind=event["kind"],
                actor=event["actor"], payload=event["payload"], authority_gateway_id=scope.authority_gateway_id,
                authority_epoch=scope.authority_epoch, expected_output=expected)
        expected["cancel_generation"] = stored["cancel_generation"]
        with pytest.raises(RoomArtifactError, match="coordinates changed"):
            rooms.append_event(service.db_path, room_id="room", event_id="late-output", kind=event["kind"],
                actor=event["actor"], payload={**event["payload"], "task_id": "other-task"},
                authority_gateway_id=scope.authority_gateway_id, authority_epoch=scope.authority_epoch,
                expected_output=expected)
        assert not any(e["event_id"] == "late-output" for e in service._events("room"))
