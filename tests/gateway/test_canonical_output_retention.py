"""Output review regressions: ordinary retention and transport classification."""

import asyncio
import json
import time

import pytest

from tests.gateway.test_canonical_hosted_outputs import owner, execute_group_turn


@pytest.mark.asyncio
async def test_published_retirement_survives_receipt_expiry_without_skipping_byte_proof(tmp_path, monkeypatch):
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_room_artifacts import (
        ACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS, RoomArtifactError, RoomArtifactOutbox, RoomArtifactScope,
    )
    from gateway.hosted_room_attachments import AttachmentIntegrityError
    from tools import hosted_room_artifact

    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        path = tmp_path / "cache" / "result.txt"
        path.parent.mkdir(exist_ok=True)
        data = b"retained canonical result"
        path.write_bytes(data)
        calls = []
        async def handle(event):
            calls.append(event.message_id)
            shared = json.loads(await asyncio.to_thread(hosted_room_artifact.share_group_file, str(path)))
            assert shared["ok"], shared
            return "Shared report."
        runner._handle_message = handle
        _, _, _, task, binding = await execute_group_turn(authority, service)
        saved = tasks.get_task(service.db_path, task["identity"])
        scope = RoomArtifactScope.from_mapping(saved["result"]["artifact_scope"])
        manifest = saved["result"]["artifacts"]
        outbox = RoomArtifactOutbox(service.db_path)
        assert outbox.prune_acknowledged_receipts(
            now=time.time() + ACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS + 1) == 1
        assert outbox.retirement_complete(scope)
        service.prepare_room(binding)
        service.send(room_id="room", event_id="follow-up", payload=dict(thread_id="next", text="@writer Continue"))
        assert len(calls) == 1
        assert tasks.list_tasks(service.db_path, room_id="room", status="queued")
        assert tasks.get_task(service.db_path, task["identity"])["status"] == "settled"

        # Missing short-lived rows alone are not positive retirement evidence.
        unproven = RoomArtifactOutbox(tmp_path / "unproven-source" / "state.db")
        assert not unproven.retirement_complete(scope)
        with pytest.raises(RoomArtifactError, match="acknowledgement scope changed"):
            service._acknowledge_output(scope, manifest, unproven)

        store = service.output_attachments
        item = store.list_published(room_id="room", authority_gateway_id=scope.authority_gateway_id,
                                    authority_epoch=scope.authority_epoch)["items"][0]
        assert store.read_viewer(room_id="room", event_id=item["event_id"], attachment_id=item["attachment_id"],
            authority_gateway_id=scope.authority_gateway_id, authority_epoch=scope.authority_epoch).data == data
        blob = next(store.blob_root.glob("blob_*"))
        blob.write_bytes(b"different canonical bytes")
        with pytest.raises(AttachmentIntegrityError):
            service.prepare_room(binding)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["transport-collision", "transport-distinct", "local-denied", "invalid-transport"])
async def test_optional_output_classifies_retained_transport_without_hiding_local_errors(tmp_path, monkeypatch, case):
    from gateway.hosted_room_driver import TaskIdentity
    from gateway.session_contract import Principal
    from gateway.session_finite import execute_finite_admission
    from gateway.session_hosted_output import _binding, current_output_binding
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from gateway.session_hosted_transport import _BINDING, _principal
    from hermes_state_runtime import RuntimeStoreError, claim_session_input

    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        monkeypatch.setattr(authority, "_schedule", lambda ref: None)
        room_id = "other-room" if case == "transport-distinct" else "room"
        transport = dict(source_home=str(tmp_path / "source"), owner="source-owner", target_home=authority.profile_id,
                         selector=dict(room_id=room_id, member_id="source-member", profile="default"))
        actor = _principal(authority, transport)
        if case == "local-denied":
            actor = Principal("denied-local-owner", authority.profile_id, actor.capabilities, "local")
        rpc = HostedRoomAuthorityRPC(authority, asyncio.get_running_loop(), room_id=room_id,
            member_id="source-member", profile="default", principal=actor, authorize=lambda *args: True)
        coords = dict(profile="default", source="bot_room")
        sid = (await asyncio.to_thread(rpc.create, **coords, title=f"Group: {room_id}"))["session_id"]
        if case != "local-denied":
            retained = transport if case != "invalid-transport" else {**transport, "target_home": "wrong-owner"}
            authority.db._execute_write(lambda conn: conn.execute("INSERT INTO state_meta(key,value) VALUES(?,?)",
                                                                 (_BINDING + sid, json.dumps(retained))))
        await asyncio.to_thread(rpc.submit, **coords, session_id=sid, prompt="retained text",
            task=TaskIdentity(room_id, "source-task", "thread", "turn"), execution_generation=1, on_terminal=lambda _: None)
        row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
        if case in {"local-denied", "invalid-transport"}:
            with pytest.raises(RuntimeStoreError, match="permission_denied"):
                _binding(authority, rpc.ref, row)
            return
        assert _binding(authority, rpc.ref, row) is None
        calls = []
        async def text_only(event):
            assert current_output_binding() is None
            calls.append(event.message_id)
            return "ordinary text result"
        runner._handle_message = text_only
        # Enter after the independently owned transport admission gate: no
        # source-owner socket or cross-host authorization is exercised here.
        assert await execute_finite_admission(authority, rpc.ref, row) == "ordinary text result"
        assert calls == [row["admission_id"]]
        assert "artifacts" not in authority.pending_results[row["admission_id"]]["result"]
        assert not (tmp_path / "hosted-room-artifact-outbox").exists()
