"""Root-local frozen recipients and exact publication/ACK refusal boundaries."""

import asyncio
import json

import pytest

from gateway import hosted_room_driver as tasks
from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactOutbox, RoomArtifactScope
from tests.gateway.test_canonical_hosted_outputs import owner, execute_group_turn
from tools.hosted_room_artifact import share_group_file


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["frozen", "missing-snapshot", "changed-bytes", "changed-generation", "cursor"])
async def test_publication_uses_admitted_recipients_and_exact_source(tmp_path, monkeypatch, case):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        path = tmp_path / "cache" / "report.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"exact source bytes")

        async def handle(event):
            shared = json.loads(await asyncio.to_thread(share_group_file, str(path)))
            assert shared["ok"], shared
            return "Shared report."

        runner._handle_message = handle

        def snapshot(task):
            payload = dict(task["payload"])
            if case == "missing-snapshot":
                payload.pop("recipient_member_ids")
            else:
                # Today's roster contains reviewer; this admission does not grant it Files.
                payload["recipient_member_ids"] = ["writer"]
            _, encoded, digest = tasks._task_payload(payload)
            authority.db._execute_write(lambda conn: conn.execute(
                "UPDATE hosted_room_driver_tasks SET payload_json=?, payload_digest=? WHERE room_id=? AND task_id=?",
                (encoded, digest, "room", task["identity"].task_id)))

        _, _, _, task, binding = await execute_group_turn(
            authority, service, before_admission=snapshot, defer_publication=True)
        saved = tasks.get_task(service.db_path, task["identity"])
        scope = RoomArtifactScope.from_mapping(saved["result"]["artifact_scope"])
        outbox = RoomArtifactOutbox(service.db_path)
        ack_calls = []
        original_ack = RoomArtifactOutbox.acknowledge

        def acknowledge(self, *args, **kwargs):
            ack_calls.append((args, kwargs))
            return original_ack(self, *args, **kwargs)

        monkeypatch.setattr(RoomArtifactOutbox, "acknowledge", acknowledge)
        if case == "changed-bytes":
            with outbox._connect() as conn:
                blob = conn.execute("SELECT blob_name FROM hosted_room_output_artifacts").fetchone()[0]
            (outbox.blob_root / blob).write_bytes(b"changed bytes")
        if case == "changed-generation":
            authority.db._execute_write(lambda conn: conn.execute(
                "UPDATE hosted_room_driver_tasks SET execution_generation=execution_generation+1 "
                "WHERE room_id=? AND task_id=?", ("room", task["identity"].task_id)))
        if case == "cursor":
            from gateway import hosted_rooms
            append = hosted_rooms.append_event

            def stale_cursor(*args, **kwargs):
                if kwargs.get("expected_output"):
                    kwargs["expected_latest_seq"] += 1
                return append(*args, **kwargs)

            with monkeypatch.context() as raced:
                raced.setattr(hosted_rooms, "append_event", stale_cursor)
                with pytest.raises(hosted_rooms.EventCursorConflictError):
                    service._publish_terminal_tasks(service._room("room"))
            with authority.db._read_ctx() as conn:
                assert conn.execute("SELECT COUNT(*) FROM hosted_room_attachments WHERE state='committed'").fetchone()[0] == 0
            assert not ack_calls and not outbox.retirement_complete(scope)
        if case not in {"frozen", "cursor"}:
            with pytest.raises((RoomArtifactError, ValueError)):
                service.prepare_room(binding)
            assert not [e for e in service._events("room") if e["kind"] == "message.member"]
            assert not ack_calls and not outbox.retirement_complete(scope)
            return
        due = authority.db._conn.execute('SELECT next_attempt_at FROM hosted_room_artifact_retries').fetchone()
        if due is not None:
            service._artifact_clock = lambda: due[0]
        service.prepare_room(binding)
        message, = [e for e in service._events("room") if e["kind"] == "message.member"]
        assert message["payload"]["recipient_member_ids"] == ["writer"]
        attachment = message["payload"]["attachments"][0]
        store = service.output_attachments
        assert store.read(room_id="room", event_id=message["event_id"], recipient_member_id="writer",
                          attachment_id=attachment["attachment_id"]).data == path.read_bytes()
        with pytest.raises(ValueError):
            store.read(room_id="room", event_id=message["event_id"], recipient_member_id="reviewer",
                       attachment_id=attachment["attachment_id"])
        assert len(ack_calls) == 1 and outbox.retirement_complete(scope)


@pytest.mark.asyncio
async def test_lost_exact_ack_response_replays_without_second_publication(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        path = tmp_path / "cache" / "result.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"durable canonical bytes")

        async def handle(event):
            assert json.loads(await asyncio.to_thread(share_group_file, str(path)))["ok"]
            return "Shared result."

        runner._handle_message = handle
        _, _, _, task, binding = await execute_group_turn(authority, service, defer_publication=True)
        saved = tasks.get_task(service.db_path, task["identity"])
        scope = RoomArtifactScope.from_mapping(saved["result"]["artifact_scope"])
        original_ack = RoomArtifactOutbox.acknowledge
        calls = []

        def lose_response(self, source, artifact_ids, *, message_event_id):
            calls.append((source, artifact_ids, message_event_id))
            original_ack(self, source, artifact_ids, message_event_id=message_event_id)
            raise ConnectionError("inert lost ACK response")

        monkeypatch.setattr(RoomArtifactOutbox, "acknowledge", lose_response)
        service.prepare_room(binding)
        retry = authority.db._conn.execute('SELECT blocked,next_attempt_at FROM hosted_room_artifact_retries').fetchone()
        assert retry is not None and retry['blocked'] == 0
        message, = [e for e in service._events("room") if e["kind"] == "message.member"]
        assert calls == [(scope, [i["artifact_id"] for i in saved["result"]["artifacts"]["items"]],
                          message["event_id"])]
        service._artifact_clock = lambda: retry['next_attempt_at']
        service.prepare_room(binding)
        assert [e for e in service._events("room") if e["kind"] == "message.member"] == [message]
        assert len(calls) == 1
        assert service.output_attachments.abort_unpublished_event(
            room_id="room", event_id=message["event_id"]) is False
        item = message["payload"]["attachments"][0]
        assert service.output_attachments.read_viewer(
            room_id="room", event_id=message["event_id"], attachment_id=item["attachment_id"],
            authority_gateway_id=scope.authority_gateway_id,
            authority_epoch=scope.authority_epoch).data == path.read_bytes()
