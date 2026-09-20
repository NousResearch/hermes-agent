"""Failed canonical producers retire private Output without a Stop request."""

import asyncio
import json

import pytest

from gateway import hosted_room_driver as tasks
from gateway.hosted_room_artifacts import RoomArtifactOutbox
from tests.gateway.test_canonical_hosted_outputs import owner, execute_group_turn


@pytest.mark.asyncio
async def test_failed_producer_retires_private_output_before_failure_publication(tmp_path, monkeypatch):
    from gateway.session_hosted_output import current_output_binding
    from tools.hosted_room_artifact import share_group_file

    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        output = tmp_path / "cache" / "failed-output.txt"
        output.parent.mkdir(exist_ok=True)
        output.write_bytes(b"private output from failed producer")
        captured = []

        async def fail_after_output(_event):
            binding = current_output_binding()
            assert binding is not None
            captured.append(binding)
            shared = json.loads(await asyncio.to_thread(share_group_file, str(output)))
            assert shared["ok"] is True
            raise RuntimeError("inert provider failure after private output")

        runner._handle_message = fail_after_output
        _, _, _, task, binding = await execute_group_turn(
            authority, service, defer_publication=True
        )
        failed = tasks.get_task(service.db_path, task["identity"])
        assert failed["status"] == "failed"
        assert captured

        service.prepare_room(binding)
        outbox = RoomArtifactOutbox(service.db_path)
        assert outbox.list(captured[0].scope) == []
        assert outbox.retirement_complete(captured[0].scope)
        assert not any(
            event["kind"] == "message.member"
            for event in service._events("room")
        )
