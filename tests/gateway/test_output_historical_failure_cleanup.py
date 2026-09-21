"""Failed canonical producers retire private Output without a Stop request."""

import asyncio
import json
from dataclasses import replace

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
        foreign = []

        async def fail_after_output(_event):
            binding = current_output_binding()
            assert binding is not None
            captured.append(binding)
            shared = json.loads(await asyncio.to_thread(share_group_file, str(output)))
            assert shared["ok"] is True
            foreign_scope = replace(
                binding.scope, member_id="foreign-member", target_profile="foreign-profile"
            )
            foreign_outbox = RoomArtifactOutbox(service.db_path)
            foreign.append(
                (foreign_outbox, foreign_scope, foreign_outbox.put_path(scope=foreign_scope, path=output))
            )
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
        foreign_outbox, foreign_scope, foreign_item = foreign[0]
        assert foreign_outbox.read(foreign_scope, foreign_item["artifact_id"])[1] == output.read_bytes()
        assert output.read_bytes() == b"private output from failed producer"
        assert not any(
            event["kind"] == "message.member"
            for event in service._events("room")
        )


@pytest.mark.asyncio
async def test_failed_producer_is_physically_retired_before_actual_failure_publication(
    tmp_path, monkeypatch
):
    from gateway.session_hosted_output import current_output_binding
    from tools.hosted_room_artifact import share_group_file

    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        output = tmp_path / "cache" / "published-failure.txt"
        output.parent.mkdir(exist_ok=True)
        output.write_bytes(b"private output removed before failure publication")
        captured, blobs, publication_checks = [], [], []

        async def fail_after_output(_event):
            binding = current_output_binding()
            assert binding is not None
            captured.append(binding)
            assert json.loads(await asyncio.to_thread(share_group_file, str(output)))["ok"] is True
            with authority.db._read_ctx() as conn:
                blobs.extend(
                    tmp_path / "hosted-room-artifact-outbox" / "blobs" / row[0]
                    for row in conn.execute(
                        "SELECT blob_name FROM hosted_room_output_artifacts WHERE scope_key=?",
                        (binding.scope.key,),
                    )
                )
            raise RuntimeError("producer failed before publication")

        publish = service._publish_one_output

        def publish_after_cleanup(room, task, progress):
            scope = captured[0].scope
            with authority.db._read_ctx() as conn:
                rows = conn.execute(
                    "SELECT 1 FROM hosted_room_output_artifacts WHERE scope_key=?", (scope.key,)
                ).fetchall()
                fence = conn.execute(
                    "SELECT retired_generation FROM hosted_room_output_generation_fences "
                    "WHERE lineage_identity=?",
                    (scope.lineage_json,),
                ).fetchone()
            publication_checks.append((rows, None if fence is None else fence[0]))
            assert rows == []
            assert fence is not None and fence[0] >= scope.execution_generation
            assert blobs and not any(path.exists() for path in blobs)
            return publish(room, task, progress)

        service._publish_one_output = publish_after_cleanup
        runner._handle_message = fail_after_output
        _, _, _, task, _ = await execute_group_turn(authority, service)

        assert tasks.get_task(service.db_path, task["identity"])["status"] == "failed"
        assert publication_checks
        assert output.read_bytes() == b"private output removed before failure publication"


@pytest.mark.asyncio
async def test_failed_producer_cleanup_fault_remains_pending_until_exact_retry(tmp_path, monkeypatch):
    from gateway.session_hosted_output import current_output_binding
    from gateway.session_hosted_output_lifecycle import records
    from tools.hosted_room_artifact import share_group_file

    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        output = tmp_path / "cache" / "pending-output.txt"
        output.parent.mkdir(exist_ok=True)
        output.write_bytes(b"private output awaiting exact cleanup")
        captured = []

        async def fail_after_output(_event):
            binding = current_output_binding()
            assert binding is not None
            captured.append(binding)
            assert json.loads(await asyncio.to_thread(share_group_file, str(output)))["ok"] is True
            raise RuntimeError("original producer failure survives cleanup fault")

        runner._handle_message = fail_after_output
        with monkeypatch.context() as cleanup_fault:
            def fail_exact_unlink(*_args, **kwargs):
                if kwargs.get("dir_fd") is not None:
                    raise OSError("test-owned physical cleanup fault")
                raise AssertionError("cleanup must use a directory-relative exact blob name")

            cleanup_fault.setattr("gateway.hosted_room_output_discard.os.unlink", fail_exact_unlink)
            _, _, _, task, room_binding = await execute_group_turn(
                authority, service, defer_publication=True
            )
            service.prepare_room(room_binding)

        failed = tasks.get_task(service.db_path, task["identity"])
        assert failed["status"] == "failed"
        with authority.db._read_ctx() as conn:
            pending, = [record for _, record in records(conn, "room")]
            row = conn.execute(
                "SELECT acknowledged_at, cleanup_required_at, blob_reclaimed_at "
                "FROM hosted_room_output_artifacts WHERE scope_key=?",
                (captured[0].scope.key,),
            ).fetchone()
        assert pending["state"] == "pending"
        assert pending["reason_code"] == "cleanup_unavailable"
        assert pending["removed"] == 1 and len(pending["blobs"]) == 1
        assert row is not None and row["acknowledged_at"] is not None
        assert row["cleanup_required_at"] is not None and row["blob_reclaimed_at"] is None
        blob = tmp_path / "hosted-room-artifact-outbox" / "blobs" / pending["blobs"][0]["blob_name"]
        assert blob.read_bytes() == b"private output awaiting exact cleanup"
        assert output.read_bytes() == b"private output awaiting exact cleanup"
        assert captured[0].cleanup_pending is True
        assert not any(event["kind"] == "message.member" for event in service._events("room"))
        status = service.status("room")["pending_actions"]
        assert any(
            item["kind"] == "output_cleanup"
            and item["state"] == "pending"
            and item["reason_code"] == "cleanup_unavailable"
            for item in status
        )

        service._artifact_clock = lambda: pending["next_attempt_at"] + 1
        service.prepare_room(room_binding)
        outbox = RoomArtifactOutbox(service.db_path)
        assert outbox.list(captured[0].scope) == []
        assert outbox.retirement_complete(captured[0].scope)
        assert not blob.exists()
        assert not service.status("room")["pending_actions"]


@pytest.mark.asyncio
@pytest.mark.parametrize("loss", ["owner", "epoch"])
async def test_failed_producer_capture_loss_recovers_after_restored_owner_cold_continuation(
    tmp_path, monkeypatch, caplog, loss
):
    from gateway import session_hosted_output
    from gateway.hosted_room_artifacts import RoomArtifactOutbox
    from gateway.session_hosted_output import current_output_binding
    from gateway.session_hosted_output_lifecycle import records
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from tools.hosted_room_artifact import share_group_file

    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        output = tmp_path / "cache" / f"{loss}-loss.txt"
        output.parent.mkdir(exist_ok=True)
        output.write_bytes(b"private output retained by invalidated owner")
        captured, changes = [], []
        capture = session_hosted_output.capture_failed_output

        def capture_while_invalidated(current_authority, row, binding):
            before = current_authority.db._conn.total_changes
            if loss == "owner":
                retained = current_authority.hosted_room_service
                current_authority.hosted_room_service = object()
            else:
                retained = current_authority.epoch
                current_authority.epoch += 1
            try:
                return capture(current_authority, row, binding)
            finally:
                if loss == "owner":
                    current_authority.hosted_room_service = retained
                else:
                    current_authority.epoch = retained
                changes.append((before, current_authority.db._conn.total_changes))

        monkeypatch.setattr(session_hosted_output, "capture_failed_output", capture_while_invalidated)

        async def fail_after_output(_event):
            binding = current_output_binding()
            assert binding is not None
            captured.append(binding)
            assert json.loads(await asyncio.to_thread(share_group_file, str(output)))["ok"] is True
            raise RuntimeError(f"original {loss} producer failure")

        runner._handle_message = fail_after_output
        _, _, _, task, _ = await execute_group_turn(authority, service, defer_publication=True)

        assert tasks.get_task(service.db_path, task["identity"])["status"] == "failed"
        assert len(changes) == 1 and changes[0][0] == changes[0][1]
        assert captured[0].cleanup_pending is True
        assert captured[0].cleanup_reason == "owner_unavailable"
        assert f"original {loss} producer failure" in caplog.text
        with authority.db._read_ctx() as conn:
            assert records(conn, "room") == []
            row = conn.execute(
                "SELECT acknowledged_at, cleanup_required_at, blob_name "
                "FROM hosted_room_output_artifacts "
                "WHERE scope_key=?",
                (captured[0].scope.key,),
            ).fetchone()
        assert row is not None
        assert row["acknowledged_at"] is None and row["cleanup_required_at"] is None
        assert output.read_bytes() == b"private output retained by invalidated owner"

        # Continue through a fresh service with no access to the process-local
        # producer flag. The exact persisted outbox row is the additional proof.
        scope = captured[0].scope
        blob = tmp_path / "hosted-room-artifact-outbox" / "blobs" / row["blob_name"]
        captured.clear()
        cold = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
        authority.hosted_room_service = cold
        room_binding = cold.bindings()[0]
        published = []
        publish = cold._publish_one_output

        def publish_after_cleanup(room, current, progress):
            with authority.db._read_ctx() as conn:
                remaining = conn.execute(
                    "SELECT 1 FROM hosted_room_output_artifacts WHERE scope_key=?", (scope.key,)
                ).fetchall()
            assert remaining == []
            assert not blob.exists()
            published.append(current["identity"].task_id)
            return publish(room, current, progress)

        cold._publish_one_output = publish_after_cleanup
        with monkeypatch.context() as cleanup_fault:
            def fail_exact_unlink(*_args, **kwargs):
                if kwargs.get("dir_fd") is not None:
                    raise OSError("test-owned restored-owner cleanup fault")
                raise AssertionError("cleanup must use a directory-relative exact blob name")

            cleanup_fault.setattr("gateway.hosted_room_output_discard.os.unlink", fail_exact_unlink)
            cold.prepare_room(room_binding)

        assert published == []
        with authority.db._read_ctx() as conn:
            pending, = [record for _, record in records(conn, "room")]
            retained = conn.execute(
                "SELECT acknowledged_at, cleanup_required_at, blob_reclaimed_at "
                "FROM hosted_room_output_artifacts WHERE scope_key=?", (scope.key,)
            ).fetchone()
        assert pending["state"] == "pending"
        assert pending["reason_code"] == "cleanup_unavailable"
        assert retained is not None and retained["acknowledged_at"] is not None
        assert retained["cleanup_required_at"] is not None
        assert retained["blob_reclaimed_at"] is None and blob.exists()
        assert any(
            item["kind"] == "output_cleanup"
            and item["state"] == "pending"
            and item["reason_code"] == "cleanup_unavailable"
            for item in cold.status("room")["pending_actions"]
        )

        cold._artifact_clock = lambda: pending["next_attempt_at"] + 1
        cold.prepare_room(room_binding)
        outbox = RoomArtifactOutbox(cold.db_path)
        assert outbox.list(scope) == []
        assert outbox.retirement_complete(scope)
        assert published == [task["identity"].task_id]
        assert not cold.status("room")["pending_actions"]
        assert output.read_bytes() == b"private output retained by invalidated owner"


@pytest.mark.asyncio
async def test_failed_producer_current_invalid_owner_refuses_recovery_and_publication(
    tmp_path, monkeypatch
):
    from gateway import session_hosted_output
    from gateway.hosted_room_artifacts import RoomArtifactError
    from gateway.session_hosted_output import current_output_binding
    from gateway.session_hosted_output_lifecycle import records
    from tools.hosted_room_artifact import share_group_file

    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        output = tmp_path / "cache" / "current-invalid-owner.txt"
        output.parent.mkdir(exist_ok=True)
        output.write_bytes(b"private output retained while owner is invalid")
        captured = []
        capture = session_hosted_output.capture_failed_output

        def capture_while_invalidated(current_authority, row, binding):
            retained = current_authority.hosted_room_service
            current_authority.hosted_room_service = object()
            try:
                return capture(current_authority, row, binding)
            finally:
                current_authority.hosted_room_service = retained

        monkeypatch.setattr(session_hosted_output, "capture_failed_output", capture_while_invalidated)

        async def fail_after_output(_event):
            binding = current_output_binding()
            assert binding is not None
            captured.append(binding)
            assert json.loads(await asyncio.to_thread(share_group_file, str(output)))["ok"] is True
            raise RuntimeError("producer failure before current-owner refusal")

        runner._handle_message = fail_after_output
        _, _, _, task, room_binding = await execute_group_turn(
            authority, service, defer_publication=True
        )
        published = []
        service._publish_one_output = lambda *_args, **_kwargs: published.append(True)
        retained_service = authority.hosted_room_service
        authority.hosted_room_service = object()
        before = authority.db._conn.total_changes
        try:
            with pytest.raises(RoomArtifactError, match="owner changed"):
                service.prepare_room(room_binding)
        finally:
            authority.hosted_room_service = retained_service

        assert authority.db._conn.total_changes == before
        assert published == []
        assert tasks.get_task(service.db_path, task["identity"])["status"] == "failed"
        with authority.db._read_ctx() as conn:
            assert records(conn, "room") == []
            row = conn.execute(
                "SELECT acknowledged_at, cleanup_required_at FROM hosted_room_output_artifacts "
                "WHERE scope_key=?", (captured[0].scope.key,)
            ).fetchone()
        assert row is not None
        assert row["acknowledged_at"] is None and row["cleanup_required_at"] is None
        assert output.read_bytes() == b"private output retained while owner is invalid"


@pytest.mark.asyncio
async def test_failed_hosted_turn_without_output_creates_no_cleanup_obligation(tmp_path, monkeypatch):
    from gateway.session_hosted_output import current_output_binding
    from gateway.session_hosted_output_lifecycle import records

    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        async def fail_without_output(_event):
            binding = current_output_binding()
            assert binding is not None and binding.used is False
            raise RuntimeError("failure before any private output")

        runner._handle_message = fail_without_output
        _, _, _, task, room_binding = await execute_group_turn(
            authority, service, defer_publication=True
        )
        service.prepare_room(room_binding)

        assert tasks.get_task(service.db_path, task["identity"])["status"] == "failed"
        with authority.db._read_ctx() as conn:
            assert records(conn, "room") == []
            assert conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='hosted_room_output_artifacts'"
            ).fetchone() is None
        assert not any(event["kind"] == "message.member" for event in service._events("room"))
