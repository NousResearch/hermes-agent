"""A default-profile Bot's shared files through the real FIFO, driver and room log.

Each test pins one guarantee: one member message per share with the files
attached, byte checks before publication and before the source ACK, and the
private copy acknowledged or discarded exactly once across retries, Stop,
failure, a lost response and a restart.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from gateway import hosted_room_driver as tasks, hosted_rooms as rooms
from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactOutbox, RoomArtifactScope
from gateway.session_contract import Principal
from tests.gateway.fixtures.hosted_output import (
    events, held_threads, obligations, outbox_rows, owner, run_turn, share, write_file,
)


def _download(service, tmp_path, event_id, attachment_id):
    from gateway.session_hosted_attachments import download
    actor = Principal("alice", str(tmp_path), frozenset({"session:read"}), "viewer")
    result = download(service, actor, dict(room_id="room", event_id=event_id, attachment_id=attachment_id))
    return base64.b64decode(result["data_base64"])


def _sharing_handler(path, results, *, reply="@reviewer Shared report.", fail=None):
    async def handle(event):
        from gateway.session_hosted_output import current_output_binding
        assert current_output_binding() is not None
        results.append(await asyncio.to_thread(share, path))
        if fail is not None:
            raise fail
        return reply
    return handle


@pytest.mark.asyncio
async def test_a_shared_file_is_published_once_on_the_member_message_and_acknowledged(tmp_path, monkeypatch):
    from gateway.session_hosted_output import current_output_binding
    from tools.registry import registry

    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        path = write_file(tmp_path)
        results, contexts = [], []

        async def handle(event):
            from contextvars import copy_context
            contexts.append(copy_context())
            results.append(await asyncio.to_thread(share, path))
            assert (await asyncio.to_thread(share, path))["artifact_id"] == results[0]["artifact_id"]
            return "@reviewer Shared report."
        runner._handle_message = handle
        turn = await run_turn(authority, service)

        assert results[0]["ok"] is True, results
        settled = tasks.get_task(service.db_path, turn.task["identity"])
        assert settled["status"] == "settled"
        assert settled["result"]["artifacts"]["items"][0]["sha256"] == results[0]["sha256"]
        message, = events(service, "message.member")
        assert message["payload"]["text"] == "@reviewer Shared report."
        attachment, = message["payload"]["attachments"]
        assert message["payload"]["recipient_member_ids"] == ["writer", "reviewer"]
        assert _download(service, tmp_path, message["event_id"], attachment["attachment_id"]) == path.read_bytes()
        # The next Bot turn receives the file through the room's own attachment feed.
        following, = tasks.list_tasks(service.db_path, room_id="room", status="queued")
        assert following["payload"]["target_member_id"] == "reviewer"
        assert following["payload"]["attachments"][0]["attachment_id"] == attachment["attachment_id"]
        # The private copy is acknowledged exactly once and its obligation is complete.
        row, = outbox_rows(service)
        assert row["acknowledged_at"] is not None and row["blob_reclaimed_at"] is not None
        obligation, = obligations(service)
        assert (obligation["operation"], obligation["state"]) == ("ack", "completed")
        # Replaying the room pass neither republishes nor re-acknowledges.
        service.prepare_room(turn.binding)
        assert len(events(service, "message.member")) == 1
        assert obligations(service) == [obligation]
        # The binding expired with the turn, including for a copied tool context.
        assert current_output_binding() is None
        assert json.loads(contexts[0].run(registry.dispatch, "share_group_file", {"path": str(path)}))["ok"] is False


@pytest.mark.asyncio
async def test_files_shared_with_a_pass_reply_still_get_their_one_member_message(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        results = []
        runner._handle_message = _sharing_handler(write_file(tmp_path, "notes.md", b"# Notes\n"), results,
                                                  reply="(pass)")
        await run_turn(authority, service)
        message, = events(service, "message.member")
        assert message["payload"]["text"] == "Shared notes.md."
        assert [a["name"] for a in message["payload"]["attachments"]] == ["notes.md"]


@pytest.mark.asyncio
async def test_a_text_only_room_publishes_exactly_as_before(tmp_path, monkeypatch):
    """Without shared files the room log, task state and status match the base publication."""
    import shutil
    from tui_gateway.hosted_room_service import HostedRoomService

    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        async def handle(event):
            return "@reviewer Text-only report ready."
        runner._handle_message = handle
        turn = await run_turn(authority, service, publish=False)
        room = service._room("room")
        snapshot = tmp_path / "base-copy.db"
        with authority.db._read_ctx() as conn:
            conn.execute("VACUUM INTO ?", (str(snapshot),))
        assert service._publish_terminal_tasks(room) is True
        ours = [(e["event_id"], e["kind"], e["actor"], e["payload"]) for e in events(service)]
        base = HostedRoomService.__new__(HostedRoomService)
        base.db_path, base.policy_checkpoint = snapshot, None
        from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
        base.policy_checkpoint = HostedRoomPolicyCheckpoint(snapshot)
        base.local_profiles = service.local_profiles
        base_room = rooms.room_state(snapshot, room_id="room")
        base.policy_checkpoint.snapshot(room_id="room", latest_seq=int(base_room["latest_seq"]))
        assert HostedRoomService._publish_terminal_tasks(base, base_room) is True
        theirs = [(e["event_id"], e["kind"], e["actor"], e["payload"])
                  for e in rooms.read_events(snapshot, room_id="room", limit=rooms.MAX_LOG_LIMIT)["events"]]
        assert ours == theirs
        assert outbox_rows(service) == [] and obligations(service) == []
        status = service.status("room")
        assert not [a for a in status["pending_actions"] if a["kind"] == "output_retry"]
        with authority.db._read_ctx() as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not {"hosted_room_output_artifacts", "hosted_room_output_obligations"} & tables
        shutil.rmtree(tmp_path / "cache", ignore_errors=True)
        del turn


@pytest.mark.asyncio
async def test_a_failed_turn_retires_its_files_before_the_failure_is_published(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        results = []
        runner._handle_message = _sharing_handler(write_file(tmp_path), results, fail=RuntimeError("model down"))
        turn = await run_turn(authority, service, publish=False)
        assert results[0]["ok"] is True
        # The producer committed the cleanup before its failed terminal existed.
        assert outbox_rows(service) == []
        failed = tasks.get_task(service.db_path, turn.task["identity"])
        assert failed["status"] == "failed" and "artifacts" not in failed["result"]
        service.prepare_room(turn.binding)
        assert [e["kind"] for e in events(service) if e["kind"].startswith("turn.")] == ["turn.failed"]
        assert not events(service, "message.member")


@pytest.mark.asyncio
async def test_a_failed_cleanup_holds_the_failure_until_the_files_are_gone(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        results = []
        runner._handle_message = _sharing_handler(write_file(tmp_path), results, fail=RuntimeError("model down"))
        original = RoomArtifactOutbox.discard_durably
        monkeypatch.setattr(RoomArtifactOutbox, "discard_durably",
                            lambda self, scope: (_ for _ in ()).throw(OSError("disk busy")))
        turn = await run_turn(authority, service, publish=False)
        assert len(outbox_rows(service)) == 1  # nothing was committed
        service.prepare_room(turn.binding)
        assert not [e for e in events(service) if e["kind"] == "turn.failed"]
        obligation, = obligations(service)
        assert (obligation["operation"], obligation["state"], obligation["reason_code"]) == (
            "discard", "pending", "transient")
        actions = [a for a in service.status("room")["pending_actions"] if a["kind"] == "output_retry"]
        assert actions and actions[0]["blocked"] is False and actions[0]["operation"] == "discard"
        assert held_threads(service) == {"thread"}
        monkeypatch.setattr(RoomArtifactOutbox, "discard_durably", original)
        monkeypatch.setattr(service, "_output_clock", lambda: obligation["next_attempt_at"] + 1)
        service.prepare_room(turn.binding)
        assert outbox_rows(service) == []
        assert [e["kind"] for e in events(service) if e["kind"].startswith("turn.")] == ["turn.failed"]
        assert obligations(service)[0]["state"] == "completed"
        assert held_threads(service) == frozenset()


@pytest.mark.asyncio
async def test_a_superseded_reply_discards_its_files_instead_of_publishing_them(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        results = []
        runner._handle_message = _sharing_handler(write_file(tmp_path), results)
        await run_turn(authority, service, publish=False)
        assert outbox_rows(service)[0]["acknowledged_at"] is None
        # A newer request in the same thread supersedes the late reply, files included.
        service.send(room_id="room", event_id="newer", payload=dict(thread_id="thread", text="@writer Never mind"))
        assert not events(service, "message.member")
        cancelled = [e for e in events(service) if e["kind"] == "turn.cancelled"]
        assert cancelled and cancelled[0]["payload"]["reason"] == "superseded_by_newer_user_event"
        assert outbox_rows(service) == []
        obligation, = obligations(service)
        assert (obligation["operation"], obligation["state"]) == ("discard", "completed")


@pytest.mark.asyncio
async def test_publication_retries_transient_faults_and_blocks_integrity_faults(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        results = []
        runner._handle_message = _sharing_handler(write_file(tmp_path), results)
        turn = await run_turn(authority, service, publish=False)
        original_read = RoomArtifactOutbox.read
        monkeypatch.setattr(RoomArtifactOutbox, "read", lambda *a, **k: (_ for _ in ()).throw(OSError("busy")))
        service.prepare_room(turn.binding)
        obligation, = obligations(service)
        assert (obligation["state"], obligation["reason_code"], obligation["attempts"]) == ("pending", "transient", 1)
        assert not events(service, "message.member") and held_threads(service) == {"thread"}
        # Before the backoff expires nothing is retried.
        service.prepare_room(turn.binding)
        assert obligations(service)[0]["attempts"] == 1
        monkeypatch.setattr(RoomArtifactOutbox, "read",
                            lambda *a, **k: (_ for _ in ()).throw(RoomArtifactError("room artifact bytes changed")))
        monkeypatch.setattr(service, "_output_clock", lambda: obligation["next_attempt_at"] + 1)
        service.prepare_room(turn.binding)
        blocked, = obligations(service)
        assert (blocked["state"], blocked["reason_code"], blocked["attempts"]) == ("blocked", "verification_failed", 2)
        action, = [a for a in service.status("room")["pending_actions"] if a["kind"] == "output_retry"]
        assert action["blocked"] is True and action["task_id"] == turn.task["identity"].task_id
        assert not events(service, "message.member") and held_threads(service) == {"thread"}
        assert "room artifact bytes changed" in service.runtime.status()["last_error"]
        monkeypatch.setattr(RoomArtifactOutbox, "read", original_read)
        service.prepare_room(turn.binding)
        assert not events(service, "message.member")  # blocked stays blocked
        # A newer request in the thread supersedes the stuck reply: its files are discarded
        # and the room moves on.
        service.send(room_id="room", event_id="retry", payload=dict(thread_id="thread", text="@writer Try again"))
        assert [e["kind"] for e in events(service) if e["kind"].startswith("turn.")] == ["turn.cancelled"]
        superseded, = obligations(service)
        assert (superseded["operation"], superseded["state"], superseded["reason_code"]) == (
            "discard", "completed", "completed")
        assert outbox_rows(service) == []
        assert held_threads(service) == frozenset()
        assert tasks.list_tasks(service.db_path, room_id="room", status="queued")  # the new request is planned


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["transient", "blocked"])
async def test_a_reply_waiting_on_its_files_holds_only_its_own_thread(tmp_path, monkeypatch, fault):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        runner._handle_message = _sharing_handler(write_file(tmp_path), [])
        turn = await run_turn(authority, service, publish=False)
        original_read = RoomArtifactOutbox.read
        error = OSError("busy") if fault == "transient" else RoomArtifactError("room artifact bytes changed")
        monkeypatch.setattr(RoomArtifactOutbox, "read", lambda *a, **k: (_ for _ in ()).throw(error))
        service.prepare_room(turn.binding)
        obligation, = obligations(service)
        assert obligation["state"] == ("pending" if fault == "transient" else "blocked")
        assert held_threads(service) == {"thread"}
        assert service._policy_snapshot(service._room("room")).events == ()  # nothing else to plan
        # The waiting thread is left out of Policy selection: another discussion goes ahead.
        service.send(room_id="room", event_id="other", payload=dict(thread_id="other", text="@reviewer Check B"))
        queued, = tasks.list_tasks(service.db_path, room_id="room", status="queued")
        assert queued["identity"].thread_id == "other"
        assert not events(service, "message.member") and held_threads(service) == {"thread"}
        if fault == "blocked":
            return  # it stays held until a newer request in its thread supersedes it (tested above)
        # Once its files go through, the waiting reply is published in its own thread.
        monkeypatch.setattr(RoomArtifactOutbox, "read", original_read)
        monkeypatch.setattr(service, "_output_clock", lambda: obligation["next_attempt_at"] + 1)
        service.prepare_room(turn.binding)
        message, = events(service, "message.member")
        assert message["payload"]["thread_id"] == "thread" and len(message["payload"]["attachments"]) == 1
        assert obligations(service)[0]["state"] == "completed" and held_threads(service) == frozenset()


@pytest.mark.asyncio
async def test_a_retry_after_a_partial_copy_publishes_every_file_once(tmp_path, monkeypatch):
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore

    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        first = write_file(tmp_path, "plan.md", b"# Plan\n")
        second = write_file(tmp_path, "data.csv", b"a,b\n1,2\n")

        async def handle(event):
            for path in (first, second):
                assert (await asyncio.to_thread(share, path))["ok"] is True
            return "@reviewer Shared both."
        runner._handle_message = handle
        turn = await run_turn(authority, service, publish=False)
        original_put, puts = HostedRoomAttachmentStore.put, []

        def put(self, **kwargs):
            puts.append(kwargs["name"])
            if len(puts) == 2:
                raise OSError("disk busy")  # the first file is already in the room store
            return original_put(self, **kwargs)
        monkeypatch.setattr(HostedRoomAttachmentStore, "put", put)
        service.prepare_room(turn.binding)
        obligation, = obligations(service)
        assert (obligation["state"], obligation["reason_code"]) == ("pending", "transient")
        assert not events(service, "message.member")
        monkeypatch.setattr(service, "_output_clock", lambda: obligation["next_attempt_at"] + 1)
        service.prepare_room(turn.binding)
        assert puts == ["plan.md", "data.csv", "plan.md", "data.csv"]
        message, = events(service, "message.member")
        attachments = message["payload"]["attachments"]
        assert [a["name"] for a in attachments] == ["plan.md", "data.csv"]
        # The retry reused the first copy: one stored file per share, both downloadable.
        with authority.db._read_ctx() as conn:
            assert conn.execute("SELECT COUNT(*) FROM hosted_room_attachments WHERE room_id='room'").fetchone()[0] == 2
        assert [_download(service, tmp_path, message["event_id"], a["attachment_id"]) for a in attachments] == [
            first.read_bytes(), second.read_bytes()]
        assert obligations(service)[0]["state"] == "completed"
        assert all(row["acknowledged_at"] is not None for row in outbox_rows(service))


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [True, False])
async def test_a_lost_ack_completes_without_republishing(tmp_path, monkeypatch, committed):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        results = []
        runner._handle_message = _sharing_handler(write_file(tmp_path), results)
        turn = await run_turn(authority, service, publish=False)
        original_ack = RoomArtifactOutbox.acknowledge
        calls = []

        def lose(self, scope, ids, *, message_event_id):
            calls.append(ids)
            if len(calls) == 1:
                if committed:
                    original_ack(self, scope, ids, message_event_id=message_event_id)
                raise TimeoutError("acknowledgement response lost")
            return original_ack(self, scope, ids, message_event_id=message_event_id)
        monkeypatch.setattr(RoomArtifactOutbox, "acknowledge", lose)
        service.prepare_room(turn.binding)
        assert len(events(service, "message.member")) == 1
        obligation, = obligations(service)
        assert (obligation["operation"], obligation["state"]) == ("ack", "pending")
        assert (outbox_rows(service)[0]["acknowledged_at"] is not None) is committed
        # The published terminal pins its task while the ACK is still owed.
        tomorrow = lambda: __import__("time").time() + 86400  # noqa: E731
        assert tasks.prune_published_terminal_tasks(service.db_path, room_id="room", clock=tomorrow, retain=0) == 0
        monkeypatch.setattr(service, "_output_clock", lambda: obligation["next_attempt_at"] + 1)
        service.prepare_room(turn.binding)
        assert len(events(service, "message.member")) == 1
        assert obligations(service)[0]["state"] == "completed"
        # A committed ACK is proven by its retirement evidence, not repeated.
        assert len(calls) == (1 if committed else 2)
        assert outbox_rows(service)[0]["acknowledged_at"] is not None
        assert tasks.prune_published_terminal_tasks(service.db_path, room_id="room", clock=tomorrow, retain=0) == 1
        service._retire_unowned_output("room")
        assert obligations(service) == []  # the receipt goes with its task


def _stopping_handler(service, path, shared, *, honor_stop):
    async def handle(event):
        from gateway.session_results import execution_result
        shared.append(await asyncio.to_thread(share, path))
        await asyncio.to_thread(service.stop_room, "room", cancel_id="stop")
        # Stop fenced the attempt: a later share is refused inside the outbox write.
        shared.append(await asyncio.to_thread(share, write_file(path.parent.parent, "later.txt", b"later\n")))
        if honor_stop:
            execution_result.get()["result"] = {"final_response": "", "messages": [], "interrupted": True}
            return ""
        return "@reviewer Finished anyway."
    return handle


async def _settle_stop(service, turn):
    current = tasks.get_task(service.db_path, turn.task["identity"])
    if current["status"] == "stopping":
        await asyncio.to_thread(service.runtime._finish_stop, turn.binding, current,
                                service.runtime._ensure_lease(turn.binding))
    service.prepare_room(turn.binding)
    return tasks.get_task(service.db_path, turn.task["identity"])


@pytest.mark.asyncio
async def test_a_stopped_turn_discards_its_files_and_publishes_no_reply(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        shared = []
        runner._handle_message = _stopping_handler(service, write_file(tmp_path), shared, honor_stop=True)
        turn = await run_turn(authority, service, publish=False)
        assert shared[0]["ok"] is True and shared[1]["ok"] is False
        assert outbox_rows(service) == []  # retired by the producer before its interrupted terminal
        assert (await _settle_stop(service, turn))["status"] == "cancelled"
        assert not events(service, "message.member")
        assert [e["kind"] for e in events(service) if e["kind"].startswith("turn.")] == ["turn.cancelled"]


@pytest.mark.asyncio
async def test_a_turn_that_finishes_before_stop_takes_effect_publishes_its_reply_and_files(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        shared = []
        runner._handle_message = _stopping_handler(service, write_file(tmp_path), shared, honor_stop=False)
        turn = await run_turn(authority, service, publish=False)
        assert shared[0]["ok"] is True and shared[1]["ok"] is False
        # Stop waited for the turn's real terminal; the completion won, as it does for text.
        assert (await _settle_stop(service, turn))["status"] == "settled"
        message, = events(service, "message.member")
        assert message["payload"]["text"] == "@reviewer Finished anyway."
        assert [a["name"] for a in message["payload"]["attachments"]] == ["report.txt"]
        assert outbox_rows(service)[0]["acknowledged_at"] is not None
        obligation, = obligations(service)
        assert (obligation["operation"], obligation["state"]) == ("ack", "completed")


@pytest.mark.asyncio
async def test_an_unknown_attempt_keeps_its_files_until_explicitly_discarded(tmp_path, monkeypatch):
    import time
    from hermes_state_runtime import begin_runtime_epoch, recover_session_inputs
    from tests.gateway.fixtures.hosted_output import admit_turn

    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        path = write_file(tmp_path)
        task, binding, _attempt = await admit_turn(service)
        rpc = service._resolve_member_transport(binding, task)
        coords = dict(profile="default", source="bot_room")
        sid = (await asyncio.to_thread(rpc.create, **coords, title="Group: room"))["session_id"]
        await asyncio.to_thread(rpc.submit, **coords, session_id=sid, prompt=task["payload"]["prompt"],
                                task=task["identity"], execution_generation=1, on_terminal=lambda r: None)
        crashed = asyncio.Event()

        async def handle(event):
            await asyncio.to_thread(share, path)
            crashed.set()
            await asyncio.sleep(3600)  # the process dies mid-turn
        runner._handle_message = handle
        drain = asyncio.ensure_future(authority._drain(rpc.ref))
        await asyncio.wait_for(crashed.wait(), timeout=5)
        drain.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain
        # Restart: the admission is unknown and the room driver finds the attempt abandoned.
        authority.epoch = begin_runtime_epoch(authority.db, instance_id="restarted")
        recover_session_inputs(authority.db, epoch=authority.epoch)
        later = lambda: time.time() + 120  # noqa: E731
        service.runtime.clock = later
        lease = service.runtime._ensure_lease(binding)
        tasks.recover_room(service.db_path, lease, clock=later)
        assert tasks.get_task(service.db_path, task["identity"])["status"] == "indeterminate"
        service.prepare_room(binding)
        assert len(outbox_rows(service)) == 1  # an unknown outcome is not guessed away
        await asyncio.to_thread(service.discard_room_task, "room", member_id="writer",
                                task_id=task["identity"].task_id, execution_generation=1)
        assert outbox_rows(service) == []
        assert tasks.get_task(service.db_path, task["identity"])["status"] == "cancelled"
        assert [e["kind"] for e in events(service) if e["kind"].startswith("turn.")] == ["turn.cancelled"]


@pytest.mark.asyncio
async def test_the_append_fence_refuses_output_for_a_changed_attempt(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        results = []
        runner._handle_message = _sharing_handler(write_file(tmp_path), results)
        turn = await run_turn(authority, service)
        stored = tasks.get_task(service.db_path, turn.task["identity"])
        scope = RoomArtifactScope.from_mapping(stored["result"]["artifact_scope"])
        message, = events(service, "message.member")
        expected = dict(scope=scope.as_mapping(), manifest=stored["result"]["artifacts"],
                        cancel_generation=stored["cancel_generation"] + 1)
        append = dict(room_id="room", event_id="late-output", kind=message["kind"], actor=message["actor"],
                      authority_gateway_id=scope.authority_gateway_id, authority_epoch=scope.authority_epoch)
        with pytest.raises(RoomArtifactError, match="attempt changed"):
            rooms.append_event(service.db_path, payload=message["payload"], expected_output=expected, **append)
        expected["cancel_generation"] -= 1
        with pytest.raises(RoomArtifactError, match="coordinates changed"):
            rooms.append_event(service.db_path, payload={**message["payload"], "task_id": "other-task"},
                               expected_output=expected, **append)
        with pytest.raises(RoomArtifactError, match="recipients changed"):
            rooms.append_event(service.db_path, payload={**message["payload"], "recipient_member_ids": ["writer"]},
                               expected_output=expected, **append)
        # The member now runs on another profile: its earlier output cannot be published as its own.
        with authority.db._read_ctx() as conn:
            members = json.loads(conn.execute("SELECT members_json FROM hosted_rooms WHERE room_id='room'")
                                 .fetchone()[0])
        moved = [{**m, "profile": "reviewer"} if m["member_id"] == "writer" else m for m in members]
        authority.db._execute_write(lambda conn: conn.execute(
            "UPDATE hosted_rooms SET members_json=? WHERE room_id='room'", (json.dumps(moved),)))
        with pytest.raises(RoomArtifactError, match="participant changed"):
            rooms.append_event(service.db_path, payload=message["payload"], expected_output=expected, **append)
        assert not [e for e in events(service) if e["event_id"] == "late-output"]


@pytest.mark.asyncio
async def test_a_late_receipt_retires_files_only_when_its_attempt_can_no_longer_publish(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        results = []
        runner._handle_message = _sharing_handler(write_file(tmp_path), results)
        turn = await run_turn(authority, service, publish=False)
        settled = tasks.get_task(service.db_path, turn.task["identity"])
        receipt = settled["result"]
        for status in ("settled", "running", "stopping", "indeterminate"):
            service.retire_stale_output(turn.binding, {**settled, "status": status}, 1, receipt)
        assert obligations(service) == []  # publication or a later harvest still owns it
        # Cancellation won: the receipt's files can never be published.
        def cancel(conn):
            conn.execute("UPDATE hosted_room_driver_tasks SET status='cancelled', result_json='{}' "
                         "WHERE room_id='room' AND task_id=?", (turn.task["identity"].task_id,))
        authority.db._execute_write(cancel)
        service.retire_stale_output(turn.binding, tasks.get_task(service.db_path, turn.task["identity"]), 1, receipt)
        obligation, = obligations(service)
        assert (obligation["operation"], obligation["state"]) == ("discard", "pending")
        service.prepare_room(turn.binding)
        assert outbox_rows(service) == []
        assert obligations(service)[0]["state"] == "completed"
        assert not events(service, "message.member")


@pytest.mark.asyncio
async def test_disband_retires_a_blocked_output_and_then_deletes_the_room(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        results = []
        runner._handle_message = _sharing_handler(write_file(tmp_path), results)
        turn = await run_turn(authority, service, publish=False)
        monkeypatch.setattr(RoomArtifactOutbox, "read",
                            lambda *a, **k: (_ for _ in ()).throw(RoomArtifactError("room artifact bytes changed")))
        service.prepare_room(turn.binding)
        assert obligations(service)[0]["state"] == "blocked"
        retired = []
        original_outcome = service._record_outcome

        def record(scope, **kwargs):
            saved = original_outcome(scope, **kwargs)
            retired.append(saved)
            return saved
        monkeypatch.setattr(service, "_record_outcome", record)
        service.stop_room("room", cancel_id="room-disbanded", require_acknowledged=True)
        assert [(row["operation"], row["state"], row["reason_code"]) for row in retired] == [
            ("discard", "completed", "room_disbanded")]
        assert outbox_rows(service) == []
        assert obligations(service) == []  # the room's receipts go with the room
        rooms.disband_room(service.db_path, room_id="room", expected_gateway_id=turn.binding.gateway_id,
                           expected_epoch=turn.binding.authority_epoch)


@pytest.mark.asyncio
async def test_a_terminal_whose_files_were_already_discarded_is_still_published(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        results = []
        runner._handle_message = _sharing_handler(write_file(tmp_path), results)
        turn = await run_turn(authority, service, publish=False)
        original = rooms.append_event
        raced = []

        def lose_cursor_race(*args, **kwargs):
            if kwargs.get("kind") == "turn.cancelled" and not raced:
                raced.append(True)
                raise rooms.EventCursorConflictError("room changed before event publication")
            return original(*args, **kwargs)
        monkeypatch.setattr(rooms, "append_event", lose_cursor_race)
        service.send(room_id="room", event_id="newer", payload=dict(thread_id="thread", text="@writer Never mind"))
        # The superseded files are discarded, then the cancelled terminal loses a cursor race.
        assert not [e for e in events(service) if e["kind"] == "turn.cancelled"]
        assert raced and outbox_rows(service) == []
        assert obligations(service)[0]["state"] == "completed"
        service.prepare_room(turn.binding)
        assert [e["kind"] for e in events(service) if e["kind"].startswith("turn.")] == ["turn.cancelled"]
        assert held_threads(service) == frozenset()


@pytest.mark.asyncio
async def test_an_owner_restart_finishes_a_committed_cleanup(tmp_path, monkeypatch):
    from gateway.session_hosted_output import replay_output_cleanups

    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        assert replay_output_cleanups(authority) is False  # no outbox: nothing is created
        results = []
        runner._handle_message = _sharing_handler(write_file(tmp_path), results, fail=RuntimeError("model down"))
        unlink = __import__("pathlib").Path.unlink
        monkeypatch.setattr(__import__("pathlib").Path, "unlink", lambda self, *a, **k: (_ for _ in ()).throw(
            OSError("disk busy")) if "hosted-room-artifact-outbox" in str(self) else unlink(self, *a, **k))
        await run_turn(authority, service, publish=False)
        row, = outbox_rows(service)
        assert row["cleanup_required_at"] is not None  # the intent committed, the bytes did not go
        monkeypatch.setattr(__import__("pathlib").Path, "unlink", unlink)
        assert replay_output_cleanups(authority) is True
        assert outbox_rows(service) == []
