"""File context for hosted Discussion turns: prompt lines, batches, watermarks and Send."""
import sqlite3
from types import SimpleNamespace

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.hosted_room_attachments import MAX_TASK_ATTACHMENTS, HostedRoomAttachmentStore
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
from gateway.hosted_rooms_common import compact_json
from gateway.session_hosted_attachments import append_user_event
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


@pytest.fixture
def service(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with SessionDB(home / "state.db") as db:
        authority = SimpleNamespace(db=db, profile_id=str(home),
                                    epoch=begin_runtime_epoch(db, instance_id="files-test"))
        current = CanonicalHostedRoomService(authority, None)
        current.local_profiles = lambda: ("default", "ops")
        current.authorize_room("alice", "room", create=True)
        hosted_rooms.create_room(db.db_path, room_id="room", name="Files",
            authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
            members=[dict(member_id=name, profile=profile, handle=name)
                     for name, profile in (("writer", "default"), ("reviewer", "ops"))])
        yield current
        assert not current.runtime.status()["running"]


def upload(service, n, *, kind="file", name=None, mime="text/plain", data=b"notes"):
    item = HostedRoomAttachmentStore(service.db_path).put(
        room_id="room", upload_id=f"upload-{n}", kind=kind, name=name or f"file-{n}.txt", mime=mime, data=data)
    return {key: item[key] for key in ("attachment_id", "kind", "name", "size", "mime")}


def send(service, n, attachments=(), text="@writer Review"):
    gateway, epoch = service._owned_authority("room")
    payload = discussion.validate_user_payload(
        dict(text=text, thread_id="work", **({"attachments": list(attachments)} if attachments else {})),
        member_ids=("writer", "reviewer"))
    return append_user_event(service, room_id="room", event_id=f"user-{n}", payload=payload,
                             gateway_id=gateway, epoch=epoch)


def plan(service):
    room = service._room("room")
    snapshot = service._policy_snapshot(room)
    return discussion.plan_next_task(room, snapshot.events, local_profiles=service.local_profiles(),
        initial_watermarks=snapshot.watermarks, freeze_input_context=True).task


def reconstruct(service, saved):
    payload = saved["payload"]
    events = service.policy_checkpoint.events_for_task(room_id="room",
        source_event_seq=payload["source_event_seq"], input_context=payload["input_context"],
        task_id=saved["identity"].task_id)
    return discussion.reconstruct_task_plan(service._room("room"), events, saved,
                                            local_profiles=service.local_profiles())


@pytest.mark.parametrize("cleanup", [False, True])
def test_checkpoint_plans_the_remaining_file_batch(service, cleanup):
    batches = []
    for batch, count in enumerate((8, 8, 1)):
        entries = [upload(service, f"{batch}-{n}") for n in range(count)]
        send(service, batch, entries)
        batches.append([{**item, "event_id": f"user-{batch}"} for item in entries])
    first = plan(service)
    assert first is not None
    assert first.payload["attachments"] == batches[0] + batches[1]
    assert len(first.payload["attachments"]) == MAX_TASK_ATTACHMENTS
    assert first.seen_through_seq < first.payload["source_event_seq"]
    assert batches[2][0]["name"] not in first.payload["prompt"]
    events = hosted_rooms.read_events(service.db_path, room_id="room")["events"]
    publication = discussion.plan_publication(service._room("room"), events, first,
        status="settled", result={"text": "First batch reviewed"}, local_profiles=service.local_profiles())
    if cleanup:
        gateway, epoch = service._owned_authority("room")
        event = hosted_rooms.append_event(service.db_path, room_id="room", event_id="cleanup",
            kind="room.activity", actor={"kind": "gateway", "id": gateway},
            authority_gateway_id=gateway, authority_epoch=epoch,
            payload=dict(status="settled", reason_code="silent_round", thread_id="work",
                         discussion_event_id=first.discussion_event_id))
        service.policy_checkpoint.sync(room_id="room", latest_seq=event["seq"])
    for effect in publication.events:
        event = hosted_rooms.append_event(service.db_path, **effect.append_kwargs("room"))
        service.policy_checkpoint.sync(room_id="room", latest_seq=event["seq"])
    if cleanup:
        assert service._policy_snapshot(service._room("room")).events == ()
        with sqlite3.connect(service.db_path) as conn:
            assert not conn.execute("SELECT seq FROM hosted_room_policy_events WHERE discussion_event_id=?",
                                    (first.discussion_event_id,)).fetchall()
        send(service, "followup")
    cold = CanonicalHostedRoomService(service.authority, None)
    cold.local_profiles = service.local_profiles
    # Preparation must admit the remaining event, not move the checkpoint past it
    # because the earlier reply was visible.
    cold.prepare_room(cold.bindings()[0])
    tasks = driver.list_tasks(service.db_path, room_id="room", status="queued")
    assert len(tasks) == 1
    second = tasks[0]
    assert second["payload"].get("attachments") == batches[2]
    assert batches[2][0]["name"] in second["payload"]["prompt"]
    assert batches[0][0]["name"] not in second["payload"]["prompt"]
    assert second["payload"]["input_context"]["watermark"] == first.seen_through_seq
    assert reconstruct(cold, second).payload == second["payload"]
    # Even removing both derived projections cannot change a frozen admission.
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("DELETE FROM hosted_room_policy_events WHERE room_id='room'")
        conn.execute("DELETE FROM hosted_room_policy_transcript WHERE room_id='room'")
    assert reconstruct(cold, second).payload == second["payload"]
    assert driver.get_task(service.db_path, second["identity"])["payload"] == second["payload"]
    assert not cold.runtime.status()["running"]


def test_prompt_reserves_bounded_file_and_media_lines(service):
    entries = [
        upload(service, "file", name='résumé "notes".txt'),
        upload(service, "image", kind="image", name="景色.png", mime="image/png", data=b"\x89PNG\r\n\x1a\n"),
        upload(service, "pdf", kind="pdf", name="研究.pdf", mime="application/pdf", data=b"%PDF-1.4\n"),
    ]
    send(service, "media", entries)
    # Fill to the driver's byte limit: file lines are reserved before whole transcript
    # lines are selected, never appended past the limit.
    send(service, "older", text="旧" * 21740)
    send(service, "latest", text="@writer " + "新" * 21740)
    service.prepare_room(service.bindings()[0])
    task, = driver.list_tasks(service.db_path, room_id="room", status="queued")
    prompt = task["payload"]["prompt"]
    for item, label in zip(entries, ("Staged file", "Queued image", "Queued PDF")):
        assert f"{label} {compact_json(item['name'])} ({item['mime']}, {item['size']} bytes)" in prompt
    assert "inspect the supplied media rather than treating its filename as content" in prompt
    assert len(prompt.encode("utf-8")) <= driver.MAX_PROMPT_BYTES
    assert "新" * 21740 in prompt
    assert "旧" * 32 not in prompt
    assert "[Earlier content omitted to fit this turn.]" in prompt
    assert task["payload"]["attachments"] == [{**item, "event_id": "user-media"} for item in entries]
    assert reconstruct(service, task).payload == task["payload"]


def test_file_lines_and_omission_notice_never_overflow_the_prompt(service):
    item = upload(service, "boundary", name="file.txt", data=b"notes")
    send(service, "file", [item], text="@writer Review")
    send(service, "older", text="o" * 64895)
    newest = "@writer " + "n" * 65522
    send(service, "latest", text=newest)

    service.prepare_room(service.bindings()[0])
    task, = driver.list_tasks(service.db_path, room_id="room", status="queued")
    prompt = task["payload"]["prompt"]
    assert len(prompt.encode("utf-8")) <= driver.MAX_PROMPT_BYTES
    assert newest in prompt
    assert 'Staged file "file.txt" (text/plain, 5 bytes)' in prompt
    assert "  User (user): @writer Review" not in prompt
    assert task["payload"]["attachments"] == [{**item, "event_id": "user-file"}]
    assert reconstruct(service, task).payload == task["payload"]


def test_text_only_prompts_and_earlier_file_admissions_stay_unchanged(service):
    send(service, "text", text="@writer Review")
    text_task = plan(service)
    assert text_task is not None
    text_prompt = text_task.payload["prompt"]
    assert "@writer Review" in text_prompt
    assert "Attachments available to you for this turn" not in text_prompt
    assert "Staged file" not in text_prompt
    assert not text_task.payload.get("attachments")
    # A file task admitted before file lines existed used exactly that prompt.
    # It must reconstruct and stay frozen as admitted, not be rebuilt with file lines.
    item = upload(service, "frozen")
    send(service, "frozen", [item], text="@writer Review")
    room = discussion.validate_room(service._room("room"), local_profiles=service.local_profiles())
    events = discussion._validated_events(hosted_rooms.read_events(service.db_path, room_id="room")["events"], room=room)
    old = discussion._make_task_plan(room=room, discussion_event=events[-1], member=room.members[0],
        member_index=0, round_index=0, seen_through_seq=events[-1].seq, prompt=text_prompt,
        input_context={"watermark": events[0].seq, "event_seqs": [events[-1].seq]},
        attachments=[{**item, "event_id": events[-1].event_id}])
    driver.admit_task(service.db_path, old.identity, payload=old.payload, clock=lambda: 10)
    saved = driver.get_task(service.db_path, old.identity)
    before_digest = driver._task_payload(saved["payload"])[2]
    later = upload(service, "later", name="later.txt", data=b"later share")
    send(service, "later", [later], text="@writer Review the later share")
    service.policy_checkpoint = HostedRoomPolicyCheckpoint(service.db_path)
    service.prepare_room(service.bindings()[0])
    assert reconstruct(service, saved).payload == old.payload
    after = driver.get_task(service.db_path, old.identity)
    assert after["payload"] == saved["payload"]
    assert driver._task_payload(after["payload"])[2] == before_digest


def test_send_defaults_the_thread_and_checks_the_roster_before_committing_files(service, tmp_path):
    event = service.send(room_id="room", event_id="user-plain", payload={"text": "@writer hello"})
    assert event["payload"]["thread_id"] == "user-plain"
    gateway = hosted_rooms.local_authority_gateway_id()
    hosted_rooms.create_room(service.db_path, room_id="empty", name="No recipients",
                             authority_gateway_id=gateway, members=[])
    store = HostedRoomAttachmentStore(service.db_path)
    item = store.put(room_id="empty", upload_id="spec", kind="file", name="spec.txt",
                     mime="text/plain", data=b"spec")
    manifest = [{key: item[key] for key in ("attachment_id", "kind", "name", "size", "mime")}]
    with pytest.raises(discussion.DiscussionValidationError, match="attachment member ids"):
        service.send(room_id="empty", event_id="user-file", payload={"text": "spec", "attachments": manifest})
    assert store.find_upload(room_id="empty", upload_id="spec")["state"] == "uploaded"


def _checkpoint_room(tmp_path):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(db, room_id="checkpoint", name="Checkpoint", members=[],
                             authority_gateway_id="home")
    for index in (1, 2):
        hosted_rooms.append_event(db, room_id="checkpoint", event_id=f"user-{index}", kind="message.user",
                                  actor={"kind": "user", "id": "owner"}, authority_gateway_id="home",
                                  authority_epoch=1, payload={"text": f"request {index}", "thread_id": "work"})
    checkpoint = HostedRoomPolicyCheckpoint(db)
    checkpoint.sync(room_id="checkpoint", latest_seq=2)
    return db, checkpoint


@pytest.mark.parametrize("seen_through_seq", [1, 2])
def test_late_settlement_keeps_a_partial_file_watermark_after_cleanup(tmp_path, seen_through_seq):
    db, checkpoint = _checkpoint_room(tmp_path)
    shared = {"thread_id": "work", "discussion_event_id": "user-2", "member_id": "builder",
              "task_id": "dtask:late"}
    appended = []
    for event_id, kind, fields in (
            ("cleanup", "room.activity", {"status": "settled"}),
            ("dmessage:late", "message.member", {"text": "Reviewed"}),
            ("dterminal:late", "turn.settled",
             {"message_event_id": "dmessage:late", "seen_through_seq": seen_through_seq})):
        actor = {"kind": "member", "id": "builder"} if kind == "message.member" else {"kind": "gateway", "id": "home"}
        event = hosted_rooms.append_event(db, room_id="checkpoint", event_id=event_id, kind=kind, actor=actor,
                                          authority_gateway_id="home", authority_epoch=1,
                                          payload={**shared, **fields})
        checkpoint.sync(room_id="checkpoint", latest_seq=event["seq"])
        appended.append(event)
    message, terminal = appended[-2:]
    cold = HostedRoomPolicyCheckpoint(db)
    assert cold.snapshot(room_id="checkpoint", latest_seq=terminal["seq"]).events == ()
    assert cold.publication_exists(room_id="checkpoint", task_id="dtask:late", status="settled",
                                   execution_generation=1)
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT seq FROM hosted_room_policy_events WHERE discussion_event_id='user-2'").fetchall() == []
        # A reply that saw only part of the input must not advance the member past the rest.
        assert conn.execute(
            "SELECT seen_through_seq FROM hosted_room_policy_watermarks WHERE member_id='builder'"
        ).fetchone()[0] == (1 if seen_through_seq == 1 else message["seq"])
        assert conn.execute("SELECT settled_seq FROM hosted_room_policy_transcript WHERE seq=?",
                            (message["seq"],)).fetchone()[0] == terminal["seq"]
