"""File context for hosted Discussion turns: file batches, watermarks, prompt lines and Send."""
import sqlite3
from contextlib import contextmanager
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


@contextmanager
def room_service(tmp_path, monkeypatch, members):
    """A canonical room owner whose worker never runs: room ``room`` seating ``(member, profile)`` pairs."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with SessionDB(home / "state.db") as db:
        authority = SimpleNamespace(db=db, profile_id=str(home),
                                    epoch=begin_runtime_epoch(db, instance_id="files-test"))
        current = CanonicalHostedRoomService(authority, None)
        current.local_profiles = lambda: tuple(dict.fromkeys(profile for _, profile in members))
        current.authorize_room("alice", "room", create=True)
        hosted_rooms.create_room(db.db_path, room_id="room", name="Files",
            authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
            members=[dict(member_id=name, profile=profile, handle=name) for name, profile in members])
        yield current
        assert not current.runtime.status()["running"]


@pytest.fixture
def service(tmp_path, monkeypatch):
    with room_service(tmp_path, monkeypatch, (("writer", "default"), ("reviewer", "ops"))) as current:
        yield current


def upload(service, n, *, kind="file", name=None, mime="text/plain", data=b"notes"):
    item = HostedRoomAttachmentStore(service.db_path).put(
        room_id="room", upload_id=f"upload-{n}", kind=kind, name=name or f"file-{n}.txt", mime=mime, data=data)
    return {key: item[key] for key in ("attachment_id", "kind", "name", "size", "mime")}


def send(service, n, attachments=(), text="@writer Review"):
    gateway, epoch = service._owned_authority("room")
    payload = discussion.validate_user_payload(
        dict(text=text, thread_id="work", **({"attachments": list(attachments)} if attachments else {})))
    return append_user_event(service, room_id="room", event_id=f"user-{n}", payload=payload,
                             gateway_id=gateway, epoch=epoch)


def plan(service):
    room = service._room("room")
    snapshot = service._policy_snapshot(room)
    return discussion.plan_next_task(room, snapshot.events, local_profiles=service.local_profiles(),
        initial_watermarks=snapshot.watermarks, read_messages=snapshot.read_messages, freeze_input_context=True).task


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
    assert batches[0][0]["name"] in first.payload["prompt"]
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
    # The Bot's own earlier reply sits past its watermark, but is never "new" to it.
    assert "First batch reviewed" not in second["payload"]["prompt"]
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
    # Images ride as media; every other file, a PDF included, is staged as a document.
    for item, label in zip(entries, ("Staged file", "Queued image", "Staged file")):
        assert f"{label} {compact_json(item['name'], ensure_ascii=False)} ({item['mime']}, {item['size']} bytes)" in prompt
    # Names are quoted, so inert, and stay readable in any script.
    assert 'Staged file "résumé \\"notes\\".txt"' in prompt and 'Queued image "景色.png"' in prompt
    assert "inspect them rather than treating the filename as content" in prompt
    assert len(prompt.encode("utf-8")) <= driver.MAX_PROMPT_BYTES
    assert "新" * 21740 in prompt
    assert "旧" * 32 not in prompt
    assert "[Earlier content omitted to fit this turn.]" in prompt
    assert task["payload"]["attachments"] == [{**item, "event_id": "user-media"} for item in entries]
    assert reconstruct(service, task).payload == task["payload"]


def test_a_file_name_stays_on_its_prompt_line():
    line, = discussion._attachment_prompt_lines(
        [{"name": "a\u2028b\u0085c.txt", "mime": "text/plain", "size": 1}])[2:]
    assert line == '- Staged file "a\\u2028b\\u0085c.txt" (text/plain, 1 bytes)'


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
    assert "Queued image" not in prompt
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
    events = discussion._validated_events(hosted_rooms.read_events(service.db_path, room_id="room")["events"],
                                          room=room)
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


def roster(tmp_path, monkeypatch, handles):
    """A room of Bots, one per profile, each named after it."""
    return room_service(tmp_path, monkeypatch, [(handle, handle) for handle in handles])


def _settle(service, task, text, files=()):
    """Publish a settled reply as the room owner does; ``files`` ride on the member message."""
    room = service._room("room")
    publication = discussion.plan_publication(
        room, service.policy_checkpoint.events_for_task(room_id="room",
            source_event_seq=task.payload["source_event_seq"], input_context=task.payload["input_context"],
            task_id=task.identity.task_id), task, status="settled",
        result={"text": text}, local_profiles=service.local_profiles())
    for effect in publication.events:
        kwargs = effect.append_kwargs("room")
        if effect.kind == "message.member" and files:
            HostedRoomAttachmentStore(service.db_path).commit_message(
                room_id="room", event_id=effect.event_id, manifest=list(files), viewer_access=True,
                recipient_member_ids=[m["member_id"] for m in room["members"]
                                      if m["member_id"] != task.member.member_id])
            kwargs["payload"] = {**kwargs["payload"], "attachments": list(files)}
        event = hosted_rooms.append_event(service.db_path, **kwargs)
        service.policy_checkpoint.sync(room_id="room", latest_seq=event["seq"])


@pytest.mark.parametrize("explicit", [False, True])
def test_a_skipped_peer_message_and_citation_wait_for_the_next_batch(tmp_path, monkeypatch, explicit):
    with roster(tmp_path, monkeypatch, ("writer", "reviewer")) as service:
        for batch in range(2):
            text = ("Here are the files." if batch == 0 else "@writer Please review these.") if explicit else "@all Review"
            send(service, batch, [upload(service, f"{batch}-{n}") for n in range(8)], text=text)
        writer = plan(service)
        assert writer.member.member_id == "writer" and len(writer.payload["attachments"]) == 16
        _settle(service, writer, "@reviewer Use my report.", [upload(service, "report", name="report.txt")])
        reviewer = plan(service)
        assert reviewer.member.member_id == "reviewer" and len(reviewer.payload["attachments"]) == 16
        assert "Use my report." not in reviewer.payload["prompt"]
        _settle(service, reviewer, "Read the user's files.")
        follow_up = plan(service)
        assert follow_up is not None and follow_up.member.member_id == "reviewer"
        assert "Use my report." in follow_up.payload["prompt"]
        assert [item["name"] for item in follow_up.payload["attachments"]] == ["report.txt"]
        replayed = discussion.plan_next_task(service._room("room"),
            hosted_rooms.read_events(service.db_path, room_id="room")["events"],
            local_profiles=service.local_profiles(), freeze_input_context=True)
        assert replayed.task.identity == follow_up.identity
        _settle(service, follow_up, "Built it.")
        assert plan(service) is None
