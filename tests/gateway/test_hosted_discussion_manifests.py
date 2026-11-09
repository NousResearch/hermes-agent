"""Exact event-bound media survives deterministic planning and reconstruction."""
from copy import deepcopy

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.hosted_room_attachments import HostedRoomAttachmentStore


ROOM = {"room_id": "room", "name": "Files", "authority_gateway_id": "gw", "authority_epoch": 1,
        "members": [{"member_id": name, "profile": name, "handle": name, "display_name": name}
                    for name in ("alice", "bob", "carol")]}
PROFILES = ("alice", "bob", "carol")


def attachment(n):
    return {"attachment_id": f"att_{n:032x}", "kind": "file", "name": "notes.txt",
            "size": 1, "mime": "text/plain"}


def event(seq, *, attachments=(), text="@alice", kind="message.user", payload=None, actor=None):
    return {"room_id": "room", "seq": seq, "event_id": f"event-{seq}", "kind": kind,
            "authority_epoch": 1, "actor": actor or {"kind": "user", "id": "user"},
            "payload": payload or {"text": text, "thread_id": "thread", "attachments": list(attachments)}}


def plan(events, **kwargs):
    return discussion.plan_next_task(ROOM, events, local_profiles=PROFILES, **kwargs).task


def test_bound_manifest_reconstructs_frozen_input_and_rejects_forgery(tmp_path):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(db, **{key: val for key, val in ROOM.items() if key != "authority_epoch"})
    store = HostedRoomAttachmentStore(db)
    meta = store.put(room_id="room", upload_id="upload", kind="file", name="notes.txt", mime="text/plain", data=b"x")
    meta = {key: meta[key] for key in ("attachment_id", "kind", "name", "size", "mime")}
    store.commit_message_with_receipt(room_id="room", event_id="event-1", manifest=[meta],
                                      recipient_member_ids=list(PROFILES), viewer_access=True, hold_until_event=True)
    raw = event(1, attachments=[meta], text="")
    hosted_rooms.append_event(db, **{key: val for key, val in raw.items() if key != "seq"}, authority_gateway_id="gw")
    events = hosted_rooms.read_events(db, room_id="room", since_seq=0)["events"]
    task = plan(events, freeze_input_context=True)
    expected = [{**meta, "event_id": "event-1"}]
    assert task.payload["attachments"] == expected
    normalized, _, digest = driver._task_payload(dict(task.payload))
    assert normalized == task.payload
    driver.admit_task(db, task.identity, payload=normalized, clock=lambda: 10)
    saved = driver.get_task(db, task.identity)
    assert saved["payload"] == normalized
    events.append(event(2, attachments=[attachment(2)]))
    assert discussion.reconstruct_task_plan(ROOM, events, saved, local_profiles=PROFILES) == task
    for change in ({"event_id": "event-2"}, {"name": "different.txt"}):
        forged = deepcopy(saved)
        forged["payload"]["attachments"][0].update(change)
        assert driver._task_payload(forged["payload"])[2] != digest
        with pytest.raises(discussion.DiscussionReconstructionError):
            discussion.reconstruct_task_plan(ROOM, events, forged, local_profiles=PROFILES)
    for invalid in ([], [{**expected[0], "size": True}], [{**expected[0], "path": "/private"}],
                    [{k: v for k, v in expected[0].items() if k != "event_id"}]):
        with pytest.raises(driver.DriverValidationError):
            driver._task_payload({**normalized, "attachments": invalid})


def test_member_recipients_and_bounded_batches_preserve_unconsumed_files():
    events = [event(i, attachments=[attachment(i * 10 + n) for n in range(8)], text="@alice")
              for i in range(1, 4)]
    first = plan(events, freeze_input_context=True)
    assert first.seen_through_seq == 2
    assert len(first.payload["attachments"]) == 16
    publication = discussion.plan_publication(ROOM, events, first, status="settled",
                                             result={"text": "Review @bob"}, local_profiles=PROFILES)
    assert publication.terminal_kind == "turn.settled"
    for effect in publication.events:
        raw = {**effect.append_kwargs("room"), "seq": len(events) + 1}
        if raw["kind"] == "message.member":
            raw["payload"] = {**raw["payload"], "attachments": [attachment(99)],
                              "recipient_member_ids": ["bob"]}
        events.append(raw)
    second = plan(events, freeze_input_context=True)
    assert second.member.member_id == "alice"
    assert second.payload["attachments"] == [{**attachment(30 + n), "event_id": "event-3"} for n in range(8)]
    assert discussion.reconstruct_task_plan(ROOM, events,
        {"identity": first.identity, "payload": first.payload}, local_profiles=PROFILES) == first
    # Bob receives the peer share; Carol and the producer do not.
    for target in PROFILES:
        tailored = deepcopy(events)
        tailored[2]["payload"]["text"] = f"@{target}"
        candidate = plan(tailored, initial_watermarks={("thread", target): 3}, freeze_input_context=True)
        if target == "alice":
            continue  # already terminal and no unconsumed eligible attachment
        assert candidate is not None
        assert candidate.payload.get("attachments", []) == (
            [{**attachment(99), "event_id": "dmessage:" + first.identity.task_id.removeprefix("dtask:")}]
            if target == "bob" else [])
    invalid = deepcopy(events)
    invalid[3]["payload"]["recipient_member_ids"] = ["outsider"]
    with pytest.raises(discussion.DiscussionValidationError):
        plan(invalid)
