"""File-producing replies spend a finite budget and do not starve later members."""
import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from tests.gateway.test_hosted_discussion_file_context import _settle, roster, send, upload


def decision(service, limits):
    room = service._room("room")
    snapshot = service._policy_snapshot(room)
    planned = discussion.plan_next_task(room, snapshot.events, local_profiles=service.local_profiles(),
        initial_watermarks=snapshot.watermarks, read_messages=snapshot.read_messages, freeze_input_context=True, limits=limits)
    replayed = discussion.plan_next_task(room,
        hosted_rooms.read_events(service.db_path, room_id="room")["events"],
        local_profiles=service.local_profiles(), freeze_input_context=True, limits=limits)
    assert planned == replayed
    return planned


@pytest.mark.parametrize("members, input_files, allowance", [("ab", 0, 10), ("abc", 17, 13)])
def test_repeating_file_outputs_pause_for_continue(tmp_path, monkeypatch, members, input_files, allowance):
    with roster(tmp_path, monkeypatch, members) as service:
        for start in range(0, input_files, 8):
            send(service, start, [upload(service, n) for n in range(start, min(start + 8, input_files))],
                 text="@all Review the files")
        if not input_files:
            send(service, "start", text="@all Work together")
        limits = discussion.DiscussionLimits(max_messages=10, max_rounds=3)
        heard = []
        for n in range(allowance):
            current = decision(service, limits)
            assert current.task is not None
            heard.append(current.task.member.member_id)
            _settle(service, current.task, "@all Here is my update", [upload(service, f"output-{n}")])
        paused = decision(service, limits)
        assert paused.status == "bounded" and paused.reason == "max_messages"
        assert heard[:len(members)] == list(members)
        service.prepare_room(service.bindings()[0])
        activity = hosted_rooms.read_events(service.db_path, room_id="room")["events"][-1]
        assert activity["kind"] == "room.activity" and activity["payload"]["status"] == "bounded"
        gateway, epoch = service._owned_authority("room")
        service._append_continue("room", paused.discussion_event_id, gateway, epoch)
        for n in range(limits.max_messages):
            current = decision(service, limits)
            assert current.task is not None
            _settle(service, current.task, "@all Another update", [upload(service, f"continued-{n}")])
        assert decision(service, limits).status == "bounded"
