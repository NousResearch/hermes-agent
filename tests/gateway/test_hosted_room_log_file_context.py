"""Admitted file backlogs use the log without overflowing policy or task caches."""
import sqlite3

import pytest

from gateway import hosted_room_discussion as discussion, hosted_rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
from tests.gateway.test_hosted_discussion_file_context import _settle, roster, send, upload


def full_log(service):
    events = []
    cursor = 0
    while True:
        page = hosted_rooms.read_events(service.db_path, room_id="room", since_seq=cursor,
                                        limit=hosted_rooms.MAX_LOG_LIMIT)
        events.extend(page["events"])
        if not page["has_more"]:
            return events
        cursor = page["cursor"]


def decision(service, recovery):
    if isinstance(recovery, int):
        with sqlite3.connect(service.db_path) as conn:
            conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=?", (recovery,))
    if recovery != "live":
        service.policy_checkpoint = HostedRoomPolicyCheckpoint(service.db_path)
    room = service._room("room")
    snapshot = service._policy_snapshot(room)
    planned = discussion.plan_next_task(room, snapshot.events, local_profiles=service.local_profiles(),
        initial_watermarks=snapshot.watermarks, read_messages=getattr(snapshot, "read_messages", None),
        freeze_input_context=True)
    replayed = discussion.plan_next_task(room, full_log(service), local_profiles=service.local_profiles(),
                                         freeze_input_context=True)
    assert planned == replayed
    if planned.task:
        payload = planned.task.payload
        assert len(payload["input_context"]["event_seqs"]) <= 128
        assert len(payload.get("attachments", [])) <= 16
    return planned


@pytest.mark.parametrize("recovery", ["live", "reopen", 4, 5])
@pytest.mark.parametrize("count", [17, 64, 320])
def test_six_bots_finish_then_accept_thanks(tmp_path, monkeypatch, recovery, count):
    with roster(tmp_path, monkeypatch, "abcdef") as service:
        inputs = set()
        for start in range(0, count, 8):
            files = [upload(service, n) for n in range(start, min(start + 8, count))]
            inputs.update(item["attachment_id"] for item in files)
            send(service, start, files, text="@all Review")
        received = {member: set() for member in "abcdef"}
        for n in range(6 * ((count + 15) // 16)):
            task = decision(service, recovery).task
            assert task is not None, f"stalled after {n} replies: {received}"
            received[task.member.member_id].update(a["attachment_id"] for a in task.payload["attachments"])
            _settle(service, task, "Reviewed.")
        assert received == dict.fromkeys("abcdef", inputs)
        settled = decision(service, recovery)
        assert settled.status == "settled"
        service._append_room_status(service._room("room"), settled)
        send(service, "thanks", text="Thanks")
        assert decision(service, recovery).task is not None
        with sqlite3.connect(service.db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM hosted_room_policy_transcript").fetchone()[0] <= 24
            # Each Send is an anchor; catch-up replies add no cached policy rows.
            assert conn.execute("SELECT COUNT(*) FROM hosted_room_policy_events").fetchone()[0] <= (count + 7) // 8 + 1


@pytest.mark.parametrize("recovery", ["live", "reopen", 4, 5])
def test_late_bot_reads_file_after_63_completed_pairs(tmp_path, monkeypatch, recovery):
    with roster(tmp_path, monkeypatch, "ab") as service:
        file = upload(service, "original")
        send(service, "original", [file], text="@a Review")
        _settle(service, decision(service, recovery).task, "Reviewed.")
        for n in range(63):
            send(service, n, text="@a Continue")
            _settle(service, decision(service, recovery).task, "Done.")
        send(service, "late", text="@b Read the file")
        task = decision(service, recovery).task
        assert task.member.member_id == "b"
        assert [a["attachment_id"] for a in task.payload["attachments"]] == [file["attachment_id"]]
