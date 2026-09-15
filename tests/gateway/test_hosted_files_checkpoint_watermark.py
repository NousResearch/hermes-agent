"""Files watermark cases extracted from I691, not its unrelated upgrade tests.

Source: 691bb08a5cd310fffc5f3d01653dc93f394fc080,
tests/gateway/test_hosted_room_checkpoint_consistency.py (helpers and two cases).
"""
import sqlite3

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint


def _append(db, index):
    return hosted_rooms.append_event(
        db,
        room_id="checkpoint",
        event_id=f"user-{index}",
        kind="message.user",
        actor={"kind": "user", "id": "owner"},
        authority_gateway_id="home",
        authority_epoch=1,
        payload={"text": f"request {index}", "thread_id": "work"},
    )



def _checkpoint(tmp_path):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db,
        room_id="checkpoint",
        name="Checkpoint",
        members=[],
        authority_gateway_id="home",
    )
    _append(db, 1)
    _append(db, 2)
    checkpoint = HostedRoomPolicyCheckpoint(db)
    checkpoint.sync(room_id="checkpoint", latest_seq=2)
    return db, checkpoint



def _late_settlement(db, checkpoint, seen_through_seq):
    payload = {
        "thread_id": "work",
        "discussion_event_id": "user-2",
        "member_id": "builder",
        "task_id": "dtask:late",
    }
    events = []
    for event_id, kind, fields in (
        ("cleanup", "room.activity", {"status": "settled"}),
        ("dmessage:late", "message.member", {"text": "Reviewed"}),
        (
            "dterminal:late",
            "turn.settled",
            {"message_event_id": "dmessage:late", "seen_through_seq": seen_through_seq},
        ),
    ):
        event = hosted_rooms.append_event(
            db,
            room_id="checkpoint",
            event_id=event_id,
            kind=kind,
            actor=(
                {"kind": "member", "id": "builder"}
                if kind == "message.member"
                else {"kind": "gateway", "id": "home"}
            ),
            authority_gateway_id="home",
            authority_epoch=1,
            payload={**payload, **fields},
        )
        checkpoint.sync(room_id="checkpoint", latest_seq=event["seq"])
        events.append(event)
    return events[-2:]



@pytest.mark.parametrize("seen_through_seq", [1, 2])
def test_late_settlement_keeps_partial_file_watermark_after_cleanup(tmp_path, seen_through_seq):
    db, checkpoint = _checkpoint(tmp_path)
    message, terminal = _late_settlement(db, checkpoint, seen_through_seq)
    cold = HostedRoomPolicyCheckpoint(db)
    assert cold.snapshot(room_id="checkpoint", latest_seq=terminal["seq"]).events == ()
    assert cold.publication_exists(
        room_id="checkpoint", task_id="dtask:late", status="settled", execution_generation=1
    )
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT seq FROM hosted_room_policy_events WHERE discussion_event_id='user-2'"
        ).fetchall() == []
        assert conn.execute(
            "SELECT seen_through_seq FROM hosted_room_policy_watermarks WHERE member_id='builder'"
        ).fetchone()[0] == (1 if seen_through_seq == 1 else message["seq"])
        assert conn.execute(
            "SELECT settled_seq FROM hosted_room_policy_transcript WHERE seq=?", (message["seq"],)
        ).fetchone()[0] == terminal["seq"]
