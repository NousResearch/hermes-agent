"""Read discussion receipts and unread input at a fixed position in the room log."""

from functools import partial

from .._backend import translated
from .hosted_rooms import _EVENT_COLUMNS, _event_from_row


def discussion_events(conn, *, room_id, discussion_event_id, kinds):
    anchor = conn.execute(
        "SELECT seq FROM hosted_room_policy_events WHERE room_id=? AND discussion_event_id=? LIMIT 1",
        (room_id, discussion_event_id),
    ).fetchone()
    if anchor is None:
        return []
    return [
        _event_from_row(row)
        for row in conn.execute(
            f"""SELECT {_EVENT_COLUMNS} FROM hosted_room_events
            WHERE room_id=? AND seq>=? AND seq<=(
                SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id=?)
              AND (seq=? OR (kind IN ({",".join("?" * len(kinds))})
                   AND json_extract(payload_json, '$.discussion_event_id')=?))
            ORDER BY seq""",
            (
                room_id,
                anchor["seq"],
                room_id,
                anchor["seq"],
                *kinds,
                discussion_event_id,
            ),
        )
    ]


def _message_page(conn, *, room_id, through_seq, thread_id, after_seq):
    rows = conn.execute(
        f"""SELECT {_EVENT_COLUMNS} FROM hosted_room_events
            WHERE room_id=? AND seq>? AND seq<=? ORDER BY seq LIMIT 64""",
        (room_id, after_seq, through_seq),
    ).fetchall()
    messages = []
    for row in rows:
        if row["kind"] not in {"message.user", "message.member"}:
            continue
        event = _event_from_row(row)
        payload = event["payload"]
        if payload.get("thread_id") != thread_id:
            continue
        if row["kind"] == "message.member":
            committed = conn.execute(
                """SELECT 1 FROM hosted_room_policy_publications AS publication
                CROSS JOIN hosted_room_events AS receipt
                  ON receipt.room_id=publication.room_id AND receipt.seq=publication.seq
                WHERE publication.room_id=? AND publication.task_id=? AND publication.kind='turn.settled'
                  AND publication.seq<=? AND json_extract(receipt.payload_json, '$.message_event_id')=?""",
                (room_id, payload.get("task_id"), through_seq, event["event_id"]),
            ).fetchone()
            if committed is None:
                continue
        messages.append(event)
    return (int(rows[-1]["seq"]) if rows else through_seq), messages


def read_thread_messages(
    files, placement, *, room_id, through_seq, thread_id, watermark
):
    while watermark < through_seq:
        with translated():
            watermark, messages = files._hosted_read(
                placement,
                partial(
                    _message_page,
                    room_id=room_id,
                    through_seq=through_seq,
                    thread_id=thread_id,
                    after_seq=watermark,
                ),
            )
        yield from messages
