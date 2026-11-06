"""Read policy receipts and unread input from the indexed, append-only group chat log."""

from collections.abc import Iterator
from contextlib import closing
from typing import Any

from gateway import hosted_rooms


def _pages(conn, *, room_id: str, after_seq: int, through_seq: int) -> Iterator[Any]:
    # The existing (room_id, seq) index bounds each read, including sparse threads.
    while after_seq < through_seq:
        rows = conn.execute(
            f"""SELECT {hosted_rooms._EVENT_COLUMNS} FROM hosted_room_events
                WHERE room_id=? AND seq>? AND seq<=? ORDER BY seq LIMIT ?""",
            (room_id, after_seq, through_seq, 64)).fetchall()
        if not rows:
            return
        yield from rows
        after_seq = int(rows[-1]["seq"])


def discussion_events(conn, *, room_id: str, discussion_event_id: str) -> Iterator[dict[str, Any]]:
    source = conn.execute(
        f"SELECT {hosted_rooms._EVENT_COLUMNS} FROM hosted_room_events WHERE room_id=? AND event_id=?",
        (room_id, discussion_event_id)).fetchone()
    if source is None:
        return
    yield hosted_rooms._event_from_row(source)
    through_seq = conn.execute(
        "SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id=?", (room_id,)).fetchone()[0]
    for row in _pages(conn, room_id=room_id, after_seq=int(source["seq"]), through_seq=through_seq):
        event = hosted_rooms._event_from_row(row)
        if event["payload"].get("discussion_event_id") == discussion_event_id:
            yield event


def read_thread_messages(db_path, *, room_id: str, through_seq: int,
                         thread_id: str, watermark: int) -> Iterator[dict[str, Any]]:
    from hermes_cli.sqlite_util import open_db

    with closing(open_db(db_path, db_label="shared-state.db (group chat input)")) as conn:
        for row in _pages(conn, room_id=room_id, after_seq=watermark, through_seq=through_seq):
            if row["kind"] not in {"message.user", "message.member"}:
                continue
            event = hosted_rooms._event_from_row(row)
            if event["payload"].get("thread_id") != thread_id:
                continue
            if row["kind"] == "message.member":
                # A visible message is input only after its terminal receipt commits it.
                # Fix the join order so lookup starts with one task, not the log prefix.
                committed = conn.execute("""SELECT 1 FROM hosted_room_policy_publications AS publication
                    CROSS JOIN hosted_room_events AS receipt
                      ON receipt.room_id=publication.room_id AND receipt.seq=publication.seq
                    WHERE publication.room_id=? AND publication.task_id=? AND publication.kind='turn.settled'
                      AND publication.seq<=? AND json_extract(receipt.payload_json, '$.message_event_id')=?""",
                    (room_id, event["payload"]["task_id"], through_seq, event["event_id"])).fetchone()
                if committed is None:
                    continue
            yield event
