"""Bounded discovery tests for canonical hosted-room attachments."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_attachments import (
    ATTACHMENT_LIST_EVENT_SCAN_LIMIT,
    AttachmentError,
    HostedRoomAttachmentStore,
    MAX_ATTACHMENT_LIST_RESPONSE_BYTES,
)


AUTHORITY = "gateway-a"
ROOM_ID = "room-1"


def _create_catalog(tmp_path):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db,
        room_id=ROOM_ID,
        name="Files",
        members=[
            {
                "member_id": "ops",
                "profile": "ops",
                "handle": "ops",
                "display_name": "Operations",
            },
            {
                "member_id": "qa",
                "profile": "qa",
                "handle": "qa",
                "display_name": "Quality",
            },
        ],
        authority_gateway_id=AUTHORITY,
        now=0.0,
    )
    return db, HostedRoomAttachmentStore(db)


def _entry(attachment_id: str, name: str) -> dict[str, object]:
    return {
        "attachment_id": attachment_id,
        "kind": "file",
        "name": name,
        "size": 1,
        "mime": "application/octet-stream",
    }


def _seed_events(
    db,
    *,
    total_events: int,
    records: Mapping[int, list[str]],
    producers: Mapping[int, str] | None = None,
    start_seq: int = 1,
) -> None:
    producers = producers or {}
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        for seq in range(start_seq, total_events + 1):
            names = records.get(seq, [])
            producer = producers.get(seq, "desktop")
            actor = (
                {"kind": "user", "id": "desktop"}
                if producer == "desktop"
                else {
                    "kind": "member",
                    "id": producer,
                    "profile": producer,
                    "display_name": "Operations" if producer == "ops" else "Quality",
                }
            )
            event_id = f"event-{seq}"
            manifest = []
            for slot, name in enumerate(names):
                attachment_id = f"att_{seq * 16 + slot:032x}"
                blob_id = f"blob_{seq * 16 + slot:032x}"
                digest = hashlib.sha256(attachment_id.encode()).hexdigest()
                manifest.append(_entry(attachment_id, name))
                conn.execute(
                    """INSERT INTO hosted_room_attachment_blobs
                       (blob_id, sha256, size, ref_count, created_at)
                       VALUES (?, ?, 1, 1, ?)""",
                    (blob_id, digest, float(total_events - seq)),
                )
                conn.execute(
                    """INSERT INTO hosted_room_attachments
                       (attachment_id, upload_id, room_id, event_id, kind, name,
                        size, mime, sha256, blob_id, recipient_member_ids_json,
                        viewer_access, state, created_at, updated_at, expires_at)
                       VALUES (?, ?, ?, ?, 'file', ?, 1,
                               'application/octet-stream', ?, ?, '["ops","qa"]',
                               1, 'committed', ?, ?, NULL)""",
                    (
                        attachment_id,
                        f"upload-{seq}-{slot}",
                        ROOM_ID,
                        event_id,
                        name,
                        digest,
                        blob_id,
                        float(total_events - seq),
                        float(total_events - seq),
                    ),
                )
            payload = {
                "text": "shared" if manifest else "ordinary message",
                "thread_id": f"thread-{seq}",
                **({"member_id": producer} if producer != "desktop" else {}),
                **({"attachments": manifest} if manifest else {}),
            }
            conn.execute(
                """INSERT INTO hosted_room_events
                   (room_id, seq, event_id, kind, actor_json, authority_epoch,
                    payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
                (
                    ROOM_ID,
                    seq,
                    event_id,
                    "message.user" if producer == "desktop" else "message.member",
                    json.dumps(actor, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    float(seq),
                ),
            )
        conn.execute(
            """UPDATE hosted_rooms
                  SET next_seq=MAX(next_seq, ?), updated_at=MAX(updated_at, ?)
                WHERE room_id=?""",
            (total_events + 1, float(total_events), ROOM_ID),
        )
        conn.commit()
    finally:
        conn.close()


def _page(store, **kwargs):
    return store.list_published(
        room_id=kwargs.pop("room_id", ROOM_ID),
        authority_gateway_id=kwargs.pop("authority_gateway_id", AUTHORITY),
        authority_epoch=kwargs.pop("authority_epoch", 1),
        **kwargs,
    )


def _all_items(store, **kwargs):
    items = []
    cursor = None
    snapshots = set()
    while True:
        page = _page(store, cursor=cursor, **kwargs)
        items.extend(page["items"])
        snapshots.add(page["snapshot_seq"])
        if not page["has_more"]:
            break
        cursor = page["next_cursor"]
        assert cursor
    assert len(snapshots) == 1
    return items


@pytest.mark.parametrize("count", [0, 1, 8, 9, 80, 2000])
def test_catalogue_pages_cover_bounded_record_counts(tmp_path, count):
    db, store = _create_catalog(tmp_path)
    _seed_events(
        db,
        total_events=count,
        records={seq: [f"file-{seq}.bin"] for seq in range(1, count + 1)},
    )

    items = _all_items(store)

    assert len(items) == count
    assert len({item["attachment_id"] for item in items}) == count
    assert [item["seq"] for item in items] == list(range(count, 0, -1))


def test_catalogue_orders_by_share_and_attachment_identity_and_searches_unicode(
    tmp_path,
):
    db, store = _create_catalog(tmp_path)
    _seed_events(
        db,
        total_events=4,
        records={
            1: ["Résumé.txt"],
            2: ["same.txt", "same.txt", "Résumé.txt"],
            3: ["STRASSE.md"],
            4: ["same.txt"],
        },
        producers={2: "qa", 3: "ops"},
    )

    items = _all_items(store, limit=32)

    assert [item["seq"] for item in items] == [4, 3, 2, 2, 2, 1]
    event_two = [item for item in items if item["seq"] == 2]
    assert [item["attachment_id"] for item in event_two] == sorted(
        item["attachment_id"] for item in event_two
    )
    assert [item["shared_at"] for item in items] == [4.0, 3.0, 2.0, 2.0, 2.0, 1.0]
    assert [item["name"] for item in _all_items(store, query="straße")] == [
        "STRASSE.md"
    ]
    assert [item["name"] for item in _all_items(store, query="RÉSUMÉ")] == [
        "Résumé.txt",
        "Résumé.txt",
    ]
    produced = _all_items(store, producer_member_id="ops")
    assert [item["name"] for item in produced] == ["STRASSE.md"]
    assert produced[0]["producer"] == {
        "kind": "member",
        "id": "ops",
        "label": "Operations",
    }


def test_catalogue_cursor_freezes_arrivals_and_rejects_request_mismatch(tmp_path):
    db, store = _create_catalog(tmp_path)
    _seed_events(
        db,
        total_events=9,
        records={seq: [f"file-{seq}.bin"] for seq in range(1, 10)},
    )
    first = _page(store)
    assert [item["seq"] for item in first["items"]] == list(range(9, 1, -1))

    _seed_events(
        db,
        total_events=10,
        records={10: ["new-arrival.bin"]},
        start_seq=10,
    )
    second = _page(store, cursor=first["next_cursor"])

    assert [item["seq"] for item in second["items"]] == [1]
    assert second["snapshot_seq"] == first["snapshot_seq"] == 9
    assert _page(store)["items"][0]["name"] == "new-arrival.bin"

    hosted_rooms.create_room(
        db,
        room_id="room-other",
        name="Other",
        members=[{"member_id": "ops", "profile": "ops", "handle": "ops"}],
        authority_gateway_id=AUTHORITY,
    )

    for changed in (
        {"query": "file"},
        {"producer_member_id": "ops"},
        {"room_id": "room-other"},
    ):
        with pytest.raises(AttachmentError, match="cursor"):
            _page(store, cursor=first["next_cursor"], **changed)
    cursor = first["next_cursor"]
    position = len(cursor) // 2
    replacement = "A" if cursor[position] != "A" else "B"
    tampered = cursor[:position] + replacement + cursor[position + 1 :]
    with pytest.raises(AttachmentError, match="cursor"):
        _page(store, cursor=tampered)


def test_cursor_pages_inside_one_eight_attachment_event_without_duplicates(tmp_path):
    db, store = _create_catalog(tmp_path)
    _seed_events(
        db,
        total_events=2,
        records={1: ["older.bin"], 2: ["same.bin"] * 8},
    )
    first = _page(store, limit=3)
    _seed_events(db, total_events=3, records={3: ["new.bin"]}, start_seq=3)
    second = _page(store, limit=3, cursor=first["next_cursor"])
    third = _page(store, limit=3, cursor=second["next_cursor"])
    items = first["items"] + second["items"] + third["items"]

    assert [item["seq"] for item in items] == [2] * 8 + [1]
    assert len({item["attachment_id"] for item in items}) == 9
    assert [item["attachment_id"] for item in items[:8]] == sorted(
        item["attachment_id"] for item in items[:8]
    )
    assert not third["has_more"]


def test_sparse_catalogue_progress_and_sql_work_are_bounded_without_blob_reads(
    tmp_path,
    monkeypatch,
):
    db, store = _create_catalog(tmp_path)
    _seed_events(db, total_events=2000, records={1: ["oldest.bin"]})
    monkeypatch.setattr(
        store,
        "_read_blob",
        lambda **_kwargs: pytest.fail("browse must not read canonical blob bytes"),
    )
    original_connect = sqlite3.connect
    progress_steps = []

    class MeasuredConnection(sqlite3.Connection):
        def set_progress_handler(self, callback, instructions):
            steps = [0]
            progress_steps.append(steps)

            def progress():
                steps[0] += instructions
                return callback()

            return super().set_progress_handler(progress, instructions)

    def measured_connect(*args, **kwargs):
        return original_connect(*args, **kwargs, factory=MeasuredConnection)

    monkeypatch.setattr(
        "gateway.hosted_room_attachment_catalog.sqlite3.connect", measured_connect
    )
    cursor = None
    pages = 0
    found = []
    while True:
        page = _page(store, cursor=cursor, query="oldest")
        pages += 1
        found.extend(page["items"])
        assert progress_steps[-1][0] < 50_000
        if not page["has_more"]:
            break
        cursor = page["next_cursor"]

    assert pages == 1
    assert [item["name"] for item in found] == ["oldest.bin"]


def test_catalogue_withholds_unpublished_expired_mismatched_and_corrupt_rows(
    tmp_path,
):
    db, store = _create_catalog(tmp_path)
    _seed_events(
        db,
        total_events=6,
        records={seq: [f"file-{seq}.bin"] for seq in range(1, 7)},
    )
    staged = store.put(
        room_id=ROOM_ID,
        upload_id="staged-upload",
        kind="file",
        name="staged.bin",
        mime="application/octet-stream",
        data=b"x",
    )
    unpublished = store.put(
        room_id=ROOM_ID,
        upload_id="unpublished-upload",
        kind="file",
        name="unpublished.bin",
        mime="application/octet-stream",
        data=b"x",
    )
    store.commit_message(
        room_id=ROOM_ID,
        event_id="never-published",
        manifest=[
            {
                key: unpublished[key]
                for key in ("attachment_id", "kind", "name", "size", "mime")
            }
        ],
        recipient_member_ids=("ops", "qa"),
        viewer_access=True,
    )
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE hosted_room_attachments SET expires_at=0 WHERE upload_id='upload-2-0'"
        )
        conn.execute(
            "UPDATE hosted_room_attachments SET event_id='wrong-event' WHERE upload_id='upload-3-0'"
        )
        payload = json.loads(
            conn.execute(
                "SELECT payload_json FROM hosted_room_events WHERE event_id='event-4'"
            ).fetchone()[0]
        )
        payload["attachments"][0]["name"] = "contradiction.bin"
        conn.execute(
            "UPDATE hosted_room_events SET payload_json=? WHERE event_id='event-4'",
            (json.dumps(payload, separators=(",", ":")),),
        )
        blob_id = conn.execute(
            "SELECT blob_id FROM hosted_room_attachments WHERE upload_id='upload-5-0'"
        ).fetchone()[0]
        conn.execute(
            "DELETE FROM hosted_room_attachment_blobs WHERE blob_id=?", (blob_id,)
        )
        conn.execute(
            "UPDATE hosted_room_attachments SET state='uploaded' WHERE upload_id='upload-6-0'"
        )
        conn.commit()
    finally:
        conn.close()

    items = _all_items(store, limit=32)

    assert [item["name"] for item in items] == ["file-1.bin"]
    assert staged["name"] not in {item["name"] for item in items}


@pytest.mark.parametrize(
    "corruption",
    [
        "actor",
        "payload",
        "manifest",
        "recipient-only",
        "recipients",
        "wrong-room",
        "missing-event",
        "blob-digest",
        "epoch",
    ],
)
def test_corrupt_or_inaccessible_metadata_does_not_become_discoverable(
    tmp_path, corruption
):
    db, store = _create_catalog(tmp_path)
    _seed_events(db, total_events=2, records={1: ["good.bin"], 2: ["bad.bin"]})
    with sqlite3.connect(db) as conn:
        if corruption == "actor":
            conn.execute("UPDATE hosted_room_events SET actor_json='{}' WHERE seq=2")
        elif corruption == "payload":
            conn.execute(
                "UPDATE hosted_room_events SET payload_json='not-json' WHERE seq=2"
            )
        elif corruption == "manifest":
            payload = json.loads(
                conn.execute(
                    "SELECT payload_json FROM hosted_room_events WHERE seq=2"
                ).fetchone()[0]
            )
            payload["attachments"][0]["extra"] = "not-canonical"
            conn.execute(
                "UPDATE hosted_room_events SET payload_json=? WHERE seq=2",
                (json.dumps(payload),),
            )
        elif corruption == "recipient-only":
            conn.execute(
                "UPDATE hosted_room_attachments SET viewer_access=0 WHERE event_id='event-2'"
            )
        elif corruption == "recipients":
            conn.execute(
                "UPDATE hosted_room_attachments SET recipient_member_ids_json='invalid' WHERE event_id='event-2'"
            )
        elif corruption == "wrong-room":
            conn.execute(
                "UPDATE hosted_room_attachments SET room_id='other-room' WHERE event_id='event-2'"
            )
        elif corruption == "missing-event":
            conn.execute("DELETE FROM hosted_room_events WHERE seq=2")
        elif corruption == "epoch":
            conn.execute(
                "UPDATE hosted_room_events SET authority_epoch='invalid' WHERE seq=2"
            )
        else:
            conn.execute(
                "UPDATE hosted_room_attachments SET sha256='invalid' WHERE event_id='event-2'"
            )

    assert [item["name"] for item in _all_items(store)] == ["good.bin"]


def test_maximum_unicode_names_keep_responses_and_cursors_bounded(tmp_path):
    db, store = _create_catalog(tmp_path)
    name = "\U00020000" * 255
    _seed_events(
        db,
        total_events=33,
        records={seq: [name] for seq in range(1, 34)},
    )

    first = _page(store, limit=32, query=name)
    second = _page(store, limit=32, query=name, cursor=first["next_cursor"])

    assert len(first["items"]) == 32
    assert len(second["items"]) == 1
    assert (
        len(json.dumps(first, ensure_ascii=False).encode())
        <= MAX_ATTACHMENT_LIST_RESPONSE_BYTES
    )


def test_catalogue_limit_and_query_inputs_are_bounded(tmp_path):
    _db, store = _create_catalog(tmp_path)

    for value in (0, 33, True, "8"):
        with pytest.raises(AttachmentError, match="limit"):
            _page(store, limit=value)
    with pytest.raises(AttachmentError, match="query"):
        _page(store, query="x" * 256)
    with pytest.raises(AttachmentError, match="query"):
        _page(store, query=" " * 256)
    for value in (False, 1, [], {}, "x" * 129):
        with pytest.raises(AttachmentError, match="producer_member_id"):
            _page(store, producer_member_id=value)


def test_same_event_staging_rows_cannot_bypass_the_metadata_scan_bound(
    tmp_path, monkeypatch
):
    db, store = _create_catalog(tmp_path)
    _seed_events(db, total_events=2, records={1: ["good.bin"], 2: ["staging.bin"]})
    with sqlite3.connect(db) as conn:
        conn.executemany(
            """INSERT INTO hosted_room_attachments
               (attachment_id, upload_id, room_id, event_id, kind, name, size,
                mime, sha256, blob_id, recipient_member_ids_json, viewer_access,
                state, created_at, updated_at, expires_at)
               SELECT ?, ?, room_id, event_id, kind, name, size, mime, sha256,
                      blob_id, recipient_member_ids_json, 0, 'uploaded',
                      created_at, updated_at, expires_at
                 FROM hosted_room_attachments WHERE upload_id='upload-2-0'""",
            ((f"att_{10000 + index:032x}", f"extra-{index}") for index in range(2000)),
        )
    original_connect = store._connect
    steps = [0]

    def measured_connect():
        conn = original_connect()

        def progress():
            steps[0] += 100
            return 0

        conn.set_progress_handler(progress, 100)
        return conn

    monkeypatch.setattr(store, "_connect", measured_connect)

    assert [item["name"] for item in _page(store)["items"]] == [
        "staging.bin",
        "good.bin",
    ]
    assert steps[0] < 5000
