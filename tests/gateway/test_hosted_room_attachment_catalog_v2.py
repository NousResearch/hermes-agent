"""Product-level catalog invariants beyond the initial bounded event scan."""

import sqlite3
import time

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_attachments import (
    AttachmentCursorError,
    AttachmentError,
    AttachmentQuotaError,
    HostedRoomAttachmentStore,
)
from tests.gateway.test_hosted_room_attachment_catalog import (
    AUTHORITY,
    ROOM_ID,
    _create_catalog,
    _seed_events,
)


def share(db, store, *, upload_id="one", name="report.md"):
    data = f"Shared {name}\n".encode()
    item = store.put(
        room_id=ROOM_ID,
        upload_id=upload_id,
        kind="file",
        name=name,
        mime="text/plain",
        data=data,
    )
    event_id = f"share-{upload_id}"
    manifest = [
        {key: item[key] for key in ("attachment_id", "kind", "name", "mime", "size")}
    ]
    store.commit_message(
        room_id=ROOM_ID,
        event_id=event_id,
        manifest=manifest,
        recipient_member_ids=["ops", "qa"],
        viewer_access=True,
        hold_until_event=True,
    )
    hosted_rooms.append_event(
        db,
        room_id=ROOM_ID,
        event_id=event_id,
        kind="message.member",
        actor={
            "kind": "member",
            "id": "ops",
            "profile": "ops",
            "display_name": "Operations",
        },
        authority_gateway_id=AUTHORITY,
        authority_epoch=1,
        payload={
            "text": "Shared a report",
            "member_id": "ops",
            "thread_id": "thread",
            "attachments": manifest,
        },
    )
    return item, event_id, data


def listing(store, **kwargs):
    return store.list_published(
        room_id=ROOM_ID, authority_gateway_id=AUTHORITY, authority_epoch=1, **kwargs
    )


def test_browsing_does_not_create_a_roomlink_secret(tmp_path):
    db, store = _create_catalog(tmp_path)
    for index in range(9):
        share(db, store, upload_id=str(index))
    secret = tmp_path / ".room-link-grant-secret"
    assert not secret.exists()
    first = listing(store)
    assert first["next_cursor"]
    second = listing(store, cursor=first["next_cursor"])
    assert len(first["items"]) == 8 and len(second["items"]) == 1
    assert not secret.exists()


def test_legacy_epochless_share_is_listed_when_the_chip_can_read_it(tmp_path):
    db, store = _create_catalog(tmp_path)
    item, event_id, data = share(db, store)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE hosted_room_events SET authority_epoch=NULL WHERE room_id=? AND event_id=?",
            (ROOM_ID, event_id),
        )
    assert (
        store.read(
            room_id=ROOM_ID,
            attachment_id=item["attachment_id"],
            event_id=event_id,
            recipient_member_id=None,
            viewer=True,
        ).data
        == data
    )
    assert [entry["attachment_id"] for entry in listing(store)["items"]] == [
        item["attachment_id"]
    ]


def test_plain_text_does_not_advance_the_latest_file_sequence(tmp_path):
    db, store = _create_catalog(tmp_path)
    share(db, store)
    first = listing(store)
    hosted_rooms.append_event(
        db,
        room_id=ROOM_ID,
        event_id="ordinary-text",
        kind="message.member",
        actor={"kind": "member", "id": "qa", "profile": "qa"},
        authority_gateway_id=AUTHORITY,
        authority_epoch=1,
        payload={"text": "Still working", "member_id": "qa", "thread_id": "thread"},
    )
    second = listing(store)
    assert second["snapshot_seq"] > first["snapshot_seq"]
    assert second["latest_seq"] == first["latest_seq"] == first["items"][0]["seq"]


@pytest.mark.parametrize("query", ["cafe", "CAFE\u0301", "operations", "OPERATIONS"])
def test_search_matches_accents_case_and_sharer(tmp_path, query):
    db, store = _create_catalog(tmp_path)
    item, _event, _data = share(db, store, name="Caf\u00e9 plan.md")
    assert [row["attachment_id"] for row in listing(store, query=query)["items"]] == [
        item["attachment_id"]
    ]


def test_manifest_order_is_retained_across_page_boundary(tmp_path):
    db, store = _create_catalog(tmp_path)
    items = [
        store.put(
            room_id=ROOM_ID,
            upload_id=f"batch-{index}",
            kind="file",
            name=f"file-{index}.txt",
            mime="text/plain",
            data=b"file",
        )
        for index in range(2)
    ]
    manifest = [
        {key: item[key] for key in ("attachment_id", "kind", "name", "mime", "size")}
        for item in sorted(items, key=lambda item: item["attachment_id"], reverse=True)
    ]
    store.commit_message(
        room_id=ROOM_ID,
        event_id="batch",
        manifest=manifest,
        recipient_member_ids=["ops", "qa"],
        viewer_access=True,
        hold_until_event=True,
    )
    hosted_rooms.append_event(
        db,
        room_id=ROOM_ID,
        event_id="batch",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        authority_gateway_id=AUTHORITY,
        authority_epoch=1,
        payload={"text": "Two files", "attachments": manifest},
    )
    first = listing(store, limit=1)
    second = listing(store, limit=1, cursor=first["next_cursor"])
    assert [row["attachment_id"] for row in first["items"] + second["items"]] == [
        item["attachment_id"] for item in manifest
    ]
    assert first["items"][0]["manifest_index"] == 0
    assert second["items"][0]["manifest_index"] == 1


def test_listing_does_not_need_a_sqlite_write_reservation(tmp_path, monkeypatch):
    db, store = _create_catalog(tmp_path)
    share(db, store)
    monkeypatch.setattr(
        store,
        "_connect",
        lambda: pytest.fail("Listing called the schema-writing connection"),
    )
    with sqlite3.connect(db) as writer:
        writer.execute("BEGIN IMMEDIATE")
        assert len(listing(store)["items"]) == 1


@pytest.mark.parametrize("count", [1000, 10000, 50000])
def test_rare_search_is_one_page_at_scale(tmp_path, count, record_property):
    db, store = _create_catalog(tmp_path)
    records = {index: [f"note-{index}.md"] for index in range(1, count + 1)}
    records[1] = ["needle.md"]
    _seed_events(db, total_events=count, records=records)
    # The bulk fixture represents publication metadata, not physical blob I/O.
    # Ordinary put() already writes this folded column for each new upload.
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_attachments SET catalog_name=lower(name)")
    started = time.perf_counter()
    page = listing(store, query="needle")
    elapsed = time.perf_counter() - started
    record_property("catalog_records", count)
    record_property("catalog_search_seconds", elapsed)
    assert [row["name"] for row in page["items"]] == ["needle.md"]
    assert not page["has_more"]
    assert elapsed < 2.0


def test_count_caps_bound_tiny_deduplicated_files_without_breaking_retry(tmp_path):
    store = HostedRoomAttachmentStore(
        tmp_path / "quota.db",
        room_quota_count=1,
        gateway_quota_count=2,
    )

    def upload(room, identity):
        return store.put(
            room_id=room,
            upload_id=identity,
            kind="file",
            name="small.txt",
            mime="text/plain",
            data=b"x",
        )

    first = upload("one", "first")
    assert upload("one", "first")["attachment_id"] == first["attachment_id"]
    with pytest.raises(AttachmentQuotaError, match="room attachment count"):
        upload("one", "second")
    upload("two", "second")
    with pytest.raises(AttachmentQuotaError, match="gateway attachment count"):
        upload("three", "third")


def test_recipient_filter_finds_old_eligible_files_without_empty_pages(tmp_path):
    db, store = _create_catalog(tmp_path)
    _seed_events(
        db,
        total_events=300,
        records={index: [f"file-{index}.md"] for index in range(1, 301)},
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE hosted_room_attachments SET recipient_member_ids_json='[\"qa\"]' WHERE event_id!='event-1'"
        )
    scoped = listing(store, recipient_member_id="ops")
    assert [row["name"] for row in scoped["items"]] == ["file-1.md"]
    assert not scoped["has_more"]
    assert scoped["latest_seq"] == 1
    root_page = listing(store)
    with pytest.raises(AttachmentCursorError, match="does not match"):
        listing(store, recipient_member_id="qa", cursor=root_page["next_cursor"])


def test_manifest_projection_does_not_decode_ordinary_message_text(
    tmp_path, monkeypatch
):
    import json

    db, store = _create_catalog(tmp_path)
    share(db, store)
    marker = "TEXT_NOT_NEEDED_FOR_FILES"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE hosted_room_events SET payload_json=json_set(payload_json,'$.text',?)",
            (marker * 1000,),
        )
    original_loads = json.loads

    def decode_metadata(value, *args, **kwargs):
        assert marker not in str(value)
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(
        "gateway.hosted_room_attachment_catalog.json.loads", decode_metadata
    )
    assert len(listing(store)["items"]) == 1


@pytest.mark.parametrize("character", ["\ud55c", "\ufdfa"])
def test_normalization_expansion_does_not_overflow_a_valid_cursor(tmp_path, character):
    db, store = _create_catalog(tmp_path)
    name = character * 255
    for index in range(9):
        share(db, store, upload_id=f"unicode-{index}", name=name)
    first = listing(store, query=name)
    assert len(first["items"]) == 8
    assert len(first["next_cursor"]) <= 4096
    second = listing(store, query=name, cursor=first["next_cursor"])
    assert len(second["items"]) == 1


def test_unpaired_surrogates_fail_before_blob_writes_or_querying(tmp_path):
    _db, store = _create_catalog(tmp_path)
    with pytest.raises(AttachmentError, match="valid Unicode"):
        store.put(
            room_id=ROOM_ID,
            upload_id="invalid",
            kind="file",
            name="\ud800.txt",
            mime="text/plain",
            data=b"private",
        )
    assert list(store.blob_root.iterdir()) == []
    with pytest.raises(AttachmentError, match="valid Unicode"):
        listing(store, query="\ud800")
