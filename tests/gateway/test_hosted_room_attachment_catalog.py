"""Files catalog over the canonical attachment store: order, search, cursors and eligibility."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping

import pytest

from gateway import hosted_room_attachment_catalog as catalog
from gateway import hosted_rooms
from gateway.hosted_room_attachments import (
    AttachmentError,
    AttachmentNotFoundError,
    HostedRoomAttachmentStore,
)

AUTHORITY = "gateway-a"
ROOM_ID = "room-1"


def _create_catalog(tmp_path):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db, room_id=ROOM_ID, name="Files", authority_gateway_id=AUTHORITY, now=0.0,
        members=[
            {"member_id": "ops", "profile": "ops", "handle": "ops", "display_name": "Operations"},
            {"member_id": "qa", "profile": "qa", "handle": "qa", "display_name": "Quality"},
        ])
    return db, HostedRoomAttachmentStore(db)


def _seed_events(db, *, total_events: int, records: Mapping[int, list[str]],
                 producers: Mapping[int, str] | None = None, start_seq: int = 1) -> None:
    """Bulk-publish metadata rows the way a committed Send leaves them (no blob bytes)."""
    producers = producers or {}
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        for seq in range(start_seq, total_events + 1):
            producer = producers.get(seq, "desktop")
            actor = ({"kind": "user", "id": "desktop"} if producer == "desktop" else
                     {"kind": "member", "id": producer, "profile": producer,
                      "display_name": "Operations" if producer == "ops" else "Quality"})
            manifest = []
            for slot, name in enumerate(records.get(seq, [])):
                attachment_id = f"att_{seq * 16 + slot:032x}"
                blob_id = f"blob_{seq * 16 + slot:032x}"
                digest = hashlib.sha256(attachment_id.encode()).hexdigest()
                manifest.append({"attachment_id": attachment_id, "kind": "file", "name": name, "size": 1,
                                 "mime": "application/octet-stream"})
                conn.execute("INSERT INTO hosted_room_attachment_blobs (blob_id, sha256, size, ref_count,"
                             " created_at) VALUES (?, ?, 1, 1, ?)", (blob_id, digest, float(seq)))
                conn.execute(
                    """INSERT INTO hosted_room_attachments
                       (attachment_id, upload_id, room_id, event_id, kind, name, size, mime, sha256, blob_id,
                        recipient_member_ids_json, viewer_access, state, created_at, updated_at, expires_at)
                       VALUES (?, ?, ?, ?, 'file', ?, 1, 'application/octet-stream', ?, ?, '["ops","qa"]',
                               1, 'committed', ?, ?, NULL)""",
                    (attachment_id, f"upload-{seq}-{slot}", ROOM_ID, f"event-{seq}", name, digest, blob_id,
                     float(seq), float(seq)))
            payload = {"text": "shared" if manifest else "ordinary message", "thread_id": f"thread-{seq}",
                       **({"member_id": producer} if producer != "desktop" else {}),
                       **({"attachments": manifest} if manifest else {})}
            conn.execute(
                "INSERT INTO hosted_room_events (room_id, seq, event_id, kind, actor_json, authority_epoch,"
                " payload_json, created_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                (ROOM_ID, seq, f"event-{seq}", "message.user" if producer == "desktop" else "message.member",
                 json.dumps(actor, ensure_ascii=False, separators=(",", ":")),
                 json.dumps(payload, ensure_ascii=False, separators=(",", ":")), float(seq)))
        conn.execute("UPDATE hosted_rooms SET next_seq=MAX(next_seq, ?) WHERE room_id=?",
                     (total_events + 1, ROOM_ID))
        conn.commit()
    finally:
        conn.close()


def share(db, store, *, upload_id="one", name="report.md", member="ops"):
    """A real upload, commitment and published message, as Send performs them."""
    data = f"Shared {name} {upload_id}\n".encode()
    item = store.put(room_id=ROOM_ID, upload_id=upload_id, kind="file", name=name, mime="text/plain", data=data)
    manifest = [{key: item[key] for key in ("attachment_id", "kind", "name", "mime", "size")}]
    event_id = f"share-{upload_id}"
    store.commit_message(room_id=ROOM_ID, event_id=event_id, manifest=manifest,
                         recipient_member_ids=["ops", "qa"], viewer_access=True, hold_until_event=True)
    hosted_rooms.append_event(
        db, room_id=ROOM_ID, event_id=event_id, kind="message.member",
        actor={"kind": "member", "id": member, "profile": member,
               "display_name": "Operations" if member == "ops" else "Quality"},
        authority_gateway_id=AUTHORITY, authority_epoch=1,
        payload={"text": "Shared a report", "member_id": member, "thread_id": "thread", "attachments": manifest})
    return item, event_id, data


def _page(db, **kwargs):
    return catalog.list_published(
        db, room_id=kwargs.pop("room_id", ROOM_ID), authority_gateway_id=kwargs.pop("authority_gateway_id", AUTHORITY),
        authority_epoch=kwargs.pop("authority_epoch", 1), **kwargs)


def _all_items(db, pages=None, **kwargs):
    """Every item of a listing, paged through one snapshot; ``pages`` collects the pages."""
    items, cursor, snapshots = [], None, set()
    while True:
        page = _page(db, cursor=cursor, **kwargs)
        items.extend(page["items"])
        snapshots.add(page["snapshot_seq"])
        if pages is not None:
            pages.append(page)
        if not page["has_more"]:
            assert page["next_cursor"] is None
            break
        cursor = page["next_cursor"]
        assert cursor
    assert len(snapshots) == 1
    return items


@pytest.mark.parametrize("count", [0, 1, 8, 9, 80, 2000])
def test_pages_cover_every_file_newest_first(tmp_path, count):
    db, _store = _create_catalog(tmp_path)
    _seed_events(db, total_events=count, records={seq: [f"file-{seq}.bin"] for seq in range(1, count + 1)})
    pages = []

    items = _all_items(db, pages)

    assert [item["seq"] for item in items] == list(range(count, 0, -1))
    assert len({item["attachment_id"] for item in items}) == count
    assert [len(page["items"]) for page in pages] == [8] * (count // 8) + ([count % 8] if count % 8 or not count else [])


def test_order_unicode_search_and_sharer_filter(tmp_path):
    db, _store = _create_catalog(tmp_path)
    _seed_events(db, total_events=4,
                 records={1: ["Résumé.txt"], 2: ["same.txt", "same.txt", "Résumé.txt"], 3: ["STRASSE.md"],
                          4: ["same.txt"]},
                 producers={2: "qa", 3: "ops"})

    items = _all_items(db, limit=32)

    assert [(item["seq"], item["manifest_index"]) for item in items] == [(4, 0), (3, 0), (2, 0), (2, 1), (2, 2), (1, 0)]
    assert [item["shared_at"] for item in items] == [4.0, 3.0, 2.0, 2.0, 2.0, 1.0]
    same = [item for item in items if item["name"] == "same.txt"]
    assert len({(item["event_id"], item["attachment_id"]) for item in same}) == 3
    assert [item["name"] for item in _all_items(db, query="straße")] == ["STRASSE.md"]
    assert [item["name"] for item in _all_items(db, query="RÉSUMÉ")] == ["Résumé.txt", "Résumé.txt"]
    produced = _all_items(db, producer_member_id="ops")
    assert [item["name"] for item in produced] == ["STRASSE.md"]
    assert produced[0]["producer"] == {"kind": "member", "id": "ops", "label": "Operations"}
    assert [item["seq"] for item in _all_items(db, query="quality")] == [2, 2, 2]
    assert [item["seq"] for item in _all_items(db, query="you")] == [4, 1]


def test_a_search_for_the_readers_word_for_you_finds_their_own_shares(tmp_path):
    db, _store = _create_catalog(tmp_path)
    _seed_events(db, total_events=3, records={1: ["notes.txt"], 2: ["rendez-vous.txt"], 3: ["plan.md"]},
                 producers={2: "ops"})
    assert [item["seq"] for item in _all_items(db, query="Vous", you_label="Vous")] == [3, 2, 1]
    assert [item["seq"] for item in _all_items(db, query="vous")] == [2]
    assert [item["seq"] for item in _all_items(db, query="あなた", you_label="あなた")] == [3, 1]
    assert [item["seq"] for item in _all_items(db, query="you", you_label="Vous")] == [3, 1]
    assert [item["seq"] for item in _all_items(db, query="operations", you_label="Vous")] == [2]
    # The word is part of a search's cursor scope: a continuation under another word is refused.
    first = _page(db, query="v", you_label="Vous", limit=1)
    assert first["has_more"]
    with pytest.raises(catalog.CatalogCursorError):
        _page(db, query="v", you_label="Tú", limit=1, cursor=first["next_cursor"])
    assert _page(db, query="v", you_label="vous ", limit=1, cursor=first["next_cursor"])["items"]
    # Without a query the word changes nothing, so a listing's cursor ignores it.
    plain = _page(db, limit=1, you_label="Vous")
    assert _page(db, limit=1, cursor=plain["next_cursor"])["items"][0]["seq"] == 2
    for bad in ("x" * 65, 7):
        with pytest.raises(AttachmentError):
            _page(db, query="x", you_label=bad)


def test_cursor_freezes_new_arrivals_and_is_refused_for_another_listing(tmp_path):
    db, _store = _create_catalog(tmp_path)
    _seed_events(db, total_events=9, records={seq: [f"file-{seq}.bin"] for seq in range(1, 10)})
    first = _page(db)
    assert [item["seq"] for item in first["items"]] == list(range(9, 1, -1))

    _seed_events(db, total_events=10, records={10: ["new-arrival.bin"]}, start_seq=10)
    second = _page(db, cursor=first["next_cursor"])

    assert [item["seq"] for item in second["items"]] == [1]
    assert second["snapshot_seq"] == first["snapshot_seq"] == 9
    assert _page(db)["items"][0]["name"] == "new-arrival.bin"
    hosted_rooms.create_room(db, room_id="room-other", name="Other", authority_gateway_id=AUTHORITY,
                             members=[{"member_id": "ops", "profile": "ops", "handle": "ops"}])
    for changed in ({"query": "file"}, {"producer_member_id": "ops"}, {"room_id": "room-other"}):
        with pytest.raises(catalog.CatalogCursorError, match="cursor"):
            _page(db, cursor=first["next_cursor"], **changed)
    for malformed in ("", "not base64!", "e30", first["next_cursor"][:-3], "A" * 4097):
        with pytest.raises(catalog.CatalogCursorError, match="cursor"):
            _page(db, cursor=malformed)
    assert catalog.CatalogCursorError.reason == "attachment_cursor_invalid"


def test_pages_inside_one_message_keep_its_manifest_order_without_duplicates(tmp_path):
    db, store = _create_catalog(tmp_path)
    older, _event_id, _data = share(db, store, upload_id="older", name="older.bin")
    items = [store.put(room_id=ROOM_ID, upload_id=f"batch-{index}", kind="file", name=f"file-{index}.txt",
                       mime="text/plain", data=f"file {index}".encode()) for index in range(8)]
    manifest = [{key: item[key] for key in ("attachment_id", "kind", "name", "mime", "size")}
                for item in sorted(items, key=lambda item: item["attachment_id"], reverse=True)]
    store.commit_message(room_id=ROOM_ID, event_id="batch", manifest=manifest, recipient_member_ids=["ops", "qa"],
                         viewer_access=True, hold_until_event=True)
    hosted_rooms.append_event(db, room_id=ROOM_ID, event_id="batch", kind="message.user",
                              actor={"kind": "user", "id": "desktop"}, authority_gateway_id=AUTHORITY,
                              authority_epoch=1, payload={"text": "Eight files", "attachments": manifest})
    first = _page(db, limit=3)
    share(db, store, upload_id="new", name="new.bin")  # after the snapshot the pages continue
    second = _page(db, limit=3, cursor=first["next_cursor"])
    third = _page(db, limit=3, cursor=second["next_cursor"])
    rows = first["items"] + second["items"] + third["items"]

    assert [row["attachment_id"] for row in rows] == [
        *(item["attachment_id"] for item in manifest), older["attachment_id"]]
    assert [row["manifest_index"] for row in rows[:8]] == list(range(8))
    assert not third["has_more"]


@pytest.fixture
def query_work(monkeypatch):
    original, work = catalog._snapshot, []
    def counted(path):
        conn = original(path)
        index = len(work)
        work.append(0)
        def progress():
            work[index] += 100
            return 0
        conn.set_progress_handler(progress, 100)
        return conn
    monkeypatch.setattr(catalog, "_snapshot", counted)
    return work


def test_rare_search_and_sparse_history_stay_paged_without_blob_reads(tmp_path, monkeypatch, query_work):
    db, _store = _create_catalog(tmp_path)
    records = {index: [f"note-{index}.md"] for index in range(1, 10001)}
    records[1] = ["needle.md"]
    _seed_events(db, total_events=10000, records=records)
    _seed_events(db, total_events=12000, records={}, start_seq=10001)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_events SET payload_json=json_set(payload_json, '$.attachments', json('[]')) WHERE seq>10000")
    monkeypatch.setattr(HostedRoomAttachmentStore, "_read_blob",
                        lambda *args, **kwargs: pytest.fail("browsing must not read file bytes"))
    pages = []
    found = _all_items(db, pages=pages, query="needle")
    assert [item["name"] for item in found] == ["needle.md"]
    assert len(pages) > 1
    assert pages[0]["items"] == [] and pages[0]["has_more"]
    assert max(query_work) < 25000  # Even an empty sparse page must seek a bounded number of rows.


# One way each to damage a published version's record, and whether the version stays listed: a
# byte failure keeps it (unavailable), private, staged or conflicting metadata never lists it.
_DAMAGE = {
    "expired": (True, "UPDATE hosted_room_attachments SET expires_at=0 WHERE event_id=:event"),
    "blob-row": (True, "DELETE FROM hosted_room_attachment_blobs WHERE blob_id IN"
                       " (SELECT blob_id FROM hosted_room_attachments WHERE event_id=:event)"),
    "blob-digest": (True, f"UPDATE hosted_room_attachments SET sha256='{'0' * 64}' WHERE event_id=:event"),
    "wrong-event": (False, "UPDATE hosted_room_attachments SET event_id='wrong-event' WHERE event_id=:event"),
    "staged": (False, "UPDATE hosted_room_attachments SET state='uploaded' WHERE event_id=:event"),
    "viewer-access": (False, "UPDATE hosted_room_attachments SET viewer_access=0 WHERE event_id=:event"),
    "recipients": (False, "UPDATE hosted_room_attachments SET recipient_member_ids_json='invalid' WHERE event_id=:event"),
    "wrong-room": (False, "UPDATE hosted_room_attachments SET room_id='other-room' WHERE event_id=:event"),
    "actor": (False, "UPDATE hosted_room_events SET actor_json='{}' WHERE event_id=:event"),
    "payload": (False, "UPDATE hosted_room_events SET payload_json='not-json' WHERE event_id=:event"),
    "missing-event": (False, "DELETE FROM hosted_room_events WHERE event_id=:event"),
    "renamed": (False, "UPDATE hosted_room_events SET payload_json=json_set(payload_json, '$.attachments[0].name',"
                       " 'contradiction.bin') WHERE event_id=:event"),
    "extra-field": (False, "UPDATE hosted_room_events SET payload_json=json_set(payload_json,"
                           " '$.attachments[0].extra', 1) WHERE event_id=:event"),
    # The manifest as a JSON string instead of a list.
    "manifest-string": (False, "UPDATE hosted_room_events SET payload_json=json_set(payload_json, '$.attachments',"
                               " json_extract(payload_json, '$.attachments') || '') WHERE event_id=:event"),
}


def test_a_damaged_version_stays_listed_only_for_a_byte_failure(tmp_path):
    db, store = _create_catalog(tmp_path)
    names = ["intact", *_DAMAGE]
    _seed_events(db, total_events=len(names), records={seq: [f"{name}.bin"] for seq, name in enumerate(names, 1)})
    # A staged upload, and one committed to a message the log never published, are no versions at all.
    store.put(room_id=ROOM_ID, upload_id="staged-upload", kind="file", name="staged.bin",
              mime="application/octet-stream", data=b"x")
    unpublished = store.put(room_id=ROOM_ID, upload_id="unpublished-upload", kind="file",
                            name="unpublished.bin", mime="application/octet-stream", data=b"y")
    store.commit_message(room_id=ROOM_ID, event_id="never-published",
                         manifest=[{key: unpublished[key] for key in ("attachment_id", "kind", "name", "size", "mime")}],
                         recipient_member_ids=("ops", "qa"), viewer_access=True)
    with sqlite3.connect(db) as conn:
        for seq, (_listed, statement) in enumerate(_DAMAGE.values(), 2):
            conn.execute(statement, {"event": f"event-{seq}"})

    rows = _all_items(db, limit=32)
    assert [item["name"] for item in rows] == [
        f"{name}.bin" for name in reversed(names) if name == "intact" or _DAMAGE[name][0]]
    # This bulk metadata fixture has no blob files: every listed version is a reference without bytes.
    assert all(item["available"] is False for item in rows)


def test_epochless_share_is_listed_because_download_serves_it(tmp_path):
    db, store = _create_catalog(tmp_path)
    item, event_id, data = share(db, store)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_events SET authority_epoch=NULL WHERE event_id=?", (event_id,))
    assert store.read_viewer(room_id=ROOM_ID, attachment_id=item["attachment_id"], event_id=event_id,
                             authority_gateway_id=AUTHORITY, authority_epoch=1).data == data
    assert [entry["attachment_id"] for entry in _page(db)["items"]] == [item["attachment_id"]]


def test_revoked_room_or_changed_authority_lists_nothing(tmp_path):
    db, store = _create_catalog(tmp_path)
    share(db, store)
    with pytest.raises(AttachmentNotFoundError):
        _page(db, authority_epoch=2)
    with pytest.raises(AttachmentNotFoundError):
        _page(db, authority_gateway_id="gateway-b")
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS hosted_room_quarantine (
            room_id TEXT PRIMARY KEY, reason TEXT NOT NULL, detected_at REAL NOT NULL)""")
        conn.execute("INSERT INTO hosted_room_quarantine (room_id, reason, detected_at) VALUES (?, ?, ?)",
                     (ROOM_ID, "imported_unsafe_history", 1))
    with pytest.raises(AttachmentNotFoundError):
        _page(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM hosted_room_quarantine")
    assert len(_page(db)["items"]) == 1
    hosted_rooms.disband_room(db, room_id=ROOM_ID, expected_gateway_id=AUTHORITY, expected_epoch=1)
    with pytest.raises(AttachmentNotFoundError):
        _page(db)


def test_listing_needs_no_write_reservation(tmp_path):
    db, store = _create_catalog(tmp_path)
    share(db, store)
    with sqlite3.connect(db) as writer:
        writer.execute("BEGIN IMMEDIATE")
        assert len(_page(db)["items"]) == 1
        writer.rollback()


def test_staging_rows_on_a_published_message_do_not_widen_the_scan(tmp_path, query_work):
    db, _store = _create_catalog(tmp_path)
    _seed_events(db, total_events=2, records={1: ["good.bin"], 2: ["staging.bin"]})
    with sqlite3.connect(db) as conn:
        conn.executemany(
            """INSERT INTO hosted_room_attachments
               (attachment_id, upload_id, room_id, event_id, kind, name, size, mime, sha256, blob_id,
                recipient_member_ids_json, viewer_access, state, created_at, updated_at, expires_at)
               SELECT ?, ?, room_id, event_id, kind, name, size, mime, sha256, blob_id,
                      recipient_member_ids_json, 0, 'uploaded', created_at, updated_at, expires_at
                 FROM hosted_room_attachments WHERE upload_id='upload-2-0'""",
            ((f"att_{10000 + index:032x}", f"extra-{index}") for index in range(2000)))
    assert [item["name"] for item in _page(db)["items"]] == ["staging.bin", "good.bin"]
    assert sum(query_work) < 5000


@pytest.mark.parametrize("character", ["\U00020000", "한", "ﷺ"])
def test_longest_unicode_names_keep_pages_and_cursors_bounded(tmp_path, character):
    db, _store = _create_catalog(tmp_path)
    name = character * 255
    _seed_events(db, total_events=33, records={seq: [name] for seq in range(1, 34)})

    first = _page(db, limit=32, query=name)
    second = _page(db, limit=32, query=name, cursor=first["next_cursor"])

    assert len(first["items"]) == 32 and len(second["items"]) == 1
    assert len(first["next_cursor"]) <= catalog.MAX_CURSOR_CHARS
    assert len(json.dumps(first, ensure_ascii=False).encode()) < 128 * 1024


@pytest.mark.parametrize("field, values", [
    ("limit", (0, 33, True, "8")),
    ("query", ("x" * 256, " " * 256, "\ud800", 7)),
    ("producer_member_id", (False, 1, [], {}, "x" * 129, "not an id")),
])
def test_limit_query_and_sharer_inputs_are_bounded(tmp_path, field, values):
    db, _store = _create_catalog(tmp_path)
    for value in values:
        with pytest.raises(AttachmentError, match=field):
            _page(db, **{field: value})
