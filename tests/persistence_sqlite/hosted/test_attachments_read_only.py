"""Attachment reads on a read-only store handle (``open_persistence_reader``) of the SQLite backend.

A reader runs ``read``, ``describe``, ``read_range`` and ``read_viewer`` without the first-touch setup a writable handle runs: no directory is created or chmodded, nothing is pruned, reconciled or migrated. Where the tables or the blob are missing, the reader gets the answer a writable handle gives for a missing attachment. The reference backend has no reader door, so these cases live here rather than in the contract suite.
"""

from __future__ import annotations

import sqlite3
import stat
from pathlib import Path

import pytest

from persistence import PermanentStoreError
from persistence.gateway.hosted_rooms import AttachmentIntegrityError
from persistence.sqlite.gateway import hosted_room_attachments as backend_module
from persistence_assembly import open_persistence_reader
from tests.persistence_sqlite.hosted._task_fixtures import (
    ATTACHMENT_DIR, ATTACHMENT_TABLES, GATEWAY, ROOM, dump, make_hosted, outcome, seed_room,
)

PROFILE = "profile-a"
DATA = b"0123456789"
FIELDS = ("attachment_id", "kind", "name", "size", "mime")
PLACEMENTS = ("profile", "shared")


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch) -> None:
    monkeypatch.setattr(backend_module, "_STARTED", set())


def _seed(placement: str, *, expired_upload: bool = False) -> tuple[Path, Path, dict]:
    """A room with one committed, published, viewer-visible attachment on ``placement``; returns the profile home, the placement's file and the attachment's metadata."""
    hosted = make_hosted(PROFILE)
    try:
        seed_room(hosted, placement)
        attachments = hosted.store.hosted_room_attachments
        meta = attachments.put(placement=placement, room_id=ROOM, upload_id="up-1", kind="file", name="a.txt", mime="text/plain", data=DATA, now=110.0)
        entry = {key: meta[key] for key in FIELDS}
        attachments.commit_message_with_receipt(placement=placement, room_id=ROOM, event_id="e1", manifest=[entry], recipient_member_ids=["m1"],
                                                viewer_access=True, hold_until_event=True, now=120.0)
        hosted.store.hosted_rooms.append_event(placement=placement, room_id=ROOM, event_id="e1", kind="message.user", actor={"kind": "user", "id": "desktop"},
                                               payload={"text": "hi", "attachments": [entry]}, authority_gateway_id=GATEWAY, authority_epoch=1, now=130.0)
        if expired_upload:
            attachments.put(placement=placement, room_id=ROOM, upload_id="up-stale", kind="file", name="b.txt", mime="text/plain", data=b"stale", now=110.0)
        return hosted.scope.profile_dir, hosted.path(placement), meta
    finally:
        hosted.backend.close()


def _reads(store, placement: str, attachment_id: str, now: float) -> dict:
    """Each read verb's outcome (value, or error class and message) for recipient ``m1`` of event ``e1``."""
    attachments = store.hosted_room_attachments
    where = {"placement": placement, "room_id": ROOM, "attachment_id": attachment_id, "now": now}
    return {
        "read": outcome(lambda: attachments.read(**where, recipient_member_id="m1", event_id="e1")),
        "describe": outcome(lambda: attachments.describe(**where, recipient_member_id="m1", event_id="e1")),
        "read_range": outcome(lambda: attachments.read_range(**where, recipient_member_id="m1", event_id="e1", offset=2, length=3)),
        "read_viewer": outcome(lambda: attachments.read_viewer(**where, event_id="e1", authority_gateway_id=GATEWAY, authority_epoch=1)),
    }


def _tree(folder: Path) -> dict[str, tuple[int, int, int]]:
    """Every path under ``folder`` with its mode, size and modification time."""
    return {str(item.relative_to(folder)): (item.stat().st_mode, item.stat().st_size, item.stat().st_mtime_ns)
            for item in sorted(folder.rglob("*"))} if folder.exists() else {}


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_a_reader_reads_a_committed_attachment_and_writes_nothing(placement):
    home, path, meta = _seed(placement, expired_upload=True)
    root = path.parent / ATTACHMENT_DIR
    root.chmod(0o755)
    backend_module._STARTED.clear()
    before = (_tree(root), dump(path, ATTACHMENT_TABLES))
    with open_persistence_reader(home) as reader:
        seen = _reads(reader, placement, meta["attachment_id"], now=140.0 + 2 * 60 * 60)
    attachment = {**meta, "state": "committed", "event_id": "e1"}
    assert seen["read"] == ("ok", {"attachment": attachment, "data": DATA})
    assert seen["describe"] == ("ok", attachment)
    assert seen["read_range"] == ("ok", {"attachment": attachment, "data": b"234"})
    assert seen["read_viewer"] == ("ok", {"attachment": attachment, "data": DATA})
    # In a process that has not set the file up, the expired upload is still stored, the folder keeps its mode, and the setup is still owed by the first writable verb.
    assert (_tree(root), dump(path, ATTACHMENT_TABLES)) == before
    assert stat.S_IMODE(root.stat().st_mode) == 0o755 and backend_module._STARTED == set()


def test_a_reader_on_a_file_without_attachment_tables_answers_as_for_a_missing_attachment():
    hosted = make_hosted(PROFILE)
    try:
        seed_room(hosted, "profile")
        path = hosted.path("profile")
    finally:
        hosted.backend.close()
    with open_persistence_reader(hosted.scope.profile_dir) as reader:
        seen = _reads(reader, "profile", "att_missing", now=140.0)
    # Nothing was created: no attachment tables and no attachment folder.
    assert dump(path, ATTACHMENT_TABLES) == {table: None for table in ATTACHMENT_TABLES}
    assert not (path.parent / ATTACHMENT_DIR).exists()
    assert set(seen.values()) == {("error", "AttachmentNotFoundError", "attachment is not committed for this room")}
    writer = make_hosted(PROFILE)
    try:
        assert _reads(writer.store, "profile", "att_missing", now=140.0) == seen
    finally:
        writer.backend.close()


def test_a_reader_whose_blob_folder_is_gone_gets_the_writable_answer_and_creates_no_folder():
    home, path, meta = _seed("profile")
    root = path.parent / ATTACHMENT_DIR
    for item in sorted(root.rglob("*"), reverse=True):
        item.unlink() if item.is_file() else item.rmdir()
    root.rmdir()
    with open_persistence_reader(home) as reader:
        seen = _reads(reader, "profile", meta["attachment_id"], now=140.0)
    missing = ("error", AttachmentIntegrityError.__name__, "canonical attachment blob is unavailable")
    assert (seen["read"], seen["read_range"], seen["read_viewer"]) == (missing, missing, missing)
    assert seen["describe"][0] == "ok" and not root.exists()
    writer = make_hosted(PROFILE)
    try:
        assert _reads(writer.store, "profile", meta["attachment_id"], now=140.0) == seen
    finally:
        writer.backend.close()


def test_a_reader_refuses_attachment_tables_older_than_this_version_and_leaves_them():
    home, path, meta = _seed("profile")
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE hosted_room_attachments DROP COLUMN catalog_name")
    before = dump(path, ATTACHMENT_TABLES)
    with open_persistence_reader(home) as reader:
        with pytest.raises(PermanentStoreError, match="a read-only store handle never migrates it"):
            reader.hosted_room_attachments.read(placement="profile", room_id=ROOM, attachment_id=meta["attachment_id"], recipient_member_id="m1", now=140.0)
    assert dump(path, ATTACHMENT_TABLES) == before


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_a_reader_reads_a_published_commitment_the_reconciliation_would_retain(placement):
    home, path, meta = _seed(placement)
    # A commitment published before retains were atomic: its event names it, its expiry was never lifted.
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE hosted_room_attachments SET expires_at=125.0 WHERE attachment_id=?", (meta["attachment_id"],))
    backend_module._STARTED.clear()
    with open_persistence_reader(home) as reader:
        seen = _reads(reader, placement, meta["attachment_id"], now=140.0)
    backend_module._STARTED.clear()
    writer = make_hosted(PROFILE)
    try:
        assert _reads(writer.store, placement, meta["attachment_id"], now=140.0) == seen
    finally:
        writer.backend.close()
    assert seen["describe"][0] == "ok"
