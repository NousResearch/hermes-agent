"""``read_viewer`` runs its room check and its attachment-row read in one SQLite snapshot, as main's viewer read
does with its explicit ``BEGIN`` (gateway/hosted_room_attachments.py ``HostedRoomAttachmentStore.read_viewer``)."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from persistence.sqlite.gateway import hosted_room_attachments as backend_module
from persistence.sqlite.gateway.hosted_room_attachments import SQLiteHostedRoomAttachmentRepository
from tests.persistence_sqlite.hosted._task_fixtures import GATEWAY, ROOM, m1_event, m1_room, make_hosted

FIELDS = ("attachment_id", "kind", "name", "size", "mime")


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setattr(backend_module, "_STARTED", set())
    hosted = make_hosted()
    yield hosted
    hosted.backend.close()


def _set_expiry(path, attachment_id: str, expires_at) -> None:
    with closing(sqlite3.connect(path)) as other:
        other.execute("UPDATE hosted_room_attachments SET expires_at=? WHERE attachment_id=?", (expires_at, attachment_id))
        other.commit()


@pytest.mark.parametrize("placement", ["profile", "shared"])
def test_viewer_read_checks_the_room_and_reads_the_row_in_one_snapshot(hosted, monkeypatch, placement):
    path = hosted.prepare(placement)
    m1_room(path)
    repo = SQLiteHostedRoomAttachmentRepository(hosted.store)
    meta = repo.put(placement=placement, room_id=ROOM, upload_id="up-1", kind="file", name="a.txt", mime="text/plain",
                    data=b"hello", now=110.0)
    attachment_id = meta["attachment_id"]
    manifest = {key: meta[key] for key in FIELDS}
    repo.commit_message_with_receipt(placement=placement, room_id=ROOM, event_id="e-v", manifest=[manifest],
                                     recipient_member_ids=["m1"], viewer_access=True, hold_until_event=True, now=120.0)
    m1_event(path, event_id="e-v", kind="message.user", payload={"text": "hi", "attachments": [manifest]}, now=121.0)
    with closing(sqlite3.connect(path)) as conn:
        expires_at = conn.execute("SELECT expires_at FROM hosted_room_attachments WHERE attachment_id=?",
                                  (attachment_id,)).fetchone()[0]

    read_row, read_blob = backend_module._read_committed_row, backend_module._read_blob
    calls = []

    def interleaved(conn, **kwargs):
        # Another connection expires the row between the first read's room check and its row read; inside one
        # snapshot that row read still sees the row as it was.
        calls.append(kwargs["attachment_id"])
        if len(calls) == 1:
            _set_expiry(path, attachment_id, 1.0)
        return read_row(conn, **kwargs)

    def restored(*args, **kwargs):
        # The byte read runs between the two reads, outside any snapshot: the row is restored before the second.
        _set_expiry(path, attachment_id, expires_at)
        return read_blob(*args, **kwargs)

    monkeypatch.setattr(backend_module, "_read_committed_row", interleaved)
    monkeypatch.setattr(backend_module, "_read_blob", restored)
    data = repo.read_viewer(placement=placement, room_id=ROOM, attachment_id=attachment_id, event_id="e-v",
                            authority_gateway_id=GATEWAY, authority_epoch=1, now=130.0)
    assert data.data == b"hello"
    assert len(calls) == 2
