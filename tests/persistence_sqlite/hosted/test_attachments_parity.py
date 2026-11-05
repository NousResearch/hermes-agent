"""The SQLite ``HostedRoomAttachmentRepository`` against main's attachment store (gateway/hosted_room_attachments.py): the attachment retention an append runs inside its transaction, and the first verb that creates the attachment tables.

Each case seeds the profile's file through the room and attachment verbs, copies it to a twin, runs main's function on the twin and the backend's on the file, and compares the outcomes and the attachment tables. ``secrets.token_hex`` is replaced by a counter, so the minted attachment and blob ids are fixed; each module's first-construction set starts empty, so both run their first-touch reconciliation.
"""

from __future__ import annotations

import itertools
import secrets
import sqlite3
from typing import Any

import pytest

from tests.persistence_sqlite._main_callers import hosted_room_attachments as m1
from persistence.gateway.hosted_rooms import AttachmentNotFoundError
from persistence.sqlite.gateway import hosted_room_attachments as backend_module
from persistence.sqlite.gateway.hosted_room_attachments import SQLiteHostedRoomAttachmentRepository
from tests.persistence_sqlite.hosted._task_fixtures import (
    ATTACHMENT_TABLES, ROOM, Hosted, dump, make_hosted, outcome, schema, seed_room, status, twin,
)


FIELDS = ("attachment_id", "kind", "name", "size", "mime")
TTL_S = 60 * 60


@pytest.fixture
def hosted(monkeypatch) -> Hosted:
    monkeypatch.setattr(m1, "_STARTED", set())
    monkeypatch.setattr(backend_module, "_STARTED", set())
    hosted = make_hosted()
    yield hosted
    hosted.backend.close()


class Tokens:
    """A ``secrets.token_hex`` that counts up from zero."""

    def __init__(self) -> None:
        self.counter = itertools.count()

    def __call__(self, nbytes: int = 16) -> str:
        return format(next(self.counter), f"0{nbytes * 2}x")


@pytest.fixture
def tokens(monkeypatch) -> Tokens:
    fake = Tokens()
    monkeypatch.setattr(secrets, "token_hex", fake)
    return fake


def test_retain_message_attachments_matches_m1_inside_the_append_transaction(hosted, tmp_path, tokens):
    path = hosted.prepare("profile")
    seed_room(hosted, "profile")
    repo = SQLiteHostedRoomAttachmentRepository(hosted.store)
    meta = repo.put(placement="profile", room_id=ROOM, upload_id="up-1", kind="file", name="a.txt", mime="text/plain",
                    data=b"hello", now=110.0)
    held = repo.put(placement="profile", room_id=ROOM, upload_id="up-2", kind="file", name="b.txt",
                    mime="text/plain", data=b"bye", now=110.0)
    manifest = [{key: meta[key] for key in FIELDS}]
    repo.commit_message_with_receipt(placement="profile", room_id=ROOM, event_id="e1", manifest=manifest,
                                     recipient_member_ids=["m1"], viewer_access=True, hold_until_event=True, now=120.0)
    legacy_path = twin(path, tmp_path / "m1")

    def legacy(**kwargs: Any) -> Any:
        conn = sqlite3.connect(legacy_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            m1.retain_message_attachments(conn, **kwargs)
            conn.commit()
        finally:
            conn.close()

    def backend(**kwargs: Any) -> Any:
        hosted.files._hosted_write(
            "profile", lambda conn: backend_module.retain_message_attachments(conn, **kwargs), immediate=True)

    cases = [
        (dict(room_id=ROOM, event_id="e1", manifest=[], now=125.0), "ok"),
        (dict(room_id=ROOM, event_id="e1", manifest=[{**manifest[0], "mime": "TEXT/plain"}], now=125.0),
         "AttachmentError"),
        (dict(room_id=ROOM, event_id="e1", manifest=[{**manifest[0], "size": 0}], now=125.0), "AttachmentError"),
        (dict(room_id=ROOM, event_id="e2", manifest=manifest, now=125.0), "EventAttachmentConflictError"),
        (dict(room_id=ROOM, event_id="e1", manifest=[{key: held[key] for key in FIELDS}], now=125.0),
         "EventAttachmentConflictError"),
        (dict(room_id=ROOM, event_id="e1", manifest=manifest, now=120.0 + TTL_S), "EventAttachmentConflictError"),
        (dict(room_id=ROOM, event_id="e1", manifest=manifest, now=125.0), "ok"),
    ]
    for kwargs, expected in cases:
        legacy_result, backend_result = outcome(lambda: legacy(**kwargs)), outcome(lambda: backend(**kwargs))
        assert backend_result == legacy_result and status(backend_result) == expected, kwargs
        assert dump(path, ATTACHMENT_TABLES) == dump(legacy_path, ATTACHMENT_TABLES), kwargs


def test_attachment_schema_matches_m1(hosted, tmp_path):
    # CONTRACT: the attachment tables appear with the first attachment verb on the file, as main creates them when it
    # first constructs its attachment store for the file; the room verbs never create them.
    path = hosted.prepare("profile")
    seed_room(hosted, "profile")
    hosted.store.hosted_rooms.room_state(placement="profile", room_id=ROOM)
    assert schema(path, "*hosted_room_attachment*") == []
    legacy = twin(path, tmp_path / "m1")
    m1.HostedRoomAttachmentStore(legacy)
    with pytest.raises(AttachmentNotFoundError):
        SQLiteHostedRoomAttachmentRepository(hosted.store).describe(
            placement="profile", room_id=ROOM, attachment_id="att_" + "0" * 32, recipient_member_id="m1",
            event_id="e1", now=125.0)
    assert schema(path, "*hosted_room_attachment*") == schema(legacy, "*hosted_room_attachment*")
    assert schema(path, "*hosted_room_attachment*")
