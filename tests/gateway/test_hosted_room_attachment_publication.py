"""Canonical message insertion and attachment rollback share one write fence."""

import sqlite3
from contextlib import contextmanager

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_attachments import (
    AttachmentNotFoundError,
    HostedRoomAttachmentStore,
    UNCOMMITTED_TTL_SECONDS,
)


def _staged(tmp_path, *, kind="message.member", viewer=True):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db, room_id="files", name="Files", members=[], authority_gateway_id="home"
    )
    store = HostedRoomAttachmentStore(db, clock=lambda: 1000.0)
    uploaded = store.put(
        room_id="files",
        upload_id="upload",
        kind="file",
        name="result.txt",
        mime="text/plain",
        data=b"exact output",
    )
    manifest = [
        {
            key: uploaded[key]
            for key in ("attachment_id", "kind", "name", "size", "mime")
        }
    ]
    store.commit_message(
        room_id="files",
        event_id="message",
        manifest=manifest,
        recipient_member_ids=["member"],
        viewer_access=viewer,
        hold_until_event=True,
    )
    args = dict(
        room_id="files",
        event_id="message",
        kind=kind,
        actor={"kind": "user", "id": "owner"}
        if kind == "message.user"
        else {"kind": "member", "id": "member"},
        payload={"text": "result", "attachments": manifest},
        authority_gateway_id="home",
        authority_epoch=1,
        now=1001.0,
        expected_latest_seq=0,
    )
    return db, store, manifest, args


def _abort(store, manifest):
    return store.abort_message_commit(
        room_id="files",
        event_id="message",
        attachment_ids=[item["attachment_id"] for item in manifest],
    )


def _read(store, manifest):
    return store.read(
        room_id="files",
        event_id="message",
        recipient_member_id="member",
        attachment_id=manifest[0]["attachment_id"],
    ).data


@pytest.mark.parametrize("kind", ["message.user", "message.member"])
@pytest.mark.parametrize("publish_first", [False, True])
def test_rollback_and_canonical_append_have_one_winner(tmp_path, kind, publish_first):
    db, store, manifest, args = _staged(tmp_path, kind=kind)
    if publish_first:
        committed = hosted_rooms.append_event(db, **args)
        assert not _abort(store, manifest)
        assert _read(store, manifest) == b"exact output"
        assert hosted_rooms.append_event(db, **args)["idempotent"]
        assert (
            hosted_rooms.read_events(db, room_id="files")["events"][0]["seq"]
            == committed["seq"]
        )
    else:
        assert _abort(store, manifest)
        with pytest.raises(hosted_rooms.EventCursorConflictError):
            hosted_rooms.append_event(db, **args)
        assert hosted_rooms.room_state(db, room_id="files")["latest_seq"] == 0
        with pytest.raises(AttachmentNotFoundError):
            _read(store, manifest)
        store.commit_message(
            room_id="files",
            event_id="message",
            manifest=manifest,
            recipient_member_ids=["member"],
            viewer_access=True,
            hold_until_event=True,
        )
        hosted_rooms.append_event(db, **args)
        assert _read(store, manifest) == b"exact output"


def test_append_validates_after_entering_its_sql_write_transaction(
    tmp_path, monkeypatch
):
    db, store, manifest, args = _staged(tmp_path)
    transaction = hosted_rooms._transaction

    @contextmanager
    def rollback_before_begin(*values, **kwargs):
        _abort(store, manifest)
        with transaction(*values, **kwargs) as conn:
            yield conn

    monkeypatch.setattr(hosted_rooms, "_transaction", rollback_before_begin)
    with pytest.raises(hosted_rooms.EventCursorConflictError):
        hosted_rooms.append_event(db, **args)
    assert hosted_rooms.read_events(db, room_id="files")["events"] == []


@pytest.mark.parametrize("invalid", ["room", "event", "metadata", "private", "expired"])
def test_append_requires_exact_live_room_visible_commitment(tmp_path, invalid):
    db, store, manifest, args = _staged(tmp_path, viewer=invalid != "private")
    if invalid == "room":
        hosted_rooms.create_room(
            db, room_id="other", name="Other", members=[], authority_gateway_id="home"
        )
        args["room_id"] = "other"
    elif invalid == "event":
        args["event_id"] = "other"
    elif invalid == "metadata":
        args["payload"]["attachments"][0]["name"] = "different.txt"
    elif invalid == "expired":
        args["now"] = 1000.0 + UNCOMMITTED_TTL_SECONDS
    with pytest.raises(hosted_rooms.EventCursorConflictError):
        hosted_rooms.append_event(db, **args)
    assert hosted_rooms.room_state(db, room_id=args["room_id"])["latest_seq"] == 0
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute("SELECT expires_at FROM hosted_room_attachments").fetchone()[0]
            is not None
        )


def test_canonical_append_retains_files_before_caller_can_crash_or_prune(tmp_path):
    db, store, manifest, args = _staged(tmp_path)
    hosted_rooms.append_event(db, **args)
    # No service retain_event/ACK call has run after append.
    assert store.prune(now=1001.0 + UNCOMMITTED_TTL_SECONDS) == 0
    assert _read(store, manifest) == b"exact output"


def test_failed_event_insert_rolls_back_attachment_retention(tmp_path, monkeypatch):
    db, store, manifest, args = _staged(tmp_path)

    def fail(*values, **kwargs):
        assert values[0].in_transaction
        assert (
            values[0]
            .execute("SELECT expires_at FROM hosted_room_attachments")
            .fetchone()[0]
            is None
        )
        raise RuntimeError("event capacity failed")

    monkeypatch.setattr(hosted_rooms, "_prepare_event", fail)
    with pytest.raises(RuntimeError, match="event capacity failed"):
        hosted_rooms.append_event(db, **args)
    assert _abort(store, manifest) == 1
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM hosted_room_events").fetchone()[0] == 0
        )
        assert (
            conn.execute("SELECT expires_at FROM hosted_room_attachments").fetchone()[0]
            is not None
        )


def test_receipt_rollback_releases_only_files_absent_from_durable_manifest(tmp_path):
    db, store, manifest, args = _staged(tmp_path)
    hosted_rooms.append_event(db, **args)
    extra = store.put(
        room_id="files",
        upload_id="extra",
        kind="file",
        name="extra.txt",
        mime="text/plain",
        data=b"not published",
    )
    extended = manifest + [{key: extra[key] for key in manifest[0]}]
    store.commit_message(
        room_id="files",
        event_id="message",
        manifest=extended,
        recipient_member_ids=["member"],
        viewer_access=True,
        hold_until_event=True,
    )
    assert _abort(store, extended) == 1
    assert _read(store, manifest) == b"exact output"
    with pytest.raises(AttachmentNotFoundError):
        _read(store, extended[1:])


@pytest.mark.parametrize("viewer", [False, True])
def test_room_visible_staging_denied_until_exact_owner_is_durable(tmp_path, viewer):
    db, store, manifest, args = _staged(tmp_path, kind="message.user")
    access = dict(
        room_id="files",
        attachment_id=manifest[0]["attachment_id"],
        event_id="message",
        recipient_member_id="member",
        viewer=viewer,
    )
    with pytest.raises(AttachmentNotFoundError):
        store.read(**access)
    hosted_rooms.append_event(db, **args)
    assert store.read(**access).data == b"exact output"


def test_append_does_not_read_blob_bytes_inside_its_write_fence(tmp_path, monkeypatch):
    db, store, manifest, args = _staged(tmp_path)

    def unexpected_io(*values, **kwargs):
        raise AssertionError("canonical append must do metadata-only SQL work")

    with monkeypatch.context() as fence:
        fence.setattr(HostedRoomAttachmentStore, "_read_blob", unexpected_io)
        hosted_rooms.append_event(db, **args)
    assert _read(store, manifest) == b"exact output"
