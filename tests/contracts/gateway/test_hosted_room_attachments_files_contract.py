"""Contract cases for the blob-file steps of ``persistence.gateway.hosted_rooms.HostedRoomAttachmentRepository`` and the quota order of ``put``, through the in-memory reference's seams.

The blob files, the quota constants and a write landing during a byte read are backend internals, so these cases reach them through the reference's own seams (its blob file map with the ``before_read`` hook, its constants module) and skip on any backend without injectable failpoints. The cases cover the file steps, the quotas and the blob, the prune, the unlinks after it and the orphan sweep, and the viewer re-check.
"""

from __future__ import annotations

import pytest

from persistence.gateway.hosted_rooms import AttachmentIntegrityError, AttachmentNotFoundError, AttachmentQuotaError
from tests.contracts._reference.gateway import hosted_room_attachment_rules as limits
from tests.contracts._reference.gateway import hosted_room_attachments as reference
from tests.contracts._reference.gateway.hosted_room_attachment_files import blob_files
from tests.contracts._reference.gateway.hosted_room_tables import write_tables
from tests.contracts.conftest import require_failpoints

MEMBERS = [{"member_id": "m1", "profile": "p1"}, {"member_id": "m2", "profile": "p2"}]
FIELDS = ("attachment_id", "kind", "name", "size", "mime")
TTL_S = 60 * 60


@pytest.fixture
def store(backend, scope_a):
    require_failpoints(backend)
    opened = backend.open(scope_a)
    opened.hosted_rooms.create_room(placement="profile", room_id="room", name="Room", members=MEMBERS, authority_gateway_id="gw", now=100.0)
    return opened


def _put(store, upload_id: str = "up-1", *, room_id: str = "room", data: bytes = b"hello", now: float = 110.0):
    return store.hosted_room_attachments.put(placement="profile", room_id=room_id, upload_id=upload_id, kind="file", name="a.txt", mime="text/plain", data=data, now=now)


def _entry(meta) -> dict:
    return {key: meta[key] for key in FIELDS}


def _commit(store, meta, *, viewer: bool = False, now: float = 120.0):
    return store.hosted_room_attachments.commit_message_with_receipt(
        placement="profile", room_id="room", event_id="e1", manifest=[_entry(meta)], recipient_member_ids=["m1"], viewer_access=viewer, hold_until_event=viewer, now=now)


def _read(store, meta, *, now: float = 130.0):
    return store.hosted_room_attachments.read(placement="profile", room_id="room", attachment_id=meta["attachment_id"], recipient_member_id="m1", event_id="e1", now=now)


def _blob_names(store) -> set[str]:
    return {name for name in blob_files(store, "profile") if name.startswith("blob_")}


# ── blob files ────────────────────────────────────────────────────────────────


def test_one_blob_file_per_content_released_after_its_last_row(store):
    first, second = _put(store, "up-1"), _put(store, "up-2")
    (blob,) = _blob_names(store)
    assert blob_files(store, "profile")[blob] == b"hello" and blob_files(store, "shared") == {}
    _commit(store, first)
    # The prune deletes the expired upload and drops one reference; the committed row keeps the file.
    _put(store, "up-3", data=b"other", now=110.0 + TTL_S)
    assert blob in _blob_names(store) and len(_blob_names(store)) == 2
    store.hosted_rooms.disband_room(placement="profile", room_id="room", expected_gateway_id="gw", expected_epoch=1, now=200.0 + TTL_S)
    _put(store, "up-4", room_id="room-2", data=b"four", now=200.0 + TTL_S)
    _put(store, "up-5", room_id="room-2", data=b"five", now=200.0 + TTL_S + 15 * 60)
    # CONTRACT: the disbanded room's committed row expired after the grace and its last reference unlinked the file after the commit.
    assert blob not in _blob_names(store)
    assert second["sha256"] == first["sha256"]


def test_the_orphan_sweep_reclaims_staging_quarantine_and_unreferenced_blobs(store):
    meta = _put(store)
    (live,) = _blob_names(store)
    files = blob_files(store, "profile")
    files.update({".tmp-1": b"x", ".prune-2": b"x", "blob_" + "a" * 32: b"x", "README": b"x"})
    _put(store, "up-2", now=120.0)
    assert set(files) == {live, "README"}
    assert _commit(store, meta)[1] == (meta["attachment_id"],)


def test_a_blob_written_by_a_refused_put_is_left_to_the_sweep(store, monkeypatch):
    def refuse(_name: str) -> str:
        raise RuntimeError("row insert failed")

    monkeypatch.setattr(reference.rules, "fold_catalog_text", refuse)
    with pytest.raises(RuntimeError, match="row insert failed"):
        _put(store)
    # The write rolled back but its file stays, unreferenced, until the next prune's sweep.
    (orphan,) = _blob_names(store)
    monkeypatch.undo()
    meta = _put(store)
    assert orphan not in _blob_names(store) and len(_blob_names(store)) == 1
    assert meta["idempotent"] is False


@pytest.mark.parametrize(("damage", "message"), [
    (lambda data: b"j" + data[1:], "canonical attachment blob failed SHA-256 validation"),
    (lambda data: data[:-1], "canonical attachment blob size changed"),
    (None, "canonical attachment blob is unavailable")])
def test_every_verified_read_refuses_a_damaged_blob(store, damage, message):
    meta = _put(store)
    _commit(store, meta)
    (blob,) = _blob_names(store)
    files = blob_files(store, "profile")
    if damage is None:
        del files[blob]
    else:
        files[blob] = damage(files[blob])
    for verb in (lambda: _put(store), lambda: _put(store, "up-2"), lambda: _commit(store, meta), lambda: _read(store, meta)):
        with pytest.raises(AttachmentIntegrityError, match=message):
            verb()


def test_read_range_fences_the_size_but_does_not_hash(store):
    meta = _put(store)
    _commit(store, meta)
    (blob,) = _blob_names(store)
    files = blob_files(store, "profile")
    files[blob] = b"jello"

    def slice_(offset: int, length: int) -> bytes:
        return store.hosted_room_attachments.read_range(
            placement="profile", room_id="room", attachment_id=meta["attachment_id"], recipient_member_id="m1", event_id="e1", offset=offset, length=length, now=130.0).data

    # CONTRACT: the receiver verifies the reassembled bytes against the whole-file digest; the slice read only fences the size.
    assert slice_(0, 2) == b"je"
    files[blob] = b"hell"
    with pytest.raises(AttachmentIntegrityError, match="canonical attachment blob size changed"):
        slice_(0, 2)


def _published(store):
    meta = _put(store)
    _commit(store, meta, viewer=True)
    store.hosted_rooms.append_event(
        placement="profile", room_id="room", event_id="e1", kind="message.user", actor={"kind": "user", "id": "desktop"},
        payload={"text": "hi", "attachments": [_entry(meta)]}, authority_gateway_id="gw", authority_epoch=1, now=125.0)
    return meta


def _during_next_read(store, write) -> None:
    files = blob_files(store, "profile")

    def once() -> None:
        files.before_read = None
        write()

    files.before_read = once


def _viewer(store, meta):
    return store.hosted_room_attachments.read_viewer(placement="profile", room_id="room", attachment_id=meta["attachment_id"], event_id="e1", authority_gateway_id="gw", authority_epoch=1, now=130.0)


def test_read_viewer_refuses_when_the_room_is_revoked_during_the_byte_read(store):
    meta = _published(store)
    _during_next_read(store, lambda: store.hosted_rooms.disband_room(placement="profile", room_id="room", expected_gateway_id="gw", expected_epoch=1, now=128.0))
    with pytest.raises(AttachmentNotFoundError, match="Group Chat viewer authority changed"):
        _viewer(store, meta)


def test_read_viewer_refuses_when_the_row_moves_during_the_byte_read(store):
    meta = _published(store)

    def move() -> None:
        with write_tables(store, "profile") as tables:
            tables["hosted_room_attachments"]["room"][meta["attachment_id"]]["created_at"] = 111.0

    _during_next_read(store, move)
    with pytest.raises(AttachmentNotFoundError, match="attachment changed during viewer read"):
        _viewer(store, meta)
    assert _viewer(store, meta).attachment["created_at"] == 111.0


# ── put quotas ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("limit", "value", "message"), [
    ("MAX_ROOM_ATTACHMENT_BYTES", 8, "room attachment quota exceeded"),
    ("MAX_ROOM_ATTACHMENT_COUNT", 1, "room attachment count quota exceeded"),
    ("MAX_ROOM_UNCOMMITTED_BYTES", 8, "room uncommitted upload quota exceeded"),
    ("MAX_GATEWAY_BLOB_BYTES", 8, "gateway attachment quota exceeded")])
def test_put_room_quotas(store, monkeypatch, limit, value, message):
    monkeypatch.setattr(limits, limit, value)
    _put(store, "up-1")
    with pytest.raises(AttachmentQuotaError, match=message):
        _put(store, "up-2", data=b"world")
    # Known content needs no new physical bytes, so only the gateway blob quota lets it through.
    if limit == "MAX_GATEWAY_BLOB_BYTES":
        assert _put(store, "up-2")["sha256"] == _put(store, "up-1")["sha256"]


def test_put_quotas_count_disbanded_rows_for_the_gateway_but_not_the_room(store, monkeypatch):
    for limit, value in (("MAX_ROOM_ATTACHMENT_BYTES", 8), ("MAX_ROOM_ATTACHMENT_COUNT", 1), ("MAX_GATEWAY_ATTACHMENT_COUNT", 2)):
        monkeypatch.setattr(limits, limit, value)
    meta = _put(store, "up-1")
    with pytest.raises(AttachmentQuotaError, match="room attachment quota exceeded"):
        _put(store, "up-2", data=b"world")
    _commit(store, meta)
    store.hosted_rooms.disband_room(placement="profile", room_id="room", expected_gateway_id="gw", expected_epoch=1, now=150.0)
    _put(store, "up-x", room_id="room-2", data=b"x", now=160.0)
    # The disbanded row no longer counts for its room but still counts for the gateway.
    with pytest.raises(AttachmentQuotaError, match="gateway attachment count quota exceeded"):
        _put(store, "up-2", data=b"world", now=170.0)
