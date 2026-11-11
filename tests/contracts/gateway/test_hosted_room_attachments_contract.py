"""Contract cases for the write verbs of ``persistence.gateway.hosted_rooms.HostedRoomAttachmentRepository``: ``prepare``, ``put``, ``commit_message_with_receipt`` and ``abort_message_commit``.

Every case runs on each registered backend through the ``backend`` fixture and seeds through public verbs. The cases pin the metadata dict, idempotent replays, the refusals with their messages in their order, the quota order with the prune that runs first, the expiry a commitment takes, and the published commitment an abort never reverts.
"""

from __future__ import annotations

import hashlib
import re

import pytest

from persistence import Conflict, NotFound
from persistence.gateway.hosted_rooms import (
    AttachmentConflictError,
    AttachmentError,
    AttachmentNotFoundError,
    AttachmentQuotaError,
)

MEMBERS = [{"member_id": "m1", "profile": "p1"}, {"member_id": "m2", "profile": "p2"}]
FIELDS = ("attachment_id", "kind", "name", "size", "mime")
TTL_S = 60 * 60


def _room(store, room_id: str = "room", *, placement: str = "profile"):
    return store.hosted_rooms.create_room(placement=placement, room_id=room_id, name="Room", members=MEMBERS, authority_gateway_id="gw", now=100.0)


def _put(store, upload_id: str = "up-1", *, room_id: str = "room", placement: str = "profile", data: bytes = b"hello", name: str = "a.txt", mime: str = "text/plain", kind: str = "file", now: float = 110.0):
    return store.hosted_room_attachments.put(placement=placement, room_id=room_id, upload_id=upload_id, kind=kind, name=name, mime=mime, data=data, now=now)


def _entry(meta) -> dict:
    return {key: meta[key] for key in FIELDS}


def _commit(store, entries, *, event_id: str = "e1", room_id: str = "room", placement: str = "profile", recipients=("m1", "m2"), viewer: bool = False, retention=None, hold: bool = False, now: float = 120.0):
    return store.hosted_room_attachments.commit_message_with_receipt(
        placement=placement, room_id=room_id, event_id=event_id, manifest=entries, recipient_member_ids=list(recipients),
        viewer_access=viewer, retention_seconds=retention, hold_until_event=hold, now=now)


def _read(store, meta, *, member: str = "m1", event_id: str | None = "e1", room_id: str = "room", now: float = 130.0):
    return store.hosted_room_attachments.read(placement="profile", room_id=room_id, attachment_id=meta["attachment_id"], recipient_member_id=member, event_id=event_id, now=now)


# ── prepare ───────────────────────────────────────────────────────────────────


def test_prepare_runs_the_file_setup_once_and_leaves_later_expiry_to_the_verbs(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    assert store.hosted_room_attachments.prepare(placement="profile", now=105.0) is None
    meta = _put(store)
    # CONTRACT: the setup runs once per file and process, so a later prepare prunes nothing, even past the upload's expiry.
    assert store.hosted_room_attachments.prepare(placement="profile", now=110.0 + TTL_S) is None
    assert _commit(store, [_entry(meta)]) == ([_entry(meta)], (meta["attachment_id"],))


# ── put ───────────────────────────────────────────────────────────────────────


def test_put_returns_the_metadata_dict_and_replays_the_same_upload(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    meta = _put(store)
    assert re.fullmatch(r"att_[0-9a-f]{32}", meta["attachment_id"])
    assert meta == {
        "attachment_id": meta["attachment_id"], "kind": "file", "name": "a.txt", "size": 5, "mime": "text/plain",
        "sha256": hashlib.sha256(b"hello").hexdigest(), "state": "uploaded", "created_at": 110.0, "idempotent": False}
    # CONTRACT: the replay keeps the row's id and creation time and reports idempotent; the metadata never carries event_id before a commit.
    assert _put(store, now=500.0) == {**meta, "idempotent": True}


def test_put_refuses_a_reused_upload_id_with_other_content(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    _put(store)
    for changes in ({"data": b"other"}, {"name": "b.txt"}, {"mime": "text/csv"}, {"kind": "pdf"}, {"data": b"hellO"}):
        with pytest.raises(AttachmentConflictError, match="upload_id was already used for different attachment content") as refused:
            _put(store, **changes)
        assert isinstance(refused.value, Conflict) and isinstance(refused.value, AttachmentError) and isinstance(refused.value, ValueError)
    # The same upload id in another room is another upload.
    assert _put(store, room_id="room-2", data=b"other")["idempotent"] is False


def test_put_shares_content_across_uploads_and_keeps_each_row(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    first, second = _put(store, "up-1"), _put(store, "up-2", name="b.txt")
    assert first["attachment_id"] != second["attachment_id"] and first["sha256"] == second["sha256"]
    _commit(store, [_entry(first), _entry(second)])
    assert _read(store, first).data == _read(store, second).data == b"hello"


def test_put_refuses_past_the_uncommitted_count_of_the_room(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    for index in range(64):
        _put(store, f"up-{index}")
    with pytest.raises(AttachmentQuotaError, match="room uncommitted upload quota exceeded") as refused:
        _put(store, "up-64")
    assert isinstance(refused.value, AttachmentError) and not isinstance(refused.value, Conflict)
    # The quota is per room, and the exact replay of a stored upload precedes it.
    assert _put(store, "up-0")["idempotent"] is True
    assert _put(store, "up-64", room_id="room-2")["idempotent"] is False


def test_put_prunes_expired_uploads_before_its_quotas(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    first = _put(store, "up-0")
    for index in range(1, 64):
        _put(store, f"up-{index}")
    # CONTRACT: the prune runs with the same `now` first, so the expired uploads no longer count and their upload ids are free.
    fresh = _put(store, "up-0", now=110.0 + TTL_S)
    assert fresh["idempotent"] is False and fresh["attachment_id"] != first["attachment_id"] and fresh["created_at"] == 110.0 + TTL_S
    with pytest.raises(AttachmentNotFoundError, match="attachment is not uploaded for this room"):
        _commit(store, [_entry(first)], now=110.0 + TTL_S)


def test_the_two_placements_keep_their_attachments_apart(backend, scope_a, scope_b):
    store_a, store_b = backend.open(scope_a), backend.open(scope_b)
    _room(store_a, placement="shared")
    shared = _put(store_a, placement="shared")
    # The shared file is the install's root shared-state.db: another profile reaches the same row, the profile files do not.
    assert _put(store_b, placement="shared") == {**shared, "idempotent": True}
    assert _put(store_a)["attachment_id"] != shared["attachment_id"]
    _commit(store_b, [_entry(shared)], placement="shared")
    with pytest.raises(AttachmentNotFoundError, match="attachment is not committed for this room"):
        _read(store_a, shared)
    assert store_a.hosted_room_attachments.read(
        placement="shared", room_id="room", attachment_id=shared["attachment_id"], recipient_member_id="m2", event_id="e1", now=130.0).data == b"hello"


# ── commit_message_with_receipt ───────────────────────────────────────────────


def test_commit_binds_uploads_to_the_event_and_reports_only_its_own_transitions(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    first, second = _put(store, "up-1"), _put(store, "up-2", data=b"world")
    entries = [_entry(first), _entry(second)]
    assert _commit(store, entries) == (entries, (first["attachment_id"], second["attachment_id"]))
    # CONTRACT: an identical commitment replays with no transition; the frozen recipient set is order-free.
    assert _commit(store, entries, recipients=("m2", "m1"), now=125.0) == (entries, ())
    meta = _read(store, first, member="m2").attachment
    assert (meta["state"], meta["event_id"], meta["created_at"], meta["idempotent"]) == ("committed", "e1", 110.0, False)


def test_commit_refusals_check_the_upload_metadata_and_expiry_then_the_owner(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    meta = _put(store)
    with pytest.raises(AttachmentNotFoundError, match="attachment is not uploaded for this room") as missing:
        _commit(store, [{**_entry(meta), "attachment_id": "att_" + "0" * 32}])
    assert isinstance(missing.value, NotFound)
    with pytest.raises(AttachmentNotFoundError, match="attachment is not uploaded for this room"):
        _commit(store, [_entry(meta)], room_id="room-2")
    # The metadata check precedes the expiry check.
    with pytest.raises(AttachmentConflictError, match="attachment manifest metadata does not match the uploaded bytes"):
        _commit(store, [{**_entry(meta), "name": "b.txt"}], now=110.0 + TTL_S)
    with pytest.raises(AttachmentNotFoundError, match="attachment has expired"):
        _commit(store, [_entry(meta)], now=110.0 + TTL_S)
    _commit(store, [_entry(meta)])
    for changes in ({"event_id": "e2"}, {"recipients": ("m1",)}, {"viewer": True}):
        with pytest.raises(AttachmentConflictError, match="attachment is already owned by a different room event"):
            _commit(store, [_entry(meta)], **changes)


def test_commit_is_one_write_and_a_refused_entry_commits_nothing(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    first = _put(store, "up-1")
    with pytest.raises(AttachmentNotFoundError, match="attachment is not uploaded for this room"):
        _commit(store, [_entry(first), {**_entry(first), "attachment_id": "att_" + "1" * 32}])
    # A manifest naming one upload twice moves it once, then meets the fenced update.
    with pytest.raises(AttachmentConflictError, match="attachment changed during message commitment"):
        _commit(store, [_entry(first), _entry(first)])
    assert _commit(store, [_entry(first)]) == ([_entry(first)], (first["attachment_id"],))


def test_commit_refuses_rows_of_a_disbanded_room(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    meta = _put(store)
    _commit(store, [_entry(meta)])
    store.hosted_rooms.disband_room(placement="profile", room_id="room", expected_gateway_id="gw", expected_epoch=1, now=200.0)
    # The next put's prune marks committed rows of disbanded rooms; they expire after the 15 minute grace.
    _put(store, "up-x", room_id="room-2", now=300.0)
    with pytest.raises(AttachmentNotFoundError, match="attachment belongs to a disbanded room"):
        _commit(store, [_entry(meta)], now=300.0)
    with pytest.raises(AttachmentNotFoundError, match="attachment has expired"):
        _commit(store, [_entry(meta)], now=200.0 + 15 * 60)


@pytest.mark.parametrize(("options", "readable_until"), [
    ({"retention": None}, None), ({"retention": 50.0}, 120.0 + 50.0), ({"hold": True, "retention": 50.0}, 120.0 + TTL_S)])
def test_commit_sets_the_expiry_from_retention_and_hold(backend, scope_a, options, readable_until):
    store = backend.open(scope_a)
    _room(store)
    meta = _put(store)
    _commit(store, [_entry(meta)], **options)
    if readable_until is None:
        assert _read(store, meta, now=10.0 ** 9).data == b"hello"
        return
    assert _read(store, meta, now=readable_until - 1).data == b"hello"
    with pytest.raises(AttachmentNotFoundError, match="attachment has expired"):
        _read(store, meta, now=readable_until)


# ── abort_message_commit ──────────────────────────────────────────────────────


def _abort(store, ids, *, event_id: str = "e1", room_id: str = "room", now: float = 140.0) -> int:
    return store.hosted_room_attachments.abort_message_commit(placement="profile", room_id=room_id, event_id=event_id, attachment_ids=list(ids), now=now)


def test_abort_returns_an_unpublished_commitment_to_a_fresh_upload(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    first, second = _put(store, "up-1"), _put(store, "up-2", data=b"world")
    _, transitioned = _commit(store, [_entry(first), _entry(second)], viewer=True, hold=True)
    assert _abort(store, []) == 0
    assert _abort(store, transitioned, event_id="e9") == 0 and _abort(store, transitioned, room_id="room-2") == 0
    assert _abort(store, transitioned) == 2
    assert _abort(store, transitioned) == 0
    # CONTRACT: the rows are uploads again (no event, no recipients, no viewer flag) that expire one TTL after the abort.
    assert _commit(store, [_entry(first)], event_id="e2", recipients=("m1",), now=150.0)[1] == (first["attachment_id"],)
    with pytest.raises(AttachmentNotFoundError, match="attachment has expired"):
        _commit(store, [_entry(second)], event_id="e2", now=140.0 + TTL_S)


def test_abort_never_reverts_a_published_commitment(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    meta = _put(store)
    _, transitioned = _commit(store, [_entry(meta)], viewer=True, hold=True)
    store.hosted_rooms.append_event(
        placement="profile", room_id="room", event_id="e1", kind="message.user", actor={"kind": "user", "id": "desktop"},
        payload={"text": "hi", "attachments": [_entry(meta)]}, authority_gateway_id="gw", authority_epoch=1, now=130.0)
    assert _abort(store, transitioned) == 0
    assert _read(store, meta, now=10.0 ** 9).attachment["state"] == "committed"
