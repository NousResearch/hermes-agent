"""Contract cases for the read verbs of ``persistence.gateway.hosted_rooms.HostedRoomAttachmentRepository`` (``read``, ``describe``, ``read_range``, ``read_viewer``) and for ``append_user_event``, the gateway composition over the attachment verbs and ``HostedRoomRepository.append_event``.

Every case runs on each registered backend through the ``backend`` fixture and seeds through public verbs. The cases pin the committed-row checks with their messages in their order, the frozen recipient set, the published-owner rule for room-visible rows, the viewer authority fence, and the compensation and crash cover of a failed publication.
"""

from __future__ import annotations

import pytest

from persistence import NotFound
from persistence.gateway.hosted_rooms import (
    AttachmentError,
    AttachmentNotFoundError,
    AuthorityConflictError,
    EventAttachmentConflictError,
)

MEMBERS = [{"member_id": "m1", "profile": "p1"}, {"member_id": "m2", "profile": "p2"}]
FIELDS = ("attachment_id", "kind", "name", "size", "mime")
TTL_S = 60 * 60
DATA = b"0123456789"


def _room(store, room_id: str = "room"):
    return store.hosted_rooms.create_room(placement="profile", room_id=room_id, name="Room", members=MEMBERS, authority_gateway_id="gw", now=100.0)


def _put(store, upload_id: str = "up-1", *, data: bytes = DATA, now: float = 110.0):
    return store.hosted_room_attachments.put(placement="profile", room_id="room", upload_id=upload_id, kind="file", name="a.txt", mime="text/plain", data=data, now=now)


def _entry(meta) -> dict:
    return {key: meta[key] for key in FIELDS}


def _commit(store, metas, *, event_id: str = "e1", recipients=("m1",), viewer: bool = False, hold: bool = False, now: float = 120.0):
    return store.hosted_room_attachments.commit_message_with_receipt(
        placement="profile", room_id="room", event_id=event_id, manifest=[_entry(meta) for meta in metas],
        recipient_member_ids=list(recipients), viewer_access=viewer, hold_until_event=hold, now=now)


def _publish(store, metas, *, event_id: str = "e1", epoch: int = 1, now: float = 130.0):
    return store.hosted_rooms.append_event(
        placement="profile", room_id="room", event_id=event_id, kind="message.user", actor={"kind": "user", "id": "desktop"},
        payload={"text": "hi", "attachments": [_entry(meta) for meta in metas]}, authority_gateway_id="gw", authority_epoch=epoch, now=now)


def append_user_event(store, *, event_id: str, payload: dict, gateway_id: str, epoch: int, now: float):
    """A user-event append with its attachment commit, over the contract verbs, all on one placement."""
    manifest, transitioned = payload.get("attachments", []), ()
    if manifest:
        members = store.hosted_rooms.room_state(placement="profile", room_id="room")["members"]
        _, transitioned = store.hosted_room_attachments.commit_message_with_receipt(
            placement="profile", room_id="room", event_id=event_id, manifest=manifest,
            recipient_member_ids=[member["member_id"] for member in members], viewer_access=True, hold_until_event=True, now=now)
    try:
        return store.hosted_rooms.append_event(
            placement="profile", room_id="room", event_id=event_id, kind="message.user", actor={"kind": "user", "id": "desktop"},
            payload=payload, authority_gateway_id=gateway_id, authority_epoch=epoch, now=now)
    except Exception:
        if transitioned:
            store.hosted_room_attachments.abort_message_commit(placement="profile", room_id="room", event_id=event_id, attachment_ids=list(transitioned), now=now)
        raise


def _read(store, meta, *, member: str | None = "m1", event_id: str | None = "e1", viewer: bool = False, now: float = 140.0):
    return store.hosted_room_attachments.read(placement="profile", room_id="room", attachment_id=meta["attachment_id"], recipient_member_id=member, event_id=event_id, viewer=viewer, now=now)


def _viewer(store, meta, *, event_id: str | None = "e1", gateway: str = "gw", epoch: int = 1, now: float = 140.0):
    return store.hosted_room_attachments.read_viewer(placement="profile", room_id="room", attachment_id=meta["attachment_id"], event_id=event_id, authority_gateway_id=gateway, authority_epoch=epoch, now=now)


# ── read and describe ─────────────────────────────────────────────────────────


def test_read_returns_verified_bytes_and_metadata_to_a_frozen_recipient(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    meta = _put(store)
    _commit(store, [meta])
    saved = _read(store, meta)
    assert saved.data == DATA and saved.attachment == {**meta, "state": "committed", "event_id": "e1"}
    assert _read(store, meta, event_id=None).data == DATA
    assert store.hosted_room_attachments.describe(placement="profile", room_id="room", attachment_id=meta["attachment_id"], recipient_member_id="m1", event_id="e1", now=140.0) == saved.attachment


def test_read_refusals_check_the_commit_expiry_and_event_then_the_recipient(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    meta = _put(store)
    for kwargs in ({}, {"member": None}):
        with pytest.raises(AttachmentNotFoundError, match="attachment is not committed for this room") as refused:
            _read(store, meta, **kwargs)
        assert isinstance(refused.value, NotFound) and isinstance(refused.value, AttachmentError)
    _commit(store, [meta], hold=True)
    cases = (
        ({"now": 120.0 + TTL_S, "event_id": "e9", "member": "m9"}, "attachment has expired"),
        ({"event_id": "e9", "member": "m9"}, "attachment is not owned by this room event"),
        ({"member": "m2"}, "attachment is unavailable to this recipient"),
        ({"member": None}, "attachment is unavailable to this recipient"),
        ({"viewer": True}, "attachment is unavailable to Group Chat viewers"))
    for kwargs, message in cases:
        with pytest.raises(AttachmentNotFoundError, match=message):
            _read(store, meta, **kwargs)
    with pytest.raises(AttachmentNotFoundError, match="attachment is unavailable to this recipient"):
        store.hosted_room_attachments.describe(placement="profile", room_id="room", attachment_id=meta["attachment_id"], recipient_member_id="m2", event_id="e1", now=140.0)
    with pytest.raises(AttachmentNotFoundError, match="attachment is not owned by this room event"):
        store.hosted_room_attachments.describe(placement="profile", room_id="room", attachment_id=meta["attachment_id"], recipient_member_id="m1", event_id="e9", now=140.0)


def test_a_room_visible_row_needs_its_published_owner_even_for_members(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    meta = _put(store)
    _commit(store, [meta], recipients=("m1", "m2"), viewer=True, hold=True)
    # CONTRACT: a pre-event commitment is staging; neither a member nor a viewer reads it until the event names it.
    for kwargs in ({}, {"viewer": True, "member": None}):
        with pytest.raises(AttachmentNotFoundError, match="attachment is not published in this room"):
            _read(store, meta, **kwargs)
    _publish(store, [meta])
    assert _read(store, meta).data == _read(store, meta, viewer=True, member=None).data == DATA
    assert _read(store, meta, now=10.0 ** 9).attachment["state"] == "committed"


# ── read_range ────────────────────────────────────────────────────────────────


def _range(store, meta, offset: int, length: int, *, member: str = "m1", event_id: str = "e1"):
    return store.hosted_room_attachments.read_range(placement="profile", room_id="room", attachment_id=meta["attachment_id"], recipient_member_id=member, event_id=event_id, offset=offset, length=length, now=140.0)


def test_read_range_slices_the_blob_and_carries_the_whole_file_digest(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    meta = _put(store)
    _commit(store, [meta])
    middle = _range(store, meta, 2, 3)
    assert middle.data == b"234" and middle.attachment["sha256"] == meta["sha256"] and middle.attachment["size"] == 10
    assert _range(store, meta, 8, 100).data == b"89"
    with pytest.raises(AttachmentError, match="attachment range starts past the end") as refused:
        _range(store, meta, 10, 1)
    assert not isinstance(refused.value, NotFound)
    # The recipient check precedes the range check.
    with pytest.raises(AttachmentNotFoundError, match="attachment is unavailable to this recipient"):
        _range(store, meta, 10, 1, member="m2")


# ── read_viewer ───────────────────────────────────────────────────────────────


def test_read_viewer_reads_a_published_room_visible_attachment(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    meta = _put(store)
    _commit(store, [meta], viewer=True, hold=True)
    with pytest.raises(AttachmentNotFoundError, match="attachment is not published in this room"):
        _viewer(store, meta)
    _publish(store, [meta])
    saved = _viewer(store, meta)
    assert saved.data == DATA and saved.attachment == {**meta, "state": "committed", "event_id": "e1"}
    assert _viewer(store, meta, event_id=None).data == DATA
    with pytest.raises(AttachmentNotFoundError, match="attachment is not owned by this room event"):
        _viewer(store, meta, event_id="e9")


def test_read_viewer_is_fenced_on_the_live_room_and_its_authority(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    hidden = _put(store, "up-0", data=b"hidden")
    _commit(store, [hidden])
    meta = _put(store)
    _commit(store, [meta], event_id="e2", viewer=True, hold=True)
    _publish(store, [meta], event_id="e2")
    # The room fence precedes the row checks.
    for kwargs in ({"gateway": "gw-2"}, {"epoch": 2}):
        with pytest.raises(AttachmentNotFoundError, match="Group Chat viewer authority changed"):
            _viewer(store, hidden, **kwargs)
    with pytest.raises(AttachmentNotFoundError, match="attachment is unavailable to Group Chat viewers"):
        _viewer(store, hidden)
    assert _viewer(store, meta, event_id="e2").data == DATA
    store.hosted_rooms.disband_room(placement="profile", room_id="room", expected_gateway_id="gw", expected_epoch=1, now=150.0)
    with pytest.raises(AttachmentNotFoundError, match="Group Chat viewer authority changed"):
        _viewer(store, meta, event_id="e2", now=160.0)
    with pytest.raises(AttachmentNotFoundError, match="Group Chat viewer authority changed"):
        store.hosted_room_attachments.read_viewer(placement="profile", room_id="nowhere", attachment_id=meta["attachment_id"], authority_gateway_id="gw", authority_epoch=1, now=160.0)


# ── append_user_event (gateway composition) ───────────────────────────────────


def test_append_user_event_publishes_and_retains_its_attachments(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    meta = _put(store)
    event = append_user_event(store, event_id="e1", payload={"text": "hi", "attachments": [_entry(meta)]}, gateway_id="gw", epoch=1, now=120.0)
    assert event["seq"] == 1 and event["payload"]["attachments"] == [_entry(meta)]
    # The recipient set is the room roster read before the commit; the published row no longer expires.
    for member in ("m1", "m2"):
        assert _read(store, meta, member=member, now=10.0 ** 9).data == DATA
    assert _viewer(store, meta, now=10.0 ** 9).data == DATA


def test_append_user_event_aborts_the_commitment_when_the_append_refuses(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    meta = _put(store)
    payload = {"text": "hi", "attachments": [_entry(meta)]}
    with pytest.raises(AuthorityConflictError, match="stale hosted room authority"):
        append_user_event(store, event_id="e1", payload=payload, gateway_id="gw", epoch=2, now=120.0)
    # CONTRACT: the compensation returned the row to an upload, so the retry commits it again and publishes.
    with pytest.raises(AttachmentNotFoundError, match="attachment is not committed for this room"):
        _read(store, meta, now=130.0)
    assert append_user_event(store, event_id="e1", payload=payload, gateway_id="gw", epoch=1, now=130.0)["seq"] == 1


def test_a_commitment_whose_publication_never_ran_expires_with_the_upload_ttl(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    meta = _put(store)
    _commit(store, [meta], recipients=("m1", "m2"), viewer=True, hold=True)
    # Crash cover: neither append_event nor the abort ran; the held rows expire and the next put's prune deletes them.
    _put(store, "up-2", data=b"later", now=120.0 + TTL_S)
    with pytest.raises(EventAttachmentConflictError, match="attachment commitment changed before event publication"):
        _publish(store, [meta], now=120.0 + TTL_S)
    with pytest.raises(AttachmentNotFoundError, match="attachment is not uploaded for this room"):
        _commit(store, [meta], recipients=("m1", "m2"), viewer=True, hold=True, now=120.0 + TTL_S)
