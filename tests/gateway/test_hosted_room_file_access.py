"""Real canonical-store coverage for the messaging backend file bridge."""

import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_room_controls as controls, hosted_rooms
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.hosted_room_file_contract import FileAccessError, MANIFEST_FIELDS
from gateway.hosted_room_messaging import MessagingRoomBackend


def publish(
    state,
    name="shared.txt",
    data=b"published bytes",
    *,
    recipients=None,
    viewer=True,
    published=True,
    actor=None,
):
    index = len(state.files) + 1
    stored = state.store.put(
        room_id="room-1",
        upload_id=f"upload-{index}",
        kind="file",
        name=name,
        mime="text/plain",
        data=data,
    )
    event_id = f"event-{index}"
    manifest = {key: stored[key] for key in MANIFEST_FIELDS}
    state.store.commit_message(
        room_id="room-1",
        event_id=event_id,
        manifest=[manifest],
        recipient_member_ids=recipients or ["peer", "ops"],
        viewer_access=viewer,
        hold_until_event=published,
    )
    if published:
        actor = actor or {"kind": "user", "id": "desktop"}
        hosted_rooms.append_event(
            state.db,
            room_id="room-1",
            event_id=event_id,
            kind="message.member" if actor["kind"] == "member" else "message.user",
            actor=actor,
            authority_gateway_id=state.authority,
            authority_epoch=1,
            payload={
                "text": "shared",
                "thread_id": "thread-1",
                "attachments": [manifest],
                **({"member_id": actor["id"]} if actor["kind"] == "member" else {}),
            },
        )
    item = {**manifest, "event_id": event_id}
    state.files.append(item)
    return item


@pytest.fixture
def file_state(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db = home / "state.db"
    authority = hosted_rooms.local_authority_gateway_id()
    room = hosted_rooms.create_room(
        db,
        room_id="room-1",
        name="Files",
        authority_gateway_id=authority,
        members=[
            {
                "member_id": "peer",
                "profile": "reviewer",
                "handle": "reviewer",
                "target": {
                    "kind": "peer",
                    "installation_id": "install:peer",
                    "peer_id": "install:peer",
                    "profile": "reviewer",
                },
            },
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    store = HostedRoomAttachmentStore(db)
    backend = MessagingRoomBackend(
        db_path=db, service=SimpleNamespace(db_path=db, attachments=store)
    )
    return SimpleNamespace(
        db=db,
        room={**room, "_room_mode": "hosted"},
        authority=authority,
        store=store,
        backend=backend,
        files=[],
    )


def test_local_owner_and_named_profile_use_published_viewer_scope(
    file_state, monkeypatch
):
    state = file_state
    shared = publish(state)
    peer_only = publish(state, "peer-only.txt", recipients=["peer"])
    publish(state, "private.txt", viewer=False, published=False)
    publish(state, "pending.txt", published=False)
    monkeypatch.setattr(
        state.store, "_read_blob", lambda **kwargs: pytest.fail("eager blob read")
    )
    assert {
        item["attachment_id"]
        for item in state.backend.list_files(room=state.room)["items"]
    } == {shared["attachment_id"], peer_only["attachment_id"]}
    assert [
        item["attachment_id"]
        for item in state.backend.list_files(room=state.room, profile="ops")["items"]
    ] == [shared["attachment_id"]]
    for profile in ("unknown", "reviewer"):
        with pytest.raises(FileAccessError) as error:
            state.backend.list_files(room=state.room, profile=profile)
        assert error.value.code == "file_access_denied"


def test_local_read_is_exact_verified_and_bounded(file_state):
    state = file_state
    item = publish(state)
    selected = dict(
        room=state.room, event_id=item["event_id"], attachment_id=item["attachment_id"]
    )
    assert state.backend.read_file(**selected, profile="ops").data == b"published bytes"
    for changed in (
        {"event_id": "wrong-event"},
        {"max_bytes": 1},
        {"attachment_id": "a3f9c2e1"},
    ):
        with pytest.raises(FileAccessError):
            state.backend.read_file(**{**selected, **changed})
    blob = next(state.store.blob_root.glob("blob_*"))
    blob.write_bytes(b"corrupt bytes")
    with pytest.raises(FileAccessError) as error:
        state.backend.read_file(**selected)
    assert error.value.code == "file_integrity_failed"


@pytest.mark.parametrize(
    "failure",
    [
        "expired",
        "disbanded",
        "disbanding",
        "authority",
        "private",
        "pending",
        "member_removed",
    ],
)
def test_local_stale_access_never_returns_bytes(file_state, failure):
    state = file_state
    item = publish(
        state,
        viewer=failure != "private",
        published=failure not in {"private", "pending"},
    )
    if failure == "expired":
        with sqlite3.connect(state.db) as conn:
            conn.execute("UPDATE hosted_room_attachments SET expires_at=1")
    elif failure == "disbanded":
        hosted_rooms.disband_room(
            state.db,
            room_id="room-1",
            expected_gateway_id=state.authority,
            expected_epoch=1,
        )
    elif failure == "disbanding":
        retire = getattr(hosted_rooms, "begin_room_link_retirement", None)
        if callable(retire):
            retire(
                state.db, room_id="room-1",
                authority_gateway_id=state.authority, authority_epoch=1,
            )
        else:
            with sqlite3.connect(state.db) as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS hosted_room_disband_fences (
                        room_id TEXT PRIMARY KEY,
                        authority_gateway_id TEXT NOT NULL,
                        authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
                        started_at REAL NOT NULL,
                        revocation_complete_at REAL
                    )"""
                )
                conn.execute(
                    "INSERT INTO hosted_room_disband_fences VALUES (?, ?, 1, 1, NULL)",
                    ("room-1", state.authority),
                )
    elif failure == "authority":
        with sqlite3.connect(state.db) as conn:
            conn.execute("UPDATE hosted_rooms SET authority_gateway_id='other'")
    elif failure == "member_removed":
        with sqlite3.connect(state.db) as conn:
            conn.execute("UPDATE hosted_rooms SET members_json='[]'")
    with pytest.raises(FileAccessError):
        state.backend.read_file(
            room=state.room,
            profile="ops",
            event_id=item["event_id"],
            attachment_id=item["attachment_id"],
        )


def test_classic_is_typed_unsupported_without_creating_a_hosted_mirror(file_state):
    state = file_state
    before = hosted_rooms.list_rooms(state.db)
    classic = {"room_id": "classic", "_room_mode": "desktop"}
    for operation in (
        lambda: state.backend.list_files(room=classic),
        lambda: state.backend.read_file(
            room=classic, event_id="event", attachment_id="att_" + "0" * 32
        ),
    ):
        with pytest.raises(FileAccessError) as error:
            operation()
        assert error.value.code == "classic_files_on_desktop"
    assert hosted_rooms.list_rooms(state.db) == before


def test_remote_resolution_requires_persisted_profile_and_active_control(
    file_state, tmp_path
):
    state = file_state
    db = tmp_path / "peer" / "state.db"
    hosted_rooms.reserve_peer_room(
        db,
        claims={
            "room_id": "room-1",
            "member_id": "peer",
            "target_profile": "reviewer",
            "authority_gateway_id": state.authority,
            "authority_epoch": 1,
        },
        expires_at=time.time() + 300,
    )
    controls.save_peer_control_link(
        db,
        room_id="room-1",
        member_id="peer",
        target_profile="reviewer",
        home_url="http://127.0.0.1:9",
        authority_gateway_id=state.authority,
        authority_epoch=1,
        room_name="Files",
        member_count=2,
        control_token="A" * 43,
        expires_at=time.time() + 300,
    )
    backend = MessagingRoomBackend(db_path=db)
    room = {**state.room, "_room_mode": "remote", "_remote_member_id": "peer"}
    with pytest.raises(FileAccessError) as error:
        backend.list_files(room=room, profile="ops")
    assert error.value.code == "file_access_denied"
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_peer_controls SET target_profile=''")
    with pytest.raises(FileAccessError) as error:
        backend.list_files(room=room)
    assert error.value.code == "file_access_denied"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE hosted_room_peer_controls SET target_profile='reviewer'"
        )
        conn.execute("UPDATE hosted_room_peer_reservations SET revoked_at=1")
    with pytest.raises(FileAccessError) as error:
        backend.list_files(room=room)
    assert error.value.code == "file_access_denied"
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_peer_reservations SET revoked_at=NULL")
    controls.revoke_peer_control_links(db, room_id="room-1", member_id="peer")
    with pytest.raises(FileAccessError) as error:
        backend.list_files(room=room)
    assert error.value.code == "file_access_denied"


def test_cross_process_backend_reuses_store_but_never_caches_authorization(
    file_state, monkeypatch
):
    state = file_state
    publish(state)
    first = MessagingRoomBackend(db_path=state.db)
    assert first.list_files(room=state.room)["items"]
    monkeypatch.setattr(
        HostedRoomAttachmentStore,
        "__init__",
        lambda *args, **kwargs: pytest.fail("store startup repeated"),
    )
    second = MessagingRoomBackend(db_path=state.db)
    assert second.list_files(room=state.room)["items"]
    hosted_rooms.disband_room(
        state.db,
        room_id="room-1",
        expected_gateway_id=state.authority,
        expected_epoch=1,
    )
    with pytest.raises(FileAccessError) as error:
        second.list_files(room=state.room)
    assert error.value.code == "file_access_denied"


def test_private_pending_and_other_recipient_bytes_are_never_opened(
    file_state, monkeypatch
):
    state = file_state
    items = [
        publish(state, viewer=False, published=False),
        publish(state, published=False),
        publish(state, recipients=["peer"]),
    ]
    monkeypatch.setattr(
        state.store,
        "_read_blob",
        lambda **kwargs: pytest.fail("ineligible blob opened"),
    )
    for item in items:
        with pytest.raises(FileAccessError):
            state.backend.read_file(
                room=state.room,
                profile="ops",
                event_id=item["event_id"],
                attachment_id=item["attachment_id"],
            )


def test_named_profile_rebind_during_read_refuses_the_result(file_state, monkeypatch):
    import json

    state = file_state
    item = publish(state)
    original = state.store._read_blob

    def rebound(**kwargs):
        data = original(**kwargs)
        with sqlite3.connect(state.db) as conn:
            members = json.loads(
                conn.execute(
                    "SELECT members_json FROM hosted_rooms WHERE room_id='room-1'"
                ).fetchone()[0]
            )
            for member in members:
                if member["member_id"] == "ops":
                    member["profile"] = "other"
            conn.execute(
                "UPDATE hosted_rooms SET members_json=? WHERE room_id='room-1'",
                (json.dumps(members),),
            )
        return data

    monkeypatch.setattr(state.store, "_read_blob", rebound)
    with pytest.raises(FileAccessError) as error:
        state.backend.read_file(
            room=state.room,
            profile="ops",
            event_id=item["event_id"],
            attachment_id=item["attachment_id"],
        )
    assert error.value.code == "file_access_denied"
