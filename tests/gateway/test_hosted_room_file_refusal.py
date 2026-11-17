"""A member's bound file is prepared before its turn is submitted, on both server routes.

A file the member can never take (over its ``gateway.max_inbound_media_bytes``) is a refusal of
work that never started: the turn fails at once instead of pausing the room for recovery as an
uncertain execution. A transient preparation failure is proven not admitted and is requeued.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gateway import hosted_room_driver as state
from gateway import hosted_rooms
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from tests.tui_gateway.test_hosted_room_attachment_routes import (
    _admit, _legacy_runtime, _run_until_settled_or_error, _server_module)
from tests.tui_gateway.test_hosted_room_driver_runtime import BINDING, PROFILE, ROOM_ID
from tests.waiting import wait_until
from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime
from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC
from persistence.gateway.hosted_rooms import HostedTaskIdentity, TERMINAL_STATUSES
from persistence_assembly import open_persistence, profile_scope


@pytest.fixture(name="store")
def store_fixture(tmp_path):
    """A store handle on the test home, closed at teardown."""
    handle = open_persistence(profile_scope(home=tmp_path))
    yield handle
    handle.close()


def _cap_member_media(monkeypatch, limit: int) -> None:
    from gateway.platforms import base

    monkeypatch.setattr(base, "get_inbound_media_max_bytes", lambda: limit)


def _bind(store: HostedRoomAttachmentStore, room_id: str, member_id: str) -> list[dict]:
    saved = store.put(room_id=room_id, upload_id="upload", kind="file", name="note.txt",
                      mime="text/plain", data=b"exact bytes")
    manifest = [{k: saved[k] for k in ("attachment_id", "kind", "name", "size", "mime")}]
    store.commit_message(room_id=room_id, event_id="event", manifest=manifest,
                         recipient_member_ids=[member_id])
    return [{**manifest[0], "event_id": "event"}]


@pytest.fixture
def tui_db(tmp_path: Path, monkeypatch, store) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "member-home"))
    path = tmp_path / "shared-state.db"
    hosted_rooms.create_room(
        store, placement="shared", room_id=ROOM_ID, name="Release room", members=[{"profile": PROFILE, "handle": PROFILE}],
        authority_gateway_id=BINDING.gateway_id, now=time.time())
    return path


def _tui_run(db: Path, store):
    identity = _admit(store, placement="shared", attachments=_bind(
        HostedRoomAttachmentStore(storage_handle(db.parent), "shared"), ROOM_ID, "member-ops"))
    server, submits = _server_module()
    server._profile_home = lambda _profile: None
    runtime = _legacy_runtime(store, HostedRoomServerRPC(server, db_path=db), placement="shared")
    task = _run_until_settled_or_error(runtime, store, identity)
    return task, submits, runtime


def test_tui_route_fails_a_turn_whose_file_exceeds_the_member_cap(tui_db, monkeypatch, store):
    _cap_member_media(monkeypatch, 4)
    task, submits, runtime = _tui_run(tui_db, store)
    assert task["status"] == "failed", runtime.status()["last_error"]
    assert submits == []
    assert ROOM_ID not in runtime._ambiguous_rooms


def test_tui_route_requeues_a_turn_whose_file_preparation_failed_transiently(tui_db, monkeypatch, store):
    def unavailable(*_args, **_kwargs):
        raise OSError("disk briefly unavailable")

    monkeypatch.setattr(HostedRoomAttachmentStore, "read", unavailable)
    task, submits, runtime = _tui_run(tui_db, store)
    assert task["status"] == "queued", runtime.status()["last_error"]
    assert "was not admitted" in runtime.status()["last_error"]
    assert submits == []
    assert ROOM_ID not in runtime._ambiguous_rooms


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_authority_route_fails_a_turn_whose_file_exceeds_the_member_cap(hosted_owner, tmp_path, monkeypatch, store):
    authority, loop, principal, _ = hosted_owner
    binding = HostedRoomBinding(room_id="room", gateway_id="gateway-a", authority_epoch=1)
    hosted_rooms.create_room(
        store, placement="shared", room_id="room", name="Release room", members=[{"profile": "default", "handle": "default"}],
        authority_gateway_id=binding.gateway_id, now=time.time())
    identity = HostedTaskIdentity("room", "task", "thread", "turn")
    state.admit_task(store, identity, placement="shared", payload={
        "target_profile": "default", "target_member_id": "member", "prompt": "read",
        "source_event_seq": 1, "attachments": _bind(HostedRoomAttachmentStore(authority.persistence), "room", "member")},
        clock=time.time)
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id="room", member_id="member",
                                 profile="default", principal=principal, authorize=lambda *args: True)
    runtime = HostedRoomRuntime(store=store, placement="shared", rooms=[binding], rpc=rpc,
                                turn_lock=lambda _profile: _NullLock(), poll_interval_seconds=0.01)
    _cap_member_media(monkeypatch, 4)

    runtime.start()
    try:
        wait_until(lambda: state.get_task(store, identity, placement="shared")["status"] in TERMINAL_STATUSES
                   or runtime.status()["last_error"] is not None, what="a settled refusal or an error")
        status = state.get_task(store, identity, placement="shared")["status"]
    finally:
        assert runtime.stop(timeout=5.0)

    assert status == "failed", runtime.status()["last_error"]
    assert rpc.callbacks == {}
    assert "room" not in runtime._ambiguous_rooms
