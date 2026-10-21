"""The canonical authority route still carries a member turn's bound file end to end."""

from __future__ import annotations

import time
from pathlib import Path

from gateway import hosted_room_driver as state
from gateway import hosted_rooms
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
from hermes_state_runtime import list_session_admissions
from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime
from tests.waiting import wait_until


def test_authority_route_admits_a_member_turn_with_its_file(hosted_owner, tmp_path):
    authority, loop, principal, _ = hosted_owner
    binding = HostedRoomBinding(room_id="room", gateway_id="gateway-a", authority_epoch=1)
    room_db = tmp_path / "rooms.db"
    hosted_rooms.create_room(
        room_db, room_id="room", name="Release room",
        members=[{"profile": "default", "handle": "default"}],
        authority_gateway_id=binding.gateway_id, now=time.time())
    store = HostedRoomAttachmentStore(authority.db.db_path)
    saved = store.put(room_id="room", upload_id="upload", kind="file", name="note.txt",
                      mime="text/plain", data=b"exact bytes")
    manifest = [{k: saved[k] for k in ("attachment_id", "kind", "name", "size", "mime")}]
    store.commit_message(room_id="room", event_id="event", manifest=manifest,
                         recipient_member_ids=["member"])
    identity = state.TaskIdentity("room", "task", "thread", "turn")
    state.admit_task(room_db, identity, payload={
        "target_profile": "default", "target_member_id": "member", "prompt": "read",
        "source_event_seq": 1, "attachments": [{**manifest[0], "event_id": "event"}]},
        clock=time.time)
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id="room", member_id="member",
                                 profile="default", principal=principal,
                                 authorize=lambda *args: True)
    runtime = HostedRoomRuntime(
        db_path=room_db, rooms=[binding], rpc=rpc, turn_lock=lambda _profile: _NullLock(),
        poll_interval_seconds=0.01)

    runtime.start()
    try:
        wait_until(lambda: rpc.callbacks or runtime.status()["last_error"] is not None,
                   what="the canonical submit or a runtime error")
        last_error = runtime.status()["last_error"]
        status = state.get_task(room_db, identity)["status"]
    finally:
        assert runtime.stop(timeout=5.0)

    # Admitted with the bytes resolved, and still running: no refusal on this route.
    assert last_error is None
    assert status == "running"
    sid = rpc.resolve_exact(profile="default", title="Group: room", source="bot_room")["session_id"]
    rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert len(rows) == 1
    path = Path(rows[0]["payload"]["text"].split("file: ")[1].split("\n")[0])
    assert path.read_bytes() == b"exact bytes"


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
