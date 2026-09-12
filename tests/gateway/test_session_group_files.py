"""Canonical Files handlers use real room ownership, versions and byte stores."""

import base64
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.session_contract import Principal
from gateway.session_hosted_service import CanonicalHostedRoomService, _OWNER
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch


@pytest.fixture
def files(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with SessionDB(home / "state.db") as db:
        authority = SimpleNamespace(db=db, profile_id=str(home),
                                    epoch=begin_runtime_epoch(db, instance_id="files-test"))
        service = CanonicalHostedRoomService(authority, None)
        service.authorize_room("alice", "room", create=True)
        gateway = hosted_rooms.local_authority_gateway_id()
        hosted_rooms.create_room(db.db_path, room_id="room", name="Files", authority_gateway_id=gateway,
                                 members=[dict(member_id="writer", profile="default", handle="writer")])
        actor = Principal("alice", str(home), frozenset({"session:read", "session:submit"}), "test")
        yield service, actor, gateway, db


def share(service, actor, version):
    from gateway.session_group_files import dispatch_group_files
    from gateway.session_hosted_attachments import append_user_event
    uploaded = dispatch_group_files(service, actor, "groups.attachment.upload", dict(
        room_id="room", upload_id=f"upload-{version}", name="report.txt", kind="file", mime="text/plain",
        data_base64=base64.b64encode(f"version {version}".encode()).decode()))
    manifest = [{key: uploaded[key] for key in ("attachment_id", "kind", "name", "size", "mime")}]
    gateway, epoch = service._owned_authority("room")
    append_user_event(service, room_id="room", event_id=f"event-{version}",
                      payload=dict(text="Shared", attachments=manifest), gateway_id=gateway, epoch=epoch)
    return uploaded


def test_handlers_preserve_catalog_versions_scope_and_canonical_wire_format(files):
    from gateway.session_group_files import dispatch_group_files
    service, actor, gateway, db = files
    first, second = share(service, actor, 1), share(service, actor, 2)
    reader = replace(actor, capabilities=frozenset({"session:read"}))
    args = dict(room_id="room", limit=1, authority_gateway_id=gateway, authority_epoch=1)
    page = dispatch_group_files(service, reader, "groups.attachment.list", args)
    assert page["room_id"] == "room" and page["authority"] == dict(gateway_id=gateway, epoch=1)
    assert page["items"][0]["attachment_id"] == second["attachment_id"]
    share(service, actor, 3)
    reopened = CanonicalHostedRoomService(service.authority, None)
    older = dispatch_group_files(reopened, reader, "groups.attachment.list", {**args, "cursor": page["next_cursor"]})
    item = older["items"][0]
    assert item["attachment_id"] == first["attachment_id"] and item["event_id"] == "event-1"
    selected = dict(room_id="room", event_id=item["event_id"], attachment_id=item["attachment_id"],
                    authority_gateway_id=gateway, authority_epoch=1)
    data = dispatch_group_files(reopened, reader, "groups.attachment.download", selected)
    assert base64.b64decode(data["data_base64"]) == b"version 1"
    assert data["sha256"] == first["sha256"] and data["event_id"] == "event-1"
    assert data["room_id"] == "room" and data["authority"] == page["authority"]
    # Existing clients need no new parameters or decoder to keep downloading.
    old_wire = {key: selected[key] for key in ("room_id", "event_id", "attachment_id")}
    assert dispatch_group_files(reopened, reader, "groups.attachment.download", old_wire)["data_base64"] == data["data_base64"]
    for changed in [dict(event_id="event-2"), dict(authority_epoch=2), dict(authority_gateway_id="other")]:
        with pytest.raises(RuntimeStoreError):
            dispatch_group_files(reopened, reader, "groups.attachment.download", {**selected, **changed})
    with pytest.raises(RuntimeStoreError, match="attachment_cursor_invalid"):
        dispatch_group_files(reopened, reader, "groups.attachment.list", {**args, "cursor": page["next_cursor"], "query": "different"})


def test_handlers_refuse_foreign_or_changed_authority_before_returning_bytes(files, monkeypatch):
    from gateway.session_group_files import dispatch_group_files
    service, actor, gateway, db = files
    saved = share(service, actor, 1)
    selected = dict(room_id="room", event_id="event-1", attachment_id=saved["attachment_id"])
    for denied in [replace(actor, subject="bob"), replace(actor, capabilities=frozenset()),
                   replace(actor, profile_id="foreign")]:
        with pytest.raises(RuntimeStoreError):
            dispatch_group_files(service, denied, "groups.attachment.download", selected)
    for changed in [dict(path="/not-a-capability"), dict(event_id=None), dict(event_id=""),
                    dict(authority_epoch=True, authority_gateway_id=gateway),
                    dict(authority_epoch=1), dict(profile="foreign")]:
        with pytest.raises(RuntimeStoreError):
            dispatch_group_files(service, actor, "groups.attachment.download", {**selected, **changed})
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        dispatch_group_files(service, replace(actor, capabilities=frozenset({"session:read"})),
                             "groups.attachment.upload", dict(room_id="room"))
    original = HostedRoomAttachmentStore._read_blob
    def read_then_reassign(store, **kwargs):
        result = original(store, **kwargs)
        db._execute_write(lambda conn: conn.execute("UPDATE state_meta SET value=? WHERE key=?", ("bob", _OWNER + "room")))
        return result
    monkeypatch.setattr(HostedRoomAttachmentStore, "_read_blob", read_then_reassign)
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        dispatch_group_files(service, actor, "groups.attachment.download", selected)
