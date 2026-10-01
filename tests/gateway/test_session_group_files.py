"""``groups.attachment.list`` on the canonical wire: authorization, exact versions and base download."""
import asyncio
import base64
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from gateway import hosted_room_attachment_catalog as catalog
from gateway import hosted_rooms
from gateway.session_controls import AuthorityConnection
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with SessionDB(home / "state.db") as db:
        authority = SimpleNamespace(db=db, profile_id=str(home), instance_id="owner", events={},
                                    epoch=begin_runtime_epoch(db, instance_id="owner"))
        service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
        service.local_profiles = lambda: ("default", "ops")
        # The room driver counts as running; nothing here starts its worker thread.
        monkeypatch.setattr(service.runtime, "status", lambda: {"running": True, "stopping": False})
        owner = AuthorityConnection(authority, object(), {"user_id": "alice"})
        service.authorize_room(owner.actor.subject, "room", create=True)
        hosted_rooms.create_room(db.db_path, room_id="room", name="Files",
                                 authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
                                 members=[{"member_id": "writer", "profile": "default", "handle": "writer"},
                                          {"member_id": "reviewer", "profile": "ops", "handle": "reviewer"}])
        yield SimpleNamespace(authority=authority, service=service, owner=owner)


def call(connection, method, **params):
    return asyncio.run(connection.dispatch({"id": 1, "method": method, "params": params}))


def reason(reply):
    assert "error" in reply, reply
    return reply["error"]["data"]["reason"]


def share(gateway, n, data, name="report.txt"):
    uploaded = call(gateway.owner, "groups.attachment.upload", room_id="room", upload_id=f"upload-{n}",
                    kind="file", name=name, mime="text/plain", data_base64=base64.b64encode(data).decode())["result"]
    manifest = [{key: uploaded[key] for key in ("attachment_id", "kind", "name", "size", "mime")}]
    sent = call(gateway.owner, "groups.send", room_id="room", event_id=f"client-{n}",
                payload={"text": f"version {n}", "thread_id": "files", "attachments": manifest})
    return sent["result"]["event"]["event_id"], uploaded["attachment_id"]


def test_list_offers_exact_same_name_versions_that_download_serves(gateway):
    versions = {share(gateway, n, f"version {n}\n".encode()): f"version {n}\n".encode() for n in (1, 2)}
    reader = AuthorityConnection(gateway.authority, object(), {"user_id": "alice", "capabilities": ["session:read"]})
    assert "groups.attachment.list" in call(reader, "groups.capabilities")["result"]["methods"]

    first = call(reader, "groups.attachment.list", room_id="room", limit=1)["result"]
    second = call(reader, "groups.attachment.list", room_id="room", limit=1, cursor=first["next_cursor"])["result"]

    assert first["room_id"] == "room" and first["has_more"] and not second["has_more"]
    assert first["authority"] == {"gateway_id": hosted_rooms.local_authority_gateway_id(), "epoch": 1}
    items = first["items"] + second["items"]
    assert [(item["event_id"], item["attachment_id"]) for item in items] == list(reversed(versions))
    assert {item["name"] for item in items} == {"report.txt"}
    assert all(item["producer"] == {"kind": "user", "id": "desktop", "label": "You"} for item in items)
    for item in items:
        saved = call(reader, "groups.attachment.download", room_id="room", event_id=item["event_id"],
                     attachment_id=item["attachment_id"])["result"]
        data = base64.b64decode(saved["data_base64"])
        assert data == versions[item["event_id"], item["attachment_id"]]
        assert saved["sha256"] == hashlib.sha256(data).hexdigest() and saved["size"] == item["size"] == len(data)
    assert reason(call(reader, "groups.attachment.upload", room_id="room", upload_id="denied", kind="file",
                       name="x.txt", mime="text/plain", data_base64="eA==")) == "permission_denied"


def test_list_is_authorized_like_download_and_refuses_other_callers(gateway, monkeypatch):
    event_id, attachment_id = share(gateway, 1, b"private\n")
    original, checks = gateway.service.authorize_room, []

    def counted(*args, **kwargs):
        checks.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(gateway.service, "authorize_room", counted)
    call(gateway.owner, "groups.attachment.download", room_id="room", event_id=event_id, attachment_id=attachment_id)
    download_checks, checks[:] = len(checks), []
    assert len(call(gateway.owner, "groups.attachment.list", room_id="room")["result"]["items"]) == 1
    assert len(checks) == download_checks

    stranger = AuthorityConnection(gateway.authority, object(), {"user_id": "bob"})
    assert reason(call(stranger, "groups.attachment.list", room_id="room")) == "permission_denied"
    blind = AuthorityConnection(gateway.authority, object(), {"user_id": "alice", "capabilities": []})
    assert reason(call(blind, "groups.attachment.list", room_id="room")) == "permission_denied"
    for params in ({"room_id": "room", "authority_epoch": 1}, {"room_id": "room", "recipient_member_id": "writer"},
                   {"room_id": "room", "limit": 33}, {"room_id": "room", "query": 7}, {}):
        assert reason(call(gateway.owner, "groups.attachment.list", **params)) == "invalid_params"
    assert reason(call(gateway.owner, "groups.attachment.list", room_id="room", profile="other")) == "profile_mismatch"

    assert "result" in call(gateway.owner, "groups.disband", room_id="room", cancel_id="files-gone")
    assert reason(call(gateway.owner, "groups.attachment.list", room_id="room")) == "invalid_params"


def test_a_cursor_for_another_listing_has_its_own_reason(gateway):
    for n in range(3):
        share(gateway, n, f"file {n}\n".encode(), name=f"report-{n}.txt")
    page = call(gateway.owner, "groups.attachment.list", room_id="room", limit=1, query="report")["result"]
    refused = call(gateway.owner, "groups.attachment.list", room_id="room", limit=1, query="other",
                   cursor=page["next_cursor"])
    assert reason(refused) == catalog.CatalogCursorError.reason == "attachment_cursor_invalid"


def test_files_need_the_running_room_driver_like_download(gateway, monkeypatch):
    share(gateway, 1, b"file\n")
    monkeypatch.setattr(gateway.service.runtime, "status", lambda: {"running": False, "stopping": False})
    assert reason(call(gateway.owner, "groups.attachment.list", room_id="room")) == "runtime_coordination_required"


def test_slow_browsing_never_blocks_stop(gateway, monkeypatch):
    share(gateway, 1, b"file\n")
    entered, release, original = threading.Event(), threading.Event(), catalog.list_published

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(catalog, "list_published", held)
    with ThreadPoolExecutor(max_workers=2) as pool:
        listing = pool.submit(call, gateway.owner, "groups.attachment.list", room_id="room")
        try:
            assert entered.wait(10)
            stop = pool.submit(call, gateway.owner, "groups.stop", room_id="room", cancel_id="while-browsing")
            assert "result" in stop.result(timeout=10)
        finally:
            release.set()
        assert len(listing.result(timeout=10)["result"]["items"]) == 1
