"""A named Bot's shared files through real served owners and the private owner transport."""

from __future__ import annotations

import base64
import json
import time

import pytest

from tests.gateway.test_hosted_mux_runtime import mux  # noqa: F401 - real owners and private socket


def _wait(predicate, *, timeout=15, message="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    pytest.fail(f"timed out waiting for {message}")


def _start(mux, monkeypatch, handle):
    from gateway.session_authority import SessionAuthority
    from gateway.session_authorities import owner_scope
    from gateway.session_hosted_service import ensure_hosted_service

    runner, homes, _, call = mux
    runner._handle_message = handle
    for authority in runner.session_authorities:
        monkeypatch.setattr(authority, "_schedule", SessionAuthority._schedule.__get__(authority))
    call(ensure_hosted_service(runner))
    source = runner.session_authorities.require(homes["default"])
    target = runner.session_authorities.require(homes["beta"])
    service = source.hosted_room_service
    service.runtime.poll_interval_seconds = service.runtime.active_poll_interval_seconds = 0.05
    with owner_scope(source):
        service.authorize_room("alice", "room", create=True)
        service.create_room(room_id="room", name="Files", members=[
            {"member_id": "host", "profile": "default", "handle": "host"},
            {"member_id": "helper", "profile": "beta", "handle": "helper"}])
        service.send(room_id="room", event_id="input",
                     payload={"text": "@helper Share the report", "thread_id": "thread"})
    return source, target, service


def _events(source, kind=None):
    from gateway import hosted_rooms
    return [e for e in hosted_rooms.read_events(source.db.db_path, room_id="room")["events"]
            if kind is None or e["kind"] == kind]


def _rows(authority, table):
    with authority.db._read_ctx() as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is None:
            return []
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]


def _sharing(path, *, fail=None, seen=None):
    async def handle(event):
        from gateway.session_hosted_output import current_output_binding
        from gateway.session_results import execution_result
        from tools import hosted_room_artifact  # noqa: F401
        from tools.registry import registry
        binding = current_output_binding()
        if seen is not None:
            seen.append(binding)
        assert binding is not None and binding.room_local is False
        result = json.loads(registry.dispatch("share_group_file", {"path": str(path)}))
        assert result["ok"] is True, result
        if fail is not None:
            raise fail
        execution_result.get()["result"] = {"final_response": "Report ready.", "messages": [], "completed": True}
        return "Report ready."
    return handle


@pytest.mark.live_system_guard_bypass
def test_a_named_bots_file_crosses_the_owner_transport_and_is_published_once(mux, monkeypatch):
    from gateway.session_authorities import owner_scope
    from gateway.session_contract import Principal
    from gateway.session_hosted_attachments import download

    _runner, homes, _, _call = mux
    path = homes["beta"] / "report.txt"
    path.write_bytes(b"named report bytes\n")
    seen = []
    source, target, service = _start(mux, monkeypatch, _sharing(path, seen=seen))
    message = _wait(lambda: next(iter(_events(source, "message.member")), None), message="the member message")
    attachment, = message["payload"]["attachments"]
    assert message["actor"]["id"] == "helper" and message["payload"]["text"] == "Report ready."
    with owner_scope(source):
        saved = download(service, Principal("alice", source.profile_id, frozenset({"session:read"}), "viewer"),
                         dict(room_id="room", event_id=message["event_id"], attachment_id=attachment["attachment_id"]))
    assert base64.b64decode(saved["data_base64"]) == b"named report bytes\n"
    obligation = _wait(lambda: next((o for o in _rows(source, "hosted_room_output_obligations")
                                     if o["state"] == "completed"), None), message="the completed ACK")
    assert obligation["operation"] == "ack"
    # The named profile kept the private copy in its own store and acknowledged it exactly once.
    row, = _rows(target, "hosted_room_output_artifacts")
    assert row["acknowledged_at"] is not None and row["blob_reclaimed_at"] is not None
    assert _rows(source, "hosted_room_output_artifacts") == []
    assert seen[0].scope.target_profile == "beta" and seen[0].active is False
    assert len(_events(source, "message.member")) == 1

    # Every later output request needs a fresh attestation from the room owner: none is pending now.
    from gateway.session_hosted_transport import HostedRoomOwnerRPC
    from hermes_state_runtime import RuntimeStoreError
    task = next(t for t in __import__("gateway.hosted_room_driver", fromlist=["list_tasks"]).list_tasks(
        source.db.db_path, room_id="room") if t["status"] == "settled")
    rpc = HostedRoomOwnerRPC(home=homes["beta"], source_home=homes["default"], room_id="room",
                             member_id="helper", profile="beta")
    from dataclasses import asdict
    params = {"task": asdict(task["identity"]), "execution_generation": task["execution_generation"],
              "artifact_scope": task["result"]["artifact_scope"],
              "manifest_digest": task["result"]["artifacts"]["manifest_digest"]}
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        rpc.output_export(**params, artifact_id=row["artifact_id"], offset=0)
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        rpc.output_discard(**params)


@pytest.mark.live_system_guard_bypass
def test_a_named_bots_failed_turn_retires_its_file_on_its_own_profile(mux, monkeypatch):
    _runner, homes, _, _call = mux
    path = homes["beta"] / "report.txt"
    path.write_bytes(b"never published\n")
    source, target, service = _start(mux, monkeypatch, _sharing(path, fail=RuntimeError("model down")))
    _wait(lambda: [e for e in _events(source) if e["kind"] == "turn.failed"], message="the failed terminal")
    assert not _events(source, "message.member")
    assert _rows(target, "hosted_room_output_artifacts") == []
    assert _rows(source, "hosted_room_output_obligations") == []


@pytest.mark.live_system_guard_bypass
def test_a_lost_named_ack_replays_through_the_transport(mux, monkeypatch):
    from gateway.hosted_room_artifacts import RoomArtifactOutbox

    _runner, homes, _, _call = mux
    path = homes["beta"] / "report.txt"
    path.write_bytes(b"named report bytes\n")
    original = RoomArtifactOutbox.acknowledge
    calls = []

    def lose_first_response(self, scope, ids, *, message_event_id):
        changed = original(self, scope, ids, message_event_id=message_event_id)
        calls.append(changed)
        if len(calls) == 1:
            raise OSError("response lost after the named ACK committed")
        return changed
    monkeypatch.setattr(RoomArtifactOutbox, "acknowledge", lose_first_response)
    source, target, service = _start(mux, monkeypatch, _sharing(path))
    pending = _wait(lambda: next((o for o in _rows(source, "hosted_room_output_obligations")
                                  if o["attempts"] >= 1), None), message="the failed first ACK")
    assert pending["state"] == "pending" and pending["reason_code"] == "transient"
    assert len(_events(source, "message.member")) == 1
    monkeypatch.setattr(service, "_output_clock", lambda: pending["next_attempt_at"] + 1)
    service.runtime.wakeup()
    done = _wait(lambda: next((o for o in _rows(source, "hosted_room_output_obligations")
                               if o["state"] == "completed"), None), message="the replayed ACK")
    assert done["operation"] == "ack" and calls == [1, 0]
    assert len(_events(source, "message.member")) == 1


@pytest.mark.live_system_guard_bypass
def test_a_named_profiles_interrupted_cleanup_finishes_when_its_service_starts(mux, monkeypatch):
    from pathlib import Path
    from gateway.hosted_room_artifacts import RoomArtifactOutbox, RoomArtifactScope
    from gateway.session_hosted_service import ensure_hosted_service

    runner, homes, _, call = mux
    target = runner.session_authorities.require(homes["beta"])
    scope = RoomArtifactScope.from_mapping(SCOPE)
    outbox = RoomArtifactOutbox(target.db.db_path)
    outbox.put_bytes(scope=scope, data=b"never published\n", source_name="report.txt")
    unlink = Path.unlink

    def busy(self, *args, **kwargs):
        if "hosted-room-artifact-outbox" in str(self):
            raise OSError("disk busy")
        return unlink(self, *args, **kwargs)
    with monkeypatch.context() as interrupted, pytest.raises(OSError, match="disk busy"):
        interrupted.setattr(Path, "unlink", busy)
        outbox.discard_durably(scope)
    row, = _rows(target, "hosted_room_output_artifacts")
    assert row["cleanup_required_at"] is not None  # the intent committed, the bytes did not go
    call(ensure_hosted_service(runner))
    assert _rows(target, "hosted_room_output_artifacts") == []
    assert not any((target.db.db_path.parent / "hosted-room-artifact-outbox" / "blobs").iterdir())


# ---------------------------------------------------------------- protocol edges, without a gateway
SCOPE = {"room_id": "room", "task_id": "dtask:abc", "execution_generation": 1, "member_id": "helper",
         "target_profile": "beta", "home_install_id": "install:x", "target_install_id": "install:x",
         "authority_gateway_id": "install:x", "authority_epoch": 1}
TASK = {"room_id": "room", "task_id": "dtask:abc", "thread_id": "thread", "turn_id": "turn"}


def _named_outbox(tmp_path):
    from types import SimpleNamespace
    from gateway.hosted_room_artifacts import RoomArtifactOutbox, RoomArtifactScope
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    outbox = RoomArtifactOutbox(db.db_path)
    scope = RoomArtifactScope.from_mapping(SCOPE)
    path = tmp_path / "report.txt"
    path.write_bytes(b"named report\n")
    from gateway.hosted_room_artifacts import open_room_artifact_path, terminal_artifact_manifest
    with open_room_artifact_path(path) as (opened, descriptor):
        item = outbox.put_open_file(scope=scope, descriptor=descriptor, source_name=opened.name)
    manifest = terminal_artifact_manifest([item])
    authority = SimpleNamespace(db=db, profile_id=str(tmp_path))
    binding = {"selector": {"room_id": "room", "member_id": "helper", "profile": "beta"}}
    params = {"task": dict(TASK), "execution_generation": 1, "artifact_scope": dict(SCOPE),
              "manifest_digest": manifest["manifest_digest"]}
    return db, outbox, scope, item, authority, binding, params


def test_the_named_profile_serves_only_exactly_attested_requests(tmp_path):
    from gateway.session_hosted_output_owner import action_digest, serve_output_operation
    from hermes_state_runtime import RuntimeStoreError

    db, outbox, scope, item, authority, binding, params = _named_outbox(tmp_path)
    try:
        export = {**params, "artifact_id": item["artifact_id"], "offset": 0}
        ok = {"action_digest": action_digest("output_export", export)}
        chunk = serve_output_operation(authority, binding, "session", "principal", "output_export", export, ok)
        assert base64.b64decode(chunk["data_base64"]) == b"named report\n"
        for attested in ({}, {"action_digest": action_digest("output_export", {**export, "offset": 1})}):
            with pytest.raises(RuntimeStoreError, match="permission_denied"):
                serve_output_operation(authority, binding, "session", "principal", "output_export", export, attested)
        other = {**binding, "selector": {**binding["selector"], "member_id": "someone-else"}}
        with pytest.raises(RuntimeStoreError, match="permission_denied"):
            serve_output_operation(authority, other, "session", "principal", "output_export", export, ok)
        stale = {**params, "manifest_digest": "0" * 64, "artifact_id": item["artifact_id"], "offset": 0}
        with pytest.raises(RuntimeStoreError, match="permission_denied"):
            serve_output_operation(authority, binding, "session", "principal", "output_export", stale,
                                   {"action_digest": action_digest("output_export", stale)})
        ack = {**params, "artifact_ids": [item["artifact_id"]], "message_event_id": "dmessage:abc"}
        result = serve_output_operation(authority, binding, "session", "principal", "output_ack", ack,
                                        {"action_digest": action_digest("output_ack", ack)})
        assert result == {"acknowledged": True, "changed": 1}
        # Published and acknowledged: never exported again; a discard finds nothing open to remove.
        with pytest.raises(RuntimeStoreError, match="permission_denied"):
            serve_output_operation(authority, binding, "session", "principal", "output_export", export, ok)
        assert serve_output_operation(authority, binding, "session", "principal", "output_discard", params,
                                      {"action_digest": action_digest("output_discard", params)}) == {
            "discarded": True, "removed": 0}
        replay = {"action_digest": action_digest("output_ack", ack)}
        assert serve_output_operation(authority, binding, "session", "principal", "output_ack", ack,
                                      replay)["changed"] == 0
        assert [row["artifact_id"] for row in outbox.scope_manifest(scope)] == [item["artifact_id"]]  # receipts kept
        # After the receipts expire, the retired generation still answers an ACK replay.
        from gateway.hosted_room_artifacts import ACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS
        outbox.prune_acknowledged_receipts(now=__import__("time").time() + ACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS + 1)
        assert outbox.scope_manifest(scope) == []
        assert serve_output_operation(authority, binding, "session", "principal", "output_ack", ack,
                                      replay) == {"acknowledged": True, "changed": 0}
    finally:
        db.close()


def test_the_named_profile_refuses_output_of_a_turn_that_is_still_running(tmp_path):
    import json as _json
    from gateway.session_hosted_output_owner import action_digest, serve_output_operation
    from hermes_state_runtime import RuntimeStoreError, admit_session_input, begin_runtime_epoch

    db, outbox, scope, item, authority, binding, params = _named_outbox(tmp_path)
    try:
        epoch = begin_runtime_epoch(db, instance_id="test")
        request_id = "hosted:" + _json.dumps([TASK, 1], sort_keys=True, separators=(",", ":"))
        db.create_session("session", source="bot_room")
        admit_session_input(db, epoch=epoch, principal_id="principal", session_id="session",
                            request_id=request_id, payload={"text": "go"})
        discard = dict(params)
        with pytest.raises(RuntimeStoreError, match="permission_denied"):
            serve_output_operation(authority, binding, "session", "principal", "output_discard", discard,
                                   {"action_digest": action_digest("output_discard", discard)})
        assert len(outbox.list(scope)) == 1
    finally:
        db.close()


def test_the_room_owner_rejects_any_chunk_that_does_not_match_the_manifest():
    from gateway.hosted_room_artifacts import RoomArtifactError, terminal_artifact_manifest
    from gateway.session_hosted_output_owner import read_exported_item

    data = b"named report\n"
    item = {"artifact_id": "rart_" + "a" * 32, "kind": "file", "name": "report.txt", "size": len(data),
            "mime": "text/plain", "sha256": __import__("hashlib").sha256(data).hexdigest()}
    manifest = terminal_artifact_manifest([item])
    good = {"artifact_id": item["artifact_id"], "offset": 0, "size": len(data), "sha256": item["sha256"],
            "data_base64": base64.b64encode(data).decode()}

    class Peer:
        def __init__(self, chunk):
            self.chunk = chunk

        def output_export(self, **params):
            return dict(self.chunk)

    assert read_exported_item(Peer(good), {}, manifest, item["artifact_id"]) == (item, data)
    for broken in ({**good, "offset": 1}, {**good, "sha256": "0" * 64}, {**good, "extra": True},
                   {**good, "data_base64": base64.b64encode(b"other bytes!!").decode()},
                   {**good, "data_base64": "not base64"}):
        with pytest.raises(RoomArtifactError):
            read_exported_item(Peer(broken), {}, manifest, item["artifact_id"])
    with pytest.raises(RoomArtifactError, match="unavailable"):
        read_exported_item(Peer(good), {}, manifest, "rart_" + "b" * 32)
