"""Peer document authority, producer budgets and truthful terminal evidence."""
import base64
from contextlib import asynccontextmanager
import hashlib
import json
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from gateway.config import Platform, PlatformConfig
from gateway.hosted_room_artifacts import RoomArtifactError, terminal_artifact_manifest
from gateway.hosted_room_peer import HostedMemberDispatch, decode_room_grant, issue_room_grant
from gateway.hosted_room_peer_output import OUTPUT_CAPABILITY, output_scope
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_peer_output import handle
from gateway.platforms.api_server_room_grants import _local_room_catalog
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from gateway.platforms.api_server_run_scope import room_run_scope_key
from gateway.session_api import bind_api_session
from gateway.session_hosted_output import output_binding, terminal_output_fields
from gateway.session_peer_output import PeerDocumentOutbox, accepted_dispatch_digest, receipt_fields
from hermes_state_runtime import admit_session_input, claim_session_input, settle_session_input
from tests.gateway.fixtures.hosted_output import owner
from tests.gateway.test_hosted_room_artifacts import _scope
from tests.tui_gateway.test_hosted_room_peer_http import _dispatch


class Request(dict):
    """Transport stub; grant validation and all accepting state are real."""
    def __init__(self, body, *, grant=None, operation=None, run_id="run-output"):
        super().__init__()
        if grant is not None:
            self["verified_room_grant"] = grant
        self.headers = {"Authorization": "Bearer ordinary-api-key"}
        self.match_info = {"run_id": run_id, "operation": operation}
        self.body = body

    async def json(self):
        return self.body


@asynccontextmanager
async def admitted(tmp_path, monkeypatch, *, consent=True, authenticated=True):
    async with owner(tmp_path, monkeypatch) as (authority, _, runner):
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "ordinary-api-key"}))
        adapter._run_idempotency_store.close()
        adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "runs.db"))
        adapter.gateway_runner = runner
        runner.adapters[Platform.API_SERVER] = adapter
        try:
            install = hosted_rooms.local_authority_gateway_id()
            _, catalog = _local_room_catalog(adapter, "default", install)
            dispatch = HostedMemberDispatch.from_mapping(_dispatch(
                authority_gateway_id="install-home", target_install_id=install, target_profile="default",
                task_id="dtask:output", capability_digest=catalog["catalog_digest"],
                execution_policy_digest=catalog["execution_policy"]["policy_digest"],
                **({"document_output": dict(OUTPUT_CAPABILITY)} if consent else {})))
            identity = [dispatch.home_install_id, dispatch.room_id, dispatch.member_id, dispatch.target_profile]
            session_id = "room_" + hashlib.sha256("\0".join(identity).encode()).hexdigest()[:32] if consent else "ordinary"
            ref = bind_api_session(authority, session_id, hosted_dispatch=dispatch.as_mapping() if consent else None)
            settings = {"room_dispatch": dispatch.as_mapping() if consent else None}
            payload = {"text": dispatch.prompt, "api_turn_v1": {"history": None, "settings": settings,
                "run_owner_scope": room_run_scope_key(dispatch.as_mapping()) if authenticated else "0" * 64}}
            admit_session_input(authority.db, epoch=authority.epoch, principal_id="api", session_id=session_id,
                                request_id="run-output", payload=payload)
            row = claim_session_input(authority.db, epoch=authority.epoch, session_id=session_id)
            authority.sessions[session_id].event_stream.execution = {
                "authority_epoch": authority.epoch, "execution_generation": row["generation"],
                "admission_id": row["admission_id"]}
            yield SimpleNamespace(authority=authority, adapter=adapter, ref=ref, row=row,
                                  dispatch=dispatch, scope=output_scope(dispatch))
        finally:
            adapter._run_idempotency_store.close()


@pytest.mark.parametrize("boundary", ["empty", "image", "file-size", "batch-size", "count"])
def test_refused_share_has_no_custody_effect_and_exact_replay_stays_allowed(tmp_path, boundary):
    outbox, scope = PeerDocumentOutbox(tmp_path / "state.db"), _scope()
    first_size = 5_000_000 if boundary == "batch-size" else 1
    first = outbox.put_bytes(scope=scope, data=b"a" * first_size, source_name="report.txt")
    if boundary == "batch-size":
        outbox.put_bytes(scope=scope, data=b"b" * 1_000_000, source_name="appendix.txt")
    elif boundary == "count":
        for index in range(7):
            outbox.put_bytes(scope=scope, data=b"x", source_name=f"note-{index}.txt")
    before, blobs = outbox.list(scope), set(outbox.blob_root.iterdir())
    data, name = {
        "empty": (b"", "empty.txt"),
        "image": (b"\x89PNG\r\n\x1a\n" + b"\0" * 32, "image.png"),
        "file-size": (b"x" * 5_000_001, "large.txt"),
        "batch-size": (b"x", "extra.txt"),
        "count": (b"x", "ninth.txt"),
    }[boundary]
    with pytest.raises(RoomArtifactError):
        outbox.put_bytes(scope=scope, data=data, source_name=name)
    assert outbox.list(scope) == before
    assert set(outbox.blob_root.iterdir()) == blobs
    assert outbox.put_bytes(scope=scope, data=b"a" * first_size, source_name="report.txt") == first
    assert outbox.read(scope, first["artifact_id"])[1] == b"a" * first_size


@pytest.mark.asyncio
@pytest.mark.parametrize("consent,authenticated,allowed", [(False, True, False), (True, False, False), (True, True, True)])
async def test_only_authenticated_canonical_consent_can_bind_or_project_output(tmp_path, monkeypatch, consent, authenticated, allowed):
    async with admitted(tmp_path, monkeypatch, consent=consent, authenticated=authenticated) as p:
        binding = await output_binding(p.authority, p.ref, p.row)
        assert (binding is not None) is allowed
        from gateway.hosted_room_peer_output import dispatch_digest
        assert accepted_dispatch_digest(p.row) == (dispatch_digest(p.dispatch) if allowed else None)
        if allowed:
            artifact = binding._outbox().put_bytes(scope=p.scope, data=b"%PDF-1.4\nreport", source_name="report.pdf")
            assert artifact["kind"] == "pdf"
            manifest = terminal_artifact_manifest([artifact])
            result = {"artifacts": manifest, "artifact_scope": p.scope.as_mapping()}
            assert receipt_fields(p.row, result) == result
        else:
            # Valid-looking metadata alone cannot turn an ordinary/forged admission into peer authority.
            result = {"peer_output_empty": p.scope.as_mapping()}
            assert receipt_fields(p.row, result) == {}
            with p.authority.db._read_ctx() as conn:
                assert conn.execute("SELECT name FROM sqlite_master WHERE name='hosted_room_output_artifacts'").fetchone() is None
        ordinary = {"input": "An ordinary API turn"}
        normalized, error = await p.adapter._normalize_room_dispatch(Request(ordinary), ordinary)
        assert normalized is ordinary and error is None
        assert terminal_output_fields(None) == {}
        body = {"input": p.dispatch.prompt, "hosted_room_dispatch": {
            **p.dispatch.as_mapping(), "document_output": dict(OUTPUT_CAPABILITY)}}
        before = len(p.authority.sessions)
        _, refused = await p.adapter._normalize_room_dispatch(Request(body), body)
        assert refused.status == 403
        assert json.loads(refused.text)["error"]["code"] == "invalid_room_output"
        assert len(p.authority.sessions) == before
        with p.authority.db._read_ctx() as conn:
            assert conn.execute("SELECT COUNT(*) FROM session_admissions").fetchone()[0] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,permissions,allowed", [
    ("read", ("status",), True), ("read", ("stop",), False),
    ("ack", ("status",), False), ("ack", ("status", "stop"), True),
    ("discard", ("status",), False), ("discard", ("status", "stop"), True),
])
async def test_observer_can_read_but_only_stop_authority_can_retire_output(tmp_path, monkeypatch, operation, permissions, allowed):
    async with admitted(tmp_path, monkeypatch) as p:
        binding = await output_binding(p.authority, p.ref, p.row)
        outbox = binding._outbox()
        artifact = outbox.put_bytes(scope=p.scope, data=b"Retained report\n", source_name="report.txt")
        manifest = terminal_artifact_manifest(outbox.list(p.scope))
        settle_session_input(p.authority.db, epoch=p.authority.epoch, admission_id=p.row["admission_id"],
            generation=p.row["generation"], outcome="completed", result={"result": {
                "final_response": "Report ready", "artifacts": manifest, "artifact_scope": p.scope.as_mapping()}})
        grant = issue_room_grant(p.adapter._room_grant_secret(), grant_id="observer",
            **{key: getattr(p.dispatch, key) for key in ("room_id", "home_install_id", "authority_gateway_id",
                "authority_epoch", "member_id", "target_install_id", "target_profile", "execution_policy_digest")},
            permissions=permissions)
        claims = decode_room_grant(p.adapter._room_grant_secret(), grant, permission=permissions[0])
        hosted_rooms.reserve_peer_room(p.authority.db.db_path, claims=claims, expires_at=claims["expires_at"])
        body = {"artifact_scope": p.scope.as_mapping(), "manifest_digest": manifest["manifest_digest"]}
        if operation == "read":
            body["artifact_id"] = artifact["artifact_id"]
        elif operation == "ack":
            body.update(artifact_ids=[artifact["artifact_id"]], message_event_id="dmessage:output")
        request = Request(body, grant=grant, operation=operation)
        response = await handle(p.adapter, request)
        assert response.status == (200 if allowed else 409), response.text
        if allowed and operation != "read":
            assert outbox.list(p.scope) == []
            repeated = await handle(p.adapter, request)
            assert repeated.status == 200
            assert json.loads(repeated.text)["changed" if operation == "ack" else "removed"] == 0
        else:
            assert outbox.list(p.scope) == [artifact]
            assert outbox.read(p.scope, artifact["artifact_id"])[1] == b"Retained report\n"
            with p.authority.db._read_ctx() as conn:
                assert conn.execute("SELECT 1 FROM state_meta WHERE key LIKE 'gateway.peer-output-disposition.v1.%'").fetchone() is None
            if allowed:
                assert base64.b64decode(json.loads(response.text)["data_base64"]) == b"Retained report\n"


@pytest.mark.parametrize("conflict", ["artifacts", "artifact_scope"])
def test_empty_output_evidence_cannot_override_file_or_scope_evidence(tmp_path, conflict):
    scope = _scope()
    outbox = PeerDocumentOutbox(tmp_path / "state.db")
    artifact = outbox.put_bytes(scope=scope, data=b"retained", source_name="report.txt")
    evidence = {"peer_output_empty": scope.as_mapping()}
    assert terminal_output_fields(evidence) == evidence
    evidence[conflict] = terminal_artifact_manifest([artifact]) if conflict == "artifacts" else scope.as_mapping()
    assert terminal_output_fields(evidence) == {}
    assert outbox.read(scope, artifact["artifact_id"])[1] == b"retained"
