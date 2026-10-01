"""A named Bot's shared files across the owner transport.

A named profile keeps its turn's output in its own outbox. The room owner
reads, acknowledges or discards it through three operations on the private
owner transport. As for every transport operation, the named profile first has
the room owner attest the exact request: the attestation here binds the
output's scope, manifest and the owner's recorded disposition, so the named
profile serves bytes, ACKs or discards only what the room owner asked for.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
import json

from gateway.hosted_room_artifacts import (
    RoomArtifactError,
    RoomArtifactOutbox,
    RoomArtifactScope,
    output_store_exists,
    terminal_artifact_manifest,
    validate_terminal_artifact_manifest,
)
from hermes_state_runtime import RuntimeStoreError

OUTPUT_OPERATIONS = frozenset({"output_export", "output_ack", "output_discard"})
# One chunk per private-socket exchange, like attachment input (see the transport's _CHUNK_BYTES).
CHUNK_BYTES = 360 * 1024
_COMMON = frozenset({"task", "execution_generation", "artifact_scope", "manifest_digest"})
_EXTRA = {
    "output_export": frozenset({"artifact_id", "offset"}),
    "output_ack": frozenset({"artifact_ids", "message_event_id"}),
    "output_discard": frozenset(),
}


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def action_digest(operation, params) -> str:
    """What the room owner attests and the named profile re-derives for one request."""
    return hashlib.sha256(_canonical({"operation": operation, "params": params}).encode()).hexdigest()


def _request(operation, params):
    """Validate one output request; returns (task identity, generation, scope)."""
    from gateway.hosted_room_driver import TaskIdentity

    if (operation not in OUTPUT_OPERATIONS or not isinstance(params, dict)
            or set(params) != _COMMON | _EXTRA[operation]):
        raise RuntimeStoreError("invalid_params")
    try:
        identity = TaskIdentity(**params["task"])
        scope = RoomArtifactScope.from_mapping(params["artifact_scope"])
    except (TypeError, ValueError) as exc:
        raise RuntimeStoreError("invalid_params") from exc
    generation = params["execution_generation"]
    digest = params["manifest_digest"]
    if (type(generation) is not int or generation != scope.execution_generation
            or asdict(identity) != params["task"]
            or (identity.room_id, identity.task_id) != (scope.room_id, scope.task_id)
            or not isinstance(digest, str) or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)):
        raise RuntimeStoreError("invalid_params")
    if operation == "output_export":
        if (not isinstance(params["artifact_id"], str) or type(params["offset"]) is not int
                or params["offset"] < 0):
            raise RuntimeStoreError("invalid_params")
    elif operation == "output_ack":
        ids = params["artifact_ids"]
        if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                or any(not isinstance(item, str) for item in ids)
                or params["message_event_id"] != "dmessage:" + scope.task_id.removeprefix("dtask:")):
            raise RuntimeStoreError("invalid_params")
    return identity, generation, scope


# ---------------------------------------------------------------------- room owner side
def attest_output_scope(service, room_id, member_id, profile, params):
    """Confirm a running attempt of this member and name the scope its output takes."""
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_room_driver import TaskIdentity

    if set(params) != {"task", "execution_generation", "_target_home"}:
        raise RuntimeStoreError("invalid_params")
    try:
        identity = TaskIdentity(**params["task"])
    except TypeError as exc:
        raise RuntimeStoreError("invalid_params") from exc
    generation = params["execution_generation"]
    if identity.room_id != room_id or type(generation) is not int or generation < 1:
        raise RuntimeStoreError("permission_denied")
    task = next((t for t in tasks.list_tasks(service.db_path, room_id=room_id)
                 if t["identity"] == identity and t["execution_generation"] == generation), None)
    if (task is None or task["status"] != "running"
            or task["payload"].get("target_member_id", task["payload"]["target_profile"]) != member_id
            or task["payload"]["target_profile"] != profile):
        raise RuntimeStoreError("permission_denied")
    gateway, epoch = service._owned_authority(room_id)
    return RoomArtifactScope.from_mapping(dict(
        room_id=room_id, task_id=identity.task_id, execution_generation=generation,
        member_id=member_id, target_profile=profile, home_install_id=gateway,
        target_install_id=gateway, authority_gateway_id=gateway, authority_epoch=epoch)).as_mapping()


def attest_output_action(service, room_id, member_id, profile, operation, params):
    """Authorize one output request against the room owner's recorded disposition."""
    from gateway.session_hosted_output_publication import OBLIGATIONS, obligations_exist

    request = {key: value for key, value in params.items() if key != "_target_home"}
    identity, generation, scope = _request(operation, request)
    gateway, epoch = service._owned_authority(room_id)
    if ((scope.room_id, scope.member_id, scope.target_profile) != (room_id, member_id, profile)
            or (scope.authority_gateway_id, scope.authority_epoch) != (gateway, epoch)):
        raise RuntimeStoreError("permission_denied")
    with service.authority.db._read_ctx() as conn:
        row = conn.execute(
            f"SELECT * FROM {OBLIGATIONS} WHERE room_id=? AND task_id=? AND execution_generation=?",
            (room_id, identity.task_id, generation)).fetchone() if obligations_exist(conn) else None
    expected = {"ack": {"output_export", "output_ack"}, "discard": {"output_discard"}}
    if (row is None or row["state"] != "pending" or operation not in expected[row["operation"]]
            or row["scope_json"] != _canonical(scope.as_mapping())
            or row["identity_json"] != _canonical(asdict(identity))
            or not row["manifest_json"]
            or json.loads(row["manifest_json"]).get("manifest_digest") != request["manifest_digest"]):
        raise RuntimeStoreError("permission_denied")
    if operation == "output_ack":
        manifest = json.loads(row["manifest_json"])
        published = service._published_files(scope, manifest)
        if published is None or request["artifact_ids"] != [item["artifact_id"] for item in published[0]]:
            raise RuntimeStoreError("permission_denied")
    return {"action_digest": action_digest(operation, request)}


def read_exported_item(rpc, params, manifest, artifact_id):
    """Reassemble one named output file and verify it against the reported manifest."""
    items = validate_terminal_artifact_manifest(manifest)
    item = next((entry for entry in items if entry["artifact_id"] == artifact_id), None)
    if item is None:
        raise RoomArtifactError("Group Chat output file is unavailable")
    data = bytearray()
    while len(data) < item["size"]:
        chunk = rpc.output_export(**params, artifact_id=artifact_id, offset=len(data))
        if (type(chunk) is not dict or set(chunk) != {"artifact_id", "offset", "size", "sha256", "data_base64"}
                or chunk["artifact_id"] != artifact_id or chunk["offset"] != len(data)
                or chunk["size"] != item["size"] or chunk["sha256"] != item["sha256"]):
            raise RoomArtifactError("Group Chat output chunk changed")
        try:
            raw = base64.b64decode(chunk["data_base64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise RoomArtifactError("Group Chat output chunk is invalid") from exc
        if len(raw) != min(CHUNK_BYTES, item["size"] - len(data)):
            raise RoomArtifactError("Group Chat output chunk changed")
        data.extend(raw)
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise RoomArtifactError("Group Chat output bytes changed")
    return item, bytes(data)


# ---------------------------------------------------------------------- named profile side
def serve_output_operation(authority, binding, session_id, principal_id, operation, params, attested):
    """Export, ACK or discard this profile's own output, exactly as the room owner attested."""
    if attested.get("action_digest") != action_digest(operation, params):
        raise RuntimeStoreError("permission_denied")
    identity, generation, scope = _request(operation, params)
    selector = binding["selector"]
    if (scope.room_id, scope.member_id, scope.target_profile) != (
            selector["room_id"], selector["member_id"], selector["profile"]):
        raise RuntimeStoreError("permission_denied")
    request_id = "hosted:" + json.dumps([asdict(identity), generation], sort_keys=True, separators=(",", ":"))
    with authority.db._read_ctx() as conn:
        admission = conn.execute(
            "SELECT status FROM session_admissions WHERE target_session_id=? AND request_id=? AND principal_id=?",
            (session_id, request_id, principal_id)).fetchone()
        if admission is not None and admission["status"] != "terminal":
            raise RuntimeStoreError("permission_denied")  # the producing turn is still open
        if not output_store_exists(conn):
            if operation == "output_discard":
                return {"discarded": True, "removed": 0}
            raise RuntimeStoreError("permission_denied")
    outbox = RoomArtifactOutbox(authority.db.db_path)
    try:
        if operation == "output_discard":
            # Only open copies are removed; a published receipt is never touched.
            return {"discarded": True, "removed": outbox.discard_durably(scope)}
        stored = outbox.scope_manifest(scope)
        if operation == "output_ack" and not stored and outbox.retirement_complete(scope):
            # The ACK committed earlier and its short-lived receipts have since expired.
            return {"acknowledged": True, "changed": 0}
        reported = terminal_artifact_manifest(stored)
        if reported is None or reported["manifest_digest"] != params["manifest_digest"]:
            raise RoomArtifactError("Group Chat output manifest changed")
        if operation == "output_ack":
            if params["artifact_ids"] != [item["artifact_id"] for item in stored]:
                raise RoomArtifactError("Group Chat output acknowledgement changed")
            changed = outbox.acknowledge(scope, params["artifact_ids"],
                                         message_event_id=params["message_event_id"])
            return {"acknowledged": True, "changed": changed}
        if outbox.list(scope) != stored:
            raise RoomArtifactError("Group Chat output is no longer open")
        item, data = outbox.read_range(scope, params["artifact_id"], offset=params["offset"], length=CHUNK_BYTES)
        return {"artifact_id": item["artifact_id"], "offset": params["offset"], "size": item["size"],
                "sha256": item["sha256"], "data_base64": base64.b64encode(data).decode("ascii")}
    except RoomArtifactError as exc:
        raise RuntimeStoreError("permission_denied") from exc
