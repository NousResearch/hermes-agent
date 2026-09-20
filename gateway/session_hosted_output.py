"""Owner-local Files binding reconstructed from a real canonical admission."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import os
from pathlib import Path

from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactOutbox, RoomArtifactScope
from hermes_state_runtime import RuntimeStoreError, _epoch


_OUTPUT = ContextVar("canonical_hosted_output", default=None)


@dataclass
class HostedOutputBinding:
    authority: object
    ref: object
    row: dict
    scope: RoomArtifactScope
    cancel_generation: int
    owner_pid: int
    active: bool = True
    used: bool = False
    task: dict | None = None
    cleanup_pending: bool = False
    cleanup_reason: str | None = None

    def check_write(self, conn, scope):
        if not self.active or os.getpid() != self.owner_pid or scope != self.scope:
            raise RoomArtifactError("Group Chat output producer is no longer active")
        _epoch(conn, self.authority.epoch)
        admission = conn.execute(
            "SELECT status, owner_epoch, generation, principal_id, target_session_id FROM session_admissions WHERE admission_id=?",
            (self.row["admission_id"],),
        ).fetchone()
        if (admission is None or admission["status"] != "started"
                or admission["owner_epoch"] != self.authority.epoch
                or admission["generation"] != self.row["generation"]
                or admission["principal_id"] != self.row["principal_id"]
                or admission["target_session_id"] != self.ref.session_id):
            raise RoomArtifactError("Group Chat output admission changed")
        from gateway.hosted_room_output_fence import require_output_task
        require_output_task(conn, self.scope, self.cancel_generation, status="running")
        from gateway.session_hosted_service import _OWNER
        owner = conn.execute("SELECT value FROM state_meta WHERE key=?", (_OWNER + scope.room_id,)).fetchone()
        if owner is None or owner[0] != self.row["principal_id"]:
            raise RoomArtifactError("Group Chat output owner changed")

    def outbox(self):
        if not self.active or os.getpid() != self.owner_pid:
            raise RoomArtifactError("Group Chat output producer is unavailable")
        self.used = True
        return RoomArtifactOutbox(self.authority.db.db_path, authorize_write=self.check_write)


def current_output_binding():
    binding = _OUTPUT.get()
    if binding is None or not binding.active or binding.owner_pid != os.getpid():
        return None
    return binding


def _is_owner_transport_admission(authority, ref, row):
    from gateway.session_hosted_transport import _BINDING, _principal

    with authority.db._read_ctx() as conn:
        retained = conn.execute("SELECT value FROM state_meta WHERE key=?", (_BINDING + ref.session_id,)).fetchone()
    if retained is None:
        return False
    try:
        transport = json.loads(retained[0])
        if (not isinstance(transport, dict)
                or not isinstance(transport.get("source_home"), str) or not Path(transport["source_home"]).is_absolute()
                or not isinstance(transport.get("owner"), str) or not transport["owner"]
                or transport.get("target_home") != authority.profile_id
                or ref.profile_id != authority.profile_id
                or row.get("target_session_id") != ref.session_id
                or row.get("principal_id") != _principal(authority, transport).subject):
            raise ValueError("transport binding mismatch")
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeStoreError("permission_denied") from exc
    return True


def _binding(authority, ref, row):
    if row.get("principal_id") == "api":
        from gateway.session_peer_output import peer_output_binding
        return peer_output_binding(authority, ref, row)
    if not row.get("request_id", "").startswith("hosted:"):
        return None
    service = getattr(authority, "hosted_room_service", None)
    if service is None or not callable(getattr(service, "check_admission", None)):
        return None
    home = Path(authority.profile_id)
    if not home.is_absolute() or Path(authority.db.db_path).resolve().parent != home.resolve():
        return None
    if home.parent.name == "profiles":
        from gateway.session_hosted_output_rpc import OWNER_OUTPUT_LIFECYCLE_READY
        if not OWNER_OUTPUT_LIFECYCLE_READY:
            return None
    from gateway.session_managed_worker import managed_policy
    if managed_policy(authority, ref) is not None:
        return None
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    if policy is None or policy.source != "bot_room" or "bot_room" not in policy.toolsets:
        return None
    # Retained owner transports have their own preclaim authorizer. Classify
    # that namespace before consulting a colliding local coordinator room.
    if _is_owner_transport_admission(authority, ref, row):
        if home.parent.name == "profiles":
            from gateway.session_hosted_transport import _BINDING
            with authority.db._read_ctx() as conn:
                retained = conn.execute(
                    "SELECT value FROM state_meta WHERE key=?", (_BINDING + ref.session_id,)
                ).fetchone()
            if retained is None:
                return None
            from gateway.session_hosted_output_rpc import owner_output_binding
            return owner_output_binding(authority, ref, row, json.loads(retained[0]))
        return None
    # Named targets require the separate owner-transport consent above.  A local
    # coordinator must never fall through to the collapsing root outbox.
    if home.parent.name == "profiles":
        return None
    # Remote hosted bindings live on the target and lack the coordinator task.
    # They must not infer local authority merely from a hosted request-id prefix.
    try:
        identity, generation = json.loads(row["request_id"][7:])
        room_id = identity["room_id"]
        room = service._room(room_id)
    except (KeyError, TypeError, ValueError):
        return None
    task = service.check_admission(ref, row)
    profile = task["payload"]["target_profile"]
    member_id = task["payload"].get("target_member_id", profile)
    member = next((m for m in room["members"] if m["member_id"] == member_id), None)
    if (member is None or member.get("target", {}).get("kind", "local") != "local"
            or Path(service.profile_homes().get(profile, "")).resolve() != home.resolve()):
        return None
    gateway, epoch = service._owned_authority(room_id)
    scope = RoomArtifactScope.from_mapping(dict(
        room_id=room_id, task_id=task["identity"].task_id, execution_generation=generation,
        member_id=member_id, target_profile=profile, home_install_id=gateway,
        target_install_id=gateway, authority_gateway_id=gateway, authority_epoch=epoch,
    ))
    binding = HostedOutputBinding(
        authority, ref, row, scope, task["cancel_generation"], os.getpid(), task=dict(task)
    )
    with authority.db._read_ctx() as conn:
        binding.check_write(conn, scope)
    return binding


@contextmanager
def hosted_output_scope(authority, ref, row):
    """Bind only owner-local execution; copied tool contexts expire on return."""
    binding = _binding(authority, ref, row)
    token = _OUTPUT.set(binding)
    try:
        yield binding
    finally:
        if binding is not None:
            binding.active = False
        _OUTPUT.reset(token)


def capture_output_result(authority, row, binding):
    if binding is None or not binding.used:
        return
    from gateway.hosted_room_artifacts import terminal_artifact_manifest
    manifest = terminal_artifact_manifest(authority.db.db_path, binding.scope, outbox=binding.outbox())
    if manifest is None:
        return
    saved = authority.pending_results.get(row["admission_id"])
    if saved is None:
        raise RuntimeStoreError("storage_unavailable")
    saved["result"].update(artifacts=manifest, artifact_scope=binding.scope.as_mapping())
    if hasattr(binding, "consent_json"):
        from gateway.session_hosted_output_rpc import capture_owner_output_receipt
        saved["result"]["owner_output_receipt"] = capture_owner_output_receipt(
            authority, row, binding, saved["result"])
    if row.get("principal_id") == "api":
        from gateway.session_peer_output import output_run_binding
        saved["result"]["peer_output_binding"] = output_run_binding(row, saved["result"])


def capture_failed_output(authority, row, binding):
    """Retain one exact cleanup obligation without replacing the producer failure."""
    if binding is None or not binding.used:
        return None
    try:
        if (
            binding.authority is not authority
            or binding.row.get("admission_id") != row.get("admission_id")
            or binding.row.get("generation") != row.get("generation")
        ):
            raise RoomArtifactError("Group Chat output owner changed")
        if hasattr(binding, "consent_json"):
            from gateway.session_hosted_output_rpc import capture_failed_owner_output
            complete = capture_failed_owner_output(authority, row, binding)
            binding.cleanup_pending = not complete
            binding.cleanup_reason = None if complete else "cleanup_pending"
            return binding.cleanup_reason
        service = getattr(authority, "hosted_room_service", None)
        capture = getattr(service, "_capture_failed_output", None)
        if not callable(capture):
            raise RoomArtifactError("Group Chat output owner changed")
        complete = capture(binding)
    except Exception as exc:
        binding.cleanup_pending = True
        binding.cleanup_reason = (
            "owner_unavailable"
            if isinstance(exc, (RoomArtifactError, RuntimeStoreError))
            else "cleanup_unavailable"
        )
        return binding.cleanup_reason
    binding.cleanup_pending = not complete
    binding.cleanup_reason = None if complete else "cleanup_pending"
    return binding.cleanup_reason


def output_receipt_fields(value):
    """Bounded, explicit metadata only; never a path or a client-created scope."""
    if not isinstance(value, dict) or not value.get("artifacts"):
        return {}
    from gateway.hosted_room_artifacts import validate_terminal_artifact_manifest
    validate_terminal_artifact_manifest(value["artifacts"])
    scope = RoomArtifactScope.from_mapping(value.get("artifact_scope") or {})
    fields = {"artifacts": value["artifacts"], "artifact_scope": scope.as_mapping()}
    if value.get("owner_output_receipt") is not None:
        from gateway.session_hosted_output_rpc import validate_owner_output_receipt
        fields["owner_output_receipt"] = validate_owner_output_receipt(
            value["owner_output_receipt"])
    peer_keys = ("peer_run_id", "peer_admission_id", "peer_execution_generation", "peer_result_digest")
    if any(value.get(key) is not None for key in peer_keys):
        from gateway.hosted_rooms_common import identifier
        import re
        for key in ("peer_run_id", "peer_admission_id"):
            fields[key] = identifier(value.get(key), label=key, error=ValueError, max_chars=256)
        generation, digest = value.get("peer_execution_generation"), value.get("peer_result_digest")
        if (type(generation) is not int or not 0 < generation <= 9007199254740991
                or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError("invalid peer output result identity")
        fields.update(peer_execution_generation=generation, peer_result_digest=digest)
    return fields
