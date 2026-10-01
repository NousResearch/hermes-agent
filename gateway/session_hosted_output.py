"""Group Chat output binding for one admitted hosted turn, on the profile that runs it.

The binding exists only while the exact claimed admission executes. Every
outbox write re-checks it inside the outbox's own write transaction, so a
stopped, superseded or replaced attempt can never add output afterwards.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sqlite3

from gateway.hosted_room_artifacts import (
    RoomArtifactError,
    RoomArtifactOutbox,
    RoomArtifactScope,
    terminal_artifact_manifest,
    validate_terminal_artifact_manifest,
)
from hermes_state_runtime import RuntimeStoreError, _epoch

logger = logging.getLogger(__name__)

_OUTPUT = ContextVar("canonical_hosted_output", default=None)


@dataclass
class HostedOutputBinding:
    """Output of one claimed admission, written to the executing profile's own outbox.

    When this profile also owns the room (the default profile) the attempt's
    driver row is checked in the outbox's write transaction.
    """

    authority: object
    ref: object
    row: dict
    scope: RoomArtifactScope
    cancel_generation: int
    owner_pid: int
    active: bool = True
    used: bool = False

    def check_write(self, conn, scope):
        if not self.active or os.getpid() != self.owner_pid or scope != self.scope:
            raise RoomArtifactError("Group Chat output producer is no longer active")
        authority, row = self.authority, self.row
        _epoch(conn, authority.epoch)
        admission = conn.execute(
            "SELECT status, owner_epoch, generation, principal_id, target_session_id, request_id "
            "FROM session_admissions WHERE admission_id=?",
            (row["admission_id"],),
        ).fetchone()
        if (admission is None or admission["status"] != "started"
                or admission["owner_epoch"] != authority.epoch
                or admission["generation"] != row["generation"]
                or admission["principal_id"] != row["principal_id"]
                or admission["request_id"] != row["request_id"]
                or admission["target_session_id"] != self.ref.session_id):
            raise RoomArtifactError("Group Chat output admission changed")
        live = authority.sessions.get(self.ref.session_id)
        if live is None or live.event_stream.execution != {
                "authority_epoch": authority.epoch, "execution_generation": row["generation"],
                "admission_id": row["admission_id"]}:
            raise RoomArtifactError("Group Chat output admission changed")
        from gateway.hosted_room_output_fence import require_output_task
        require_output_task(conn, self.scope, self.cancel_generation, status="running")
        from gateway.session_hosted_service import _OWNER
        owner = conn.execute("SELECT value FROM state_meta WHERE key=?", (_OWNER + scope.room_id,)).fetchone()
        if owner is None or owner[0] != row["principal_id"]:
            raise RoomArtifactError("Group Chat output owner changed")

    def _outbox(self, *, authorize=False):
        return RoomArtifactOutbox(self.authority.db.db_path,
                                  authorize_write=self.check_write if authorize else None)

    def outbox(self):
        if not self.active or os.getpid() != self.owner_pid:
            raise RoomArtifactError("Group Chat output producer is unavailable")
        self.used = True
        return self._outbox(authorize=True)


def current_output_binding():
    binding = _OUTPUT.get()
    if binding is None or not binding.active or binding.owner_pid != os.getpid():
        return None
    return binding


def _hosted_identity(row):
    from gateway.hosted_room_driver import TaskIdentity
    identity, generation = json.loads(row["request_id"][len("hosted:"):])
    identity = TaskIdentity(**identity)
    if type(generation) is not int or generation < 1:
        raise ValueError("invalid hosted generation")
    return identity, generation


def _bot_room_session(authority, ref):
    live = authority.sessions.get(ref.session_id)
    if live is None:
        return False
    from gateway.session_managed_worker import managed_policy
    if managed_policy(authority, ref) is not None:
        return False
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, live.source)
    return policy is not None and policy.source == "bot_room" and "bot_room" in policy.toolsets


def _transport_binding(authority, ref, row):
    """The owner-transport binding this session was admitted under, if any."""
    from gateway.session_hosted_transport import _BINDING, _principal

    with authority.db._read_ctx() as conn:
        retained = conn.execute("SELECT value FROM state_meta WHERE key=?",
                                (_BINDING + ref.session_id,)).fetchone()
    if retained is None:
        return None
    transport = json.loads(retained[0])
    if (not isinstance(transport, dict)
            or set(transport) != {"source_home", "selector", "target_home", "owner"}
            or not isinstance(transport["source_home"], str) or not Path(transport["source_home"]).is_absolute()
            or not isinstance(transport["owner"], str) or not transport["owner"]
            or transport["target_home"] != authority.profile_id
            or ref.profile_id != authority.profile_id
            or row.get("target_session_id") != ref.session_id
            or row.get("principal_id") != _principal(authority, transport).subject):
        raise RoomArtifactError("Group Chat output transport binding changed")
    return retained[0], transport


def _room_local_binding(authority, ref, row, identity, generation):
    """Default profile: this owner runs the turn and owns the room."""
    from gateway import hosted_room_driver as tasks
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from tui_gateway.hosted_room_driver import HostedRoomBinding

    service = getattr(authority, "hosted_room_service", None)
    if service is None or not callable(getattr(service, "_owner", None)):
        return None
    if service._owner(identity.room_id) != row.get("principal_id"):
        raise RoomArtifactError("Group Chat output owner changed")
    gateway, epoch = service._owned_authority(identity.room_id)
    task = next((t for t in tasks.list_tasks(service.db_path, room_id=identity.room_id)
                 if t["identity"] == identity and t["execution_generation"] == generation), None)
    if task is None or task["status"] != "running":
        raise RoomArtifactError("Group Chat output attempt changed")
    payload = task["payload"]
    profile = payload["target_profile"]
    member_id = payload.get("target_member_id", profile)
    room = service._room(identity.room_id)
    member = next((m for m in room["members"] if m["member_id"] == member_id), None)
    if (member is None or member.get("target", {}).get("kind", "local") != "local"
            or member["profile"] != profile
            or Path(service.profile_homes().get(profile, "")).resolve() != Path(authority.profile_id).resolve()):
        return None
    rpc = service._resolve_member_transport(HostedRoomBinding(identity.room_id, gateway, epoch), task)
    if type(rpc) is not HostedRoomAuthorityRPC or rpc.ref != ref:
        raise RoomArtifactError("Group Chat output session changed")
    scope = RoomArtifactScope.from_mapping(dict(
        room_id=identity.room_id, task_id=identity.task_id, execution_generation=generation,
        member_id=member_id, target_profile=profile, home_install_id=gateway,
        target_install_id=gateway, authority_gateway_id=gateway, authority_epoch=epoch))
    return HostedOutputBinding(authority, ref, dict(row), scope, task["cancel_generation"], os.getpid())


def _binding(authority, ref, row):
    try:
        identity, generation = _hosted_identity(row)
    except (TypeError, ValueError, KeyError):
        return None
    if not _bot_room_session(authority, ref):
        return None
    home = Path(authority.profile_id)
    if not home.is_absolute() or Path(authority.db.db_path).resolve().parent != home.resolve():
        return None
    if _transport_binding(authority, ref, row) is not None:
        return None  # a named profile's turn: its owner reaches its output over the owner transport
    binding = _room_local_binding(authority, ref, row, identity, generation)
    if binding is not None:
        with authority.db._read_ctx() as conn:
            binding.check_write(conn, binding.scope)
    return binding


async def output_binding(authority, ref, row):
    """Bind output for a hosted Group Chat admission; other admissions get none.

    Its reads run off the owner's event loop. A refusal leaves the turn
    running without file sharing rather than failing it.
    """
    if not str(row.get("request_id") or "").startswith("hosted:"):
        return None
    try:
        return await asyncio.to_thread(_binding, authority, ref, row)
    except (ValueError, LookupError, OSError, sqlite3.Error, AttributeError, TypeError) as exc:
        logger.warning("Group Chat file sharing is unavailable for admission %s: %s",
                       row.get("admission_id"), exc)
        return None


@contextmanager
def hosted_output_scope(binding):
    """Expose the binding to this turn only; copied tool contexts expire on return."""
    token = _OUTPUT.set(binding)
    try:
        yield binding
    finally:
        if binding is not None:
            binding.active = False
        _OUTPUT.reset(token)


def capture_output_result(authority, row, binding):
    """Report the open output of a finished turn with its result, or retire it."""
    if binding is None or not binding.used:
        return
    binding.active = False  # no later write can join the reported manifest
    saved = authority.pending_results.get(row["admission_id"])
    if saved is None:
        raise RuntimeStoreError("storage_unavailable")
    value = saved.get("result")
    if not isinstance(value, dict) or (
            value.get("interrupted") is True or value.get("failed") is True or value.get("error")):
        capture_failed_output(authority, row, binding)
        return
    manifest = terminal_artifact_manifest(binding._outbox().list(binding.scope))
    if manifest is None:
        return
    fields = {"artifacts": manifest, "artifact_scope": binding.scope.as_mapping()}
    from gateway.session_results import _redacted
    if _redacted(fields) != fields:
        # The stored result is redacted; metadata that would change cannot be published.
        raise RoomArtifactError("Group Chat output metadata cannot be stored exactly")
    value.update(fields)


def capture_failed_output(authority, row, binding):
    """Retire a failed or stopped turn's output before its terminal is exposed."""
    if binding is None or not binding.used:
        return None
    binding.active = False
    try:
        binding._outbox().discard_durably(binding.scope)
    except (OSError, ValueError, sqlite3.Error) as exc:
        # A committed intent replays when this outbox next opens; an uncommitted
        # one is retired by the room owner's sweep or the outbox's expiry.
        logger.warning("Group Chat output cleanup for admission %s is pending: %s",
                       row.get("admission_id"), exc)
        return "cleanup_pending"
    return None


def replay_output_cleanups(authority):
    """Reopen this profile's outbox, if it has one, so committed discards finish now."""
    from gateway.hosted_room_artifacts import output_store_exists

    try:
        with authority.db._read_ctx() as conn:
            if not output_store_exists(conn):
                return False
        RoomArtifactOutbox(authority.db.db_path)
    except (OSError, ValueError, sqlite3.Error) as exc:
        logger.warning("Group Chat output cleanup replay is pending: %s", exc)
        return False
    return True


def output_receipt_fields(value):
    """Bounded output metadata for a terminal receipt; never a path, never a guess.

    Invalid metadata is dropped rather than raised: a receipt must stay readable
    so the turn settles, and the owner retires the unreported output instead.
    """
    if not isinstance(value, dict) or not value.get("artifacts"):
        return {}
    try:
        validate_terminal_artifact_manifest(value["artifacts"])
        scope = RoomArtifactScope.from_mapping(value.get("artifact_scope") or {})
    except (RoomArtifactError, TypeError, ValueError):
        logger.warning("Dropping invalid Group Chat output metadata from a terminal receipt")
        return {}
    return {"artifacts": value["artifacts"], "artifact_scope": scope.as_mapping()}
