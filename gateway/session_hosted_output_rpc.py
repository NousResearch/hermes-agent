"""Served named-profile Output consent, custody, and owner-local actions.

The source coordinator owns publication and retry.  The named target owns consent,
its initialized outbox, receipt, bytes, ACK, and discard.  No function in this
module opens another authority's database or returns an owner-local path.
"""
from __future__ import annotations

import base64
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping

from gateway.hosted_room_artifacts import (
    RoomArtifactError,
    RoomArtifactOutbox,
    RoomArtifactScope,
    UNACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS,
    validate_terminal_artifact_manifest,
)
from gateway.hosted_room_driver import TaskIdentity, list_tasks
from hermes_state_runtime import RuntimeStoreError, _admission, _epoch, _json, _row
from hermes_state_terminal import RESULT_PREFIX

CONSENT_PREFIX = "gateway.hosted.owner_output.v1:"
FAILURE_PREFIX = "gateway.hosted.owner_output_failure.v1:"
RIGHTS = (
    "hosted.output.receipt",
    "hosted.output.export",
    "hosted.output.ack",
    "hosted.output.discard",
)
OUTPUT_OPERATIONS = frozenset({"output_export", "output_ack", "output_discard"})
# Deliberately false until stopped/unknown/disband retirement is wired end to end.
# The provider may initialize its owner-local store, but NEW admissions receive no
# Output consent while this hold remains.
OWNER_OUTPUT_LIFECYCLE_READY = False
CHUNK_BYTES = 360 * 1024
MAX_ITEM_BYTES = 15_000_000
_MAX_SAFE_INTEGER = 9007199254740991
_HEX = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_ID = re.compile(r"rart_[0-9a-f]{32}")
_RECEIPT_ID = re.compile(r"hor_[0-9a-f]{64}")
_COMMON = {
    "session_id",
    "task",
    "execution_generation",
    "artifact_scope",
    "manifest_digest",
    "owner_output_receipt",
}


def _canonical(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeStoreError("invalid_params") from exc


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _positive(value, *, reason="invalid_params") -> int:
    if type(value) is not int or not 0 < value <= _MAX_SAFE_INTEGER:
        raise RuntimeStoreError(reason)
    return value


def _hex(value, *, reason="invalid_params") -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise RuntimeStoreError(reason)
    return value


def _identifier(value, label="identifier") -> str:
    from gateway.hosted_rooms_common import identifier

    return identifier(value, label=label, error=RuntimeStoreError, max_chars=256)


def consent_key(session_id: str, principal_id: str, request_id: str) -> str:
    return CONSENT_PREFIX + _digest([session_id, principal_id, request_id])


def _owner_identity(authority):
    return (
        authority,
        authority.db,
        authority.epoch,
        authority.instance_id,
        authority.runner.session_authorities,
        os.getpid(),
    )


def _named_home(authority) -> Path | None:
    home = Path(authority.profile_id)
    registry = getattr(authority.runner, "session_authorities", None)
    if (
        not home.is_absolute()
        or home != home.resolve()
        or home.parent.name != "profiles"
        or registry is None
        or registry.for_home(home) is not authority
        or Path(authority.db.db_path).resolve().parent != home
        or Path(registry.launch.profile_id).resolve() != home.parent.parent
    ):
        return None
    return home


def initialize_owner_output(service) -> bool:
    """Initialize a named owner's local store during service preparation only."""
    authority = service.authority
    home = _named_home(authority)
    if home is None:
        return False
    from gateway.runtime_ownership import process_ownership

    if not process_ownership.owns(home) or authority.db._db_file_was_replaced():
        raise RuntimeStoreError("output_owner_unavailable")
    service._owner_output_outbox = RoomArtifactOutbox(
        authority.db.db_path, root=home / "hosted-room-artifact-outbox"
    )
    service._owner_output_owner = _owner_identity(authority)
    retry_owner_output_cleanups(service)
    return True


def _provider(service, conn=None) -> RoomArtifactOutbox:
    authority = service.authority
    home = _named_home(authority)
    from gateway.runtime_ownership import process_ownership

    outbox = getattr(service, "_owner_output_outbox", None)
    if (
        home is None
        or authority.hosted_room_service is not service
        or getattr(service, "_owner_output_owner", None) != _owner_identity(authority)
        or type(outbox) is not RoomArtifactOutbox
        or authority.db._db_file_was_replaced()
        or not process_ownership.owns(home)
        or outbox.db_path.resolve() != Path(authority.db.db_path).resolve()
        or outbox.root.resolve() != (home / "hosted-room-artifact-outbox").resolve()
        or not outbox.blob_root.is_dir()
        or outbox.blob_root.is_symlink()
    ):
        raise RuntimeStoreError("output_owner_unavailable")
    if conn is not None:
        _epoch(conn, authority.epoch)
        database = conn.execute("PRAGMA database_list").fetchone()[2]
        if Path(database).resolve() != outbox.db_path.resolve():
            raise RuntimeStoreError("output_owner_unavailable")
        conn.execute(
            "SELECT scope_json,ack_message_event_id,blob_reclaimed_at FROM hosted_room_output_artifacts LIMIT 0"
        )
        conn.execute(
            "SELECT lineage_identity,max_generation,retired_generation FROM hosted_room_output_generation_fences LIMIT 0"
        )
    return outbox


def source_output_admission(service, selector, task, generation, *, owner, target_home):
    """Capture source facts before target NEW; this grants no target right."""
    try:
        identity = task if isinstance(task, TaskIdentity) else TaskIdentity(**task)
        generation = _positive(generation, reason="permission_denied")
        room = service._room(identity.room_id)
        member = selector["member_id"]
        profile = selector["profile"]
        if (
            identity.room_id != selector["room_id"]
            or Path(target_home).resolve() == Path(service.authority.profile_id).resolve()
            or room["authority_gateway_id"] is None
            or type(room["authority_epoch"]) is not int
            or room["authority_epoch"] < 1
        ):
            return None
        gateway = room["authority_gateway_id"]
        scope = RoomArtifactScope.from_mapping(
            {
                "room_id": identity.room_id,
                "task_id": identity.task_id,
                "execution_generation": generation,
                "member_id": member,
                "target_profile": profile,
                "home_install_id": gateway,
                "target_install_id": gateway,
                "authority_gateway_id": gateway,
                "authority_epoch": room["authority_epoch"],
            }
        )
        from hermes_constants import hermes_home_key

        return {
            "version": 1,
            "selector": dict(selector),
            "task": asdict(identity),
            "hosted_execution_generation": generation,
            "scope": scope.as_mapping(),
            "source_home": str(Path(service.authority.profile_id).resolve()),
            "source_home_key": hermes_home_key(service.authority.profile_id),
            "owner_subject": owner,
            "target_home": str(Path(target_home).resolve()),
        }
    except (KeyError, TypeError, ValueError, RoomArtifactError, RuntimeStoreError):
        return None


@dataclass(frozen=True)
class OwnerOutputContext:
    authority: object
    service: object
    binding: dict
    attested: dict
    peer_subject: str


def capture_owner_output_context(authority, binding, attested, peer_subject):
    if not OWNER_OUTPUT_LIFECYCLE_READY:
        return None
    service = getattr(authority, "hosted_room_service", None)
    candidate = attested.get("owner_output_admission") if isinstance(attested, dict) else None
    if not isinstance(peer_subject, str) or not peer_subject or not isinstance(candidate, dict):
        return None
    try:
        _provider(service)
        registry = authority.runner.session_authorities
        source = registry.for_home(candidate["source_home"])
        if (
            source is None
            or registry.for_home(authority.profile_id) is not authority
            or candidate["target_home"] != authority.profile_id
            or binding["source_home"] != candidate["source_home"]
            or binding["target_home"] != authority.profile_id
            or candidate["selector"] != binding["selector"]
        ):
            return None
    except (AttributeError, KeyError, RuntimeStoreError):
        return None
    return OwnerOutputContext(authority, service, copy.deepcopy(binding), copy.deepcopy(candidate), peer_subject)


def new_admission_authorizer(rpc, context, *, request_id, payload, task, generation):
    """Return a target-local NEW guard; exact replay bypass is intentional."""
    if type(context) is not OwnerOutputContext or context.authority is not rpc.authority:
        return None
    candidate = context.attested
    try:
        identity = task if isinstance(task, TaskIdentity) else TaskIdentity(**task)
        scope = RoomArtifactScope.from_mapping(candidate["scope"])
        if (
            candidate["task"] != asdict(identity)
            or candidate["hosted_execution_generation"] != generation
            or scope.execution_generation != generation
            or scope.target_profile != rpc.profile
            or scope.member_id != rpc.member_id
            or candidate["selector"]
            != {"room_id": rpc.room_id, "member_id": rpc.member_id, "profile": rpc.profile}
        ):
            raise RuntimeStoreError("permission_denied")
        from gateway.session_admission import admission_fingerprint

        payload_digest = admission_fingerprint(
            canonical_target=rpc.ref.session_id,
            payload={"input": payload, "intent": "queue"},
        )
        record = {
            "version": 1,
            "state": "admitted",
            "rights": list(RIGHTS),
            "selector": candidate["selector"],
            "task": candidate["task"],
            "hosted_execution_generation": generation,
            "scope": scope.as_mapping(),
            "source": {
                "home_key": candidate["source_home_key"],
                "owner_subject": candidate["owner_subject"],
                "peer_subject_sha256": hashlib.sha256(context.peer_subject.encode()).hexdigest(),
            },
            "target": {
                "profile_id": rpc.authority.profile_id,
                "session_id": rpc.ref.session_id,
                "principal_id": rpc.principal.subject,
                "request_id": request_id,
                "payload_digest": payload_digest,
                "admitting_owner_epoch": rpc.authority.epoch,
                "admitting_instance_id": rpc.authority.instance_id,
            },
        }
        encoded = _canonical(record)
        key = consent_key(rpc.ref.session_id, rpc.principal.subject, request_id)
    except (KeyError, TypeError, ValueError, RoomArtifactError) as exc:
        raise RuntimeStoreError("permission_denied") from exc

    def authorize(conn):
        outbox = _provider(context.service, conn)
        del outbox
        registry = rpc.authority.runner.session_authorities
        if (
            registry.for_home(candidate["source_home"]) is None
            or registry.for_home(rpc.authority.profile_id) is not rpc.authority
            or _owner_identity(rpc.authority) != context.service._owner_output_owner
        ):
            raise RuntimeStoreError("permission_denied")
        old = conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
        if old is not None and old[0] != encoded:
            raise RuntimeStoreError("admission_conflict")
        conn.execute("INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)", (key, encoded))

    return authorize


def _load_consent(conn, row):
    key = consent_key(row["target_session_id"], row["principal_id"], row["request_id"])
    saved = conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
    if saved is None:
        return None
    try:
        record = json.loads(saved[0])
        scope = RoomArtifactScope.from_mapping(record["scope"])
        if (
            set(record) != {
                "version",
                "state",
                "rights",
                "selector",
                "task",
                "hosted_execution_generation",
                "scope",
                "source",
                "target",
            }
            or record["version"] != 1
            or record["state"] != "admitted"
            or record["rights"] != list(RIGHTS)
            or record["target"]["session_id"] != row["target_session_id"]
            or record["target"]["principal_id"] != row["principal_id"]
            or record["target"]["request_id"] != row["request_id"]
            or record["target"]["payload_digest"] != row["payload_digest"]
            or record["hosted_execution_generation"] != scope.execution_generation
        ):
            raise ValueError("consent changed")
    except (KeyError, TypeError, ValueError, RoomArtifactError) as exc:
        raise RuntimeStoreError("permission_denied") from exc
    return key, saved[0], record, scope


def owner_output_binding(authority, ref, row, transport):
    """Reconstruct a consented target-local binding; no source DB is opened."""
    service = getattr(authority, "hosted_room_service", None)
    if _named_home(authority) is None:
        return None
    from gateway.session_hosted_output import HostedOutputBinding

    with authority.db._read_ctx() as conn:
        _provider(service, conn)
        loaded = _load_consent(conn, {
            **row,
            "payload_digest": _stored_payload_digest(conn, row["admission_id"]),
        })
        if loaded is None:
            return None
        key, encoded, record, scope = loaded
        if (
            record["target"]["profile_id"] != authority.profile_id
            or record["target"]["admitting_owner_epoch"] != authority.epoch
            or record["target"]["admitting_instance_id"] != authority.instance_id
            or record["source"]["owner_subject"] != transport["owner"]
            or record["selector"] != transport["selector"]
            or record["source"]["home_key"] != _home_key(transport["source_home"])
        ):
            raise RuntimeStoreError("permission_denied")

    class NamedOwnerOutputBinding(HostedOutputBinding):
        consent_key = key
        consent_json = encoded
        consent = record
        target_outbox = service._owner_output_outbox

        def check_write(self, conn, checked_scope):
            if (
                not self.active
                or os.getpid() != self.owner_pid
                or checked_scope != self.scope
                or self.authority is not authority
            ):
                raise RoomArtifactError("Group Chat output producer is no longer active")
            _provider(service, conn)
            raw = conn.execute("SELECT * FROM session_admissions WHERE admission_id=?", (row["admission_id"],)).fetchone()
            live = authority.sessions.get(ref.session_id)
            if (
                raw is None
                or raw["status"] != "started"
                or raw["owner_epoch"] != authority.epoch
                or raw["generation"] != row["generation"]
                or raw["principal_id"] != row["principal_id"]
                or raw["target_session_id"] != ref.session_id
                or live is None
                or live.event_stream.execution
                != {
                    "authority_epoch": authority.epoch,
                    "execution_generation": row["generation"],
                    "admission_id": row["admission_id"],
                }
            ):
                raise RoomArtifactError("Group Chat output admission changed")
            current = conn.execute("SELECT value FROM state_meta WHERE key=?", (self.consent_key,)).fetchone()
            if current is None or current[0] != self.consent_json:
                raise RoomArtifactError("Group Chat output consent changed")

        def outbox(self):
            if not self.active or os.getpid() != self.owner_pid:
                raise RoomArtifactError("Group Chat output producer is unavailable")
            self.used = True
            result = copy.copy(self.target_outbox)
            result.authorize_write = self.check_write
            return result

    binding = NamedOwnerOutputBinding(authority, ref, copy.deepcopy(row), scope, 0, os.getpid())
    with authority.db._read_ctx() as conn:
        binding.check_write(conn, scope)
    return binding


def _home_key(home):
    from hermes_constants import hermes_home_key

    return hermes_home_key(home)


def _stored_payload_digest(conn, admission_id):
    row = conn.execute("SELECT payload_digest FROM session_admissions WHERE admission_id=?", (admission_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError("permission_denied")
    return row[0]


def _failure_key(admission_id):
    return FAILURE_PREFIX + _digest([_identifier(admission_id, "admission_id")])


@dataclass(frozen=True)
class UnknownOwnerOutputContext:
    authority: object
    service: object
    source_home: str
    selector: dict
    peer_subject: str
    owner: str
    source_discard_digest: str


def capture_unknown_output_context(authority, binding, attested, peer_subject, params):
    """Bind reverse-attested cleanup facts without opening the source database."""
    service = getattr(authority, "hosted_room_service", None)
    digest = attested.get("source_discard_digest") if isinstance(attested, dict) else None
    try:
        _provider(service)
    except (AttributeError, RuntimeStoreError):
        return None
    if (
        _named_home(authority) is None
        or not isinstance(peer_subject, str)
        or not peer_subject
        or not isinstance(digest, str)
        or digest != params.get("_source_discard_digest")
        or _HEX.fullmatch(digest) is None
        or not isinstance(attested.get("owner"), str)
        or not attested["owner"]
    ):
        return None
    return UnknownOwnerOutputContext(
        authority=authority,
        service=service,
        source_home=binding["source_home"],
        selector=copy.deepcopy(binding["selector"]),
        peer_subject=peer_subject,
        owner=attested["owner"],
        source_discard_digest=digest,
    )


def _owner_cleanup_snapshot(
    authority, row, scope, consent_json, conn, *, allowed, context=None
):
    service = authority.hosted_room_service
    outbox = _provider(service, conn)
    raw = _admission(conn, row["admission_id"])
    current = _row(raw)
    loaded = _load_consent(conn, {**current, "payload_digest": raw["payload_digest"]})
    if loaded is None:
        raise RoomArtifactError("Group Chat output cleanup consent is unavailable")
    _, encoded, consent, saved_scope = loaded
    terminal = (current["status"], current.get("outcome"))
    if (
        encoded != consent_json
        or saved_scope != scope
        or current["target_session_id"] != row["target_session_id"]
        or current["principal_id"] != row["principal_id"]
        or current["request_id"] != row["request_id"]
        or current["generation"] != row["generation"]
        or terminal not in allowed
        or consent["target"]["profile_id"] != authority.profile_id
        or consent["task"]["task_id"] != scope.task_id
        or consent["hosted_execution_generation"] != scope.execution_generation
    ):
        raise RoomArtifactError("Group Chat output cleanup binding changed")
    if context is not None and (
        type(context) is not UnknownOwnerOutputContext
        or context.authority is not authority
        or context.service is not service
        or consent["selector"] != context.selector
        or consent["source"]["home_key"] != _home_key(context.source_home)
        or consent["source"]["owner_subject"] != context.owner
        or consent["source"]["peer_subject_sha256"]
            != hashlib.sha256(context.peer_subject.encode()).hexdigest()
    ):
        raise RoomArtifactError("Group Chat output cleanup source changed")
    return outbox, current, consent


def _stage_owner_cleanup(authority, row, scope, consent_json, *, reason, allowed, context=None):
    key = _failure_key(row["admission_id"])
    consent_digest = hashlib.sha256(consent_json.encode()).hexdigest()
    commitment = {
        "version": 1,
        "admission_id": row["admission_id"],
        "target_session_id": row["target_session_id"],
        "execution_generation": row["generation"],
        "scope": scope.as_mapping(),
        "consent_digest": consent_digest,
        "reason_code": reason,
        "source_discard_digest": (
            context.source_discard_digest if context is not None else None
        ),
    }

    def stage(conn):
        outbox, _, _ = _owner_cleanup_snapshot(
            authority, row, scope, consent_json, conn, allowed=allowed, context=context
        )
        saved = conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
        old = json.loads(saved[0]) if saved is not None else None
        if old is not None:
            if old.get("commitment") != commitment:
                raise RoomArtifactError("Group Chat output cleanup commitment changed")
            return old
        rows = conn.execute(
            "SELECT * FROM hosted_room_output_artifacts WHERE scope_key=? "
            "ORDER BY created_at,artifact_id LIMIT 9",
            (scope.key,),
        ).fetchall()
        if len(rows) > 8:
            raise RoomArtifactError("Group Chat output cleanup inventory is too large")
        items = [outbox._manifest(saved) for saved in rows]
        if items:
            authorize = lambda checked: _owner_cleanup_snapshot(
                authority, row, scope, consent_json, checked,
                allowed=allowed, context=context,
            )
            from gateway.hosted_room_output_discard import retire_exact
            blobs = retire_exact(outbox, conn, scope, items, authorize=authorize)
            state = "pending"
        else:
            outbox._retire_generation(conn, scope)
            blobs, state = [], "completed"
        record = {
            "version": 1,
            "commitment": commitment,
            "state": state,
            "items": items,
            "blobs": blobs,
            "removed": len(items),
            "attempts": 0,
            "last_error": None,
        }
        conn.execute(
            "INSERT INTO state_meta(key,value) VALUES(?,?)", (key, _canonical(record))
        )
        return record

    record = authority.db._execute_write(stage)
    if record["state"] == "completed":
        return True
    return _complete_owner_cleanup(
        authority, row, scope, consent_json, record,
        allowed=allowed, context=context,
    )


def _complete_owner_cleanup(
    authority, row, scope, consent_json, record, *, allowed, context=None
):
    key = _failure_key(row["admission_id"])
    try:
        def complete(conn):
            outbox, _, _ = _owner_cleanup_snapshot(
                authority, row, scope, consent_json, conn,
                allowed=allowed, context=context,
            )
            saved = conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
            current = json.loads(saved[0]) if saved is not None else None
            if current != record:
                raise RoomArtifactError("Group Chat output cleanup reservation changed")
            if current["state"] == "completed":
                return current
            from gateway.hosted_room_output_discard import cleanup_exact
            authorize = lambda checked: _owner_cleanup_snapshot(
                authority, row, scope, consent_json, checked,
                allowed=allowed, context=context,
            )
            cleanup_exact(
                outbox, conn, scope, current["items"], current["blobs"],
                authorize=authorize,
            )
            done = {**current, "state": "completed", "blobs": [], "last_error": None}
            conn.execute("UPDATE state_meta SET value=? WHERE key=?", (_canonical(done), key))
            return done
        authority.db._execute_write(complete)
        return True
    except Exception as exc:
        def retain(conn):
            saved = conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
            current = json.loads(saved[0]) if saved is not None else None
            if current != record:
                return
            pending = {
                **current,
                "attempts": min(2147483647, current["attempts"] + 1),
                "last_error": (
                    "storage_unavailable"
                    if isinstance(exc, (OSError, RoomArtifactError))
                    else "cleanup_unavailable"
                ),
            }
            conn.execute("UPDATE state_meta SET value=? WHERE key=?", (_canonical(pending), key))
        authority.db._execute_write(retain)
        return False


def capture_failed_owner_output(authority, row, binding):
    """Commit named target cleanup before exposing a failed/interrupted terminal."""
    if not hasattr(binding, "consent_json"):
        raise RoomArtifactError("Group Chat named output consent is unavailable")
    with authority.db._read_ctx() as conn:
        binding.check_write(conn, binding.scope)
    return _stage_owner_cleanup(
        authority, row, binding.scope, binding.consent_json,
        reason="producer_failed",
        allowed={("started", None)},
    )


def discard_unknown_owner_output(authority, row, task, generation, context):
    """Resolve target-local Output before an unknown admission becomes terminal."""
    if context is None:
        return True
    with authority.db._read_ctx() as conn:
        raw = _admission(conn, row["admission_id"])
        loaded = _load_consent(conn, {**_row(raw), "payload_digest": raw["payload_digest"]})
    if loaded is None:
        return True
    _, consent_json, consent, scope = loaded
    if consent["task"] != asdict(task) or scope.execution_generation != generation:
        raise RuntimeStoreError("permission_denied")
    complete = _stage_owner_cleanup(
        authority, _row(raw), scope, consent_json,
        reason="unknown_discard",
        allowed={("unknown", None), ("terminal", "interrupted")},
        context=context,
    )
    if not complete:
        raise RuntimeStoreError("storage_unavailable")
    return True


def retry_owner_output_cleanups(service):
    """Replay owner-local committed cleanup intents at owner preparation."""
    authority = service.authority
    with authority.db._read_ctx() as conn:
        rows = conn.execute(
            "SELECT value FROM state_meta WHERE key LIKE ? "
            "AND json_extract(value,'$.state')='pending' ORDER BY key LIMIT 64",
            (FAILURE_PREFIX + "%",),
        ).fetchall()
    for saved in rows:
        try:
            record = json.loads(saved[0])
            commitment = record["commitment"]
            scope = RoomArtifactScope.from_mapping(commitment["scope"])
            with authority.db._read_ctx() as conn:
                raw = _admission(conn, commitment["admission_id"])
                loaded = _load_consent(conn, {**_row(raw), "payload_digest": raw["payload_digest"]})
            if loaded is None or hashlib.sha256(loaded[1].encode()).hexdigest() != commitment["consent_digest"]:
                continue
            allowed = (
                {("unknown", None), ("terminal", "interrupted")}
                if commitment["reason_code"] == "unknown_discard"
                else {("started", None), ("terminal", "failed"), ("terminal", "interrupted")}
            )
            _complete_owner_cleanup(
                authority, _row(raw), scope, loaded[1], record, allowed=allowed
            )
        except Exception:
            continue


def compact_owner_output_for_retirement(conn, raw):
    """Block pending named Output and compact completed target-only evidence."""
    row = _row(raw)
    key = consent_key(row["target_session_id"], row["principal_id"], row["request_id"])
    saved = conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
    if saved is None:
        return
    consent = json.loads(saved[0])
    if consent.get("version") == 2:
        return
    loaded = _load_consent(conn, {**row, "payload_digest": raw["payload_digest"]})
    if loaded is None:
        raise RuntimeStoreError("storage_unavailable")
    _, consent_json, _, scope = loaded
    result_key = RESULT_PREFIX + row["admission_id"]
    result_row = conn.execute("SELECT value FROM state_meta WHERE key=?", (result_key,)).fetchone()
    if result_row is None:
        raise RuntimeStoreError("storage_unavailable")
    result_wrapper = json.loads(result_row[0])
    result = result_wrapper.get("result")
    if not isinstance(result, dict):
        raise RuntimeStoreError("storage_unavailable")
    failure_key = _failure_key(row["admission_id"])
    failure_row = conn.execute("SELECT value FROM state_meta WHERE key=?", (failure_key,)).fetchone()
    failure = json.loads(failure_row[0]) if failure_row is not None else None
    artifacts = conn.execute(
        "SELECT acknowledged_at,blob_reclaimed_at FROM hosted_room_output_artifacts "
        "WHERE scope_key=?", (scope.key,),
    ).fetchall()

    if failure is not None:
        if failure.get("state") != "completed" or failure.get("blobs") != []:
            raise RuntimeStoreError("session_busy")
        disposition = {"kind": "cleanup", "digest": _digest(failure),
                       "removed": failure.get("removed")}
    elif result.get("owner_output_receipt") is not None:
        ack = result_wrapper.get("owner_output_ack")
        discard = result_wrapper.get("owner_output_discard")
        if ack is not None and discard is None:
            fence = conn.execute(
                "SELECT retired_generation FROM hosted_room_output_generation_fences "
                "WHERE lineage_identity=?", (scope.lineage_json,),
            ).fetchone()
            if (fence is None or int(fence[0]) < scope.execution_generation
                    or any(item["acknowledged_at"] is None
                           or item["blob_reclaimed_at"] is None for item in artifacts)):
                raise RuntimeStoreError("session_busy")
            disposition = {"kind": "ack", "digest": _digest(ack)}
        elif (discard is not None and ack is None
              and discard.get("state") == "completed"
              and discard.get("blobs") == [] and not artifacts):
            disposition = {"kind": "discard", "digest": _digest(discard)}
        else:
            raise RuntimeStoreError("session_busy")
    else:
        if artifacts:
            raise RuntimeStoreError("session_busy")
        disposition = {"kind": "unused", "digest": _digest([scope.key, row["admission_id"]])}

    compact = {
        "version": 2, "state": "completed",
        "consent_digest": hashlib.sha256(consent_json.encode()).hexdigest(),
        "scope_digest": _digest(scope.as_mapping()),
        "disposition_digest": _digest(disposition),
    }
    conn.execute("UPDATE state_meta SET value=? WHERE key=?", (_canonical(compact), key))
    if failure is not None:
        compact_failure = {
            "version": 2, "state": "completed",
            "commitment_digest": _digest(failure["commitment"]),
            "disposition_digest": _digest(disposition),
            "removed": failure.get("removed"), "attempts": failure.get("attempts"),
        }
        conn.execute("UPDATE state_meta SET value=? WHERE key=?",
                     (_canonical(compact_failure), failure_key))
    compact_result = {"result": {"owner_output_retirement": disposition}, "usage": {}}
    conn.execute("UPDATE state_meta SET value=? WHERE key=?",
                 (_canonical(compact_result), result_key))


def capture_owner_output_receipt(authority, row, binding, result):
    if not hasattr(binding, "consent_json"):
        return None
    manifest = result["artifacts"]
    scope = binding.scope
    items = validate_terminal_artifact_manifest(manifest)
    with authority.db._read_ctx() as conn:
        binding.check_write(conn, scope)
        placeholders = ",".join("?" for _ in items)
        artifact_ids = [item["artifact_id"] for item in items]
        rows = conn.execute(
            f"SELECT artifact_id,created_at FROM hosted_room_output_artifacts WHERE scope_key=? AND artifact_id IN ({placeholders})",
            (scope.key, *artifact_ids),
        ).fetchall()
        if len(rows) != len(items):
            raise RuntimeStoreError("storage_unavailable")
        expires_at = min(float(saved["created_at"]) for saved in rows) + UNACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS
    without = {key: value for key, value in result.items() if key != "owner_output_receipt"}
    result_digest = _digest(without)
    consent_digest = hashlib.sha256(binding.consent_json.encode()).hexdigest()
    identity = [
        consent_digest,
        row["admission_id"],
        row["target_session_id"],
        row["generation"],
        binding.consent["target"]["payload_digest"],
        scope.key,
        manifest["manifest_digest"],
        result_digest,
    ]
    return {
        "version": 1,
        "receipt_id": "hor_" + _digest(identity),
        "consent_digest": consent_digest,
        "target_admission_id": row["admission_id"],
        "target_session_id": row["target_session_id"],
        "target_request_id_sha256": hashlib.sha256(row["request_id"].encode()).hexdigest(),
        "target_execution_generation": row["generation"],
        "producing_owner_epoch": authority.epoch,
        "producing_instance_id": authority.instance_id,
        "target_payload_digest": binding.consent["target"]["payload_digest"],
        "result_digest": result_digest,
        "expires_at": expires_at,
    }


def validate_owner_output_receipt(value):
    required = {
        "version",
        "receipt_id",
        "consent_digest",
        "target_admission_id",
        "target_session_id",
        "target_request_id_sha256",
        "target_execution_generation",
        "producing_owner_epoch",
        "producing_instance_id",
        "target_payload_digest",
        "result_digest",
        "expires_at",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("version") != 1:
        raise ValueError("invalid owner Output receipt")
    result = dict(value)
    for key in ("target_admission_id", "target_session_id", "producing_instance_id"):
        result[key] = _identifier(result[key], key)
    if not isinstance(result["receipt_id"], str) or _RECEIPT_ID.fullmatch(result["receipt_id"]) is None:
        raise ValueError("invalid owner Output receipt")
    for key in (
        "consent_digest",
        "target_request_id_sha256",
        "target_payload_digest",
        "result_digest",
    ):
        try:
            _hex(result[key])
        except RuntimeStoreError as exc:
            raise ValueError("invalid owner Output receipt") from exc
    for key in ("target_execution_generation", "producing_owner_epoch"):
        try:
            _positive(result[key])
        except RuntimeStoreError as exc:
            raise ValueError("invalid owner Output receipt") from exc
    expiry = result["expires_at"]
    if type(expiry) not in {int, float} or not math.isfinite(expiry):
        raise ValueError("invalid owner Output receipt")
    result["expires_at"] = float(expiry)
    return result


def _validate_action_params(operation, params):
    extra = {
        "output_export": {"artifact_id", "offset"},
        "output_ack": {"artifact_ids", "message_event_id"},
        "output_discard": {"reason_code", "result_digest"},
    }[operation]
    if not isinstance(params, dict) or set(params) != _COMMON | extra:
        raise RuntimeStoreError("invalid_params")
    task = TaskIdentity(**params["task"])
    scope = RoomArtifactScope.from_mapping(params["artifact_scope"])
    receipt = validate_owner_output_receipt(params["owner_output_receipt"])
    if (
        _positive(params["execution_generation"]) != scope.execution_generation
        or asdict(task) != params["task"]
        or task.room_id != scope.room_id
        or task.task_id != scope.task_id
        or params["manifest_digest"] != _hex(params["manifest_digest"])
        or receipt["target_session_id"] != params["session_id"]
    ):
        raise RuntimeStoreError("invalid_params")
    if operation == "output_export":
        if not isinstance(params["artifact_id"], str) or _ARTIFACT_ID.fullmatch(params["artifact_id"]) is None:
            raise RuntimeStoreError("invalid_params")
        if type(params["offset"]) is not int or params["offset"] < 0:
            raise RuntimeStoreError("invalid_params")
    elif operation == "output_ack":
        ids = params["artifact_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or len(ids) > 8
            or len(set(ids)) != len(ids)
            or any(not isinstance(item, str) or _ARTIFACT_ID.fullmatch(item) is None for item in ids)
        ):
            raise RuntimeStoreError("invalid_params")
        _identifier(params["message_event_id"], "message_event_id")
    else:
        if params["reason_code"] not in {"verification_failed", "unpublished"}:
            raise RuntimeStoreError("invalid_params")
        if params["result_digest"] != receipt["result_digest"]:
            raise RuntimeStoreError("invalid_params")
    return task, scope, receipt


def source_output_action_attestation(service, selector, operation, params):
    """Authorize one reserved source action without any reverse call."""
    action_params = {key: value for key, value in params.items() if key != "_target_home"}
    task_identity, scope, receipt = _validate_action_params(operation, action_params)
    matches = [
        task
        for task in list_tasks(service.db_path, room_id=scope.room_id)
        if task["identity"] == task_identity
        and task["execution_generation"] == scope.execution_generation
    ]
    if len(matches) != 1:
        raise RuntimeStoreError("permission_denied")
    task = matches[0]
    result = task.get("result")
    if (
        selector
        != {"room_id": scope.room_id, "member_id": scope.member_id, "profile": scope.target_profile}
        or not isinstance(result, dict)
        or result.get("artifact_scope") != scope.as_mapping()
        or result.get("artifacts", {}).get("manifest_digest") != action_params["manifest_digest"]
        or result.get("owner_output_receipt") != receipt
    ):
        raise RuntimeStoreError("permission_denied")
    expected_operation = {
        "output_export": "publish",
        "output_ack": "ack",
        "output_discard": "discard",
    }[operation]
    key = service._output_key(task)
    with service.authority.db._read_ctx() as conn:
        current = service._output_metadata(conn, key)
        retry = conn.execute(
            "SELECT * FROM hosted_room_artifact_retries WHERE room_id=? AND task_id=? AND execution_generation=?",
            key,
        ).fetchone()
        if retry is None or retry["blocked"] or retry["operation"] != expected_operation:
            raise RuntimeStoreError("permission_denied")
        retained = json.loads(retry["metadata_json"])
        if any(retained.get(name) != current[name] for name in ("work", "route", "lineage", "member_id")):
            raise RuntimeStoreError("permission_denied")
        if operation == "output_ack" and service._publication_operation(conn, key) != "ack":
            raise RuntimeStoreError("permission_denied")
    return {"action_digest": _digest({"operation": operation, "params": action_params})}


def _target_snapshot(authority, source_home, selector, peer_subject, operation, params, attested, conn):
    task, scope, receipt = _validate_action_params(operation, params)
    service = authority.hosted_room_service
    outbox = _provider(service, conn)
    try:
        raw = _admission(conn, receipt["target_admission_id"])
        row = _row(raw)
        stored_digest = raw["payload_digest"]
        loaded = _load_consent(conn, {**row, "payload_digest": stored_digest})
        if loaded is None:
            raise RuntimeStoreError("permission_denied")
        _, encoded, consent, consent_scope = loaded
        result_row = conn.execute(
            "SELECT value FROM state_meta WHERE key=?", (RESULT_PREFIX + receipt["target_admission_id"],)
        ).fetchone()
        saved = json.loads(result_row[0]) if result_row else None
        result = saved["result"] if isinstance(saved, dict) else None
        manifest = result.get("artifacts") if isinstance(result, dict) else None
        manifest_items = validate_terminal_artifact_manifest(manifest)
    except (KeyError, TypeError, ValueError, RoomArtifactError) as exc:
        raise RuntimeStoreError("permission_denied") from exc
    expected_action = _digest({"operation": operation, "params": params})
    if (
        row["status"] != "terminal"
        or row["outcome"] != "completed"
        or row["target_session_id"] != params["session_id"]
        or row["generation"] != receipt["target_execution_generation"]
        or row["owner_epoch"] != receipt["producing_owner_epoch"]
        or hashlib.sha256(row["request_id"].encode()).hexdigest() != receipt["target_request_id_sha256"]
        or stored_digest != receipt["target_payload_digest"]
        or receipt["producing_instance_id"] != consent["target"]["admitting_instance_id"]
        or hashlib.sha256(encoded.encode()).hexdigest() != receipt["consent_digest"]
        or consent_scope != scope
        or consent["task"] != asdict(task)
        or consent["selector"] != selector
        or consent["source"]["home_key"] != _home_key(source_home)
        or consent["source"]["owner_subject"] != attested.get("owner")
        or consent["source"]["peer_subject_sha256"] != hashlib.sha256(peer_subject.encode()).hexdigest()
        or attested.get("target_home") != authority.profile_id
        or attested.get("action_digest") != expected_action
        or manifest["manifest_digest"] != params["manifest_digest"]
        or result.get("artifact_scope") != scope.as_mapping()
        or result.get("owner_output_receipt") != receipt
    ):
        raise RuntimeStoreError("permission_denied")
    if not (saved.get("owner_output_ack") or saved.get("owner_output_discard")):
        without = {key: value for key, value in result.items() if key != "owner_output_receipt"}
        if _digest(without) != receipt["result_digest"]:
            raise RuntimeStoreError("permission_denied")
    if saved.get("owner_output_ack") is not None and operation == "output_discard":
        raise RuntimeStoreError("permission_denied")
    if saved.get("owner_output_discard") is not None and operation in {"output_export", "output_ack"}:
        raise RuntimeStoreError("permission_denied")
    return {
        "row": row,
        "saved": saved,
        "result": result,
        "manifest": manifest,
        "items": manifest_items,
        "scope": scope,
        "receipt": receipt,
        "outbox": outbox,
    }


def handle_target_output_operation(
    authority, *, source_home, selector, peer_subject, operation, params, attested
):
    """Perform one target-local action after source I/O, never while a writer is held."""
    if operation not in OUTPUT_OPERATIONS or not isinstance(peer_subject, str) or not peer_subject:
        raise RuntimeStoreError("invalid_params")
    with authority.db._read_ctx() as conn:
        snapshot = _target_snapshot(
            authority, source_home, selector, peer_subject, operation, params, attested, conn
        )
    scope, receipt, outbox = snapshot["scope"], snapshot["receipt"], snapshot["outbox"]
    if operation == "output_export":
        item = next((item for item in snapshot["items"] if item["artifact_id"] == params["artifact_id"]), None)
        if item is None or item["size"] > MAX_ITEM_BYTES or not 0 <= params["offset"] < item["size"]:
            raise RuntimeStoreError("permission_denied")
        actual, data = outbox.read(scope, params["artifact_id"])
        if actual != item or len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeStoreError("storage_unavailable")
        offset = params["offset"]
        raw = data[offset : offset + CHUNK_BYTES]
        with authority.db._read_ctx() as conn:
            _target_snapshot(authority, source_home, selector, peer_subject, operation, params, attested, conn)
        return {
            "version": 1,
            "receipt_id": receipt["receipt_id"],
            "artifact_id": item["artifact_id"],
            "offset": offset,
            "total_size": item["size"],
            "sha256": item["sha256"],
            "data_base64": base64.b64encode(raw).decode("ascii"),
            "eof": offset + len(raw) == item["size"],
            "target_profile_id_sha256": hashlib.sha256(authority.profile_id.encode()).hexdigest(),
            "serving_owner_epoch": authority.epoch,
            "serving_instance_id": authority.instance_id,
        }
    if operation == "output_ack":
        expected_ids = [item["artifact_id"] for item in snapshot["items"]]
        if (
            params["artifact_ids"] != expected_ids
            or params["message_event_id"] != "dmessage:" + scope.task_id.removeprefix("dtask:")
        ):
            raise RuntimeStoreError("permission_denied")
        commitment = {
            "version": 1,
            "receipt_id": receipt["receipt_id"],
            "manifest_digest": snapshot["manifest"]["manifest_digest"],
            "artifact_ids": expected_ids,
            "message_event_id": params["message_event_id"],
        }
        old = snapshot["saved"].get("owner_output_ack")
        if old is not None and old != commitment:
            raise RuntimeStoreError("permission_denied")
        if old == commitment and outbox.retirement_complete(scope):
            return {"acknowledged": True, "changed": 0}
        target = copy.copy(outbox)

        def authorize(conn, checked_scope):
            if checked_scope != scope:
                raise RoomArtifactError("Group Chat output ACK scope changed")
            current = _target_snapshot(
                authority, source_home, selector, peer_subject, operation, params, attested, conn
            )
            saved = current["saved"]
            previous = saved.get("owner_output_ack")
            if previous is not None and previous != commitment:
                raise RoomArtifactError("Group Chat output ACK changed")

        def commit(conn, checked_scope):
            authorize(conn, checked_scope)
            key = RESULT_PREFIX + receipt["target_admission_id"]
            saved = json.loads(conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()[0])
            saved["owner_output_ack"] = commitment
            conn.execute("UPDATE state_meta SET value=? WHERE key=?", (_json(saved), key))

        target.authorize_write = authorize
        target.commit_acknowledgement = commit
        try:
            changed = target.acknowledge(scope, expected_ids, message_event_id=params["message_event_id"])
        except (OSError, RoomArtifactError) as exc:
            raise RuntimeStoreError("storage_unavailable") from exc
        return {"acknowledged": True, "changed": changed}
    return _discard_target(
        authority,
        source_home=source_home,
        selector=selector,
        peer_subject=peer_subject,
        params=params,
        attested=attested,
        snapshot=snapshot,
    )


def _discard_target(authority, *, source_home, selector, peer_subject, params, attested, snapshot):
    from gateway.hosted_room_output_discard import cleanup_exact, retire_exact

    scope, receipt = snapshot["scope"], snapshot["receipt"]
    commitment = {
        "version": 1,
        "receipt_id": receipt["receipt_id"],
        "manifest_digest": snapshot["manifest"]["manifest_digest"],
        "reason_code": params["reason_code"],
        "result_digest": params["result_digest"],
    }
    existing = snapshot["saved"].get("owner_output_discard")
    if existing is not None:
        if existing.get("commitment") != commitment:
            raise RuntimeStoreError("permission_denied")
        if existing.get("state") == "completed":
            return {"discarded": True, "removed": existing["removed"]}
    key = RESULT_PREFIX + receipt["target_admission_id"]

    if existing is None:
        def stage(conn):
            current = _target_snapshot(
                authority, source_home, selector, peer_subject, "output_discard", params, attested, conn
            )
            target = copy.copy(current["outbox"])
            authorize = lambda checked: _target_snapshot(
                authority, source_home, selector, peer_subject, "output_discard", params, attested, checked
            )
            blobs = retire_exact(target, conn, scope, current["items"], authorize=authorize)
            record = {
                "version": 1,
                "commitment": commitment,
                "state": "pending",
                "removed": len(current["items"]),
                "items": current["items"],
                "blobs": blobs,
            }
            saved = current["saved"]
            saved["owner_output_discard"] = record
            conn.execute("UPDATE state_meta SET value=? WHERE key=?", (_json(saved), key))
            return record
        try:
            existing = authority.db._execute_write(stage)
        except (OSError, RoomArtifactError) as exc:
            raise RuntimeStoreError("storage_unavailable") from exc

    def complete(conn):
        current = _target_snapshot(
            authority, source_home, selector, peer_subject, "output_discard", params, attested, conn
        )
        saved = current["saved"]
        record = saved.get("owner_output_discard")
        if record != existing or record["state"] != "pending":
            raise RoomArtifactError("Group Chat output discard changed")
        target = copy.copy(current["outbox"])
        authorize = lambda checked: _target_snapshot(
            authority, source_home, selector, peer_subject, "output_discard", params, attested, checked
        )
        cleanup_exact(
            target,
            conn,
            scope,
            record["items"],
            record["blobs"],
            authorize=authorize,
        )
        done = {**record, "state": "completed", "items": [], "blobs": []}
        saved["owner_output_discard"] = done
        conn.execute("UPDATE state_meta SET value=? WHERE key=?", (_json(saved), key))
        return done

    try:
        done = authority.db._execute_write(complete)
    except (OSError, RoomArtifactError) as exc:
        raise RuntimeStoreError("storage_unavailable") from exc
    return {"discarded": True, "removed": done["removed"]}


@dataclass
class ServedNamedOutputCustody:
    service: object
    scope: RoomArtifactScope
    manifest: dict
    result: dict
    cancel_generation: int

    def __post_init__(self):
        self.manifest = copy.deepcopy(self.manifest)
        self.result = copy.deepcopy(self.result)
        self.receipt = validate_owner_output_receipt(self.result.get("owner_output_receipt"))
        validate_terminal_artifact_manifest(self.manifest)
        self.authority = self.service.authority
        self.registry = self.authority.runner.session_authorities
        self.source_epoch = self.authority.epoch
        self.source_instance = self.authority.instance_id
        self.target_home = Path(self.service.profile_homes()[self.scope.target_profile])
        from gateway.session_hosted_transport import HostedRoomOwnerRPC

        self.rpc = HostedRoomOwnerRPC(
            home=self.target_home,
            source_home=self.authority.profile_id,
            room_id=self.scope.room_id,
            member_id=self.scope.member_id,
            profile=self.scope.target_profile,
        )
        with self.authority.db._read_ctx() as conn:
            self.check_current(conn, self.scope)

    def check_current(self, conn, scope):
        from gateway.runtime_ownership import process_ownership
        from gateway.session_authorities import authority_for_home
        from gateway.hosted_room_output_fence import require_output_task

        if (
            scope != self.scope
            or self.service.authority is not self.authority
            or self.authority.hosted_room_service is not self.service
            or self.authority.runner.session_authorities is not self.registry
            or authority_for_home(self.authority.runner, self.authority.profile_id) is not self.authority
            or self.registry.for_home(self.target_home) is None
            or (self.authority.epoch, self.authority.instance_id)
            != (self.source_epoch, self.source_instance)
            or self.authority.db._db_file_was_replaced()
            or not process_ownership.owns(Path(self.authority.profile_id))
            or Path(self.service.profile_homes().get(scope.target_profile, "")).resolve()
            != self.target_home.resolve()
        ):
            raise RoomArtifactError("Group Chat output owner changed")
        _epoch(conn, self.source_epoch)
        task = require_output_task(conn, scope, self.cancel_generation, cleanup=True)
        if (
            json.loads(task["result_json"]) != self.result
            or self.result.get("artifact_scope") != scope.as_mapping()
            or self.result.get("artifacts") != self.manifest
            or self.result.get("owner_output_receipt") != self.receipt
        ):
            raise RoomArtifactError("Group Chat output receipt changed")
        key = (scope.room_id, scope.task_id, scope.execution_generation)
        current = self.service._output_metadata(conn, key)
        retry = conn.execute(
            "SELECT * FROM hosted_room_artifact_retries WHERE room_id=? AND task_id=? AND execution_generation=?",
            key,
        ).fetchone()
        if retry is None or retry["blocked"]:
            raise RoomArtifactError("Group Chat output reservation missing")
        retained = json.loads(retry["metadata_json"])
        if any(retained.get(name) != current[name] for name in ("work", "route", "lineage", "member_id")):
            raise RoomArtifactError("Group Chat output reservation changed")
        return task

    def _common(self):
        return {
            "session_id": self.receipt["target_session_id"],
            "task": {
                "room_id": self.scope.room_id,
                "task_id": self.scope.task_id,
                "thread_id": self._task_identity().thread_id,
                "turn_id": self._task_identity().turn_id,
            },
            "execution_generation": self.scope.execution_generation,
            "artifact_scope": self.scope.as_mapping(),
            "manifest_digest": self.manifest["manifest_digest"],
            "owner_output_receipt": self.receipt,
        }

    def _task_identity(self):
        matches = [
            task["identity"]
            for task in list_tasks(self.service.db_path, room_id=self.scope.room_id)
            if task["identity"].task_id == self.scope.task_id
            and task["execution_generation"] == self.scope.execution_generation
        ]
        if len(matches) != 1:
            raise RoomArtifactError("Group Chat output task changed")
        return matches[0]

    def _recheck(self):
        with self.authority.db._read_ctx() as conn:
            self.check_current(conn, self.scope)

    def read(self, scope, artifact_id):
        item = next((item for item in self.manifest["items"] if item["artifact_id"] == artifact_id), None)
        if item is None or item["size"] > MAX_ITEM_BYTES:
            raise RoomArtifactError("Group Chat output file is unavailable")
        data = bytearray()
        owner = None
        while len(data) < item["size"]:
            result = self.rpc.output_export(
                **self._common(), artifact_id=artifact_id, offset=len(data)
            )
            raw = _validate_export_response(result, item, len(data), self.receipt["receipt_id"])
            identity = (
                result["target_profile_id_sha256"],
                result["serving_owner_epoch"],
                result["serving_instance_id"],
            )
            if owner is None:
                owner = identity
            elif identity != owner:
                raise RoomArtifactError("Group Chat output owner changed during read")
            data.extend(raw)
            self._recheck()
        if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RoomArtifactError("Group Chat output bytes changed")
        return item, bytes(data)

    def retirement_complete(self, scope):
        return False

    def acknowledge(self, scope, artifact_ids, *, message_event_id):
        expected = [item["artifact_id"] for item in self.manifest["items"]]
        if scope != self.scope or artifact_ids != expected:
            raise RoomArtifactError("Group Chat output acknowledgement changed")
        result = self.rpc.output_ack(
            **self._common(), artifact_ids=artifact_ids, message_event_id=message_event_id
        )
        if (
            type(result) is not dict
            or set(result) != {"acknowledged", "changed"}
            or result["acknowledged"] is not True
            or type(result["changed"]) is not int
            or not 0 <= result["changed"] <= len(expected)
        ):
            raise RoomArtifactError("Group Chat output ACK was not confirmed")
        self._recheck()
        return result["changed"]

    def discard_durably(self, scope):
        if scope != self.scope:
            raise RoomArtifactError("Group Chat output discard changed")
        result = self.rpc.output_discard(
            **self._common(), reason_code="unpublished", result_digest=self.receipt["result_digest"]
        )
        if (
            type(result) is not dict
            or set(result) != {"discarded", "removed"}
            or result["discarded"] is not True
            or type(result["removed"]) is not int
            or result["removed"] != len(self.manifest["items"])
        ):
            raise RoomArtifactError("Group Chat output discard was not confirmed")
        self._recheck()
        return result["removed"]


def _validate_export_response(result, item, offset, receipt_id):
    required = {
        "version",
        "receipt_id",
        "artifact_id",
        "offset",
        "total_size",
        "sha256",
        "data_base64",
        "eof",
        "target_profile_id_sha256",
        "serving_owner_epoch",
        "serving_instance_id",
    }
    if type(result) is not dict or set(result) != required:
        raise RoomArtifactError("Group Chat output chunk is invalid")
    try:
        raw = base64.b64decode(result["data_base64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise RoomArtifactError("Group Chat output chunk is invalid") from exc
    expected = min(CHUNK_BYTES, item["size"] - offset)
    if (
        type(result["version"]) is not int
        or result["version"] != 1
        or result["receipt_id"] != receipt_id
        or result["artifact_id"] != item["artifact_id"]
        or type(result["offset"]) is not int
        or result["offset"] != offset
        or type(result["total_size"]) is not int
        or result["total_size"] != item["size"]
        or result["sha256"] != item["sha256"]
        or len(raw) != expected
        or result["eof"] is not (offset + expected == item["size"])
        or not isinstance(result["target_profile_id_sha256"], str)
        or _HEX.fullmatch(result["target_profile_id_sha256"]) is None
        or type(result["serving_owner_epoch"]) is not int
        or result["serving_owner_epoch"] < 1
        or not isinstance(result["serving_instance_id"], str)
        or not result["serving_instance_id"]
    ):
        raise RoomArtifactError("Group Chat output chunk changed")
    return raw
