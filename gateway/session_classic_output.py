"""Canonical admission and lifecycle adapter for classic Desktop exports."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from gateway.classic_output_exports import (
    CANONICAL_BINDING_VERSION,
    CANONICAL_MARKER_FIELDS,
    ClassicExports,
    transition_canonical_export,
    validate_write,
)
from gateway.config import Platform
from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactOutbox
from gateway.hosted_room_artifacts_classic import ClassicExportScope, identifier
from gateway.hosted_room_output_discard import OutputCleanupUnavailable, unlink_blob_names
from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError, _epoch, get_session_admission


@dataclass
class ClassicOutputBinding:
    authority: object
    ref: SessionRef
    row: dict
    store: ClassicExports
    export: dict
    group_id: str
    principal_id: str
    owner_pid: int
    active: bool = True
    used: bool = False

    @property
    def scope(self) -> ClassicExportScope:
        return self.store.scope(self.export)

    def check_write(self, conn, scope) -> None:
        if (
            not self.active
            or os.getpid() != self.owner_pid
            or scope != self.scope
        ):
            raise RoomArtifactError("Classic export producer is no longer active")
        _epoch(conn, self.authority.epoch)
        admission = conn.execute(
            "SELECT status,owner_epoch,generation,principal_id,target_session_id,request_id "
            "FROM session_admissions WHERE admission_id=?",
            (self.row["admission_id"],),
        ).fetchone()
        if (
            admission is None
            or admission["status"] != "started"
            or admission["owner_epoch"] != self.authority.epoch
            or admission["generation"] != self.row["generation"]
            or admission["principal_id"] != self.principal_id
            or admission["target_session_id"] != self.ref.session_id
            or admission["request_id"] != self.export["request_id"]
        ):
            raise RoomArtifactError("Classic export admission changed")
        validate_write(conn, scope)

    def outbox(self) -> RoomArtifactOutbox:
        if not self.active or os.getpid() != self.owner_pid:
            raise RoomArtifactError("Classic export producer is unavailable")
        self.used = True
        return RoomArtifactOutbox(
            self.authority.db.db_path,
            authorize_write=self.check_write,
        )


_CURRENT: ContextVar[ClassicOutputBinding | None] = ContextVar(
    "canonical_classic_output", default=None
)


def _unavailable(exc=None):
    error = RuntimeStoreError("classic_export_unavailable")
    if exc is not None:
        error.__cause__ = exc
    return error


def _root_home(authority) -> Path:
    from gateway.session_authorities import served_profile_name

    home = Path(authority.profile_id)
    path = Path(authority.db.db_path)
    if (
        not home.is_absolute()
        or home != home.resolve()
        or home.parent.name == "profiles"
        or served_profile_name(home) != "default"
        or path != home / "state.db"
        or path != path.resolve()
    ):
        raise _unavailable()
    return home


def _installation(authority) -> str:
    _root_home(authority)
    from gateway.hosted_rooms import HostedRoomError, local_authority_gateway_id

    try:
        return local_authority_gateway_id()
    except (HostedRoomError, OSError, ValueError) as exc:
        raise _unavailable(exc) from exc


def _status(store: ClassicExports, row: dict) -> dict:
    try:
        return store.status(row["export_id"])
    except (RoomArtifactError, ValueError, TypeError, KeyError) as exc:
        raise _unavailable(exc) from exc


def _canonical_marker(row: dict):
    marker = row.get("payload", {}).get("classic_export_v1")
    if (
        not isinstance(marker, dict)
        or set(marker) != CANONICAL_MARKER_FIELDS
        or marker.get("binding_version") != CANONICAL_BINDING_VERSION
        or marker.get("principal_id") != row.get("principal_id")
    ):
        return None
    return marker


def prepare_submission(connection, ref: SessionRef, request, text: str, request_id: str):
    """Admit one explicit classic export before the canonical input is committed."""

    authority, actor = connection.authority, connection.actor
    authority.authorize(actor, ref, "session:submit")
    home = _root_home(authority)
    live = authority.sessions.get(ref.session_id)
    if (
        live is None
        or live.source is None
        or live.source.platform != Platform.LOCAL
        or not isinstance(request, dict)
        or request.get("request_id") != request_id
    ):
        raise _unavailable()
    try:
        store = ClassicExports(home)
        row, fresh = store.admit(
            ref.session_id, request, text, principal_id=actor.subject
        )
        binding = json.loads(row["binding"])
        marker = {
            "export_id": row["export_id"],
            "generation": row["generation"],
            "group_id": binding["group_id"],
            "principal_id": binding["principal_id"],
            "binding_version": binding["binding_version"],
        }
        return store, row, fresh, marker
    except (RoomArtifactError, ValueError, TypeError, KeyError) as exc:
        raise _unavailable(exc) from exc


def abort_submission(prepared, authority, expected_payload) -> None:
    if prepared is None:
        return
    store, row, fresh, _marker = prepared
    if not fresh:
        return
    try:
        binding = json.loads(row["binding"])
        with authority.db._read_ctx() as conn:
            admission = conn.execute(
                """SELECT payload_json FROM session_admissions
                   WHERE principal_id=? AND target_session_id=? AND request_id=?""",
                (
                    binding["principal_id"],
                    row["session_key"],
                    row["request_id"],
                ),
            ).fetchone()
        if (
            admission is not None
            and expected_payload is not None
            and json.loads(admission["payload_json"]) == expected_payload
        ):
            return
        store.release_prepared(row)
    except (RoomArtifactError, OSError, sqlite3.Error, ValueError, KeyError):
        # A preparation that cannot be proved removable remains running and
        # unreadable; preserve the original submit error for retry/recovery.
        pass


def submission_status(prepared) -> dict:
    store, row, _fresh, _marker = prepared
    return _status(store, row)


def _binding(authority, ref: SessionRef, row: dict):
    marker = _canonical_marker(row)
    if marker is None:
        return None
    home = _root_home(authority)
    live = authority.sessions.get(ref.session_id)
    if live is None or live.source is None or live.source.platform != Platform.LOCAL:
        return None
    try:
        with authority.db._read_ctx() as conn:
            export = conn.execute(
                "SELECT export_id,profile_home,session_key,request_id,generation,binding,state,expires,text "
                "FROM classic_output_exports WHERE export_id=?",
                (marker["export_id"],),
            ).fetchone()
        if export is None:
            return None
        export = dict(export)
        binding = json.loads(export["binding"])
        if (
            export["profile_home"] != str(home)
            or export["session_key"] != ref.session_id
            or export["request_id"] != row.get("request_id")
            or export["generation"] != marker["generation"]
            or export["state"] != "running"
            or binding.get("group_id") != marker["group_id"]
            or binding.get("binding_version") != CANONICAL_BINDING_VERSION
            or binding.get("principal_id") != marker["principal_id"]
            or binding.get("prompt_sha256")
            != hashlib.sha256(row["payload"]["text"].encode()).hexdigest()
        ):
            return None
        store = ClassicExports(home)
        result = ClassicOutputBinding(
            authority,
            ref,
            row,
            store,
            export,
            binding["group_id"],
            marker["principal_id"],
            os.getpid(),
        )
        with authority.db._read_ctx() as conn:
            result.check_write(conn, result.scope)
        return result
    except (RoomArtifactError, RuntimeStoreError, sqlite3.Error, ValueError, TypeError, KeyError):
        return None


@contextmanager
def classic_output_scope(authority, ref: SessionRef, row: dict):
    binding = _binding(authority, ref, row)
    token = _CURRENT.set(binding)
    try:
        yield binding
    finally:
        if binding is not None:
            binding.active = False
        _CURRENT.reset(token)


def current_classic_output_binding():
    binding = _CURRENT.get()
    if (
        binding is None
        or not binding.active
        or binding.owner_pid != os.getpid()
    ):
        return None
    return binding


def classic_turn_toolsets(enabled):
    if current_classic_output_binding() is None or enabled is None:
        return enabled
    values = list(enabled)
    if "bot_room" not in values:
        values.append("bot_room")
    return sorted(values)


def terminal_write(authority, row: dict):
    """Return the classic mutation consumed by canonical terminal writers."""

    marker = _canonical_marker(row)
    if marker is None:
        return None
    home = _root_home(authority)
    with authority.db._read_ctx() as conn:
        if conn.execute(
            "SELECT 1 FROM classic_output_exports WHERE export_id=?",
            (marker["export_id"],),
        ).fetchone() is None:
            # A private in-process marker without producer custody is inert.
            return None

    def write(conn, admission, outcome, result):
        if (
            admission.get("admission_id") != row.get("admission_id")
            or _canonical_marker(admission) != marker
        ):
            raise RoomArtifactError("Classic canonical admission changed")
        transition_canonical_export(
            conn,
            admission=admission,
            marker=marker,
            outcome=outcome,
            result=result,
            home=home,
        )

    return write


def _cleanup_retired_scopes(authority, scopes, *, authorize=None):
    """Remove only committed retired custody on the still-current owner."""
    def cleanup(conn):
        _epoch(conn, authority.epoch)
        if authorize is not None:
            authorize(conn)
        outbox = RoomArtifactOutbox.borrow_existing(authority.db, conn)
        for scope in scopes:
            export = conn.execute(
                "SELECT state FROM classic_output_exports WHERE export_id=? AND generation=?",
                (scope.export_id, scope.execution_generation),
            ).fetchone()
            rows = conn.execute(
                "SELECT blob_name,cleanup_required_at FROM hosted_room_output_artifacts WHERE scope_key=?",
                (scope.key,),
            ).fetchall()
            if export is None and not rows:
                continue
            if export is None or export["state"] != "retired":
                raise RoomArtifactError("Classic custody is not retired")
            if any(row["cleanup_required_at"] is None for row in rows):
                raise RoomArtifactError("Classic cleanup obligation is unavailable")
            outbox._retire_generation(conn, scope)
            if rows:
                unlink_blob_names(outbox, [row["blob_name"] for row in rows])
            conn.execute("DELETE FROM hosted_room_output_artifacts WHERE scope_key=?", (scope.key,))
    authority.db._execute_write(cleanup)


def cleanup_terminal(authority, row: dict, terminal: dict) -> bool:
    """Attempt physical cleanup only after canonical retirement committed."""

    marker = _canonical_marker(row)
    if marker is None:
        return True
    try:
        _root_home(authority)
        with authority.db._read_ctx() as conn:
            export = conn.execute(
                "SELECT state FROM classic_output_exports WHERE export_id=? AND generation=?",
                (marker["export_id"], marker["generation"]),
            ).fetchone()
        if export is not None and export["state"] != "retired":
            return True
        _cleanup_retired_scopes(authority, [ClassicExportScope(marker["export_id"], marker["generation"])])
    except (RoomArtifactError, OutputCleanupUnavailable, RuntimeStoreError,
            OSError, sqlite3.Error, ValueError, TypeError, KeyError):
        return False
    return True


def receipt_status(connection, ref: SessionRef, admission_id: str):
    row = get_session_admission(connection.authority.db, admission_id=admission_id)
    if row is None or row["target_session_id"] != ref.session_id:
        return None
    marker = _canonical_marker(row)
    if marker is None:
        return None
    connection.authority.authorize(connection.actor, ref, "session:read")
    store = ClassicExports(_root_home(connection.authority))
    try:
        export = store.lookup(marker.get("export_id"))
    except (RoomArtifactError, ValueError, TypeError) as exc:
        raise _unavailable(exc) from exc
    if export["session_key"] != ref.session_id or export["request_id"] != row["request_id"]:
        raise _unavailable()
    return _status(store, export)


def capabilities(connection, params):
    if params:
        raise RuntimeStoreError("invalid_params")
    if not connection.actor.capabilities:
        raise RuntimeStoreError("permission_denied")
    try:
        installation = _installation(connection.authority)
    except RuntimeStoreError:
        return {"classic_output_export_v1": False}
    from hermes_cli.active_sessions import PER_SESSION_EXCLUSIVE_SUBMIT

    return {
        "per_session_exclusive_submit": bool(PER_SESSION_EXCLUSIVE_SUBMIT),
        "classic_output_export_v1": True,
        "installation": installation,
        "classic_exact_generation_v1": True,
    }


def _status_read(connection, ref: SessionRef, params):
    allowed = {"session_id", "installation", "group_id", "request_id", "export_id", "profile"}
    if (
        not isinstance(params, dict)
        or set(params) - allowed
        or ("request_id" in params) == ("export_id" in params)
        or "artifact_id" in params
    ):
        raise RuntimeStoreError("invalid_params")
    try:
        for key in ("session_id", "installation", "group_id"):
            identifier(params[key])
        selector = params.get("request_id", params.get("export_id"))
        identifier(selector)
        if params.get("profile", "default") != "default":
            raise ValueError("invalid profile")
    except (KeyError, RoomArtifactError, ValueError, TypeError) as exc:
        raise RuntimeStoreError("invalid_params") from exc
    from gateway.session_classic_exports import _authorize_bound

    _authorize_bound(connection, ref, connection.authority.epoch)
    if params["installation"] != _installation(connection.authority):
        raise _unavailable()
    store = ClassicExports(_root_home(connection.authority))
    try:
        row = (
            store.prior(ref.session_id, params["request_id"])
            if "request_id" in params
            else store.lookup(params["export_id"])
        )
    except (RoomArtifactError, ValueError, TypeError) as exc:
        raise _unavailable(exc) from exc
    status = _status(store, row)
    if row["session_key"] != ref.session_id or status["group_id"] != params["group_id"]:
        raise _unavailable()
    return status


async def read(connection, ref: SessionRef, params):
    if isinstance(params, dict) and "artifact_id" in params:
        from gateway.session_classic_exports import read_classic_export

        return await read_classic_export(connection, ref, params)
    return _status_read(connection, ref, params)


async def discard(connection, ref: SessionRef, params):
    allowed = {"session_id", "installation", "group_id", "export_id", "profile"}
    if not isinstance(params, dict) or set(params) - allowed:
        raise RuntimeStoreError("invalid_params")
    try:
        for key in ("session_id", "installation", "group_id"):
            identifier(params[key])
        if "export_id" in params:
            identifier(params["export_id"])
        if params.get("profile", "default") != "default":
            raise ValueError("invalid profile")
    except (KeyError, RoomArtifactError, ValueError, TypeError) as exc:
        raise RuntimeStoreError("invalid_params") from exc
    from gateway.session_classic_exports import _current

    authority = connection.authority
    epoch = authority.epoch

    def authorize(conn):
        _epoch(conn, epoch)
        _home, _path, installation, _live = _current(connection, ref, epoch)
        authority.authorize(connection.actor, ref, "session:control")
        if params["installation"] != installation:
            raise _unavailable()

    def retire(conn):
        authorize(conn)
        home = str(_root_home(authority))
        rows = conn.execute(
            "SELECT * FROM classic_output_exports WHERE profile_home=? "
            "AND json_extract(binding,'$.group_id')=?",
            (home, params["group_id"]),
        ).fetchall()
        if "export_id" in params:
            rows = [row for row in rows if row["export_id"] == params["export_id"]]
            if len(rows) != 1 or rows[0]["session_key"] != ref.session_id:
                raise _unavailable()
        elif not any(row["session_key"] == ref.session_id for row in rows):
            raise _unavailable()
        outbox = RoomArtifactOutbox.borrow_existing(authority.db, conn)
        if "export_id" not in params:
            from gateway.classic_output_exports import MAX_EXPORTS
            existing = conn.execute(
                "SELECT 1 FROM classic_retired_groups WHERE profile_home=? AND group_id=?",
                (home, params["group_id"]),
            ).fetchone()
            if not existing and conn.execute("SELECT COUNT(*) FROM classic_retired_groups").fetchone()[0] >= MAX_EXPORTS:
                raise _unavailable()
            conn.execute("INSERT OR IGNORE INTO classic_retired_groups VALUES (?,?)", (home, params["group_id"]))
        scopes = [ClassicExports.scope(row) for row in rows]
        import time
        for row, scope in zip(rows, scopes):
            conn.execute("UPDATE classic_output_exports SET state='retired' WHERE export_id=?", (row["export_id"],))
            outbox._retire_generation(conn, scope)
            conn.execute("UPDATE hosted_room_output_artifacts SET cleanup_required_at=? WHERE scope_key=?",
                         (time.time(), scope.key))
        return scopes

    try:
        scopes = authority.db._execute_write(retire)
        _cleanup_retired_scopes(authority, scopes, authorize=authorize)
    except RuntimeStoreError:
        raise
    except (RoomArtifactError, OutputCleanupUnavailable, OSError, sqlite3.Error,
            ValueError, TypeError, KeyError) as exc:
        raise _unavailable(exc) from exc
    return {"retired": True}
