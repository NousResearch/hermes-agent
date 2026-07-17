"""Fixed live edge for one stopped Canonical Brain schema reconciliation.

The module is deliberately not a migration runner.  Every selectable value is
derived from the sealed interpreter, the stopped-release receipt, or the one
reviewed reconciliation plan.  The only secret crossing this boundary is the
64-byte owner-created temporary administrator credential.  It is copied to a
root-owned 0400 memfd, used to authenticate one bounded PostgreSQL session,
and then zeroized before the signed preflight challenge is returned.

Cloud SQL user creation/deletion remains owner-side.  This runtime performs no
Cloud mutation.  After the owner proves that the temporary administrator was
deleted, it opens a distinct fixed-writer session which can prove the terminal
authority and schema contract without reusing the administrator credential or
broadening the writer's permanent Canonical Brain data-read authority.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import stat
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Protocol, Sequence

from gateway import canonical_writer_foundation as foundation
from gateway import canonical_writer_foundation_phase_b as phase_b
from gateway import canonical_writer_phase_b_runtime as phase_b_runtime
from gateway import canonical_writer_schema_reconciliation_bootstrap as bootstrap
from gateway.canonical_canary_host_identity import (
    FULL_CANARY_HOST_IDENTITY_SCHEMA,
    collect_dedicated_canary_host_identity_receipt,
)
from gateway.canonical_writer_activation import CANARY_WRITER_GID
from gateway.canonical_writer_db import (
    CredentialSource,
    ManagedCloudSQLAdminHBAReceipt,
    WriterDBConfig,
    _open_postgres_session,
    collect_managed_cloudsqladmin_hba_receipt,
)
from gateway.canonical_writer_planner import load_release_manifest
from gateway.canonical_writer_preflight_publisher import (
    _load_host_receipt,
    _load_stopped_release_receipt,
    _read_trusted_file,
)
from gateway.canonical_writer_schema_reconciliation import (
    AppendOnlySchemaReconciliationJournal,
    CanonicalTruthReceipt,
    SchemaContract,
    SchemaReconciliationAuthorization,
    SchemaReconciliationError,
    SchemaReconciliationPlan,
    build_schema_reconciliation_plan,
    execute_schema_reconciliation,
    load_release_schema_contract_asset,
    preflight_schema_reconciliation,
)
from gateway.canonical_writer_schema_reconciliation_db import (
    PostDeleteTerminalReceipt,
    PostgresSchemaReconciliationDatabase,
    _split_sealed_mutation_sql,
    collect_post_delete_terminal_receipt,
    validate_post_delete_terminal_receipt,
)


OWNER_PUBLIC_KEY_ED25519_HEX = (
    "4f928fd117e2e62f1e52b0095d6ab5524370707f5e9b295efdc62479ce887e26"
)
OWNER_KEY_ID = "d9229ecb5084f4a78c8887b07effc6259355ccceecbe9b4ad55994e070d674c1"
OWNER_PUBLIC_FINGERPRINT = "SHA256:7Ea5WNys9ui7FL/p0FlOnL1ZLr6NPFuewekwqRw/rdw"
OWNER_SUBJECT_SHA256 = (
    "d75f30ec2d057e67b14ed88bc7e8bcebb429b1a40a4171030fe809908d93491a"
)

RUNTIME_SCHEMA = "muncho-canonical-writer-schema-reconciliation-runtime.v2"
DATABASE_IDENTITY_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-database-identity.v1"
)
HOST_STATE_SCHEMA = "muncho-canonical-writer-schema-reconciliation-host-state.v1"
SERVICES_STATE_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-services-state.v1"
)

EVIDENCE_ROOT = Path("/var/lib/muncho-writer-canary-evidence/schema-reconciliation")
GATE_TTL_SECONDS = bootstrap.MAX_GATE_TTL_SECONDS
TEMPORARY_ADMIN_PREFIX = "muncho_canary_admin_"
OPAQUE_CREDENTIAL_BYTES = 64
EXPECTED_PYTHON_VERSION = "3.11.15"
_ROOT_UID = 0
_ROOT_GID = 0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_URLSAFE_CREDENTIAL = re.compile(rb"^[A-Za-z0-9_-]{64}$")

_PACKAGED_RELEASE_FILES: Mapping[str, str] = {
    "gateway/canonical_writer_schema_reconciliation.py": "0444",
    "gateway/canonical_writer_schema_reconciliation_db.py": "0444",
    "gateway/canonical_writer_schema_reconciliation_bootstrap.py": "0444",
    "gateway/canonical_writer_schema_reconciliation_runtime.py": "0444",
    "gateway/assets/canonical_writer_schema_contract_v1.json": "0444",
}
_ROOT_RELEASE_FILES: Mapping[str, str] = {
    "gateway/assets/canonical_writer_schema_contract_v1.json": "0444",
    "scripts/sql/canonical_writer_v1.sql": "0444",
}


class SchemaReconciliationRuntimeError(
    bootstrap.SchemaReconciliationBootstrapError
):
    """Stable, secret-free fixed-runtime failure."""

    def __init__(self, code: str) -> None:
        if (
            not isinstance(code, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{2,95}", code) is None
        ):
            code = "schema_reconciliation_runtime_failed"
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise SchemaReconciliationRuntimeError(code)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_value_invalid"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _hashed(unsigned: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    value = copy.deepcopy(dict(unsigned))
    value[field_name] = _sha256_json(value)
    return value


def _zeroize(value: bytearray | None) -> None:
    if value is None:
        return
    try:
        value[:] = b"\x00" * len(value)
    except (BufferError, TypeError, ValueError):
        pass


def _derived_owner_identity() -> Mapping[str, str]:
    try:
        public = bytes.fromhex(OWNER_PUBLIC_KEY_ED25519_HEX)
    except ValueError as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_owner_pin_invalid"
        ) from exc
    algorithm = b"ssh-ed25519"
    wire = (
        struct.pack(">I", len(algorithm))
        + algorithm
        + struct.pack(">I", len(public))
        + public
    )
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(wire).digest()).decode(
        "ascii"
    ).rstrip("=")
    key_id = _sha256_bytes(public)
    if (
        len(public) != 32
        or key_id != OWNER_KEY_ID
        or fingerprint != OWNER_PUBLIC_FINGERPRINT
        or _SHA256.fullmatch(OWNER_SUBJECT_SHA256) is None
    ):
        _fail("schema_reconciliation_runtime_owner_pin_invalid")
    return {
        "owner_public_key_ed25519_hex": OWNER_PUBLIC_KEY_ED25519_HEX,
        "owner_key_id": key_id,
        "owner_public_fingerprint": fingerprint,
        "owner_subject_sha256": OWNER_SUBJECT_SHA256,
    }


def _require_root_hardened() -> None:
    try:
        phase_b_runtime._require_root_linux()
        phase_b_runtime._harden_phase_b_process()
    except BaseException as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_process_boundary_invalid"
        ) from exc


def _read_fixed_ca() -> bytes:
    return _read_trusted_file(
        foundation.DATABASE_CA_PATH,
        expected_uid=_ROOT_UID,
        expected_gid=CANARY_WRITER_GID,
        allowed_modes=frozenset({0o440}),
        maximum=2 * 1024 * 1024,
        allowed_parent_gids=frozenset({0, CANARY_WRITER_GID}),
    )


def _stable_regular_file_sha256(
    path: Path,
    *,
    expected_mode: int,
    maximum_bytes: int,
) -> str:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != _ROOT_UID
            or before.st_gid != _ROOT_GID
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != expected_mode
            or not 0 < before.st_size <= maximum_bytes
        ):
            _fail("schema_reconciliation_runtime_release_invalid")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > maximum_bytes:
                _fail("schema_reconciliation_runtime_release_invalid")
            digest.update(chunk)
        after = os.fstat(descriptor)
        reachable = path.lstat()
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_uid,
            item.st_gid,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if (
            size != before.st_size
            or identity(before) != identity(opened)
            or identity(before) != identity(after)
            or identity(before) != identity(reachable)
        ):
            _fail("schema_reconciliation_runtime_release_invalid")
        return digest.hexdigest()
    except SchemaReconciliationRuntimeError:
        raise
    except OSError as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_release_invalid"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _required_release_files(manifest: Any) -> Mapping[str, str]:
    python_version = getattr(manifest, "python_version", None)
    if python_version != EXPECTED_PYTHON_VERSION:
        _fail("schema_reconciliation_runtime_release_invalid")
    major, minor, _patch = python_version.split(".")
    site_packages = f"venv/lib/python{major}.{minor}/site-packages"
    return {
        **{
            f"{site_packages}/{path}": mode
            for path, mode in _PACKAGED_RELEASE_FILES.items()
        },
        **_ROOT_RELEASE_FILES,
    }


def _validate_release_files(manifest: Any) -> None:
    required = _required_release_files(manifest)
    entries = {
        entry.path: entry
        for entry in getattr(manifest, "entries", ())
        if entry.path in required
    }
    if set(entries) != set(required) or any(
        entries[path].kind != "file"
        or entries[path].mode != mode
        or entries[path].size <= 0
        or _SHA256.fullmatch(entries[path].sha256) is None
        for path, mode in required.items()
    ):
        _fail("schema_reconciliation_runtime_release_invalid")


def _require_activation_inventory_absent(
    stopped: Mapping[str, Any],
    *,
    path_exists: Callable[[str], bool] = os.path.lexists,
) -> None:
    inventory = stopped.get("activation_inventory")
    if not isinstance(inventory, list) or not inventory:
        _fail("schema_reconciliation_runtime_activation_inventory_invalid")
    for item in inventory:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "state"}
            or item.get("state") != "absent"
            or not isinstance(item.get("path"), str)
            or not str(item["path"]).startswith("/")
            or path_exists(str(item["path"]))
        ):
            _fail("schema_reconciliation_runtime_activation_inventory_invalid")


def _release_binding(
    *,
    revision: str,
    manifest: Any,
    manifest_raw: bytes,
    stopped: Mapping[str, Any],
    stopped_raw: bytes,
    interpreter_sha256: Callable[[Path], str],
    path_exists: Callable[[str], bool],
) -> Mapping[str, Any]:
    """Return the exact sealed-release identity proved at both protocol ends."""

    _validate_release_files(manifest)
    _require_activation_inventory_absent(stopped, path_exists=path_exists)
    release_root = Path("/opt/muncho-canary-releases") / revision
    interpreter = release_root / "venv/bin/python"
    try:
        current_interpreter_sha256 = interpreter_sha256(interpreter)
        stopped_receipt_sha256 = stopped.get("receipt_sha256")
        activation_inventory_sha256 = _sha256_json(
            stopped.get("activation_inventory")
        )
    except SchemaReconciliationRuntimeError:
        raise
    except BaseException as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_release_binding_invalid"
        ) from exc
    manifest_sha256 = _sha256_bytes(manifest_raw)
    stopped_file_sha256 = _sha256_bytes(stopped_raw)
    artifact_sha256 = getattr(manifest, "artifact_sha256", None)
    if (
        getattr(manifest, "revision", None) != revision
        or getattr(manifest, "artifact_root", None) != str(release_root)
        or getattr(manifest, "python_version", None) != EXPECTED_PYTHON_VERSION
        or getattr(manifest, "interpreter", None) != str(interpreter)
        or stopped.get("release_revision") != revision
        or stopped.get("release_root") != str(release_root)
        or stopped.get("release_manifest_path")
        != str(release_root / "release-manifest.json")
        or stopped.get("release_manifest_file_sha256") != manifest_sha256
        or stopped.get("release_artifact_sha256") != artifact_sha256
        or stopped.get("python_version") != EXPECTED_PYTHON_VERSION
        or stopped.get("interpreter") != str(interpreter)
        or stopped.get("interpreter_sha256") != current_interpreter_sha256
        or not isinstance(stopped_receipt_sha256, str)
        or _SHA256.fullmatch(stopped_receipt_sha256) is None
        or not isinstance(artifact_sha256, str)
        or _SHA256.fullmatch(artifact_sha256) is None
        or _SHA256.fullmatch(current_interpreter_sha256) is None
    ):
        _fail("schema_reconciliation_runtime_release_binding_invalid")
    return {
        "release_manifest_sha256": manifest_sha256,
        "stopped_release_receipt_file_sha256": stopped_file_sha256,
        "stopped_release_receipt_sha256": stopped_receipt_sha256,
        "release_artifact_sha256": artifact_sha256,
        "python_version": EXPECTED_PYTHON_VERSION,
        "interpreter_sha256": current_interpreter_sha256,
        "activation_inventory_sha256": activation_inventory_sha256,
    }


def _validate_host_receipt(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != FULL_CANARY_HOST_IDENTITY_SCHEMA
        or value.get("collector_authority") != "trusted_root_read_only_host_collector"
        or type(value.get("observed_at_unix")) is not int
        or value.get("receipt_sha256")
        != _sha256_json({
            name: item for name, item in value.items() if name != "receipt_sha256"
        })
    ):
        _fail("schema_reconciliation_runtime_host_invalid")
    for name in (
        "gce_identity_sha256",
        "machine_id_sha256",
        "hostname_sha256",
        "host_identity_sha256",
        "boot_id_sha256",
        "receipt_sha256",
    ):
        if _SHA256.fullmatch(str(value.get(name))) is None:
            _fail("schema_reconciliation_runtime_host_invalid")
    return copy.deepcopy(dict(value))


def _host_state(value: Mapping[str, Any]) -> Mapping[str, Any]:
    receipt = _validate_host_receipt(value)
    unsigned = {
        "schema": HOST_STATE_SCHEMA,
        "host_identity_sha256": receipt["host_identity_sha256"],
        "boot_id_sha256": receipt["boot_id_sha256"],
    }
    return _hashed(unsigned, "state_sha256")


def _services_state(value: Mapping[str, Any], *, revision: str) -> Mapping[str, Any]:
    try:
        validated = phase_b._validate_services(
            value,
            release_revision=revision,
        )
    except BaseException as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_services_invalid"
        ) from exc
    unsigned = {
        "schema": SERVICES_STATE_SCHEMA,
        "release_revision": revision,
        "services": copy.deepcopy(validated["services"]),
        "services_stopped_and_disabled": True,
    }
    return _hashed(unsigned, "state_sha256")


def _journal_head(
    plan: SchemaReconciliationPlan,
    journal: AppendOnlySchemaReconciliationJournal,
) -> Mapping[str, Any]:
    with journal.lock():
        intent = journal.load_authorized_intent(plan)
        terminal = journal.load_terminal(plan)
    if terminal is not None:
        if intent is None:
            _fail("schema_reconciliation_runtime_journal_invalid")
        state = "terminal"
    elif intent is not None:
        state = "authorized_intent"
    else:
        state = "empty"
    unsigned = {
        "schema": bootstrap.JOURNAL_HEAD_SCHEMA,
        "state": state,
        "authorized_intent_sha256": (
            None if intent is None else intent["authorized_intent_sha256"]
        ),
        "terminal_receipt_sha256": (
            None if terminal is None else terminal["receipt_sha256"]
        ),
    }
    return _hashed(unsigned, "head_sha256")


def _bridge_sql_sha256(plan: SchemaReconciliationPlan) -> str:
    segments = _split_sealed_mutation_sql(plan)
    return _sha256_bytes(
        (segments.authority_open + "\n\n" + segments.authority_close).encode(
            "utf-8", errors="strict"
        )
    )


class _Session(Protocol):
    username: str
    tls_peer_certificate_sha256: str

    def query(self, sql: str, *, maximum_rows: int) -> Any: ...

    def close(self) -> None: ...


class _BorrowedSession:
    def __init__(self, owner: "_AuthenticatedSessionLease") -> None:
        self._owner = owner
        self._closed = False

    @property
    def username(self) -> str:
        return self._owner.session.username

    @property
    def tls_peer_certificate_sha256(self) -> str:
        return self._owner.session.tls_peer_certificate_sha256

    def query(self, sql: str, *, maximum_rows: int) -> Any:
        if self._closed:
            _fail("schema_reconciliation_runtime_database_session_closed")
        return self._owner.session.query(sql, maximum_rows=maximum_rows)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._owner.release()


class _AuthenticatedSessionLease:
    """Sequential adapter borrows over one already-authenticated session."""

    def __init__(self, session: _Session, expected_config: Any) -> None:
        self.session = session
        self._expected_config = expected_config
        self._borrowed = False
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def borrow(self, config: WriterDBConfig) -> _BorrowedSession:
        if config is not self._expected_config or self._borrowed or self._closed:
            _fail("schema_reconciliation_runtime_database_session_busy")
        self._borrowed = True
        return _BorrowedSession(self)

    def release(self) -> None:
        if not self._borrowed or self._closed:
            _fail("schema_reconciliation_runtime_database_session_invalid")
        self._borrowed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.session.close()
        except BaseException as exc:
            raise SchemaReconciliationRuntimeError(
                "schema_reconciliation_runtime_database_close_failed"
            ) from exc


_DATABASE_IDENTITY_SQL = """
SELECT current_database()::text AS database_name,
       pg_catalog.current_setting('server_version_num')::int::text AS version_num,
       pg_catalog.pg_get_userbyid(database.datdba)::text AS database_owner,
       pg_catalog.pg_postmaster_start_time()::text AS postmaster_started
  FROM pg_catalog.pg_database AS database
 WHERE database.datname = current_database()
""".strip()


def _database_identity(session: _Session) -> Mapping[str, Any]:
    try:
        result = session.query(_DATABASE_IDENTITY_SQL, maximum_rows=1)
    except BaseException as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_database_identity_invalid"
        ) from exc
    if (
        tuple(getattr(result, "columns", ()))
        != ("database_name", "version_num", "database_owner", "postmaster_started")
        or len(tuple(getattr(result, "rows", ()))) != 1
        or len(tuple(result.rows[0])) != 4
        or any(not isinstance(item, str) or not item for item in result.rows[0])
    ):
        _fail("schema_reconciliation_runtime_database_identity_invalid")
    database, version_text, owner, postmaster_started = result.rows[0]
    try:
        version = int(version_text)
    except ValueError as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_database_identity_invalid"
        ) from exc
    peer = getattr(session, "tls_peer_certificate_sha256", None)
    if (
        database != foundation.SQL_DATABASE
        or version // 10000 != bootstrap.EXPECTED_POSTGRESQL_MAJOR
        or owner != foundation.DATABASE_OWNER_ROLE
        or not isinstance(peer, str)
        or _SHA256.fullmatch(peer) is None
    ):
        _fail("schema_reconciliation_runtime_database_identity_invalid")
    unsigned = {
        "schema": DATABASE_IDENTITY_SCHEMA,
        "project": foundation.PROJECT,
        "instance": foundation.SQL_INSTANCE,
        "host": foundation.SQL_HOST,
        "port": foundation.SQL_PORT,
        "database": database,
        "database_owner": owner,
        "postgresql_major": version // 10000,
        "tls_server_name": foundation.SQL_TLS_SERVER_NAME,
        "tls_peer_certificate_sha256": peer,
        "postmaster_started": postmaster_started,
    }
    return _hashed(unsigned, "identity_sha256")


def _temporary_owner_bridge_receipt(observed_at_unix: int) -> Mapping[str, Any]:
    unsigned = {
        "schema": bootstrap.TEMPORARY_OWNER_BRIDGE_SCHEMA,
        "transaction_isolation": "SERIALIZABLE",
        "database_roles": list(
            bootstrap.SCHEMA_RECONCILIATION_DATABASE_ROLES
        ),
        "provider_membership_count": 2,
        "admin_option": False,
        "inherit_option": True,
        "set_option": False,
        "cloudsqlsuperuser_absent": True,
        "canonical_truth_share_lock": True,
        "owner_authority_active_during_locked_collection": True,
        "current_user_remained_temporary_login": True,
        "exact_provider_memberships_during_locked_collection": True,
        "contract_collected_while_truth_lock_held": True,
        "canonical_truth_collected_while_truth_lock_held": True,
        "temporary_login_owned_objects": False,
        "memberships_remain_until_cloud_user_cleanup": True,
        "transaction_committed": True,
        "observed_at_unix": observed_at_unix,
        "secret_material_recorded": False,
    }
    return _hashed(unsigned, "receipt_sha256")


def _database_commit_attestation(
    *,
    database_identity: Mapping[str, Any],
    contract: SchemaContract,
    truth: CanonicalTruthReceipt,
    observed_at_unix: int,
) -> Mapping[str, Any]:
    unsigned = {
        "schema": bootstrap.DATABASE_COMMIT_ATTESTATION_SCHEMA,
        "ok": True,
        "database_identity_sha256": database_identity["identity_sha256"],
        "tls_peer_certificate_sha256": database_identity["tls_peer_certificate_sha256"],
        "postgresql_major": database_identity["postgresql_major"],
        "observed_contract_sha256": contract.sha256,
        "canonical_truth_receipt": truth.value,
        "transaction_committed": True,
        "temporary_owner_memberships_present": True,
        "temporary_login_owns_zero_objects": True,
        "trampoline_restored_before_commit": True,
        "cloud_user_cleanup_required": True,
        "database_session_closed": True,
        "re_attested_before_temporary_admin_delete": True,
        "observed_at_unix": observed_at_unix,
        "secret_material_recorded": False,
    }
    return _hashed(unsigned, "attestation_sha256")


@dataclass(frozen=True)
class _RuntimeDependencies:
    current_revision: Callable[[], str] = phase_b_runtime._current_release_revision
    load_manifest: Callable[[str], tuple[Any, bytes]] = load_release_manifest
    load_stopped: Callable[[str], tuple[Mapping[str, Any], bytes]] = (
        _load_stopped_release_receipt
    )
    load_historical_host: Callable[
        [Mapping[str, Any]], tuple[Mapping[str, Any], bytes]
    ] = _load_host_receipt
    collect_host: Callable[..., Mapping[str, Any]] = (
        collect_dedicated_canary_host_identity_receipt
    )
    collect_services: Callable[[str, int], Mapping[str, Any]] = (
        phase_b_runtime._collect_services
    )
    build_plan: Callable[[str], SchemaReconciliationPlan] = (
        build_schema_reconciliation_plan
    )
    load_target_asset: Callable[[str], Any] = load_release_schema_contract_asset
    writer_config: Callable[[], WriterDBConfig] = foundation._fixed_writer_config
    collect_hba: Callable[..., ManagedCloudSQLAdminHBAReceipt] = (
        collect_managed_cloudsqladmin_hba_receipt
    )
    open_session: Callable[[WriterDBConfig], _Session] = _open_postgres_session
    collect_post_delete_terminal: Callable[..., PostDeleteTerminalReceipt] = (
        collect_post_delete_terminal_receipt
    )
    journal_factory: Callable[[], AppendOnlySchemaReconciliationJournal] = lambda: (
        AppendOnlySchemaReconciliationJournal(EVIDENCE_ROOT)
    )
    random_bytes: Callable[[int], bytes] = os.urandom
    now: Callable[[], int] = lambda: int(time.time())
    path_exists: Callable[[str], bool] = os.path.lexists
    interpreter_sha256: Callable[[Path], str] = lambda path: (
        _stable_regular_file_sha256(
            path,
            expected_mode=0o555,
            maximum_bytes=256 * 1024 * 1024,
        )
    )
    read_ca: Callable[[], bytes] = _read_fixed_ca
    harden: Callable[[], None] = _require_root_hardened
    protocol_runner: Callable[..., Mapping[str, Any]] = bootstrap.run_protocol_v2


@dataclass
class _RuntimeContext:
    revision: str
    plan: SchemaReconciliationPlan
    target: SchemaContract
    journal: AppendOnlySchemaReconciliationJournal
    stopped: Mapping[str, Any]
    gate: Mapping[str, Any]
    initial_host_state: Mapping[str, Any]
    initial_services_state: Mapping[str, Any]
    initial_release_binding: Mapping[str, Any]
    dependencies: _RuntimeDependencies
    lease: _AuthenticatedSessionLease | None = None
    database: PostgresSchemaReconciliationDatabase | None = None
    preflight: Mapping[str, Any] | None = None
    truth: CanonicalTruthReceipt | None = None
    database_identity: Mapping[str, Any] | None = None
    writer_config: WriterDBConfig | None = None
    temporary_admin_database_closed_before_cleanup: bool = False

    def close_temporary_admin_database(self) -> None:
        if self.lease is not None:
            self.lease.close()
        self.temporary_admin_database_closed_before_cleanup = True


def _collect_stopped_boundary(
    context: _RuntimeContext,
    observed_at_unix: int,
) -> Mapping[str, Any]:
    """Re-read the exact sealed release, host, and stopped service boundary."""

    try:
        manifest, manifest_raw = context.dependencies.load_manifest(
            context.revision
        )
        stopped, stopped_raw = context.dependencies.load_stopped(
            context.revision
        )
        release_binding = _release_binding(
            revision=context.revision,
            manifest=manifest,
            manifest_raw=manifest_raw,
            stopped=stopped,
            stopped_raw=stopped_raw,
            interpreter_sha256=context.dependencies.interpreter_sha256,
            path_exists=context.dependencies.path_exists,
        )
        historical_host, _historical_host_raw = (
            context.dependencies.load_historical_host(stopped)
        )
        host = _validate_host_receipt(
            context.dependencies.collect_host(
                observed_at_unix=observed_at_unix
            )
        )
        services = context.dependencies.collect_services(
            context.revision,
            observed_at_unix,
        )
        return {
            "release_binding": release_binding,
            "historical_host_state": _host_state(historical_host),
            "host_state": _host_state(host),
            "services_state": _services_state(
                services,
                revision=context.revision,
            ),
            "host_receipt": host,
            "services_receipt": services,
        }
    except SchemaReconciliationRuntimeError:
        raise
    except BaseException as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_stopped_boundary_invalid"
        ) from exc


def _require_unchanged_stopped_boundary(
    context: _RuntimeContext,
    observation: Mapping[str, Any],
    *,
    code: str,
) -> None:
    if (
        observation["release_binding"] != context.initial_release_binding
        or observation["historical_host_state"] != context.initial_host_state
        or observation["host_state"] != context.initial_host_state
        or observation["services_state"] != context.initial_services_state
    ):
        _fail(code)


def _revalidate_stopped_boundary(
    context: _RuntimeContext,
    *,
    code: str,
) -> Mapping[str, Any]:
    observation = _collect_stopped_boundary(
        context,
        context.dependencies.now(),
    )
    _require_unchanged_stopped_boundary(context, observation, code=code)
    return observation


def _prepare_runtime(dependencies: _RuntimeDependencies) -> _RuntimeContext:
    dependencies.harden()
    owner = _derived_owner_identity()
    try:
        revision = dependencies.current_revision()
        if _REVISION.fullmatch(revision) is None:
            _fail("schema_reconciliation_runtime_release_invalid")
        manifest, manifest_raw = dependencies.load_manifest(revision)
        stopped, stopped_raw = dependencies.load_stopped(revision)
        release_binding = _release_binding(
            revision=revision,
            manifest=manifest,
            manifest_raw=manifest_raw,
            stopped=stopped,
            stopped_raw=stopped_raw,
            interpreter_sha256=dependencies.interpreter_sha256,
            path_exists=dependencies.path_exists,
        )
        historical_host, _historical_host_raw = dependencies.load_historical_host(
            stopped
        )
        plan = dependencies.build_plan(revision)
        target_asset = dependencies.load_target_asset(revision)
        target = target_asset.contract
        journal = dependencies.journal_factory()
        current = dependencies.now()
        fresh_host = _validate_host_receipt(
            dependencies.collect_host(observed_at_unix=current)
        )
        historical_state = _host_state(historical_host)
        host_state = _host_state(fresh_host)
        services = dependencies.collect_services(revision, current)
        services_state = _services_state(services, revision=revision)
    except SchemaReconciliationRuntimeError:
        raise
    except BaseException as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_pre_gate_invalid"
        ) from exc

    if (
        plan.revision != revision
        or target.sha256 != plan.value["target_contract_sha256"]
        or target_asset.sha256 != plan.value["target_asset_sha256"]
        or host_state != historical_state
    ):
        _fail("schema_reconciliation_runtime_release_binding_invalid")

    try:
        ca_raw = dependencies.read_ca()
    except BaseException as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_ca_invalid"
        ) from exc

    journal_head = _journal_head(plan, journal)
    admin_username = TEMPORARY_ADMIN_PREFIX + plan.sha256[:16]
    issued_at = dependencies.now()
    nonce = dependencies.random_bytes(32)
    if (
        type(issued_at) is not int
        or issued_at < current
        or not isinstance(nonce, bytes)
        or len(nonce) != 32
    ):
        _fail("schema_reconciliation_runtime_clock_invalid")
    unsigned_gate = {
        "schema": bootstrap.GATE_SCHEMA,
        "ok": True,
        "state": "stopped_release_admin_preflight_ready",
        "release_revision": revision,
        **release_binding,
        "plan_sha256": plan.sha256,
        "base_artifact_sha256": plan.value["base_artifact_sha256"],
        "target_asset_sha256": plan.value["target_asset_sha256"],
        "expected_old_contract_sha256": plan.value["expected_old_contract_sha256"],
        "target_contract_sha256": plan.value["target_contract_sha256"],
        "mutation_sql_sha256": plan.value["mutation_sql_sha256"],
        "preflight_bridge_sql_sha256": _bridge_sql_sha256(plan),
        "advisory_lock_key": plan.value["advisory_lock_key"],
        "host_identity_sha256": host_state["state_sha256"],
        "services_stopped_sha256": services_state["state_sha256"],
        "project": foundation.PROJECT,
        "sql_instance": foundation.SQL_INSTANCE,
        "database": foundation.SQL_DATABASE,
        "postgresql_major": bootstrap.EXPECTED_POSTGRESQL_MAJOR,
        "tls_server_name": foundation.SQL_TLS_SERVER_NAME,
        "ca_file_sha256": _sha256_bytes(ca_raw),
        "temporary_admin_username": admin_username,
        "temporary_admin_username_sha256": _sha256_bytes(
            admin_username.encode("ascii")
        ),
        "owner_subject_sha256": owner["owner_subject_sha256"],
        "owner_public_key_ed25519_hex": owner["owner_public_key_ed25519_hex"],
        "owner_key_id": owner["owner_key_id"],
        "owner_public_fingerprint": owner["owner_public_fingerprint"],
        "journal_head": journal_head,
        "run_nonce_sha256": _sha256_bytes(nonce),
        "issued_at_unix": issued_at,
        "expires_at_unix": issued_at + GATE_TTL_SECONDS,
        "temporary_admin_required": True,
        "secret_material_recorded": False,
    }
    gate = _hashed(unsigned_gate, "gate_sha256")
    return _RuntimeContext(
        revision=revision,
        plan=plan,
        target=target,
        journal=journal,
        stopped=copy.deepcopy(dict(stopped)),
        gate=gate,
        initial_host_state=host_state,
        initial_services_state=services_state,
        initial_release_binding=release_binding,
        dependencies=dependencies,
    )


def _open_admin_config(
    context: _RuntimeContext,
    credential: bytearray,
) -> tuple[WriterDBConfig, _Session]:
    try:
        if (
            not isinstance(credential, bytearray)
            or len(credential) != OPAQUE_CREDENTIAL_BYTES
            or _URLSAFE_CREDENTIAL.fullmatch(credential) is None
        ):
            _fail("schema_reconciliation_runtime_credential_invalid")
        with phase_b_runtime._secret_descriptor(credential) as descriptor:
            admin_config = phase_b_runtime._database_config(
                str(context.gate["temporary_admin_username"]),
                credential=CredentialSource(
                    fd=descriptor,
                    expected_uid=_ROOT_UID,
                    expected_gid=_ROOT_GID,
                    allowed_modes=frozenset({0o400}),
                ),
                application_name="muncho-schema-reconciliation-admin",
            )
            session = context.dependencies.open_session(admin_config)
        return admin_config, session
    except SchemaReconciliationRuntimeError:
        raise
    except phase_b_runtime.PhaseBRuntimeError as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_admin_secret_transport_failed"
        ) from exc
    except BaseException as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_admin_authentication_failed"
        ) from exc
    finally:
        _zeroize(credential)


def _preflight_callback(
    context: _RuntimeContext,
    gate: Mapping[str, Any],
    _admin: Mapping[str, Any],
    credential: bytearray,
) -> Mapping[str, Any]:
    if gate != context.gate or context.lease is not None:
        _fail("schema_reconciliation_runtime_preflight_state_invalid")
    writer_config = context.dependencies.writer_config()
    try:
        hba = context.dependencies.collect_hba(
            writer_config,
            now_unix=context.dependencies.now(),
            ttl_seconds=300,
        )
        _revalidate_stopped_boundary(
            context,
            code=(
                "schema_reconciliation_runtime_preflight_stopped_boundary_drifted"
            ),
        )
        admin_config, raw_session = _open_admin_config(context, credential)
        lease = _AuthenticatedSessionLease(raw_session, admin_config)
        context.lease = lease
        identity = _database_identity(raw_session)
        if (
            raw_session.username != context.gate["temporary_admin_username"]
            or identity["tls_peer_certificate_sha256"] != hba.server_certificate_sha256
        ):
            _fail("schema_reconciliation_runtime_database_identity_invalid")
        database = PostgresSchemaReconciliationDatabase(
            plan=context.plan,
            target=context.target,
            admin_config=admin_config,
            writer_config=writer_config,
            managed_hba_receipt=hba,
            _session_factory=lease.borrow,
        )
        with database.transaction(
            advisory_lock_key=context.plan.value["advisory_lock_key"]
        ) as transaction:
            transaction.lock_canonical_truth()
            observed = transaction.observe_contract()
            truth = transaction.observe_canonical_truth()
        observed_at = context.dependencies.now()
        preflight = preflight_schema_reconciliation(
            context.plan,
            target=context.target,
            observed=observed,
            truth=truth,
            observed_at_unix=observed_at,
        )
        context.database = database
        context.writer_config = writer_config
        context.preflight = preflight
        context.truth = truth
        context.database_identity = identity
        return {
            "database_identity_sha256": identity["identity_sha256"],
            "tls_peer_certificate_sha256": identity["tls_peer_certificate_sha256"],
            "managed_hba_receipt_sha256": hba.sha256,
            "postgresql_major": identity["postgresql_major"],
            "temporary_owner_bridge_receipt": (
                _temporary_owner_bridge_receipt(observed_at)
            ),
            "preflight": preflight,
            "canonical_truth_receipt": truth.value,
            "observed_at_unix": observed_at,
        }
    except BaseException as exc:
        _zeroize(credential)
        context.close_temporary_admin_database()
        if isinstance(exc, SchemaReconciliationRuntimeError):
            raise
        if isinstance(exc, SchemaReconciliationError):
            raise SchemaReconciliationRuntimeError(exc.code) from exc
        raise


def _admission_for_apply(
    context: _RuntimeContext,
    core_authorization: SchemaReconciliationAuthorization | None,
    owner_frame_receipt: Mapping[str, Any] | None,
) -> tuple[
    SchemaReconciliationAuthorization,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    if context.preflight is None or context.truth is None:
        _fail("schema_reconciliation_runtime_apply_state_invalid")
    current_head = _journal_head(context.plan, context.journal)
    if current_head != context.gate["journal_head"]:
        _fail("schema_reconciliation_runtime_journal_changed")
    if current_head["state"] == "empty":
        if not isinstance(
            core_authorization,
            SchemaReconciliationAuthorization,
        ) or not isinstance(owner_frame_receipt, Mapping):
            _fail("schema_reconciliation_runtime_authorization_invalid")
        return (
            core_authorization,
            copy.deepcopy(dict(owner_frame_receipt)),
            context.preflight,
        )

    if core_authorization is not None or owner_frame_receipt is not None:
        _fail("schema_reconciliation_runtime_authorization_invalid")

    with context.journal.lock():
        intent = context.journal.load_authorized_intent(context.plan)
    if (
        intent is None
        or intent.get("authorized_intent_sha256")
        != current_head["authorized_intent_sha256"]
    ):
        _fail("schema_reconciliation_runtime_journal_invalid")
    try:
        authorization = SchemaReconciliationAuthorization.from_mapping(
            intent["authorization"]
        )
        owner_frame = copy.deepcopy(dict(intent["owner_authorization_frame"]))
        durable_preflight = copy.deepcopy(dict(intent["preflight"]))
        durable_truth = CanonicalTruthReceipt.from_mapping(
            intent["initial_canonical_truth"]
        )
    except (KeyError, TypeError, SchemaReconciliationError) as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_journal_invalid"
        ) from exc
    if (
        durable_truth != context.truth
        or durable_preflight.get("observed_contract_sha256")
        != intent.get("initial_contract_sha256")
        or durable_preflight.get("truth_receipt_sha256") != context.truth.sha256
        or context.preflight.get("truth_receipt_sha256") != context.truth.sha256
    ):
        _fail("schema_reconciliation_runtime_journal_drifted")
    return authorization, owner_frame, durable_preflight


def _apply_callback(
    context: _RuntimeContext,
    gate: Mapping[str, Any],
    _admin: Mapping[str, Any],
    challenge: Mapping[str, Any],
    _authorization_frame: Mapping[str, Any],
    core_authorization: SchemaReconciliationAuthorization | None,
    owner_frame_receipt: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if (
        gate != context.gate
        or context.database is None
        or context.lease is None
        or context.preflight is None
        or context.truth is None
        or challenge.get("preflight") != context.preflight
        or challenge.get("canonical_truth_receipt") != context.truth.value
    ):
        _fail("schema_reconciliation_runtime_apply_state_invalid")
    try:
        authorization, owner_frame, core_preflight = _admission_for_apply(
            context,
            core_authorization,
            owner_frame_receipt,
        )
        _revalidate_stopped_boundary(
            context,
            code="schema_reconciliation_runtime_apply_stopped_boundary_drifted",
        )
        terminal = execute_schema_reconciliation(
            context.plan,
            target=context.target,
            preflight=core_preflight,
            authorization=authorization,
            owner_authorization_frame=owner_frame,
            database=context.database,
            journal=context.journal,
            now=context.dependencies.now,
        )
        with context.database.transaction(
            advisory_lock_key=context.plan.value["advisory_lock_key"]
        ) as transaction:
            transaction.lock_canonical_truth()
            final_contract = transaction.observe_contract()
            final_truth = transaction.observe_canonical_truth()
        final_identity = _database_identity(context.lease.session)
        if (
            final_contract.sha256 != context.target.sha256
            or final_truth != context.truth
            or final_identity != context.database_identity
        ):
            _fail("schema_reconciliation_runtime_database_reattestation_failed")
        observed_at = context.dependencies.now()
        context.close_temporary_admin_database()
        return {
            "authorized_intent_sha256": terminal["authorized_intent_sha256"],
            "core_terminal_receipt": terminal,
            "database_commit_attestation": _database_commit_attestation(
                database_identity=final_identity,
                contract=final_contract,
                truth=final_truth,
                observed_at_unix=observed_at,
            ),
        }
    except BaseException as exc:
        context.close_temporary_admin_database()
        if isinstance(exc, SchemaReconciliationRuntimeError):
            raise
        if isinstance(exc, SchemaReconciliationError):
            raise SchemaReconciliationRuntimeError(exc.code) from exc
        raise


def _post_cleanup_callback(
    context: _RuntimeContext,
    gate: Mapping[str, Any],
    challenge: Mapping[str, Any],
    intermediate: Mapping[str, Any],
    cleanup: Mapping[str, Any],
) -> Mapping[str, Any]:
    absence = (
        cleanup.get("cloud_sql_absence_receipt")
        if isinstance(cleanup, Mapping)
        else None
    )
    if (
        gate != context.gate
        or not context.temporary_admin_database_closed_before_cleanup
        or context.lease is None
        or not context.lease.closed
        or context.truth is None
        or context.database_identity is None
        or context.writer_config is None
        or not isinstance(challenge, Mapping)
        or not isinstance(intermediate, Mapping)
        or not isinstance(cleanup, Mapping)
        or not isinstance(absence, Mapping)
        or absence.get("temporary_admin_absent") is not True
        or cleanup.get("temporary_admin_username_sha256")
        != gate["temporary_admin_username_sha256"]
    ):
        _fail("schema_reconciliation_runtime_cleanup_order_invalid")

    try:
        before_observed_at = context.dependencies.now()
        before = _collect_stopped_boundary(context, before_observed_at)
        _require_unchanged_stopped_boundary(
            context,
            before,
            code="schema_reconciliation_runtime_post_cleanup_drifted",
        )

        pre_delete_truth = CanonicalTruthReceipt.from_mapping(
            intermediate.get("final_canonical_truth")
        )
        if (
            pre_delete_truth != context.truth
            or challenge.get("database_identity_sha256")
            != context.database_identity["identity_sha256"]
            or challenge.get("tls_peer_certificate_sha256")
            != context.database_identity["tls_peer_certificate_sha256"]
        ):
            _fail("schema_reconciliation_runtime_post_cleanup_invalid")

        writer_config = context.dependencies.writer_config()
        if writer_config != context.writer_config:
            _fail("schema_reconciliation_runtime_post_cleanup_invalid")
        fresh_hba = context.dependencies.collect_hba(
            writer_config,
            now_unix=before_observed_at,
            ttl_seconds=300,
        )
        if (
            fresh_hba.host != writer_config.host
            or fresh_hba.tls_server_name != writer_config.tls_server_name
            or fresh_hba.port != writer_config.port
            or fresh_hba.user != writer_config.user
            or fresh_hba.observed_at_unix < cleanup["issued_at_unix"]
            or not fresh_hba.is_fresh(before_observed_at)
            or fresh_hba.server_certificate_sha256
            != challenge["tls_peer_certificate_sha256"]
        ):
            _fail("schema_reconciliation_runtime_post_cleanup_invalid")
        collected = context.dependencies.collect_post_delete_terminal(
            plan=context.plan,
            target=context.target,
            temporary_login=gate["temporary_admin_username"],
            writer_config=writer_config,
            managed_hba_receipt=fresh_hba,
            pre_delete_canonical_truth=pre_delete_truth,
            observed_at_unix=before_observed_at,
            _session_factory=context.dependencies.open_session,
        )
        collected_value = (
            collected.value
            if isinstance(collected, PostDeleteTerminalReceipt)
            else collected
        )
        post_delete_terminal = validate_post_delete_terminal_receipt(
            collected_value,
            plan=context.plan,
            target=context.target,
            temporary_login=gate["temporary_admin_username"],
            managed_hba_receipt=fresh_hba,
            pre_delete_canonical_truth=pre_delete_truth,
        )
        if (
            post_delete_terminal.tls_peer_certificate_sha256
            != challenge["tls_peer_certificate_sha256"]
        ):
            _fail("schema_reconciliation_runtime_post_cleanup_invalid")

        observed_at = context.dependencies.now()
        if not fresh_hba.is_fresh(observed_at):
            _fail("schema_reconciliation_runtime_post_cleanup_invalid")
        after = _collect_stopped_boundary(context, observed_at)
        _require_unchanged_stopped_boundary(
            context,
            after,
            code="schema_reconciliation_runtime_post_cleanup_drifted",
        )
        if before != after:
            # Receipt timestamps may change, but the release and stopped-state
            # identities must remain byte-stable around the writer proof.
            if (
                before["release_binding"] != after["release_binding"]
                or before["historical_host_state"]
                != after["historical_host_state"]
                or before["host_state"] != after["host_state"]
                or before["services_state"] != after["services_state"]
            ):
                _fail("schema_reconciliation_runtime_post_cleanup_drifted")
    except SchemaReconciliationRuntimeError:
        raise
    except BaseException as exc:
        raise SchemaReconciliationRuntimeError(
            "schema_reconciliation_runtime_post_cleanup_invalid"
        ) from exc
    post_delete_value = post_delete_terminal.value
    fresh_hba_value = fresh_hba.as_dict()
    unsigned = {
        "schema": bootstrap.POST_CLEANUP_OBSERVATION_SCHEMA,
        **after["release_binding"],
        "host_identity_sha256": after["host_state"]["state_sha256"],
        "services_stopped_sha256": after["services_state"]["state_sha256"],
        "host_observation_receipt_sha256": after["host_receipt"][
            "receipt_sha256"
        ],
        "services_observation_receipt_sha256": after["services_receipt"][
            "attestation_sha256"
        ],
        "fresh_managed_hba_receipt": fresh_hba_value,
        "fresh_managed_hba_receipt_sha256": fresh_hba.sha256,
        "post_delete_terminal_receipt": post_delete_value,
        "post_delete_terminal_receipt_sha256": post_delete_value[
            "receipt_sha256"
        ],
        "observed_at_unix": observed_at,
        "secret_material_recorded": False,
    }
    return _hashed(unsigned, "observation_sha256")


def run(
    *,
    input_stream: BinaryIO | None = None,
    output_stream: BinaryIO | None = None,
    _dependencies: _RuntimeDependencies | None = None,
) -> Mapping[str, Any]:
    """Run the fixed G0/A1/P1/A2/I2/C3/T3 dialogue."""

    dependencies = _dependencies or _RuntimeDependencies()
    context = _prepare_runtime(dependencies)
    try:
        return dependencies.protocol_runner(
            context.gate,
            owner_public_key_ed25519_hex=OWNER_PUBLIC_KEY_ED25519_HEX,
            owner_public_fingerprint=OWNER_PUBLIC_FINGERPRINT,
            preflight_callback=lambda gate, admin, credential: _preflight_callback(
                context,
                gate,
                admin,
                credential,
            ),
            apply_callback=lambda gate, admin, challenge, authorization, core_authorization, owner_frame_receipt: (
                _apply_callback(
                    context,
                    gate,
                    admin,
                    challenge,
                    authorization,
                    core_authorization,
                    owner_frame_receipt,
                )
            ),
            post_cleanup_callback=lambda gate, challenge, intermediate, cleanup: (
                _post_cleanup_callback(
                    context,
                    gate,
                    challenge,
                    intermediate,
                    cleanup,
                )
            ),
            input_stream=input_stream,
            output_stream=output_stream,
            now=dependencies.now,
        )
    finally:
        context.close_temporary_admin_database()


def main(argv: Sequence[str] | None = None) -> int:
    """Secret-free CLI used only by the packaged bootstrap delegation."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments:
            _fail("schema_reconciliation_runtime_arguments_forbidden")
        run()
    except BaseException:
        print("schema reconciliation runtime failed closed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DATABASE_IDENTITY_SCHEMA",
    "HOST_STATE_SCHEMA",
    "OWNER_KEY_ID",
    "OWNER_PUBLIC_FINGERPRINT",
    "OWNER_PUBLIC_KEY_ED25519_HEX",
    "OWNER_SUBJECT_SHA256",
    "RUNTIME_SCHEMA",
    "SERVICES_STATE_SCHEMA",
    "SchemaReconciliationRuntimeError",
    "main",
    "run",
]
