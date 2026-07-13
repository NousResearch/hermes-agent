"""Isolated-canary persistent Canonical Writer database foundation.

This module is a narrow privileged mechanical boundary.  It has no CLI and no
caller-selectable host, database, role, SQL artifact, credential path, or
deployment scope.  The model/owner approves an exact digest-bound plan; this
module only observes, validates, applies sealed SQL, journals receipts, and
recovers the same plan after a process loss.

The owner transport supplies an ``OpaqueStdinAdminFrame`` to
``VerifiedTLSBootstrapAdminSession`` outside this module and injects the
resulting verified session.  Keeping that boundary injected makes Phase A
fully testable without introducing another credential transport.

This reviewed Phase A subset implements exact-terminal adoption only.  The
sealed creation/password artifacts and CopyData primitive are unreachable
Phase-B material until an external administrator-deletion and post-delete
observation workflow exists.  The retirement SQL is likewise sealed for the
next gate, but this module deliberately exposes no retirement plan/apply API.
An adoption receipt explicitly does not attest retirement coverage.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import dataclasses
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from gateway.canonical_writer_config_collector import (
    _attestation_projection,
    _collect_live_policy,
)
from gateway.canonical_writer_db import (
    CanonicalWriterDB,
    CredentialSource,
    ManagedCloudSQLAdminHBAReceipt,
    PostgresServerError,
    WriterDBConfig,
    _open_postgres_session,
    collect_managed_cloudsqladmin_hba_receipt,
)
from gateway.canonical_writer_handlers import RuntimeContext
from gateway.canonical_writer_planner import load_release_manifest
from gateway.canonical_writer_postgres_backend import (
    PRODUCTION_CATALOG_SHA256,
    PRODUCTION_STATEMENT_CATALOG,
    PostgresCanonicalWriterBackend,
)


FOUNDATION_OBSERVATION_SCHEMA = "muncho-canonical-writer-foundation-observation.v1"
FOUNDATION_PLAN_SCHEMA = "muncho-canonical-writer-foundation-plan.v1"
FOUNDATION_JOURNAL_SCHEMA = "muncho-canonical-writer-foundation-journal.v1"
FOUNDATION_RECEIPT_SCHEMA = "muncho-canonical-writer-foundation-receipt.v1"
_FOUNDATION_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "ok",
        "mode",
        "release_revision",
        "plan_sha256",
        "artifact_set_sha256",
        "terminal_entry_sha256",
        "database_observation_sha256",
        "credential",
        "catalog_sha256",
        "privilege_attestation_sha256",
        "private_schema_identity_sha256",
        "managed_hba_receipt_sha256",
        "writer_ping_ok",
        "secret_material_recorded",
        "retirement_covered",
        "terminal_at_unix",
        "verified_at_unix",
        "hba_observed_at_unix",
        "hba_expires_at_unix",
        "receipt_sha256",
    }
)
PROJECT = "adventico-ai-platform"
ZONE = "europe-west3-a"
VM_NAME = "muncho-canary-v2-01"
VM_INSTANCE_ID = "9153645328899914617"
SQL_INSTANCE = "muncho-canary-pg18-v2"
SQL_HOST = "10.91.0.3"
SQL_TLS_SERVER_NAME = (
    "14-0d81ef63-2cac-4a64-84ad-c4f58c0cfd56.europe-west3.sql.goog"
)
SQL_PORT = 5432
SQL_DATABASE = "muncho_canary_brain"
SQL_USER = "muncho_canary_writer_login"
DATABASE_OWNER_ROLE = "cloudsqlsuperuser"
PERSISTENT_MEMBERSHIP_GRANTOR = "cloudsqladmin"
MIGRATION_OWNER_ROLE = "canonical_brain_migration_owner"
WRITER_ROLE = "canonical_brain_writer"
CANARY_BOOTSTRAP_ROLE = "canonical_brain_canary_bootstrap"
CANARY_BOOTSTRAP_LOGIN = "canonical_brain_canary_bootstrap_login"
WRITER_UID = 999
WRITER_GID = 994
DATABASE_CA_PATH = Path("/etc/muncho/trust/cloudsql-server-ca.pem")
DATABASE_CREDENTIAL_PATH = Path(
    "/etc/muncho/credentials/canonical-writer-db-password"
)
FOUNDATION_EVIDENCE_ROOT = Path(
    "/var/lib/muncho-writer-canary-evidence/persistent-foundation"
)
FOUNDATION_LOCK_PATH = FOUNDATION_EVIDENCE_ROOT / ".foundation.lock"

FOUNDATION_STATES = (
    "intent",
    "prerequisites",
    "truth_reconciled",
    "secret_staged",
    "login_enabled",
    "base_migration",
    "writer_membership",
    "attested",
    "terminal",
)
FOUNDATION_ADOPTION_STATES = (
    "adopted_existing_terminal",
    "terminal",
)
FOUNDATION_PLAN_MODES = frozenset(
    {"create_pristine", "adopted_existing_terminal"}
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_ADMIN_RE = re.compile(r"^muncho_canary_admin_[0-9a-f]{16}$")
_LEGACY_SOURCE_OWNER_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,63}$")
_WRITER_SECRET_RE = re.compile(rb"^[A-Za-z0-9_-]{64}$")
_MAX_SQL_BYTES = 16 * 1024 * 1024
_MAX_PUBLIC_JSON_BYTES = 4 * 1024 * 1024

_ARTIFACT_FILENAMES: Mapping[str, str] = {
    "observe": "canonical_writer_foundation_observe_v1.sql",
    "legacy_observe": "canonical_writer_foundation_legacy_observe_v1.sql",
    "prerequisites": "canonical_writer_foundation_prerequisites_v1.sql",
    "legacy_reconcile": "canonical_writer_foundation_legacy_reconcile_v1.sql",
    "base_migration": "canonical_writer_v1.sql",
    "login_enable": "canonical_writer_foundation_login_v1.sql",
    "writer_membership": "canonical_writer_foundation_membership_v1.sql",
    "retirement": "canonical_writer_foundation_retire_v1.sql",
}

_OBSERVATION_FIELDS = frozenset(
    {
        "schema",
        "database",
        "database_owner",
        "postgres_version_num",
        "session_user",
        "current_user",
        "temporary_admin_roles",
        "tls_peer_certificate_sha256",
        "roles",
        "memberships",
        "event_log_shape",
        "event_log_owner",
        "legacy_archive_present",
        "legacy_archive_identity",
        "canonical_schema_owner",
        "writer_ping_present",
        "database_acl",
        "public_schema_acl",
        "legacy_truth",
        "credential",
        "observation_sha256",
    }
)
_ROLE_FIELDS = frozenset(
    {
        "name",
        "can_login",
        "inherits",
        "superuser",
        "create_database",
        "create_role",
        "replication",
        "bypass_row_security",
        "connection_limit",
        "validity_is_unbounded",
        "configuration_is_empty",
    }
)
_MEMBERSHIP_FIELDS = frozenset(
    {
        "granted_role",
        "member_role",
        "grantor",
        "admin_option",
        "inherit_option",
        "set_option",
    }
)
_LEGACY_ARCHIVE_IDENTITY_FIELDS = frozenset(
    {
        "oid",
        "owner",
        "relation_kind",
        "persistence",
        "owner_superuser",
        "owner_create_database",
        "owner_create_role",
        "owner_replication",
        "owner_bypass_row_security",
        "owner_connection_limit",
        "owner_validity_is_unbounded",
        "owner_configuration_is_empty",
    }
)
_LEGACY_ARCHIVE_RESERVED_OWNERS = frozenset(
    {
        "postgres",
        "cloudsqladmin",
        "cloudsqlsuperuser",
        MIGRATION_OWNER_ROLE,
        WRITER_ROLE,
        CANARY_BOOTSTRAP_ROLE,
        CANARY_BOOTSTRAP_LOGIN,
        SQL_USER,
    }
)
_CREDENTIAL_FIELDS = frozenset(
    {
        "state",
        "path",
        "stage_path",
        "device",
        "inode",
        "owner_uid",
        "group_gid",
        "mode",
        "link_count",
        "modification_time_ns",
        "change_time_ns",
        "content_or_digest_recorded",
    }
)


class CanonicalWriterFoundationError(RuntimeError):
    """Fail-closed public error whose message contains no database detail."""

    def __init__(self, code: str) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,95}", code):
            code = "canonical_writer_foundation_failed"
        self.code = code
        super().__init__(code)


class AdminSession(Protocol):
    username: str
    tls_peer_certificate_sha256: str

    def query(self, sql: str, *, maximum_rows: int) -> Any: ...


@dataclass(frozen=True)
class WriterPasswordCopyBoundary:
    """Fixed, secret-free contract for one PostgreSQL CopyData mutation.

    PostgreSQL 18 does not let a managed/non-superuser administrator lower
    ``log_parameter_max_length``.  Bind parameters and ``psql \\password`` are
    therefore unsafe with statement logging enabled: the former can log the
    plaintext bind and the latter can log the generated verifier.  This
    boundary sends the password only as CopyData to a private one-row temporary
    relation.  SQL, argv, errors, results, logs, and evidence remain secret-free.
    """

    name: str = "canonical_writer_login_password_copy_v1"
    role: str = SQL_USER
    database: str = SQL_DATABASE
    password_encryption: str = "scram-sha-256"
    secret_transport: str = "postgres-copy-data-v1"
    temporary_security_definer: bool = True

    @property
    def setup_sql(self) -> str:
        return _WRITER_PASSWORD_SETUP_SQL

    @property
    def copy_sql(self) -> str:
        return _WRITER_PASSWORD_COPY_SQL

    @property
    def apply_sql(self) -> str:
        return _WRITER_PASSWORD_APPLY_SQL

    @property
    def cleanup_sql(self) -> str:
        return _WRITER_PASSWORD_CLEANUP_SQL

    @property
    def rollback_sql(self) -> str:
        return "ROLLBACK;"


_WRITER_PASSWORD_SETUP_SQL = r"""
BEGIN;
SET LOCAL password_encryption = 'scram-sha-256';
SET LOCAL lock_timeout = '15s';
SET LOCAL statement_timeout = '2min';
SELECT pg_catalog.pg_advisory_xact_lock(4841739663211427921);

CREATE TEMPORARY TABLE pg_temp._muncho_canonical_writer_password_input (
    secret_value text NOT NULL
) ON COMMIT DROP;
REVOKE ALL ON TABLE pg_temp._muncho_canonical_writer_password_input FROM PUBLIC;

CREATE FUNCTION pg_temp._muncho_install_canonical_writer_password()
RETURNS void
LANGUAGE plpgsql
VOLATILE
PARALLEL UNSAFE
SECURITY DEFINER
SET search_path = pg_catalog
AS $password_statement$
DECLARE
    secret_value text;
    secret_count bigint;
    input_relation oid := pg_catalog.to_regclass(
        'pg_temp._muncho_canonical_writer_password_input'
    );
BEGIN
    SELECT pg_catalog.count(*), min(input.secret_value)
      INTO secret_count, secret_value
      FROM pg_temp._muncho_canonical_writer_password_input AS input;
    IF pg_catalog.current_database() <> 'muncho_canary_brain'
       OR pg_catalog.current_setting('server_version_num')::integer / 10000 <> 18
       OR SESSION_USER !~ '^muncho_canary_admin_[0-9a-f]{16}$'
       OR CURRENT_USER <> SESSION_USER
       OR input_relation IS NULL
       OR secret_count <> 1
       OR secret_value !~ '^[A-Za-z0-9_-]{64}$'
       OR NOT EXISTS (
            SELECT 1
              FROM pg_catalog.pg_class AS relation
              JOIN pg_catalog.pg_namespace AS namespace
                ON namespace.oid = relation.relnamespace
             WHERE relation.oid = input_relation
               AND relation.relkind = 'r'
               AND relation.relpersistence = 't'
               AND relation.relowner = (
                    SELECT oid FROM pg_catalog.pg_roles
                     WHERE rolname = SESSION_USER
               )
               AND namespace.oid = pg_catalog.pg_my_temp_schema()
               AND NOT EXISTS (
                    SELECT 1
                      FROM pg_catalog.aclexplode(COALESCE(
                          relation.relacl,
                          pg_catalog.acldefault('r', relation.relowner)
                      )) AS acl
                     WHERE acl.grantee <> relation.relowner
               )
       )
       OR (
            SELECT pg_catalog.count(*)
              FROM pg_catalog.pg_attribute AS attribute
             WHERE attribute.attrelid = input_relation
               AND attribute.attnum > 0 AND NOT attribute.attisdropped
       ) <> 1
       OR (
            SELECT pg_catalog.count(*)
              FROM pg_catalog.pg_attribute AS attribute
              LEFT JOIN pg_catalog.pg_attrdef AS default_row
                ON default_row.adrelid = attribute.attrelid
               AND default_row.adnum = attribute.attnum
             WHERE attribute.attrelid = input_relation
               AND attribute.attnum > 0 AND NOT attribute.attisdropped
               AND attribute.attname = 'secret_value'
               AND attribute.atttypid = 'pg_catalog.text'::regtype
               AND attribute.attnotnull
               AND default_row.oid IS NULL
               AND attribute.attidentity = ''
               AND attribute.attgenerated = ''
               AND NOT attribute.atthasmissing
       ) <> 1
       OR EXISTS (
            SELECT 1 FROM pg_catalog.pg_constraint
             WHERE conrelid = input_relation
               AND contype <> 'n'
       )
       OR EXISTS (
            SELECT 1 FROM pg_catalog.pg_trigger
             WHERE tgrelid = input_relation AND NOT tgisinternal
       )
       OR EXISTS (
            SELECT 1 FROM pg_catalog.pg_rewrite
             WHERE ev_class = input_relation
       )
       OR EXISTS (
            SELECT 1 FROM pg_catalog.pg_policy
             WHERE polrelid = input_relation
       )
       OR NOT EXISTS (
            SELECT 1 FROM pg_catalog.pg_roles AS role
             WHERE role.rolname = 'muncho_canary_writer_login'
               AND NOT role.rolcanlogin AND role.rolinherit
               AND NOT role.rolsuper AND NOT role.rolcreatedb
               AND NOT role.rolcreaterole AND NOT role.rolreplication
               AND NOT role.rolbypassrls AND role.rolconnlimit = -1
               AND role.rolvaliduntil IS NULL AND role.rolconfig IS NULL
       ) OR (
            SELECT pg_catalog.count(*)
              FROM pg_catalog.pg_auth_members AS membership
              JOIN pg_catalog.pg_roles AS granted
                ON granted.oid = membership.roleid
              JOIN pg_catalog.pg_roles AS member
                ON member.oid = membership.member
              JOIN pg_catalog.pg_roles AS grantor
                ON grantor.oid = membership.grantor
             WHERE granted.rolname = 'muncho_canary_writer_login'
               AND member.rolname = SESSION_USER
               AND membership.admin_option
               AND NOT membership.inherit_option
               AND NOT membership.set_option
               AND grantor.rolname IN (
                    'postgres', 'cloudsqladmin', 'cloudsqlsuperuser'
               )
       ) <> 1 OR EXISTS (
            SELECT 1
              FROM pg_catalog.pg_auth_members AS membership
              JOIN pg_catalog.pg_roles AS granted
                ON granted.oid = membership.roleid
              JOIN pg_catalog.pg_roles AS member
                ON member.oid = membership.member
              JOIN pg_catalog.pg_roles AS grantor
                ON grantor.oid = membership.grantor
             WHERE (
                    granted.rolname = 'muncho_canary_writer_login'
                    OR member.rolname = 'muncho_canary_writer_login'
               )
               AND NOT (
                    granted.rolname = 'muncho_canary_writer_login'
                    AND member.rolname = SESSION_USER
                    AND membership.admin_option
                    AND NOT membership.inherit_option
                    AND NOT membership.set_option
                    AND grantor.rolname IN (
                        'postgres', 'cloudsqladmin', 'cloudsqlsuperuser'
                    )
               )
       ) THEN
        TRUNCATE TABLE pg_temp._muncho_canonical_writer_password_input;
        secret_value := NULL;
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'canonical writer password statement prerequisite failed';
    END IF;
    BEGIN
        EXECUTE pg_catalog.format(
            'ALTER ROLE muncho_canary_writer_login LOGIN PASSWORD %L',
            secret_value
        );
    EXCEPTION WHEN OTHERS THEN
        TRUNCATE TABLE pg_temp._muncho_canonical_writer_password_input;
        secret_value := NULL;
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'canonical writer password statement failed';
    END;
    TRUNCATE TABLE pg_temp._muncho_canonical_writer_password_input;
    secret_value := NULL;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
         WHERE rolname = 'muncho_canary_writer_login' AND rolcanlogin
    ) THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'canonical writer password statement unconfirmed';
    END IF;
END
$password_statement$;

REVOKE ALL ON FUNCTION
    pg_temp._muncho_install_canonical_writer_password()
    FROM PUBLIC;

DO $password_statement_identity$
DECLARE
    function_oid oid := pg_catalog.to_regprocedure(
        'pg_temp._muncho_install_canonical_writer_password()'
    );
    function_row record;
BEGIN
    SELECT routine.prosecdef,
           routine.proleakproof,
           routine.provolatile,
           routine.proparallel,
           routine.proconfig,
           routine.pronargs,
           routine.proargtypes,
           language.lanname,
           pg_catalog.pg_get_userbyid(routine.proowner) AS owner_name
      INTO STRICT function_row
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_language AS language
        ON language.oid = routine.prolang
     WHERE routine.oid = function_oid;
    IF NOT function_row.prosecdef
       OR function_row.proleakproof
       OR function_row.provolatile <> 'v'
       OR function_row.proparallel <> 'u'
       OR function_row.proconfig <> ARRAY['search_path=pg_catalog']::text[]
       OR function_row.pronargs <> 0
       OR function_row.proargtypes <> ''::pg_catalog.oidvector
       OR function_row.lanname <> 'plpgsql'
       OR function_row.owner_name <> SESSION_USER
       OR EXISTS (
            SELECT 1
              FROM pg_catalog.pg_proc AS routine
              CROSS JOIN LATERAL pg_catalog.aclexplode(routine.proacl) AS acl
             WHERE routine.oid = function_oid
               AND acl.grantee = 0
               AND acl.privilege_type = 'EXECUTE'
       ) THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'canonical writer password statement identity failed';
    END IF;
END
$password_statement_identity$;
""".strip()

_WRITER_PASSWORD_COPY_SQL = (
    "COPY pg_temp._muncho_canonical_writer_password_input (secret_value) "
    "FROM STDIN WITH (FORMAT text)"
)

_WRITER_PASSWORD_APPLY_SQL = (
    "SELECT pg_temp._muncho_install_canonical_writer_password() "
    "AS password_statement_applied"
)

_WRITER_PASSWORD_CLEANUP_SQL = """
DROP FUNCTION pg_temp._muncho_install_canonical_writer_password();
DROP TABLE pg_temp._muncho_canonical_writer_password_input;
COMMIT;
""".strip()


WRITER_PASSWORD_BOUNDARY = WriterPasswordCopyBoundary()


class PasswordCopySession(Protocol):
    def execute_password_copy(
        self,
        boundary: WriterPasswordCopyBoundary,
        *,
        password: bytearray,
    ) -> Mapping[str, Any]: ...


class SecretStore(Protocol):
    def observe(self, plan_sha256: str | None) -> Mapping[str, Any]: ...

    def allocate(self, plan_sha256: str) -> Mapping[str, Any]: ...

    def materialize(
        self,
        plan_sha256: str,
        expected: Mapping[str, Any],
        factory: Callable[[], bytes],
    ) -> tuple[bytearray, Mapping[str, Any]]: ...

    def publish(
        self,
        plan_sha256: str,
        expected: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def read(self, expected: Mapping[str, Any]) -> bytearray: ...

    def remove(self, expected: Mapping[str, Any]) -> None: ...


def _effective_uid() -> int:
    """Return the POSIX effective uid without a bare Windows-only lookup."""

    getter = getattr(os, "geteuid", None)
    if not callable(getter):
        raise CanonicalWriterFoundationError("foundation_posix_identity_unavailable")
    return int(getter())


def _effective_gid() -> int:
    """Return the POSIX effective gid without a bare Windows-only lookup."""

    getter = getattr(os, "getegid", None)
    if not callable(getter):
        raise CanonicalWriterFoundationError("foundation_posix_identity_unavailable")
    return int(getter())


def _canonical_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalWriterFoundationError("noncanonical_foundation_value") from exc
    return rendered.encode("utf-8", errors="strict")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CanonicalWriterFoundationError(code)
    return value


def _revision(value: Any) -> str:
    if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
        raise CanonicalWriterFoundationError("foundation_revision_invalid")
    return value


def _strict_json(raw: str, code: str) -> Mapping[str, Any]:
    def reject_duplicates(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("non-JSON constant")
            ),
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise CanonicalWriterFoundationError(code) from exc
    if not isinstance(value, Mapping):
        raise CanonicalWriterFoundationError(code)
    return value


def _zeroize(value: bytearray | None) -> None:
    if value is None:
        return
    try:
        value[:] = b"\x00" * len(value)
    except (BufferError, TypeError, ValueError):
        pass


@dataclass(frozen=True)
class SealedSQLArtifact:
    name: str
    path: Path
    sha256: str
    payload: bytes = dataclasses.field(repr=False, compare=False)


def _read_sealed_artifact(
    name: str,
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    require_root_sealed: bool,
) -> SealedSQLArtifact:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise CanonicalWriterFoundationError("foundation_sql_artifact_untrusted")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            raw = bytearray()
            while len(raw) <= _MAX_SQL_BYTES:
                chunk = os.read(descriptor, min(1024 * 1024, _MAX_SQL_BYTES + 1))
                if not chunk:
                    break
                raw.extend(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        reachable = path.lstat()
    except OSError as exc:
        raise CanonicalWriterFoundationError("foundation_sql_artifact_unavailable") from exc
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    payload = bytes(raw)
    if (
        not payload
        or len(payload) > _MAX_SQL_BYTES
        or len(payload) != expected_size
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
        or identity(before) != identity(reachable)
        or b"\x00" in payload
        or _sha256_bytes(payload) != expected_sha256
        or (require_root_sealed and (
            reachable.st_uid != 0
            or reachable.st_gid != 0
            or stat.S_IMODE(reachable.st_mode) != 0o444
            or reachable.st_nlink != 1
        ))
    ):
        raise CanonicalWriterFoundationError("foundation_sql_artifact_drifted")
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CanonicalWriterFoundationError("foundation_sql_artifact_encoding_invalid") from exc
    return SealedSQLArtifact(name, path, expected_sha256, payload)


def _load_sealed_artifacts(revision: str) -> Mapping[str, SealedSQLArtifact]:
    """Load every SQL byte only from the exact root-sealed release manifest."""

    release_revision = _revision(revision)
    try:
        manifest, _raw = load_release_manifest(release_revision)
    except BaseException as exc:
        raise CanonicalWriterFoundationError("foundation_release_manifest_invalid") from exc
    root = Path(manifest.artifact_root)
    if root != Path("/opt/muncho-canary-releases") / release_revision:
        raise CanonicalWriterFoundationError("foundation_release_root_drifted")
    entries = {entry.path: entry for entry in manifest.entries}
    artifacts: dict[str, SealedSQLArtifact] = {}
    for name, filename in _ARTIFACT_FILENAMES.items():
        relative = f"scripts/sql/{filename}"
        entry = entries.get(relative)
        if (
            entry is None
            or entry.kind != "file"
            or entry.mode != "0444"
            or entry.size <= 0
            or entry.size > _MAX_SQL_BYTES
            or _SHA256_RE.fullmatch(entry.sha256) is None
        ):
            raise CanonicalWriterFoundationError(
                "foundation_release_sql_entry_missing"
            )
        artifacts[name] = _read_sealed_artifact(
            name,
            root / relative,
            expected_sha256=entry.sha256,
            expected_size=entry.size,
            require_root_sealed=True,
        )
    return artifacts


def _load_source_artifacts_for_tests() -> Mapping[str, SealedSQLArtifact]:
    """Private test fixture; production code never falls back to source files."""

    root = Path(__file__).resolve().parents[1] / "scripts" / "sql"
    artifacts: dict[str, SealedSQLArtifact] = {}
    for name, filename in _ARTIFACT_FILENAMES.items():
        path = root / filename
        payload = path.read_bytes()
        artifacts[name] = _read_sealed_artifact(
            name,
            path,
            expected_sha256=_sha256_bytes(payload),
            expected_size=len(payload),
            require_root_sealed=False,
        )
    return artifacts


def _artifact_digest_mapping(
    artifacts: Mapping[str, SealedSQLArtifact],
) -> dict[str, str]:
    if set(artifacts) != set(_ARTIFACT_FILENAMES):
        raise CanonicalWriterFoundationError("foundation_sql_artifact_set_invalid")
    result: dict[str, str] = {}
    for name in sorted(artifacts):
        artifact = artifacts[name]
        if (
            not isinstance(artifact, SealedSQLArtifact)
            or artifact.name != name
            or artifact.path.name != _ARTIFACT_FILENAMES[name]
            or not isinstance(artifact.payload, bytes)
            or not artifact.payload
            or len(artifact.payload) > _MAX_SQL_BYTES
        ):
            raise CanonicalWriterFoundationError(
                "foundation_sql_artifact_identity_invalid"
            )
        digest = _sha256_bytes(artifact.payload)
        if digest != artifact.sha256:
            raise CanonicalWriterFoundationError(
                "foundation_sql_artifact_payload_drifted"
            )
        reread = _read_sealed_artifact(
            name,
            artifact.path,
            expected_sha256=digest,
            expected_size=len(artifact.payload),
            require_root_sealed=False,
        )
        if reread.payload != artifact.payload:
            raise CanonicalWriterFoundationError(
                "foundation_sql_artifact_payload_drifted"
            )
        result[name] = digest
    return result


def _query_json(
    session: AdminSession,
    artifact: SealedSQLArtifact,
    *,
    expected_column: str,
) -> Mapping[str, Any]:
    try:
        sql = artifact.payload.decode("utf-8", errors="strict")
        result = session.query(sql, maximum_rows=1)
    except CanonicalWriterFoundationError:
        raise
    except BaseException as exc:
        raise CanonicalWriterFoundationError("foundation_observation_query_failed") from exc
    columns = tuple(getattr(result, "columns", ()))
    rows = tuple(getattr(result, "rows", ()))
    if columns != (expected_column,) or len(rows) != 1 or len(rows[0]) != 1:
        raise CanonicalWriterFoundationError("foundation_observation_shape_invalid")
    value = rows[0][0]
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_PUBLIC_JSON_BYTES:
        raise CanonicalWriterFoundationError("foundation_observation_value_invalid")
    return _strict_json(value, "foundation_observation_json_invalid")


_EXPECTED_ROLE_SHAPES: Mapping[str, tuple[bool, bool]] = {
    MIGRATION_OWNER_ROLE: (False, False),
    WRITER_ROLE: (False, True),
    CANARY_BOOTSTRAP_ROLE: (False, False),
    CANARY_BOOTSTRAP_LOGIN: (True, True),
    SQL_USER: (False, True),
}


def _validate_role_rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise CanonicalWriterFoundationError("foundation_role_observation_invalid")
    rows: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _ROLE_FIELDS:
            raise CanonicalWriterFoundationError("foundation_role_observation_invalid")
        row = dict(item)
        name = row["name"]
        if name not in _EXPECTED_ROLE_SHAPES or name in seen:
            raise CanonicalWriterFoundationError("foundation_role_observation_invalid")
        seen.add(name)
        for field in _ROLE_FIELDS - {"name", "connection_limit"}:
            if type(row[field]) is not bool:
                raise CanonicalWriterFoundationError("foundation_role_observation_invalid")
        if row["connection_limit"] != -1:
            raise CanonicalWriterFoundationError("foundation_role_authority_drifted")
        expected_login, expected_inherit = _EXPECTED_ROLE_SHAPES[name]
        if name == SQL_USER:
            expected_login = bool(row["can_login"])
        if (
            row["can_login"] is not expected_login
            or row["inherits"] is not expected_inherit
            or row["superuser"]
            or row["create_database"]
            or row["create_role"]
            or row["replication"]
            or row["bypass_row_security"]
            or not row["validity_is_unbounded"]
            or not row["configuration_is_empty"]
        ):
            raise CanonicalWriterFoundationError("foundation_role_authority_drifted")
        rows.append(row)
    return tuple(sorted(rows, key=lambda row: str(row["name"])))


def _validate_membership_rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise CanonicalWriterFoundationError("foundation_membership_observation_invalid")
    rows: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    allowed = {
        (CANARY_BOOTSTRAP_ROLE, CANARY_BOOTSTRAP_LOGIN),
        (WRITER_ROLE, SQL_USER),
    }
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _MEMBERSHIP_FIELDS:
            raise CanonicalWriterFoundationError(
                "foundation_membership_observation_invalid"
            )
        row = dict(item)
        pair = (row["granted_role"], row["member_role"])
        if pair not in allowed or pair in seen:
            raise CanonicalWriterFoundationError("foundation_membership_authority_drifted")
        seen.add(pair)
        exact_options = (
            row["grantor"] == PERSISTENT_MEMBERSHIP_GRANTOR
            and row["admin_option"] is False
            and row["inherit_option"] is True
            and row["set_option"] is False
        )
        if not exact_options:
            raise CanonicalWriterFoundationError("foundation_membership_authority_drifted")
        rows.append(row)
    return tuple(sorted(rows, key=lambda row: (row["granted_role"], row["member_role"])))


def _credential_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CREDENTIAL_FIELDS:
        raise CanonicalWriterFoundationError("foundation_credential_metadata_invalid")
    raw = dict(value)
    if raw["path"] != str(DATABASE_CREDENTIAL_PATH):
        raise CanonicalWriterFoundationError("foundation_credential_path_drifted")
    if raw["state"] == "absent":
        if raw["stage_path"] is not None or any(
            raw[name] is not None
            for name in (
                "device",
                "inode",
                "owner_uid",
                "group_gid",
                "mode",
                "link_count",
                "modification_time_ns",
                "change_time_ns",
            )
        ):
            raise CanonicalWriterFoundationError("foundation_credential_metadata_invalid")
    elif raw["state"] in {"allocated", "staged"}:
        if (
            not isinstance(raw["stage_path"], str)
            or type(raw["device"]) is not int
            or type(raw["inode"]) is not int
            or raw["device"] <= 0
            or raw["inode"] <= 0
            or raw["owner_uid"] != WRITER_UID
            or raw["group_gid"] != WRITER_GID
            or raw["mode"] != "0400"
            or raw["link_count"] != 1
            or type(raw["modification_time_ns"]) is not int
            or type(raw["change_time_ns"]) is not int
        ):
            raise CanonicalWriterFoundationError("foundation_credential_metadata_invalid")
    elif raw["state"] == "installed":
        if (
            raw["stage_path"] is not None
            or type(raw["device"]) is not int
            or type(raw["inode"]) is not int
            or raw["device"] <= 0
            or raw["inode"] <= 0
            or raw["owner_uid"] != WRITER_UID
            or raw["group_gid"] != WRITER_GID
            or raw["mode"] != "0400"
            or raw["link_count"] != 1
            or type(raw["modification_time_ns"]) is not int
            or type(raw["change_time_ns"]) is not int
        ):
            raise CanonicalWriterFoundationError("foundation_credential_metadata_invalid")
    else:
        raise CanonicalWriterFoundationError("foundation_credential_metadata_invalid")
    if raw["content_or_digest_recorded"] is not False:
        raise CanonicalWriterFoundationError("foundation_secret_material_recorded")
    return raw


def _is_foundation_observer_username(value: Any) -> bool:
    return isinstance(value, str) and (
        value == SQL_USER or _ADMIN_RE.fullmatch(value) is not None
    )


@dataclass(frozen=True)
class FoundationObservation:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "FoundationObservation":
        if not isinstance(value, Mapping) or set(value) != _OBSERVATION_FIELDS:
            raise CanonicalWriterFoundationError("foundation_observation_fields_invalid")
        raw = copy.deepcopy(dict(value))
        unsigned = {key: item for key, item in raw.items() if key != "observation_sha256"}
        if (
            raw["schema"] != FOUNDATION_OBSERVATION_SCHEMA
            or raw["database"] != SQL_DATABASE
            or raw["postgres_version_num"] // 10000 != 18
            or not _is_foundation_observer_username(raw["session_user"])
            or raw["current_user"] != raw["session_user"]
            or _digest(
                raw["tls_peer_certificate_sha256"],
                "foundation_tls_peer_digest_invalid",
            )
            != raw["tls_peer_certificate_sha256"]
            or raw["event_log_shape"] not in {
                "absent",
                "canonical14",
                "legacy19",
            }
            or type(raw["legacy_archive_present"]) is not bool
            or type(raw["writer_ping_present"]) is not bool
            or not isinstance(raw["database_acl"], list)
            or not isinstance(raw["public_schema_acl"], list)
            or raw["observation_sha256"] != _sha256_json(unsigned)
        ):
            raise CanonicalWriterFoundationError("foundation_observation_identity_invalid")
        if raw["database_owner"] != DATABASE_OWNER_ROLE:
            raise CanonicalWriterFoundationError("foundation_database_owner_drifted")
        archive_identity = raw["legacy_archive_identity"]
        if raw["legacy_archive_present"] is False:
            if archive_identity is not None:
                raise CanonicalWriterFoundationError(
                    "foundation_legacy_archive_identity_invalid"
                )
        elif (
            not isinstance(archive_identity, Mapping)
            or set(archive_identity) != _LEGACY_ARCHIVE_IDENTITY_FIELDS
            or not isinstance(archive_identity.get("oid"), str)
            or re.fullmatch(r"[1-9][0-9]*", archive_identity["oid"]) is None
            or not isinstance(archive_identity.get("owner"), str)
            or _LEGACY_SOURCE_OWNER_RE.fullmatch(archive_identity["owner"])
            is None
            or _ADMIN_RE.fullmatch(archive_identity["owner"]) is not None
            or archive_identity["owner"] in _LEGACY_ARCHIVE_RESERVED_OWNERS
            or archive_identity.get("relation_kind") != "r"
            or archive_identity.get("persistence") != "p"
            or archive_identity.get("owner_superuser") is not False
            or archive_identity.get("owner_create_database") is not False
            or archive_identity.get("owner_create_role") is not False
            or archive_identity.get("owner_replication") is not False
            or archive_identity.get("owner_bypass_row_security") is not False
            or type(archive_identity.get("owner_connection_limit")) is not int
            or archive_identity.get("owner_connection_limit") != -1
            or archive_identity.get("owner_validity_is_unbounded") is not True
            or archive_identity.get("owner_configuration_is_empty") is not True
        ):
            raise CanonicalWriterFoundationError(
                "foundation_legacy_archive_identity_invalid"
            )
        temporary_admin_roles = raw["temporary_admin_roles"]
        if (
            not isinstance(temporary_admin_roles, list)
            or any(
                not isinstance(name, str) or _ADMIN_RE.fullmatch(name) is None
                for name in temporary_admin_roles
            )
            or temporary_admin_roles != sorted(set(temporary_admin_roles))
        ):
            raise CanonicalWriterFoundationError(
                "foundation_temporary_admin_observation_invalid"
            )
        expected_temporary_admin_roles = (
            [] if raw["session_user"] == SQL_USER else [raw["session_user"]]
        )
        if temporary_admin_roles != expected_temporary_admin_roles:
            raise CanonicalWriterFoundationError(
                "foundation_temporary_admin_authority_drifted"
            )
        roles = _validate_role_rows(raw["roles"])
        memberships = _validate_membership_rows(raw["memberships"])
        role_names = {str(row["name"]) for row in roles}
        membership_pairs = {
            (str(row["granted_role"]), str(row["member_role"]))
            for row in memberships
        }
        if membership_pairs - {
            (CANARY_BOOTSTRAP_ROLE, CANARY_BOOTSTRAP_LOGIN),
            (WRITER_ROLE, SQL_USER),
        }:
            raise CanonicalWriterFoundationError("foundation_membership_authority_drifted")
        if any(
            pair[0] not in role_names
            or (
                pair[1] not in role_names
                and pair[1] != raw["session_user"]
            )
            for pair in membership_pairs
        ):
            raise CanonicalWriterFoundationError("foundation_membership_role_missing")
        if raw["canonical_schema_owner"] not in {None, MIGRATION_OWNER_ROLE}:
            raise CanonicalWriterFoundationError("foundation_schema_owner_drifted")
        if raw["writer_ping_present"] and raw["canonical_schema_owner"] is None:
            raise CanonicalWriterFoundationError("foundation_schema_state_ambiguous")
        event_shape = raw["event_log_shape"]
        event_owner = raw["event_log_owner"]
        if event_shape == "absent":
            if event_owner is not None or raw["legacy_archive_present"]:
                raise CanonicalWriterFoundationError("foundation_event_truth_ambiguous")
        elif event_shape == "canonical14":
            if event_owner != MIGRATION_OWNER_ROLE:
                raise CanonicalWriterFoundationError("foundation_event_owner_drifted")
        elif (
            not isinstance(event_owner, str)
            or not event_owner
            or event_owner in {
                "postgres",
                "cloudsqladmin",
                "cloudsqlsuperuser",
                MIGRATION_OWNER_ROLE,
                WRITER_ROLE,
                CANARY_BOOTSTRAP_ROLE,
                CANARY_BOOTSTRAP_LOGIN,
                SQL_USER,
            }
            or raw["legacy_archive_present"]
        ):
            raise CanonicalWriterFoundationError("foundation_legacy_truth_ambiguous")
        legacy = raw["legacy_truth"]
        if event_shape == "legacy19":
            if not isinstance(legacy, Mapping) or set(legacy) != {
                "source_owner",
                "source_row_count",
                "canonical14_sha256",
                "extended19_sha256",
                "occurred_at_cutoff",
                "inserted_at_cutoff",
                "bridge_admin",
                "bridge_admin_option",
                "bridge_inherit_option",
                "bridge_set_option",
                "bridge_membership_count",
            }:
                raise CanonicalWriterFoundationError("foundation_legacy_observation_invalid")
            if (
                legacy["source_owner"] != event_owner
                or type(legacy["source_row_count"]) is not int
                or legacy["source_row_count"] <= 0
                or _SHA256_RE.fullmatch(str(legacy["canonical14_sha256"])) is None
                or _SHA256_RE.fullmatch(str(legacy["extended19_sha256"])) is None
                or not isinstance(legacy["occurred_at_cutoff"], str)
                or not legacy["occurred_at_cutoff"]
                or not isinstance(legacy["inserted_at_cutoff"], str)
                or not legacy["inserted_at_cutoff"]
                or legacy["bridge_admin"] != raw["session_user"]
                or legacy["bridge_admin_option"] is not True
                or legacy["bridge_inherit_option"] is not True
                or legacy["bridge_set_option"] is not True
                or legacy["bridge_membership_count"] != 1
            ):
                raise CanonicalWriterFoundationError("foundation_legacy_observation_invalid")
        elif legacy is not None:
            raise CanonicalWriterFoundationError("foundation_legacy_observation_unexpected")
        credential = _credential_mapping(raw["credential"])
        observation = cls(json.loads(_canonical_bytes(raw).decode("utf-8")))
        if raw["session_user"] == SQL_USER and (
            not observation.membership_ready
            or credential["state"] != "installed"
            or raw["legacy_truth"] is not None
        ):
            raise CanonicalWriterFoundationError(
                "foundation_writer_observation_not_terminal"
            )
        return observation

    @property
    def sha256(self) -> str:
        return str(self.value["observation_sha256"])

    @property
    def role_map(self) -> Mapping[str, Mapping[str, Any]]:
        return {str(item["name"]): item for item in self.value["roles"]}

    @property
    def membership_pairs(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            (str(item["granted_role"]), str(item["member_role"]))
            for item in self.value["memberships"]
        )

    @property
    def prerequisites_ready(self) -> bool:
        roles = self.role_map
        login = roles.get(SQL_USER)
        return (
            set(roles) == set(_EXPECTED_ROLE_SHAPES)
            and self.membership_pairs
            == frozenset(
                {(CANARY_BOOTSTRAP_ROLE, CANARY_BOOTSTRAP_LOGIN)}
            )
            and login is not None
            and login["can_login"] is False
            and self.value["event_log_shape"] == "canonical14"
        )

    @property
    def truth_ready(self) -> bool:
        return (
            self.value["event_log_shape"] == "canonical14"
            and self.value["event_log_owner"] == MIGRATION_OWNER_ROLE
        )

    @property
    def login_ready(self) -> bool:
        login = self.role_map.get(SQL_USER)
        return (
            self.truth_ready
            and login is not None
            and login["can_login"] is True
            and self.membership_pairs
            in {
                frozenset(
                    {(CANARY_BOOTSTRAP_ROLE, CANARY_BOOTSTRAP_LOGIN)}
                ),
                frozenset(
                    {
                        (CANARY_BOOTSTRAP_ROLE, CANARY_BOOTSTRAP_LOGIN),
                        (WRITER_ROLE, SQL_USER),
                    }
                ),
            }
        )

    @property
    def base_ready(self) -> bool:
        return self.login_ready and self.value["writer_ping_present"] is True

    @property
    def membership_ready(self) -> bool:
        return self.base_ready and self.membership_pairs == frozenset(
            {
                (CANARY_BOOTSTRAP_ROLE, CANARY_BOOTSTRAP_LOGIN),
                (WRITER_ROLE, SQL_USER),
            }
        )

    @property
    def retired(self) -> bool:
        login = self.role_map.get(SQL_USER)
        return (
            self.value["writer_ping_present"] is True
            and login is not None
            and login["can_login"] is False
            and (WRITER_ROLE, SQL_USER) not in self.membership_pairs
        )

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(_canonical_bytes(dict(self.value)).decode("utf-8"))


def observe_foundation(
    release_revision: str,
    admin_session: AdminSession,
    *,
    _secret_store: SecretStore | None = None,
    _artifacts: Mapping[str, SealedSQLArtifact] | None = None,
    _plan_sha256: str | None = None,
) -> FoundationObservation:
    """Observe only the one fixed isolated database and credential target."""

    session_username = getattr(admin_session, "username", None)
    if not _is_foundation_observer_username(session_username):
        raise CanonicalWriterFoundationError(
            "foundation_observation_session_identity_invalid"
        )
    tls_peer = _digest(
        getattr(admin_session, "tls_peer_certificate_sha256", None),
        "foundation_admin_tls_peer_invalid",
    )
    artifacts = (
        _load_sealed_artifacts(_revision(release_revision))
        if _artifacts is None
        else _artifacts
    )
    observed = dict(
        _query_json(
            admin_session,
            artifacts["observe"],
            expected_column="foundation_observation",
        )
    )
    if observed.get("session_user") != session_username:
        raise CanonicalWriterFoundationError("foundation_observation_session_drifted")
    if session_username == SQL_USER and observed.get("event_log_shape") != "canonical14":
        raise CanonicalWriterFoundationError(
            "foundation_writer_observation_not_terminal"
        )
    legacy: Mapping[str, Any] | None = None
    if observed.get("event_log_shape") == "legacy19":
        legacy = _query_json(
            admin_session,
            artifacts["legacy_observe"],
            expected_column="legacy_truth_observation",
        )
    store = PersistentWriterSecretStore() if _secret_store is None else _secret_store
    unsigned = {
        "schema": FOUNDATION_OBSERVATION_SCHEMA,
        **observed,
        "tls_peer_certificate_sha256": tls_peer,
        "legacy_truth": None if legacy is None else dict(legacy),
        "credential": dict(store.observe(_plan_sha256)),
    }
    return FoundationObservation.from_mapping(
        {**unsigned, "observation_sha256": _sha256_json(unsigned)}
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _list_xattrs(path: Path) -> tuple[str, ...]:
    lister = getattr(os, "listxattr", None)
    if not callable(lister):
        if os.name == "posix" and "linux" in sys.platform:
            raise CanonicalWriterFoundationError("foundation_xattr_inspection_unavailable")
        return ()
    try:
        return tuple(lister(path, follow_symlinks=False))
    except OSError as exc:
        raise CanonicalWriterFoundationError("foundation_xattr_inspection_failed") from exc


def _validate_secret_parent() -> None:
    parent = DATABASE_CREDENTIAL_PATH.parent
    if not parent.is_absolute():
        raise CanonicalWriterFoundationError("foundation_credential_parent_untrusted")
    descriptor = os.open(
        parent.anchor,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    cursor = Path(parent.anchor)
    try:
        for component in parent.parts[1:]:
            cursor /= component
            before = cursor.lstat()
            child = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            opened = os.fstat(child)
            reachable = cursor.lstat()
            os.close(descriptor)
            descriptor = child
            if (
                not stat.S_ISDIR(before.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or _filesystem_identity(before) != _filesystem_identity(opened)
                or _filesystem_identity(before) != _filesystem_identity(reachable)
                or opened.st_uid != 0
                or stat.S_IMODE(opened.st_mode) & 0o022
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_credential_ancestor_untrusted"
                )
        if _list_xattrs(parent):
            raise CanonicalWriterFoundationError(
                "foundation_credential_parent_untrusted"
            )
    except CanonicalWriterFoundationError:
        raise
    except OSError as exc:
        raise CanonicalWriterFoundationError(
            "foundation_credential_parent_missing"
        ) from exc
    finally:
        os.close(descriptor)


def _secret_stage_path(plan_sha256: str) -> Path:
    digest = _digest(plan_sha256, "foundation_secret_plan_digest_invalid")
    return DATABASE_CREDENTIAL_PATH.with_name(
        ".canonical-writer-db-password.foundation." + digest[:24]
    )


def _secret_metadata(
    item: os.stat_result,
    *,
    state: str,
    stage_path: Path | None,
) -> Mapping[str, Any]:
    value = {
        "state": state,
        "path": str(DATABASE_CREDENTIAL_PATH),
        "stage_path": None if stage_path is None else str(stage_path),
        "device": item.st_dev,
        "inode": item.st_ino,
        "owner_uid": item.st_uid,
        "group_gid": item.st_gid,
        "mode": f"{stat.S_IMODE(item.st_mode):04o}",
        "link_count": item.st_nlink,
        "modification_time_ns": item.st_mtime_ns,
        "change_time_ns": item.st_ctime_ns,
        "content_or_digest_recorded": False,
    }
    return _credential_mapping(value)


def _absent_credential() -> Mapping[str, Any]:
    return _credential_mapping(
        {
            "state": "absent",
            "path": str(DATABASE_CREDENTIAL_PATH),
            "stage_path": None,
            "device": None,
            "inode": None,
            "owner_uid": None,
            "group_gid": None,
            "mode": None,
            "link_count": None,
            "modification_time_ns": None,
            "change_time_ns": None,
            "content_or_digest_recorded": False,
        }
    )


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _filesystem_identity(item: os.stat_result) -> tuple[int, ...]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _credential_identity(value: Mapping[str, Any]) -> tuple[Any, ...]:
    metadata = _credential_mapping(value)
    return tuple(
        metadata[field]
        for field in (
            "device",
            "inode",
            "owner_uid",
            "group_gid",
            "mode",
            "link_count",
            "modification_time_ns",
            "change_time_ns",
        )
    )


def _require_same_credential_identity(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    allow_state_change: bool = False,
) -> None:
    left = _credential_mapping(expected)
    right = _credential_mapping(observed)
    if (
        (not allow_state_change and left["state"] != right["state"])
        or left["path"] != right["path"]
        or left["stage_path"] != right["stage_path"]
        or _credential_identity(left) != _credential_identity(right)
    ):
        raise CanonicalWriterFoundationError(
            "foundation_credential_identity_drifted"
        )


def _expected_secret_filesystem_identity(
    metadata: Mapping[str, Any],
) -> tuple[int, ...]:
    value = _credential_mapping(metadata)
    expected_size = 0 if value["state"] == "allocated" else 64
    return (
        value["device"],
        value["inode"],
        stat.S_IFREG | int(str(value["mode"]), 8),
        value["owner_uid"],
        value["group_gid"],
        value["link_count"],
        expected_size,
        value["modification_time_ns"],
        value["change_time_ns"],
    )


def _validate_secret_stat(item: os.stat_result, *, allow_two_links: bool) -> None:
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISREG(item.st_mode)
        or item.st_uid != WRITER_UID
        or item.st_gid != WRITER_GID
        or stat.S_IMODE(item.st_mode) != 0o400
        or item.st_nlink not in ({1, 2} if allow_two_links else {1})
        or item.st_size not in {0, 64}
    ):
        raise CanonicalWriterFoundationError("foundation_credential_inode_untrusted")


def _read_secret_inode(path: Path, expected: Mapping[str, Any]) -> bytearray:
    metadata = _credential_mapping(expected)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    raw = bytearray()
    try:
        before = path.lstat()
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            while len(raw) <= 64:
                chunk = os.read(descriptor, 65)
                if not chunk:
                    break
                raw.extend(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        reachable = path.lstat()
    except OSError as exc:
        _zeroize(raw)
        raise CanonicalWriterFoundationError("foundation_credential_read_failed") from exc
    expected_identity = _expected_secret_filesystem_identity(metadata)
    for item in (before, opened, after, reachable):
        _validate_secret_stat(item, allow_two_links=metadata["link_count"] == 2)
        if _filesystem_identity(item) != expected_identity:
            _zeroize(raw)
            raise CanonicalWriterFoundationError(
                "foundation_credential_identity_drifted"
            )
    if len(raw) != 64 or _WRITER_SECRET_RE.fullmatch(raw) is None:
        _zeroize(raw)
        raise CanonicalWriterFoundationError("foundation_credential_material_invalid")
    return raw


class PersistentWriterSecretStore:
    """Root-only no-replace credential staging at one compile-time path."""

    @staticmethod
    def _stage_candidates() -> tuple[Path, ...]:
        _validate_secret_parent()
        return tuple(
            sorted(
                DATABASE_CREDENTIAL_PATH.parent.glob(
                    ".canonical-writer-db-password.foundation.*"
                )
            )
        )

    def observe(self, plan_sha256: str | None) -> Mapping[str, Any]:
        expected_stage = None if plan_sha256 is None else _secret_stage_path(plan_sha256)
        candidates = self._stage_candidates()
        if any(candidate != expected_stage for candidate in candidates):
            raise CanonicalWriterFoundationError("foundation_cross_plan_secret_residue")
        target = (
            DATABASE_CREDENTIAL_PATH.lstat()
            if os.path.lexists(DATABASE_CREDENTIAL_PATH)
            else None
        )
        stage = (
            expected_stage.lstat()
            if expected_stage is not None and os.path.lexists(expected_stage)
            else None
        )
        if target is None and stage is None:
            return _absent_credential()
        for item in (target, stage):
            if item is not None:
                _validate_secret_stat(item, allow_two_links=True)
        if target is not None and stage is not None:
            if not _same_inode(target, stage) or target.st_nlink != 2 or stage.st_nlink != 2:
                raise CanonicalWriterFoundationError("foundation_credential_link_state_invalid")
            return _secret_metadata(
                target,
                state="staged" if target.st_size == 64 else "allocated",
                stage_path=expected_stage,
            )
        if stage is not None:
            return _secret_metadata(
                stage,
                state="staged" if stage.st_size == 64 else "allocated",
                stage_path=expected_stage,
            )
        assert target is not None
        if target.st_size != 64:
            raise CanonicalWriterFoundationError("foundation_credential_target_incomplete")
        return _secret_metadata(target, state="installed", stage_path=None)

    def allocate(self, plan_sha256: str) -> Mapping[str, Any]:
        _validate_secret_parent()
        stage = _secret_stage_path(plan_sha256)
        if os.path.lexists(DATABASE_CREDENTIAL_PATH):
            raise CanonicalWriterFoundationError("foundation_credential_target_not_fresh")
        if os.path.lexists(stage):
            item = stage.lstat()
            _validate_secret_stat(item, allow_two_links=False)
            if item.st_size != 0:
                raise CanonicalWriterFoundationError("foundation_secret_allocation_ambiguous")
            return _secret_metadata(item, state="allocated", stage_path=stage)
        descriptor = -1
        completed = False
        try:
            descriptor = os.open(
                stage,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            os.fchown(descriptor, WRITER_UID, WRITER_GID)
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            item = os.fstat(descriptor)
            _validate_secret_stat(item, allow_two_links=False)
            if item.st_size != 0:
                raise CanonicalWriterFoundationError("foundation_secret_allocation_invalid")
            os.close(descriptor)
            descriptor = -1
            _fsync_directory(stage.parent)
            reachable = stage.lstat()
            if not _same_inode(item, reachable):
                raise CanonicalWriterFoundationError("foundation_secret_allocation_drifted")
            completed = True
            return _secret_metadata(reachable, state="allocated", stage_path=stage)
        except CanonicalWriterFoundationError:
            raise
        except OSError as exc:
            raise CanonicalWriterFoundationError("foundation_secret_allocation_failed") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if not completed and os.path.lexists(stage):
                # An unjournaled allocation is empty by construction.  Remove
                # only this call's exact fresh inode; otherwise preserve it.
                try:
                    current = stage.lstat()
                    if "item" in locals() and _same_inode(item, current) and current.st_size == 0:
                        stage.unlink()
                        _fsync_directory(stage.parent)
                except OSError:
                    pass

    def materialize(
        self,
        plan_sha256: str,
        expected: Mapping[str, Any],
        factory: Callable[[], bytes],
    ) -> tuple[bytearray, Mapping[str, Any]]:
        metadata = _credential_mapping(expected)
        stage = _secret_stage_path(plan_sha256)
        if metadata["stage_path"] != str(stage) or metadata["state"] not in {
            "allocated",
            "staged",
        }:
            raise CanonicalWriterFoundationError("foundation_secret_stage_binding_invalid")
        observed = self.observe(plan_sha256)
        _require_same_credential_identity(metadata, observed)
        if not os.path.lexists(stage) or os.path.lexists(DATABASE_CREDENTIAL_PATH):
            raise CanonicalWriterFoundationError("foundation_secret_stage_drifted")
        if metadata["state"] == "staged":
            return _read_secret_inode(stage, metadata), metadata
        descriptor = os.open(
            stage,
            os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        generated: bytearray | None = None
        try:
            before = stage.lstat()
            item = os.fstat(descriptor)
            if (
                _filesystem_identity(before)
                != _expected_secret_filesystem_identity(metadata)
                or _filesystem_identity(item)
                != _expected_secret_filesystem_identity(metadata)
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_credential_identity_drifted"
                )
            candidate = factory()
            if not isinstance(candidate, bytes):
                raise CanonicalWriterFoundationError("foundation_secret_factory_invalid")
            generated = bytearray(candidate)
            candidate = b""
            if len(generated) != 64 or _WRITER_SECRET_RE.fullmatch(generated) is None:
                raise CanonicalWriterFoundationError("foundation_secret_factory_invalid")
            offset = 0
            while offset < len(generated):
                written = os.write(descriptor, generated[offset:])
                if written <= 0:
                    raise OSError("secret write stalled")
                offset += written
            os.fsync(descriptor)
            after = os.fstat(descriptor)
            _validate_secret_stat(after, allow_two_links=False)
            if after.st_size != 64:
                raise CanonicalWriterFoundationError(
                    "foundation_secret_materialize_unconfirmed"
                )
        except CanonicalWriterFoundationError:
            _zeroize(generated)
            raise
        except OSError as exc:
            _zeroize(generated)
            raise CanonicalWriterFoundationError("foundation_secret_materialize_failed") from exc
        finally:
            os.close(descriptor)
        reachable = stage.lstat()
        if _filesystem_identity(after) != _filesystem_identity(reachable):
            _zeroize(generated)
            raise CanonicalWriterFoundationError(
                "foundation_credential_identity_drifted"
            )
        staged = _secret_metadata(reachable, state="staged", stage_path=stage)
        staged_secret = _read_secret_inode(stage, staged)
        _zeroize(generated)
        return staged_secret, staged

    def publish(
        self,
        plan_sha256: str,
        expected: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        metadata = _credential_mapping(expected)
        stage = _secret_stage_path(plan_sha256)
        if metadata["state"] != "staged" or metadata["stage_path"] != str(stage):
            raise CanonicalWriterFoundationError("foundation_secret_stage_binding_invalid")
        _validate_secret_parent()
        observed = self.observe(plan_sha256)
        _require_same_credential_identity(metadata, observed)
        if os.path.lexists(DATABASE_CREDENTIAL_PATH) or not os.path.lexists(stage):
            raise CanonicalWriterFoundationError("foundation_secret_publish_state_invalid")
        try:
            source_before = stage.lstat()
            if (
                _filesystem_identity(source_before)
                != _expected_secret_filesystem_identity(metadata)
                or source_before.st_nlink != 1
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_credential_identity_drifted"
                )
            os.link(stage, DATABASE_CREDENTIAL_PATH, follow_symlinks=False)
            source_linked = stage.lstat()
            target_linked = DATABASE_CREDENTIAL_PATH.lstat()
            if (
                not _same_inode(source_linked, target_linked)
                or source_linked.st_nlink != 2
                or target_linked.st_nlink != 2
                or _filesystem_identity(source_linked)
                != _filesystem_identity(target_linked)
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_secret_link_transition_invalid"
                )
            linked_fd = os.open(
                DATABASE_CREDENTIAL_PATH,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(linked_fd)
            finally:
                os.close(linked_fd)
            _fsync_directory(DATABASE_CREDENTIAL_PATH.parent)
            stage.unlink()
            final = DATABASE_CREDENTIAL_PATH.lstat()
            if (
                not _same_inode(source_before, final)
                or final.st_nlink != 1
                or final.st_size != 64
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_secret_unlink_transition_invalid"
                )
            final_fd = os.open(
                DATABASE_CREDENTIAL_PATH,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(final_fd)
            finally:
                os.close(final_fd)
            _fsync_directory(DATABASE_CREDENTIAL_PATH.parent)
        except OSError as exc:
            raise CanonicalWriterFoundationError("foundation_secret_publish_failed") from exc
        final_metadata = self.observe(plan_sha256)
        if final_metadata["state"] != "installed" or (
            final_metadata["device"], final_metadata["inode"]
        ) != (metadata["device"], metadata["inode"]):
            raise CanonicalWriterFoundationError("foundation_secret_publish_unconfirmed")
        return final_metadata

    def read(self, expected: Mapping[str, Any]) -> bytearray:
        metadata = _credential_mapping(expected)
        if metadata["state"] != "installed":
            raise CanonicalWriterFoundationError("foundation_credential_not_installed")
        return _read_secret_inode(DATABASE_CREDENTIAL_PATH, metadata)

    def remove(self, expected: Mapping[str, Any]) -> None:
        metadata = _credential_mapping(expected)
        if metadata["state"] != "installed":
            raise CanonicalWriterFoundationError("foundation_credential_not_installed")
        try:
            current = DATABASE_CREDENTIAL_PATH.lstat()
            _validate_secret_stat(current, allow_two_links=False)
            if _filesystem_identity(current) != _expected_secret_filesystem_identity(
                metadata
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_credential_identity_drifted"
                )
            DATABASE_CREDENTIAL_PATH.unlink()
            _fsync_directory(DATABASE_CREDENTIAL_PATH.parent)
        except CanonicalWriterFoundationError:
            raise
        except OSError as exc:
            raise CanonicalWriterFoundationError("foundation_credential_remove_failed") from exc
        if os.path.lexists(DATABASE_CREDENTIAL_PATH):
            raise CanonicalWriterFoundationError("foundation_credential_remove_unconfirmed")


_PLAN_FIELDS = frozenset(
    {
        "schema",
        "mode",
        "release_revision",
        "target",
        "initial_observation",
        "initial_observation_sha256",
        "artifact_sha256",
        "artifact_set_sha256",
        "states",
        "legacy_reconciliation_required",
        "legacy_source_owner",
        "credential_contract",
        "terminal_attestation_catalog_sha256",
        "adoption_terminal_attestation",
        "plan_sha256",
    }
)
_TARGET_FIELDS = frozenset(
    {
        "project",
        "zone",
        "vm_name",
        "vm_instance_id",
        "sql_instance",
        "host",
        "tls_server_name",
        "port",
        "database",
        "writer_login",
        "migration_owner_role",
        "writer_role",
        "bootstrap_role",
        "bootstrap_login",
        "database_ca_path",
        "credential_path",
        "evidence_root",
    }
)


def _fixed_target() -> Mapping[str, Any]:
    return {
        "project": PROJECT,
        "zone": ZONE,
        "vm_name": VM_NAME,
        "vm_instance_id": VM_INSTANCE_ID,
        "sql_instance": SQL_INSTANCE,
        "host": SQL_HOST,
        "tls_server_name": SQL_TLS_SERVER_NAME,
        "port": SQL_PORT,
        "database": SQL_DATABASE,
        "writer_login": SQL_USER,
        "migration_owner_role": MIGRATION_OWNER_ROLE,
        "writer_role": WRITER_ROLE,
        "bootstrap_role": CANARY_BOOTSTRAP_ROLE,
        "bootstrap_login": CANARY_BOOTSTRAP_LOGIN,
        "database_ca_path": str(DATABASE_CA_PATH),
        "credential_path": str(DATABASE_CREDENTIAL_PATH),
        "evidence_root": str(FOUNDATION_EVIDENCE_ROOT),
    }


@dataclass(frozen=True)
class FoundationPlan:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "FoundationPlan":
        if not isinstance(value, Mapping) or set(value) != _PLAN_FIELDS:
            raise CanonicalWriterFoundationError("foundation_plan_fields_invalid")
        raw = copy.deepcopy(dict(value))
        unsigned = {key: item for key, item in raw.items() if key != "plan_sha256"}
        target = raw["target"]
        artifact_sha = raw["artifact_sha256"]
        mode = raw["mode"]
        expected_states = (
            FOUNDATION_STATES
            if mode == "create_pristine"
            else FOUNDATION_ADOPTION_STATES
        )
        if (
            raw["schema"] != FOUNDATION_PLAN_SCHEMA
            or mode not in FOUNDATION_PLAN_MODES
            or _revision(raw["release_revision"]) != raw["release_revision"]
            or not isinstance(target, Mapping)
            or set(target) != _TARGET_FIELDS
            or dict(target) != dict(_fixed_target())
            or FoundationObservation.from_mapping(raw["initial_observation"]).sha256
            != raw["initial_observation_sha256"]
            or not isinstance(artifact_sha, Mapping)
            or set(artifact_sha) != set(_ARTIFACT_FILENAMES)
            or any(_SHA256_RE.fullmatch(str(item)) is None for item in artifact_sha.values())
            or raw["artifact_set_sha256"] != _sha256_json(dict(artifact_sha))
            or tuple(raw["states"]) != expected_states
            or type(raw["legacy_reconciliation_required"]) is not bool
            or _digest(
                raw["terminal_attestation_catalog_sha256"],
                "foundation_catalog_digest_invalid",
            )
            != PRODUCTION_CATALOG_SHA256
            or raw["plan_sha256"] != _sha256_json(unsigned)
        ):
            raise CanonicalWriterFoundationError("foundation_plan_identity_invalid")
        initial = FoundationObservation.from_mapping(raw["initial_observation"])
        legacy_required = initial.value["event_log_shape"] == "legacy19"
        legacy_source = (
            initial.value["legacy_truth"]["source_owner"]
            if legacy_required
            else None
        )
        credential_contract = raw["credential_contract"]
        adoption_attestation = raw["adoption_terminal_attestation"]
        if (
            raw["legacy_reconciliation_required"] is not legacy_required
            or raw["legacy_source_owner"] != legacy_source
            or not isinstance(credential_contract, Mapping)
            or dict(credential_contract)
            != {
                "generated_on_vm": True,
                "bytes": 64,
                "alphabet": "base64url-no-padding",
                "owner_uid": WRITER_UID,
                "group_gid": WRITER_GID,
                "mode": "0400",
                "scram_mechanism": "SCRAM-SHA-256",
                "password_transport": "postgres-copy-data-v1",
                "server_generated_scram_salt": True,
                "copy_data_only": True,
                "statement_logging_safe_without_privileged_guc": True,
                "temporary_security_definer": True,
                "password_or_verifier_serialized": False,
                "content_or_digest_recorded": False,
            }
            or (
                mode == "create_pristine"
                and adoption_attestation is not None
            )
            or (
                mode == "adopted_existing_terminal"
                and _validate_terminal_attestation(adoption_attestation)
                != adoption_attestation
            )
        ):
            raise CanonicalWriterFoundationError("foundation_plan_contract_invalid")
        return cls(json.loads(_canonical_bytes(raw).decode("utf-8")))

    @property
    def sha256(self) -> str:
        return str(self.value["plan_sha256"])

    @property
    def revision(self) -> str:
        return str(self.value["release_revision"])

    @property
    def initial_observation(self) -> FoundationObservation:
        return FoundationObservation.from_mapping(self.value["initial_observation"])

    @property
    def mode(self) -> str:
        return str(self.value["mode"])

    @property
    def states(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.value["states"])

    @property
    def artifacts(self) -> Mapping[str, str]:
        return dict(self.value["artifact_sha256"])

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(_canonical_bytes(dict(self.value)).decode("utf-8"))


def build_foundation_plan(
    release_revision: str,
    observation: FoundationObservation,
    *,
    _artifacts: Mapping[str, SealedSQLArtifact] | None = None,
) -> FoundationPlan:
    """Fail closed until the Cloud-admin deletion workflow exists.

    PostgreSQL 18 managed/non-superuser creation necessarily leaves authority
    granted by the external role creator.  This Phase A module cannot delete
    that administrator and therefore cannot produce a truthful terminal state.
    """

    if not isinstance(observation, FoundationObservation):
        raise TypeError("foundation observation is required")
    _revision(release_revision)
    raise CanonicalWriterFoundationError(
        "foundation_creation_requires_admin_delete_integration"
    )


def _validate_plan_artifacts(
    plan: FoundationPlan,
    artifacts: Mapping[str, SealedSQLArtifact],
) -> None:
    observed = _artifact_digest_mapping(artifacts)
    if observed != dict(plan.artifacts) or _sha256_json(observed) != plan.value[
        "artifact_set_sha256"
    ]:
        raise CanonicalWriterFoundationError("foundation_plan_artifact_drifted")


_JOURNAL_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "plan_sha256",
        "artifact_set_sha256",
        "sequence",
        "state",
        "transition_phase",
        "transition_to",
        "previous_entry_sha256",
        "database_observation_sha256",
        "credential",
        "scram_salt_base64",
        "managed_hba_receipt_sha256",
        "terminal_attestation",
        "recorded_at_unix",
        "secret_material_recorded",
        "entry_sha256",
    }
)
_TERMINAL_ATTESTATION_FIELDS = frozenset(
    {
        "catalog_sha256",
        "privilege_attestation_sha256",
        "private_schema_identity_sha256",
        "managed_hba_receipt_sha256",
        "writer_ping_ok",
        "writer_ping_service",
        "writer_ping_protocol",
        "table_grants",
        "general_sql_available",
        "cross_database_connect_available",
        "dangerous_role_attributes",
        "migration_admin_membership_present",
    }
)


def _validate_terminal_attestation(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TERMINAL_ATTESTATION_FIELDS:
        raise CanonicalWriterFoundationError("foundation_terminal_attestation_invalid")
    raw = dict(value)
    if (
        _digest(raw["catalog_sha256"], "foundation_catalog_digest_invalid")
        != PRODUCTION_CATALOG_SHA256
        or _digest(
            raw["privilege_attestation_sha256"],
            "foundation_privilege_attestation_digest_invalid",
        )
        != raw["privilege_attestation_sha256"]
        or _digest(
            raw["private_schema_identity_sha256"],
            "foundation_private_schema_digest_invalid",
        )
        != raw["private_schema_identity_sha256"]
        or _digest(
            raw["managed_hba_receipt_sha256"],
            "foundation_hba_digest_invalid",
        )
        != raw["managed_hba_receipt_sha256"]
        or raw["writer_ping_ok"] is not True
        or raw["writer_ping_service"] != "canonical_writer"
        or raw["writer_ping_protocol"] != "v1"
        or raw["table_grants"] != []
        or raw["general_sql_available"] is not False
        or raw["cross_database_connect_available"] is not False
        or raw["dangerous_role_attributes"] != []
        or raw["migration_admin_membership_present"] is not False
    ):
        raise CanonicalWriterFoundationError("foundation_terminal_attestation_invalid")
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


@dataclass(frozen=True)
class FoundationJournalEntry:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        plan: FoundationPlan,
    ) -> "FoundationJournalEntry":
        if not isinstance(value, Mapping) or set(value) != _JOURNAL_FIELDS:
            raise CanonicalWriterFoundationError("foundation_journal_fields_invalid")
        raw = copy.deepcopy(dict(value))
        unsigned = {key: item for key, item in raw.items() if key != "entry_sha256"}
        state = raw["state"]
        phase = raw["transition_phase"]
        target = raw["transition_to"]
        states = plan.states
        if (
            raw["schema"] != FOUNDATION_JOURNAL_SCHEMA
            or raw["release_revision"] != plan.revision
            or raw["plan_sha256"] != plan.sha256
            or raw["artifact_set_sha256"] != plan.value["artifact_set_sha256"]
            or type(raw["sequence"]) is not int
            or raw["sequence"] < 0
            or state not in states
            or phase not in {"complete", "prepared"}
            or (
                phase == "complete" and target is not None
            )
            or (
                phase == "prepared"
                and (
                    state == states[-1]
                    or target
                    != states[states.index(state) + 1]
                )
            )
            or (
                raw["previous_entry_sha256"] is not None
                and _SHA256_RE.fullmatch(str(raw["previous_entry_sha256"])) is None
            )
            or _SHA256_RE.fullmatch(str(raw["database_observation_sha256"])) is None
            or type(raw["recorded_at_unix"]) is not int
            or raw["recorded_at_unix"] < 0
            or raw["secret_material_recorded"] is not False
            or raw["entry_sha256"] != _sha256_json(unsigned)
        ):
            raise CanonicalWriterFoundationError("foundation_journal_identity_invalid")
        _credential_mapping(raw["credential"])
        if raw["scram_salt_base64"] is not None:
            raise CanonicalWriterFoundationError(
                "foundation_deprecated_scram_salt_forbidden"
            )
        hba_digest = raw["managed_hba_receipt_sha256"]
        if hba_digest is not None:
            _digest(hba_digest, "foundation_hba_digest_invalid")
        attestation = raw["terminal_attestation"]
        if attestation is not None:
            _validate_terminal_attestation(attestation)
        return cls(json.loads(_canonical_bytes(raw).decode("utf-8")))

    @property
    def sha256(self) -> str:
        return str(self.value["entry_sha256"])

    @property
    def state(self) -> str:
        return str(self.value["state"])

    @property
    def prepared(self) -> bool:
        return self.value["transition_phase"] == "prepared"


def _secure_directory(
    path: Path,
    *,
    strict_root: bool,
    mkdir_callback: Callable[[], None] | None = None,
) -> None:
    if not path.is_absolute():
        raise CanonicalWriterFoundationError(
            "foundation_evidence_directory_untrusted"
        )
    missing: list[Path] = []
    cursor = path
    while not os.path.lexists(cursor):
        missing.append(cursor)
        if cursor.parent == cursor:
            raise CanonicalWriterFoundationError(
                "foundation_evidence_parent_missing"
            )
        cursor = cursor.parent
    try:
        existing = cursor.lstat()
    except OSError as exc:
        raise CanonicalWriterFoundationError(
            "foundation_evidence_parent_missing"
        ) from exc
    if stat.S_ISLNK(existing.st_mode) or not stat.S_ISDIR(existing.st_mode):
        raise CanonicalWriterFoundationError(
            "foundation_evidence_ancestor_untrusted"
        )
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
            os.chmod(directory, 0o700, follow_symlinks=False)
            _fsync_directory(directory)
            _fsync_directory(directory.parent)
        except OSError as exc:
            raise CanonicalWriterFoundationError(
                "foundation_evidence_directory_create_failed"
            ) from exc
        if mkdir_callback is not None:
            mkdir_callback()
    descriptor = os.open(
        path.anchor,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    cursor = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            cursor /= component
            before = cursor.lstat()
            child = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            opened = os.fstat(child)
            reachable = cursor.lstat()
            os.close(descriptor)
            descriptor = child
            if (
                not stat.S_ISDIR(before.st_mode)
                or _filesystem_identity(before) != _filesystem_identity(opened)
                or _filesystem_identity(before) != _filesystem_identity(reachable)
                or (
                    strict_root
                    and (
                        opened.st_uid != 0
                        or opened.st_gid != 0
                        or stat.S_IMODE(opened.st_mode) & 0o022
                    )
                )
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_evidence_ancestor_untrusted"
                )
        item = os.fstat(descriptor)
        expected_uid = 0 if strict_root else _effective_uid()
        if (
            stat.S_IMODE(item.st_mode) != 0o700
            or item.st_uid != expected_uid
            or (strict_root and item.st_gid != 0)
            or _list_xattrs(path)
        ):
            raise CanonicalWriterFoundationError(
                "foundation_evidence_directory_untrusted"
            )
    except CanonicalWriterFoundationError:
        raise
    except OSError as exc:
        raise CanonicalWriterFoundationError(
            "foundation_evidence_ancestor_untrusted"
        ) from exc
    finally:
        os.close(descriptor)


class _AppendOnlyFoundationJournal:
    def __init__(
        self,
        root: Path = FOUNDATION_EVIDENCE_ROOT,
        *,
        strict_root: bool = True,
        publication_fault_injector: Callable[[str, str], None] | None = None,
    ) -> None:
        self.root = root
        self.strict_root = strict_root
        self.publication_fault_injector = publication_fault_injector

    def _plan_root(self, plan: FoundationPlan) -> Path:
        return self.root / plan.revision / plan.sha256

    def _journal_root(self, plan: FoundationPlan) -> Path:
        return self._plan_root(plan) / "journal"

    def _terminal_root(self, plan: FoundationPlan) -> Path:
        return self._plan_root(plan) / "terminal"

    def _staging_root(self, plan: FoundationPlan) -> Path:
        return self._plan_root(plan) / "staging"

    def _fault(self, kind: str, point: str) -> None:
        if self.publication_fault_injector is not None:
            self.publication_fault_injector(kind, point)

    def _ensure_directory(self, path: Path, *, kind: str) -> None:
        _secure_directory(
            path,
            strict_root=self.strict_root,
            mkdir_callback=lambda: self._fault(kind, "after_mkdir"),
        )

    def _read_publication_bytes(
        self,
        path: Path,
        *,
        code: str,
        expected_link_count: int = 1,
    ) -> bytes:
        try:
            before = path.lstat()
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o400
                or before.st_nlink != expected_link_count
                or before.st_size <= 0
                or before.st_size > _MAX_PUBLIC_JSON_BYTES
                or (self.strict_root and (before.st_uid != 0 or before.st_gid != 0))
                or _list_xattrs(path)
            ):
                raise CanonicalWriterFoundationError(code)
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                opened = os.fstat(descriptor)
                raw = bytearray()
                while len(raw) <= _MAX_PUBLIC_JSON_BYTES:
                    chunk = os.read(
                        descriptor,
                        min(1024 * 1024, _MAX_PUBLIC_JSON_BYTES + 1),
                    )
                    if not chunk:
                        break
                    raw.extend(chunk)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            reachable = path.lstat()
        except CanonicalWriterFoundationError:
            raise
        except OSError as exc:
            raise CanonicalWriterFoundationError(code) from exc
        payload = bytes(raw)
        if (
            _filesystem_identity(before) != _filesystem_identity(opened)
            or _filesystem_identity(before) != _filesystem_identity(after)
            or _filesystem_identity(before) != _filesystem_identity(reachable)
            or len(payload) != before.st_size
        ):
            raise CanonicalWriterFoundationError(code)
        return payload

    @staticmethod
    def _fsync_publication_file(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _publish_payload(
        self,
        *,
        kind: str,
        stage: Path,
        final: Path,
        payload: bytes,
    ) -> None:
        self._ensure_directory(stage.parent, kind=kind)
        self._ensure_directory(final.parent, kind=kind)
        stage_exists = os.path.lexists(stage)
        final_exists = os.path.lexists(final)
        if final_exists and not stage_exists:
            if self._read_publication_bytes(
                final,
                code="foundation_final_publication_invalid",
                expected_link_count=1,
            ) != payload:
                raise CanonicalWriterFoundationError(
                    "foundation_final_publication_drifted"
                )
            self._fsync_publication_file(final)
            _fsync_directory(final.parent)
            _fsync_directory(stage.parent)
            return
        if stage_exists and final_exists:
            staged_payload = self._read_publication_bytes(
                stage,
                code="foundation_staged_publication_invalid",
                expected_link_count=2,
            )
            final_payload = self._read_publication_bytes(
                final,
                code="foundation_final_publication_invalid",
                expected_link_count=2,
            )
            if (
                staged_payload != payload
                or final_payload != payload
                or not _same_inode(stage.lstat(), final.lstat())
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_publication_link_state_invalid"
                )
            self._fsync_publication_file(final)
            _fsync_directory(final.parent)
            _fsync_directory(stage.parent)
        elif stage_exists:
            if self._read_publication_bytes(
                stage,
                code="foundation_staged_publication_invalid",
                expected_link_count=1,
            ) != payload:
                raise CanonicalWriterFoundationError(
                    "foundation_staged_publication_drifted"
                )
            self._fsync_publication_file(stage)
            self._fault(kind, "after_fsync")
            _fsync_directory(stage.parent)
        else:
            descriptor = os.open(
                stage,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o400,
            )
            try:
                if self.strict_root:
                    os.fchown(descriptor, 0, 0)
                os.fchmod(descriptor, 0o400)
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written <= 0:
                        raise OSError("publication write stalled")
                    offset += written
                self._fault(kind, "after_write")
                os.fsync(descriptor)
                self._fault(kind, "after_fsync")
            finally:
                os.close(descriptor)
            _fsync_directory(stage.parent)
        if not final_exists:
            stage_before_link = stage.lstat()
            if stage_before_link.st_nlink != 1:
                raise CanonicalWriterFoundationError(
                    "foundation_publication_link_state_invalid"
                )
            try:
                os.link(stage, final, follow_symlinks=False)
            except FileExistsError:
                raise CanonicalWriterFoundationError(
                    "foundation_publication_collided"
                ) from None
            except OSError as exc:
                raise CanonicalWriterFoundationError(
                    "foundation_publication_failed"
                ) from exc
            stage_linked = stage.lstat()
            final_linked = final.lstat()
            if (
                not _same_inode(stage_linked, final_linked)
                or stage_linked.st_nlink != 2
                or final_linked.st_nlink != 2
                or _filesystem_identity(stage_linked)
                != _filesystem_identity(final_linked)
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_publication_link_transition_invalid"
                )
            self._fault(kind, "after_publish")
            self._fsync_publication_file(final)
            _fsync_directory(final.parent)
            _fsync_directory(stage.parent)
            self._fault(kind, "after_publish_fsync")
        if os.path.lexists(stage):
            stage_before_unlink = stage.lstat()
            final_before_unlink = final.lstat()
            if (
                not _same_inode(stage_before_unlink, final_before_unlink)
                or stage_before_unlink.st_nlink != 2
                or final_before_unlink.st_nlink != 2
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_publication_link_state_invalid"
                )
            stage.unlink()
            self._fault(kind, "after_unlink")
            final_after_unlink = final.lstat()
            if (
                not _same_inode(final_before_unlink, final_after_unlink)
                or final_after_unlink.st_nlink != 1
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_publication_unlink_transition_invalid"
                )
            self._fsync_publication_file(final)
            _fsync_directory(stage.parent)
            _fsync_directory(final.parent)
        if self._read_publication_bytes(
            final,
            code="foundation_final_publication_invalid",
            expected_link_count=1,
        ) != payload:
            raise CanonicalWriterFoundationError(
                "foundation_final_publication_drifted"
            )

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        self._ensure_directory(self.root, kind="lock")
        lock_path = self.root / ".foundation.lock"
        flags = (
            os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        created = False
        try:
            descriptor = os.open(
                lock_path,
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            created = True
        except FileExistsError:
            descriptor = os.open(lock_path, flags)
        try:
            if created:
                if self.strict_root:
                    os.fchown(descriptor, 0, 0)
                else:
                    # Some sticky/setgid temporary roots (notably /private/tmp
                    # on macOS) inherit a parent group instead of the process
                    # effective group.  Pin the test/non-root lock identity so
                    # the subsequent exact lstat/fstat contract is portable.
                    os.fchown(descriptor, -1, _effective_gid())
                os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
                _fsync_directory(lock_path.parent)
            before = lock_path.lstat()
            opened = os.fstat(descriptor)
            reachable = lock_path.lstat()
            expected_uid = 0 if self.strict_root else _effective_uid()
            expected_gid = 0 if self.strict_root else _effective_gid()
            if (
                not stat.S_ISREG(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != 0o600
                or opened.st_uid != expected_uid
                or opened.st_gid != expected_gid
                or opened.st_nlink != 1
                or opened.st_size != 0
                or _list_xattrs(lock_path)
                or _filesystem_identity(before) != _filesystem_identity(opened)
                or _filesystem_identity(before) != _filesystem_identity(reachable)
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_lock_inode_untrusted"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
            after = os.fstat(descriptor)
            reachable_after = lock_path.lstat()
            if (
                _filesystem_identity(opened) != _filesystem_identity(after)
                or _filesystem_identity(opened)
                != _filesystem_identity(reachable_after)
                or _list_xattrs(lock_path)
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_lock_inode_drifted"
                )
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def assert_no_cross_plan(self, plan: FoundationPlan) -> None:
        self._ensure_directory(self.root, kind="evidence")
        expected = self._plan_root(plan)
        for revision_path in self.root.iterdir():
            if revision_path.name.startswith("."):
                continue
            if not revision_path.is_dir():
                raise CanonicalWriterFoundationError("foundation_evidence_residue_invalid")
            for plan_path in revision_path.iterdir():
                if plan_path != expected:
                    raise CanonicalWriterFoundationError("foundation_cross_plan_residue")

    def _read_entry_file(
        self,
        path: Path,
        *,
        plan: FoundationPlan,
        expected_link_count: int = 1,
    ) -> FoundationJournalEntry:
        raw = self._read_publication_bytes(
            path,
            code="foundation_journal_file_untrusted",
            expected_link_count=expected_link_count,
        )
        entry = self._decode_entry_payload(raw, plan=plan)
        if path.name != f"{entry.value['sequence']:08d}-{entry.sha256}.json":
            raise CanonicalWriterFoundationError("foundation_journal_path_drifted")
        return entry

    def _decode_entry_payload(
        self,
        raw: bytes,
        *,
        plan: FoundationPlan,
    ) -> FoundationJournalEntry:
        if not raw.endswith(b"\n"):
            raise CanonicalWriterFoundationError("foundation_journal_bytes_invalid")
        try:
            decoded = raw[:-1].decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise CanonicalWriterFoundationError("foundation_journal_bytes_invalid") from exc
        value = _strict_json(decoded, "foundation_journal_json_invalid")
        if raw != _canonical_bytes(value) + b"\n":
            raise CanonicalWriterFoundationError("foundation_journal_not_canonical")
        return FoundationJournalEntry.from_mapping(value, plan=plan)

    def load(self, plan: FoundationPlan) -> tuple[FoundationJournalEntry, ...]:
        root = self._journal_root(plan)
        if not root.exists():
            return ()
        self._ensure_directory(root, kind="journal")
        paths = sorted(root.iterdir())
        staging = self._staging_root(plan)
        entries_list: list[FoundationJournalEntry] = []
        for path in paths:
            sequence_match = re.fullmatch(
                r"([0-9]{8})-[0-9a-f]{64}\.json",
                path.name,
            )
            if sequence_match is None:
                raise CanonicalWriterFoundationError(
                    "foundation_journal_path_drifted"
                )
            stage_path = staging / f"journal-{sequence_match.group(1)}.stage"
            entries_list.append(
                self._read_entry_file(
                    path,
                    plan=plan,
                    expected_link_count=(
                        2 if os.path.lexists(stage_path) else 1
                    ),
                )
            )
        entries = tuple(entries_list)
        current_index = -1
        pending_target: str | None = None
        previous: str | None = None
        retained_salt: str | None = None
        retained_attestation: Mapping[str, Any] | None = None
        previous_credential: Mapping[str, Any] | None = None
        credential_rank = {
            "absent": 0,
            "allocated": 1,
            "staged": 2,
            "installed": 3,
        }
        for sequence, entry in enumerate(entries):
            value = entry.value
            if value["sequence"] != sequence or value["previous_entry_sha256"] != previous:
                raise CanonicalWriterFoundationError("foundation_journal_chain_broken")
            states = plan.states
            state_index = states.index(entry.state)
            if sequence == 0:
                if entry.state != states[0] or entry.prepared:
                    raise CanonicalWriterFoundationError("foundation_journal_initial_state_invalid")
                current_index = 0
            elif entry.prepared:
                if state_index != current_index:
                    raise CanonicalWriterFoundationError("foundation_journal_state_regressed")
                if pending_target not in {None, value["transition_to"]}:
                    raise CanonicalWriterFoundationError("foundation_journal_pending_drifted")
                pending_target = str(value["transition_to"])
            else:
                if pending_target is None or entry.state != pending_target:
                    raise CanonicalWriterFoundationError("foundation_journal_transition_unprepared")
                if state_index != current_index + 1:
                    raise CanonicalWriterFoundationError("foundation_journal_state_skipped")
                current_index = state_index
                pending_target = None
            salt = value["scram_salt_base64"]
            if retained_salt is not None and salt != retained_salt:
                raise CanonicalWriterFoundationError("foundation_journal_salt_drifted")
            if salt is not None:
                retained_salt = str(salt)
            attestation = value["terminal_attestation"]
            if (
                retained_attestation is not None
                and attestation is not None
                and _attestation_static_projection(attestation)
                != _attestation_static_projection(retained_attestation)
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_journal_attestation_drifted"
                )
            if attestation is not None:
                retained_attestation = attestation
            credential = _credential_mapping(value["credential"])
            if previous_credential is not None:
                previous_rank = credential_rank[str(previous_credential["state"])]
                current_rank = credential_rank[str(credential["state"])]
                if current_rank < previous_rank or current_rank > previous_rank + 1:
                    raise CanonicalWriterFoundationError(
                        "foundation_journal_credential_state_invalid"
                    )
                if current_rank == previous_rank:
                    _require_same_credential_identity(
                        previous_credential,
                        credential,
                    )
                elif previous_rank > 0:
                    if (
                        previous_credential["device"] != credential["device"]
                        or previous_credential["inode"] != credential["inode"]
                        or previous_credential["owner_uid"]
                        != credential["owner_uid"]
                        or previous_credential["group_gid"]
                        != credential["group_gid"]
                        or previous_credential["mode"] != credential["mode"]
                        or previous_credential["link_count"] != 1
                        or credential["link_count"] != 1
                    ):
                        raise CanonicalWriterFoundationError(
                            "foundation_journal_credential_identity_invalid"
                        )
            previous_credential = credential
            previous = entry.sha256
        if staging.exists():
            self._ensure_directory(staging, kind="journal")
            for stage_path in sorted(staging.iterdir()):
                if re.fullmatch(
                    r"terminal-[0-9a-f]{24}\.stage",
                    stage_path.name,
                ):
                    continue
                match = re.fullmatch(r"journal-([0-9]{8})\.stage", stage_path.name)
                if match is None:
                    raise CanonicalWriterFoundationError(
                        "foundation_staged_publication_unknown"
                    )
                sequence = int(match.group(1))
                if sequence > len(entries):
                    raise CanonicalWriterFoundationError(
                        "foundation_staged_publication_sequence_invalid"
                    )
                if sequence == len(entries):
                    continue
                raw = self._read_publication_bytes(
                    stage_path,
                    code="foundation_staged_publication_invalid",
                    expected_link_count=(
                        2 if sequence < len(entries) else 1
                    ),
                )
                staged = self._decode_entry_payload(raw, plan=plan)
                committed = entries[sequence]
                if staged.value != committed.value:
                    raise CanonicalWriterFoundationError(
                        "foundation_staged_publication_drifted"
                    )
                self._publish_payload(
                    kind="journal",
                    stage=stage_path,
                    final=root / f"{sequence:08d}-{committed.sha256}.json",
                    payload=raw,
                )
        return entries

    def append(
        self,
        plan: FoundationPlan,
        *,
        state: str,
        transition_phase: str,
        transition_to: str | None,
        database_observation_sha256: str,
        credential: Mapping[str, Any],
        scram_salt_base64: str | None,
        managed_hba_receipt_sha256: str | None,
        terminal_attestation: Mapping[str, Any] | None,
        now_unix: int,
    ) -> FoundationJournalEntry:
        entries = self.load(plan)
        staging = self._staging_root(plan)
        self._ensure_directory(staging, kind="journal")
        staged_entries: dict[int, tuple[Path, FoundationJournalEntry, bytes]] = {}
        for stage_path in sorted(staging.iterdir()):
            if re.fullmatch(
                r"terminal-[0-9a-f]{24}\.stage",
                stage_path.name,
            ):
                continue
            match = re.fullmatch(r"journal-([0-9]{8})\.stage", stage_path.name)
            if match is None:
                raise CanonicalWriterFoundationError(
                    "foundation_staged_publication_unknown"
                )
            raw = self._read_publication_bytes(
                stage_path,
                code="foundation_staged_publication_invalid",
            )
            staged_entry = self._decode_entry_payload(raw, plan=plan)
            sequence = int(match.group(1))
            if staged_entry.value["sequence"] != sequence or sequence in staged_entries:
                raise CanonicalWriterFoundationError(
                    "foundation_staged_publication_drifted"
                )
            staged_entries[sequence] = (stage_path, staged_entry, raw)

        for sequence in sorted(tuple(staged_entries)):
            if sequence >= len(entries):
                continue
            stage_path, staged_entry, raw = staged_entries.pop(sequence)
            committed = entries[sequence]
            if committed.value != staged_entry.value:
                raise CanonicalWriterFoundationError(
                    "foundation_staged_publication_drifted"
                )
            self._publish_payload(
                kind="journal",
                stage=stage_path,
                final=self._journal_root(plan)
                / f"{sequence:08d}-{committed.sha256}.json",
                payload=raw,
            )
        if any(sequence != len(entries) for sequence in staged_entries):
            raise CanonicalWriterFoundationError(
                "foundation_staged_publication_sequence_invalid"
            )
        recovered = staged_entries.get(len(entries))
        previous = entries[-1].sha256 if entries else None
        unsigned = {
            "schema": FOUNDATION_JOURNAL_SCHEMA,
            "release_revision": plan.revision,
            "plan_sha256": plan.sha256,
            "artifact_set_sha256": plan.value["artifact_set_sha256"],
            "sequence": len(entries),
            "state": state,
            "transition_phase": transition_phase,
            "transition_to": transition_to,
            "previous_entry_sha256": previous,
            "database_observation_sha256": _digest(
                database_observation_sha256,
                "foundation_database_observation_digest_invalid",
            ),
            "credential": dict(_credential_mapping(credential)),
            "scram_salt_base64": scram_salt_base64,
            "managed_hba_receipt_sha256": managed_hba_receipt_sha256,
            "terminal_attestation": (
                None
                if terminal_attestation is None
                else dict(_validate_terminal_attestation(terminal_attestation))
            ),
            "recorded_at_unix": (
                recovered[1].value["recorded_at_unix"]
                if recovered is not None
                else now_unix
            ),
            "secret_material_recorded": False,
        }
        entry = FoundationJournalEntry.from_mapping(
            {**unsigned, "entry_sha256": _sha256_json(unsigned)},
            plan=plan,
        )
        if recovered is not None and recovered[1].value != entry.value:
            raise CanonicalWriterFoundationError(
                "foundation_staged_publication_drifted"
            )
        root = self._journal_root(plan)
        self._ensure_directory(root, kind="journal")
        path = root / f"{len(entries):08d}-{entry.sha256}.json"
        payload = _canonical_bytes(entry.value) + b"\n"
        stage = (
            recovered[0]
            if recovered is not None
            else staging / f"journal-{len(entries):08d}.stage"
        )
        self._publish_payload(
            kind="journal",
            stage=stage,
            final=path,
            payload=payload,
        )
        loaded = self._read_entry_file(path, plan=plan)
        if loaded.value != entry.value:
            raise CanonicalWriterFoundationError("foundation_journal_readback_drifted")
        return loaded

    def _decode_terminal_receipt_payload(
        self,
        raw: bytes,
        *,
        plan: FoundationPlan,
    ) -> Mapping[str, Any]:
        if not raw.endswith(b"\n"):
            raise CanonicalWriterFoundationError(
                "foundation_terminal_receipt_invalid"
            )
        try:
            decoded = raw[:-1].decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise CanonicalWriterFoundationError(
                "foundation_terminal_receipt_invalid"
            ) from exc
        value = _strict_json(decoded, "foundation_terminal_receipt_invalid")
        unsigned = {
            key: item for key, item in value.items() if key != "receipt_sha256"
        }
        credential = value.get("credential")
        if (
            set(value) != _FOUNDATION_RECEIPT_FIELDS
            or value.get("schema") != FOUNDATION_RECEIPT_SCHEMA
            or value.get("ok") is not True
            or value.get("mode") != "adopted_existing_terminal"
            or plan.mode != "adopted_existing_terminal"
            or value.get("release_revision") != plan.revision
            or value.get("plan_sha256") != plan.sha256
            or value.get("artifact_set_sha256")
            != plan.value["artifact_set_sha256"]
            or value.get("catalog_sha256") != PRODUCTION_CATALOG_SHA256
            or value.get("writer_ping_ok") is not True
            or value.get("secret_material_recorded") is not False
            or value.get("retirement_covered") is not False
            or not isinstance(credential, Mapping)
            or _credential_mapping(credential)["state"] != "installed"
            or any(
                _SHA256_RE.fullmatch(str(value.get(field))) is None
                for field in (
                    "terminal_entry_sha256",
                    "database_observation_sha256",
                    "privilege_attestation_sha256",
                    "private_schema_identity_sha256",
                    "managed_hba_receipt_sha256",
                )
            )
            or any(
                type(value.get(field)) is not int
                for field in (
                    "terminal_at_unix",
                    "verified_at_unix",
                    "hba_observed_at_unix",
                    "hba_expires_at_unix",
                )
            )
            or value.get("receipt_sha256") != _sha256_json(unsigned)
            or raw != _canonical_bytes(value) + b"\n"
        ):
            raise CanonicalWriterFoundationError(
                "foundation_terminal_receipt_invalid"
            )
        return json.loads(_canonical_bytes(value).decode("utf-8"))

    def _recover_terminal_staging(
        self,
        plan: FoundationPlan,
        *,
        terminal_root: Path,
    ) -> None:
        staging_root = self._staging_root(plan)
        if not staging_root.exists():
            return
        self._ensure_directory(staging_root, kind="terminal")
        for stage in sorted(staging_root.glob("terminal-*.stage")):
            if re.fullmatch(r"terminal-[0-9a-f]{24}\.stage", stage.name) is None:
                raise CanonicalWriterFoundationError(
                    "foundation_staged_publication_unknown"
                )
            link_count = stage.lstat().st_nlink
            if link_count not in {1, 2}:
                raise CanonicalWriterFoundationError(
                    "foundation_publication_link_state_invalid"
                )
            payload = self._read_publication_bytes(
                stage,
                code="foundation_terminal_receipt_invalid",
                expected_link_count=link_count,
            )
            value = self._decode_terminal_receipt_payload(payload, plan=plan)
            digest = str(value["receipt_sha256"])
            if stage.name != f"terminal-{digest[:24]}.stage":
                raise CanonicalWriterFoundationError(
                    "foundation_staged_publication_drifted"
                )
            self._publish_payload(
                kind="terminal",
                stage=stage,
                final=terminal_root / f"{digest}.json",
                payload=payload,
            )

    def publish_terminal_receipt(
        self,
        plan: FoundationPlan,
        entry: FoundationJournalEntry,
        *,
        observation: FoundationObservation,
        terminal_attestation: Mapping[str, Any],
        hba_receipt: ManagedCloudSQLAdminHBAReceipt,
        now_unix: int,
    ) -> Mapping[str, Any]:
        if plan.mode != "adopted_existing_terminal":
            raise CanonicalWriterFoundationError(
                "foundation_terminal_requires_exact_adoption"
            )
        if entry.state != "terminal" or entry.prepared:
            raise CanonicalWriterFoundationError("foundation_terminal_entry_invalid")
        if not isinstance(observation, FoundationObservation):
            raise TypeError("foundation terminal observation is required")
        _require_state_observation("terminal", observation)
        _require_same_credential_identity(
            entry.value["credential"],
            observation.value["credential"],
        )
        approved_attestation = _validate_terminal_attestation(
            entry.value["terminal_attestation"]
        )
        attestation = _validate_terminal_attestation(terminal_attestation)
        if _attestation_static_projection(attestation) != _attestation_static_projection(
            approved_attestation
        ):
            raise CanonicalWriterFoundationError(
                "foundation_terminal_attestation_changed"
            )
        if (
            not isinstance(hba_receipt, ManagedCloudSQLAdminHBAReceipt)
            or attestation["managed_hba_receipt_sha256"] != hba_receipt.sha256
            or not hba_receipt.is_fresh(now_unix)
            or type(now_unix) is not int
            or now_unix < entry.value["recorded_at_unix"]
        ):
            raise CanonicalWriterFoundationError("foundation_terminal_time_invalid")
        unsigned = {
            "schema": FOUNDATION_RECEIPT_SCHEMA,
            "ok": True,
            "mode": "adopted_existing_terminal",
            "release_revision": plan.revision,
            "plan_sha256": plan.sha256,
            "artifact_set_sha256": plan.value["artifact_set_sha256"],
            "terminal_entry_sha256": entry.sha256,
            "database_observation_sha256": observation.sha256,
            "credential": dict(
                _credential_mapping(observation.value["credential"])
            ),
            "catalog_sha256": PRODUCTION_CATALOG_SHA256,
            "privilege_attestation_sha256": attestation[
                "privilege_attestation_sha256"
            ],
            "private_schema_identity_sha256": attestation[
                "private_schema_identity_sha256"
            ],
            "managed_hba_receipt_sha256": attestation[
                "managed_hba_receipt_sha256"
            ],
            "writer_ping_ok": True,
            "secret_material_recorded": False,
            "retirement_covered": False,
            "terminal_at_unix": entry.value["recorded_at_unix"],
            "verified_at_unix": now_unix,
            "hba_observed_at_unix": hba_receipt.observed_at_unix,
            "hba_expires_at_unix": hba_receipt.expires_at_unix,
        }
        receipt = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
        root = self._terminal_root(plan)
        self._ensure_directory(root, kind="terminal")
        self._recover_terminal_staging(plan, terminal_root=root)
        path = root / f"{receipt['receipt_sha256']}.json"
        payload = _canonical_bytes(receipt) + b"\n"
        stage = self._staging_root(plan) / (
            f"terminal-{receipt['receipt_sha256'][:24]}.stage"
        )
        for existing in tuple(root.iterdir()):
            if re.fullmatch(r"[0-9a-f]{64}\.json", existing.name) is None:
                raise CanonicalWriterFoundationError(
                    "foundation_terminal_receipt_residue"
                )
            existing_payload = self._read_publication_bytes(
                existing,
                code="foundation_terminal_receipt_invalid",
                expected_link_count=(
                    2
                    if existing == path and os.path.lexists(stage)
                    else 1
                ),
            )
            existing_value = self._decode_terminal_receipt_payload(
                existing_payload,
                plan=plan,
            )
            if (
                existing.name
                != f"{existing_value.get('receipt_sha256')}.json"
            ):
                raise CanonicalWriterFoundationError(
                    "foundation_terminal_receipt_invalid"
                )
        self._publish_payload(
            kind="terminal",
            stage=stage,
            final=path,
            payload=payload,
        )
        if self._read_publication_bytes(
            path,
            code="foundation_terminal_receipt_invalid",
        ) != payload:
            raise CanonicalWriterFoundationError("foundation_terminal_receipt_drifted")
        return receipt


def _fixed_writer_config() -> WriterDBConfig:
    return WriterDBConfig(
        host=SQL_HOST,
        tls_server_name=SQL_TLS_SERVER_NAME,
        port=SQL_PORT,
        database=SQL_DATABASE,
        user=SQL_USER,
        ca_file=DATABASE_CA_PATH,
        credential=CredentialSource(
            path=DATABASE_CREDENTIAL_PATH,
            expected_uid=WRITER_UID,
            expected_gid=WRITER_GID,
            allowed_modes=frozenset({0o400}),
        ),
        connect_timeout_seconds=5.0,
        io_timeout_seconds=10.0,
        application_name="muncho-writer-foundation-attestation",
    )


def _sql_text_literal(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 4096
        or "\x00" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise CanonicalWriterFoundationError("foundation_setting_value_invalid")
    return "'" + value.replace("'", "''") + "'"


_ALLOWED_SESSION_SETTINGS = frozenset(
    {
        "muncho.canonical_writer_reconcile_scope",
        "muncho.canonical_writer_reconcile_database",
        "muncho.canonical_writer_reconcile_server_identity_sha256",
        "muncho.canonical_writer_reconcile_source_owner",
        "muncho.canonical_writer_reconcile_expected_row_count",
        "muncho.canonical_writer_reconcile_expected_canonical14_sha256",
        "muncho.canonical_writer_reconcile_expected_extended19_sha256",
        "muncho.canonical_writer_reconcile_expected_occurred_at_cutoff",
        "muncho.canonical_writer_reconcile_approval_receipt_sha256",
        "muncho.canonical_writer_foundation_legacy_source_owner",
        "muncho.canonical_writer_migration_scope",
        "muncho.canonical_writer_migration_database",
        "muncho.canonical_writer_migration_approval_receipt_sha256",
        "muncho.canonical_writer_cloudsqladmin_hba_rejection_sha256",
    }
)


def _require_command(result: Any, prefix: str, code: str) -> None:
    if (
        not str(getattr(result, "command_tag", "")).upper().startswith(prefix)
        or tuple(getattr(result, "rows", ()))
    ):
        raise CanonicalWriterFoundationError(code)


@contextlib.contextmanager
def _temporary_session_settings(
    session: AdminSession,
    settings: Mapping[str, str],
) -> Iterator[None]:
    if not settings or set(settings) - _ALLOWED_SESSION_SETTINGS:
        raise CanonicalWriterFoundationError("foundation_setting_name_invalid")
    set_sql = "\n".join(
        f"SET {name} = {_sql_text_literal(value)};"
        for name, value in sorted(settings.items())
    )
    reset_sql = "ROLLBACK;\n" + "\n".join(
        f"RESET {name};" for name in sorted(settings)
    )
    try:
        result = session.query(set_sql, maximum_rows=0)
        _require_command(result, "SET", "foundation_setting_install_unconfirmed")
        yield
    except BaseException as primary:
        try:
            session.query(reset_sql, maximum_rows=0)
        except BaseException as cleanup:
            raise ExceptionGroup(
                "foundation session setting cleanup blocked",
                [primary, cleanup],
            ) from None
        raise
    else:
        try:
            result = session.query(reset_sql, maximum_rows=0)
            if not str(getattr(result, "command_tag", "")).upper().startswith("RESET"):
                raise CanonicalWriterFoundationError(
                    "foundation_setting_cleanup_unconfirmed"
                )
        except BaseException as exc:
            raise CanonicalWriterFoundationError(
                "foundation_setting_cleanup_failed"
            ) from exc


def _execute_artifact(
    session: AdminSession,
    artifact: SealedSQLArtifact,
    *,
    settings: Mapping[str, str] | None = None,
) -> None:
    try:
        sql = artifact.payload.decode("utf-8", errors="strict")
        if settings:
            with _temporary_session_settings(session, settings):
                result = session.query(sql, maximum_rows=0)
        else:
            result = session.query(sql, maximum_rows=0)
        _require_command(result, "COMMIT", "foundation_sql_commit_unconfirmed")
    except CanonicalWriterFoundationError:
        raise
    except BaseException as exc:
        raise CanonicalWriterFoundationError("foundation_sql_execution_failed") from exc


_PASSWORD_COPY_RESULT_FIELDS = frozenset(
    {
        "boundary",
        "database",
        "role",
        "password_encryption",
        "login_enabled",
        "secret_transport",
        "copy_data_completed",
        "temporary_security_definer_removed",
        "temporary_admin_delete_required",
        "secret_material_recorded",
    }
)


def _execute_writer_password_copy(
    session: AdminSession,
    password: bytearray,
) -> Mapping[str, Any]:
    """Invoke the dedicated CopyData boundary, never generic SQL."""

    if (
        not isinstance(password, bytearray)
        or len(password) != 64
        or _WRITER_SECRET_RE.fullmatch(password) is None
    ):
        raise CanonicalWriterFoundationError(
            "foundation_password_copy_secret_invalid"
        )
    executor = getattr(session, "execute_password_copy", None)
    if not callable(executor):
        raise CanonicalWriterFoundationError(
            "foundation_password_copy_boundary_unavailable"
        )
    try:
        result = executor(WRITER_PASSWORD_BOUNDARY, password=password)
    except CanonicalWriterFoundationError:
        raise
    except BaseException as exc:
        raise CanonicalWriterFoundationError(
            "foundation_password_copy_failed"
        ) from exc
    if not isinstance(result, Mapping) or set(result) != _PASSWORD_COPY_RESULT_FIELDS:
        raise CanonicalWriterFoundationError(
            "foundation_password_copy_receipt_invalid"
        )
    value = dict(result)
    if value != {
        "boundary": WRITER_PASSWORD_BOUNDARY.name,
        "database": SQL_DATABASE,
        "role": SQL_USER,
        "password_encryption": "scram-sha-256",
        "login_enabled": True,
        "secret_transport": "postgres-copy-data-v1",
        "copy_data_completed": True,
        "temporary_security_definer_removed": True,
        "temporary_admin_delete_required": True,
        "secret_material_recorded": False,
    }:
        raise CanonicalWriterFoundationError(
            "foundation_password_copy_receipt_invalid"
        )
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _default_secret_factory() -> bytes:
    return base64.urlsafe_b64encode(secrets.token_bytes(48))


def _default_hba_collector() -> ManagedCloudSQLAdminHBAReceipt:
    return collect_managed_cloudsqladmin_hba_receipt(_fixed_writer_config())


def _validate_hba_receipt(
    receipt: Any,
    *,
    admin_session: AdminSession,
    now_unix: int,
) -> ManagedCloudSQLAdminHBAReceipt:
    return _validate_hba_receipt_for_peer(
        receipt,
        tls_peer_certificate_sha256=admin_session.tls_peer_certificate_sha256,
        now_unix=now_unix,
    )


def _validate_hba_receipt_for_peer(
    receipt: Any,
    *,
    tls_peer_certificate_sha256: str,
    now_unix: int,
) -> ManagedCloudSQLAdminHBAReceipt:
    if not isinstance(receipt, ManagedCloudSQLAdminHBAReceipt):
        raise CanonicalWriterFoundationError("foundation_hba_receipt_type_invalid")
    if (
        receipt.host != SQL_HOST
        or receipt.tls_server_name != SQL_TLS_SERVER_NAME
        or receipt.port != SQL_PORT
        or receipt.user != SQL_USER
        or receipt.database != "cloudsqladmin"
        or receipt.server_certificate_sha256
        != tls_peer_certificate_sha256
        or receipt.tls_peer_verified is not True
        or not receipt.is_fresh(now_unix)
    ):
        raise CanonicalWriterFoundationError("foundation_hba_receipt_binding_invalid")
    return receipt


def _default_authentication_probe(enabled: bool) -> Mapping[str, Any]:
    session = None
    try:
        session = _open_postgres_session(_fixed_writer_config())
        result = session.query(
            "SELECT CURRENT_USER::text AS writer_identity",
            maximum_rows=1,
        )
        if not enabled:
            raise CanonicalWriterFoundationError(
                "foundation_retired_writer_still_authenticates"
            )
        if (
            tuple(result.columns) != ("writer_identity",)
            or tuple(result.rows) != ((SQL_USER,),)
            or not result.command_tag.upper().startswith("SELECT")
        ):
            raise CanonicalWriterFoundationError(
                "foundation_writer_authentication_identity_invalid"
            )
        return {"authenticated": True, "user": SQL_USER}
    except PostgresServerError as exc:
        if enabled or exc.sqlstate not in {"28000", "28P01"}:
            raise CanonicalWriterFoundationError(
                "foundation_writer_authentication_failed"
            ) from exc
        return {"authenticated": False, "user": SQL_USER, "sqlstate": exc.sqlstate}
    finally:
        if session is not None:
            session.close()


def _default_terminal_attestor(
    hba_receipt: ManagedCloudSQLAdminHBAReceipt,
) -> Mapping[str, Any]:
    config = _fixed_writer_config()
    policy, _seed_attestation, hba = _collect_live_policy(
        config,
        hba_collector=lambda _config: hba_receipt,
    )
    if hba.sha256 != hba_receipt.sha256:
        raise CanonicalWriterFoundationError("foundation_hba_receipt_drifted")
    database = CanonicalWriterDB(
        config=config,
        privilege_policy=policy,
        statements=PRODUCTION_STATEMENT_CATALOG,
        _managed_hba_probe=lambda _config: hba_receipt,
    )
    if database.statement_catalog_sha256 != PRODUCTION_CATALOG_SHA256:
        raise CanonicalWriterFoundationError("foundation_catalog_digest_drifted")
    attestation = database.startup_attest()
    backend = PostgresCanonicalWriterBackend(database)
    ping = backend.ping(
        RuntimeContext(
            request_id="canonical-writer-foundation-attestation",
            platform="api_server",
            service_internal=True,
        )
    )
    private_identity = attestation.canonical_private_schema_identity
    if private_identity is None:
        raise CanonicalWriterFoundationError("foundation_private_schema_identity_missing")
    return _validate_terminal_attestation(
        {
            "catalog_sha256": database.statement_catalog_sha256,
            "privilege_attestation_sha256": _sha256_json(
                _attestation_projection(attestation)
            ),
            "private_schema_identity_sha256": private_identity.sha256,
            "managed_hba_receipt_sha256": hba.sha256,
            "writer_ping_ok": True,
            "writer_ping_service": ping.get("service"),
            "writer_ping_protocol": ping.get("protocol"),
            "table_grants": [],
            "general_sql_available": False,
            "cross_database_connect_available": False,
            "dangerous_role_attributes": [],
            "migration_admin_membership_present": False,
        }
    )


def _require_authentication_success(value: Any) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"authenticated", "user"}
        or value["authenticated"] is not True
        or value["user"] != SQL_USER
    ):
        raise CanonicalWriterFoundationError(
            "foundation_writer_authentication_unconfirmed"
        )
    return {"authenticated": True, "user": SQL_USER}


def _attestation_static_projection(value: Any) -> Mapping[str, Any]:
    attestation = dict(_validate_terminal_attestation(value))
    attestation.pop("managed_hba_receipt_sha256")
    return attestation


def _collect_terminal_proof(
    *,
    admin_session: AdminSession,
    hba_collector: HBACollector,
    authentication_probe: AuthenticationProbe,
    terminal_attestor: TerminalAttestor,
    now_unix: int,
    approved_attestation: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], ManagedCloudSQLAdminHBAReceipt]:
    _require_authentication_success(authentication_probe(True))
    hba = _validate_hba_receipt(
        hba_collector(),
        admin_session=admin_session,
        now_unix=now_unix,
    )
    current = _validate_terminal_attestation(terminal_attestor(hba))
    if current["managed_hba_receipt_sha256"] != hba.sha256:
        raise CanonicalWriterFoundationError(
            "foundation_terminal_hba_proof_drifted"
        )
    if approved_attestation is not None and _attestation_static_projection(
        current
    ) != _attestation_static_projection(approved_attestation):
        raise CanonicalWriterFoundationError(
            "foundation_terminal_attestation_changed"
        )
    return current, hba


def build_foundation_adoption_plan(
    release_revision: str,
    observation: FoundationObservation,
    *,
    _artifacts: Mapping[str, SealedSQLArtifact] | None = None,
    _hba_collector: HBACollector = _default_hba_collector,
    _authentication_probe: AuthenticationProbe = _default_authentication_probe,
    _terminal_attestor: TerminalAttestor = _default_terminal_attestor,
    _clock: Callable[[], float] = time.time,
) -> FoundationPlan:
    """Plan read-only adoption of an already exact terminal foundation."""

    if not isinstance(observation, FoundationObservation):
        raise TypeError("foundation observation is required")
    revision = _revision(release_revision)
    credential = _credential_mapping(observation.value["credential"])
    if (
        observation.value["session_user"] != SQL_USER
        or not observation.membership_ready
        or credential["state"] != "installed"
        or observation.value["legacy_truth"] is not None
    ):
        raise CanonicalWriterFoundationError(
            "foundation_existing_authority_not_exact_terminal"
        )
    _require_authentication_success(_authentication_probe(True))
    hba = _validate_hba_receipt_for_peer(
        _hba_collector(),
        tls_peer_certificate_sha256=str(
            observation.value["tls_peer_certificate_sha256"]
        ),
        now_unix=int(_clock()),
    )
    attestation = _validate_terminal_attestation(_terminal_attestor(hba))
    if attestation["managed_hba_receipt_sha256"] != hba.sha256:
        raise CanonicalWriterFoundationError(
            "foundation_terminal_hba_proof_drifted"
        )
    artifacts = (
        _load_sealed_artifacts(revision) if _artifacts is None else _artifacts
    )
    artifact_sha = _artifact_digest_mapping(artifacts)
    unsigned = {
        "schema": FOUNDATION_PLAN_SCHEMA,
        "mode": "adopted_existing_terminal",
        "release_revision": revision,
        "target": dict(_fixed_target()),
        "initial_observation": observation.to_mapping(),
        "initial_observation_sha256": observation.sha256,
        "artifact_sha256": artifact_sha,
        "artifact_set_sha256": _sha256_json(artifact_sha),
        "states": list(FOUNDATION_ADOPTION_STATES),
        "legacy_reconciliation_required": False,
        "legacy_source_owner": None,
        "credential_contract": {
            "generated_on_vm": True,
            "bytes": 64,
            "alphabet": "base64url-no-padding",
            "owner_uid": WRITER_UID,
            "group_gid": WRITER_GID,
            "mode": "0400",
            "scram_mechanism": "SCRAM-SHA-256",
            "password_transport": "postgres-copy-data-v1",
            "server_generated_scram_salt": True,
            "copy_data_only": True,
            "statement_logging_safe_without_privileged_guc": True,
            "temporary_security_definer": True,
            "password_or_verifier_serialized": False,
            "content_or_digest_recorded": False,
        },
        "terminal_attestation_catalog_sha256": PRODUCTION_CATALOG_SHA256,
        "adoption_terminal_attestation": dict(attestation),
    }
    return FoundationPlan.from_mapping(
        {**unsigned, "plan_sha256": _sha256_json(unsigned)}
    )


def _legacy_settings(plan: FoundationPlan) -> Mapping[str, str]:
    initial = plan.initial_observation.value
    legacy = initial["legacy_truth"]
    if not isinstance(legacy, Mapping):
        raise CanonicalWriterFoundationError("foundation_legacy_plan_missing")
    return {
        "muncho.canonical_writer_reconcile_scope": "isolated_canary_copy",
        "muncho.canonical_writer_reconcile_database": SQL_DATABASE,
        "muncho.canonical_writer_reconcile_server_identity_sha256": (
            plan.initial_observation.sha256
        ),
        "muncho.canonical_writer_reconcile_source_owner": str(
            legacy["source_owner"]
        ),
        "muncho.canonical_writer_reconcile_expected_row_count": str(
            legacy["source_row_count"]
        ),
        "muncho.canonical_writer_reconcile_expected_canonical14_sha256": str(
            legacy["canonical14_sha256"]
        ),
        "muncho.canonical_writer_reconcile_expected_extended19_sha256": str(
            legacy["extended19_sha256"]
        ),
        "muncho.canonical_writer_reconcile_expected_occurred_at_cutoff": str(
            legacy["occurred_at_cutoff"]
        ),
        "muncho.canonical_writer_reconcile_approval_receipt_sha256": plan.sha256,
        "muncho.canonical_writer_foundation_legacy_source_owner": str(
            legacy["source_owner"]
        ),
    }


def _base_settings(
    plan: FoundationPlan,
    receipt: ManagedCloudSQLAdminHBAReceipt,
) -> Mapping[str, str]:
    return {
        "muncho.canonical_writer_migration_scope": "isolated_canary_copy",
        "muncho.canonical_writer_migration_database": SQL_DATABASE,
        "muncho.canonical_writer_migration_approval_receipt_sha256": plan.sha256,
        "muncho.canonical_writer_cloudsqladmin_hba_rejection_sha256": receipt.sha256,
    }


def _observe_for_plan(
    plan: FoundationPlan,
    session: AdminSession,
    *,
    secret_store: SecretStore,
    artifacts: Mapping[str, SealedSQLArtifact],
    expected_credential: Mapping[str, Any] | None = None,
) -> FoundationObservation:
    observed = observe_foundation(
        plan.revision,
        session,
        _secret_store=secret_store,
        _artifacts=artifacts,
        _plan_sha256=plan.sha256,
    )
    if observed.value["tls_peer_certificate_sha256"] != plan.initial_observation.value[
        "tls_peer_certificate_sha256"
    ]:
        raise CanonicalWriterFoundationError("foundation_tls_peer_changed_after_approval")
    if observed.value["session_user"] != plan.initial_observation.value["session_user"]:
        raise CanonicalWriterFoundationError(
            "foundation_observation_session_changed_after_approval"
        )
    if observed.value["legacy_archive_present"] is not plan.initial_observation.value[
        "legacy_archive_present"
    ]:
        raise CanonicalWriterFoundationError(
            "foundation_legacy_archive_changed_after_approval"
        )
    if observed.value["legacy_archive_identity"] != plan.initial_observation.value[
        "legacy_archive_identity"
    ]:
        raise CanonicalWriterFoundationError(
            "foundation_legacy_archive_changed_after_approval"
        )
    if expected_credential is not None:
        _require_same_credential_identity(
            expected_credential,
            observed.value["credential"],
        )
    return observed


def _require_state_observation(
    state: str,
    observation: FoundationObservation,
) -> None:
    credential = _credential_mapping(observation.value["credential"])
    if state == "intent":
        if credential["state"] != "absent":
            raise CanonicalWriterFoundationError("foundation_intent_state_drifted")
        return
    if state == "prerequisites":
        valid = observation.prerequisites_ready and credential["state"] == "absent"
    elif state == "truth_reconciled":
        valid = observation.truth_ready and credential["state"] == "absent"
    elif state == "secret_staged":
        valid = observation.truth_ready and credential["state"] == "installed"
    elif state == "login_enabled":
        valid = (
            observation.login_ready
            and observation.membership_pairs
            == frozenset({(CANARY_BOOTSTRAP_ROLE, CANARY_BOOTSTRAP_LOGIN)})
            and credential["state"] == "installed"
        )
    elif state == "base_migration":
        valid = (
            observation.base_ready
            and observation.membership_pairs
            == frozenset({(CANARY_BOOTSTRAP_ROLE, CANARY_BOOTSTRAP_LOGIN)})
            and credential["state"] == "installed"
        )
    elif state in {
        "writer_membership",
        "attested",
        "adopted_existing_terminal",
        "terminal",
    }:
        valid = observation.membership_ready and credential["state"] == "installed"
    else:
        raise CanonicalWriterFoundationError("foundation_state_invalid")
    if not valid:
        raise CanonicalWriterFoundationError("foundation_state_postcondition_failed")


def _next_state(plan: FoundationPlan, state: str) -> str:
    states = plan.states
    index = states.index(state)
    if index + 1 >= len(states):
        raise CanonicalWriterFoundationError("foundation_state_terminal")
    return states[index + 1]


FaultInjector = Callable[[str, str], None]
AuthenticationProbe = Callable[[bool], Mapping[str, Any]]
TerminalAttestor = Callable[[ManagedCloudSQLAdminHBAReceipt], Mapping[str, Any]]
HBACollector = Callable[[], ManagedCloudSQLAdminHBAReceipt]


def _noop_fault_injector(_state: str, _point: str) -> None:
    return None


def _append_journal_entry(
    journal: _AppendOnlyFoundationJournal,
    plan: FoundationPlan,
    *,
    state: str,
    phase: str,
    target: str | None,
    observation: FoundationObservation,
    credential: Mapping[str, Any] | None = None,
    salt: str | None,
    hba_digest: str | None,
    attestation: Mapping[str, Any] | None,
    clock: Callable[[], float],
) -> FoundationJournalEntry:
    return journal.append(
        plan,
        state=state,
        transition_phase=phase,
        transition_to=target,
        database_observation_sha256=observation.sha256,
        credential=(
            observation.value["credential"]
            if credential is None
            else credential
        ),
        scram_salt_base64=salt,
        managed_hba_receipt_sha256=hba_digest,
        terminal_attestation=attestation,
        now_unix=int(clock()),
    )


def _prove_adopted_terminal(
    plan: FoundationPlan,
    observation: FoundationObservation,
    *,
    admin_session: AdminSession,
    hba_collector: HBACollector,
    authentication_probe: AuthenticationProbe,
    terminal_attestor: TerminalAttestor,
    now_unix: int,
) -> tuple[Mapping[str, Any], ManagedCloudSQLAdminHBAReceipt]:
    _require_state_observation("adopted_existing_terminal", observation)
    approved = plan.value["adoption_terminal_attestation"]
    try:
        return _collect_terminal_proof(
            admin_session=admin_session,
            hba_collector=hba_collector,
            authentication_probe=authentication_probe,
            terminal_attestor=terminal_attestor,
            now_unix=now_unix,
            approved_attestation=approved,
        )
    except CanonicalWriterFoundationError as exc:
        if exc.code == "foundation_terminal_attestation_changed":
            raise CanonicalWriterFoundationError(
                "foundation_adoption_attestation_changed"
            ) from exc
        raise


def apply_approved_foundation(
    plan: FoundationPlan,
    *,
    approved_plan_sha256: str,
    admin_session: AdminSession,
    _journal: _AppendOnlyFoundationJournal | None = None,
    _secret_store: SecretStore | None = None,
    _artifacts: Mapping[str, SealedSQLArtifact] | None = None,
    _hba_collector: HBACollector = _default_hba_collector,
    _authentication_probe: AuthenticationProbe = _default_authentication_probe,
    _terminal_attestor: TerminalAttestor = _default_terminal_attestor,
    _secret_factory: Callable[[], bytes] = _default_secret_factory,
    _clock: Callable[[], float] = time.time,
    _fault_injector: FaultInjector = _noop_fault_injector,
) -> Mapping[str, Any]:
    """Apply or resume only the exact owner-approved persistent foundation."""

    if not isinstance(plan, FoundationPlan):
        raise TypeError("foundation plan is required")
    if plan.mode != "adopted_existing_terminal":
        raise CanonicalWriterFoundationError(
            "foundation_creation_requires_admin_delete_integration"
        )
    approved = _digest(
        approved_plan_sha256,
        "foundation_approved_plan_digest_invalid",
    )
    if not hmac.compare_digest(plan.sha256, approved):
        raise CanonicalWriterFoundationError("foundation_plan_not_approved")
    if (
        getattr(admin_session, "username", None) != SQL_USER
        or plan.initial_observation.value["session_user"] != SQL_USER
    ):
        raise CanonicalWriterFoundationError(
            "foundation_adoption_writer_session_required"
        )
    artifacts = (
        _load_sealed_artifacts(plan.revision) if _artifacts is None else _artifacts
    )
    _validate_plan_artifacts(plan, artifacts)
    journal = _AppendOnlyFoundationJournal() if _journal is None else _journal
    secret_store = PersistentWriterSecretStore() if _secret_store is None else _secret_store

    with journal.lock():
        journal.assert_no_cross_plan(plan)
        entries = journal.load(plan)
        live = _observe_for_plan(
            plan,
            admin_session,
            secret_store=secret_store,
            artifacts=artifacts,
            expected_credential=(
                entries[-1].value["credential"]
                if entries
                else plan.initial_observation.value["credential"]
            ),
        )
        adoption_attestation: Mapping[str, Any] | None = None
        adoption_hba: ManagedCloudSQLAdminHBAReceipt | None = None
        if plan.mode == "adopted_existing_terminal":
            adoption_attestation, adoption_hba = _prove_adopted_terminal(
                plan,
                live,
                admin_session=admin_session,
                hba_collector=_hba_collector,
                authentication_probe=_authentication_probe,
                terminal_attestor=_terminal_attestor,
                now_unix=int(_clock()),
            )
        if not entries:
            if live.value != plan.initial_observation.value:
                raise CanonicalWriterFoundationError(
                    "foundation_initial_observation_changed"
                )
            initial_state = plan.states[0]
            _append_journal_entry(
                journal,
                plan,
                state=initial_state,
                phase="complete",
                target=None,
                observation=live,
                salt=None,
                hba_digest=(
                    None if adoption_hba is None else adoption_hba.sha256
                ),
                attestation=adoption_attestation,
                clock=_clock,
            )
            _fault_injector(initial_state, "after_transition")

        while True:
            entries = journal.load(plan)
            if not entries:
                raise CanonicalWriterFoundationError("foundation_journal_missing")
            last = entries[-1]
            if last.state == "terminal" and not last.prepared:
                final_observation = _observe_for_plan(
                    plan,
                    admin_session,
                    secret_store=secret_store,
                    artifacts=artifacts,
                    expected_credential=last.value["credential"],
                )
                _require_state_observation("terminal", final_observation)
                terminal_attestation, terminal_hba = _collect_terminal_proof(
                    admin_session=admin_session,
                    hba_collector=_hba_collector,
                    authentication_probe=_authentication_probe,
                    terminal_attestor=_terminal_attestor,
                    now_unix=int(_clock()),
                    approved_attestation=last.value["terminal_attestation"],
                )
                return journal.publish_terminal_receipt(
                    plan,
                    last,
                    observation=final_observation,
                    terminal_attestation=terminal_attestation,
                    hba_receipt=terminal_hba,
                    now_unix=int(_clock()),
                )

            current_state = last.state
            target_state = (
                str(last.value["transition_to"])
                if last.prepared
                else _next_state(plan, current_state)
            )
            live = _observe_for_plan(
                plan,
                admin_session,
                secret_store=secret_store,
                artifacts=artifacts,
                expected_credential=last.value["credential"],
            )
            if not last.prepared:
                _require_state_observation(current_state, live)

            salt_b64 = None
            hba_digest = last.value["managed_hba_receipt_sha256"]
            attestation = last.value["terminal_attestation"]
            terminal_hba: ManagedCloudSQLAdminHBAReceipt | None = None

            if not last.prepared:
                last = _append_journal_entry(
                    journal,
                    plan,
                    state=current_state,
                    phase="prepared",
                    target=target_state,
                    observation=live,
                    salt=salt_b64,
                    hba_digest=hba_digest,
                    attestation=attestation,
                    clock=_clock,
                )
                _fault_injector(target_state, "after_prepare")

            if target_state == "prerequisites":
                if live.value["credential"]["state"] != "absent":
                    raise CanonicalWriterFoundationError(
                        "foundation_prerequisite_secret_residue"
                    )
                _execute_artifact(admin_session, artifacts["prerequisites"])

            elif target_state == "truth_reconciled":
                if plan.value["legacy_reconciliation_required"]:
                    _execute_artifact(
                        admin_session,
                        artifacts["legacy_reconcile"],
                        settings=_legacy_settings(plan),
                    )

            elif target_state == "secret_staged":
                expected = last.value["credential"]
                if expected["state"] == "absent":
                    allocated = secret_store.allocate(plan.sha256)
                    _fault_injector(
                        target_state,
                        "after_secret_allocation_before_journal",
                    )
                    allocation_observation = _observe_for_plan(
                        plan,
                        admin_session,
                        secret_store=secret_store,
                        artifacts=artifacts,
                    )
                    last = _append_journal_entry(
                        journal,
                        plan,
                        state=current_state,
                        phase="prepared",
                        target=target_state,
                        observation=allocation_observation,
                        credential=allocated,
                        salt=salt_b64,
                        hba_digest=hba_digest,
                        attestation=attestation,
                        clock=_clock,
                    )
                    expected = last.value["credential"]
                    _fault_injector(target_state, "after_secret_allocation")
                secret, staged = secret_store.materialize(
                    plan.sha256,
                    expected,
                    _secret_factory,
                )
                _zeroize(secret)
                staged_observation = _observe_for_plan(
                    plan,
                    admin_session,
                    secret_store=secret_store,
                    artifacts=artifacts,
                    expected_credential=staged,
                )
                last = _append_journal_entry(
                    journal,
                    plan,
                    state=current_state,
                    phase="prepared",
                    target=target_state,
                    observation=staged_observation,
                    credential=staged,
                    salt=None,
                    hba_digest=hba_digest,
                    attestation=attestation,
                    clock=_clock,
                )
                _fault_injector(target_state, "after_secret_materialized")
                secret_store.publish(plan.sha256, last.value["credential"])

            elif target_state == "login_enabled":
                if not live.login_ready:
                    _execute_artifact(
                        admin_session,
                        artifacts["login_enable"],
                    )
                    secret = secret_store.read(last.value["credential"])
                    try:
                        _execute_writer_password_copy(
                            admin_session,
                            secret,
                        )
                    finally:
                        _zeroize(secret)
                # The login intentionally has no CONNECT path until the later
                # writer-role membership.  Authentication is proven only by
                # the final, freshly collected terminal attestation; treating
                # this pre-membership state as network-authenticatable would be
                # a contradictory privilege requirement.

            elif target_state == "base_migration":
                receipt = _validate_hba_receipt(
                    _hba_collector(),
                    admin_session=admin_session,
                    now_unix=int(_clock()),
                )
                hba_digest = receipt.sha256
                # A refreshed, exact HBA observation is itself journaled before
                # the migration on every retry; stale evidence is never reused.
                live_before_base = _observe_for_plan(
                    plan,
                    admin_session,
                    secret_store=secret_store,
                    artifacts=artifacts,
                    expected_credential=last.value["credential"],
                )
                last = _append_journal_entry(
                    journal,
                    plan,
                    state=current_state,
                    phase="prepared",
                    target=target_state,
                    observation=live_before_base,
                    salt=salt_b64,
                    hba_digest=hba_digest,
                    attestation=attestation,
                    clock=_clock,
                )
                _fault_injector(target_state, "after_hba_receipt")
                _execute_artifact(
                    admin_session,
                    artifacts["base_migration"],
                    settings=_base_settings(plan, receipt),
                )

            elif target_state == "writer_membership":
                _execute_artifact(admin_session, artifacts["writer_membership"])

            elif target_state == "attested":
                attestation, terminal_hba = _collect_terminal_proof(
                    admin_session=admin_session,
                    hba_collector=_hba_collector,
                    authentication_probe=_authentication_probe,
                    terminal_attestor=_terminal_attestor,
                    now_unix=int(_clock()),
                )
                hba_digest = terminal_hba.sha256

            elif target_state == "terminal":
                if attestation is None:
                    raise CanonicalWriterFoundationError(
                        "foundation_terminal_attestation_missing"
                    )
                attestation, terminal_hba = _collect_terminal_proof(
                    admin_session=admin_session,
                    hba_collector=_hba_collector,
                    authentication_probe=_authentication_probe,
                    terminal_attestor=_terminal_attestor,
                    now_unix=int(_clock()),
                    approved_attestation=attestation,
                )
                hba_digest = terminal_hba.sha256

            else:  # pragma: no cover - state table is validated above.
                raise CanonicalWriterFoundationError("foundation_state_invalid")

            _fault_injector(target_state, "after_mutation")
            post = _observe_for_plan(
                plan,
                admin_session,
                secret_store=secret_store,
                artifacts=artifacts,
                expected_credential=(
                    None
                    if target_state == "secret_staged"
                    else last.value["credential"]
                ),
            )
            _require_state_observation(target_state, post)
            complete = _append_journal_entry(
                journal,
                plan,
                state=target_state,
                phase="complete",
                target=None,
                observation=post,
                salt=salt_b64,
                hba_digest=hba_digest,
                attestation=attestation,
                clock=_clock,
            )
            _fault_injector(target_state, "after_transition")
            if target_state == "terminal":
                if terminal_hba is None:
                    raise CanonicalWriterFoundationError(
                        "foundation_terminal_hba_proof_missing"
                    )
                return journal.publish_terminal_receipt(
                    plan,
                    complete,
                    observation=post,
                    terminal_attestation=attestation,
                    hba_receipt=terminal_hba,
                    now_unix=int(_clock()),
                )
