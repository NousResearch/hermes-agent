#!/usr/bin/env python3
"""Root-only mechanical coordinator for one isolated full Muncho canary.

This module deliberately owns no task semantics.  It binds an exact reviewed
canary input, two fresh owner approvals, one inherited-stdin administrator
credential, fixed PostgreSQL operations, and the existing honest live driver.
It never accepts SQL, a host, a database, a path, or a secret through argv or
the environment, and it never enables a service.
"""

from __future__ import annotations

import copy
import base64
import fcntl
import grp
import hashlib
import json
import os
import pwd
import re
import select
import secrets
import signal
import ssl
import stat
import struct
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway.canonical_full_canary_runtime import (
    DEFAULT_CANARY_BOOTSTRAP_RETIRE_SQL_RELATIVE,
    DEFAULT_CANARY_BOOTSTRAP_SQL_RELATIVE,
    DEFAULT_CANARY_BOOTSTRAP_CREDENTIAL,
    DEFAULT_APPROVAL_PATH,
    DEFAULT_E2E_FIXTURE,
    DEFAULT_EDGE_CONFIG,
    DEFAULT_EDGE_CONFIG_SOURCE,
    DEFAULT_EDGE_TOKEN_PATH,
    DEFAULT_GATEWAY_CONFIG,
    DEFAULT_GATEWAY_CONFIG_SOURCE,
    DEFAULT_HOST_IDENTITY_RECEIPT,
    DEFAULT_PLAN_PATH,
    DEFAULT_STAGED_PLAN_PATH,
    DEFAULT_WRITER_CONFIG,
    DEFAULT_WRITER_CONFIG_SOURCE,
    EDGE_UNIT_NAME,
    GATEWAY_UNIT_NAME,
    WRITER_UNIT_NAME,
    ExactArtifact,
    FullCanaryIdentities,
    FullCanaryLifecycle,
    FullCanaryOwnerApproval,
    FullCanaryPlan,
    PreopenedSessionBootstrapProvisioner,
    _validate_edge_config,
    _validate_gateway_config,
    _validate_writer_config,
    _validate_writer_only_receipt,
    _read_stable_file,
    _validated_e2e_fixture,
    _validated_release_file,
    _MAX_HOST_IDENTITY_RECEIPT_BYTES,
    build_full_canary_plan,
    collect_service_state,
    evaluate_service_states,
    _lifecycle_lock,
    _require_root_linux,
    validate_dedicated_canary_host,
    load_full_canary_plan,
)
from gateway.canonical_full_canary_live_driver import (
    HonestFullCanaryDriver,
    _atomic_stage_writer_config,
    _read_staged_writer_config,
    prepare_session_bound_plan,
    wait_for_fresh_owner_approval,
)
from gateway.canonical_writer_activation import (
    DEFAULT_PLAN_PATH as WRITER_ACTIVATION_PLAN_PATH,
    GATEWAY_GROUP,
    GATEWAY_USER,
    SOCKET_CLIENT_GROUP,
    WRITER_GROUP,
    WRITER_USER,
    ActivationPlan,
    _success_receipt_path,
    load_activation_plan,
)
from gateway.canonical_writer_db import (
    CredentialSource,
    ManagedCloudSQLAdminHBAReceipt,
    managed_cloudsqladmin_hba_receipt_from_mapping,
    PostgresProtocolError,
    WriterDBConfig,
    _PostgresWireSession,
    _authenticate,
    collect_managed_cloudsqladmin_hba_receipt,
    _open_verified_tls_connection,
    _send_startup_message,
)
from gateway.canonical_writer_boundary import (
    current_process_hardening_state,
    harden_current_process_against_dumping,
)
from gateway.canonical_writer_readiness import (
    boot_identity,
    module_file_identity,
    process_start_time_ticks,
)


COORDINATOR_INPUT_SCHEMA = "muncho-full-canary-coordinator-input.v1"
COORDINATOR_INPUT_PUBLICATION_SCHEMA = (
    "muncho-full-canary-coordinator-input-publication.v1"
)
CREDENTIAL_PREPARE_APPROVAL_SCHEMA = "muncho-full-canary-credential-prepare-approval.v1"
DISCORD_TOKEN_INSTALL_APPROVAL_SCHEMA = (
    "muncho-full-canary-discord-token-install-approval.v1"
)
OWNER_LAUNCH_GATE_SCHEMA = "muncho-full-canary-owner-launch-gate.v1"
DISCORD_TOKEN_INSTALL_GATE_SCHEMA = "muncho-full-canary-discord-token-install-gate.v1"
DISCORD_TOKEN_INSTALL_RECEIPT_SCHEMA = (
    "muncho-full-canary-discord-token-install-receipt.v1"
)
DISCORD_TOKEN_INSTALL_JOURNAL_SCHEMA = (
    "muncho-full-canary-discord-token-install-journal.v1"
)
DISCORD_TOKEN_RETIREMENT_RECEIPT_SCHEMA = (
    "muncho-full-canary-discord-token-retirement-receipt.v1"
)
SECRET_GATE_SCHEMA = "muncho-full-canary-coordinator-secret-gate.v1"
OWNER_APPROVAL_REQUEST_SCHEMA = "muncho-full-canary-owner-approval-request.v1"
COORDINATOR_RECEIPT_SCHEMA = "muncho-full-canary-coordinator-receipt.v1"
COORDINATOR_FAILURE_SCHEMA = "muncho-full-canary-coordinator-failure.v1"
LEGACY_RECOVERY_RECEIPT_SCHEMA = "muncho-full-canary-recovery-receipt.v1"
RECOVERY_TAKEOVER_GATE_SCHEMA = "muncho-full-canary-recovery-takeover-gate.v1"
RECOVERY_TAKEOVER_ACK_SCHEMA = "muncho-full-canary-owner-recovery-takeover-ack.v1"
RECOVERY_WORKER_LEASE_SCHEMA = "muncho-full-canary-recovery-worker-lease.v1"
RECOVERY_SECRET_GATE_SCHEMA = "muncho-full-canary-recovery-admin-secret-gate.v1"
RECOVERY_CONCURRENT_LOSER_RECEIPT_SCHEMA = (
    "muncho-full-canary-recovery-worker-claim-lost-receipt.v1"
)
RECOVERY_WORKER_COMPLETION_SCHEMA = "muncho-full-canary-recovery-worker-completion.v1"
RECOVERY_FINALIZE_PENDING_RECEIPT_SCHEMA = (
    "muncho-full-canary-recovery-finalize-pending-receipt.v1"
)
RECOVERY_RECEIPT_SCHEMA = "muncho-full-canary-recovery-receipt.v2"
# Public stage-one names remain concise while the legacy strings stay explicit.
RECOVERY_GATE_SCHEMA = RECOVERY_TAKEOVER_GATE_SCHEMA
RECOVERY_ACK_SCHEMA = RECOVERY_TAKEOVER_ACK_SCHEMA
PREPLAN_STOPPED_REPORT_SCHEMA = "muncho-full-canary-preplan-stopped-report.v1"
DISCORD_RETIREMENT_GATE_SCHEMA = "muncho-full-canary-discord-token-retirement-gate.v1"
DISCORD_RETIREMENT_ACK_SCHEMA = "muncho-full-canary-owner-discord-retirement-ack.v1"

COORDINATOR_INPUT_PATH = Path("/etc/muncho/full-canary/coordinator-input.json")
COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH = Path(
    "/etc/muncho/full-canary/coordinator-input-publication.json"
)
CREDENTIAL_PREPARE_APPROVAL_PATH = Path(
    "/etc/muncho/full-canary/credential-prepare-approval.json"
)
DISCORD_TOKEN_INSTALL_APPROVAL_PATH = Path(
    "/etc/muncho/full-canary/discord-token-install-approval.json"
)
DISCORD_TOKEN_INSTALL_RECEIPT_PATH = Path(
    "/etc/muncho/full-canary/discord-token-install-receipt.json"
)
DISCORD_TOKEN_RETIREMENT_RECEIPT_PATH = Path(
    "/etc/muncho/full-canary/discord-token-retirement-receipt.json"
)
OWNER_APPROVAL_REQUEST_PATH = Path("/etc/muncho/full-canary/approval-request.json")
COORDINATOR_PROCESS_LEASE_PATH = Path(
    "/etc/muncho/full-canary/coordinator-process-lease.json"
)
COORDINATOR_PROCESS_LOCK_PATH = Path("/run/muncho-full-canary/coordinator-process.lock")
PREPLAN_STOPPED_REPORT_PATH = Path(
    "/etc/muncho/full-canary/preplan-stopped-report.json"
)
DISCORD_TOKEN_PATH = DEFAULT_EDGE_TOKEN_PATH
DISCORD_TOKEN_STAGE_PATH = DISCORD_TOKEN_PATH.with_name(".discord-bot-token.installing")
CANARY_BOOTSTRAP_CREDENTIAL_PATH = DEFAULT_CANARY_BOOTSTRAP_CREDENTIAL
CANARY_DATABASE_CA_PATH = Path("/etc/muncho/trust/cloudsql-server-ca.pem")

CANARY_DATABASE_HOST = "10.91.0.3"
CANARY_DATABASE_PORT = 5432
CANARY_DATABASE_NAME = "muncho_canary_brain"
CANARY_BOOTSTRAP_LOGIN = "canonical_brain_canary_bootstrap_login"

ADMIN_FRAME_SCHEMA = "MCA2-u16be-u32be-utf8-eof.v1"
ADMIN_FRAME_MAGIC = b"MCA2"
ADMIN_FRAME_FD = 0
MAX_ADMIN_USERNAME_BYTES = 63
MIN_ADMIN_PASSWORD_BYTES = 24
MAX_ADMIN_PASSWORD_BYTES = 4096
MAX_COORDINATOR_INPUT_BYTES = 8 * 1024 * 1024
MAX_OWNER_APPROVAL_BYTES = 128 * 1024
MAX_RELEASE_MANIFEST_BYTES = 16 * 1024 * 1024
DISCORD_TOKEN_FRAME_SCHEMA = "DCT1-u32be-opaque-eof.v1"
DISCORD_TOKEN_FRAME_MAGIC = b"DCT1"
MIN_DISCORD_TOKEN_BYTES = 24
MAX_DISCORD_TOKEN_BYTES = 4096
FINAL_APPROVAL_FRAME_SCHEMA = "MFA1-u32be-canonical-json-eof.v1"
FINAL_APPROVAL_FRAME_MAGIC = b"MFA1"
MAX_FINAL_APPROVAL_FRAME_BYTES = 128 * 1024
MAX_FIXED_CANARY_ARTIFACT_BYTES = 2 * 1024 * 1024
FINAL_APPROVAL_INSTALL_RECEIPT_SCHEMA = (
    "muncho-full-canary-final-approval-install-receipt.v1"
)
FINAL_APPROVAL_CANCEL_RECEIPT_SCHEMA = (
    "muncho-full-canary-final-approval-cancel-receipt.v2"
)
FINAL_APPROVAL_MAX_WAIT_SECONDS = 240
FINAL_APPROVAL_TRANSMIT_MARGIN_SECONDS = 30
HBA_EXPIRY_SAFETY_MARGIN_SECONDS = 30
RECOVERY_ACK_FRAME_MAGIC = b"MRA1"
RECOVERY_ACK_FRAME_SCHEMA = "MRA1-u32be-canonical-json-no-secret.v1"
RECOVERY_ADMIN_FRAME_MAGIC = b"MRC2"
RECOVERY_ADMIN_FRAME_SCHEMA = "MRC2-gate-sha256-nonce-sha256-u16be-u32be-utf8-eof.v1"
DISCORD_RETIREMENT_ACK_FRAME_MAGIC = b"DRA1"
DISCORD_RETIREMENT_ACK_FRAME_SCHEMA = "DRA1-u32be-canonical-json-eof.v1"
RECOVERY_GATE_MAX_SECONDS = 300
RECOVERY_PROCESS_TERM_SECONDS = 10.0
RECOVERY_PROCESS_KILL_SECONDS = 10.0
RECOVERY_CONCURRENT_OBSERVE_SECONDS = 5.0
RECOVERY_WORKER_LEASE_MAX_SECONDS = 300

_FRAME_HEADER = struct.Struct("!4sHI")
_TOKEN_FRAME_HEADER = struct.Struct("!4sI")
_FINAL_APPROVAL_FRAME_HEADER = struct.Struct("!4sI")
_ACK_FRAME_HEADER = struct.Struct("!4sI")
_RECOVERY_ADMIN_FRAME_HEADER = struct.Struct("!4s32s32sHI")
_RECOVERY_ADMIN_FRAME_TAIL = struct.Struct("!32s32sHI")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_ADMIN_USERNAME_RE = re.compile(r"^muncho_canary_admin_[0-9a-f]{16}$")
_TLS_NAME_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.europe-west3\.sql\.goog$"
)

_BOOTSTRAP_ROLE_PASSWORD_TEMPLATE = """DO $muncho_coordinator$
DECLARE
    password_value text := pg_catalog.convert_from(
        pg_catalog.decode('{password_base64}','base64'),'UTF8'
    );
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
         WHERE rolname = 'canonical_brain_canary_bootstrap'
           AND NOT rolcanlogin AND NOT rolinherit AND NOT rolsuper
           AND NOT rolcreatedb AND NOT rolcreaterole AND NOT rolreplication
           AND NOT rolbypassrls
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
         WHERE rolname = 'canonical_brain_canary_bootstrap_login'
           AND rolcanlogin AND rolinherit AND NOT rolsuper
           AND NOT rolcreatedb AND NOT rolcreaterole AND NOT rolreplication
           AND NOT rolbypassrls
    ) OR (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted_role
            ON granted_role.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member_role
            ON member_role.oid = membership.member
         WHERE granted_role.rolname = 'canonical_brain_canary_bootstrap'
           AND member_role.rolname = 'canonical_brain_canary_bootstrap_login'
           AND NOT membership.admin_option
           AND membership.inherit_option
           AND membership.set_option
    ) <> 1 OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted_role
            ON granted_role.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member_role
            ON member_role.oid = membership.member
         WHERE (
                granted_role.rolname = 'canonical_brain_canary_bootstrap'
                OR member_role.rolname = 'canonical_brain_canary_bootstrap_login'
                OR member_role.rolname = 'canonical_brain_canary_bootstrap'
               )
           AND NOT (
                granted_role.rolname = 'canonical_brain_canary_bootstrap'
                AND member_role.rolname = 'canonical_brain_canary_bootstrap_login'
                AND NOT membership.admin_option
                AND membership.inherit_option
                AND membership.set_option
           )
    ) THEN
        RAISE EXCEPTION 'canonical canary bootstrap role shape is invalid';
    END IF;
    EXECUTE pg_catalog.format(
        'ALTER ROLE canonical_brain_canary_bootstrap_login PASSWORD %L',
        password_value
    );
END
$muncho_coordinator$"""

_BOOTSTRAP_ROLE_DISABLE_SQL = """DO $muncho_coordinator$
DECLARE
    admin_name text := SESSION_USER;
BEGIN
    IF admin_name !~ '^muncho_canary_admin_[0-9a-f]{16}$' THEN
        RAISE EXCEPTION 'coordinator administrator identity is invalid';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS owner_role
            ON owner_role.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member_role
            ON member_role.oid = membership.member
         WHERE owner_role.rolname = 'canonical_brain_migration_owner'
           AND member_role.rolname = admin_name
    ) THEN
        EXECUTE pg_catalog.format(
            'REVOKE canonical_brain_migration_owner FROM %I', admin_name
        );
    END IF;
    ALTER ROLE canonical_brain_canary_bootstrap_login PASSWORD NULL;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS owner_role
            ON owner_role.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member_role
            ON member_role.oid = membership.member
         WHERE owner_role.rolname = 'canonical_brain_migration_owner'
           AND member_role.rolname = admin_name
    ) THEN
        RAISE EXCEPTION 'coordinator administrator membership survived cleanup';
    END IF;
END
$muncho_coordinator$"""

_COORDINATOR_INPUT_FIELDS = frozenset({
    "schema",
    "revision",
    "writer_activation_plan",
    "writer_activation_receipt",
    "writer_activation_receipt_file_sha256",
    "identities",
    "writer_config",
    "artifacts",
    "bootstrap_sql_sha256",
    "bootstrap_retire_sql_sha256",
    "coordinator_input_sha256",
})
_CREDENTIAL_PREPARE_APPROVAL_FIELDS = frozenset({
    "schema",
    "scope",
    "coordinator_input_sha256",
    "release_sha",
    "authority_kind",
    "cryptographic_owner_proof",
    "owner_subject_sha256",
    "approval_source_sha256",
    "nonce_sha256",
    "approved_at_unix",
    "expires_at_unix",
})
_DISCORD_TOKEN_INSTALL_APPROVAL_FIELDS = frozenset(_CREDENTIAL_PREPARE_APPROVAL_FIELDS)
_OWNER_APPROVAL_REQUEST_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "approval_source_sha256",
    "ephemeral_admin_username",
    "full_canary_plan_sha256",
    "staged_plan_path",
    "staged_plan_file_sha256",
    "approval_request_path",
    "approval_path",
    "hba_receipt_sha256",
    "hba_expires_at_unix",
    "fixture_expires_at_unix",
    "credential_approval_expires_at_unix",
    "requested_at_unix",
    "approval_deadline_unix",
    "owner_input_cutoff_unix",
    "final_approval_transmit_margin_seconds",
    "max_wait_seconds",
    "prior_approval_file_sha256",
    "final_approval_frame_schema",
    "request_sha256",
})
FINAL_APPROVAL_CANCEL_RECEIPT_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "reason",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "full_canary_plan_sha256",
    "approval_request_sha256",
    "approval_request_path",
    "expected_approval_request_file_sha256",
    "observed_approval_request_file_sha256",
    "approval_request_artifact_state",
    "approval_request_present",
    "approval_request_remains_active",
    "staged_plan_path",
    "expected_staged_plan_file_sha256",
    "observed_staged_plan_file_sha256",
    "staged_plan_artifact_state",
    "staged_plan_present",
    "approval_path",
    "prior_approval_file_sha256",
    "observed_approval_file_sha256",
    "owner_approval_artifact_state",
    "approval_path_matches_prior",
    "new_owner_approval_installed",
    "frame_bytes_received",
    "owner_approval_mutation_performed_by_this_helper",
    "cancelled_at_unix",
    "receipt_sha256",
})


class CoordinatorError(RuntimeError):
    """Stable secret-free coordinator failure."""

    def __init__(self, code: str, *, phase: str = "") -> None:
        self.code = code
        self.phase = phase
        super().__init__(code)


class CoordinatorCleanupBlocked(CoordinatorError):
    """Exact cleanup could not be proven; recovery material is preserved."""


class _FinalApprovalNoSecretCancellation(CoordinatorError):
    """FD 0 reached EOF before any MFA1 byte was disclosed."""

    def __init__(self) -> None:
        super().__init__(
            "final_approval_cancelled_no_secret",
            phase="final_approval_read",
        )


_ACTIVE_SIGNAL_FENCE = threading.local()


def _begin_active_signal_cleanup() -> None:
    fence = getattr(_ACTIVE_SIGNAL_FENCE, "value", None)
    if fence is not None:
        fence.begin_cleanup()


@contextmanager
def _defer_termination_signals():
    pthread_sigmask = getattr(signal, "pthread_sigmask", None)
    handled = {
        value
        for value in (
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGINT", None),
            getattr(signal, "SIGTERM", None),
        )
        if value is not None
    }
    if not callable(pthread_sigmask) or not handled:
        _fail("coordinator_signal_defer_unavailable", phase="signal")
    old_mask = pthread_sigmask(signal.SIG_BLOCK, handled)
    try:
        yield
    finally:
        pthread_sigmask(signal.SIG_SETMASK, old_mask)


def _fail(code: str, *, phase: str = "") -> None:
    raise CoordinatorError(code, phase=phase)


def _harden_secret_process() -> None:
    """Fail closed before emitting a gate for any inherited secret frame."""

    _require_root_linux()
    try:
        harden_current_process_against_dumping()
        dumpable, core_soft, core_hard = current_process_hardening_state()
    except BaseException as exc:
        raise CoordinatorError(
            "secret_process_hardening_failed",
            phase="process_hardening",
        ) from exc
    if dumpable or core_soft != 0 or core_hard != 0:
        _fail("secret_process_hardening_unconfirmed", phase="process_hardening")
    os.umask(0o077)


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
        raise CoordinatorError("non_canonical_json") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(code)
    return value


def _revision(value: Any, code: str = "release_revision_invalid") -> str:
    if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
        _fail(code)
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str) or not key or key in result:
            _fail("strict_json_invalid")
        result[key] = value
    return result


def _decode_mapping(raw: bytes, *, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                ValueError("non-JSON constant")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CoordinatorError(code) from exc
    if not isinstance(value, Mapping) or raw != _canonical_bytes(value):
        _fail(code)
    return value


def _stable_root_read(path: Path, *, maximum: int) -> bytes:
    """Read one root:root 0400 regular file through a stable no-follow FD."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise CoordinatorError("trusted_root_file_unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != 0
        or before.st_gid != 0
        or stat.S_IMODE(before.st_mode) != 0o400
        or not 0 < before.st_size <= maximum
    ):
        _fail("trusted_root_file_identity_invalid")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    chunks: list[bytes] = []
    total = 0
    try:
        opened = os.fstat(descriptor)
        while total <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        reachable = path.lstat()
    except OSError as exc:
        raise CoordinatorError("trusted_root_file_replaced") from exc

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
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
        total > maximum
        or total != before.st_size
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
        or identity(before) != identity(reachable)
    ):
        _fail("trusted_root_file_replaced")
    return b"".join(chunks)


_ROOT_PUBLICATION_PATHS = frozenset({
    COORDINATOR_INPUT_PATH,
    COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH,
    DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
    DISCORD_TOKEN_RETIREMENT_RECEIPT_PATH,
    OWNER_APPROVAL_REQUEST_PATH,
    DEFAULT_STAGED_PLAN_PATH,
    DEFAULT_PLAN_PATH,
    DEFAULT_APPROVAL_PATH,
    COORDINATOR_PROCESS_LEASE_PATH,
    PREPLAN_STOPPED_REPORT_PATH,
})


@dataclass(frozen=True)
class _RootFileSnapshot:
    path: Path
    raw: bytes
    item: os.stat_result

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.raw)


@dataclass
class _RootRemovalState:
    unlinked: bool = False


def _capture_root_snapshot(
    path: Path,
    *,
    maximum: int = MAX_COORDINATOR_INPUT_BYTES,
) -> _RootFileSnapshot | None:
    if path not in _ROOT_PUBLICATION_PATHS:
        _fail("root_publication_path_not_fixed")
    if not os.path.lexists(path):
        return None
    raw = _stable_root_read(path, maximum=maximum)
    return _RootFileSnapshot(path=path, raw=raw, item=path.lstat())


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_nlink,
        left.st_uid,
        left.st_gid,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_nlink,
        right.st_uid,
        right.st_gid,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


def _remove_exact_root_snapshot(
    snapshot: _RootFileSnapshot,
    *,
    state: _RootRemovalState,
) -> None:
    if snapshot.path not in _ROOT_PUBLICATION_PATHS:
        _fail("root_snapshot_cleanup_path_not_fixed")
    errors: list[BaseException] = []
    if not state.unlinked:
        try:
            current = _capture_root_snapshot(snapshot.path)
            if (
                current is None
                or current.raw != snapshot.raw
                or not _same_file_identity(current.item, snapshot.item)
            ):
                raise RuntimeError("root snapshot cleanup identity changed")
            snapshot.path.unlink()
            state.unlinked = True
        except BaseException as exc:
            errors.append(exc)
    fsync_error: BaseException | None = None
    for _attempt in range(2):
        try:
            _fsync_directory(snapshot.path.parent)
            fsync_error = None
            break
        except BaseException as exc:
            fsync_error = exc
    if fsync_error is not None:
        errors.append(fsync_error)
    try:
        snapshot.path.lstat()
    except FileNotFoundError:
        pass
    except BaseException as exc:
        errors.append(exc)
    else:
        errors.append(RuntimeError("root snapshot cleanup unconfirmed"))
    if errors:
        raise CoordinatorCleanupBlocked(
            "root_snapshot_cleanup_blocked",
            phase="root_snapshot_cleanup",
        ) from ExceptionGroup("root snapshot cleanup failures", errors)


def _write_root_publication_temp(
    path: Path, payload: bytes
) -> tuple[Path, os.stat_result]:
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid != 0
        or stat.S_IMODE(parent.st_mode) & 0o022
        or not 0 < len(payload) <= MAX_COORDINATOR_INPUT_BYTES
    ):
        _fail("root_publication_parent_or_payload_invalid")
    temporary = path.with_name(f".{path.name}.publish.{os.getpid()}.{uuid.uuid4().hex}")
    descriptor = -1
    opened: os.stat_result | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        opened = os.fstat(descriptor)
        os.fchown(descriptor, 0, 0)
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail("root_publication_write_stalled")
            offset += written
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        os.close(descriptor)
        descriptor = -1
        return temporary, opened
    except BaseException as primary:
        _begin_active_signal_cleanup()
        cleanup_errors: list[BaseException] = []
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as exc:
                cleanup_errors.append(exc)
        try:
            current = temporary.lstat()
        except FileNotFoundError:
            pass
        except BaseException as exc:
            cleanup_errors.append(exc)
        else:
            if opened is None or (current.st_dev, current.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                cleanup_errors.append(
                    RuntimeError("root publication temp identity changed")
                )
            else:
                try:
                    temporary.unlink()
                except BaseException as exc:
                    cleanup_errors.append(exc)
        try:
            _fsync_directory(path.parent)
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            temporary.lstat()
        except FileNotFoundError:
            pass
        except BaseException as exc:
            cleanup_errors.append(exc)
        else:
            cleanup_errors.append(
                RuntimeError("root publication temp cleanup unconfirmed")
            )
        if cleanup_errors:
            raise CoordinatorCleanupBlocked(
                "root_publication_temp_cleanup_blocked",
                phase="root_publication_cleanup",
            ) from ExceptionGroup(
                "root publication temp creation and cleanup failed",
                [primary, *cleanup_errors],
            )
        raise


@dataclass
class _RootPublication:
    path: Path
    before: _RootFileSnapshot | None
    after: _RootFileSnapshot
    changed: bool
    _rolled_back: bool = False
    _unlinked: bool = False
    _removal_state: _RootRemovalState = field(
        default_factory=_RootRemovalState,
    )

    def rollback(self) -> None:
        if not self.changed or self._rolled_back:
            self._rolled_back = True
            return
        current = _capture_root_snapshot(self.path)
        if (
            current is None
            or current.raw != self.after.raw
            or not _same_file_identity(current.item, self.after.item)
        ):
            raise CoordinatorCleanupBlocked(
                "root_publication_cleanup_identity_changed",
                phase="root_publication_cleanup",
            )
        if self.before is None:
            _remove_exact_root_snapshot(
                self.after,
                state=self._removal_state,
            )
        else:
            try:
                restored = _publish_root_payload(
                    self.path,
                    self.before.raw,
                    expected_previous_sha256=self.after.sha256,
                )
                if restored.after.raw != self.before.raw:
                    raise RuntimeError("root publication restore drifted")
            except BaseException as exc:
                raise CoordinatorCleanupBlocked(
                    "root_publication_restore_blocked",
                    phase="root_publication_cleanup",
                ) from exc
        self._rolled_back = True


def _publish_root_payload(
    path: Path,
    payload: bytes,
    *,
    expected_previous_sha256: str | None,
) -> _RootPublication:
    """Publish one fixed root artifact, replacing only an exact predecessor."""

    if path not in _ROOT_PUBLICATION_PATHS:
        _fail("root_publication_path_not_fixed")
    before = _capture_root_snapshot(path)
    if expected_previous_sha256 is None:
        if before is not None:
            _fail("root_publication_target_not_fresh")
    else:
        expected = _digest(
            expected_previous_sha256,
            "root_publication_previous_digest_invalid",
        )
        if before is None or before.sha256 != expected:
            _fail("root_publication_previous_drifted")
        if before.raw == payload:
            return _RootPublication(
                path=path,
                before=before,
                after=before,
                changed=False,
            )
    temporary, opened = _write_root_publication_temp(path, payload)
    published = False
    completed = False
    try:
        if before is None:
            os.link(temporary, path, follow_symlinks=False)
            published = True
            temporary.unlink()
        else:
            current = _capture_root_snapshot(path)
            if (
                current is None
                or current.sha256 != before.sha256
                or not _same_file_identity(current.item, before.item)
            ):
                _fail("root_publication_previous_replaced")
            os.replace(temporary, path)
            published = True
        _fsync_directory(path.parent)
        after = _capture_root_snapshot(path)
        if (
            after is None
            or after.raw != payload
            or (after.item.st_dev, after.item.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            _fail("root_publication_readback_invalid")
        result = _RootPublication(
            path=path,
            before=before,
            after=after,
            changed=True,
        )
        completed = True
        return result
    except BaseException as primary:
        _begin_active_signal_cleanup()
        cleanup_errors: list[BaseException] = []
        if published:
            try:
                current = _capture_root_snapshot(path)
                if current is None or current.raw != payload:
                    raise RuntimeError("published root artifact changed")
                if before is None:
                    path.unlink()
                    _fsync_directory(path.parent)
                    if os.path.lexists(path):
                        raise RuntimeError("published root artifact survived")
                else:
                    rollback_temp, _rollback_opened = _write_root_publication_temp(
                        path,
                        before.raw,
                    )
                    try:
                        current_again = _capture_root_snapshot(path)
                        if (
                            current_again is None
                            or current_again.sha256 != current.sha256
                            or not _same_file_identity(
                                current_again.item,
                                current.item,
                            )
                        ):
                            raise RuntimeError("published root artifact raced")
                        os.replace(rollback_temp, path)
                        _fsync_directory(path.parent)
                    finally:
                        try:
                            rollback_temp.unlink()
                        except FileNotFoundError:
                            pass
                    restored = _capture_root_snapshot(path)
                    if restored is None or restored.raw != before.raw:
                        raise RuntimeError("root artifact rollback unconfirmed")
            except BaseException as cleanup:
                cleanup_errors.append(cleanup)
        if cleanup_errors:
            raise CoordinatorCleanupBlocked(
                "root_publication_cleanup_blocked",
                phase="root_publication_cleanup",
            ) from ExceptionGroup(
                "root publication failure and cleanup failure",
                [primary, *cleanup_errors],
            )
        raise
    finally:
        if not completed:
            cleanup_errors: list[BaseException] = []
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                temporary.lstat()
            except FileNotFoundError:
                pass
            except BaseException as exc:
                cleanup_errors.append(exc)
            else:
                cleanup_errors.append(
                    RuntimeError("root publication temporary survived")
                )
            if cleanup_errors:
                raise CoordinatorCleanupBlocked(
                    "root_publication_temporary_cleanup_blocked",
                    phase="root_publication_cleanup",
                ) from ExceptionGroup(
                    "root publication temporary cleanup failures",
                    cleanup_errors,
                )


def _read_exact_artifact(artifact: ExactArtifact, *, label: str) -> bytes:
    try:
        before = artifact.source_path.lstat()
    except OSError as exc:
        raise CoordinatorError(f"{label}_artifact_unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != artifact.uid
        or before.st_gid != artifact.gid
        or stat.S_IMODE(before.st_mode) != artifact.mode
        or not 0 < before.st_size <= artifact.maximum_bytes
    ):
        _fail(f"{label}_artifact_identity_invalid")
    descriptor = os.open(
        artifact.source_path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        raw = bytearray()
        while len(raw) <= artifact.maximum_bytes:
            chunk = os.read(
                descriptor,
                min(64 * 1024, artifact.maximum_bytes + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = artifact.source_path.lstat()
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
    result = bytes(raw)
    if (
        len(result) > artifact.maximum_bytes
        or len(result) != before.st_size
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
        or identity(before) != identity(reachable)
        or _sha256_bytes(result) != artifact.sha256
    ):
        _fail(f"{label}_artifact_replaced_or_drifted")
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_install_secret(
    path: Path,
    payload: bytearray,
    *,
    uid: int,
    gid: int,
) -> os.stat_result:
    """Install one fixed credential without replace and return exact identity."""

    if path not in {DISCORD_TOKEN_PATH, CANARY_BOOTSTRAP_CREDENTIAL_PATH}:
        _fail("secret_install_path_not_fixed")
    if not payload or len(payload) > MAX_DISCORD_TOKEN_BYTES:
        _fail("secret_install_payload_bound_invalid")
    if type(uid) is not int or type(gid) is not int or uid <= 0 or gid <= 0:
        _fail("secret_install_owner_invalid")
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid != 0
        or stat.S_IMODE(parent.st_mode) & 0o022
        or os.path.lexists(path)
    ):
        _fail("secret_install_target_not_fresh")
    temporary = path.with_name(f".{path.name}.install.{os.getpid()}.{uuid.uuid4().hex}")
    descriptor = -1
    linked = False
    completed = False
    opened: os.stat_result | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        opened = os.fstat(descriptor)
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail("secret_install_write_stalled")
            offset += written
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        os.close(descriptor)
        descriptor = -1
        # Hard-link publication is no-replace on every supported Linux Python
        # and keeps the operation shell-free.  The temporary name is removed
        # immediately, leaving a single-link credential.
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        temporary.unlink()
        _fsync_directory(path.parent)
        installed = path.lstat()
        if (
            not stat.S_ISREG(installed.st_mode)
            or stat.S_ISLNK(installed.st_mode)
            or installed.st_nlink != 1
            or installed.st_uid != uid
            or installed.st_gid != gid
            or stat.S_IMODE(installed.st_mode) != 0o400
            or installed.st_size != len(payload)
            or (installed.st_dev, installed.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            _fail("secret_install_readback_invalid")
        completed = True
        return installed
    except FileExistsError as exc:
        raise CoordinatorError("secret_install_target_not_fresh") from exc
    finally:
        cleanup_errors: list[BaseException] = []
        if not completed:
            _begin_active_signal_cleanup()
        if not completed and descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as exc:
                cleanup_errors.append(exc)

        # Every cleanup target is attempted independently.  A failure on the
        # still-open descriptor or on either unlink must never skip the other
        # path.  We only unlink names that still resolve to the inode created
        # by this call; an unexpected identity is preserved for recovery.
        def remove_owned_name(candidate: Path) -> None:
            try:
                current = candidate.lstat()
            except FileNotFoundError:
                return
            except BaseException as exc:
                cleanup_errors.append(exc)
                return
            if opened is None or (current.st_dev, current.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                cleanup_errors.append(RuntimeError("secret cleanup identity changed"))
                return
            try:
                candidate.unlink()
            except BaseException as exc:
                cleanup_errors.append(exc)

        # On success the temporary name must still be absent, while the
        # published name is retained.  On every non-success path both names
        # owned by this call are removed.
        if not completed:
            remove_owned_name(temporary)
        if linked and not completed:
            remove_owned_name(path)

        if not completed:
            try:
                _fsync_directory(path.parent)
            except BaseException as exc:
                cleanup_errors.append(exc)

        def prove_absent(candidate: Path) -> None:
            try:
                candidate.lstat()
            except FileNotFoundError:
                return
            except BaseException as exc:
                cleanup_errors.append(exc)
                return
            cleanup_errors.append(RuntimeError("secret cleanup not absent"))

        if not completed:
            prove_absent(temporary)
        if linked and not completed:
            prove_absent(path)

        if cleanup_errors:
            raise CoordinatorCleanupBlocked(
                "secret_install_cleanup_blocked",
                phase="secret_install_cleanup",
            ) from ExceptionGroup(
                "secret install cleanup failures",
                cleanup_errors,
            )


@dataclass
class _SecretRemovalState:
    unlinked: bool = False


def _remove_exact_secret(
    path: Path,
    expected: os.stat_result,
    *,
    state: _SecretRemovalState | None = None,
) -> None:
    if path not in {DISCORD_TOKEN_PATH, CANARY_BOOTSTRAP_CREDENTIAL_PATH}:
        _fail("secret_cleanup_path_not_fixed")
    removal = state if state is not None else _SecretRemovalState()
    errors: list[BaseException] = []
    if not removal.unlinked:
        try:
            current = path.lstat()
            if (
                not stat.S_ISREG(current.st_mode)
                or stat.S_ISLNK(current.st_mode)
                or current.st_nlink != 1
                or (current.st_dev, current.st_ino)
                != (expected.st_dev, expected.st_ino)
                or current.st_uid != expected.st_uid
                or current.st_gid != expected.st_gid
                or current.st_mode != expected.st_mode
            ):
                raise RuntimeError("secret cleanup identity changed")
            path.unlink()
            removal.unlinked = True
        except BaseException as exc:
            errors.append(exc)
    try:
        _fsync_directory(path.parent)
    except BaseException as exc:
        errors.append(exc)
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    except BaseException as exc:
        errors.append(exc)
    else:
        errors.append(RuntimeError("secret cleanup removal unconfirmed"))
    if errors:
        raise CoordinatorCleanupBlocked(
            "secret_cleanup_blocked",
            phase="secret_cleanup",
        ) from ExceptionGroup("secret cleanup failures", errors)


def _validate_secret_metadata(
    path: Path,
    *,
    uid: int,
    gid: int,
    maximum: int,
) -> os.stat_result:
    try:
        item = path.lstat()
    except OSError as exc:
        raise CoordinatorError("required_secret_metadata_unavailable") from exc
    if (
        not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid != uid
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) != 0o400
        or not 0 < item.st_size <= maximum
    ):
        _fail("required_secret_metadata_invalid")
    return item


@dataclass(frozen=True)
class CoordinatorInput:
    value: Mapping[str, Any]
    writer_activation_plan: ActivationPlan
    identities: FullCanaryIdentities
    artifacts: Mapping[str, ExactArtifact]
    base_plan: FullCanaryPlan

    @classmethod
    def from_mapping(cls, value: Any) -> "CoordinatorInput":
        if not isinstance(value, Mapping) or set(value) != _COORDINATOR_INPUT_FIELDS:
            _fail("coordinator_input_fields_invalid")
        raw = copy.deepcopy(dict(value))
        if raw["schema"] != COORDINATOR_INPUT_SCHEMA:
            _fail("coordinator_input_schema_invalid")
        revision = _revision(raw["revision"])
        expected_digest = _digest(
            raw["coordinator_input_sha256"],
            "coordinator_input_digest_invalid",
        )
        unsigned = {
            name: copy.deepcopy(item)
            for name, item in raw.items()
            if name != "coordinator_input_sha256"
        }
        if _sha256_json(unsigned) != expected_digest:
            _fail("coordinator_input_self_digest_mismatch")
        try:
            writer_plan = ActivationPlan.from_mapping(raw["writer_activation_plan"])
            identities = FullCanaryIdentities.from_mapping(raw["identities"])
        except (TypeError, ValueError) as exc:
            raise CoordinatorError("coordinator_input_nested_contract_invalid") from exc
        if writer_plan.revision != revision:
            _fail("coordinator_input_release_mismatch")
        if not isinstance(raw["writer_activation_receipt"], Mapping):
            _fail("coordinator_input_writer_receipt_invalid")
        try:
            writer_receipt = _validate_writer_only_receipt(
                raw["writer_activation_receipt"],
                plan=writer_plan,
            )
        except (TypeError, ValueError) as exc:
            raise CoordinatorError(
                "coordinator_input_writer_receipt_semantics_invalid"
            ) from exc
        _digest(
            raw["writer_activation_receipt_file_sha256"],
            "coordinator_input_writer_receipt_digest_invalid",
        )
        if not isinstance(raw["writer_config"], Mapping):
            _fail("coordinator_input_writer_config_invalid")
        artifact_raw = raw["artifacts"]
        if not isinstance(artifact_raw, Mapping) or set(artifact_raw) != {
            "gateway_config",
            "edge_config",
            "e2e_fixture",
            "host_identity_receipt",
        }:
            _fail("coordinator_input_artifacts_invalid")
        try:
            artifacts = {
                name: ExactArtifact.from_mapping(item, label=name)
                for name, item in artifact_raw.items()
            }
        except (TypeError, ValueError) as exc:
            raise CoordinatorError(
                "coordinator_input_artifact_contract_invalid"
            ) from exc
        expected_sources = {
            "gateway_config": DEFAULT_GATEWAY_CONFIG_SOURCE,
            "edge_config": DEFAULT_EDGE_CONFIG_SOURCE,
            "e2e_fixture": DEFAULT_E2E_FIXTURE,
            "host_identity_receipt": DEFAULT_HOST_IDENTITY_RECEIPT,
        }
        if any(
            artifacts[name].source_path != expected
            for name, expected in expected_sources.items()
        ):
            _fail("coordinator_input_artifact_path_not_fixed")
        expected_targets = {
            "gateway_config": DEFAULT_GATEWAY_CONFIG,
            "edge_config": DEFAULT_EDGE_CONFIG,
            "e2e_fixture": DEFAULT_E2E_FIXTURE,
            "host_identity_receipt": DEFAULT_HOST_IDENTITY_RECEIPT,
        }
        if any(
            artifacts[name].target_path != expected
            for name, expected in expected_targets.items()
        ):
            _fail("coordinator_input_artifact_target_not_fixed")
        _digest(raw["bootstrap_sql_sha256"], "bootstrap_sql_digest_invalid")
        _digest(
            raw["bootstrap_retire_sql_sha256"],
            "bootstrap_retire_sql_digest_invalid",
        )
        database = raw["writer_config"].get("database")
        if (
            not isinstance(database, Mapping)
            or database.get("host") != CANARY_DATABASE_HOST
            or database.get("port") != CANARY_DATABASE_PORT
            or database.get("database") != CANARY_DATABASE_NAME
            or not isinstance(database.get("tls_server_name"), str)
            or _TLS_NAME_RE.fullmatch(database["tls_server_name"]) is None
        ):
            _fail("coordinator_input_database_not_fixed")
        scope = raw["writer_config"].get("canary_scope_preapproval")
        hba = (
            scope.get("bootstrap_managed_cloudsqladmin_hba_rejection_receipt")
            if isinstance(scope, Mapping)
            else None
        )
        if (
            not isinstance(hba, Mapping)
            or hba.get("host") != CANARY_DATABASE_HOST
            or hba.get("tls_server_name") != database["tls_server_name"]
            or hba.get("port") != CANARY_DATABASE_PORT
            or hba.get("user") != CANARY_BOOTSTRAP_LOGIN
            or hba.get("database") != "cloudsqladmin"
            or hba.get("tls_peer_verified") is not True
        ):
            _fail("coordinator_input_hba_binding_invalid")
        _digest(
            hba.get("server_certificate_sha256"),
            "coordinator_input_hba_peer_digest_invalid",
        )
        try:
            parsed_hba = managed_cloudsqladmin_hba_receipt_from_mapping(hba)
        except (TypeError, ValueError) as exc:
            raise CoordinatorError("coordinator_input_hba_receipt_invalid") from exc
        configured_hba_sha256 = scope.get(
            "bootstrap_managed_cloudsqladmin_hba_rejection_sha256"
        )
        if parsed_hba.sha256 != configured_hba_sha256:
            _fail("coordinator_input_hba_self_digest_mismatch")
        writer_config_raw = _canonical_bytes(raw["writer_config"])
        writer_artifact = ExactArtifact(
            source_path=DEFAULT_WRITER_CONFIG_SOURCE,
            target_path=DEFAULT_WRITER_CONFIG,
            sha256=_sha256_bytes(writer_config_raw),
            mode=0o440,
            uid=0,
            gid=identities.writer_gid,
        )
        try:
            base_plan = build_full_canary_plan(
                writer_activation_plan=writer_plan,
                writer_activation_receipt=writer_receipt,
                writer_activation_receipt_file_sha256=raw[
                    "writer_activation_receipt_file_sha256"
                ],
                identities=identities,
                writer_config=writer_artifact,
                gateway_config=artifacts["gateway_config"],
                edge_config=artifacts["edge_config"],
                e2e_fixture=artifacts["e2e_fixture"],
                host_identity_receipt=artifacts["host_identity_receipt"],
            )
            _validate_writer_config(
                writer_config_raw,
                identities,
                plan=base_plan,
                expected_approval_source_sha256=str(scope["approval_source_sha256"]),
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise CoordinatorError(
                "coordinator_input_writer_config_semantics_invalid"
            ) from exc
        return cls(
            value=raw,
            writer_activation_plan=writer_plan,
            identities=identities,
            artifacts=artifacts,
            base_plan=base_plan,
        )

    @property
    def sha256(self) -> str:
        return str(self.value["coordinator_input_sha256"])

    @property
    def revision(self) -> str:
        return str(self.value["revision"])

    @property
    def tls_server_name(self) -> str:
        return str(self.value["writer_config"]["database"]["tls_server_name"])

    @property
    def tls_peer_certificate_sha256(self) -> str:
        return str(
            self.value["writer_config"]["canary_scope_preapproval"][
                "bootstrap_managed_cloudsqladmin_hba_rejection_receipt"
            ]["server_certificate_sha256"]
        )


@dataclass(frozen=True)
class CredentialPrepareApproval:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "CredentialPrepareApproval":
        if (
            not isinstance(value, Mapping)
            or set(value) != _CREDENTIAL_PREPARE_APPROVAL_FIELDS
        ):
            _fail("credential_prepare_approval_fields_invalid")
        raw = copy.deepcopy(dict(value))
        if (
            raw["schema"] != CREDENTIAL_PREPARE_APPROVAL_SCHEMA
            or raw["scope"] != "full_canary_ephemeral_admin_prepare"
            or raw["authority_kind"] != "trusted_root_bootstrap_out_of_band_owner"
            or raw["cryptographic_owner_proof"] is not False
        ):
            _fail("credential_prepare_approval_semantics_invalid")
        _digest(
            raw["coordinator_input_sha256"], "credential_prepare_input_digest_invalid"
        )
        _revision(raw["release_sha"], "credential_prepare_release_invalid")
        for name in (
            "owner_subject_sha256",
            "approval_source_sha256",
            "nonce_sha256",
        ):
            _digest(raw[name], f"credential_prepare_{name}_invalid")
        approved = raw["approved_at_unix"]
        expires = raw["expires_at_unix"]
        if (
            type(approved) is not int
            or type(expires) is not int
            or approved < 0
            or not 1 <= expires - approved <= 900
        ):
            _fail("credential_prepare_approval_window_invalid")
        return cls(value=raw)

    @property
    def sha256(self) -> str:
        return _sha256_json(self.value)

    def require(self, *, coordinator_input: CoordinatorInput, now_unix: int) -> None:
        scope = coordinator_input.value["writer_config"].get("canary_scope_preapproval")
        if (
            self.value["coordinator_input_sha256"] != coordinator_input.sha256
            or self.value["release_sha"] != coordinator_input.revision
            or not isinstance(scope, Mapping)
            or self.value["approval_source_sha256"]
            != scope.get("approval_source_sha256")
            or type(now_unix) is not int
            or not self.value["approved_at_unix"]
            <= now_unix
            <= self.value["expires_at_unix"]
        ):
            _fail("credential_prepare_approval_not_fresh_or_bound")


@dataclass(frozen=True)
class DiscordTokenInstallApproval:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "DiscordTokenInstallApproval":
        if (
            not isinstance(value, Mapping)
            or set(value) != _DISCORD_TOKEN_INSTALL_APPROVAL_FIELDS
        ):
            _fail("discord_token_install_approval_fields_invalid")
        raw = copy.deepcopy(dict(value))
        if (
            raw["schema"] != DISCORD_TOKEN_INSTALL_APPROVAL_SCHEMA
            or raw["scope"] != "full_canary_discord_token_install"
            or raw["authority_kind"] != "trusted_root_bootstrap_out_of_band_owner"
            or raw["cryptographic_owner_proof"] is not False
        ):
            _fail("discord_token_install_approval_semantics_invalid")
        _digest(raw["coordinator_input_sha256"], "discord_token_input_digest_invalid")
        _revision(raw["release_sha"], "discord_token_release_invalid")
        for name in (
            "owner_subject_sha256",
            "approval_source_sha256",
            "nonce_sha256",
        ):
            _digest(raw[name], f"discord_token_{name}_invalid")
        approved = raw["approved_at_unix"]
        expires = raw["expires_at_unix"]
        if (
            type(approved) is not int
            or type(expires) is not int
            or approved < 0
            or not 1 <= expires - approved <= 900
        ):
            _fail("discord_token_install_approval_window_invalid")
        return cls(value=raw)

    @property
    def sha256(self) -> str:
        return _sha256_json(self.value)

    def require(self, *, coordinator_input: CoordinatorInput, now_unix: int) -> None:
        scope = coordinator_input.value["writer_config"].get("canary_scope_preapproval")
        if (
            self.value["coordinator_input_sha256"] != coordinator_input.sha256
            or self.value["release_sha"] != coordinator_input.revision
            or not isinstance(scope, Mapping)
            or self.value["approval_source_sha256"]
            != scope.get("approval_source_sha256")
            or type(now_unix) is not int
            or not self.value["approved_at_unix"]
            <= now_unix
            <= self.value["expires_at_unix"]
        ):
            _fail("discord_token_install_approval_not_fresh_or_bound")


@dataclass(frozen=True)
class OwnerApprovalRequest:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "OwnerApprovalRequest":
        if (
            not isinstance(value, Mapping)
            or set(value) != _OWNER_APPROVAL_REQUEST_FIELDS
        ):
            _fail("owner_approval_request_fields_invalid")
        raw = copy.deepcopy(dict(value))
        if (
            raw["schema"] != OWNER_APPROVAL_REQUEST_SCHEMA
            or raw["ok"] is not True
            or raw["state"] != "awaiting_final_owner_approval"
            or raw["staged_plan_path"] != str(DEFAULT_STAGED_PLAN_PATH)
            or raw["approval_request_path"] != str(OWNER_APPROVAL_REQUEST_PATH)
            or raw["approval_path"] != str(DEFAULT_APPROVAL_PATH)
            or raw["final_approval_frame_schema"] != FINAL_APPROVAL_FRAME_SCHEMA
        ):
            _fail("owner_approval_request_semantics_invalid")
        _revision(raw["release_sha"], "owner_approval_request_release_invalid")
        for name in (
            "coordinator_input_sha256",
            "credential_prepare_approval_sha256",
            "owner_subject_sha256",
            "approval_source_sha256",
            "full_canary_plan_sha256",
            "staged_plan_file_sha256",
            "hba_receipt_sha256",
        ):
            _digest(raw[name], f"owner_approval_request_{name}_invalid")
        prior = raw["prior_approval_file_sha256"]
        if prior is not None:
            _digest(prior, "owner_approval_request_prior_approval_invalid")
        if (
            not isinstance(raw["ephemeral_admin_username"], str)
            or _ADMIN_USERNAME_RE.fullmatch(raw["ephemeral_admin_username"]) is None
        ):
            _fail("owner_approval_request_admin_username_invalid")
        requested = raw["requested_at_unix"]
        deadline = raw["approval_deadline_unix"]
        input_cutoff = raw["owner_input_cutoff_unix"]
        transmit_margin = raw["final_approval_transmit_margin_seconds"]
        hba_expires = raw["hba_expires_at_unix"]
        fixture_expires = raw["fixture_expires_at_unix"]
        credential_expires = raw["credential_approval_expires_at_unix"]
        wait = raw["max_wait_seconds"]
        if (
            any(
                type(item) is not int
                for item in (
                    requested,
                    deadline,
                    input_cutoff,
                    transmit_margin,
                    hba_expires,
                    fixture_expires,
                    credential_expires,
                    wait,
                )
            )
            or requested < 0
            or transmit_margin != FINAL_APPROVAL_TRANSMIT_MARGIN_SECONDS
            or not FINAL_APPROVAL_TRANSMIT_MARGIN_SECONDS + 1
            <= wait
            <= FINAL_APPROVAL_MAX_WAIT_SECONDS
            or deadline != requested + wait
            or input_cutoff != deadline - transmit_margin
            or not requested < input_cutoff < deadline
            or deadline > hba_expires - HBA_EXPIRY_SAFETY_MARGIN_SECONDS
            or deadline > fixture_expires
            or deadline > credential_expires - FINAL_APPROVAL_TRANSMIT_MARGIN_SECONDS
        ):
            _fail("owner_approval_request_window_invalid")
        expected = _sha256_json({
            key: item for key, item in raw.items() if key != "request_sha256"
        })
        if raw["request_sha256"] != expected:
            _fail("owner_approval_request_self_digest_mismatch")
        return cls(value=raw)

    @property
    def sha256(self) -> str:
        return str(self.value["request_sha256"])


def _read_final_approval_bytes(
    fd: int,
    size: int,
    *,
    cancel_on_initial_eof: bool,
) -> bytearray:
    result = bytearray()
    while len(result) < size:
        try:
            chunk = os.read(fd, size - len(result))
        except OSError as exc:
            _zeroize(result)
            raise CoordinatorError(
                "final_approval_frame_read_failed",
                phase="final_approval_read",
            ) from exc
        if not chunk:
            if cancel_on_initial_eof and not result:
                raise _FinalApprovalNoSecretCancellation()
            _zeroize(result)
            _fail(
                "final_approval_frame_truncated",
                phase="final_approval_read",
            )
        result.extend(chunk)
    return result


def _read_final_owner_approval_frame(
    *,
    fd: int = ADMIN_FRAME_FD,
) -> Mapping[str, Any]:
    if fd != ADMIN_FRAME_FD:
        _fail("final_approval_frame_fd_not_fixed", phase="final_approval_read")
    try:
        if os.isatty(fd):
            _fail("final_approval_frame_tty_forbidden", phase="final_approval_read")
    except OSError as exc:
        raise CoordinatorError(
            "final_approval_frame_fd_unavailable",
            phase="final_approval_read",
        ) from exc
    header = _read_final_approval_bytes(
        fd,
        _FINAL_APPROVAL_FRAME_HEADER.size,
        cancel_on_initial_eof=True,
    )
    raw: bytearray | None = None
    try:
        magic, size = _FINAL_APPROVAL_FRAME_HEADER.unpack(header)
        if magic != FINAL_APPROVAL_FRAME_MAGIC:
            _fail("final_approval_frame_magic_invalid", phase="final_approval_read")
        if not 1 <= size <= MAX_FINAL_APPROVAL_FRAME_BYTES:
            _fail("final_approval_frame_bound_invalid", phase="final_approval_read")
        raw = _read_final_approval_bytes(
            fd,
            size,
            cancel_on_initial_eof=False,
        )
        if os.read(fd, 1):
            _fail("final_approval_frame_trailing_data", phase="final_approval_read")
        return _decode_mapping(
            bytes(raw),
            code="final_approval_frame_json_invalid",
        )
    finally:
        _zeroize(header)
        _zeroize(raw)


class OpaqueDiscordTokenFrame:
    """One-shot DCT1 token holder with best-effort in-place zeroization."""

    __slots__ = ("_consumed", "_lock", "_token")

    def __init__(self, token: bytearray) -> None:
        self._token: bytearray | None = token
        self._consumed = False
        self._lock = threading.Lock()

    @classmethod
    def read(cls, *, fd: int = ADMIN_FRAME_FD) -> "OpaqueDiscordTokenFrame":
        if fd != ADMIN_FRAME_FD:
            _fail("discord_token_frame_fd_not_fixed", phase="discord_token_read")
        try:
            if os.isatty(fd):
                _fail("discord_token_frame_tty_forbidden", phase="discord_token_read")
        except OSError as exc:
            raise CoordinatorError(
                "discord_token_frame_fd_unavailable", phase="discord_token_read"
            ) from exc
        header = _read_exact(fd, _TOKEN_FRAME_HEADER.size)
        token: bytearray | None = None
        try:
            magic, token_size = _TOKEN_FRAME_HEADER.unpack(header)
            if magic != DISCORD_TOKEN_FRAME_MAGIC:
                _fail("discord_token_frame_magic_invalid", phase="discord_token_read")
            if not MIN_DISCORD_TOKEN_BYTES <= token_size <= MAX_DISCORD_TOKEN_BYTES:
                _fail("discord_token_frame_bound_invalid", phase="discord_token_read")
            token = _read_exact(fd, token_size)
            try:
                token_text = token.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise CoordinatorError(
                    "discord_token_frame_utf8_invalid", phase="discord_token_read"
                ) from exc
            if (
                token_text != token_text.strip()
                or any(ord(char) < 32 or ord(char) == 127 for char in token_text)
                or "\x00" in token_text
            ):
                _fail("discord_token_frame_value_invalid", phase="discord_token_read")
            token_text = ""
            if os.read(fd, 1):
                _fail("discord_token_frame_trailing_data", phase="discord_token_read")
            result_token = token
            token = None
            return cls(result_token)
        finally:
            _zeroize(header)
            _zeroize(token)

    def consume(self) -> bytearray:
        with self._lock:
            if self._consumed or self._token is None:
                _fail(
                    "discord_token_frame_replay_forbidden", phase="discord_token_read"
                )
            result = self._token
            self._token = None
            self._consumed = True
            return result

    def close(self) -> None:
        with self._lock:
            _zeroize(self._token)
            self._token = None
            self._consumed = True

    def __del__(self) -> None:  # pragma: no cover - best effort.
        self.close()

    def __repr__(self) -> str:
        return (
            "OpaqueDiscordTokenFrame(token=<redacted>, consumed="
            + repr(self._consumed)
            + ")"
        )


def _require_exact_nss_identity(
    *,
    user_name: str,
    group_name: str,
    expected_uid: int | None,
    expected_gid: int | None,
) -> tuple[int, int]:
    """Resolve one fixed service identity without accepting aliases."""

    try:
        user = pwd.getpwnam(user_name)
        user_by_uid = pwd.getpwuid(user.pw_uid)
        group = grp.getgrnam(group_name)
        group_by_gid = grp.getgrgid(group.gr_gid)
    except (KeyError, OSError) as exc:
        raise CoordinatorError("coordinator_input_nss_identity_unavailable") from exc
    if (
        user.pw_name != user_name
        or user_by_uid.pw_name != user_name
        or group.gr_name != group_name
        or group_by_gid.gr_name != group_name
        or user.pw_gid != group.gr_gid
        or user.pw_uid <= 0
        or group.gr_gid <= 0
        or (expected_uid is not None and user.pw_uid != expected_uid)
        or (expected_gid is not None and group.gr_gid != expected_gid)
    ):
        _fail("coordinator_input_nss_identity_drifted")
    return user.pw_uid, group.gr_gid


def _collect_full_canary_identities(
    writer_activation_plan: ActivationPlan,
) -> FullCanaryIdentities:
    """Extend the exact writer-only identities with the fixed Discord edge."""

    numeric = writer_activation_plan.identities
    gateway_uid, gateway_gid = _require_exact_nss_identity(
        user_name=GATEWAY_USER,
        group_name=GATEWAY_GROUP,
        expected_uid=numeric.gateway_uid,
        expected_gid=numeric.gateway_gid,
    )
    writer_uid, writer_gid = _require_exact_nss_identity(
        user_name=WRITER_USER,
        group_name=WRITER_GROUP,
        expected_uid=numeric.writer_uid,
        expected_gid=numeric.writer_gid,
    )
    try:
        socket_group = grp.getgrnam(SOCKET_CLIENT_GROUP)
        socket_group_by_gid = grp.getgrgid(socket_group.gr_gid)
    except (KeyError, OSError) as exc:
        raise CoordinatorError("coordinator_input_nss_identity_unavailable") from exc
    if (
        socket_group.gr_name != SOCKET_CLIENT_GROUP
        or socket_group_by_gid.gr_name != SOCKET_CLIENT_GROUP
        or socket_group.gr_gid != numeric.socket_client_gid
    ):
        _fail("coordinator_input_nss_identity_drifted")
    edge_uid, edge_gid = _require_exact_nss_identity(
        user_name="muncho-discord-egress",
        group_name="muncho-discord-egress",
        expected_uid=None,
        expected_gid=None,
    )
    try:
        return FullCanaryIdentities.from_mapping({
            "writer_user": WRITER_USER,
            "writer_group": WRITER_GROUP,
            "writer_uid": writer_uid,
            "writer_gid": writer_gid,
            "gateway_user": GATEWAY_USER,
            "gateway_group": GATEWAY_GROUP,
            "gateway_uid": gateway_uid,
            "gateway_gid": gateway_gid,
            "socket_client_group": SOCKET_CLIENT_GROUP,
            "socket_client_gid": socket_group.gr_gid,
            "edge_user": "muncho-discord-egress",
            "edge_group": "muncho-discord-egress",
            "edge_uid": edge_uid,
            "edge_gid": edge_gid,
        })
    except (TypeError, ValueError) as exc:
        raise CoordinatorError("coordinator_input_service_identities_invalid") from exc


def _collect_exact_artifact(
    *,
    source_path: Path,
    target_path: Path,
    mode: int,
    uid: int,
    gid: int,
    maximum_bytes: int = MAX_FIXED_CANARY_ARTIFACT_BYTES,
) -> tuple[ExactArtifact, bytes]:
    try:
        raw, _item = _read_stable_file(
            source_path,
            maximum=maximum_bytes,
            expected_uid=uid,
            expected_gid=gid,
            allowed_modes=frozenset({mode}),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CoordinatorError("coordinator_input_artifact_collection_failed") from exc
    if not raw:
        _fail("coordinator_input_artifact_empty")
    artifact = ExactArtifact(
        source_path=source_path,
        target_path=target_path,
        sha256=_sha256_bytes(raw),
        mode=mode,
        uid=uid,
        gid=gid,
        maximum_bytes=maximum_bytes,
    )
    return artifact, raw


def collect_coordinator_input() -> CoordinatorInput:
    """Build the input solely from fixed, installed, root-controlled state."""

    _require_root_linux()
    if os.path.lexists(COORDINATOR_PROCESS_LEASE_PATH):
        _fail("coordinator_process_recovery_required", phase="input_bootstrap")
    if not _services_are_exactly_stopped_and_disabled():
        _fail("coordinator_input_bootstrap_services_not_stopped")
    writer_plan = load_activation_plan(WRITER_ACTIVATION_PLAN_PATH)
    identities = _collect_full_canary_identities(writer_plan)
    receipt_path = _success_receipt_path(writer_plan, create_parent=False)
    receipt_raw = _stable_root_read(
        receipt_path,
        maximum=MAX_COORDINATOR_INPUT_BYTES,
    )
    writer_receipt = _decode_mapping(
        receipt_raw,
        code="coordinator_input_writer_receipt_file_invalid",
    )
    try:
        writer_receipt = _validate_writer_only_receipt(
            writer_receipt,
            plan=writer_plan,
        )
    except (TypeError, ValueError) as exc:
        raise CoordinatorError("coordinator_input_writer_receipt_invalid") from exc

    writer_artifact, writer_raw = _collect_exact_artifact(
        source_path=DEFAULT_WRITER_CONFIG_SOURCE,
        target_path=DEFAULT_WRITER_CONFIG,
        mode=0o440,
        uid=0,
        gid=identities.writer_gid,
    )
    writer_config = _decode_mapping(
        writer_raw,
        code="coordinator_input_writer_config_json_invalid",
    )
    gateway_artifact, _gateway_raw = _collect_exact_artifact(
        source_path=DEFAULT_GATEWAY_CONFIG_SOURCE,
        target_path=DEFAULT_GATEWAY_CONFIG,
        mode=0o440,
        uid=0,
        gid=identities.gateway_gid,
    )
    edge_artifact, _edge_raw = _collect_exact_artifact(
        source_path=DEFAULT_EDGE_CONFIG_SOURCE,
        target_path=DEFAULT_EDGE_CONFIG,
        mode=0o440,
        uid=0,
        gid=identities.edge_gid,
    )
    fixture_artifact, _fixture_raw = _collect_exact_artifact(
        source_path=DEFAULT_E2E_FIXTURE,
        target_path=DEFAULT_E2E_FIXTURE,
        mode=0o440,
        uid=0,
        gid=identities.gateway_gid,
    )
    host_artifact, _host_raw = _collect_exact_artifact(
        source_path=DEFAULT_HOST_IDENTITY_RECEIPT,
        target_path=DEFAULT_HOST_IDENTITY_RECEIPT,
        mode=0o400,
        uid=0,
        gid=0,
        maximum_bytes=_MAX_HOST_IDENTITY_RECEIPT_BYTES,
    )
    preliminary_plan = build_full_canary_plan(
        writer_activation_plan=writer_plan,
        writer_activation_receipt=writer_receipt,
        writer_activation_receipt_file_sha256=_sha256_bytes(receipt_raw),
        identities=identities,
        writer_config=writer_artifact,
        gateway_config=gateway_artifact,
        edge_config=edge_artifact,
        e2e_fixture=fixture_artifact,
        host_identity_receipt=host_artifact,
    )
    _bootstrap_path, _bootstrap_raw, bootstrap_sha256 = _validated_release_file(
        preliminary_plan,
        DEFAULT_CANARY_BOOTSTRAP_SQL_RELATIVE,
        maximum_bytes=MAX_FIXED_CANARY_ARTIFACT_BYTES,
    )
    _retire_path, _retire_raw, retire_sha256 = _validated_release_file(
        preliminary_plan,
        DEFAULT_CANARY_BOOTSTRAP_RETIRE_SQL_RELATIVE,
        maximum_bytes=MAX_FIXED_CANARY_ARTIFACT_BYTES,
    )
    value = build_coordinator_input(
        writer_activation_plan=writer_plan,
        writer_activation_receipt=writer_receipt,
        writer_activation_receipt_file_sha256=_sha256_bytes(receipt_raw),
        identities=identities,
        writer_config=writer_config,
        gateway_config=gateway_artifact,
        edge_config=edge_artifact,
        e2e_fixture=fixture_artifact,
        host_identity_receipt=host_artifact,
        bootstrap_sql_sha256=bootstrap_sha256,
        bootstrap_retire_sql_sha256=retire_sha256,
    )
    _validate_coordinator_input_live(value)
    return value


def build_coordinator_input(
    *,
    writer_activation_plan: ActivationPlan,
    writer_activation_receipt: Mapping[str, Any],
    writer_activation_receipt_file_sha256: str,
    identities: FullCanaryIdentities,
    writer_config: Mapping[str, Any],
    gateway_config: ExactArtifact,
    edge_config: ExactArtifact,
    e2e_fixture: ExactArtifact,
    host_identity_receipt: ExactArtifact,
    bootstrap_sql_sha256: str,
    bootstrap_retire_sql_sha256: str,
) -> CoordinatorInput:
    """Build the exact secret-free input after writer-only success."""

    unsigned = {
        "schema": COORDINATOR_INPUT_SCHEMA,
        "revision": writer_activation_plan.revision,
        "writer_activation_plan": writer_activation_plan.to_mapping(),
        "writer_activation_receipt": copy.deepcopy(dict(writer_activation_receipt)),
        "writer_activation_receipt_file_sha256": writer_activation_receipt_file_sha256,
        "identities": identities.to_mapping(),
        "writer_config": copy.deepcopy(dict(writer_config)),
        "artifacts": {
            "gateway_config": gateway_config.to_mapping(),
            "edge_config": edge_config.to_mapping(),
            "e2e_fixture": e2e_fixture.to_mapping(),
            "host_identity_receipt": host_identity_receipt.to_mapping(),
        },
        "bootstrap_sql_sha256": bootstrap_sql_sha256,
        "bootstrap_retire_sql_sha256": bootstrap_retire_sql_sha256,
    }
    return CoordinatorInput.from_mapping({
        **unsigned,
        "coordinator_input_sha256": _sha256_json(unsigned),
    })


_COORDINATOR_INPUT_PUBLICATION_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "coordinator_input_path",
    "coordinator_input_file_sha256",
    "publication_receipt_path",
    "owner_uid",
    "group_gid",
    "mode",
    "published_at_unix",
    "receipt_sha256",
})


def _parse_coordinator_input_publication_receipt(
    value: Any,
    *,
    coordinator_input: CoordinatorInput,
    payload: bytes,
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _COORDINATOR_INPUT_PUBLICATION_FIELDS
    ):
        _fail("coordinator_input_publication_receipt_invalid")
    raw = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    if (
        raw["schema"] != COORDINATOR_INPUT_PUBLICATION_SCHEMA
        or raw["ok"] is not True
        or raw["state"] != "published"
        or raw["release_sha"] != coordinator_input.revision
        or raw["coordinator_input_sha256"] != coordinator_input.sha256
        or raw["coordinator_input_path"] != str(COORDINATOR_INPUT_PATH)
        or raw["coordinator_input_file_sha256"] != _sha256_bytes(payload)
        or raw["publication_receipt_path"]
        != str(COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH)
        or raw["owner_uid"] != 0
        or raw["group_gid"] != 0
        or raw["mode"] != "0400"
        or type(raw["published_at_unix"]) is not int
        or raw["published_at_unix"] < 0
        or raw["receipt_sha256"] != _sha256_json(unsigned)
    ):
        _fail("coordinator_input_publication_receipt_invalid")
    return raw


def publish_coordinator_input(
    value: CoordinatorInput,
    *,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Durably no-replace publish input and its retry-safe terminal receipt."""

    _require_root_linux()
    if not isinstance(value, CoordinatorInput):
        raise TypeError("coordinator input is required")
    observed_at = int(time.time()) if now_unix is None else now_unix
    if type(observed_at) is not int or observed_at < 0:
        _fail("coordinator_input_publication_clock_invalid")
    payload = _canonical_bytes(value.value)
    with _lifecycle_lock():
        if os.path.lexists(COORDINATOR_PROCESS_LEASE_PATH):
            _fail("coordinator_process_recovery_required", phase="input_bootstrap")
        if not _services_are_exactly_stopped_and_disabled():
            _fail("coordinator_input_bootstrap_services_not_stopped")
        _validate_coordinator_input_live(value)
        input_snapshot = _capture_root_snapshot(COORDINATOR_INPUT_PATH)
        receipt_snapshot = _capture_root_snapshot(
            COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH,
            maximum=MAX_OWNER_APPROVAL_BYTES,
        )
        if receipt_snapshot is not None and input_snapshot is None:
            _fail("coordinator_input_publication_orphan_receipt")
        if input_snapshot is None:
            _publish_root_payload(
                COORDINATOR_INPUT_PATH,
                payload,
                expected_previous_sha256=None,
            )
        elif input_snapshot.raw != payload:
            _fail("coordinator_input_publication_input_drifted")
        if receipt_snapshot is not None:
            receipt_value = _decode_mapping(
                receipt_snapshot.raw,
                code="coordinator_input_publication_receipt_invalid",
            )
            return _parse_coordinator_input_publication_receipt(
                receipt_value,
                coordinator_input=value,
                payload=payload,
            )
        unsigned = {
            "schema": COORDINATOR_INPUT_PUBLICATION_SCHEMA,
            "ok": True,
            "state": "published",
            "release_sha": value.revision,
            "coordinator_input_sha256": value.sha256,
            "coordinator_input_path": str(COORDINATOR_INPUT_PATH),
            "coordinator_input_file_sha256": _sha256_bytes(payload),
            "publication_receipt_path": str(COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH),
            "owner_uid": 0,
            "group_gid": 0,
            "mode": "0400",
            "published_at_unix": observed_at,
        }
        terminal_receipt = {
            **unsigned,
            "receipt_sha256": _sha256_json(unsigned),
        }
        _parse_coordinator_input_publication_receipt(
            terminal_receipt,
            coordinator_input=value,
            payload=payload,
        )
        _publish_root_payload(
            COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH,
            _canonical_bytes(terminal_receipt),
            expected_previous_sha256=None,
        )
        return terminal_receipt


def collect_and_publish_coordinator_input() -> Mapping[str, Any]:
    """Packaged fixed-path bootstrap; accepts no operator-selected inputs."""

    coordinator_input = collect_coordinator_input()
    _attest_current_cli_process(
        coordinator_input,
        command="publish-coordinator-input",
    )
    return publish_coordinator_input(coordinator_input)


def _validate_coordinator_input_live(value: CoordinatorInput) -> None:
    installed = load_activation_plan(WRITER_ACTIVATION_PLAN_PATH)
    if installed.to_mapping() != value.writer_activation_plan.to_mapping():
        _fail("coordinator_input_writer_plan_drifted")
    expected_receipt_path = _success_receipt_path(
        value.writer_activation_plan,
        create_parent=False,
    )
    receipt_raw = _stable_root_read(
        expected_receipt_path,
        maximum=MAX_COORDINATOR_INPUT_BYTES,
    )
    receipt_value = _decode_mapping(
        receipt_raw,
        code="coordinator_input_writer_receipt_file_invalid",
    )
    if (
        receipt_value != value.value["writer_activation_receipt"]
        or _sha256_bytes(receipt_raw)
        != value.value["writer_activation_receipt_file_sha256"]
    ):
        _fail("coordinator_input_writer_receipt_drifted")
    for name, artifact in value.artifacts.items():
        raw = _read_exact_artifact(artifact, label=name)
        if name == "gateway_config":
            _validate_gateway_config(raw)
        elif name == "edge_config":
            _validate_edge_config(raw, value.identities)
    _validated_e2e_fixture(value.base_plan)
    validate_dedicated_canary_host(value.base_plan)
    manifest_path = Path(value.base_plan.release["manifest_path"])
    manifest_raw = _stable_root_read(
        manifest_path,
        maximum=MAX_RELEASE_MANIFEST_BYTES,
    )
    if _sha256_bytes(manifest_raw) != value.base_plan.release["manifest_file_sha256"]:
        _fail("coordinator_release_manifest_digest_drifted")
    manifest = _decode_mapping(
        manifest_raw,
        code="coordinator_release_manifest_invalid",
    )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        _fail("coordinator_release_manifest_entries_invalid")
    expected = {
        DEFAULT_CANARY_BOOTSTRAP_SQL_RELATIVE.as_posix(): value.value[
            "bootstrap_sql_sha256"
        ],
        DEFAULT_CANARY_BOOTSTRAP_RETIRE_SQL_RELATIVE.as_posix(): value.value[
            "bootstrap_retire_sql_sha256"
        ],
    }
    observed = {
        entry.get("path"): entry.get("sha256")
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("path") in expected
    }
    if observed != expected:
        _fail("coordinator_bootstrap_manifest_binding_invalid")


def load_coordinator_input(
    path: Path = COORDINATOR_INPUT_PATH,
) -> CoordinatorInput:
    _require_root_linux()
    if path != COORDINATOR_INPUT_PATH:
        _fail("coordinator_input_path_not_fixed")
    raw = _stable_root_read(path, maximum=MAX_COORDINATOR_INPUT_BYTES)
    value = CoordinatorInput.from_mapping(
        _decode_mapping(raw, code="coordinator_input_json_invalid")
    )
    _validate_coordinator_input_live(value)
    return value


def load_credential_prepare_approval(
    coordinator_input: CoordinatorInput,
    *,
    path: Path = CREDENTIAL_PREPARE_APPROVAL_PATH,
    now_unix: int | None = None,
) -> CredentialPrepareApproval:
    _require_root_linux()
    if path != CREDENTIAL_PREPARE_APPROVAL_PATH:
        _fail("credential_prepare_approval_path_not_fixed")
    raw = _stable_root_read(
        DISCORD_TOKEN_INSTALL_APPROVAL_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    approval = CredentialPrepareApproval.from_mapping(
        _decode_mapping(raw, code="credential_prepare_approval_json_invalid")
    )
    approval.require(
        coordinator_input=coordinator_input,
        now_unix=int(time.time()) if now_unix is None else now_unix,
    )
    return approval


def _load_recovery_credential_approval(
    coordinator_input: CoordinatorInput,
) -> CredentialPrepareApproval:
    """Load an exact historical approval without treating expiry as drift."""

    raw = _stable_root_read(
        CREDENTIAL_PREPARE_APPROVAL_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    approval = CredentialPrepareApproval.from_mapping(
        _decode_mapping(
            raw,
            code="credential_prepare_approval_json_invalid",
        )
    )
    scope = coordinator_input.value["writer_config"].get("canary_scope_preapproval")
    if (
        approval.value["coordinator_input_sha256"] != coordinator_input.sha256
        or approval.value["release_sha"] != coordinator_input.revision
        or not isinstance(scope, Mapping)
        or approval.value["approval_source_sha256"]
        != scope.get("approval_source_sha256")
    ):
        _fail(
            "recovery_credential_prepare_approval_not_bound",
            phase="recovery_preflight",
        )
    return approval


def load_discord_token_install_approval(
    coordinator_input: CoordinatorInput,
    *,
    path: Path = DISCORD_TOKEN_INSTALL_APPROVAL_PATH,
    now_unix: int | None = None,
) -> DiscordTokenInstallApproval:
    _require_root_linux()
    if path != DISCORD_TOKEN_INSTALL_APPROVAL_PATH:
        _fail("discord_token_install_approval_path_not_fixed")
    approval = _load_bound_discord_token_install_approval(coordinator_input)
    approval.require(
        coordinator_input=coordinator_input,
        now_unix=int(time.time()) if now_unix is None else now_unix,
    )
    return approval


def _load_bound_discord_token_install_approval(
    coordinator_input: CoordinatorInput,
) -> DiscordTokenInstallApproval:
    """Load the immutable approval binding without reusing its install TTL.

    The short approval window authorizes secret installation.  A durable
    retirement journal may need owner recovery later, so cleanup uses the
    same fixed approval only as an identity/source binding and does not turn
    expiry into an unretirable secret.
    """

    _require_root_linux()
    raw = _stable_root_read(
        DISCORD_TOKEN_INSTALL_APPROVAL_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    approval = DiscordTokenInstallApproval.from_mapping(
        _decode_mapping(raw, code="discord_token_install_approval_json_invalid")
    )
    scope = coordinator_input.value["writer_config"].get("canary_scope_preapproval")
    if (
        approval.value["coordinator_input_sha256"] != coordinator_input.sha256
        or approval.value["release_sha"] != coordinator_input.revision
        or not isinstance(scope, Mapping)
        or approval.value["approval_source_sha256"]
        != scope.get("approval_source_sha256")
    ):
        _fail("discord_token_install_approval_not_bound")
    return approval


def load_discord_token_install_receipt(
    coordinator_input: CoordinatorInput,
    *,
    require_token: bool = True,
) -> tuple[Mapping[str, Any], os.stat_result | None, _RootFileSnapshot]:
    """Load the secret-free token lease and bind it to the exact inode."""

    _require_root_linux()
    snapshot = _capture_root_snapshot(
        DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    if snapshot is None:
        _fail("discord_token_install_receipt_missing")
    receipt = _decode_mapping(
        snapshot.raw,
        code="discord_token_install_receipt_invalid",
    )
    fields = {
        "schema",
        "ok",
        "release_sha",
        "coordinator_input_sha256",
        "discord_token_install_approval_sha256",
        "owner_subject_sha256",
        "token_path",
        "device",
        "inode",
        "owner_uid",
        "group_gid",
        "mode",
        "size",
        "link_count",
        "content_or_digest_recorded",
        "installed_at_unix",
        "receipt_sha256",
    }
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        set(receipt) != fields
        or receipt.get("schema") != DISCORD_TOKEN_INSTALL_RECEIPT_SCHEMA
        or receipt.get("ok") is not True
        or receipt.get("release_sha") != coordinator_input.revision
        or receipt.get("coordinator_input_sha256") != coordinator_input.sha256
        or receipt.get("owner_subject_sha256") is None
        or receipt.get("token_path") != str(DISCORD_TOKEN_PATH)
        or receipt.get("owner_uid") != coordinator_input.identities.edge_uid
        or receipt.get("group_gid") != coordinator_input.identities.edge_gid
        or receipt.get("mode") != "0400"
        or receipt.get("link_count") != 1
        or receipt.get("content_or_digest_recorded") is not False
        or type(receipt.get("installed_at_unix")) is not int
        or receipt.get("receipt_sha256") != _sha256_json(unsigned)
    ):
        _fail("discord_token_install_receipt_invalid")
    approval = _load_bound_discord_token_install_approval(coordinator_input)
    if (
        receipt["discord_token_install_approval_sha256"] != approval.sha256
        or receipt["owner_subject_sha256"] != approval.value["owner_subject_sha256"]
    ):
        _fail("discord_token_install_receipt_approval_drifted")
    installed = None
    if os.path.lexists(DISCORD_TOKEN_PATH):
        installed = _validate_secret_metadata(
            DISCORD_TOKEN_PATH,
            uid=coordinator_input.identities.edge_uid,
            gid=coordinator_input.identities.edge_gid,
            maximum=MAX_DISCORD_TOKEN_BYTES,
        )
    elif require_token:
        _fail("discord_token_install_receipt_token_missing")
    if installed is not None and (
        receipt["device"] != installed.st_dev
        or receipt["inode"] != installed.st_ino
        or receipt["size"] != installed.st_size
    ):
        _fail("discord_token_install_receipt_inode_drifted")
    return receipt, installed, snapshot


_DISCORD_INSTALL_JOURNAL_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "discord_token_install_approval_sha256",
    "owner_subject_sha256",
    "token_path",
    "staging_path",
    "device",
    "inode",
    "owner_uid",
    "group_gid",
    "mode",
    "size",
    "content_or_digest_recorded",
    "prepared_at_unix",
    "receipt_sha256",
})


@dataclass(frozen=True)
class _DiscordInstallState:
    value: Mapping[str, Any]
    snapshot: _RootFileSnapshot
    state: str
    token_item: os.stat_result | None
    stage_item: os.stat_result | None

    @property
    def receipt_sha256(self) -> str:
        return str(self.value["receipt_sha256"])

    @property
    def owner_subject_sha256(self) -> str:
        return str(self.value["owner_subject_sha256"])

    @property
    def device(self) -> int | None:
        value = self.value.get("device")
        if type(value) is int:
            return value
        item = self.token_item or self.stage_item
        return None if item is None else item.st_dev

    @property
    def inode(self) -> int | None:
        value = self.value.get("inode")
        if type(value) is int:
            return value
        item = self.token_item or self.stage_item
        return None if item is None else item.st_ino


def _discord_install_journal(
    *,
    coordinator_input: CoordinatorInput,
    approval: DiscordTokenInstallApproval,
    state: str,
    prepared_at_unix: int,
    device: int | None,
    inode: int | None,
    size: int | None,
) -> Mapping[str, Any]:
    unsigned = {
        "schema": DISCORD_TOKEN_INSTALL_JOURNAL_SCHEMA,
        "ok": False,
        "state": state,
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "discord_token_install_approval_sha256": approval.sha256,
        "owner_subject_sha256": approval.value["owner_subject_sha256"],
        "token_path": str(DISCORD_TOKEN_PATH),
        "staging_path": str(DISCORD_TOKEN_STAGE_PATH),
        "device": device,
        "inode": inode,
        "owner_uid": coordinator_input.identities.edge_uid,
        "group_gid": coordinator_input.identities.edge_gid,
        "mode": "0400",
        "size": size,
        "content_or_digest_recorded": False,
        "prepared_at_unix": prepared_at_unix,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _parse_discord_install_journal(
    value: Any,
    *,
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _DISCORD_INSTALL_JOURNAL_FIELDS:
        _fail("discord_token_install_journal_invalid")
    raw = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    state = raw["state"]
    if (
        raw["schema"] != DISCORD_TOKEN_INSTALL_JOURNAL_SCHEMA
        or raw["ok"] is not False
        or state not in {"install_intent", "stage_allocated", "secret_staged"}
        or raw["release_sha"] != coordinator_input.revision
        or raw["coordinator_input_sha256"] != coordinator_input.sha256
        or raw["token_path"] != str(DISCORD_TOKEN_PATH)
        or raw["staging_path"] != str(DISCORD_TOKEN_STAGE_PATH)
        or raw["owner_uid"] != coordinator_input.identities.edge_uid
        or raw["group_gid"] != coordinator_input.identities.edge_gid
        or raw["mode"] != "0400"
        or raw["content_or_digest_recorded"] is not False
        or type(raw["prepared_at_unix"]) is not int
        or raw["prepared_at_unix"] < 0
        or raw["receipt_sha256"] != _sha256_json(unsigned)
    ):
        _fail("discord_token_install_journal_invalid")
    _digest(
        raw["discord_token_install_approval_sha256"],
        "discord_token_install_journal_approval_invalid",
    )
    _digest(
        raw["owner_subject_sha256"],
        "discord_token_install_journal_owner_invalid",
    )
    if state == "install_intent":
        if any(raw[name] is not None for name in ("device", "inode", "size")):
            _fail("discord_token_install_intent_invalid")
    elif (
        type(raw["device"]) is not int
        or raw["device"] <= 0
        or type(raw["inode"]) is not int
        or raw["inode"] <= 0
        or type(raw["size"]) is not int
        or not 0 <= raw["size"] <= MAX_DISCORD_TOKEN_BYTES
        or (state == "stage_allocated" and raw["size"] != 0)
        or (state == "secret_staged" and raw["size"] < MIN_DISCORD_TOKEN_BYTES)
    ):
        _fail("discord_token_install_stage_journal_invalid")
    approval = _load_bound_discord_token_install_approval(coordinator_input)
    if (
        raw["discord_token_install_approval_sha256"] != approval.sha256
        or raw["owner_subject_sha256"] != approval.value["owner_subject_sha256"]
    ):
        _fail("discord_token_install_journal_approval_drifted")
    return raw


def _load_discord_install_state(
    coordinator_input: CoordinatorInput,
    *,
    require_terminal_token: bool = True,
) -> _DiscordInstallState:
    snapshot = _capture_root_snapshot(
        DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    if snapshot is None:
        _fail("discord_token_install_state_missing")
    value = _decode_mapping(
        snapshot.raw,
        code="discord_token_install_state_invalid",
    )
    if value.get("schema") == DISCORD_TOKEN_INSTALL_RECEIPT_SCHEMA:
        receipt, token_item, same_snapshot = load_discord_token_install_receipt(
            coordinator_input,
            require_token=require_terminal_token,
        )
        if snapshot.raw != same_snapshot.raw or not _same_file_identity(
            snapshot.item, same_snapshot.item
        ):
            _fail("discord_token_install_terminal_receipt_drifted")
        if os.path.lexists(DISCORD_TOKEN_STAGE_PATH):
            _fail("discord_token_install_terminal_stage_survived")
        return _DiscordInstallState(
            value=receipt,
            snapshot=same_snapshot,
            state="installed",
            token_item=token_item,
            stage_item=None,
        )
    journal = _parse_discord_install_journal(
        value,
        coordinator_input=coordinator_input,
    )
    token_item = (
        DISCORD_TOKEN_PATH.lstat() if os.path.lexists(DISCORD_TOKEN_PATH) else None
    )
    stage_item = (
        DISCORD_TOKEN_STAGE_PATH.lstat()
        if os.path.lexists(DISCORD_TOKEN_STAGE_PATH)
        else None
    )
    for item in (token_item, stage_item):
        if item is None:
            continue
        if (
            not stat.S_ISREG(item.st_mode)
            or stat.S_ISLNK(item.st_mode)
            or item.st_uid not in {0, coordinator_input.identities.edge_uid}
            or item.st_gid not in {0, coordinator_input.identities.edge_gid}
            or stat.S_IMODE(item.st_mode) & 0o077
            or not 0 <= item.st_size <= MAX_DISCORD_TOKEN_BYTES
        ):
            _fail("discord_token_install_recovery_artifact_invalid")
    state = str(journal["state"])
    if state == "install_intent":
        if token_item is not None or (
            stage_item is not None and stage_item.st_size != 0
        ):
            _fail("discord_token_install_intent_state_drifted")
    else:
        expected = (journal["device"], journal["inode"])
        if token_item is None and stage_item is None:
            pass
        elif any(
            (item.st_dev, item.st_ino) != expected
            or item.st_uid != coordinator_input.identities.edge_uid
            or item.st_gid != coordinator_input.identities.edge_gid
            or stat.S_IMODE(item.st_mode) != 0o400
            for item in (token_item, stage_item)
            if item is not None
        ):
            _fail("discord_token_install_stage_identity_drifted")
        if state == "stage_allocated" and token_item is not None:
            _fail("discord_token_install_allocated_target_unexpected")
        if state == "secret_staged" and any(
            item.st_size != journal["size"]
            for item in (token_item, stage_item)
            if item is not None
        ):
            _fail("discord_token_install_staged_size_drifted")
        if token_item is not None and stage_item is not None:
            if (
                (token_item.st_dev, token_item.st_ino)
                != (stage_item.st_dev, stage_item.st_ino)
                or token_item.st_nlink != 2
                or stage_item.st_nlink != 2
            ):
                _fail("discord_token_install_link_state_invalid")
        elif any(
            item.st_nlink != 1 for item in (token_item, stage_item) if item is not None
        ):
            _fail("discord_token_install_link_count_invalid")
    return _DiscordInstallState(
        value=journal,
        snapshot=snapshot,
        state=state,
        token_item=token_item,
        stage_item=stage_item,
    )


def _load_owner_approval_request_live(
    *,
    now_unix: int | None = None,
) -> tuple[
    OwnerApprovalRequest,
    FullCanaryPlan,
    CoordinatorInput,
    CredentialPrepareApproval,
]:
    _require_root_linux()
    current_time = int(time.time()) if now_unix is None else now_unix
    raw = _stable_root_read(
        OWNER_APPROVAL_REQUEST_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    request = OwnerApprovalRequest.from_mapping(
        _decode_mapping(raw, code="owner_approval_request_json_invalid")
    )
    coordinator_input = load_coordinator_input()
    credential_approval = load_credential_prepare_approval(
        coordinator_input,
        now_unix=current_time,
    )
    if (
        request.value["release_sha"] != coordinator_input.revision
        or request.value["coordinator_input_sha256"] != coordinator_input.sha256
        or request.value["credential_prepare_approval_sha256"]
        != credential_approval.sha256
        or request.value["owner_subject_sha256"]
        != credential_approval.value["owner_subject_sha256"]
        or request.value["ephemeral_admin_username"]
        != derive_ephemeral_admin_username(credential_approval.sha256)
        or not request.value["requested_at_unix"]
        <= current_time
        <= request.value["approval_deadline_unix"]
    ):
        _fail("owner_approval_request_not_active_or_bound")
    plan_raw = _stable_root_read(
        DEFAULT_STAGED_PLAN_PATH,
        maximum=MAX_COORDINATOR_INPUT_BYTES,
    )
    if _sha256_bytes(plan_raw) != request.value["staged_plan_file_sha256"]:
        _fail("owner_approval_request_staged_plan_drifted")
    try:
        plan = FullCanaryPlan.from_mapping(
            _decode_mapping(plan_raw, code="staged_full_canary_plan_invalid")
        )
    except (TypeError, ValueError) as exc:
        raise CoordinatorError("staged_full_canary_plan_invalid") from exc
    if (
        plan.sha256 != request.value["full_canary_plan_sha256"]
        or plan.revision != coordinator_input.revision
    ):
        _fail("owner_approval_request_plan_binding_invalid")
    validate_dedicated_canary_host(plan)
    prior = _capture_root_snapshot(
        DEFAULT_APPROVAL_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    prior_sha256 = None if prior is None else prior.sha256
    if prior_sha256 != request.value["prior_approval_file_sha256"]:
        _fail("owner_approval_request_prior_approval_drifted")
    return request, plan, coordinator_input, credential_approval


def _final_approval_hard_deadline(
    *,
    hba_expires_at_unix: int,
    fixture_expires_at_unix: int,
    credential_approval_expires_at_unix: int,
) -> int:
    if any(
        type(item) is not int or item < 0
        for item in (
            hba_expires_at_unix,
            fixture_expires_at_unix,
            credential_approval_expires_at_unix,
        )
    ):
        _fail("owner_approval_request_authority_deadline_invalid")
    return min(
        hba_expires_at_unix - HBA_EXPIRY_SAFETY_MARGIN_SECONDS,
        fixture_expires_at_unix,
        credential_approval_expires_at_unix - FINAL_APPROVAL_TRANSMIT_MARGIN_SECONDS,
    )


def build_owner_approval_request(
    *,
    coordinator_input: CoordinatorInput,
    credential_approval: CredentialPrepareApproval,
    plan: FullCanaryPlan,
    staged_plan_publication: _RootPublication,
    hba_receipt: ManagedCloudSQLAdminHBAReceipt,
    fixture: Mapping[str, Any],
    approval_source_sha256: str,
    now_unix: int | None = None,
) -> tuple[OwnerApprovalRequest, _RootPublication]:
    """Publish and return the exact active final-approval request."""

    _require_root_linux()
    current_time = int(time.time()) if now_unix is None else now_unix
    if (
        not isinstance(coordinator_input, CoordinatorInput)
        or not isinstance(credential_approval, CredentialPrepareApproval)
        or not isinstance(plan, FullCanaryPlan)
        or staged_plan_publication.path != DEFAULT_STAGED_PLAN_PATH
        or staged_plan_publication.after.raw != _canonical_bytes(plan.to_mapping())
        or not isinstance(hba_receipt, ManagedCloudSQLAdminHBAReceipt)
    ):
        _fail("owner_approval_request_inputs_invalid")
    credential_approval.require(
        coordinator_input=coordinator_input,
        now_unix=current_time,
    )
    approval_source_sha256 = _digest(
        approval_source_sha256,
        "owner_approval_request_approval_source_invalid",
    )
    if approval_source_sha256 != credential_approval.value["approval_source_sha256"]:
        _fail("owner_approval_request_approval_source_drifted")
    fixture_expiry_ms = fixture.get("valid_until_unix_ms")
    if type(fixture_expiry_ms) is not int or fixture_expiry_ms <= 0:
        _fail("owner_approval_request_fixture_expiry_invalid")
    fixture_expiry = fixture_expiry_ms // 1000
    hard_deadline = _final_approval_hard_deadline(
        hba_expires_at_unix=hba_receipt.expires_at_unix,
        fixture_expires_at_unix=fixture_expiry,
        credential_approval_expires_at_unix=credential_approval.value[
            "expires_at_unix"
        ],
    )
    max_wait = min(
        FINAL_APPROVAL_MAX_WAIT_SECONDS,
        hard_deadline - current_time,
    )
    if max_wait <= FINAL_APPROVAL_TRANSMIT_MARGIN_SECONDS:
        _fail("owner_approval_request_window_exhausted")
    approval_deadline = current_time + max_wait
    prior_approval = _capture_root_snapshot(
        DEFAULT_APPROVAL_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    unsigned = {
        "schema": OWNER_APPROVAL_REQUEST_SCHEMA,
        "ok": True,
        "state": "awaiting_final_owner_approval",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "credential_prepare_approval_sha256": credential_approval.sha256,
        "owner_subject_sha256": credential_approval.value["owner_subject_sha256"],
        "approval_source_sha256": approval_source_sha256,
        "ephemeral_admin_username": derive_ephemeral_admin_username(
            credential_approval.sha256
        ),
        "full_canary_plan_sha256": plan.sha256,
        "staged_plan_path": str(DEFAULT_STAGED_PLAN_PATH),
        "staged_plan_file_sha256": staged_plan_publication.after.sha256,
        "approval_request_path": str(OWNER_APPROVAL_REQUEST_PATH),
        "approval_path": str(DEFAULT_APPROVAL_PATH),
        "hba_receipt_sha256": hba_receipt.sha256,
        "hba_expires_at_unix": hba_receipt.expires_at_unix,
        "fixture_expires_at_unix": fixture_expiry,
        "credential_approval_expires_at_unix": credential_approval.value[
            "expires_at_unix"
        ],
        "requested_at_unix": current_time,
        "approval_deadline_unix": approval_deadline,
        "owner_input_cutoff_unix": (
            approval_deadline - FINAL_APPROVAL_TRANSMIT_MARGIN_SECONDS
        ),
        "final_approval_transmit_margin_seconds": (
            FINAL_APPROVAL_TRANSMIT_MARGIN_SECONDS
        ),
        "max_wait_seconds": max_wait,
        "prior_approval_file_sha256": (
            None if prior_approval is None else prior_approval.sha256
        ),
        "final_approval_frame_schema": FINAL_APPROVAL_FRAME_SCHEMA,
    }
    request = OwnerApprovalRequest.from_mapping({
        **unsigned,
        "request_sha256": _sha256_json(unsigned),
    })
    previous_request = _capture_root_snapshot(
        OWNER_APPROVAL_REQUEST_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    expected_previous: str | None = None
    if previous_request is not None:
        try:
            old = OwnerApprovalRequest.from_mapping(
                _decode_mapping(
                    previous_request.raw,
                    code="prior_owner_approval_request_invalid",
                )
            )
        except CoordinatorError as exc:
            raise CoordinatorCleanupBlocked(
                "prior_owner_approval_request_cleanup_blocked",
                phase="owner_approval_request",
            ) from exc
        if current_time <= old.value["approval_deadline_unix"]:
            _fail("owner_approval_request_already_active")
        expected_previous = previous_request.sha256
    publication = _publish_root_payload(
        OWNER_APPROVAL_REQUEST_PATH,
        _canonical_bytes(request.value),
        expected_previous_sha256=expected_previous,
    )
    return request, publication


def _approval_path_snapshot_unchanged(
    before: _RootFileSnapshot | None,
    after: _RootFileSnapshot | None,
) -> bool:
    if before is None or after is None:
        return before is None and after is None
    return bool(
        before.raw == after.raw and _same_file_identity(before.item, after.item)
    )


def _final_approval_cancel_has_state_conflict(
    *,
    request_state: str,
    staged_state: str,
    owner_state: str,
) -> bool:
    request_and_plan_are_coherent = (
        request_state in {"matching_active", "matching_expired"}
        and staged_state == "matching_present"
    ) or (request_state == "retired_absent" and staged_state == "retired_absent")
    return not request_and_plan_are_coherent or owner_state != "matching_prior"


def _parse_final_approval_cancel_receipt(
    value: Any,
    *,
    request: OwnerApprovalRequest,
    plan: FullCanaryPlan,
    coordinator_input: CoordinatorInput,
    credential_approval: CredentialPrepareApproval,
    approval_request_before: _RootFileSnapshot,
    staged_plan_before: _RootFileSnapshot,
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != FINAL_APPROVAL_CANCEL_RECEIPT_FIELDS
    ):
        _fail("final_approval_cancel_receipt_fields_invalid")
    raw = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    prior = request.value["prior_approval_file_sha256"]
    request_state = raw["approval_request_artifact_state"]
    staged_state = raw["staged_plan_artifact_state"]
    owner_state = raw["owner_approval_artifact_state"]
    conflict = _final_approval_cancel_has_state_conflict(
        request_state=request_state,
        staged_state=staged_state,
        owner_state=owner_state,
    )
    if (
        raw["schema"] != FINAL_APPROVAL_CANCEL_RECEIPT_SCHEMA
        or raw["ok"] is not False
        or raw["state"]
        != ("cancelled_no_secret_state_conflict" if conflict else "cancelled_no_secret")
        or raw["reason"] != "eof_before_mfa1"
        or raw["release_sha"] != coordinator_input.revision
        or raw["coordinator_input_sha256"] != coordinator_input.sha256
        or raw["credential_prepare_approval_sha256"] != credential_approval.sha256
        or raw["owner_subject_sha256"] != request.value["owner_subject_sha256"]
        or raw["full_canary_plan_sha256"] != plan.sha256
        or raw["approval_request_sha256"] != request.sha256
        or raw["approval_request_path"] != str(OWNER_APPROVAL_REQUEST_PATH)
        or raw["expected_approval_request_file_sha256"]
        != approval_request_before.sha256
        or request_state
        not in {
            "matching_active",
            "matching_expired",
            "retired_absent",
            "superseded",
            "drifted",
        }
        or raw["approval_request_present"] is not (request_state != "retired_absent")
        or raw["approval_request_remains_active"]
        is not (request_state == "matching_active")
        or raw["staged_plan_path"] != str(DEFAULT_STAGED_PLAN_PATH)
        or raw["expected_staged_plan_file_sha256"] != staged_plan_before.sha256
        or staged_state
        not in {"matching_present", "retired_absent", "superseded", "drifted"}
        or raw["staged_plan_present"] is not (staged_state != "retired_absent")
        or raw["approval_path"] != str(DEFAULT_APPROVAL_PATH)
        or raw["prior_approval_file_sha256"] != prior
        or owner_state not in {"matching_prior", "drifted"}
        or raw["approval_path_matches_prior"] is not (owner_state == "matching_prior")
        or raw["new_owner_approval_installed"]
        is not (False if owner_state == "matching_prior" else None)
        or raw["frame_bytes_received"] != 0
        or raw["owner_approval_mutation_performed_by_this_helper"] is not False
        or type(raw["cancelled_at_unix"]) is not int
        or raw["cancelled_at_unix"] < request.value["requested_at_unix"]
        or raw["receipt_sha256"] != _sha256_json(unsigned)
    ):
        _fail("final_approval_cancel_receipt_invalid")
    for name in (
        "observed_approval_request_file_sha256",
        "observed_staged_plan_file_sha256",
        "observed_approval_file_sha256",
    ):
        if raw[name] is not None:
            _digest(raw[name], "final_approval_cancel_observed_digest_invalid")
    if (
        (request_state == "retired_absent")
        != (raw["observed_approval_request_file_sha256"] is None)
        or (staged_state == "retired_absent")
        != (raw["observed_staged_plan_file_sha256"] is None)
        or (
            request_state in {"matching_active", "matching_expired"}
            and raw["observed_approval_request_file_sha256"]
            != raw["expected_approval_request_file_sha256"]
        )
        or (
            request_state == "superseded"
            and raw["observed_approval_request_file_sha256"]
            == raw["expected_approval_request_file_sha256"]
        )
        or (
            request_state == "drifted"
            and raw["observed_approval_request_file_sha256"]
            != raw["expected_approval_request_file_sha256"]
        )
        or (
            request_state == "matching_active"
            and raw["cancelled_at_unix"] > request.value["approval_deadline_unix"]
        )
        or (
            request_state == "matching_expired"
            and raw["cancelled_at_unix"] <= request.value["approval_deadline_unix"]
        )
        or (
            staged_state == "matching_present"
            and raw["observed_staged_plan_file_sha256"]
            != raw["expected_staged_plan_file_sha256"]
        )
        or (
            staged_state == "superseded"
            and raw["observed_staged_plan_file_sha256"]
            == raw["expected_staged_plan_file_sha256"]
        )
        or (
            staged_state == "drifted"
            and raw["observed_staged_plan_file_sha256"]
            != raw["expected_staged_plan_file_sha256"]
        )
        or (
            owner_state == "matching_prior"
            and raw["observed_approval_file_sha256"] != prior
        )
        or (
            owner_state == "drifted"
            and prior is None
            and raw["observed_approval_file_sha256"] is None
        )
    ):
        _fail("final_approval_cancel_receipt_invalid")
    return raw


def _inspect_final_approval_request_artifact(
    *,
    expected: _RootFileSnapshot,
    observed: _RootFileSnapshot | None,
    now_unix: int,
    request: OwnerApprovalRequest,
) -> str:
    if observed is None:
        return "retired_absent"
    if _snapshot_is_exact(expected, observed):
        return (
            "matching_active"
            if now_unix <= request.value["approval_deadline_unix"]
            else "matching_expired"
        )
    if observed.raw == expected.raw:
        return "drifted"
    # Any different content is a superseding/conflicting replacement.  The
    # narrower ``drifted`` state is reserved for same bytes through a changed
    # inode identity (an ABA replacement), which the captured snapshot proves.
    return "superseded"


def _inspect_final_staged_plan_artifact(
    *,
    expected: _RootFileSnapshot,
    observed: _RootFileSnapshot | None,
) -> str:
    if observed is None:
        return "retired_absent"
    if _snapshot_is_exact(expected, observed):
        return "matching_present"
    if observed.raw == expected.raw:
        return "drifted"
    return "superseded"


def _build_final_approval_cancel_receipt(
    *,
    request: OwnerApprovalRequest,
    plan: FullCanaryPlan,
    coordinator_input: CoordinatorInput,
    credential_approval: CredentialPrepareApproval,
    approval_path_before: _RootFileSnapshot | None,
    approval_request_before: _RootFileSnapshot,
    staged_plan_before: _RootFileSnapshot,
) -> Mapping[str, Any]:
    approval_request_after = _capture_root_snapshot(
        OWNER_APPROVAL_REQUEST_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    staged_plan_after = _capture_root_snapshot(
        DEFAULT_STAGED_PLAN_PATH,
        maximum=MAX_COORDINATOR_INPUT_BYTES,
    )
    approval_path_after = _capture_root_snapshot(
        DEFAULT_APPROVAL_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    prior = request.value["prior_approval_file_sha256"]
    observed = None if approval_path_after is None else approval_path_after.sha256
    now_unix = int(time.time())
    request_state = _inspect_final_approval_request_artifact(
        expected=approval_request_before,
        observed=approval_request_after,
        now_unix=now_unix,
        request=request,
    )
    staged_state = _inspect_final_staged_plan_artifact(
        expected=staged_plan_before,
        observed=staged_plan_after,
    )
    owner_matches = (
        _approval_path_snapshot_unchanged(
            approval_path_before,
            approval_path_after,
        )
        and observed == prior
    )
    owner_state = "matching_prior" if owner_matches else "drifted"
    conflict = _final_approval_cancel_has_state_conflict(
        request_state=request_state,
        staged_state=staged_state,
        owner_state=owner_state,
    )
    unsigned = {
        "schema": FINAL_APPROVAL_CANCEL_RECEIPT_SCHEMA,
        "ok": False,
        "state": (
            "cancelled_no_secret_state_conflict" if conflict else "cancelled_no_secret"
        ),
        "reason": "eof_before_mfa1",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "credential_prepare_approval_sha256": credential_approval.sha256,
        "owner_subject_sha256": request.value["owner_subject_sha256"],
        "full_canary_plan_sha256": plan.sha256,
        "approval_request_sha256": request.sha256,
        "approval_request_path": str(OWNER_APPROVAL_REQUEST_PATH),
        "expected_approval_request_file_sha256": approval_request_before.sha256,
        "observed_approval_request_file_sha256": (
            None if approval_request_after is None else approval_request_after.sha256
        ),
        "approval_request_artifact_state": request_state,
        "approval_request_present": approval_request_after is not None,
        "approval_request_remains_active": request_state == "matching_active",
        "staged_plan_path": str(DEFAULT_STAGED_PLAN_PATH),
        "expected_staged_plan_file_sha256": staged_plan_before.sha256,
        "observed_staged_plan_file_sha256": (
            None if staged_plan_after is None else staged_plan_after.sha256
        ),
        "staged_plan_artifact_state": staged_state,
        "staged_plan_present": staged_plan_after is not None,
        "approval_path": str(DEFAULT_APPROVAL_PATH),
        "prior_approval_file_sha256": prior,
        "observed_approval_file_sha256": observed,
        "owner_approval_artifact_state": owner_state,
        "approval_path_matches_prior": owner_matches,
        "new_owner_approval_installed": False if owner_matches else None,
        "frame_bytes_received": 0,
        "owner_approval_mutation_performed_by_this_helper": False,
        "cancelled_at_unix": now_unix,
    }
    receipt = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
    return _parse_final_approval_cancel_receipt(
        receipt,
        request=request,
        plan=plan,
        coordinator_input=coordinator_input,
        credential_approval=credential_approval,
        approval_request_before=approval_request_before,
        staged_plan_before=staged_plan_before,
    )


def _final_owner_approval_is_bound_to_request(
    approval: FullCanaryOwnerApproval,
    request: OwnerApprovalRequest,
) -> bool:
    return bool(
        approval.value["plan_sha256"] == request.value["full_canary_plan_sha256"]
        and approval.value["owner_subject_sha256"]
        == request.value["owner_subject_sha256"]
        and approval.value["approval_source_sha256"]
        == request.value["approval_source_sha256"]
        and request.value["requested_at_unix"]
        <= approval.value["approved_at_unix"]
        <= request.value["owner_input_cutoff_unix"]
        and approval.value["expires_at_unix"]
        <= min(
            request.value["hba_expires_at_unix"],
            request.value["fixture_expires_at_unix"],
            request.value["credential_approval_expires_at_unix"],
        )
    )


def _require_install_final_owner_approval_binding(
    approval: FullCanaryOwnerApproval,
    request: OwnerApprovalRequest,
) -> None:
    if not _final_owner_approval_is_bound_to_request(approval, request):
        _fail(
            "final_owner_approval_not_bound",
            phase="final_approval_install",
        )


def _require_runtime_final_owner_approval_binding(
    approval: FullCanaryOwnerApproval,
    request: OwnerApprovalRequest,
) -> None:
    if not _final_owner_approval_is_bound_to_request(approval, request):
        _fail("owner_approval_ttl_or_binding_invalid")


def install_final_owner_approval(
    *,
    gate_emitter: Callable[[Mapping[str, Any]], None],
    frame_reader: Callable[[], Mapping[str, Any]] = _read_final_owner_approval_frame,
) -> Mapping[str, Any]:
    """Install one exact final approval through the fixed OOB MFA1 frame."""

    _harden_secret_process()
    request, plan, coordinator_input, credential_approval = (
        _load_owner_approval_request_live()
    )
    approval_path_before = _capture_root_snapshot(
        DEFAULT_APPROVAL_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    approval_request_before = _capture_root_snapshot(
        OWNER_APPROVAL_REQUEST_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    staged_plan_before = _capture_root_snapshot(
        DEFAULT_STAGED_PLAN_PATH,
        maximum=MAX_COORDINATOR_INPUT_BYTES,
    )
    if (
        None if approval_path_before is None else approval_path_before.sha256
    ) != request.value["prior_approval_file_sha256"]:
        _fail(
            "final_approval_path_drifted_before_gate",
            phase="final_approval_install",
        )
    if (
        approval_request_before is None
        or approval_request_before.raw != _canonical_bytes(request.value)
        or staged_plan_before is None
        or staged_plan_before.raw != _canonical_bytes(plan.to_mapping())
        or staged_plan_before.sha256 != request.value["staged_plan_file_sha256"]
    ):
        _fail(
            "final_approval_gate_artifact_drifted",
            phase="final_approval_install",
        )
    gate_emitter(request.value)
    try:
        approval_raw = frame_reader()
    except _FinalApprovalNoSecretCancellation:
        with _lifecycle_lock():
            return _build_final_approval_cancel_receipt(
                request=request,
                plan=plan,
                coordinator_input=coordinator_input,
                credential_approval=credential_approval,
                approval_path_before=approval_path_before,
                approval_request_before=approval_request_before,
                staged_plan_before=staged_plan_before,
            )
    try:
        approval = FullCanaryOwnerApproval.from_mapping(approval_raw)
    except (TypeError, ValueError) as exc:
        raise CoordinatorError(
            "final_owner_approval_contract_invalid",
            phase="final_approval_install",
        ) from exc
    with _lifecycle_lock():
        request_again, plan_again, input_again, credential_again = (
            _load_owner_approval_request_live()
        )
        approval_path_again = _capture_root_snapshot(
            DEFAULT_APPROVAL_PATH,
            maximum=MAX_OWNER_APPROVAL_BYTES,
        )
        now_unix = int(time.time())
        approval.require(plan_sha256=plan.sha256, now_unix=now_unix)
        if (
            request_again.value != request.value
            or plan_again.to_mapping() != plan.to_mapping()
            or input_again.sha256 != coordinator_input.sha256
            or credential_again.sha256 != credential_approval.sha256
            or not _approval_path_snapshot_unchanged(
                approval_path_before,
                approval_path_again,
            )
        ):
            _fail(
                "final_owner_approval_not_bound",
                phase="final_approval_install",
            )
        _require_install_final_owner_approval_binding(approval, request)
        installed_at_unix = now_unix
        unsigned = {
            "schema": FINAL_APPROVAL_INSTALL_RECEIPT_SCHEMA,
            "ok": True,
            "release_sha": coordinator_input.revision,
            "coordinator_input_sha256": coordinator_input.sha256,
            "credential_prepare_approval_sha256": credential_approval.sha256,
            "owner_subject_sha256": request.value["owner_subject_sha256"],
            "full_canary_plan_sha256": plan.sha256,
            "approval_request_sha256": request.sha256,
            "owner_approval_sha256": approval.sha256,
            "approval_path": str(DEFAULT_APPROVAL_PATH),
            "installed_at_unix": installed_at_unix,
        }
        terminal_receipt = {
            **unsigned,
            "receipt_sha256": _sha256_json(unsigned),
        }
        _publish_root_payload(
            DEFAULT_APPROVAL_PATH,
            _canonical_bytes(approval.value),
            expected_previous_sha256=request.value["prior_approval_file_sha256"],
        )
        return terminal_receipt


def discord_token_install_gate(
    *,
    coordinator_input: CoordinatorInput | None = None,
    approval: DiscordTokenInstallApproval | None = None,
) -> Mapping[str, Any]:
    _require_root_linux()
    coordinator_input = coordinator_input or load_coordinator_input()
    approval = approval or load_discord_token_install_approval(coordinator_input)
    approval.require(coordinator_input=coordinator_input, now_unix=int(time.time()))
    if not _services_are_exactly_stopped_and_disabled():
        _fail(
            "full_canary_services_not_stopped_for_token_install",
            phase="discord_token_preflight",
        )
    if os.path.lexists(DISCORD_TOKEN_PATH):
        _fail("discord_token_target_not_fresh", phase="discord_token_preflight")
    if os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH):
        _fail(
            "discord_token_receipt_target_not_fresh",
            phase="discord_token_preflight",
        )
    if os.path.lexists(DISCORD_TOKEN_STAGE_PATH):
        _fail(
            "discord_token_stage_not_fresh",
            phase="discord_token_preflight",
        )
    unsigned = {
        "schema": DISCORD_TOKEN_INSTALL_GATE_SCHEMA,
        "ok": True,
        "state": "token_install_authorized",
        "coordinator_input_sha256": coordinator_input.sha256,
        "discord_token_install_approval_sha256": approval.sha256,
        "owner_subject_sha256": approval.value["owner_subject_sha256"],
        "release_sha": coordinator_input.revision,
        "token_path": str(DISCORD_TOKEN_PATH),
        "edge_uid": coordinator_input.identities.edge_uid,
        "edge_gid": coordinator_input.identities.edge_gid,
        "expires_at_unix": approval.value["expires_at_unix"],
        "frame_schema": DISCORD_TOKEN_FRAME_SCHEMA,
    }
    return {**unsigned, "gate_sha256": _sha256_json(unsigned)}


def _publish_discord_install_journal(
    value: Mapping[str, Any],
    *,
    expected_previous_sha256: str | None,
) -> _RootPublication:
    return _publish_root_payload(
        DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
        _canonical_bytes(value),
        expected_previous_sha256=expected_previous_sha256,
    )


def _create_empty_discord_token_stage(
    coordinator_input: CoordinatorInput,
) -> os.stat_result:
    if os.path.lexists(DISCORD_TOKEN_STAGE_PATH):
        _fail("discord_token_stage_not_fresh", phase="discord_token_install")
    parent = DISCORD_TOKEN_STAGE_PATH.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid != 0
        or stat.S_IMODE(parent.st_mode) & 0o022
    ):
        _fail("discord_token_stage_parent_invalid", phase="discord_token_install")
    descriptor = os.open(
        DISCORD_TOKEN_STAGE_PATH,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        os.fchown(
            descriptor,
            coordinator_input.identities.edge_uid,
            coordinator_input.identities.edge_gid,
        )
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        item = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(DISCORD_TOKEN_STAGE_PATH.parent)
    reachable = DISCORD_TOKEN_STAGE_PATH.lstat()
    if (
        not _same_file_identity(item, reachable)
        or not stat.S_ISREG(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid != coordinator_input.identities.edge_uid
        or item.st_gid != coordinator_input.identities.edge_gid
        or stat.S_IMODE(item.st_mode) != 0o400
        or item.st_size != 0
    ):
        _fail("discord_token_stage_allocation_invalid", phase="discord_token_install")
    return item


def _write_discord_token_stage(
    coordinator_input: CoordinatorInput,
    expected: os.stat_result,
    token: bytearray,
) -> os.stat_result:
    descriptor = os.open(
        DISCORD_TOKEN_STAGE_PATH,
        os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (
            not _same_file_identity(opened, expected)
            or opened.st_nlink != 1
            or opened.st_uid != coordinator_input.identities.edge_uid
            or opened.st_gid != coordinator_input.identities.edge_gid
            or stat.S_IMODE(opened.st_mode) != 0o400
            or opened.st_size != 0
        ):
            _fail("discord_token_stage_identity_drifted", phase="discord_token_install")
        offset = 0
        while offset < len(token):
            written = os.write(descriptor, token[offset:])
            if written <= 0:
                _fail(
                    "discord_token_stage_write_stalled", phase="discord_token_install"
                )
            offset += written
        os.fsync(descriptor)
        written_item = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = DISCORD_TOKEN_STAGE_PATH.lstat()
    if (
        not _same_file_identity(written_item, reachable)
        or (written_item.st_dev, written_item.st_ino)
        != (expected.st_dev, expected.st_ino)
        or written_item.st_nlink != 1
        or written_item.st_size != len(token)
    ):
        _fail("discord_token_stage_write_unconfirmed", phase="discord_token_install")
    return written_item


def _link_discord_token_stage(
    coordinator_input: CoordinatorInput,
    expected: os.stat_result,
) -> os.stat_result:
    current = DISCORD_TOKEN_STAGE_PATH.lstat()
    if (
        not _same_file_identity(current, expected)
        or current.st_nlink != 1
        or current.st_uid != coordinator_input.identities.edge_uid
        or current.st_gid != coordinator_input.identities.edge_gid
        or stat.S_IMODE(current.st_mode) != 0o400
        or not MIN_DISCORD_TOKEN_BYTES <= current.st_size <= MAX_DISCORD_TOKEN_BYTES
        or os.path.lexists(DISCORD_TOKEN_PATH)
    ):
        _fail("discord_token_stage_link_precondition_invalid")
    os.link(DISCORD_TOKEN_STAGE_PATH, DISCORD_TOKEN_PATH, follow_symlinks=False)
    _fsync_directory(DISCORD_TOKEN_PATH.parent)
    linked = DISCORD_TOKEN_PATH.lstat()
    staged_link = DISCORD_TOKEN_STAGE_PATH.lstat()
    if (
        (linked.st_dev, linked.st_ino) != (expected.st_dev, expected.st_ino)
        or (staged_link.st_dev, staged_link.st_ino)
        != (expected.st_dev, expected.st_ino)
        or linked.st_nlink != 2
        or staged_link.st_nlink != 2
    ):
        _fail("discord_token_stage_link_invalid")
    DISCORD_TOKEN_STAGE_PATH.unlink()
    _fsync_directory(DISCORD_TOKEN_PATH.parent)
    installed = _validate_secret_metadata(
        DISCORD_TOKEN_PATH,
        uid=coordinator_input.identities.edge_uid,
        gid=coordinator_input.identities.edge_gid,
        maximum=MAX_DISCORD_TOKEN_BYTES,
    )
    if (
        (installed.st_dev, installed.st_ino) != (expected.st_dev, expected.st_ino)
        or installed.st_size != expected.st_size
        or os.path.lexists(DISCORD_TOKEN_STAGE_PATH)
    ):
        _fail("discord_token_stage_link_completion_invalid")
    return installed


def _load_terminal_recovery_journal(
    coordinator_input: CoordinatorInput,
) -> tuple[Mapping[str, Any], _RootFileSnapshot] | None:
    snapshot = _capture_root_snapshot(
        COORDINATOR_PROCESS_LEASE_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    if snapshot is None:
        return None
    journal = _decode_mapping(
        snapshot.raw,
        code="coordinator_process_journal_invalid",
    )
    if journal.get("schema") == LEGACY_RECOVERY_RECEIPT_SCHEMA:
        _parse_legacy_recovery_receipt(
            journal,
            coordinator_input=coordinator_input,
        )
        _fail(
            "legacy_recovery_receipt_reconciliation_required",
            phase="process_identity",
        )
    if journal.get("schema") != RECOVERY_RECEIPT_SCHEMA:
        _fail("coordinator_process_recovery_required", phase="process_identity")
    receipt = _parse_recovery_receipt_v2(
        journal,
        coordinator_input=coordinator_input,
    )
    return receipt, snapshot


def _consume_terminal_recovery_journal(
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any] | None:
    """Retire only an exact terminal recovery journal before a fresh token.

    An active process lease remains a hard blocker.  Moving consumption to
    the token publication lock closes the terminal-receipt -> token -> run
    gap: a failure before ``CoordinatorProcessLease.acquire`` can then use the
    ordinary token-only retirement path without mistaking old terminal truth
    for a live coordinator.
    """

    loaded = _load_terminal_recovery_journal(coordinator_input)
    if loaded is None:
        return None
    receipt, snapshot = loaded
    _remove_exact_root_snapshot(
        snapshot,
        state=_RootRemovalState(),
    )
    return receipt


def install_discord_token(
    *,
    gate_emitter: Any,
    frame_reader: Any = OpaqueDiscordTokenFrame.read,
) -> Mapping[str, Any]:
    """Gate, read, and no-replace install the isolated Discord token."""

    _harden_secret_process()
    coordinator_input = load_coordinator_input()
    _consume_terminal_discord_retirement(coordinator_input)
    approval = load_discord_token_install_approval(coordinator_input)
    gate = discord_token_install_gate(
        coordinator_input=coordinator_input,
        approval=approval,
    )
    gate_emitter(gate)
    # A gate can expire while the owner transport is preparing its one frame.
    approval.require(coordinator_input=coordinator_input, now_unix=int(time.time()))
    frame = frame_reader()
    if not isinstance(frame, OpaqueDiscordTokenFrame):
        raise TypeError("opaque Discord token frame is required")
    token = frame.consume()
    installed: os.stat_result | None = None
    receipt_publication: _RootPublication | None = None
    try:
        # The owner transport can take an arbitrary amount of time after the
        # non-secret gate was emitted.  Serialize the final service-state
        # proof, token publication, and receipt publication with the exact
        # lifecycle lock used by every canary start/stop path.  This closes
        # the gate->frame TOCTOU: no service can start between the final proof
        # and the completed token lease publication.
        with _lifecycle_lock():
            approval.require(
                coordinator_input=coordinator_input,
                now_unix=int(time.time()),
            )
            _consume_terminal_recovery_journal(coordinator_input)
            if not _services_are_exactly_stopped_and_disabled():
                _fail(
                    "full_canary_services_not_stopped_for_token_install",
                    phase="discord_token_install",
                )
            if any(
                os.path.lexists(path)
                for path in (
                    DISCORD_TOKEN_PATH,
                    DISCORD_TOKEN_STAGE_PATH,
                    DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
                )
            ):
                _fail("discord_token_install_state_not_fresh")
            prepared_at_unix = int(time.time())
            intent = _discord_install_journal(
                coordinator_input=coordinator_input,
                approval=approval,
                state="install_intent",
                prepared_at_unix=prepared_at_unix,
                device=None,
                inode=None,
                size=None,
            )
            receipt_publication = _publish_discord_install_journal(
                intent,
                expected_previous_sha256=None,
            )
            allocated_item = _create_empty_discord_token_stage(coordinator_input)
            allocated = _discord_install_journal(
                coordinator_input=coordinator_input,
                approval=approval,
                state="stage_allocated",
                prepared_at_unix=prepared_at_unix,
                device=allocated_item.st_dev,
                inode=allocated_item.st_ino,
                size=0,
            )
            receipt_publication = _publish_discord_install_journal(
                allocated,
                expected_previous_sha256=receipt_publication.after.sha256,
            )
            staged_item = _write_discord_token_stage(
                coordinator_input,
                allocated_item,
                token,
            )
            staged = _discord_install_journal(
                coordinator_input=coordinator_input,
                approval=approval,
                state="secret_staged",
                prepared_at_unix=prepared_at_unix,
                device=staged_item.st_dev,
                inode=staged_item.st_ino,
                size=staged_item.st_size,
            )
            receipt_publication = _publish_discord_install_journal(
                staged,
                expected_previous_sha256=receipt_publication.after.sha256,
            )
            installed = _link_discord_token_stage(
                coordinator_input,
                staged_item,
            )
            installed_at_unix = int(time.time())
            unsigned = {
                "schema": DISCORD_TOKEN_INSTALL_RECEIPT_SCHEMA,
                "ok": True,
                "release_sha": coordinator_input.revision,
                "coordinator_input_sha256": coordinator_input.sha256,
                "discord_token_install_approval_sha256": approval.sha256,
                "owner_subject_sha256": approval.value["owner_subject_sha256"],
                "token_path": str(DISCORD_TOKEN_PATH),
                "device": installed.st_dev,
                "inode": installed.st_ino,
                "owner_uid": installed.st_uid,
                "group_gid": installed.st_gid,
                "mode": "0400",
                "size": installed.st_size,
                "link_count": installed.st_nlink,
                "content_or_digest_recorded": False,
                "installed_at_unix": installed_at_unix,
            }
            terminal_receipt = {
                **unsigned,
                "receipt_sha256": _sha256_json(unsigned),
            }
            receipt_publication = _publish_discord_install_journal(
                terminal_receipt,
                expected_previous_sha256=receipt_publication.after.sha256,
            )
        return terminal_receipt
    except BaseException:
        _begin_active_signal_cleanup()
        # Once the intent is durable, every later crash point is represented
        # by that exact journal or a monotonic replacement.  Preserve it and
        # any bound inode for the owner-approved DRA1 retirement path; never
        # guess that an orphan token belongs to this operation.
        raise
    finally:
        _zeroize(token)
        frame.close()


def preflight_owner_launch() -> Mapping[str, Any]:
    """Return a self-digested no-secret gate before temporary-user mutation."""

    _require_root_linux()
    coordinator_input = load_coordinator_input()
    if os.path.lexists(COORDINATOR_PROCESS_LEASE_PATH):
        journal = _capture_root_snapshot(
            COORDINATOR_PROCESS_LEASE_PATH,
            maximum=MAX_OWNER_APPROVAL_BYTES,
        )
        if journal is None:
            _fail("coordinator_process_journal_drifted")
        journal_value = _decode_mapping(
            journal.raw,
            code="coordinator_process_journal_invalid",
        )
        if journal_value.get("schema") == LEGACY_RECOVERY_RECEIPT_SCHEMA:
            _parse_legacy_recovery_receipt(
                journal_value,
                coordinator_input=coordinator_input,
            )
            _fail(
                "legacy_recovery_receipt_reconciliation_required",
                phase="credential_prepare_preflight",
            )
        if journal_value.get("schema") == RECOVERY_RECEIPT_SCHEMA:
            _parse_recovery_receipt_v2(
                journal_value,
                coordinator_input=coordinator_input,
            )
        else:
            _fail(
                "coordinator_process_recovery_required",
                phase="credential_prepare_preflight",
            )
    retirement = _load_discord_retirement(coordinator_input)
    if retirement is not None and retirement[0]["state"] != "retired":
        _fail(
            "discord_token_retirement_recovery_required",
            phase="credential_prepare_preflight",
        )
    approval = load_credential_prepare_approval(coordinator_input)
    if not _services_are_exactly_stopped_and_disabled():
        _fail(
            "full_canary_services_not_stopped_for_owner_preflight",
            phase="credential_prepare_preflight",
        )
    if os.path.lexists(DISCORD_TOKEN_PATH):
        _fail(
            "discord_token_target_not_fresh",
            phase="credential_prepare_preflight",
        )
    if os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH):
        _fail(
            "discord_token_receipt_target_not_fresh",
            phase="credential_prepare_preflight",
        )
    if os.path.lexists(DISCORD_TOKEN_STAGE_PATH):
        _fail(
            "discord_token_stage_not_fresh",
            phase="credential_prepare_preflight",
        )
    if os.path.lexists(CANARY_BOOTSTRAP_CREDENTIAL_PATH):
        _fail(
            "bootstrap_credential_target_not_fresh",
            phase="credential_prepare_preflight",
        )
    unsigned = {
        "schema": OWNER_LAUNCH_GATE_SCHEMA,
        "ok": True,
        "state": "credential_prepare_authorized",
        "coordinator_input_sha256": coordinator_input.sha256,
        "credential_prepare_approval_sha256": approval.sha256,
        "owner_subject_sha256": approval.value["owner_subject_sha256"],
        "release_sha": coordinator_input.revision,
        "database_host": CANARY_DATABASE_HOST,
        "database_port": CANARY_DATABASE_PORT,
        "database_name": CANARY_DATABASE_NAME,
        "admin_username": derive_ephemeral_admin_username(approval.sha256),
        "expires_at_unix": approval.value["expires_at_unix"],
    }
    return {**unsigned, "gate_sha256": _sha256_json(unsigned)}


def _read_exact(fd: int, size: int) -> bytearray:
    if type(fd) is not int or fd < 0 or type(size) is not int or size < 0:
        _fail("admin_frame_bounds_invalid", phase="credential_read")
    result = bytearray()
    while len(result) < size:
        try:
            chunk = os.read(fd, size - len(result))
        except OSError as exc:
            _zeroize(result)
            raise CoordinatorError(
                "admin_frame_read_failed", phase="credential_read"
            ) from exc
        if not chunk:
            _zeroize(result)
            _fail("admin_frame_truncated", phase="credential_read")
        result.extend(chunk)
    return result


def _zeroize(value: bytearray | memoryview | None) -> None:
    """Best-effort in-place clearing for caller-owned mutable secret bytes."""

    if value is None:
        return
    try:
        value[:] = b"\x00" * len(value)
    except (BufferError, TypeError, ValueError):
        pass


def derive_ephemeral_admin_username(
    credential_prepare_approval_sha256: str,
) -> str:
    """Derive the unique bounded login from one exact fresh approval."""

    digest = _digest(
        credential_prepare_approval_sha256,
        "credential_prepare_approval_digest_invalid",
    )
    username = "muncho_canary_admin_" + digest[:16]
    if (
        len(username.encode("ascii")) > MAX_ADMIN_USERNAME_BYTES
        or _ADMIN_USERNAME_RE.fullmatch(username) is None
    ):
        _fail("ephemeral_admin_username_invalid")
    return username


class OpaqueStdinAdminFrame:
    """One-shot inherited-FD frame whose repr never contains credentials.

    Wire format is exactly ``MCA2`` + username length (u16 big endian) +
    password length (u32 big endian) + username UTF-8 + password UTF-8 + EOF.
    The object may be consumed once.  Its password remains a mutable bytearray
    and is cleared on consume/close/destruction on a best-effort basis.
    """

    __slots__ = ("_consumed", "_lock", "_password", "_username")

    def __init__(self, *, username: str, password: bytearray) -> None:
        self._username = username
        self._password: bytearray | None = password
        self._consumed = False
        self._lock = threading.Lock()

    @classmethod
    def read(
        cls,
        *,
        expected_username: str,
        fd: int = ADMIN_FRAME_FD,
    ) -> "OpaqueStdinAdminFrame":
        if fd != ADMIN_FRAME_FD:
            _fail("admin_frame_fd_not_fixed", phase="credential_read")
        try:
            if os.isatty(fd):
                _fail("admin_frame_tty_forbidden", phase="credential_read")
        except OSError as exc:
            raise CoordinatorError(
                "admin_frame_fd_unavailable", phase="credential_read"
            ) from exc
        header = _read_exact(fd, _FRAME_HEADER.size)
        username_raw: bytearray | None = None
        password_raw: bytearray | None = None
        try:
            magic, username_size, password_size = _FRAME_HEADER.unpack(header)
            if magic != ADMIN_FRAME_MAGIC:
                _fail("admin_frame_magic_invalid", phase="credential_read")
            if not 1 <= username_size <= MAX_ADMIN_USERNAME_BYTES:
                _fail("admin_frame_username_bound_invalid", phase="credential_read")
            if (
                not MIN_ADMIN_PASSWORD_BYTES
                <= password_size
                <= MAX_ADMIN_PASSWORD_BYTES
            ):
                _fail("admin_frame_password_bound_invalid", phase="credential_read")
            username_raw = _read_exact(fd, username_size)
            password_raw = _read_exact(fd, password_size)
            try:
                username = username_raw.decode("utf-8", errors="strict")
                password_text = password_raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise CoordinatorError(
                    "admin_frame_utf8_invalid", phase="credential_read"
                ) from exc
            if (
                username != expected_username
                or _ADMIN_USERNAME_RE.fullmatch(username) is None
            ):
                _fail("admin_frame_username_mismatch", phase="credential_read")
            if (
                password_text != password_text.strip()
                or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in password_text
                )
                or "\x00" in password_text
            ):
                _fail("admin_frame_password_invalid", phase="credential_read")
            try:
                trailing = os.read(fd, 1)
            except OSError as exc:
                raise CoordinatorError(
                    "admin_frame_eof_check_failed", phase="credential_read"
                ) from exc
            if trailing:
                _fail("admin_frame_trailing_data", phase="credential_read")
            # Drop the immutable decoder result before returning.  Python
            # cannot guarantee clearing it; the authoritative retained copy is
            # the mutable bytearray and is zeroized below on every path.
            password_text = ""
            result_password = password_raw
            password_raw = None
            return cls(username=username, password=result_password)
        finally:
            _zeroize(header)
            _zeroize(username_raw)
            _zeroize(password_raw)

    @property
    def username(self) -> str:
        return self._username

    @property
    def consumed(self) -> bool:
        with self._lock:
            return self._consumed

    def consume_password(self) -> bytearray:
        with self._lock:
            if self._consumed or self._password is None:
                _fail("admin_frame_replay_forbidden", phase="credential_read")
            value = self._password
            self._password = None
            self._consumed = True
            return value

    def close(self) -> None:
        with self._lock:
            _zeroize(self._password)
            self._password = None
            self._consumed = True

    def __enter__(self) -> "OpaqueStdinAdminFrame":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort fallback.
        self.close()

    def __repr__(self) -> str:
        return (
            "OpaqueStdinAdminFrame("
            f"username={self.username!r}, consumed={self.consumed!r}, "
            "password=<redacted>)"
        )


class VerifiedTLSBootstrapAdminSession:
    """Concrete bounded-wire PostgreSQL session with verified TLS identity."""

    __slots__ = (
        "_closed",
        "_lock",
        "_session",
        "tls_peer_certificate_sha256",
        "username",
    )

    def __init__(
        self,
        session: _PostgresWireSession,
        *,
        username: str,
        tls_peer_certificate_sha256: str,
    ) -> None:
        self._session = session
        self.username = username
        self.tls_peer_certificate_sha256 = _digest(
            tls_peer_certificate_sha256,
            "admin_tls_peer_certificate_digest_invalid",
        )
        self._closed = False
        self._lock = threading.Lock()

    @classmethod
    def open(
        cls,
        *,
        frame: OpaqueStdinAdminFrame,
        tls_server_name: str,
        expected_tls_peer_certificate_sha256: str,
    ) -> "VerifiedTLSBootstrapAdminSession":
        if not isinstance(frame, OpaqueStdinAdminFrame):
            raise TypeError("opaque admin frame is required")
        if _TLS_NAME_RE.fullmatch(tls_server_name) is None:
            _fail("admin_tls_server_name_invalid", phase="admin_connect")
        expected_peer = _digest(
            expected_tls_peer_certificate_sha256,
            "admin_tls_peer_certificate_digest_invalid",
        )
        password = frame.consume_password()
        protected: ssl.SSLSocket | None = None
        ownership_transferred = False
        try:
            # The bounded connection helper only consults expected_uid while
            # validating the public CA.  Authentication consumes the opaque
            # bytes below; no credential source/path fallback is used.
            config = WriterDBConfig(
                host=CANARY_DATABASE_HOST,
                tls_server_name=tls_server_name,
                port=CANARY_DATABASE_PORT,
                database=CANARY_DATABASE_NAME,
                user=frame.username,
                ca_file=CANARY_DATABASE_CA_PATH,
                credential=CredentialSource(expected_uid=0, fd=ADMIN_FRAME_FD),
                application_name="muncho-full-canary-coordinator",
            )
            protected, observed_peer = _open_verified_tls_connection(config)
            if observed_peer != expected_peer:
                _fail("admin_tls_peer_certificate_mismatch", phase="admin_connect")
            _send_startup_message(
                protected,
                config=config,
                database=CANARY_DATABASE_NAME,
            )
            try:
                password_text = password.decode("utf-8", errors="strict")
                _authenticate(
                    protected,
                    user=frame.username,
                    password=password_text,
                )
            finally:
                password_text = ""
            wire = _PostgresWireSession(protected)
            ownership_transferred = True
            return cls(
                wire,
                username=frame.username,
                tls_peer_certificate_sha256=observed_peer,
            )
        except CoordinatorError:
            raise
        except (PostgresProtocolError, OSError, ssl.SSLError, UnicodeError) as exc:
            raise CoordinatorError(
                "admin_verified_tls_session_failed", phase="admin_connect"
            ) from exc
        finally:
            _zeroize(password)
            frame.close()
            if protected is not None and not ownership_transferred:
                try:
                    protected.close()
                except OSError:
                    pass

    def query(self, sql: str, *, maximum_rows: int) -> Any:
        with self._lock:
            if self._closed:
                _fail("admin_session_closed", phase="database_operation")
            return self._session.query(sql, maximum_rows=maximum_rows)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._session.close()
            self._closed = True

    def __repr__(self) -> str:
        return (
            "VerifiedTLSBootstrapAdminSession("
            f"username={self.username!r}, "
            f"tls_peer_certificate_sha256={self.tls_peer_certificate_sha256!r}, "
            f"closed={self._closed!r}, credential=<redacted>)"
        )


class BootstrapCredentialLease:
    """Rotate, install, attest, then provably retire one bootstrap password."""

    def __init__(
        self,
        *,
        coordinator_input: CoordinatorInput,
        admin_session: VerifiedTLSBootstrapAdminSession,
        password_factory: Callable[[], bytes] = lambda: base64.urlsafe_b64encode(
            secrets.token_bytes(48)
        ),
        hba_collector: Callable[[WriterDBConfig], ManagedCloudSQLAdminHBAReceipt] = (
            collect_managed_cloudsqladmin_hba_receipt
        ),
    ) -> None:
        if not isinstance(coordinator_input, CoordinatorInput):
            raise TypeError("coordinator input is required")
        if not isinstance(admin_session, VerifiedTLSBootstrapAdminSession):
            raise TypeError("verified bootstrap administrator session is required")
        self.coordinator_input = coordinator_input
        self.admin_session = admin_session
        self._password_factory = password_factory
        self._hba_collector = hba_collector
        self._installed: os.stat_result | None = None
        self._removal_state = _SecretRemovalState()
        self._password_disabled = False
        self._credential_removed = False
        self._prepared = False
        self._retired = False
        self._lock = threading.Lock()

    @property
    def password_disabled(self) -> bool:
        return self._password_disabled

    @property
    def credential_removed(self) -> bool:
        return self._credential_removed

    @property
    def recovery_material_preserved(self) -> bool:
        return (self._installed is not None and not self._credential_removed) or (
            self._installed is None
            and not self._credential_removed
            and os.path.lexists(CANARY_BOOTSTRAP_CREDENTIAL_PATH)
        )

    @staticmethod
    def _require_do_result(result: Any, *, code: str) -> None:
        if (
            str(getattr(result, "command_tag", "")).upper() != "DO"
            or tuple(getattr(result, "rows", ()))
            or tuple(getattr(result, "columns", ()))
        ):
            _fail(code, phase="bootstrap_credential")

    def prepare(self) -> tuple[Mapping[str, Any], ManagedCloudSQLAdminHBAReceipt]:
        """Create the exact login credential and collect fresh user-bound HBA."""

        with self._lock:
            if self._prepared or self._retired:
                _fail("bootstrap_credential_lease_replay_forbidden")
            if os.path.lexists(CANARY_BOOTSTRAP_CREDENTIAL_PATH):
                _fail("bootstrap_credential_target_not_fresh")
            generated = self._password_factory()
            if not isinstance(generated, bytes):
                _fail("bootstrap_password_factory_invalid")
            password = bytearray(generated)
            generated = b""
            try:
                if not MIN_ADMIN_PASSWORD_BYTES <= len(
                    password
                ) <= MAX_ADMIN_PASSWORD_BYTES or any(
                    byte < 33 or byte == 127 for byte in password
                ):
                    _fail("bootstrap_password_generated_invalid")
                password_base64 = base64.b64encode(password).decode("ascii")
                sql = _BOOTSTRAP_ROLE_PASSWORD_TEMPLATE.format(
                    password_base64=password_base64
                )
                result = self.admin_session.query(sql, maximum_rows=0)
                self._require_do_result(
                    result,
                    code="bootstrap_role_shape_or_rotation_failed",
                )
                password_base64 = ""
                self._installed = _atomic_install_secret(
                    CANARY_BOOTSTRAP_CREDENTIAL_PATH,
                    password,
                    uid=self.coordinator_input.identities.writer_uid,
                    gid=self.coordinator_input.identities.writer_gid,
                )
                config = WriterDBConfig(
                    host=CANARY_DATABASE_HOST,
                    tls_server_name=self.coordinator_input.tls_server_name,
                    port=CANARY_DATABASE_PORT,
                    database=CANARY_DATABASE_NAME,
                    user=CANARY_BOOTSTRAP_LOGIN,
                    ca_file=CANARY_DATABASE_CA_PATH,
                    credential=CredentialSource(
                        expected_uid=self.coordinator_input.identities.writer_uid,
                        expected_gid=self.coordinator_input.identities.writer_gid,
                        path=CANARY_BOOTSTRAP_CREDENTIAL_PATH,
                        allowed_modes=frozenset({0o400}),
                    ),
                    application_name="muncho-canary-bootstrap-hba-probe",
                )
                receipt = self._hba_collector(config)
                if not isinstance(receipt, ManagedCloudSQLAdminHBAReceipt):
                    _fail("bootstrap_hba_receipt_type_invalid")
                now_unix = int(time.time())
                if (
                    not receipt.is_fresh(now_unix)
                    or receipt.host != CANARY_DATABASE_HOST
                    or receipt.tls_server_name != self.coordinator_input.tls_server_name
                    or receipt.port != CANARY_DATABASE_PORT
                    or receipt.user != CANARY_BOOTSTRAP_LOGIN
                    or receipt.database != "cloudsqladmin"
                    or receipt.server_certificate_sha256
                    != self.admin_session.tls_peer_certificate_sha256
                    or receipt.tls_peer_verified is not True
                ):
                    _fail("bootstrap_hba_receipt_binding_invalid")
                writer_config = copy.deepcopy(
                    dict(self.coordinator_input.value["writer_config"])
                )
                scope = writer_config.get("canary_scope_preapproval")
                if not isinstance(scope, dict):
                    _fail("bootstrap_writer_scope_invalid")
                scope["bootstrap_credential_file"] = str(
                    CANARY_BOOTSTRAP_CREDENTIAL_PATH
                )
                scope["bootstrap_managed_cloudsqladmin_hba_rejection_receipt"] = (
                    receipt.as_dict()
                )
                scope["bootstrap_managed_cloudsqladmin_hba_rejection_sha256"] = (
                    receipt.sha256
                )
                self._prepared = True
                return writer_config, receipt
            except BaseException as primary:
                try:
                    self._retire_locked()
                except BaseException as cleanup:
                    raise ExceptionGroup(
                        "bootstrap credential preparation cleanup blocked",
                        [primary, cleanup],
                    ) from None
                raise
            finally:
                _zeroize(password)

    def _retire_locked(self) -> None:
        if self._retired:
            return
        try:
            result = self.admin_session.query(
                _BOOTSTRAP_ROLE_DISABLE_SQL,
                maximum_rows=0,
            )
            self._require_do_result(
                result,
                code="bootstrap_password_disable_unconfirmed",
            )
            self._password_disabled = True
        except BaseException as exc:
            raise CoordinatorCleanupBlocked(
                "bootstrap_password_cleanup_blocked",
                phase="bootstrap_credential_cleanup",
            ) from exc
        if self._installed is not None:
            try:
                _remove_exact_secret(
                    CANARY_BOOTSTRAP_CREDENTIAL_PATH,
                    self._installed,
                    state=self._removal_state,
                )
                self._credential_removed = True
            except BaseException as exc:
                raise CoordinatorCleanupBlocked(
                    "bootstrap_credential_file_cleanup_blocked",
                    phase="bootstrap_credential_cleanup",
                ) from exc
        else:
            if os.path.lexists(CANARY_BOOTSTRAP_CREDENTIAL_PATH):
                raise CoordinatorCleanupBlocked(
                    "bootstrap_credential_unowned_cleanup_blocked",
                    phase="bootstrap_credential_cleanup",
                )
            self._credential_removed = True
        self._retired = True

    def retire(self) -> None:
        with self._lock:
            self._retire_locked()


class CredentialRetiringAdminSession:
    """Session facade that retires bootstrap authority before closing."""

    def __init__(
        self,
        session: VerifiedTLSBootstrapAdminSession,
        lease: BootstrapCredentialLease,
    ) -> None:
        self._session = session
        self._lease = lease
        self._closed = False
        self._lock = threading.Lock()

    def query(self, sql: str, *, maximum_rows: int) -> Any:
        return self._session.query(sql, maximum_rows=maximum_rows)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._lease.retire()
            self._session.close()
            self._closed = True

    def __repr__(self) -> str:
        return "CredentialRetiringAdminSession(credential=<redacted>)"


@dataclass(frozen=True)
class _WriterSnapshot:
    raw: bytes
    item: os.stat_result

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.raw)


def _capture_writer_snapshot(writer_gid: int) -> _WriterSnapshot | None:
    if not os.path.lexists(DEFAULT_WRITER_CONFIG_SOURCE):
        return None
    raw, item = _read_staged_writer_config(
        DEFAULT_WRITER_CONFIG_SOURCE,
        mode=0o440,
        uid=0,
        gid=writer_gid,
    )
    return _WriterSnapshot(raw=raw, item=item)


@dataclass
class _WriterPublication:
    writer_gid: int
    before: _WriterSnapshot | None
    after: _WriterSnapshot
    _rolled_back: bool = False
    _removal_state: _RootRemovalState = field(
        default_factory=_RootRemovalState,
    )

    def rollback(self) -> None:
        if self._rolled_back:
            return
        if self.before is None:
            if not self._removal_state.unlinked:
                current = _capture_writer_snapshot(self.writer_gid)
                if (
                    current is None
                    or current.raw != self.after.raw
                    or not _same_file_identity(current.item, self.after.item)
                ):
                    raise CoordinatorCleanupBlocked(
                        "staged_writer_cleanup_identity_changed",
                        phase="staged_writer_cleanup",
                    )
                DEFAULT_WRITER_CONFIG_SOURCE.unlink()
                self._removal_state.unlinked = True
            try:
                _fsync_directory(DEFAULT_WRITER_CONFIG_SOURCE.parent)
            except BaseException as exc:
                raise CoordinatorCleanupBlocked(
                    "staged_writer_cleanup_fsync_blocked",
                    phase="staged_writer_cleanup",
                ) from exc
            if os.path.lexists(DEFAULT_WRITER_CONFIG_SOURCE):
                raise CoordinatorCleanupBlocked(
                    "staged_writer_cleanup_unconfirmed",
                    phase="staged_writer_cleanup",
                )
        else:
            current = _capture_writer_snapshot(self.writer_gid)
            if (
                current is None
                or current.raw != self.after.raw
                or not _same_file_identity(current.item, self.after.item)
            ):
                raise CoordinatorCleanupBlocked(
                    "staged_writer_cleanup_identity_changed",
                    phase="staged_writer_cleanup",
                )
            _atomic_stage_writer_config(
                DEFAULT_WRITER_CONFIG_SOURCE,
                self.before.raw,
                mode=0o440,
                uid=0,
                gid=self.writer_gid,
                expected_existing_sha256=self.after.sha256,
            )
            restored = _capture_writer_snapshot(self.writer_gid)
            if restored is None or restored.raw != self.before.raw:
                raise CoordinatorCleanupBlocked(
                    "staged_writer_restore_unconfirmed",
                    phase="staged_writer_cleanup",
                )
        self._rolled_back = True


def coordinator_secret_gate(
    *,
    coordinator_input: CoordinatorInput,
    approval: CredentialPrepareApproval,
    process_lease: "CoordinatorProcessLease",
) -> Mapping[str, Any]:
    """Validate post-token preconditions and authorize one MCA2 read."""

    _require_root_linux()
    approval.require(
        coordinator_input=coordinator_input,
        now_unix=int(time.time()),
    )
    if (
        not isinstance(process_lease, CoordinatorProcessLease)
        or process_lease.retired
        or process_lease.value["coordinator_input_sha256"] != coordinator_input.sha256
        or process_lease.value["credential_prepare_approval_sha256"] != approval.sha256
    ):
        _fail("coordinator_process_lease_not_bound")
    _validate_secret_metadata(
        DISCORD_TOKEN_PATH,
        uid=coordinator_input.identities.edge_uid,
        gid=coordinator_input.identities.edge_gid,
        maximum=MAX_DISCORD_TOKEN_BYTES,
    )
    if os.path.lexists(CANARY_BOOTSTRAP_CREDENTIAL_PATH):
        _fail(
            "bootstrap_credential_target_not_fresh",
            phase="coordinator_secret_gate",
        )
    unsigned = {
        "schema": SECRET_GATE_SCHEMA,
        "ok": True,
        "state": "awaiting_admin_credential",
        "coordinator_input_sha256": coordinator_input.sha256,
        "credential_prepare_approval_sha256": approval.sha256,
        "owner_subject_sha256": approval.value["owner_subject_sha256"],
        "release_sha": coordinator_input.revision,
        "admin_username": derive_ephemeral_admin_username(approval.sha256),
        "database_host": CANARY_DATABASE_HOST,
        "database_port": CANARY_DATABASE_PORT,
        "database_name": CANARY_DATABASE_NAME,
        "tls_server_name": coordinator_input.tls_server_name,
        "tls_peer_certificate_sha256": (coordinator_input.tls_peer_certificate_sha256),
        "expires_at_unix": approval.value["expires_at_unix"],
        "frame_schema": ADMIN_FRAME_SCHEMA,
        "coordinator_process_lease_sha256": process_lease.value["lease_sha256"],
        "coordinator_pid": process_lease.value["pid"],
        "coordinator_start_time_ticks": process_lease.value["process_start_time_ticks"],
        "coordinator_boot_id_sha256": process_lease.value["boot_id_sha256"],
    }
    return {**unsigned, "gate_sha256": _sha256_json(unsigned)}


def _services_are_exactly_stopped_and_disabled() -> bool:
    states = {
        unit: collect_service_state(unit)
        for unit in (EDGE_UNIT_NAME, WRITER_UNIT_NAME, GATEWAY_UNIT_NAME)
    }
    checks = evaluate_service_states(states, phase="stopped")
    return bool(checks) and all(checks.values())


def _remove_active_approval_request(publication: _RootPublication) -> None:
    errors: list[BaseException] = []
    if not publication._unlinked:
        try:
            current = _capture_root_snapshot(
                OWNER_APPROVAL_REQUEST_PATH,
                maximum=MAX_OWNER_APPROVAL_BYTES,
            )
            if (
                current is None
                or current.raw != publication.after.raw
                or not _same_file_identity(current.item, publication.after.item)
            ):
                raise RuntimeError("approval request cleanup identity changed")
            OWNER_APPROVAL_REQUEST_PATH.unlink()
            publication._unlinked = True
        except BaseException as exc:
            errors.append(exc)
    fsync_error: BaseException | None = None
    for _attempt in range(2):
        try:
            _fsync_directory(OWNER_APPROVAL_REQUEST_PATH.parent)
            fsync_error = None
            break
        except BaseException as exc:
            fsync_error = exc
    if fsync_error is not None:
        errors.append(fsync_error)
    try:
        OWNER_APPROVAL_REQUEST_PATH.lstat()
    except FileNotFoundError:
        pass
    except BaseException as exc:
        errors.append(exc)
    else:
        errors.append(RuntimeError("approval request cleanup unconfirmed"))
    if errors:
        raise CoordinatorCleanupBlocked(
            "owner_approval_request_cleanup_blocked",
            phase="owner_approval_request_cleanup",
        ) from ExceptionGroup("owner approval request cleanup failures", errors)


class _SignalFence:
    """Convert graceful termination signals into the normal cleanup path."""

    def __init__(self) -> None:
        self._signal = signal
        self._previous: dict[int, Any] = {}
        self._previous_active: Any = None
        self.cleaning = False
        self.received = False

    def __enter__(self) -> "_SignalFence":
        for name in ("SIGHUP", "SIGINT", "SIGTERM"):
            number = getattr(self._signal, name, None)
            if number is None:
                continue
            self._previous[number] = self._signal.getsignal(number)
            self._signal.signal(number, self._handle)
        self._previous_active = getattr(_ACTIVE_SIGNAL_FENCE, "value", None)
        _ACTIVE_SIGNAL_FENCE.value = self
        return self

    def _handle(self, _number: int, _frame: Any) -> None:
        self.received = True
        if not self.cleaning:
            # Flip before raising so every later signal is suppressed while
            # Python unwinds through publication rollback and secret cleanup.
            self.cleaning = True
            raise CoordinatorError(
                "coordinator_graceful_termination_requested",
                phase="signal",
            )

    def begin_cleanup(self) -> None:
        self.cleaning = True

    def __exit__(self, *_args: object) -> None:
        self.cleaning = True
        handled = set(self._previous)
        pthread_sigmask = getattr(self._signal, "pthread_sigmask", None)
        old_mask: Any = None
        if callable(pthread_sigmask) and handled:
            old_mask = pthread_sigmask(self._signal.SIG_BLOCK, handled)
        try:
            # Discard any signal queued during cleanup without exposing a
            # one-handler-at-a-time restoration window.  Re-block before the
            # caller's handlers are restored, then restore the original mask
            # only after the handler set is complete.
            if old_mask is not None:
                discardable = handled.difference(set(old_mask))
                for number in handled:
                    self._signal.signal(number, self._signal.SIG_IGN)
                if discardable:
                    pthread_sigmask(self._signal.SIG_UNBLOCK, discardable)
                    pthread_sigmask(self._signal.SIG_BLOCK, discardable)
            for number, handler in self._previous.items():
                self._signal.signal(number, handler)
        finally:
            if old_mask is not None:
                pthread_sigmask(self._signal.SIG_SETMASK, old_mask)
            if self._previous_active is None:
                try:
                    del _ACTIVE_SIGNAL_FENCE.value
                except AttributeError:
                    pass
            else:
                _ACTIVE_SIGNAL_FENCE.value = self._previous_active


_PROCESS_LEASE_FIELDS = frozenset({
    "schema",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "pid",
    "process_start_time_ticks",
    "boot_id_sha256",
    "boot_time_ns",
    "module_origin",
    "module_sha256",
    "process_exe_sha256",
    "process_cmdline_sha256",
    "created_at_unix",
    "lease_sha256",
})


def _sealed_coordinator_module_identity(
    coordinator_input: CoordinatorInput,
) -> tuple[str, str]:
    origin, module_sha256 = module_file_identity(__file__)
    artifact_root = Path(coordinator_input.base_plan.release["artifact_root"])
    try:
        relative = Path(origin).relative_to(artifact_root).as_posix()
    except ValueError as exc:
        raise CoordinatorError(
            "coordinator_module_outside_sealed_release",
            phase="process_identity",
        ) from exc
    if (
        re.fullmatch(
            r"venv/lib/python3\.[0-9]+/site-packages/"
            r"gateway/canonical_full_canary_coordinator\.py",
            relative,
        )
        is None
    ):
        _fail(
            "coordinator_module_origin_invalid",
            phase="process_identity",
        )
    try:
        sealed_path, _sealed_raw, sealed_sha256 = _validated_release_file(
            coordinator_input.base_plan,
            Path(relative),
            maximum_bytes=MAX_FIXED_CANARY_ARTIFACT_BYTES,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CoordinatorError(
            "coordinator_module_not_manifest_bound",
            phase="process_identity",
        ) from exc
    if str(sealed_path) != origin or sealed_sha256 != module_sha256:
        _fail(
            "coordinator_module_not_manifest_bound",
            phase="process_identity",
        )
    return origin, module_sha256


def _process_executable_sha256(pid: int) -> str:
    descriptor = os.open(
        f"/proc/{pid}/exe",
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    digest = hashlib.sha256()
    try:
        item = os.fstat(descriptor)
        if not stat.S_ISREG(item.st_mode) or item.st_size <= 0:
            _fail("coordinator_process_executable_invalid")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


_ATTESTED_CLI_COMMANDS = frozenset({
    "publish-coordinator-input",
    "preflight-owner-launch",
    "preflight-recovery",
    "run",
    "recover",
    "finalize-recovery",
    "install-discord-token",
    "install-final-approval",
    "stop-and-retire-discord-token",
})


def _expected_command_cmdline(
    coordinator_input: CoordinatorInput,
    *,
    command: str,
) -> bytes:
    if command not in _ATTESTED_CLI_COMMANDS:
        _fail("coordinator_process_command_invalid", phase="process_identity")
    values = (
        str(coordinator_input.base_plan.release["interpreter"]),
        "-B",
        "-I",
        "-m",
        "gateway.canonical_full_canary_coordinator",
        command,
    )
    return b"\0".join(item.encode("utf-8") for item in values) + b"\0"


def _expected_run_cmdline(coordinator_input: CoordinatorInput) -> bytes:
    return _expected_command_cmdline(coordinator_input, command="run")


def _current_executable_manifest_identity(
    coordinator_input: CoordinatorInput,
) -> str:
    expected_path = Path(coordinator_input.base_plan.release["interpreter"])
    artifact_root = Path(coordinator_input.base_plan.release["artifact_root"])
    try:
        relative = expected_path.relative_to(artifact_root)
        sealed_path, sealed_raw, sealed_sha256 = _validated_release_file(
            coordinator_input.base_plan,
            relative,
            maximum_bytes=MAX_RELEASE_MANIFEST_BYTES,
        )
        expected_item = sealed_path.lstat()
        before_link = os.readlink("/proc/self/exe")
        descriptor = os.open(
            "/proc/self/exe",
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CoordinatorError(
            "coordinator_process_executable_not_manifest_bound",
            phase="process_identity",
        ) from exc
    digest = hashlib.sha256()
    try:
        running_item = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        running_item_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_link = os.readlink("/proc/self/exe")
        reachable = sealed_path.lstat()
    except OSError as exc:
        raise CoordinatorError(
            "coordinator_process_executable_not_manifest_bound",
            phase="process_identity",
        ) from exc
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
    observed_sha256 = digest.hexdigest()
    if (
        sealed_path != expected_path
        or before_link != str(expected_path)
        or after_link != str(expected_path)
        or identity(expected_item) != identity(reachable)
        or identity(expected_item) != identity(running_item)
        or identity(running_item) != identity(running_item_after)
        or running_item.st_uid != 0
        or running_item.st_gid != 0
        or not stat.S_ISREG(running_item.st_mode)
        or running_item.st_size != len(sealed_raw)
        or observed_sha256 != sealed_sha256
    ):
        _fail(
            "coordinator_process_executable_not_manifest_bound",
            phase="process_identity",
        )
    return observed_sha256


def _attest_current_cli_process(
    coordinator_input: CoordinatorInput,
    *,
    command: str,
) -> Mapping[str, str]:
    """Prove sealed code, interpreter, and exact command before any frame."""

    _require_root_linux()
    module_origin, module_sha256 = _sealed_coordinator_module_identity(
        coordinator_input
    )
    executable_sha256 = _current_executable_manifest_identity(coordinator_input)
    try:
        cmdline = Path("/proc/self/cmdline").read_bytes()
    except OSError as exc:
        raise CoordinatorError(
            "coordinator_process_cmdline_unavailable",
            phase="process_identity",
        ) from exc
    expected = _expected_command_cmdline(
        coordinator_input,
        command=command,
    )
    if cmdline != expected:
        _fail("coordinator_process_cmdline_invalid", phase="process_identity")
    return {
        "module_origin": module_origin,
        "module_sha256": module_sha256,
        "process_exe_sha256": executable_sha256,
        "process_cmdline_sha256": _sha256_bytes(cmdline),
    }


def _process_exec_binding(
    pid: int,
    *,
    coordinator_input: CoordinatorInput,
) -> tuple[str, str]:
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    expected = _expected_run_cmdline(coordinator_input)
    if cmdline != expected:
        _fail("coordinator_process_cmdline_invalid", phase="process_identity")
    return _process_executable_sha256(pid), _sha256_bytes(cmdline)


def _parse_process_lease(
    value: Any,
    *,
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PROCESS_LEASE_FIELDS:
        _fail("coordinator_process_lease_invalid", phase="recovery_preflight")
    raw = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in raw.items() if key != "lease_sha256"}
    origin, module_sha256 = _sealed_coordinator_module_identity(coordinator_input)
    current_exe_sha256 = _process_executable_sha256(os.getpid())
    if (
        raw["schema"] != "muncho-full-canary-coordinator-process-lease.v1"
        or raw["release_sha"] != coordinator_input.revision
        or raw["coordinator_input_sha256"] != coordinator_input.sha256
        or _SHA256_RE.fullmatch(str(raw["credential_prepare_approval_sha256"])) is None
        or _SHA256_RE.fullmatch(str(raw["owner_subject_sha256"])) is None
        or raw["ephemeral_admin_username"]
        != derive_ephemeral_admin_username(raw["credential_prepare_approval_sha256"])
        or type(raw["pid"]) is not int
        or raw["pid"] <= 1
        or type(raw["process_start_time_ticks"]) is not int
        or raw["process_start_time_ticks"] <= 0
        or type(raw["boot_time_ns"]) is not int
        or raw["boot_time_ns"] < 0
        or raw["module_origin"] != origin
        or raw["module_sha256"] != module_sha256
        or raw["process_exe_sha256"] != current_exe_sha256
        or type(raw["created_at_unix"]) is not int
        or raw["created_at_unix"] < 0
        or raw["lease_sha256"] != _sha256_json(unsigned)
    ):
        _fail("coordinator_process_lease_invalid", phase="recovery_preflight")
    _digest(raw["boot_id_sha256"], "coordinator_process_boot_id_invalid")
    _digest(raw["process_cmdline_sha256"], "coordinator_process_cmdline_digest_invalid")
    return raw


def _load_process_lease_record(
    *,
    coordinator_input: CoordinatorInput,
) -> tuple[Mapping[str, Any], _RootFileSnapshot]:
    snapshot = _capture_root_snapshot(
        COORDINATOR_PROCESS_LEASE_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    if snapshot is None:
        _fail(
            "coordinator_process_recovery_not_required",
            phase="recovery_preflight",
        )
    return (
        _parse_process_lease(
            _decode_mapping(
                snapshot.raw,
                code="coordinator_process_lease_invalid",
            ),
            coordinator_input=coordinator_input,
        ),
        snapshot,
    )


def _wait_for_pidfd_exit(pidfd: int, *, timeout_seconds: float) -> bool:
    if type(pidfd) is not int or pidfd < 0 or not 0 <= timeout_seconds <= 60:
        _fail("recovery_pidfd_wait_invalid", phase="recovery_process")
    try:
        readable, _writable, _exceptional = select.select(
            [pidfd],
            [],
            [],
            timeout_seconds,
        )
    except (OSError, ValueError) as exc:
        raise CoordinatorError(
            "recovery_pidfd_wait_failed",
            phase="recovery_process",
        ) from exc
    return pidfd in readable


def _pidfd_signal(pidfd: int, signum: int, *, code: str) -> bool:
    sender = getattr(signal, "pidfd_send_signal", None)
    if not callable(sender):
        _fail("recovery_pidfd_api_unavailable", phase="recovery_process")
    try:
        sender(pidfd, signum, None, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError as exc:
        raise CoordinatorError(code, phase="recovery_process") from exc


@dataclass
class CoordinatorProcessLease:
    value: Mapping[str, Any]
    publication: _RootPublication
    lock_fd: int
    retired: bool = False

    @classmethod
    def acquire(
        cls,
        *,
        coordinator_input: CoordinatorInput,
        credential_approval: CredentialPrepareApproval,
    ) -> "CoordinatorProcessLease":
        _require_root_linux()
        parent = COORDINATOR_PROCESS_LEASE_PATH.parent
        parent_item = parent.lstat()
        if (
            not stat.S_ISDIR(parent_item.st_mode)
            or stat.S_ISLNK(parent_item.st_mode)
            or parent_item.st_uid != 0
            or parent_item.st_gid != 0
            or stat.S_IMODE(parent_item.st_mode) & 0o022
        ):
            _fail("coordinator_process_lease_parent_invalid")
        lock_parent = COORDINATOR_PROCESS_LOCK_PATH.parent
        if not os.path.lexists(lock_parent):
            anchor = lock_parent.parent.lstat()
            if (
                not stat.S_ISDIR(anchor.st_mode)
                or stat.S_ISLNK(anchor.st_mode)
                or anchor.st_uid != 0
                or stat.S_IMODE(anchor.st_mode) & 0o022
            ):
                _fail("coordinator_process_lock_parent_invalid")
            os.mkdir(lock_parent, 0o700)
            os.chown(lock_parent, 0, 0)
            os.chmod(lock_parent, 0o700)
            _fsync_directory(lock_parent.parent)
        lock_parent_item = lock_parent.lstat()
        if (
            not stat.S_ISDIR(lock_parent_item.st_mode)
            or stat.S_ISLNK(lock_parent_item.st_mode)
            or lock_parent_item.st_uid != 0
            or lock_parent_item.st_gid != 0
            or stat.S_IMODE(lock_parent_item.st_mode) != 0o700
        ):
            _fail("coordinator_process_lock_parent_invalid")
        descriptor = os.open(
            COORDINATOR_PROCESS_LOCK_PATH,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.fchown(descriptor, 0, 0)
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_item = os.fstat(descriptor)
            if (
                not stat.S_ISREG(lock_item.st_mode)
                or lock_item.st_uid != 0
                or lock_item.st_gid != 0
                or stat.S_IMODE(lock_item.st_mode) != 0o600
            ):
                _fail("coordinator_process_lock_invalid")
            if os.path.lexists(COORDINATOR_PROCESS_LEASE_PATH):
                journal = _capture_root_snapshot(
                    COORDINATOR_PROCESS_LEASE_PATH,
                    maximum=MAX_OWNER_APPROVAL_BYTES,
                )
                if journal is None:
                    _fail("coordinator_process_journal_drifted")
                journal_value = _decode_mapping(
                    journal.raw,
                    code="coordinator_process_journal_invalid",
                )
                if journal_value.get("schema") == LEGACY_RECOVERY_RECEIPT_SCHEMA:
                    _parse_legacy_recovery_receipt(
                        journal_value,
                        coordinator_input=coordinator_input,
                    )
                    _fail(
                        "legacy_recovery_receipt_reconciliation_required",
                        phase="process_identity",
                    )
                if journal_value.get("schema") != RECOVERY_RECEIPT_SCHEMA:
                    _fail("coordinator_process_recovery_required")
                _parse_recovery_receipt_v2(
                    journal_value,
                    coordinator_input=coordinator_input,
                )
                _remove_exact_root_snapshot(
                    journal,
                    state=_RootRemovalState(),
                )
            pid = os.getpid()
            boot_sha256, boottime_ns = boot_identity()
            module_origin, module_sha256 = _sealed_coordinator_module_identity(
                coordinator_input
            )
            process_exe_sha256, process_cmdline_sha256 = _process_exec_binding(
                pid,
                coordinator_input=coordinator_input,
            )
            unsigned = {
                "schema": "muncho-full-canary-coordinator-process-lease.v1",
                "release_sha": coordinator_input.revision,
                "coordinator_input_sha256": coordinator_input.sha256,
                "credential_prepare_approval_sha256": (credential_approval.sha256),
                "owner_subject_sha256": credential_approval.value[
                    "owner_subject_sha256"
                ],
                "ephemeral_admin_username": derive_ephemeral_admin_username(
                    credential_approval.sha256
                ),
                "pid": pid,
                "process_start_time_ticks": process_start_time_ticks(pid),
                "boot_id_sha256": boot_sha256,
                "boot_time_ns": boottime_ns,
                "module_origin": module_origin,
                "module_sha256": module_sha256,
                "process_exe_sha256": process_exe_sha256,
                "process_cmdline_sha256": process_cmdline_sha256,
                "created_at_unix": int(time.time()),
            }
            value = {**unsigned, "lease_sha256": _sha256_json(unsigned)}
            publication = _publish_root_payload(
                COORDINATOR_PROCESS_LEASE_PATH,
                _canonical_bytes(value),
                expected_previous_sha256=None,
            )
            return cls(
                value=value,
                publication=publication,
                lock_fd=descriptor,
            )
        except BaseException:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except BaseException:
                pass
            os.close(descriptor)
            raise

    def retire(self) -> None:
        if self.retired:
            return
        _remove_exact_root_snapshot(
            self.publication.after,
            state=self.publication._removal_state,
        )
        fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
        os.close(self.lock_fd)
        self.lock_fd = -1
        self.retired = True

    def abandon_lock_on_process_exit(self) -> None:
        if self.lock_fd >= 0:
            try:
                os.close(self.lock_fd)
            except OSError:
                pass
            self.lock_fd = -1


def _error_code_and_phase(
    error: BaseException,
    *,
    fallback_phase: str,
) -> tuple[str, str]:
    candidates: list[BaseException] = [error]
    flattened: list[BaseException] = []
    while candidates:
        candidate = candidates.pop(0)
        flattened.append(candidate)
        if isinstance(candidate, BaseExceptionGroup):
            candidates.extend(candidate.exceptions)
    selected = next(
        (item for item in flattened if isinstance(item, CoordinatorCleanupBlocked)),
        None,
    ) or next(
        (item for item in flattened if isinstance(item, CoordinatorError)),
        None,
    )
    if isinstance(selected, CoordinatorError):
        return selected.code, selected.phase or fallback_phase
    return "coordinator_execution_failed", fallback_phase


def run_full_canary(
    *,
    frame_emitter: Callable[[Mapping[str, Any]], None],
    admin_frame_reader: Callable[..., OpaqueStdinAdminFrame] = (
        OpaqueStdinAdminFrame.read
    ),
    admin_session_opener: Callable[..., VerifiedTLSBootstrapAdminSession] = (
        VerifiedTLSBootstrapAdminSession.open
    ),
    driver_factory: Callable[..., HonestFullCanaryDriver] = HonestFullCanaryDriver,
) -> Mapping[str, Any]:
    """Execute one same-process full canary and return one terminal receipt."""

    coordinator_input: CoordinatorInput | None = None
    credential_approval: CredentialPrepareApproval | None = None
    admin_session: VerifiedTLSBootstrapAdminSession | None = None
    credential_lease: BootstrapCredentialLease | None = None
    writer_publication: _WriterPublication | None = None
    staged_plan_publication: _RootPublication | None = None
    runtime_plan_publication: _RootPublication | None = None
    request_publication: _RootPublication | None = None
    prepared: Any = None
    plan: FullCanaryPlan | None = None
    live_result: Mapping[str, Any] | None = None
    driver_started = False
    services_terminal = False
    discord_token_identity: os.stat_result | None = None
    discord_token_removed = False
    discord_receipt_snapshot: _RootFileSnapshot | None = None
    discord_receipt_removal = _RootRemovalState()
    discord_install_receipt: Mapping[str, Any] | None = None
    phase = "process_hardening"
    primary: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    prior_writer: _WriterSnapshot | None = None
    prior_staged_plan: _RootFileSnapshot | None = None
    prior_runtime_plan: _RootFileSnapshot | None = None
    process_lease: CoordinatorProcessLease | None = None

    with _SignalFence() as signal_fence:
        try:
            _harden_secret_process()
            phase = "coordinator_input"
            coordinator_input = load_coordinator_input()
            (
                discord_install_receipt,
                discord_token_identity,
                discord_receipt_snapshot,
            ) = load_discord_token_install_receipt(
                coordinator_input,
            )
            services_terminal = _services_are_exactly_stopped_and_disabled()
            if not services_terminal:
                _fail(
                    "full_canary_services_not_initially_stopped",
                    phase="coordinator_input",
                )
            credential_approval = load_credential_prepare_approval(coordinator_input)
            process_lease = CoordinatorProcessLease.acquire(
                coordinator_input=coordinator_input,
                credential_approval=credential_approval,
            )
            phase = "secret_gate"
            gate = coordinator_secret_gate(
                coordinator_input=coordinator_input,
                approval=credential_approval,
                process_lease=process_lease,
            )
            frame_emitter(gate)
            credential_approval.require(
                coordinator_input=coordinator_input,
                now_unix=int(time.time()),
            )
            phase = "admin_credential"
            frame = admin_frame_reader(
                expected_username=gate["admin_username"],
            )
            phase = "admin_connect"
            admin_session = admin_session_opener(
                frame=frame,
                tls_server_name=coordinator_input.tls_server_name,
                expected_tls_peer_certificate_sha256=(
                    coordinator_input.tls_peer_certificate_sha256
                ),
            )
            credential_lease = BootstrapCredentialLease(
                coordinator_input=coordinator_input,
                admin_session=admin_session,
            )
            phase = "bootstrap_credential"
            final_writer_config, hba_receipt = credential_lease.prepare()
            fixture = _validated_e2e_fixture(coordinator_input.base_plan)
            prior_writer = _capture_writer_snapshot(
                coordinator_input.identities.writer_gid
            )

            def reconcile_previous() -> None:
                nonlocal prior_staged_plan, prior_runtime_plan
                prior_runtime_plan = _capture_root_snapshot(DEFAULT_PLAN_PATH)
                if prior_runtime_plan is None:
                    _fail("prestage_runtime_plan_missing")
                previous = load_full_canary_plan()
                prior_staged_plan = _capture_root_snapshot(DEFAULT_STAGED_PLAN_PATH)
                if prior_staged_plan is not None:
                    try:
                        staged_previous = FullCanaryPlan.from_mapping(
                            _decode_mapping(
                                prior_staged_plan.raw,
                                code="prestage_plan_invalid",
                            )
                        )
                    except (TypeError, ValueError) as exc:
                        raise CoordinatorError("prestage_plan_invalid") from exc
                    if staged_previous.to_mapping() != previous.to_mapping():
                        _fail("prestage_plan_runtime_drifted")
                FullCanaryLifecycle(previous).stop(reason="operator_requested")

            def publish_staged_writer(
                path: Path,
                payload: bytes,
                **kwargs: Any,
            ) -> None:
                nonlocal writer_publication
                with _defer_termination_signals():
                    _atomic_stage_writer_config(path, payload, **kwargs)
                    writer_raw, writer_item = _read_staged_writer_config(
                        DEFAULT_WRITER_CONFIG_SOURCE,
                        mode=0o440,
                        uid=0,
                        gid=coordinator_input.identities.writer_gid,
                    )
                    if writer_raw != payload:
                        _fail("staged_writer_publication_drifted")
                    writer_publication = _WriterPublication(
                        writer_gid=coordinator_input.identities.writer_gid,
                        before=prior_writer,
                        after=_WriterSnapshot(
                            raw=writer_raw,
                            item=writer_item,
                        ),
                    )

            def build_plan() -> FullCanaryPlan:
                nonlocal writer_publication, staged_plan_publication
                nonlocal request_publication, plan
                writer_raw, writer_item = _read_staged_writer_config(
                    DEFAULT_WRITER_CONFIG_SOURCE,
                    mode=0o440,
                    uid=0,
                    gid=coordinator_input.identities.writer_gid,
                )
                if (
                    writer_publication is None
                    or writer_publication.after.raw != writer_raw
                    or not _same_file_identity(
                        writer_publication.after.item,
                        writer_item,
                    )
                ):
                    _fail("staged_writer_publication_untracked")
                writer_artifact = ExactArtifact(
                    source_path=DEFAULT_WRITER_CONFIG_SOURCE,
                    target_path=DEFAULT_WRITER_CONFIG,
                    sha256=_sha256_bytes(writer_raw),
                    mode=0o440,
                    uid=0,
                    gid=coordinator_input.identities.writer_gid,
                )
                plan = build_full_canary_plan(
                    writer_activation_plan=(coordinator_input.writer_activation_plan),
                    writer_activation_receipt=coordinator_input.value[
                        "writer_activation_receipt"
                    ],
                    writer_activation_receipt_file_sha256=coordinator_input.value[
                        "writer_activation_receipt_file_sha256"
                    ],
                    identities=coordinator_input.identities,
                    writer_config=writer_artifact,
                    gateway_config=coordinator_input.artifacts["gateway_config"],
                    edge_config=coordinator_input.artifacts["edge_config"],
                    e2e_fixture=coordinator_input.artifacts["e2e_fixture"],
                    host_identity_receipt=coordinator_input.artifacts[
                        "host_identity_receipt"
                    ],
                )
                existing_staged = _capture_root_snapshot(DEFAULT_STAGED_PLAN_PATH)
                expected_staged = (
                    None
                    if existing_staged is None
                    else (
                        existing_staged.sha256
                        if prior_staged_plan is not None
                        and existing_staged.sha256 == prior_staged_plan.sha256
                        else _fail("staged_plan_unreconciled")
                    )
                )
                staged_plan_publication = _publish_root_payload(
                    DEFAULT_STAGED_PLAN_PATH,
                    _canonical_bytes(plan.to_mapping()),
                    expected_previous_sha256=expected_staged,
                )
                scope = final_writer_config.get("canary_scope_preapproval")
                if not isinstance(scope, Mapping):
                    _fail("final_writer_scope_missing")
                request, request_publication = build_owner_approval_request(
                    coordinator_input=coordinator_input,
                    credential_approval=credential_approval,
                    plan=plan,
                    staged_plan_publication=staged_plan_publication,
                    hba_receipt=hba_receipt,
                    fixture=fixture,
                    approval_source_sha256=str(scope["approval_source_sha256"]),
                )
                approval_state["request"] = request
                return plan

            approval_state: dict[str, Any] = {}

            def approve(final_plan: FullCanaryPlan) -> FullCanaryOwnerApproval:
                nonlocal runtime_plan_publication, request_publication
                request = approval_state.get("request")
                if not isinstance(request, OwnerApprovalRequest):
                    _fail("owner_approval_request_missing")
                remaining_seconds = (
                    float(request.value["approval_deadline_unix"]) - time.time()
                )
                if remaining_seconds < 1:
                    _fail("owner_approval_wait_window_exhausted")
                approval = wait_for_fresh_owner_approval(
                    final_plan,
                    timeout_seconds=min(
                        float(request.value["max_wait_seconds"]),
                        remaining_seconds,
                    ),
                    ready_callback=lambda: frame_emitter(request.value),
                )
                now_unix = int(time.time())
                approval.require(
                    plan_sha256=final_plan.sha256,
                    now_unix=now_unix,
                )
                _require_runtime_final_owner_approval_binding(
                    approval,
                    request,
                )
                if (
                    now_unix > request.value["approval_deadline_unix"]
                    or now_unix
                    > request.value["hba_expires_at_unix"]
                    - HBA_EXPIRY_SAFETY_MARGIN_SECONDS
                ):
                    _fail("owner_approval_ttl_or_binding_invalid")
                existing_runtime = _capture_root_snapshot(DEFAULT_PLAN_PATH)
                expected_runtime = (
                    None
                    if existing_runtime is None
                    else (
                        existing_runtime.sha256
                        if prior_runtime_plan is not None
                        and existing_runtime.sha256 == prior_runtime_plan.sha256
                        else _fail("runtime_plan_unreconciled")
                    )
                )
                runtime_plan_publication = _publish_root_payload(
                    DEFAULT_PLAN_PATH,
                    _canonical_bytes(final_plan.to_mapping()),
                    expected_previous_sha256=expected_runtime,
                )
                if request_publication is None:
                    _fail("owner_approval_request_publication_missing")
                _remove_active_approval_request(request_publication)
                request_publication = None
                return approval

            phase = "plan_and_final_approval"
            prepared = prepare_session_bound_plan(
                writer_config=final_writer_config,
                fixture=fixture,
                writer_gid=coordinator_input.identities.writer_gid,
                bootstrap_sql_sha256=coordinator_input.value["bootstrap_sql_sha256"],
                bootstrap_retire_sql_sha256=coordinator_input.value[
                    "bootstrap_retire_sql_sha256"
                ],
                staged_writer_config=DEFAULT_WRITER_CONFIG_SOURCE,
                plan_builder=build_plan,
                approval_provider=approve,
                writer=publish_staged_writer,
                prestage_reconciler=reconcile_previous,
            )
            plan = prepared.plan
            now_unix = int(time.time())
            prepared.approval.require(
                plan_sha256=plan.sha256,
                now_unix=now_unix,
            )
            if (
                now_unix
                > hba_receipt.expires_at_unix - HBA_EXPIRY_SAFETY_MARGIN_SECONDS
            ):
                _fail("hba_receipt_expired_before_canary")
            wrapped_session = CredentialRetiringAdminSession(
                admin_session,
                credential_lease,
            )
            provisioner = PreopenedSessionBootstrapProvisioner(
                wrapped_session,
                tls_peer_certificate_sha256=(admin_session.tls_peer_certificate_sha256),
            )
            phase = "live_driver"
            driver_started = True
            services_terminal = False
            live_result = driver_factory(
                prepared,
                bootstrap_provisioner=provisioner,
            ).run()
            if not _services_are_exactly_stopped_and_disabled():
                _fail(
                    "full_canary_terminal_service_truth_unconfirmed",
                    phase="terminal_cleanup",
                )
            services_terminal = True
        except BaseException as exc:
            primary = exc
        finally:
            signal_fence.begin_cleanup()
            if prepared is not None:
                try:
                    prepared.discard_session_key()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if driver_started and live_result is None and plan is not None:
                try:
                    FullCanaryLifecycle(plan).stop(reason="verification_failed")
                    if not _services_are_exactly_stopped_and_disabled():
                        raise CoordinatorCleanupBlocked(
                            "full_canary_terminal_service_truth_unconfirmed",
                            phase="terminal_cleanup",
                        )
                    services_terminal = True
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if credential_lease is not None:
                try:
                    credential_lease.retire()
                except BaseException as first:
                    try:
                        credential_lease.retire()
                    except BaseException as second:
                        cleanup_errors.append(
                            ExceptionGroup(
                                "bootstrap credential retirement retries failed",
                                [first, second],
                            )
                        )
            if admin_session is not None and not admin_session.closed:
                try:
                    admin_session.close()
                except BaseException as first:
                    try:
                        admin_session.close()
                    except BaseException as second:
                        cleanup_errors.append(
                            ExceptionGroup(
                                "admin session close retries failed",
                                [first, second],
                            )
                        )
            if discord_token_identity is not None and services_terminal:
                try:
                    retirement = _retire_discord_token_lease(
                        coordinator_input=coordinator_input,
                        install_receipt=discord_install_receipt,
                        installed=discord_token_identity,
                        install_snapshot=discord_receipt_snapshot,
                    )
                    discord_token_removed = (
                        retirement["state"] == "retired"
                        and retirement["token_removed"] is True
                        and retirement["install_receipt_removed"] is True
                    )
                    discord_receipt_removal.unlinked = discord_token_removed
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if request_publication is not None:
                try:
                    _remove_active_approval_request(request_publication)
                    request_publication = None
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if not driver_started:
                for publication in (
                    runtime_plan_publication,
                    staged_plan_publication,
                ):
                    if publication is not None:
                        try:
                            publication.rollback()
                        except BaseException as exc:
                            cleanup_errors.append(exc)
                if writer_publication is not None:
                    try:
                        writer_publication.rollback()
                    except BaseException as exc:
                        cleanup_errors.append(exc)
            if process_lease is not None and not process_lease.retired:
                credential_cleanup_complete = credential_lease is None or (
                    credential_lease.password_disabled
                    and credential_lease.credential_removed
                )
                durable_cleanup_complete = (
                    primary is None
                    and not signal_fence.received
                    and not cleanup_errors
                    and (admin_session is None or admin_session.closed)
                    and credential_cleanup_complete
                    and services_terminal
                    and not os.path.lexists(DISCORD_TOKEN_PATH)
                    and not os.path.lexists(DISCORD_TOKEN_STAGE_PATH)
                    and not os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH)
                    and not os.path.lexists(CANARY_BOOTSTRAP_CREDENTIAL_PATH)
                    and not os.path.lexists(OWNER_APPROVAL_REQUEST_PATH)
                )
                if durable_cleanup_complete:
                    try:
                        process_lease.retire()
                    except BaseException as first:
                        try:
                            process_lease.retire()
                        except BaseException as second:
                            cleanup_errors.append(
                                ExceptionGroup(
                                    "coordinator process lease retirement retries failed",
                                    [first, second],
                                )
                            )
                else:
                    # Preserve the exact root lease as the recovery journal,
                    # but release this process's flock before returning the
                    # terminal cleanup-blocked receipt.
                    process_lease.abandon_lock_on_process_exit()

    session_closed = admin_session is None or admin_session.closed
    password_disabled = (
        credential_lease is not None and credential_lease.password_disabled
    )
    credential_removed = (
        credential_lease is not None and credential_lease.credential_removed
    )
    discord_pair_absent = (
        not os.path.lexists(DISCORD_TOKEN_PATH)
        and not os.path.lexists(DISCORD_TOKEN_STAGE_PATH)
        and not os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH)
    )
    if discord_token_identity is None and discord_pair_absent:
        discord_token_removed = True
    recovery_preserved = (
        bool(
            credential_lease is not None
            and credential_lease.recovery_material_preserved
        )
        or not session_closed
        or (
            discord_token_identity is not None
            and (not discord_token_removed or not discord_receipt_removal.unlinked)
        )
        or (process_lease is not None and not process_lease.retired)
        or any(
            os.path.lexists(path)
            for path in (
                DISCORD_TOKEN_PATH,
                DISCORD_TOKEN_STAGE_PATH,
                DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
                CANARY_BOOTSTRAP_CREDENTIAL_PATH,
                OWNER_APPROVAL_REQUEST_PATH,
                COORDINATOR_PROCESS_LEASE_PATH,
            )
        )
    )
    plan_sha256 = None if plan is None else plan.sha256
    if primary is None and not cleanup_errors and live_result is not None:
        if (
            not session_closed
            or not password_disabled
            or not credential_removed
            or not discord_token_removed
            or not discord_receipt_removal.unlinked
        ):
            primary = CoordinatorCleanupBlocked(
                "terminal_cleanup_attestation_incomplete",
                phase="terminal_cleanup",
            )
        else:
            unsigned = {
                "schema": COORDINATOR_RECEIPT_SCHEMA,
                "ok": True,
                "release_sha": coordinator_input.revision,
                "coordinator_input_sha256": coordinator_input.sha256,
                "credential_prepare_approval_sha256": credential_approval.sha256,
                "owner_subject_sha256": credential_approval.value[
                    "owner_subject_sha256"
                ],
                "ephemeral_admin_username": derive_ephemeral_admin_username(
                    credential_approval.sha256
                ),
                "temporary_admin_delete_required": True,
                "full_canary_plan_sha256": plan.sha256,
                "owner_approval_sha256": prepared.approval.sha256,
                "live_driver_result": copy.deepcopy(dict(live_result)),
                "live_driver_receipt_sha256": _sha256_json(live_result),
                "bootstrap_login_password_disabled": True,
                "bootstrap_credential_removed": True,
                "discord_token_removed": discord_token_removed,
                "admin_session_closed": True,
                "coordinator_process_lease_retired": True,
                "services_enabled": False,
                "completed_at_unix": int(time.time()),
            }
            return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}

    if primary is None:
        primary = (
            cleanup_errors[0]
            if cleanup_errors
            else CoordinatorError(
                "coordinator_terminal_result_missing",
                phase="terminal",
            )
        )
    code, failure_phase = _error_code_and_phase(
        primary,
        fallback_phase=phase,
    )
    cleanup_blocked = bool(cleanup_errors) or not session_closed or recovery_preserved
    unsigned_failure = {
        "schema": COORDINATOR_FAILURE_SCHEMA,
        "ok": False,
        "phase": failure_phase,
        "error_code": code,
        "release_sha": (
            None if coordinator_input is None else coordinator_input.revision
        ),
        "coordinator_input_sha256": (
            None if coordinator_input is None else coordinator_input.sha256
        ),
        "credential_prepare_approval_sha256": (
            None if credential_approval is None else credential_approval.sha256
        ),
        "owner_subject_sha256": (
            None
            if credential_approval is None
            else credential_approval.value["owner_subject_sha256"]
        ),
        "ephemeral_admin_username": (
            None
            if credential_approval is None
            else derive_ephemeral_admin_username(credential_approval.sha256)
        ),
        "full_canary_plan_sha256": plan_sha256,
        "cleanup_status": ("cleanup_blocked" if cleanup_blocked else "complete"),
        "recovery_material_preserved": recovery_preserved,
        "admin_session_closed": session_closed,
        "coordinator_process_lease_retired": (
            process_lease is None or process_lease.retired
        ),
        "bootstrap_login_password_disabled": password_disabled,
        "bootstrap_credential_removed": credential_removed,
        "discord_token_removed": discord_token_removed,
        "services_enabled": False if services_terminal else None,
    }
    return {
        **unsigned_failure,
        "receipt_sha256": _sha256_json(unsigned_failure),
    }


_DISCORD_RETIREMENT_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "discord_token_install_receipt_sha256",
    "token_path",
    "token_device",
    "token_inode",
    "services_stopped_proven",
    "services_enabled",
    "token_removed",
    "install_receipt_removed",
    "prepared_at_unix",
    "retired_at_unix",
    "receipt_sha256",
})


def _parse_discord_retirement(
    value: Any,
    *,
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _DISCORD_RETIREMENT_FIELDS:
        _fail("discord_token_retirement_receipt_invalid")
    raw = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    state = raw["state"]
    token_identity = (raw["token_device"], raw["token_inode"])
    token_identity_valid = token_identity == (None, None) or (
        type(token_identity[0]) is int
        and token_identity[0] > 0
        and type(token_identity[1]) is int
        and token_identity[1] > 0
    )
    if (
        raw["schema"] != DISCORD_TOKEN_RETIREMENT_RECEIPT_SCHEMA
        or state not in {"retirement_prepared", "retired"}
        or raw["ok"] is not (state == "retired")
        or raw["release_sha"] != coordinator_input.revision
        or raw["coordinator_input_sha256"] != coordinator_input.sha256
        or raw["token_path"] != str(DISCORD_TOKEN_PATH)
        or not token_identity_valid
        or raw["services_stopped_proven"] is not True
        or raw["services_enabled"] is not False
        or type(raw["prepared_at_unix"]) is not int
        or raw["prepared_at_unix"] < 0
        or raw["receipt_sha256"] != _sha256_json(unsigned)
    ):
        _fail("discord_token_retirement_receipt_invalid")
    _digest(
        raw["discord_token_install_receipt_sha256"],
        "discord_token_retirement_install_digest_invalid",
    )
    if state == "retirement_prepared":
        if (
            raw["token_removed"] is not False
            or raw["install_receipt_removed"] is not False
            or raw["retired_at_unix"] is not None
        ):
            _fail("discord_token_retirement_prepared_invalid")
    elif (
        raw["token_removed"] is not True
        or raw["install_receipt_removed"] is not True
        or type(raw["retired_at_unix"]) is not int
        or raw["retired_at_unix"] < raw["prepared_at_unix"]
    ):
        _fail("discord_token_retirement_terminal_invalid")
    return raw


def _prepared_discord_retirement_sha256(
    retirement: Mapping[str, Any],
) -> str:
    """Return the causal prepared-state digest for either monotonic state."""

    prepared_unsigned = {
        **{
            key: item
            for key, item in retirement.items()
            if key
            not in {
                "receipt_sha256",
                "ok",
                "state",
                "token_removed",
                "install_receipt_removed",
                "retired_at_unix",
            }
        },
        "ok": False,
        "state": "retirement_prepared",
        "token_removed": False,
        "install_receipt_removed": False,
        "retired_at_unix": None,
    }
    return _sha256_json(prepared_unsigned)


def _load_discord_retirement(
    coordinator_input: CoordinatorInput,
) -> tuple[Mapping[str, Any], _RootFileSnapshot] | None:
    snapshot = _capture_root_snapshot(
        DISCORD_TOKEN_RETIREMENT_RECEIPT_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    if snapshot is None:
        return None
    return (
        _parse_discord_retirement(
            _decode_mapping(
                snapshot.raw,
                code="discord_token_retirement_receipt_invalid",
            ),
            coordinator_input=coordinator_input,
        ),
        snapshot,
    )


def _discord_install_state_matches_retirement(
    install_state: _DiscordInstallState,
    retirement: Mapping[str, Any],
) -> bool:
    if (
        install_state.receipt_sha256
        != retirement["discord_token_install_receipt_sha256"]
    ):
        return False
    observed_identity = (install_state.device, install_state.inode)
    retirement_identity = (
        retirement["token_device"],
        retirement["token_inode"],
    )
    if observed_identity == retirement_identity:
        return True
    # install_intent deliberately does not place an inode in the install
    # journal.  If it had an empty stage, the prepared retirement receipt
    # captures that inode before unlinking it.  A retry after that unlink sees
    # only the unchanged intent journal; the durable prepared receipt is the
    # causal proof of the now-absent stage.
    return bool(
        install_state.state == "install_intent"
        and observed_identity == (None, None)
        and install_state.token_item is None
        and install_state.stage_item is None
        and (
            retirement_identity == (None, None)
            or (
                type(retirement_identity[0]) is int
                and retirement_identity[0] > 0
                and type(retirement_identity[1]) is int
                and retirement_identity[1] > 0
            )
        )
    )


def _load_prepared_discord_retirement_source(
    coordinator_input: CoordinatorInput,
    *,
    require_terminal_install: bool,
) -> tuple[
    Mapping[str, Any],
    _RootFileSnapshot,
    _DiscordInstallState | None,
]:
    """Load the exact monotonic subsets authorized by a prepared receipt.

    Retirement publishes ``retirement_prepared`` before removing any secret
    or install-state artifact.  Its durable receipt therefore becomes the
    causal authority for retries.  With the fixed removal order the only
    valid terminal-install subsets are token+receipt, receipt-only, and
    neither; token-only is impossible.  Nonterminal install journals retain
    their own stricter stage/link invariants through
    ``_load_discord_install_state``.
    """

    loaded = _load_discord_retirement(coordinator_input)
    if loaded is None or loaded[0]["state"] != "retirement_prepared":
        _fail("discord_token_retirement_prepared_source_missing")
    retirement, retirement_snapshot = loaded
    install_state: _DiscordInstallState | None = None
    if os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH):
        install_state = _load_discord_install_state(
            coordinator_input,
            require_terminal_token=False,
        )
        if require_terminal_install and install_state.state != "installed":
            _fail("discord_token_recovery_install_state_not_terminal")
        if not _discord_install_state_matches_retirement(
            install_state,
            retirement,
        ):
            _fail("discord_token_retirement_install_state_drifted")
    elif os.path.lexists(DISCORD_TOKEN_PATH) or os.path.lexists(
        DISCORD_TOKEN_STAGE_PATH
    ):
        # Both removal implementations unlink secret artifacts before the
        # install journal.  A token/stage without its causal journal cannot be
        # attributed safely and must never be guessed away.
        _fail("discord_token_retirement_impossible_artifact_subset")
    return retirement, retirement_snapshot, install_state


def _recovery_discord_retirement_inputs(
    *,
    coordinator_input: CoordinatorInput,
    gate: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any] | None,
    os.stat_result | None,
    _RootFileSnapshot | None,
]:
    """Resolve a gate to the exact current monotonic retirement subset."""

    gate_state = gate["discord_token_state"]
    if gate_state not in {"installed", "retirement_prepared", "retired"}:
        _fail("recovery_discord_token_gate_state_invalid")
    retirement_loaded = _load_discord_retirement(coordinator_input)
    if retirement_loaded is None:
        if gate_state != "installed" or os.path.lexists(DISCORD_TOKEN_STAGE_PATH):
            _fail("recovery_discord_token_drifted")
        state = _load_discord_install_state(coordinator_input)
        if (
            state.state != "installed"
            or state.token_item is None
            or state.receipt_sha256 != gate["discord_token_install_receipt_sha256"]
            or state.device != gate["token_device"]
            or state.inode != gate["token_inode"]
        ):
            _fail("recovery_discord_token_drifted")
        return state.value, state.token_item, state.snapshot

    retirement = retirement_loaded[0]
    if (
        retirement["discord_token_install_receipt_sha256"]
        != gate["discord_token_install_receipt_sha256"]
        or retirement["token_device"] != gate["token_device"]
        or retirement["token_inode"] != gate["token_inode"]
    ):
        _fail("recovery_discord_token_drifted")
    if gate_state == "retirement_prepared" and (
        gate["discord_retirement_receipt_sha256"] is None
        or _prepared_discord_retirement_sha256(retirement)
        != gate["discord_retirement_receipt_sha256"]
    ):
        _fail("recovery_discord_retirement_causal_drifted")
    if gate_state == "retired" and (
        retirement["state"] != "retired"
        or retirement["receipt_sha256"] != gate["discord_retirement_receipt_sha256"]
    ):
        _fail("recovery_discord_retirement_terminal_drifted")

    if retirement["state"] == "retirement_prepared":
        _same_retirement, _snapshot, state = _load_prepared_discord_retirement_source(
            coordinator_input,
            require_terminal_install=True,
        )
        if state is None:
            return None, None, None
        return state.value, state.token_item, state.snapshot
    if any(
        os.path.lexists(path)
        for path in (
            DISCORD_TOKEN_PATH,
            DISCORD_TOKEN_STAGE_PATH,
            DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
        )
    ):
        _fail("recovery_discord_token_terminal_artifact_survived")
    return None, None, None


def _consume_terminal_discord_retirement(
    coordinator_input: CoordinatorInput,
) -> None:
    existing = _load_discord_retirement(coordinator_input)
    if existing is None:
        return
    retirement, snapshot = existing
    if (
        retirement["state"] != "retired"
        or os.path.lexists(DISCORD_TOKEN_PATH)
        or os.path.lexists(DISCORD_TOKEN_STAGE_PATH)
        or os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH)
    ):
        _fail("discord_token_retirement_recovery_required")
    _remove_exact_root_snapshot(
        snapshot,
        state=_RootRemovalState(),
    )


def _retire_discord_token_lease(
    *,
    coordinator_input: CoordinatorInput,
    install_receipt: Mapping[str, Any] | None,
    installed: os.stat_result | None,
    install_snapshot: _RootFileSnapshot | None,
) -> Mapping[str, Any]:
    if not _services_are_exactly_stopped_and_disabled():
        _fail(
            "discord_token_retirement_services_not_stopped",
            phase="discord_token_retirement",
        )
    existing = _load_discord_retirement(coordinator_input)
    if existing is not None:
        retirement, retirement_snapshot = existing
        if retirement["state"] == "retired":
            if (
                os.path.lexists(DISCORD_TOKEN_PATH)
                or os.path.lexists(DISCORD_TOKEN_STAGE_PATH)
                or os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH)
            ):
                _fail("discord_token_terminal_retirement_drifted")
            return retirement
        install_digest = retirement["discord_token_install_receipt_sha256"]
        token_device = retirement["token_device"]
        token_inode = retirement["token_inode"]
        prepared_at = retirement["prepared_at_unix"]
    else:
        if install_receipt is None or installed is None or install_snapshot is None:
            _fail("discord_token_retirement_source_missing")
        install_digest = install_receipt["receipt_sha256"]
        token_device = installed.st_dev
        token_inode = installed.st_ino
        prepared_at = int(time.time())
        prepared_unsigned = {
            "schema": DISCORD_TOKEN_RETIREMENT_RECEIPT_SCHEMA,
            "ok": False,
            "state": "retirement_prepared",
            "release_sha": coordinator_input.revision,
            "coordinator_input_sha256": coordinator_input.sha256,
            "discord_token_install_receipt_sha256": install_digest,
            "token_path": str(DISCORD_TOKEN_PATH),
            "token_device": token_device,
            "token_inode": token_inode,
            "services_stopped_proven": True,
            "services_enabled": False,
            "token_removed": False,
            "install_receipt_removed": False,
            "prepared_at_unix": prepared_at,
            "retired_at_unix": None,
        }
        prepared = {
            **prepared_unsigned,
            "receipt_sha256": _sha256_json(prepared_unsigned),
        }
        publication = _publish_root_payload(
            DISCORD_TOKEN_RETIREMENT_RECEIPT_PATH,
            _canonical_bytes(prepared),
            expected_previous_sha256=None,
        )
        retirement = prepared
        retirement_snapshot = publication.after
    if (
        install_receipt is not None
        and install_receipt["receipt_sha256"] != install_digest
    ):
        _fail("discord_token_retirement_install_receipt_drifted")
    if os.path.lexists(DISCORD_TOKEN_PATH):
        if installed is None:
            installed = _validate_secret_metadata(
                DISCORD_TOKEN_PATH,
                uid=coordinator_input.identities.edge_uid,
                gid=coordinator_input.identities.edge_gid,
                maximum=MAX_DISCORD_TOKEN_BYTES,
            )
        if (installed.st_dev, installed.st_ino) != (token_device, token_inode):
            _fail("discord_token_retirement_token_identity_drifted")
        _remove_exact_secret(
            DISCORD_TOKEN_PATH,
            installed,
            state=_SecretRemovalState(),
        )
    if os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH):
        if install_snapshot is None:
            install_snapshot = _capture_root_snapshot(
                DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
                maximum=MAX_OWNER_APPROVAL_BYTES,
            )
        if install_snapshot is None:
            _fail("discord_token_retirement_install_snapshot_invalid")
        parsed_install = _decode_mapping(
            install_snapshot.raw,
            code="discord_token_install_receipt_invalid",
        )
        if parsed_install.get("receipt_sha256") != install_digest:
            _fail("discord_token_retirement_install_snapshot_drifted")
        _remove_exact_root_snapshot(
            install_snapshot,
            state=_RootRemovalState(),
        )
    if (
        os.path.lexists(DISCORD_TOKEN_PATH)
        or os.path.lexists(DISCORD_TOKEN_STAGE_PATH)
        or os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH)
    ):
        _fail("discord_token_retirement_removal_unconfirmed")
    retired_at = int(time.time())
    terminal_unsigned = {
        **{
            key: item
            for key, item in retirement.items()
            if key
            not in {
                "receipt_sha256",
                "ok",
                "state",
                "token_removed",
                "install_receipt_removed",
                "retired_at_unix",
            }
        },
        "ok": True,
        "state": "retired",
        "token_removed": True,
        "install_receipt_removed": True,
        "retired_at_unix": retired_at,
    }
    terminal = {
        **terminal_unsigned,
        "receipt_sha256": _sha256_json(terminal_unsigned),
    }
    _publish_root_payload(
        DISCORD_TOKEN_RETIREMENT_RECEIPT_PATH,
        _canonical_bytes(terminal),
        expected_previous_sha256=retirement_snapshot.sha256,
    )
    return terminal


def _remove_discord_install_artifact(
    path: Path,
    *,
    expected_device: int,
    expected_inode: int,
    coordinator_input: CoordinatorInput,
    allow_empty_root_stage: bool = False,
) -> None:
    if path not in {DISCORD_TOKEN_PATH, DISCORD_TOKEN_STAGE_PATH}:
        _fail("discord_install_cleanup_path_not_fixed")
    if not os.path.lexists(path):
        return
    item = path.lstat()
    allowed_owner = {coordinator_input.identities.edge_uid}
    allowed_group = {coordinator_input.identities.edge_gid}
    if allow_empty_root_stage:
        allowed_owner.add(0)
        allowed_group.add(0)
    if (
        not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or (item.st_dev, item.st_ino) != (expected_device, expected_inode)
        or item.st_uid not in allowed_owner
        or item.st_gid not in allowed_group
        or stat.S_IMODE(item.st_mode) & 0o077
        or item.st_nlink not in {1, 2}
        or not 0 <= item.st_size <= MAX_DISCORD_TOKEN_BYTES
        or allow_empty_root_stage
        and item.st_size != 0
    ):
        _fail("discord_install_cleanup_identity_drifted")
    path.unlink()
    _fsync_directory(path.parent)
    if os.path.lexists(path):
        _fail("discord_install_cleanup_unconfirmed")


def _retire_discord_install_journal(
    *,
    coordinator_input: CoordinatorInput,
    install_state: _DiscordInstallState,
) -> Mapping[str, Any]:
    if install_state.state == "installed":
        return _retire_discord_token_lease(
            coordinator_input=coordinator_input,
            install_receipt=install_state.value,
            installed=install_state.token_item,
            install_snapshot=install_state.snapshot,
        )
    if not _services_are_exactly_stopped_and_disabled():
        _fail("discord_token_retirement_services_not_stopped")
    existing = _load_discord_retirement(coordinator_input)
    if existing is not None:
        retirement, retirement_snapshot = existing
        if retirement["state"] == "retired":
            if any(
                os.path.lexists(path)
                for path in (
                    DISCORD_TOKEN_PATH,
                    DISCORD_TOKEN_STAGE_PATH,
                    DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
                )
            ):
                _fail("discord_token_terminal_retirement_drifted")
            return retirement
        if not _discord_install_state_matches_retirement(
            install_state,
            retirement,
        ):
            _fail("discord_token_retirement_journal_drifted")
    else:
        prepared_at = int(time.time())
        prepared_unsigned = {
            "schema": DISCORD_TOKEN_RETIREMENT_RECEIPT_SCHEMA,
            "ok": False,
            "state": "retirement_prepared",
            "release_sha": coordinator_input.revision,
            "coordinator_input_sha256": coordinator_input.sha256,
            "discord_token_install_receipt_sha256": (install_state.receipt_sha256),
            "token_path": str(DISCORD_TOKEN_PATH),
            "token_device": install_state.device,
            "token_inode": install_state.inode,
            "services_stopped_proven": True,
            "services_enabled": False,
            "token_removed": False,
            "install_receipt_removed": False,
            "prepared_at_unix": prepared_at,
            "retired_at_unix": None,
        }
        retirement = {
            **prepared_unsigned,
            "receipt_sha256": _sha256_json(prepared_unsigned),
        }
        publication = _publish_root_payload(
            DISCORD_TOKEN_RETIREMENT_RECEIPT_PATH,
            _canonical_bytes(retirement),
            expected_previous_sha256=None,
        )
        retirement_snapshot = publication.after

    # Re-read after the prepared retirement journal is durable.  Every
    # removal below is bound either by the install journal's exact inode or by
    # the DRA1 gate's observation of the only allowed empty intent stage.
    current = _load_discord_install_state(coordinator_input)
    if not _discord_install_state_matches_retirement(current, retirement):
        _fail("discord_token_retirement_install_state_drifted")
    if current.stage_item is not None:
        _remove_discord_install_artifact(
            DISCORD_TOKEN_STAGE_PATH,
            expected_device=current.stage_item.st_dev,
            expected_inode=current.stage_item.st_ino,
            coordinator_input=coordinator_input,
            allow_empty_root_stage=current.state == "install_intent",
        )
    if current.token_item is not None:
        _remove_discord_install_artifact(
            DISCORD_TOKEN_PATH,
            expected_device=current.token_item.st_dev,
            expected_inode=current.token_item.st_ino,
            coordinator_input=coordinator_input,
        )
    _remove_exact_root_snapshot(
        current.snapshot,
        state=_RootRemovalState(),
    )
    if any(
        os.path.lexists(path)
        for path in (
            DISCORD_TOKEN_PATH,
            DISCORD_TOKEN_STAGE_PATH,
            DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
        )
    ):
        _fail("discord_token_retirement_removal_unconfirmed")
    retired_at = int(time.time())
    terminal_unsigned = {
        **{
            key: item
            for key, item in retirement.items()
            if key
            not in {
                "receipt_sha256",
                "ok",
                "state",
                "token_removed",
                "install_receipt_removed",
                "retired_at_unix",
            }
        },
        "ok": True,
        "state": "retired",
        "token_removed": True,
        "install_receipt_removed": True,
        "retired_at_unix": retired_at,
    }
    terminal = {
        **terminal_unsigned,
        "receipt_sha256": _sha256_json(terminal_unsigned),
    }
    _publish_root_payload(
        DISCORD_TOKEN_RETIREMENT_RECEIPT_PATH,
        _canonical_bytes(terminal),
        expected_previous_sha256=retirement_snapshot.sha256,
    )
    return terminal


def _finish_prepared_discord_retirement_without_install_state(
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    existing = _load_discord_retirement(coordinator_input)
    if existing is None:
        _fail("discord_token_retirement_source_missing")
    retirement, snapshot = existing
    if retirement["state"] == "retired":
        return retirement
    if any(
        os.path.lexists(path)
        for path in (
            DISCORD_TOKEN_PATH,
            DISCORD_TOKEN_STAGE_PATH,
            DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
        )
    ):
        _fail("discord_token_retirement_install_state_lost")
    retired_at = int(time.time())
    terminal_unsigned = {
        **{
            key: item
            for key, item in retirement.items()
            if key
            not in {
                "receipt_sha256",
                "ok",
                "state",
                "token_removed",
                "install_receipt_removed",
                "retired_at_unix",
            }
        },
        "ok": True,
        "state": "retired",
        "token_removed": True,
        "install_receipt_removed": True,
        "retired_at_unix": retired_at,
    }
    terminal = {
        **terminal_unsigned,
        "receipt_sha256": _sha256_json(terminal_unsigned),
    }
    _publish_root_payload(
        DISCORD_TOKEN_RETIREMENT_RECEIPT_PATH,
        _canonical_bytes(terminal),
        expected_previous_sha256=snapshot.sha256,
    )
    return terminal


def _require_recovery_do_result(result: Any, *, code: str) -> None:
    if (
        str(getattr(result, "command_tag", "")).upper() != "DO"
        or tuple(getattr(result, "rows", ()))
        or tuple(getattr(result, "columns", ()))
    ):
        _fail(code, phase="recovery_database_cleanup")


def _load_staged_plan_for_recovery(
    snapshot: _RootFileSnapshot,
    *,
    coordinator_input: CoordinatorInput,
) -> FullCanaryPlan:
    try:
        plan = FullCanaryPlan.from_mapping(
            _decode_mapping(snapshot.raw, code="recovery_staged_plan_invalid")
        )
    except (TypeError, ValueError) as exc:
        raise CoordinatorError(
            "recovery_staged_plan_invalid",
            phase="recovery_lifecycle",
        ) from exc
    if plan.revision != coordinator_input.revision or snapshot.raw != _canonical_bytes(
        plan.to_mapping()
    ):
        _fail("recovery_staged_plan_drifted", phase="recovery_lifecycle")
    return plan


def _validate_recovery_stop_receipt(
    value: Mapping[str, Any],
    *,
    plan: FullCanaryPlan,
) -> tuple[str, str, str]:
    if not isinstance(value, Mapping):
        _fail("recovery_stop_receipt_invalid", phase="recovery_lifecycle")
    receipt = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    preclaim = receipt.get("preclaim_reconciliation")
    result = preclaim.get("result") if isinstance(preclaim, Mapping) else None
    preclaim_state = result.get("outcome") if isinstance(result, Mapping) else None
    receipt_path = receipt.get("receipt_path")
    if (
        receipt.get("stage") != "stopped"
        or receipt.get("revision") != plan.revision
        or receipt.get("full_canary_plan_sha256") != plan.sha256
        or receipt.get("units_enabled") is not False
        or receipt.get("receipt_sha256") != _sha256_json(unsigned)
        or not isinstance(receipt_path, str)
        or preclaim_state not in {"retired", "claimed", "not_preapproved"}
        or not isinstance(preclaim, Mapping)
        or preclaim.get("receipt_sha256") is None
    ):
        _fail("recovery_stop_receipt_invalid", phase="recovery_lifecycle")
    persisted = _stable_root_read(
        Path(receipt_path),
        maximum=MAX_COORDINATOR_INPUT_BYTES,
    )
    if persisted != _canonical_bytes(receipt):
        _fail("recovery_stop_receipt_not_durable", phase="recovery_lifecycle")
    _digest(
        preclaim["receipt_sha256"],
        "recovery_preclaim_receipt_digest_invalid",
    )
    return (
        str(receipt["receipt_sha256"]),
        str(preclaim["receipt_sha256"]),
        str(preclaim_state),
    )


def _remove_recovery_root_artifact(path: Path) -> None:
    snapshot = _capture_root_snapshot(path)
    if snapshot is not None:
        _remove_exact_root_snapshot(snapshot, state=_RootRemovalState())


def _remove_recovery_writer_artifact(writer_gid: int) -> None:
    snapshot = _capture_writer_snapshot(writer_gid)
    if snapshot is not None:
        _WriterPublication(
            writer_gid=writer_gid,
            before=None,
            after=snapshot,
        ).rollback()


def _publish_preplan_stopped_report(
    *,
    coordinator_input: CoordinatorInput,
    lease: Mapping[str, Any],
) -> Mapping[str, Any]:
    if (
        any(
            os.path.lexists(path)
            for path in (
                DEFAULT_PLAN_PATH,
                DEFAULT_STAGED_PLAN_PATH,
                DEFAULT_WRITER_CONFIG_SOURCE,
                OWNER_APPROVAL_REQUEST_PATH,
                CANARY_BOOTSTRAP_CREDENTIAL_PATH,
            )
        )
        or not _services_are_exactly_stopped_and_disabled()
    ):
        _fail("recovery_preplan_truth_unresolved", phase="recovery_lifecycle")
    unsigned = {
        "schema": PREPLAN_STOPPED_REPORT_SCHEMA,
        "ok": True,
        "state": "preplan_stopped",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "credential_prepare_approval_sha256": lease[
            "credential_prepare_approval_sha256"
        ],
        "owner_subject_sha256": lease["owner_subject_sha256"],
        "base_plan_sha256": coordinator_input.base_plan.sha256,
        "runtime_plan_absent": True,
        "staged_plan_absent": True,
        "staged_writer_config_absent": True,
        "owner_approval_request_absent": True,
        "bootstrap_scope_activation_artifacts_absent": True,
        "migration_owner_membership_removed": True,
        "bootstrap_login_password_disabled": True,
        "bootstrap_credential_removed": True,
        "services_stopped_proven": True,
        "services_enabled": False,
        "completed_at_unix": int(time.time()),
    }
    report = {**unsigned, "report_sha256": _sha256_json(unsigned)}
    existing = _capture_root_snapshot(
        PREPLAN_STOPPED_REPORT_PATH,
        maximum=MAX_OWNER_APPROVAL_BYTES,
    )
    if existing is None:
        _publish_root_payload(
            PREPLAN_STOPPED_REPORT_PATH,
            _canonical_bytes(report),
            expected_previous_sha256=None,
        )
        return report
    prior = _decode_mapping(existing.raw, code="recovery_preplan_report_invalid")
    prior_unsigned = {
        key: item for key, item in prior.items() if key != "report_sha256"
    }
    stable_names = set(unsigned).difference({"completed_at_unix"})
    if (
        set(prior) != set(report)
        or any(prior.get(name) != unsigned[name] for name in stable_names)
        or prior.get("report_sha256") != _sha256_json(prior_unsigned)
    ):
        _fail("recovery_preplan_report_drifted", phase="recovery_lifecycle")
    return prior


_RECOVERY_RECEIPT_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "recovery_gate_sha256",
    "owner_recovery_ack_sha256",
    "stale_process_lease_sha256",
    "process_termination_proven",
    "process_lock_acquired",
    "process_lease_removed",
    "canonical_stop_receipt_sha256",
    "preplan_stopped_report_sha256",
    "preclaim_reconciliation_receipt_sha256",
    "preclaim_reconciliation_state",
    "admin_session_closed",
    "migration_owner_membership_removed",
    "bootstrap_login_password_disabled",
    "bootstrap_credential_removed",
    "discord_token_removed",
    "discord_install_receipt_removed",
    "discord_retirement_receipt_sha256",
    "services_stopped_proven",
    "services_enabled",
    "safe_to_delete_temporary_admin",
    "completed_at_unix",
    "receipt_sha256",
})


def _parse_legacy_recovery_receipt(
    value: Any,
    *,
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECOVERY_RECEIPT_FIELDS:
        _fail("recovery_receipt_invalid", phase="recovery_terminal")
    raw = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    canonical = raw["canonical_stop_receipt_sha256"]
    preplan = raw["preplan_stopped_report_sha256"]
    if (
        raw["schema"] != LEGACY_RECOVERY_RECEIPT_SCHEMA
        or raw["ok"] is not True
        or raw["state"] != "recovered"
        or raw["release_sha"] != coordinator_input.revision
        or raw["coordinator_input_sha256"] != coordinator_input.sha256
        or (canonical is None) == (preplan is None)
        or any(
            raw[name] is not True
            for name in (
                "process_termination_proven",
                "process_lock_acquired",
                "process_lease_removed",
                "admin_session_closed",
                "migration_owner_membership_removed",
                "bootstrap_login_password_disabled",
                "bootstrap_credential_removed",
                "discord_token_removed",
                "discord_install_receipt_removed",
                "services_stopped_proven",
                "safe_to_delete_temporary_admin",
            )
        )
        or raw["services_enabled"] is not False
        or raw["receipt_sha256"] != _sha256_json(unsigned)
    ):
        _fail("recovery_receipt_invalid", phase="recovery_terminal")
    for name in (
        "credential_prepare_approval_sha256",
        "owner_subject_sha256",
        "recovery_gate_sha256",
        "owner_recovery_ack_sha256",
        "stale_process_lease_sha256",
        "discord_retirement_receipt_sha256",
    ):
        _digest(raw[name], "recovery_receipt_digest_invalid")
    if canonical is None:
        _digest(preplan, "recovery_preplan_receipt_digest_invalid")
        if (
            raw["preclaim_reconciliation_receipt_sha256"] is not None
            or raw["preclaim_reconciliation_state"] is not None
        ):
            _fail("recovery_receipt_preplan_matrix_invalid")
    else:
        _digest(canonical, "recovery_stop_receipt_digest_invalid")
        _digest(
            raw["preclaim_reconciliation_receipt_sha256"],
            "recovery_preclaim_receipt_digest_invalid",
        )
        if raw["preclaim_reconciliation_state"] not in {
            "retired",
            "claimed",
            "not_preapproved",
        }:
            _fail("recovery_receipt_preclaim_state_invalid")
    return raw


_RECOVERY_CAUSAL_STATE_FIELDS = frozenset({
    "schema",
    "discord_token_state",
    "discord_token_install_receipt_sha256",
    "token_device",
    "token_inode",
    "discord_retirement_receipt_sha256",
    "causal_state_sha256",
})
_RECOVERY_CAUSAL_STATE_SCHEMA = "muncho-full-canary-recovery-causal-state.v1"

_RECOVERY_TAKEOVER_GATE_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "predecessor_kind",
    "predecessor_schema",
    "predecessor_journal_sha256",
    "predecessor_generation",
    "original_run_process_lease_sha256",
    "causal_recovery_state_sha256",
    "target_pid",
    "target_process_start_time_ticks",
    "target_boot_id_sha256",
    "target_boot_time_ns",
    "target_uid",
    "target_gid",
    "target_module_origin",
    "target_module_sha256",
    "target_process_exe_sha256",
    "target_process_cmdline_sha256",
    "target_process_identity_state",
    "discord_token_state",
    "discord_token_install_receipt_sha256",
    "token_device",
    "token_inode",
    "discord_retirement_receipt_sha256",
    "db_secret_accepted",
    "frame_schema",
    "observed_at_unix",
    "expires_at_unix",
    "gate_sha256",
})

_RECOVERY_TAKEOVER_ACK_FIELDS = frozenset({
    "schema",
    "scope",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "recovery_takeover_gate_sha256",
    "predecessor_kind",
    "predecessor_schema",
    "predecessor_journal_sha256",
    "predecessor_generation",
    "original_run_process_lease_sha256",
    "causal_recovery_state_sha256",
    "target_pid",
    "target_process_start_time_ticks",
    "target_boot_id_sha256",
    "target_boot_time_ns",
    "target_uid",
    "target_gid",
    "target_module_origin",
    "target_module_sha256",
    "target_process_exe_sha256",
    "target_process_cmdline_sha256",
    "target_process_identity_state",
    "discord_token_state",
    "discord_token_install_receipt_sha256",
    "token_device",
    "token_inode",
    "discord_retirement_receipt_sha256",
    "nonce_sha256",
    "approved_at_unix",
    "expires_at_unix",
    "ack_sha256",
})

_RECOVERY_WORKER_IDENTITY_FIELDS = (
    "recovery_worker_pid",
    "recovery_worker_process_start_time_ticks",
    "recovery_worker_boot_id_sha256",
    "recovery_worker_boot_time_ns",
    "recovery_worker_uid",
    "recovery_worker_gid",
    "recovery_worker_module_origin",
    "recovery_worker_module_sha256",
    "recovery_worker_process_exe_sha256",
    "recovery_worker_process_cmdline_sha256",
)

_RECOVERY_WORKER_LEASE_FIELDS = frozenset({
    "schema",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "original_run_process_lease",
    "original_run_process_lease_sha256",
    "causal_recovery_state",
    "causal_recovery_state_sha256",
    "predecessor_kind",
    "predecessor_schema",
    "predecessor_journal_sha256",
    "predecessor_generation",
    "recovery_generation",
    "transition_seq",
    "previous_transition_sha256",
    "recovery_takeover_gate_sha256",
    "owner_recovery_takeover_ack_sha256",
    *_RECOVERY_WORKER_IDENTITY_FIELDS,
    "predecessor_termination_proven",
    "predecessor_process_lock_acquired",
    "predecessor_journal_replaced",
    "claimed_at_unix",
    "updated_at_unix",
    "owner_authority_expires_at_unix",
    "worker_lease_sha256",
})

_RECOVERY_SECRET_GATE_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "recovery_takeover_gate_sha256",
    "owner_recovery_takeover_ack_sha256",
    "predecessor_kind",
    "predecessor_journal_sha256",
    "predecessor_generation",
    "original_run_process_lease_sha256",
    "causal_recovery_state_sha256",
    "recovery_worker_lease_sha256",
    "recovery_worker_state",
    "recovery_worker_transition_seq",
    *_RECOVERY_WORKER_IDENTITY_FIELDS,
    "database_host",
    "database_port",
    "database_name",
    "tls_server_name",
    "tls_peer_certificate_sha256",
    "admin_frame_schema",
    "gate_nonce_sha256",
    "expires_at_unix",
    "gate_sha256",
})

_RECOVERY_CLEANUP_FIELDS = frozenset({
    "canonical_stop_receipt_sha256",
    "preplan_stopped_report_sha256",
    "preclaim_reconciliation_receipt_sha256",
    "preclaim_reconciliation_state",
    "admin_frame_zeroized",
    "admin_session_closed",
    "migration_owner_membership_removed",
    "bootstrap_login_password_disabled",
    "bootstrap_credential_removed",
    "discord_token_removed",
    "discord_install_receipt_removed",
    "discord_retirement_receipt_sha256",
    "services_stopped_proven",
    "services_enabled",
})

_RECOVERY_WORKER_COMPLETION_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "original_run_process_lease",
    "original_run_process_lease_sha256",
    "causal_recovery_state_sha256",
    "predecessor_kind",
    "predecessor_journal_sha256",
    "predecessor_generation",
    "recovery_generation",
    "recovery_takeover_gate_sha256",
    "owner_recovery_takeover_ack_sha256",
    "recovery_worker_lease_sha256",
    *_RECOVERY_WORKER_IDENTITY_FIELDS,
    "predecessor_termination_proven",
    "predecessor_process_lock_acquired",
    "predecessor_journal_replaced",
    *_RECOVERY_CLEANUP_FIELDS,
    "recovery_worker_exit_proven",
    "safe_to_delete_temporary_admin",
    "cleanup_completed_at_unix",
    "completion_sha256",
})

_RECOVERY_RECEIPT_V2_FIELDS = frozenset({
    *(
        _RECOVERY_WORKER_COMPLETION_FIELDS
        - {
            "completion_sha256",
            "recovery_worker_exit_proven",
            "safe_to_delete_temporary_admin",
        }
    ),
    "recovery_worker_completion_sha256",
    "recovery_worker_lock_acquired",
    "recovery_worker_exit_proven",
    "safe_to_delete_temporary_admin",
    "finalized_at_unix",
    "receipt_sha256",
})

_RECOVERY_CONCURRENT_LOSER_RECEIPT_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "recovery_takeover_gate_sha256",
    "owner_recovery_takeover_ack_sha256",
    "predecessor_kind",
    "predecessor_schema",
    "predecessor_journal_sha256",
    "predecessor_generation",
    "original_run_process_lease_sha256",
    "target_pid",
    "target_process_start_time_ticks",
    "target_boot_id_sha256",
    "target_boot_time_ns",
    "target_uid",
    "target_gid",
    "target_module_origin",
    "target_module_sha256",
    "target_process_exe_sha256",
    "target_process_cmdline_sha256",
    "target_signal_attempted_by_loser",
    "target_termination_proven_by_loser",
    "process_lock_acquired_by_loser",
    "journal_cas_attempted_by_loser",
    "journal_cas_succeeded_by_loser",
    "observed_successor_schema",
    "observed_successor_state",
    "observed_successor_journal_sha256",
    "observed_successor_generation",
    "observed_successor_worker_pid",
    "observed_successor_worker_process_start_time_ticks",
    "observed_successor_worker_boot_id_sha256",
    "observed_successor_worker_module_sha256",
    "observed_successor_worker_process_exe_sha256",
    "observed_successor_worker_process_cmdline_sha256",
    "secret_gate_emitted_by_loser",
    "admin_frame_bytes_received_by_loser",
    "admin_session_opened_by_loser",
    "admin_credential_mutation_performed_by_loser",
    "worker_lease_published_by_loser",
    "retryable",
    "completed_at_unix",
    "receipt_sha256",
})

_RECOVERY_FINALIZE_PENDING_RECEIPT_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "credential_prepare_approval_sha256",
    "owner_subject_sha256",
    "ephemeral_admin_username",
    "recovery_worker_completion_sha256",
    *_RECOVERY_WORKER_IDENTITY_FIELDS,
    "completion_admin_authority_may_have_been_used",
    "completion_admin_frame_zeroized",
    "completion_admin_session_closed",
    "worker_identity_state",
    "target_signal_attempted_by_finalizer",
    "target_termination_proven_by_finalizer",
    "process_lock_acquired_by_finalizer",
    "completion_cas_attempted_by_finalizer",
    "completion_cas_succeeded_by_finalizer",
    "observed_journal_sha256",
    "secret_gate_emitted_by_finalizer",
    "admin_frame_bytes_received_by_finalizer",
    "admin_session_opened_by_finalizer",
    "admin_credential_mutation_performed_by_finalizer",
    "retryable",
    "completed_at_unix",
    "receipt_sha256",
})


def _recovery_predecessor_kind_generation_valid(
    value: Mapping[str, Any],
) -> bool:
    return bool(
        value.get("predecessor_kind") == "run_process_lease"
        and value.get("predecessor_generation") == 0
        or value.get("predecessor_kind") == "recovery_worker_lease"
        and type(value.get("predecessor_generation")) is int
        and value["predecessor_generation"] >= 1
    )


def _recovery_predecessor_kind_schema_generation_valid(
    value: Mapping[str, Any],
) -> bool:
    return bool(
        _recovery_predecessor_kind_generation_valid(value)
        and (
            value["predecessor_kind"] == "run_process_lease"
            and value.get("predecessor_schema")
            == "muncho-full-canary-coordinator-process-lease.v1"
            or value["predecessor_kind"] == "recovery_worker_lease"
            and value.get("predecessor_schema") == RECOVERY_WORKER_LEASE_SCHEMA
        )
    )


def _parse_recovery_causal_state(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECOVERY_CAUSAL_STATE_FIELDS:
        _fail("recovery_causal_state_invalid", phase="recovery_preflight")
    raw = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in raw.items() if key != "causal_state_sha256"}
    state = raw["discord_token_state"]
    positive_pair = (
        type(raw["token_device"]) is int
        and raw["token_device"] > 0
        and type(raw["token_inode"]) is int
        and raw["token_inode"] > 0
    )
    null_pair = raw["token_device"] is None and raw["token_inode"] is None
    if (
        raw["schema"] != _RECOVERY_CAUSAL_STATE_SCHEMA
        or state not in {"installed", "retirement_prepared", "retired"}
        or (state == "installed" and not positive_pair)
        or (state != "installed" and not (positive_pair or null_pair))
        or (state == "installed") != (raw["discord_retirement_receipt_sha256"] is None)
        or raw["causal_state_sha256"] != _sha256_json(unsigned)
    ):
        _fail("recovery_causal_state_invalid", phase="recovery_preflight")
    _digest(
        raw["discord_token_install_receipt_sha256"],
        "recovery_causal_install_digest_invalid",
    )
    if raw["discord_retirement_receipt_sha256"] is not None:
        _digest(
            raw["discord_retirement_receipt_sha256"],
            "recovery_causal_retirement_digest_invalid",
        )
    return raw


def _observe_recovery_causal_state(
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    retirement_loaded = _load_discord_retirement(coordinator_input)
    if (
        retirement_loaded is not None
        and retirement_loaded[0]["state"] == "retirement_prepared"
    ):
        retirement, _snapshot, _install = _load_prepared_discord_retirement_source(
            coordinator_input,
            require_terminal_install=True,
        )
        state = "retirement_prepared"
        install_sha256 = retirement["discord_token_install_receipt_sha256"]
        device = retirement["token_device"]
        inode = retirement["token_inode"]
        retirement_sha256 = retirement["receipt_sha256"]
    elif os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH):
        receipt, identity, _snapshot = load_discord_token_install_receipt(
            coordinator_input
        )
        if identity is None:
            _fail(
                "recovery_discord_token_identity_missing",
                phase="recovery_preflight",
            )
        state = "installed"
        install_sha256 = receipt["receipt_sha256"]
        device = identity.st_dev
        inode = identity.st_ino
        retirement_sha256 = None
    else:
        if (
            retirement_loaded is None
            or retirement_loaded[0]["state"] != "retired"
            or any(
                os.path.lexists(path)
                for path in (
                    DISCORD_TOKEN_PATH,
                    DISCORD_TOKEN_STAGE_PATH,
                    DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
                )
            )
        ):
            _fail(
                "recovery_discord_token_state_invalid",
                phase="recovery_preflight",
            )
        retirement = retirement_loaded[0]
        state = "retired"
        install_sha256 = retirement["discord_token_install_receipt_sha256"]
        device = retirement["token_device"]
        inode = retirement["token_inode"]
        retirement_sha256 = retirement["receipt_sha256"]
    unsigned = {
        "schema": _RECOVERY_CAUSAL_STATE_SCHEMA,
        "discord_token_state": state,
        "discord_token_install_receipt_sha256": install_sha256,
        "token_device": device,
        "token_inode": inode,
        "discord_retirement_receipt_sha256": retirement_sha256,
    }
    return _parse_recovery_causal_state({
        **unsigned,
        "causal_state_sha256": _sha256_json(unsigned),
    })


def _validate_recovery_worker_identity_fields(value: Mapping[str, Any]) -> None:
    for name in (
        "recovery_worker_pid",
        "recovery_worker_process_start_time_ticks",
        "recovery_worker_boot_time_ns",
        "recovery_worker_uid",
        "recovery_worker_gid",
    ):
        if type(value.get(name)) is not int:
            _fail("recovery_worker_identity_invalid", phase="recovery_preflight")
    if (
        value["recovery_worker_pid"] <= 1
        or value["recovery_worker_process_start_time_ticks"] <= 0
        or value["recovery_worker_boot_time_ns"] < 0
        or value["recovery_worker_uid"] != 0
        or value["recovery_worker_gid"] != 0
        or not isinstance(value.get("recovery_worker_module_origin"), str)
        or not value["recovery_worker_module_origin"]
    ):
        _fail("recovery_worker_identity_invalid", phase="recovery_preflight")
    for name in (
        "recovery_worker_boot_id_sha256",
        "recovery_worker_module_sha256",
        "recovery_worker_process_exe_sha256",
        "recovery_worker_process_cmdline_sha256",
    ):
        _digest(value.get(name), "recovery_worker_identity_digest_invalid")


def _parse_recovery_worker_lease(
    value: Any,
    *,
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECOVERY_WORKER_LEASE_FIELDS:
        _fail("recovery_worker_lease_invalid", phase="recovery_preflight")
    raw = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in raw.items() if key != "worker_lease_sha256"}
    original = raw["original_run_process_lease"]
    parsed_original = _parse_process_lease(
        original,
        coordinator_input=coordinator_input,
    )
    causal = _parse_recovery_causal_state(raw["causal_recovery_state"])
    expected_state = {
        1: "claimed_awaiting_admin",
        2: "admin_authority_may_be_in_use",
    }.get(raw["transition_seq"])
    if (
        raw["schema"] != RECOVERY_WORKER_LEASE_SCHEMA
        or expected_state is None
        or raw["state"] != expected_state
        or raw["release_sha"] != coordinator_input.revision
        or raw["coordinator_input_sha256"] != coordinator_input.sha256
        or raw["credential_prepare_approval_sha256"]
        != parsed_original["credential_prepare_approval_sha256"]
        or raw["owner_subject_sha256"] != parsed_original["owner_subject_sha256"]
        or raw["ephemeral_admin_username"]
        != parsed_original["ephemeral_admin_username"]
        or raw["original_run_process_lease_sha256"] != parsed_original["lease_sha256"]
        or raw["causal_recovery_state_sha256"] != causal["causal_state_sha256"]
        or not _recovery_predecessor_kind_schema_generation_valid(raw)
        or type(raw["predecessor_generation"]) is not int
        or raw["predecessor_generation"] < 0
        or type(raw["recovery_generation"]) is not int
        or raw["recovery_generation"] != raw["predecessor_generation"] + 1
        or raw["predecessor_termination_proven"] is not True
        or raw["predecessor_process_lock_acquired"] is not True
        or raw["predecessor_journal_replaced"] is not True
        or type(raw["claimed_at_unix"]) is not int
        or type(raw["updated_at_unix"]) is not int
        or type(raw["owner_authority_expires_at_unix"]) is not int
        or not 0 <= raw["claimed_at_unix"] <= raw["updated_at_unix"]
        or not 1
        <= raw["owner_authority_expires_at_unix"] - raw["claimed_at_unix"]
        <= RECOVERY_WORKER_LEASE_MAX_SECONDS
        or raw["updated_at_unix"] > raw["owner_authority_expires_at_unix"]
        or raw["worker_lease_sha256"] != _sha256_json(unsigned)
    ):
        _fail("recovery_worker_lease_invalid", phase="recovery_preflight")
    for name in (
        "predecessor_journal_sha256",
        "previous_transition_sha256",
        "recovery_takeover_gate_sha256",
        "owner_recovery_takeover_ack_sha256",
    ):
        _digest(raw[name], "recovery_worker_lease_digest_invalid")
    if raw["transition_seq"] == 1 and (
        raw["previous_transition_sha256"] != raw["predecessor_journal_sha256"]
    ):
        _fail("recovery_worker_transition_invalid", phase="recovery_preflight")
    _validate_recovery_worker_identity_fields(raw)
    return raw


def _parse_recovery_worker_completion(
    value: Any,
    *,
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _RECOVERY_WORKER_COMPLETION_FIELDS
    ):
        _fail("recovery_worker_completion_invalid", phase="recovery_terminal")
    raw = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in raw.items() if key != "completion_sha256"}
    original = _parse_process_lease(
        raw["original_run_process_lease"],
        coordinator_input=coordinator_input,
    )
    canonical = raw["canonical_stop_receipt_sha256"]
    preplan = raw["preplan_stopped_report_sha256"]
    if (
        raw["schema"] != RECOVERY_WORKER_COMPLETION_SCHEMA
        or raw["ok"] is not False
        or raw["state"] != "cleanup_complete_awaiting_worker_exit"
        or raw["release_sha"] != coordinator_input.revision
        or raw["coordinator_input_sha256"] != coordinator_input.sha256
        or raw["original_run_process_lease_sha256"] != original["lease_sha256"]
        or raw["credential_prepare_approval_sha256"]
        != original["credential_prepare_approval_sha256"]
        or raw["owner_subject_sha256"] != original["owner_subject_sha256"]
        or raw["ephemeral_admin_username"] != original["ephemeral_admin_username"]
        or not _recovery_predecessor_kind_generation_valid(raw)
        or type(raw["predecessor_generation"]) is not int
        or type(raw["recovery_generation"]) is not int
        or raw["recovery_generation"] != raw["predecessor_generation"] + 1
        or raw["predecessor_termination_proven"] is not True
        or raw["predecessor_process_lock_acquired"] is not True
        or raw["predecessor_journal_replaced"] is not True
        or (canonical is None) == (preplan is None)
        or raw["admin_frame_zeroized"] is not True
        or raw["admin_session_closed"] is not True
        or any(
            raw[name] is not True
            for name in (
                "migration_owner_membership_removed",
                "bootstrap_login_password_disabled",
                "bootstrap_credential_removed",
                "discord_token_removed",
                "discord_install_receipt_removed",
                "services_stopped_proven",
            )
        )
        or raw["services_enabled"] is not False
        or raw["recovery_worker_exit_proven"] is not False
        or raw["safe_to_delete_temporary_admin"] is not False
        or type(raw["cleanup_completed_at_unix"]) is not int
        or raw["cleanup_completed_at_unix"] < 0
        or raw["completion_sha256"] != _sha256_json(unsigned)
    ):
        _fail("recovery_worker_completion_invalid", phase="recovery_terminal")
    for name in (
        "credential_prepare_approval_sha256",
        "owner_subject_sha256",
        "original_run_process_lease_sha256",
        "causal_recovery_state_sha256",
        "predecessor_journal_sha256",
        "recovery_takeover_gate_sha256",
        "owner_recovery_takeover_ack_sha256",
        "recovery_worker_lease_sha256",
        "discord_retirement_receipt_sha256",
    ):
        _digest(raw[name], "recovery_worker_completion_digest_invalid")
    _validate_recovery_worker_identity_fields(raw)
    _validate_recovery_cleanup_matrix(raw)
    return raw


def _validate_recovery_cleanup_matrix(raw: Mapping[str, Any]) -> None:
    canonical = raw["canonical_stop_receipt_sha256"]
    preplan = raw["preplan_stopped_report_sha256"]
    if canonical is None:
        _digest(preplan, "recovery_preplan_receipt_digest_invalid")
        if (
            raw["preclaim_reconciliation_receipt_sha256"] is not None
            or raw["preclaim_reconciliation_state"] is not None
        ):
            _fail("recovery_receipt_preplan_matrix_invalid")
    else:
        _digest(canonical, "recovery_stop_receipt_digest_invalid")
        _digest(
            raw["preclaim_reconciliation_receipt_sha256"],
            "recovery_preclaim_receipt_digest_invalid",
        )
        if raw["preclaim_reconciliation_state"] not in {
            "retired",
            "claimed",
            "not_preapproved",
        }:
            _fail("recovery_receipt_preclaim_state_invalid")


def _project_recovery_worker_completion_from_receipt(
    raw: Mapping[str, Any],
    *,
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    """Reconstruct and authenticate the exact completion sealed by final v2."""

    projected_unsigned = {
        key: copy.deepcopy(raw[key])
        for key in _RECOVERY_WORKER_COMPLETION_FIELDS
        if key
        not in {
            "schema",
            "ok",
            "state",
            "recovery_worker_exit_proven",
            "safe_to_delete_temporary_admin",
            "completion_sha256",
        }
    }
    projected_unsigned.update({
        "schema": RECOVERY_WORKER_COMPLETION_SCHEMA,
        "ok": False,
        "state": "cleanup_complete_awaiting_worker_exit",
        "recovery_worker_exit_proven": False,
        "safe_to_delete_temporary_admin": False,
    })
    projected_sha256 = _sha256_json(projected_unsigned)
    if raw["recovery_worker_completion_sha256"] != projected_sha256:
        _fail(
            "recovery_receipt_completion_projection_invalid",
            phase="recovery_terminal",
        )
    return _parse_recovery_worker_completion(
        {**projected_unsigned, "completion_sha256": projected_sha256},
        coordinator_input=coordinator_input,
    )


def _parse_recovery_receipt_v2(
    value: Any,
    *,
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECOVERY_RECEIPT_V2_FIELDS:
        _fail("recovery_receipt_invalid", phase="recovery_terminal")
    raw = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    original = _parse_process_lease(
        raw["original_run_process_lease"],
        coordinator_input=coordinator_input,
    )
    if (
        raw["schema"] != RECOVERY_RECEIPT_SCHEMA
        or raw["ok"] is not True
        or raw["state"] != "recovered"
        or raw["release_sha"] != coordinator_input.revision
        or raw["coordinator_input_sha256"] != coordinator_input.sha256
        or raw["original_run_process_lease_sha256"] != original["lease_sha256"]
        or raw["credential_prepare_approval_sha256"]
        != original["credential_prepare_approval_sha256"]
        or raw["owner_subject_sha256"] != original["owner_subject_sha256"]
        or raw["ephemeral_admin_username"] != original["ephemeral_admin_username"]
        or not _recovery_predecessor_kind_generation_valid(raw)
        or type(raw["predecessor_generation"]) is not int
        or type(raw["recovery_generation"]) is not int
        or raw["recovery_generation"] != raw["predecessor_generation"] + 1
        or raw["predecessor_termination_proven"] is not True
        or raw["predecessor_process_lock_acquired"] is not True
        or raw["predecessor_journal_replaced"] is not True
        or (raw["canonical_stop_receipt_sha256"] is None)
        == (raw["preplan_stopped_report_sha256"] is None)
        or raw["admin_frame_zeroized"] is not True
        or raw["admin_session_closed"] is not True
        or any(
            raw[name] is not True
            for name in (
                "migration_owner_membership_removed",
                "bootstrap_login_password_disabled",
                "bootstrap_credential_removed",
                "discord_token_removed",
                "discord_install_receipt_removed",
                "services_stopped_proven",
            )
        )
        or raw["services_enabled"] is not False
        or raw["recovery_worker_lock_acquired"] is not True
        or raw["recovery_worker_exit_proven"] is not True
        or raw["safe_to_delete_temporary_admin"] is not True
        or type(raw["finalized_at_unix"]) is not int
        or raw["finalized_at_unix"] < raw["cleanup_completed_at_unix"]
        or raw["receipt_sha256"] != _sha256_json(unsigned)
    ):
        _fail("recovery_receipt_invalid", phase="recovery_terminal")
    for name in (
        "credential_prepare_approval_sha256",
        "owner_subject_sha256",
        "original_run_process_lease_sha256",
        "causal_recovery_state_sha256",
        "predecessor_journal_sha256",
        "recovery_takeover_gate_sha256",
        "owner_recovery_takeover_ack_sha256",
        "recovery_worker_lease_sha256",
        "recovery_worker_completion_sha256",
        "discord_retirement_receipt_sha256",
    ):
        _digest(raw[name], "recovery_receipt_digest_invalid")
    _validate_recovery_worker_identity_fields(raw)
    _validate_recovery_cleanup_matrix(raw)
    _project_recovery_worker_completion_from_receipt(
        raw,
        coordinator_input=coordinator_input,
    )
    return raw


def _current_recovery_worker_identity(
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    pid = os.getpid()
    boot_sha256, boot_time_ns = boot_identity()
    module_origin, module_sha256 = _sealed_coordinator_module_identity(
        coordinator_input
    )
    expected = _expected_command_cmdline(coordinator_input, command="recover")
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        owner = Path(f"/proc/{pid}").stat()
    except OSError as exc:
        raise CoordinatorError(
            "recovery_worker_identity_unavailable",
            phase="recovery_process",
        ) from exc
    if cmdline != expected or owner.st_uid != 0 or owner.st_gid != 0:
        _fail("recovery_worker_identity_invalid", phase="recovery_process")
    return {
        "recovery_worker_pid": pid,
        "recovery_worker_process_start_time_ticks": process_start_time_ticks(pid),
        "recovery_worker_boot_id_sha256": boot_sha256,
        "recovery_worker_boot_time_ns": boot_time_ns,
        "recovery_worker_uid": owner.st_uid,
        "recovery_worker_gid": owner.st_gid,
        "recovery_worker_module_origin": module_origin,
        "recovery_worker_module_sha256": module_sha256,
        "recovery_worker_process_exe_sha256": _process_executable_sha256(pid),
        "recovery_worker_process_cmdline_sha256": _sha256_bytes(cmdline),
    }


def _run_lease_recovery_target(lease: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "target_pid": lease["pid"],
        "target_process_start_time_ticks": lease["process_start_time_ticks"],
        "target_boot_id_sha256": lease["boot_id_sha256"],
        "target_boot_time_ns": lease["boot_time_ns"],
        "target_uid": 0,
        "target_gid": 0,
        "target_module_origin": lease["module_origin"],
        "target_module_sha256": lease["module_sha256"],
        "target_process_exe_sha256": lease["process_exe_sha256"],
        "target_process_cmdline_sha256": lease["process_cmdline_sha256"],
        "target_command": "run",
    }


def _worker_recovery_target(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "target_pid": value["recovery_worker_pid"],
        "target_process_start_time_ticks": value[
            "recovery_worker_process_start_time_ticks"
        ],
        "target_boot_id_sha256": value["recovery_worker_boot_id_sha256"],
        "target_boot_time_ns": value["recovery_worker_boot_time_ns"],
        "target_uid": value["recovery_worker_uid"],
        "target_gid": value["recovery_worker_gid"],
        "target_module_origin": value["recovery_worker_module_origin"],
        "target_module_sha256": value["recovery_worker_module_sha256"],
        "target_process_exe_sha256": value["recovery_worker_process_exe_sha256"],
        "target_process_cmdline_sha256": value[
            "recovery_worker_process_cmdline_sha256"
        ],
        "target_command": "recover",
    }


def _target_public_identity(target: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: target[key]
        for key in (
            "target_pid",
            "target_process_start_time_ticks",
            "target_boot_id_sha256",
            "target_boot_time_ns",
            "target_uid",
            "target_gid",
            "target_module_origin",
            "target_module_sha256",
            "target_process_exe_sha256",
            "target_process_cmdline_sha256",
        )
    }


def _validate_recovery_target_public_fields(value: Mapping[str, Any]) -> None:
    for name in (
        "target_pid",
        "target_process_start_time_ticks",
        "target_boot_time_ns",
        "target_uid",
        "target_gid",
    ):
        if type(value.get(name)) is not int:
            _fail("recovery_target_identity_invalid", phase="recovery_preflight")
    if (
        value["target_pid"] <= 1
        or value["target_process_start_time_ticks"] <= 0
        or value["target_boot_time_ns"] < 0
        or value["target_uid"] != 0
        or value["target_gid"] != 0
        or not isinstance(value.get("target_module_origin"), str)
        or not value["target_module_origin"]
    ):
        _fail("recovery_target_identity_invalid", phase="recovery_preflight")


def _recovery_target_is_exactly_alive(
    target: Mapping[str, Any],
    *,
    coordinator_input: CoordinatorInput,
) -> bool:
    current_boot_sha256, _boot_time_ns = boot_identity()
    if current_boot_sha256 != target["target_boot_id_sha256"]:
        return False
    pid = target["target_pid"]
    try:
        observed_start = process_start_time_ticks(pid)
    except (FileNotFoundError, ProcessLookupError):
        return False
    except (OSError, RuntimeError) as exc:
        raise CoordinatorError(
            "recovery_process_identity_observation_failed",
            phase="recovery_process",
        ) from exc
    try:
        owner = Path(f"/proc/{pid}").stat()
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError as exc:
        raise CoordinatorError(
            "recovery_process_identity_observation_failed",
            phase="recovery_process",
        ) from exc
    if observed_start != target["target_process_start_time_ticks"]:
        # PID reuse means the exact target is absent.  Never inspect or signal
        # the replacement as the predecessor recovery process.
        return False
    if owner.st_uid != target["target_uid"] or owner.st_gid != target["target_gid"]:
        _fail("recovery_process_identity_drifted", phase="recovery_process")
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        process_state = stat_line[stat_line.rfind(")") + 2 :].split(" ", 1)[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    except (OSError, UnicodeError, ValueError, IndexError) as exc:
        raise CoordinatorError(
            "recovery_process_identity_observation_failed",
            phase="recovery_process",
        ) from exc
    if re.fullmatch(r"[A-Z]", process_state) is None:
        _fail(
            "recovery_process_identity_observation_failed",
            phase="recovery_process",
        )
    if process_state in {"Z", "X"}:
        # The exact task has exited; an unreaped zombie must not be treated as
        # an executable drift or signalled as a live recovery worker.
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        expected = _expected_command_cmdline(
            coordinator_input,
            command=target["target_command"],
        )
        exe_sha256 = _process_executable_sha256(pid)
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError as exc:
        raise CoordinatorError(
            "recovery_process_identity_observation_failed",
            phase="recovery_process",
        ) from exc
    if (
        cmdline != expected
        or _sha256_bytes(cmdline) != target["target_process_cmdline_sha256"]
        or exe_sha256 != target["target_process_exe_sha256"]
    ):
        _fail(
            "recovery_process_exec_identity_drifted",
            phase="recovery_process",
        )
    return True


def _open_exact_recovery_target_pidfd(
    *,
    coordinator_input: CoordinatorInput,
    target: Mapping[str, Any],
) -> int | None:
    opener = getattr(os, "pidfd_open", None)
    sender = getattr(signal, "pidfd_send_signal", None)
    if not callable(opener) or not callable(sender):
        _fail("recovery_pidfd_api_unavailable", phase="recovery_process")
    try:
        pidfd = opener(target["target_pid"], 0)
    except ProcessLookupError:
        return None
    except OSError as exc:
        raise CoordinatorError(
            "recovery_pidfd_open_failed",
            phase="recovery_process",
        ) from exc
    try:
        if not _recovery_target_is_exactly_alive(
            target,
            coordinator_input=coordinator_input,
        ):
            os.close(pidfd)
            return None
        return pidfd
    except BaseException:
        os.close(pidfd)
        raise


def _terminate_exact_recovery_target(
    *,
    coordinator_input: CoordinatorInput,
    target: Mapping[str, Any],
) -> tuple[bool, bool]:
    if target["target_pid"] == os.getpid():
        _fail("recovery_process_identity_is_self", phase="recovery_process")
    if not _recovery_target_is_exactly_alive(
        target,
        coordinator_input=coordinator_input,
    ):
        return False, True
    pidfd = _open_exact_recovery_target_pidfd(
        coordinator_input=coordinator_input,
        target=target,
    )
    if pidfd is None:
        return False, True
    signal_attempted = False
    pidfd_exit_proven = False
    try:
        signal_attempted = _pidfd_signal(
            pidfd,
            signal.SIGTERM,
            code="recovery_process_sigterm_failed",
        )
        if _wait_for_pidfd_exit(
            pidfd,
            timeout_seconds=RECOVERY_PROCESS_TERM_SECONDS,
        ):
            pidfd_exit_proven = True
        else:
            if _recovery_target_is_exactly_alive(
                target,
                coordinator_input=coordinator_input,
            ):
                signal_attempted = (
                    _pidfd_signal(
                        pidfd,
                        signal.SIGKILL,  # windows-footgun: ok
                        code="recovery_process_sigkill_failed",
                    )
                    or signal_attempted
                )
                if not _wait_for_pidfd_exit(
                    pidfd,
                    timeout_seconds=RECOVERY_PROCESS_KILL_SECONDS,
                ):
                    _fail(
                        "recovery_process_termination_unconfirmed",
                        phase="recovery_process",
                    )
                pidfd_exit_proven = True
    finally:
        os.close(pidfd)
    if not pidfd_exit_proven and _recovery_target_is_exactly_alive(
        target,
        coordinator_input=coordinator_input,
    ):
        _fail(
            "recovery_process_termination_unconfirmed",
            phase="recovery_process",
        )
    return signal_attempted, True


def _open_recovery_process_lock(*, nonblocking: bool) -> int | None:
    descriptor = os.open(
        COORDINATOR_PROCESS_LOCK_PATH,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        item = os.fstat(descriptor)
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_uid != 0
            or item.st_gid != 0
            or stat.S_IMODE(item.st_mode) != 0o600
        ):
            _fail("coordinator_process_lock_invalid", phase="recovery_process")
        operation = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError:
            os.close(descriptor)
            return None
        return descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _close_recovery_process_lock(descriptor: int) -> None:
    if descriptor < 0:
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _snapshot_is_exact(
    expected: _RootFileSnapshot,
    observed: _RootFileSnapshot | None,
) -> bool:
    return bool(
        observed is not None
        and observed.raw == expected.raw
        and _same_file_identity(observed.item, expected.item)
    )


def _cas_recovery_journal(
    *,
    expected: _RootFileSnapshot,
    value: Mapping[str, Any],
) -> _RootPublication:
    current = _capture_root_snapshot(
        COORDINATOR_PROCESS_LEASE_PATH,
        maximum=MAX_COORDINATOR_INPUT_BYTES,
    )
    if not _snapshot_is_exact(expected, current):
        _fail("recovery_journal_cas_lost", phase="recovery_process")
    return _publish_root_payload(
        COORDINATOR_PROCESS_LEASE_PATH,
        _canonical_bytes(value),
        expected_previous_sha256=expected.sha256,
    )


@dataclass(frozen=True)
class _RecoveryPredecessor:
    kind: str
    schema: str
    generation: int
    value: Mapping[str, Any]
    snapshot: _RootFileSnapshot
    original_run_lease: Mapping[str, Any]
    causal_state: Mapping[str, Any]
    target: Mapping[str, Any]


def _load_recovery_predecessor(
    coordinator_input: CoordinatorInput,
) -> _RecoveryPredecessor:
    snapshot = _capture_root_snapshot(
        COORDINATOR_PROCESS_LEASE_PATH,
        maximum=MAX_COORDINATOR_INPUT_BYTES,
    )
    if snapshot is None:
        _fail(
            "coordinator_process_recovery_not_required",
            phase="recovery_preflight",
        )
    value = _decode_mapping(
        snapshot.raw,
        code="coordinator_process_journal_invalid",
    )
    schema = value.get("schema")
    if schema == "muncho-full-canary-coordinator-process-lease.v1":
        lease = _parse_process_lease(
            value,
            coordinator_input=coordinator_input,
        )
        return _RecoveryPredecessor(
            kind="run_process_lease",
            schema=str(schema),
            generation=0,
            value=lease,
            snapshot=snapshot,
            original_run_lease=lease,
            causal_state=_observe_recovery_causal_state(coordinator_input),
            target=_run_lease_recovery_target(lease),
        )
    if schema == RECOVERY_WORKER_LEASE_SCHEMA:
        worker = _parse_recovery_worker_lease(
            value,
            coordinator_input=coordinator_input,
        )
        return _RecoveryPredecessor(
            kind="recovery_worker_lease",
            schema=str(schema),
            generation=worker["recovery_generation"],
            value=worker,
            snapshot=snapshot,
            original_run_lease=worker["original_run_process_lease"],
            causal_state=worker["causal_recovery_state"],
            target=_worker_recovery_target(worker),
        )
    if schema == RECOVERY_WORKER_COMPLETION_SCHEMA:
        _parse_recovery_worker_completion(
            value,
            coordinator_input=coordinator_input,
        )
        _fail(
            "recovery_completion_requires_finalization",
            phase="recovery_preflight",
        )
    if schema == LEGACY_RECOVERY_RECEIPT_SCHEMA:
        _parse_legacy_recovery_receipt(
            value,
            coordinator_input=coordinator_input,
        )
        _fail(
            "legacy_recovery_receipt_reconciliation_required",
            phase="recovery_preflight",
        )
    if schema == RECOVERY_RECEIPT_SCHEMA:
        _parse_recovery_receipt_v2(value, coordinator_input=coordinator_input)
        _fail(
            "coordinator_process_recovery_not_required",
            phase="recovery_preflight",
        )
    _fail("coordinator_process_journal_invalid", phase="recovery_preflight")


def _parse_recovery_takeover_gate(
    value: Any,
    *,
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECOVERY_TAKEOVER_GATE_FIELDS:
        _fail("recovery_takeover_gate_invalid", phase="recovery_preflight")
    raw = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in raw.items() if key != "gate_sha256"}
    positive_pair = (
        type(raw["token_device"]) is int
        and raw["token_device"] > 0
        and type(raw["token_inode"]) is int
        and raw["token_inode"] > 0
    )
    null_pair = raw["token_device"] is None and raw["token_inode"] is None
    if (
        raw["schema"] != RECOVERY_TAKEOVER_GATE_SCHEMA
        or raw["ok"] is not True
        or raw["state"] != "awaiting_owner_recovery_takeover_ack"
        or raw["release_sha"] != coordinator_input.revision
        or raw["coordinator_input_sha256"] != coordinator_input.sha256
        or not _recovery_predecessor_kind_schema_generation_valid(raw)
        or type(raw["predecessor_generation"]) is not int
        or raw["predecessor_generation"] < 0
        or raw["target_process_identity_state"] not in {"exact_alive", "not_alive"}
        or raw["discord_token_state"]
        not in {"installed", "retirement_prepared", "retired"}
        or (raw["discord_token_state"] == "installed" and not positive_pair)
        or (
            raw["discord_token_state"] != "installed"
            and not (positive_pair or null_pair)
        )
        or raw["db_secret_accepted"] is not False
        or raw["frame_schema"] != RECOVERY_ACK_FRAME_SCHEMA
        or type(raw["observed_at_unix"]) is not int
        or type(raw["expires_at_unix"]) is not int
        or not 1
        <= raw["expires_at_unix"] - raw["observed_at_unix"]
        <= RECOVERY_GATE_MAX_SECONDS
        or raw["gate_sha256"] != _sha256_json(unsigned)
    ):
        _fail("recovery_takeover_gate_invalid", phase="recovery_preflight")
    for name in (
        "credential_prepare_approval_sha256",
        "owner_subject_sha256",
        "predecessor_journal_sha256",
        "original_run_process_lease_sha256",
        "causal_recovery_state_sha256",
        "target_boot_id_sha256",
        "target_module_sha256",
        "target_process_exe_sha256",
        "target_process_cmdline_sha256",
        "discord_token_install_receipt_sha256",
    ):
        _digest(raw[name], "recovery_takeover_gate_digest_invalid")
    if raw["discord_retirement_receipt_sha256"] is not None:
        _digest(
            raw["discord_retirement_receipt_sha256"],
            "recovery_takeover_gate_digest_invalid",
        )
    _validate_recovery_target_public_fields(raw)
    return raw


def recovery_gate(*, now_unix: int | None = None) -> Mapping[str, Any]:
    """Return one no-secret owner gate for the exact current predecessor."""

    _require_root_linux()
    coordinator_input = load_coordinator_input()
    predecessor = _load_recovery_predecessor(coordinator_input)
    observed_at = int(time.time()) if now_unix is None else now_unix
    if type(observed_at) is not int or observed_at < 0:
        _fail("recovery_clock_invalid", phase="recovery_preflight")
    target_alive = _recovery_target_is_exactly_alive(
        predecessor.target,
        coordinator_input=coordinator_input,
    )
    causal = predecessor.causal_state
    unsigned = {
        "schema": RECOVERY_TAKEOVER_GATE_SCHEMA,
        "ok": True,
        "state": "awaiting_owner_recovery_takeover_ack",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "credential_prepare_approval_sha256": predecessor.original_run_lease[
            "credential_prepare_approval_sha256"
        ],
        "owner_subject_sha256": predecessor.original_run_lease["owner_subject_sha256"],
        "ephemeral_admin_username": predecessor.original_run_lease[
            "ephemeral_admin_username"
        ],
        "predecessor_kind": predecessor.kind,
        "predecessor_schema": predecessor.schema,
        "predecessor_journal_sha256": predecessor.snapshot.sha256,
        "predecessor_generation": predecessor.generation,
        "original_run_process_lease_sha256": predecessor.original_run_lease[
            "lease_sha256"
        ],
        "causal_recovery_state_sha256": causal["causal_state_sha256"],
        **_target_public_identity(predecessor.target),
        "target_process_identity_state": (
            "exact_alive" if target_alive else "not_alive"
        ),
        "discord_token_state": causal["discord_token_state"],
        "discord_token_install_receipt_sha256": causal[
            "discord_token_install_receipt_sha256"
        ],
        "token_device": causal["token_device"],
        "token_inode": causal["token_inode"],
        "discord_retirement_receipt_sha256": causal[
            "discord_retirement_receipt_sha256"
        ],
        "db_secret_accepted": False,
        "frame_schema": RECOVERY_ACK_FRAME_SCHEMA,
        "observed_at_unix": observed_at,
        "expires_at_unix": observed_at + RECOVERY_GATE_MAX_SECONDS,
    }
    return _parse_recovery_takeover_gate(
        {**unsigned, "gate_sha256": _sha256_json(unsigned)},
        coordinator_input=coordinator_input,
    )


def _read_recovery_ack(
    *,
    gate: Mapping[str, Any],
    fd: int = ADMIN_FRAME_FD,
) -> Mapping[str, Any]:
    """Read only MRA1; the SQL credential is a later gated frame."""

    if fd != ADMIN_FRAME_FD:
        _fail("recovery_ack_fd_not_fixed", phase="recovery_ack")
    if os.isatty(fd):
        _fail("recovery_ack_tty_forbidden", phase="recovery_ack")
    header = _read_exact(fd, _ACK_FRAME_HEADER.size)
    raw = bytearray()
    try:
        magic, size = _ACK_FRAME_HEADER.unpack(header)
        if magic != RECOVERY_ACK_FRAME_MAGIC:
            _fail("recovery_ack_magic_invalid", phase="recovery_ack")
        if not 1 <= size <= MAX_OWNER_APPROVAL_BYTES:
            _fail("recovery_ack_bound_invalid", phase="recovery_ack")
        raw = _read_exact(fd, size)
        value = _decode_mapping(bytes(raw), code="recovery_ack_json_invalid")
    finally:
        _zeroize(header)
        _zeroize(raw)
    if set(value) != _RECOVERY_TAKEOVER_ACK_FIELDS:
        _fail("recovery_ack_fields_invalid", phase="recovery_ack")
    unsigned = {key: item for key, item in value.items() if key != "ack_sha256"}
    now_unix = int(time.time())
    mirrored = {
        "release_sha",
        "coordinator_input_sha256",
        "credential_prepare_approval_sha256",
        "owner_subject_sha256",
        "ephemeral_admin_username",
        "predecessor_kind",
        "predecessor_schema",
        "predecessor_journal_sha256",
        "predecessor_generation",
        "original_run_process_lease_sha256",
        "causal_recovery_state_sha256",
        "target_pid",
        "target_process_start_time_ticks",
        "target_boot_id_sha256",
        "target_boot_time_ns",
        "target_uid",
        "target_gid",
        "target_module_origin",
        "target_module_sha256",
        "target_process_exe_sha256",
        "target_process_cmdline_sha256",
        "target_process_identity_state",
        "discord_token_state",
        "discord_token_install_receipt_sha256",
        "token_device",
        "token_inode",
        "discord_retirement_receipt_sha256",
    }
    if (
        value["schema"] != RECOVERY_TAKEOVER_ACK_SCHEMA
        or value["scope"] != "terminate_exact_recovery_predecessor_and_claim_worker"
        or value["recovery_takeover_gate_sha256"] != gate["gate_sha256"]
        or any(value[name] != gate[name] for name in mirrored)
        or type(value["approved_at_unix"]) is not int
        or type(value["expires_at_unix"]) is not int
        or not gate["observed_at_unix"]
        <= value["approved_at_unix"]
        <= now_unix
        <= value["expires_at_unix"]
        or not 1
        <= value["expires_at_unix"] - value["approved_at_unix"]
        <= RECOVERY_GATE_MAX_SECONDS
        or value["expires_at_unix"] > gate["expires_at_unix"]
        or value["ack_sha256"] != _sha256_json(unsigned)
    ):
        _fail("recovery_ack_not_bound", phase="recovery_ack")
    _digest(value["nonce_sha256"], "recovery_ack_nonce_invalid")
    return copy.deepcopy(dict(value))


def _successor_worker_identity(
    value: Mapping[str, Any],
    *,
    coordinator_input: CoordinatorInput,
) -> tuple[Mapping[str, Any], int]:
    schema = value.get("schema")
    if schema == RECOVERY_WORKER_LEASE_SCHEMA:
        parsed = _parse_recovery_worker_lease(
            value,
            coordinator_input=coordinator_input,
        )
    elif schema == RECOVERY_WORKER_COMPLETION_SCHEMA:
        parsed = _parse_recovery_worker_completion(
            value,
            coordinator_input=coordinator_input,
        )
    elif schema == RECOVERY_RECEIPT_SCHEMA:
        parsed = _parse_recovery_receipt_v2(
            value,
            coordinator_input=coordinator_input,
        )
    else:
        _fail("recovery_worker_successor_invalid", phase="recovery_process")
    return parsed, int(parsed["recovery_generation"])


def _observe_concurrent_recovery_successor(
    *,
    coordinator_input: CoordinatorInput,
    predecessor: _RecoveryPredecessor,
) -> tuple[Mapping[str, Any], _RootFileSnapshot]:
    deadline = time.monotonic() + RECOVERY_CONCURRENT_OBSERVE_SECONDS
    while True:
        snapshot = _capture_root_snapshot(
            COORDINATOR_PROCESS_LEASE_PATH,
            maximum=MAX_COORDINATOR_INPUT_BYTES,
        )
        if snapshot is not None and snapshot.sha256 != predecessor.snapshot.sha256:
            value = _decode_mapping(
                snapshot.raw,
                code="recovery_worker_successor_invalid",
            )
            _successor_worker_identity(
                value,
                coordinator_input=coordinator_input,
            )
            return value, snapshot
        if time.monotonic() >= deadline:
            _fail(
                "recovery_worker_claim_contended_unresolved",
                phase="recovery_process",
            )
        time.sleep(0.01)


def _build_recovery_concurrent_loser_receipt(
    *,
    coordinator_input: CoordinatorInput,
    predecessor: _RecoveryPredecessor,
    gate: Mapping[str, Any],
    ack: Mapping[str, Any],
    signal_attempted: bool,
    target_termination_proven: bool,
) -> Mapping[str, Any]:
    successor, successor_snapshot = _observe_concurrent_recovery_successor(
        coordinator_input=coordinator_input,
        predecessor=predecessor,
    )
    parsed, generation = _successor_worker_identity(
        successor,
        coordinator_input=coordinator_input,
    )
    if (
        parsed["release_sha"] != coordinator_input.revision
        or parsed["coordinator_input_sha256"] != coordinator_input.sha256
        or parsed["credential_prepare_approval_sha256"]
        != gate["credential_prepare_approval_sha256"]
        or parsed["owner_subject_sha256"] != gate["owner_subject_sha256"]
        or parsed["original_run_process_lease_sha256"]
        != gate["original_run_process_lease_sha256"]
        or parsed["causal_recovery_state_sha256"]
        != gate["causal_recovery_state_sha256"]
        or generation <= predecessor.generation
        or (
            generation == predecessor.generation + 1
            and parsed["predecessor_journal_sha256"] != predecessor.snapshot.sha256
        )
    ):
        _fail("recovery_worker_successor_state_conflict", phase="recovery_process")
    unsigned = {
        "schema": RECOVERY_CONCURRENT_LOSER_RECEIPT_SCHEMA,
        "ok": False,
        "state": "recovery_worker_claim_lost_no_secret",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "credential_prepare_approval_sha256": gate[
            "credential_prepare_approval_sha256"
        ],
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "ephemeral_admin_username": gate["ephemeral_admin_username"],
        "recovery_takeover_gate_sha256": gate["gate_sha256"],
        "owner_recovery_takeover_ack_sha256": ack["ack_sha256"],
        "predecessor_kind": predecessor.kind,
        "predecessor_schema": predecessor.schema,
        "predecessor_journal_sha256": predecessor.snapshot.sha256,
        "predecessor_generation": predecessor.generation,
        "original_run_process_lease_sha256": gate["original_run_process_lease_sha256"],
        **_target_public_identity(predecessor.target),
        "target_signal_attempted_by_loser": signal_attempted,
        "target_termination_proven_by_loser": target_termination_proven,
        "process_lock_acquired_by_loser": False,
        "journal_cas_attempted_by_loser": False,
        "journal_cas_succeeded_by_loser": False,
        "observed_successor_schema": parsed["schema"],
        "observed_successor_state": parsed["state"],
        "observed_successor_journal_sha256": successor_snapshot.sha256,
        "observed_successor_generation": generation,
        "observed_successor_worker_pid": parsed["recovery_worker_pid"],
        "observed_successor_worker_process_start_time_ticks": parsed[
            "recovery_worker_process_start_time_ticks"
        ],
        "observed_successor_worker_boot_id_sha256": parsed[
            "recovery_worker_boot_id_sha256"
        ],
        "observed_successor_worker_module_sha256": parsed[
            "recovery_worker_module_sha256"
        ],
        "observed_successor_worker_process_exe_sha256": parsed[
            "recovery_worker_process_exe_sha256"
        ],
        "observed_successor_worker_process_cmdline_sha256": parsed[
            "recovery_worker_process_cmdline_sha256"
        ],
        "secret_gate_emitted_by_loser": False,
        "admin_frame_bytes_received_by_loser": 0,
        "admin_session_opened_by_loser": False,
        "admin_credential_mutation_performed_by_loser": False,
        "worker_lease_published_by_loser": False,
        "retryable": True,
        "completed_at_unix": int(time.time()),
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


@dataclass(frozen=True)
class _RecoveryWorkerClaim:
    lock_fd: int
    predecessor: _RecoveryPredecessor
    gate: Mapping[str, Any]
    ack: Mapping[str, Any]
    lease: Mapping[str, Any]
    snapshot: _RootFileSnapshot


def _claim_recovery_worker(
    *,
    coordinator_input: CoordinatorInput,
    predecessor: _RecoveryPredecessor,
    gate: Mapping[str, Any],
    ack: Mapping[str, Any],
) -> _RecoveryWorkerClaim | Mapping[str, Any]:
    now_unix = int(time.time())
    current = _capture_root_snapshot(
        COORDINATOR_PROCESS_LEASE_PATH,
        maximum=MAX_COORDINATOR_INPUT_BYTES,
    )
    if (
        not _snapshot_is_exact(predecessor.snapshot, current)
        or now_unix > gate["expires_at_unix"]
        or now_unix > ack["expires_at_unix"]
    ):
        _fail("recovery_predecessor_drifted_before_signal", phase="recovery_process")
    signal_attempted, termination_proven = _terminate_exact_recovery_target(
        coordinator_input=coordinator_input,
        target=predecessor.target,
    )
    lock_fd = _open_recovery_process_lock(nonblocking=True)
    if lock_fd is None:
        return _build_recovery_concurrent_loser_receipt(
            coordinator_input=coordinator_input,
            predecessor=predecessor,
            gate=gate,
            ack=ack,
            signal_attempted=signal_attempted,
            target_termination_proven=termination_proven,
        )
    try:
        now_unix = int(time.time())
        current = _capture_root_snapshot(
            COORDINATOR_PROCESS_LEASE_PATH,
            maximum=MAX_COORDINATOR_INPUT_BYTES,
        )
        if (
            not _snapshot_is_exact(predecessor.snapshot, current)
            or _recovery_target_is_exactly_alive(
                predecessor.target,
                coordinator_input=coordinator_input,
            )
            or now_unix > gate["expires_at_unix"]
            or now_unix > ack["expires_at_unix"]
        ):
            _fail("recovery_predecessor_drifted_before_cas", phase="recovery_process")
        identity = _current_recovery_worker_identity(coordinator_input)
        unsigned = {
            "schema": RECOVERY_WORKER_LEASE_SCHEMA,
            "state": "claimed_awaiting_admin",
            "release_sha": coordinator_input.revision,
            "coordinator_input_sha256": coordinator_input.sha256,
            "credential_prepare_approval_sha256": gate[
                "credential_prepare_approval_sha256"
            ],
            "owner_subject_sha256": gate["owner_subject_sha256"],
            "ephemeral_admin_username": gate["ephemeral_admin_username"],
            "original_run_process_lease": copy.deepcopy(
                dict(predecessor.original_run_lease)
            ),
            "original_run_process_lease_sha256": gate[
                "original_run_process_lease_sha256"
            ],
            "causal_recovery_state": copy.deepcopy(dict(predecessor.causal_state)),
            "causal_recovery_state_sha256": gate["causal_recovery_state_sha256"],
            "predecessor_kind": predecessor.kind,
            "predecessor_schema": predecessor.schema,
            "predecessor_journal_sha256": predecessor.snapshot.sha256,
            "predecessor_generation": predecessor.generation,
            "recovery_generation": predecessor.generation + 1,
            "transition_seq": 1,
            "previous_transition_sha256": predecessor.snapshot.sha256,
            "recovery_takeover_gate_sha256": gate["gate_sha256"],
            "owner_recovery_takeover_ack_sha256": ack["ack_sha256"],
            **identity,
            "predecessor_termination_proven": termination_proven,
            "predecessor_process_lock_acquired": True,
            "predecessor_journal_replaced": True,
            "claimed_at_unix": now_unix,
            "updated_at_unix": now_unix,
            "owner_authority_expires_at_unix": min(
                gate["expires_at_unix"],
                ack["expires_at_unix"],
                now_unix + RECOVERY_WORKER_LEASE_MAX_SECONDS,
            ),
        }
        lease = {
            **unsigned,
            "worker_lease_sha256": _sha256_json(unsigned),
        }
        publication = _cas_recovery_journal(
            expected=predecessor.snapshot,
            value=lease,
        )
        parsed = _parse_recovery_worker_lease(
            lease,
            coordinator_input=coordinator_input,
        )
        return _RecoveryWorkerClaim(
            lock_fd=lock_fd,
            predecessor=predecessor,
            gate=gate,
            ack=ack,
            lease=parsed,
            snapshot=publication.after,
        )
    except BaseException:
        _close_recovery_process_lock(lock_fd)
        raise


def _transition_recovery_worker_to_admin_authority(
    *,
    coordinator_input: CoordinatorInput,
    claim: _RecoveryWorkerClaim,
) -> _RecoveryWorkerClaim:
    current_identity = _current_recovery_worker_identity(coordinator_input)
    if any(
        current_identity[name] != claim.lease[name]
        for name in _RECOVERY_WORKER_IDENTITY_FIELDS
    ):
        _fail("recovery_worker_identity_drifted", phase="recovery_process")
    now_unix = int(time.time())
    if now_unix > claim.lease["owner_authority_expires_at_unix"]:
        _fail("recovery_owner_authority_expired", phase="recovery_secret_gate")
    unsigned = {
        **{
            key: copy.deepcopy(item)
            for key, item in claim.lease.items()
            if key
            not in {
                "state",
                "transition_seq",
                "previous_transition_sha256",
                "updated_at_unix",
                "worker_lease_sha256",
            }
        },
        "state": "admin_authority_may_be_in_use",
        "transition_seq": 2,
        "previous_transition_sha256": claim.snapshot.sha256,
        "updated_at_unix": now_unix,
    }
    lease = {**unsigned, "worker_lease_sha256": _sha256_json(unsigned)}
    publication = _cas_recovery_journal(
        expected=claim.snapshot,
        value=lease,
    )
    return _RecoveryWorkerClaim(
        lock_fd=claim.lock_fd,
        predecessor=claim.predecessor,
        gate=claim.gate,
        ack=claim.ack,
        lease=_parse_recovery_worker_lease(
            lease,
            coordinator_input=coordinator_input,
        ),
        snapshot=publication.after,
    )


def _build_recovery_secret_gate(
    *,
    coordinator_input: CoordinatorInput,
    claim: _RecoveryWorkerClaim,
) -> Mapping[str, Any]:
    if (
        claim.lease["state"] != "admin_authority_may_be_in_use"
        or claim.lease["transition_seq"] != 2
    ):
        _fail("recovery_secret_gate_worker_phase_invalid", phase="recovery_secret_gate")
    now_unix = int(time.time())
    expires_at = min(
        claim.gate["expires_at_unix"],
        claim.ack["expires_at_unix"],
        claim.lease["owner_authority_expires_at_unix"],
    )
    if now_unix > expires_at:
        _fail("recovery_secret_gate_expired", phase="recovery_secret_gate")
    unsigned = {
        "schema": RECOVERY_SECRET_GATE_SCHEMA,
        "ok": True,
        "state": "awaiting_recovery_admin_credential",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "credential_prepare_approval_sha256": claim.lease[
            "credential_prepare_approval_sha256"
        ],
        "owner_subject_sha256": claim.lease["owner_subject_sha256"],
        "ephemeral_admin_username": claim.lease["ephemeral_admin_username"],
        "recovery_takeover_gate_sha256": claim.gate["gate_sha256"],
        "owner_recovery_takeover_ack_sha256": claim.ack["ack_sha256"],
        "predecessor_kind": claim.lease["predecessor_kind"],
        "predecessor_journal_sha256": claim.lease["predecessor_journal_sha256"],
        "predecessor_generation": claim.lease["predecessor_generation"],
        "original_run_process_lease_sha256": claim.lease[
            "original_run_process_lease_sha256"
        ],
        "causal_recovery_state_sha256": claim.lease["causal_recovery_state_sha256"],
        "recovery_worker_lease_sha256": claim.lease["worker_lease_sha256"],
        "recovery_worker_state": claim.lease["state"],
        "recovery_worker_transition_seq": claim.lease["transition_seq"],
        **{name: claim.lease[name] for name in _RECOVERY_WORKER_IDENTITY_FIELDS},
        "database_host": CANARY_DATABASE_HOST,
        "database_port": CANARY_DATABASE_PORT,
        "database_name": CANARY_DATABASE_NAME,
        "tls_server_name": coordinator_input.tls_server_name,
        "tls_peer_certificate_sha256": (coordinator_input.tls_peer_certificate_sha256),
        "admin_frame_schema": RECOVERY_ADMIN_FRAME_SCHEMA,
        "gate_nonce_sha256": _sha256_bytes(secrets.token_bytes(32)),
        "expires_at_unix": expires_at,
    }
    gate = {**unsigned, "gate_sha256": _sha256_json(unsigned)}
    if set(gate) != _RECOVERY_SECRET_GATE_FIELDS:
        _fail("recovery_secret_gate_fields_invalid", phase="recovery_secret_gate")
    return gate


def _revalidate_recovery_worker_snapshot(
    *,
    coordinator_input: CoordinatorInput,
    claim: _RecoveryWorkerClaim,
) -> None:
    current = _capture_root_snapshot(
        COORDINATOR_PROCESS_LEASE_PATH,
        maximum=MAX_COORDINATOR_INPUT_BYTES,
    )
    if not _snapshot_is_exact(claim.snapshot, current):
        _fail("recovery_worker_lease_drifted", phase="recovery_process")
    parsed = _parse_recovery_worker_lease(
        _decode_mapping(current.raw, code="recovery_worker_lease_invalid"),
        coordinator_input=coordinator_input,
    )
    if parsed != claim.lease:
        _fail("recovery_worker_lease_drifted", phase="recovery_process")
    identity = _current_recovery_worker_identity(coordinator_input)
    if any(
        identity[name] != claim.lease[name] for name in _RECOVERY_WORKER_IDENTITY_FIELDS
    ):
        _fail("recovery_worker_identity_drifted", phase="recovery_process")


def _perform_recovery_cleanup(
    *,
    coordinator_input: CoordinatorInput,
    original_run_lease: Mapping[str, Any],
    causal_state: Mapping[str, Any],
    admin_session: VerifiedTLSBootstrapAdminSession,
) -> Mapping[str, Any]:
    database_cleanup = admin_session.query(
        _BOOTSTRAP_ROLE_DISABLE_SQL,
        maximum_rows=0,
    )
    _require_recovery_do_result(
        database_cleanup,
        code="recovery_database_cleanup_unconfirmed",
    )
    if os.path.lexists(CANARY_BOOTSTRAP_CREDENTIAL_PATH):
        bootstrap_item = _validate_secret_metadata(
            CANARY_BOOTSTRAP_CREDENTIAL_PATH,
            uid=coordinator_input.identities.writer_uid,
            gid=coordinator_input.identities.writer_gid,
            maximum=MAX_ADMIN_PASSWORD_BYTES,
        )
        _remove_exact_secret(
            CANARY_BOOTSTRAP_CREDENTIAL_PATH,
            bootstrap_item,
            state=_SecretRemovalState(),
        )
    admin_session.close()

    runtime_snapshot = _capture_root_snapshot(DEFAULT_PLAN_PATH)
    staged_snapshot = _capture_root_snapshot(DEFAULT_STAGED_PLAN_PATH)
    canonical_stop_sha256: str | None = None
    preplan_report_sha256: str | None = None
    preclaim_sha256: str | None = None
    preclaim_state: str | None = None
    if runtime_snapshot is not None:
        plan = load_full_canary_plan()
        if plan.revision != coordinator_input.revision:
            _fail("recovery_runtime_plan_drifted", phase="recovery_lifecycle")
        if staged_snapshot is not None:
            staged_plan = _load_staged_plan_for_recovery(
                staged_snapshot,
                coordinator_input=coordinator_input,
            )
            if staged_plan.to_mapping() != plan.to_mapping():
                _fail("recovery_plan_pair_drifted", phase="recovery_lifecycle")
        stop_receipt = FullCanaryLifecycle(plan).stop(reason="operator_requested")
        (
            canonical_stop_sha256,
            preclaim_sha256,
            preclaim_state,
        ) = _validate_recovery_stop_receipt(stop_receipt, plan=plan)
    elif staged_snapshot is not None:
        plan = _load_staged_plan_for_recovery(
            staged_snapshot,
            coordinator_input=coordinator_input,
        )
        stop_receipt = FullCanaryLifecycle(plan).stop(reason="operator_requested")
        (
            canonical_stop_sha256,
            preclaim_sha256,
            preclaim_state,
        ) = _validate_recovery_stop_receipt(stop_receipt, plan=plan)
        _remove_recovery_root_artifact(DEFAULT_STAGED_PLAN_PATH)
        _remove_recovery_writer_artifact(coordinator_input.identities.writer_gid)
    else:
        if not _services_are_exactly_stopped_and_disabled():
            _fail(
                "recovery_preplan_services_not_stopped",
                phase="recovery_lifecycle",
            )
        _remove_recovery_writer_artifact(coordinator_input.identities.writer_gid)
        _remove_recovery_root_artifact(OWNER_APPROVAL_REQUEST_PATH)
        report = _publish_preplan_stopped_report(
            coordinator_input=coordinator_input,
            lease=original_run_lease,
        )
        preplan_report_sha256 = str(report["report_sha256"])
    _remove_recovery_root_artifact(OWNER_APPROVAL_REQUEST_PATH)

    install_receipt, installed, install_snapshot = _recovery_discord_retirement_inputs(
        coordinator_input=coordinator_input,
        gate=causal_state,
    )
    retirement = _retire_discord_token_lease(
        coordinator_input=coordinator_input,
        install_receipt=install_receipt,
        installed=installed,
        install_snapshot=install_snapshot,
    )
    if (
        retirement["state"] != "retired"
        or retirement["discord_token_install_receipt_sha256"]
        != causal_state["discord_token_install_receipt_sha256"]
        or retirement["token_device"] != causal_state["token_device"]
        or retirement["token_inode"] != causal_state["token_inode"]
        or not _services_are_exactly_stopped_and_disabled()
        or any(
            os.path.lexists(path)
            for path in (
                DISCORD_TOKEN_PATH,
                DISCORD_TOKEN_STAGE_PATH,
                DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
                CANARY_BOOTSTRAP_CREDENTIAL_PATH,
            )
        )
        or not admin_session.closed
    ):
        _fail("recovery_terminal_truth_unconfirmed", phase="recovery_terminal")
    return {
        "canonical_stop_receipt_sha256": canonical_stop_sha256,
        "preplan_stopped_report_sha256": preplan_report_sha256,
        "preclaim_reconciliation_receipt_sha256": preclaim_sha256,
        "preclaim_reconciliation_state": preclaim_state,
        "admin_frame_zeroized": True,
        "admin_session_closed": True,
        "migration_owner_membership_removed": True,
        "bootstrap_login_password_disabled": True,
        "bootstrap_credential_removed": True,
        "discord_token_removed": True,
        "discord_install_receipt_removed": True,
        "discord_retirement_receipt_sha256": retirement["receipt_sha256"],
        "services_stopped_proven": True,
        "services_enabled": False,
    }


def _publish_recovery_worker_completion(
    *,
    coordinator_input: CoordinatorInput,
    claim: _RecoveryWorkerClaim,
    cleanup: Mapping[str, Any],
) -> Mapping[str, Any]:
    if set(cleanup) != _RECOVERY_CLEANUP_FIELDS:
        _fail("recovery_cleanup_fields_invalid", phase="recovery_terminal")
    _revalidate_recovery_worker_snapshot(
        coordinator_input=coordinator_input,
        claim=claim,
    )
    unsigned = {
        "schema": RECOVERY_WORKER_COMPLETION_SCHEMA,
        "ok": False,
        "state": "cleanup_complete_awaiting_worker_exit",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "credential_prepare_approval_sha256": claim.lease[
            "credential_prepare_approval_sha256"
        ],
        "owner_subject_sha256": claim.lease["owner_subject_sha256"],
        "ephemeral_admin_username": claim.lease["ephemeral_admin_username"],
        "original_run_process_lease": copy.deepcopy(
            dict(claim.lease["original_run_process_lease"])
        ),
        "original_run_process_lease_sha256": claim.lease[
            "original_run_process_lease_sha256"
        ],
        "causal_recovery_state_sha256": claim.lease["causal_recovery_state_sha256"],
        "predecessor_kind": claim.lease["predecessor_kind"],
        "predecessor_journal_sha256": claim.lease["predecessor_journal_sha256"],
        "predecessor_generation": claim.lease["predecessor_generation"],
        "recovery_generation": claim.lease["recovery_generation"],
        "recovery_takeover_gate_sha256": claim.lease["recovery_takeover_gate_sha256"],
        "owner_recovery_takeover_ack_sha256": claim.lease[
            "owner_recovery_takeover_ack_sha256"
        ],
        "recovery_worker_lease_sha256": claim.lease["worker_lease_sha256"],
        **{name: claim.lease[name] for name in _RECOVERY_WORKER_IDENTITY_FIELDS},
        "predecessor_termination_proven": claim.lease["predecessor_termination_proven"],
        "predecessor_process_lock_acquired": claim.lease[
            "predecessor_process_lock_acquired"
        ],
        "predecessor_journal_replaced": claim.lease["predecessor_journal_replaced"],
        **copy.deepcopy(dict(cleanup)),
        "recovery_worker_exit_proven": False,
        "safe_to_delete_temporary_admin": False,
        "cleanup_completed_at_unix": int(time.time()),
    }
    completion = {
        **unsigned,
        "completion_sha256": _sha256_json(unsigned),
    }
    parsed = _parse_recovery_worker_completion(
        completion,
        coordinator_input=coordinator_input,
    )
    _cas_recovery_journal(expected=claim.snapshot, value=parsed)
    return parsed


def _read_recovery_admin_frame(
    *,
    expected_username: str,
    expected_gate_sha256: str,
    expected_gate_nonce_sha256: str,
    fd: int = ADMIN_FRAME_FD,
) -> OpaqueStdinAdminFrame:
    """Read MRC2 bound to the just-emitted unpredictable stage-two gate."""

    if fd != ADMIN_FRAME_FD:
        _fail("recovery_admin_frame_fd_not_fixed", phase="credential_read")
    if os.isatty(fd):
        _fail("recovery_admin_frame_tty_forbidden", phase="credential_read")
    expected_gate = bytes.fromhex(
        _digest(expected_gate_sha256, "recovery_secret_gate_digest_invalid")
    )
    expected_nonce = bytes.fromhex(
        _digest(expected_gate_nonce_sha256, "recovery_secret_nonce_digest_invalid")
    )
    magic_raw = _read_exact(fd, 4)
    if bytes(magic_raw) != RECOVERY_ADMIN_FRAME_MAGIC:
        _zeroize(magic_raw)
        _fail("recovery_admin_frame_magic_invalid", phase="credential_read")
    header_tail = bytearray()
    username_raw: bytearray | None = None
    password_raw: bytearray | None = None
    try:
        (
            gate_raw,
            nonce_raw,
            username_size,
            password_size,
        ) = _RECOVERY_ADMIN_FRAME_TAIL.unpack(
            header_tail := _read_exact(fd, _RECOVERY_ADMIN_FRAME_TAIL.size)
        )
        if gate_raw != expected_gate or nonce_raw != expected_nonce:
            _fail("recovery_admin_frame_gate_not_bound", phase="credential_read")
        if not 1 <= username_size <= MAX_ADMIN_USERNAME_BYTES:
            _fail("admin_frame_username_bound_invalid", phase="credential_read")
        if not MIN_ADMIN_PASSWORD_BYTES <= password_size <= MAX_ADMIN_PASSWORD_BYTES:
            _fail("admin_frame_password_bound_invalid", phase="credential_read")
        username_raw = _read_exact(fd, username_size)
        password_raw = _read_exact(fd, password_size)
        try:
            username = username_raw.decode("utf-8", errors="strict")
            password_text = password_raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CoordinatorError(
                "admin_frame_utf8_invalid",
                phase="credential_read",
            ) from exc
        if (
            username != expected_username
            or _ADMIN_USERNAME_RE.fullmatch(username) is None
        ):
            _fail("admin_frame_username_mismatch", phase="credential_read")
        if (
            password_text != password_text.strip()
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in password_text
            )
            or "\x00" in password_text
        ):
            _fail("admin_frame_password_invalid", phase="credential_read")
        try:
            trailing = os.read(fd, 1)
        except OSError as exc:
            raise CoordinatorError(
                "admin_frame_eof_check_failed",
                phase="credential_read",
            ) from exc
        if trailing:
            _fail("admin_frame_trailing_data", phase="credential_read")
        password_text = ""
        result_password = password_raw
        password_raw = None
        return OpaqueStdinAdminFrame(
            username=username,
            password=result_password,
        )
    finally:
        _zeroize(magic_raw)
        _zeroize(header_tail)
        _zeroize(username_raw)
        _zeroize(password_raw)


def recover_full_canary(
    *,
    gate_emitter: Callable[[Mapping[str, Any]], None],
    ack_reader: Callable[..., Mapping[str, Any]] = _read_recovery_ack,
    admin_frame_reader: Callable[..., OpaqueStdinAdminFrame] = (
        _read_recovery_admin_frame
    ),
    admin_session_opener: Callable[..., VerifiedTLSBootstrapAdminSession] = (
        VerifiedTLSBootstrapAdminSession.open
    ),
) -> Mapping[str, Any]:
    """Run the two-stage no-secret claim then exact recovery worker cleanup."""

    _harden_secret_process()
    coordinator_input = load_coordinator_input()
    predecessor = _load_recovery_predecessor(coordinator_input)
    gate = recovery_gate()
    if (
        gate["predecessor_journal_sha256"] != predecessor.snapshot.sha256
        or gate["predecessor_kind"] != predecessor.kind
    ):
        _fail("recovery_gate_predecessor_drifted", phase="recovery_preflight")
    gate_emitter(gate)
    ack = ack_reader(gate=gate)
    claim_or_loser = _claim_recovery_worker(
        coordinator_input=coordinator_input,
        predecessor=predecessor,
        gate=gate,
        ack=ack,
    )
    if isinstance(claim_or_loser, Mapping):
        return claim_or_loser
    claim = claim_or_loser
    admin_frame: OpaqueStdinAdminFrame | None = None
    admin_session: VerifiedTLSBootstrapAdminSession | None = None
    try:
        claim = _transition_recovery_worker_to_admin_authority(
            coordinator_input=coordinator_input,
            claim=claim,
        )
        _revalidate_recovery_worker_snapshot(
            coordinator_input=coordinator_input,
            claim=claim,
        )
        secret_gate = _build_recovery_secret_gate(
            coordinator_input=coordinator_input,
            claim=claim,
        )
        gate_emitter(secret_gate)
        admin_frame = admin_frame_reader(
            expected_username=secret_gate["ephemeral_admin_username"],
            expected_gate_sha256=secret_gate["gate_sha256"],
            expected_gate_nonce_sha256=secret_gate["gate_nonce_sha256"],
        )
        _revalidate_recovery_worker_snapshot(
            coordinator_input=coordinator_input,
            claim=claim,
        )
        if int(time.time()) > secret_gate["expires_at_unix"]:
            _fail("recovery_secret_gate_expired", phase="recovery_secret_gate")
        admin_session = admin_session_opener(
            frame=admin_frame,
            tls_server_name=secret_gate["tls_server_name"],
            expected_tls_peer_certificate_sha256=secret_gate[
                "tls_peer_certificate_sha256"
            ],
        )
        # The frame is explicitly zeroized before any completion claim.  The
        # connection owns no reusable password after open succeeds.
        admin_frame.close()
        cleanup = _perform_recovery_cleanup(
            coordinator_input=coordinator_input,
            original_run_lease=claim.lease["original_run_process_lease"],
            causal_state=claim.lease["causal_recovery_state"],
            admin_session=admin_session,
        )
        if not admin_session.closed or not admin_frame.consumed:
            _fail("recovery_secret_cleanup_unconfirmed", phase="recovery_terminal")
        return _publish_recovery_worker_completion(
            coordinator_input=coordinator_input,
            claim=claim,
            cleanup=cleanup,
        )
    finally:
        _begin_active_signal_cleanup()
        if admin_frame is not None:
            admin_frame.close()
        if admin_session is not None and not admin_session.closed:
            try:
                admin_session.close()
            except BaseException:
                pass
        _close_recovery_process_lock(claim.lock_fd)


def _parse_recovery_finalize_pending_receipt(
    value: Any,
    *,
    coordinator_input: CoordinatorInput,
    completion: Mapping[str, Any],
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _RECOVERY_FINALIZE_PENDING_RECEIPT_FIELDS
    ):
        _fail("recovery_finalize_pending_receipt_invalid", phase="recovery_terminal")
    raw = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    if (
        raw["schema"] != RECOVERY_FINALIZE_PENDING_RECEIPT_SCHEMA
        or raw["ok"] is not False
        or raw["state"] != "recovery_finalization_pending_no_secret"
        or raw["release_sha"] != coordinator_input.revision
        or raw["coordinator_input_sha256"] != coordinator_input.sha256
        or raw["credential_prepare_approval_sha256"]
        != completion["credential_prepare_approval_sha256"]
        or raw["owner_subject_sha256"] != completion["owner_subject_sha256"]
        or raw["ephemeral_admin_username"] != completion["ephemeral_admin_username"]
        or raw["recovery_worker_completion_sha256"] != completion["completion_sha256"]
        or any(
            raw[name] != completion[name] for name in _RECOVERY_WORKER_IDENTITY_FIELDS
        )
        or raw["completion_admin_authority_may_have_been_used"] is not True
        or raw["completion_admin_frame_zeroized"] is not True
        or raw["completion_admin_session_closed"] is not True
        or raw["worker_identity_state"] not in {"exact_alive", "not_alive"}
        or type(raw["target_signal_attempted_by_finalizer"]) is not bool
        or type(raw["target_termination_proven_by_finalizer"]) is not bool
        or type(raw["process_lock_acquired_by_finalizer"]) is not bool
        or raw["completion_cas_attempted_by_finalizer"] is not False
        or raw["completion_cas_succeeded_by_finalizer"] is not False
        or (
            raw["worker_identity_state"] == "exact_alive"
            and raw["target_termination_proven_by_finalizer"] is True
        )
        or (
            raw["process_lock_acquired_by_finalizer"] is True
            and raw["worker_identity_state"] != "exact_alive"
        )
        or raw["observed_journal_sha256"] != _sha256_bytes(_canonical_bytes(completion))
        or raw["secret_gate_emitted_by_finalizer"] is not False
        or raw["admin_frame_bytes_received_by_finalizer"] != 0
        or raw["admin_session_opened_by_finalizer"] is not False
        or raw["admin_credential_mutation_performed_by_finalizer"] is not False
        or raw["retryable"] is not True
        or type(raw["completed_at_unix"]) is not int
        or raw["completed_at_unix"] < completion["cleanup_completed_at_unix"]
        or raw["receipt_sha256"] != _sha256_json(unsigned)
    ):
        _fail("recovery_finalize_pending_receipt_invalid", phase="recovery_terminal")
    _digest(raw["observed_journal_sha256"], "recovery_finalize_journal_digest_invalid")
    return raw


def _build_recovery_finalize_pending_receipt(
    *,
    coordinator_input: CoordinatorInput,
    completion: Mapping[str, Any],
    observed: _RootFileSnapshot,
    worker_identity_state: str,
    signal_attempted: bool,
    termination_proven: bool,
    lock_acquired: bool,
    cas_attempted: bool,
) -> Mapping[str, Any]:
    unsigned = {
        "schema": RECOVERY_FINALIZE_PENDING_RECEIPT_SCHEMA,
        "ok": False,
        "state": "recovery_finalization_pending_no_secret",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "credential_prepare_approval_sha256": completion[
            "credential_prepare_approval_sha256"
        ],
        "owner_subject_sha256": completion["owner_subject_sha256"],
        "ephemeral_admin_username": completion["ephemeral_admin_username"],
        "recovery_worker_completion_sha256": completion["completion_sha256"],
        **{name: completion[name] for name in _RECOVERY_WORKER_IDENTITY_FIELDS},
        "completion_admin_authority_may_have_been_used": True,
        "completion_admin_frame_zeroized": completion["admin_frame_zeroized"],
        "completion_admin_session_closed": completion["admin_session_closed"],
        "worker_identity_state": worker_identity_state,
        "target_signal_attempted_by_finalizer": signal_attempted,
        "target_termination_proven_by_finalizer": termination_proven,
        "process_lock_acquired_by_finalizer": lock_acquired,
        "completion_cas_attempted_by_finalizer": cas_attempted,
        "completion_cas_succeeded_by_finalizer": False,
        "observed_journal_sha256": observed.sha256,
        "secret_gate_emitted_by_finalizer": False,
        "admin_frame_bytes_received_by_finalizer": 0,
        "admin_session_opened_by_finalizer": False,
        "admin_credential_mutation_performed_by_finalizer": False,
        "retryable": True,
        "completed_at_unix": int(time.time()),
    }
    return _parse_recovery_finalize_pending_receipt(
        {**unsigned, "receipt_sha256": _sha256_json(unsigned)},
        coordinator_input=coordinator_input,
        completion=completion,
    )


def _final_recovery_receipt_from_completion(
    *,
    coordinator_input: CoordinatorInput,
    completion: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = {
        **{
            key: copy.deepcopy(item)
            for key, item in completion.items()
            if key
            not in {
                "schema",
                "ok",
                "state",
                "completion_sha256",
                "recovery_worker_exit_proven",
                "safe_to_delete_temporary_admin",
            }
        },
        "schema": RECOVERY_RECEIPT_SCHEMA,
        "ok": True,
        "state": "recovered",
        "recovery_worker_completion_sha256": completion["completion_sha256"],
        "recovery_worker_lock_acquired": True,
        "recovery_worker_exit_proven": True,
        "safe_to_delete_temporary_admin": True,
        "finalized_at_unix": int(time.time()),
    }
    return _parse_recovery_receipt_v2(
        {**unsigned, "receipt_sha256": _sha256_json(unsigned)},
        coordinator_input=coordinator_input,
    )


def _parse_concurrent_final_receipt_for_completion(
    value: Any,
    *,
    coordinator_input: CoordinatorInput,
    completion: Mapping[str, Any],
) -> Mapping[str, Any]:
    receipt = _parse_recovery_receipt_v2(
        value,
        coordinator_input=coordinator_input,
    )
    if receipt["recovery_worker_completion_sha256"] != completion["completion_sha256"]:
        _fail("recovery_final_successor_state_conflict", phase="recovery_terminal")
    return receipt


def finalize_recovery() -> Mapping[str, Any]:
    """No-secret exact worker-exit proof and completion-to-v2 CAS."""

    _require_root_linux()
    coordinator_input = load_coordinator_input()
    snapshot = _capture_root_snapshot(
        COORDINATOR_PROCESS_LEASE_PATH,
        maximum=MAX_COORDINATOR_INPUT_BYTES,
    )
    if snapshot is None:
        _fail(
            "coordinator_process_recovery_not_required",
            phase="recovery_preflight",
        )
    value = _decode_mapping(snapshot.raw, code="coordinator_process_journal_invalid")
    if value.get("schema") == LEGACY_RECOVERY_RECEIPT_SCHEMA:
        _parse_legacy_recovery_receipt(
            value,
            coordinator_input=coordinator_input,
        )
        _fail(
            "legacy_recovery_receipt_reconciliation_required",
            phase="recovery_preflight",
        )
    if value.get("schema") == RECOVERY_RECEIPT_SCHEMA:
        return _parse_recovery_receipt_v2(
            value,
            coordinator_input=coordinator_input,
        )
    completion = _parse_recovery_worker_completion(
        value,
        coordinator_input=coordinator_input,
    )
    target = _worker_recovery_target(completion)
    signal_attempted = False
    termination_proven = False
    try:
        signal_attempted, termination_proven = _terminate_exact_recovery_target(
            coordinator_input=coordinator_input,
            target=target,
        )
    except CoordinatorError as exc:
        if exc.code != "recovery_process_termination_unconfirmed":
            raise
        current = _capture_root_snapshot(
            COORDINATOR_PROCESS_LEASE_PATH,
            maximum=MAX_COORDINATOR_INPUT_BYTES,
        )
        if not _snapshot_is_exact(snapshot, current):
            _fail("recovery_completion_drifted", phase="recovery_terminal")
        return _build_recovery_finalize_pending_receipt(
            coordinator_input=coordinator_input,
            completion=completion,
            observed=current,
            worker_identity_state="exact_alive",
            signal_attempted=True,
            termination_proven=False,
            lock_acquired=False,
            cas_attempted=False,
        )
    lock_fd = _open_recovery_process_lock(nonblocking=True)
    if lock_fd is None:
        current = _capture_root_snapshot(
            COORDINATOR_PROCESS_LEASE_PATH,
            maximum=MAX_COORDINATOR_INPUT_BYTES,
        )
        if current is None:
            _fail("recovery_completion_lost", phase="recovery_terminal")
        current_value = _decode_mapping(
            current.raw,
            code="coordinator_process_journal_invalid",
        )
        if current_value.get("schema") == RECOVERY_RECEIPT_SCHEMA:
            return _parse_concurrent_final_receipt_for_completion(
                current_value,
                coordinator_input=coordinator_input,
                completion=completion,
            )
        if not _snapshot_is_exact(snapshot, current):
            _fail("recovery_completion_drifted", phase="recovery_terminal")
        return _build_recovery_finalize_pending_receipt(
            coordinator_input=coordinator_input,
            completion=completion,
            observed=current,
            worker_identity_state=(
                "exact_alive"
                if _recovery_target_is_exactly_alive(
                    target,
                    coordinator_input=coordinator_input,
                )
                else "not_alive"
            ),
            signal_attempted=signal_attempted,
            termination_proven=termination_proven,
            lock_acquired=False,
            cas_attempted=False,
        )
    try:
        current = _capture_root_snapshot(
            COORDINATOR_PROCESS_LEASE_PATH,
            maximum=MAX_COORDINATOR_INPUT_BYTES,
        )
        if current is not None:
            current_value = _decode_mapping(
                current.raw,
                code="coordinator_process_journal_invalid",
            )
            if current_value.get("schema") == RECOVERY_RECEIPT_SCHEMA:
                return _parse_concurrent_final_receipt_for_completion(
                    current_value,
                    coordinator_input=coordinator_input,
                    completion=completion,
                )
        if not _snapshot_is_exact(snapshot, current):
            _fail("recovery_completion_drifted", phase="recovery_terminal")
        if _recovery_target_is_exactly_alive(
            target,
            coordinator_input=coordinator_input,
        ):
            return _build_recovery_finalize_pending_receipt(
                coordinator_input=coordinator_input,
                completion=completion,
                observed=current,
                worker_identity_state="exact_alive",
                signal_attempted=signal_attempted,
                termination_proven=False,
                lock_acquired=True,
                cas_attempted=False,
            )
        receipt = _final_recovery_receipt_from_completion(
            coordinator_input=coordinator_input,
            completion=completion,
        )
        _cas_recovery_journal(expected=snapshot, value=receipt)
        return receipt
    finally:
        _close_recovery_process_lock(lock_fd)


def preflight_recovery() -> Mapping[str, Any]:
    """Read-only discovery of run/worker/completion/final journal truth."""

    _require_root_linux()
    coordinator_input = load_coordinator_input()
    snapshot = _capture_root_snapshot(
        COORDINATOR_PROCESS_LEASE_PATH,
        maximum=MAX_COORDINATOR_INPUT_BYTES,
    )
    if snapshot is None:
        _fail(
            "coordinator_process_recovery_not_required",
            phase="recovery_preflight",
        )
    value = _decode_mapping(snapshot.raw, code="coordinator_process_journal_invalid")
    schema = value.get("schema")
    if schema == LEGACY_RECOVERY_RECEIPT_SCHEMA:
        _parse_legacy_recovery_receipt(
            value,
            coordinator_input=coordinator_input,
        )
        _fail(
            "legacy_recovery_receipt_reconciliation_required",
            phase="recovery_preflight",
        )
    if schema == RECOVERY_RECEIPT_SCHEMA:
        return _parse_recovery_receipt_v2(
            value,
            coordinator_input=coordinator_input,
        )
    if schema == RECOVERY_WORKER_COMPLETION_SCHEMA:
        return _parse_recovery_worker_completion(
            value,
            coordinator_input=coordinator_input,
        )
    return recovery_gate()


_DISCORD_RETIREMENT_ACK_FIELDS = frozenset({
    "schema",
    "scope",
    "release_sha",
    "coordinator_input_sha256",
    "owner_subject_sha256",
    "retirement_gate_sha256",
    "discord_token_install_receipt_sha256",
    "token_device",
    "token_inode",
    "nonce_sha256",
    "approved_at_unix",
    "expires_at_unix",
    "ack_sha256",
})


def discord_retirement_gate(
    *,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    _require_root_linux()
    coordinator_input = load_coordinator_input()
    # A terminal recovery receipt is historical truth, not an active lease.
    # Validate and preserve it; only an active/invalid journal blocks DRA1.
    _load_terminal_recovery_journal(coordinator_input)
    install_state: _DiscordInstallState | None = None
    existing_retirement = _load_discord_retirement(coordinator_input)
    if (
        existing_retirement is not None
        and existing_retirement[0]["state"] == "retirement_prepared"
    ):
        retirement, _retirement_snapshot, install_state = (
            _load_prepared_discord_retirement_source(
                coordinator_input,
                require_terminal_install=False,
            )
        )
        if install_state is None:
            approval = _load_bound_discord_token_install_approval(coordinator_input)
            owner_subject_sha256 = approval.value["owner_subject_sha256"]
        else:
            owner_subject_sha256 = install_state.owner_subject_sha256
        install_sha256 = retirement["discord_token_install_receipt_sha256"]
        token_device = retirement["token_device"]
        token_inode = retirement["token_inode"]
    elif os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH):
        install_state = _load_discord_install_state(coordinator_input)
        owner_subject_sha256 = install_state.owner_subject_sha256
        install_sha256 = install_state.receipt_sha256
        token_device = install_state.device
        token_inode = install_state.inode
    else:
        _fail("discord_token_retirement_source_missing")
    if not _services_are_exactly_stopped_and_disabled():
        _fail(
            "discord_token_retirement_preflight_not_stopped",
            phase="discord_token_retirement_preflight",
        )
    current_time = int(time.time()) if now_unix is None else now_unix
    if type(current_time) is not int or current_time < 0:
        _fail("discord_token_retirement_clock_invalid")
    unsigned = {
        "schema": DISCORD_RETIREMENT_GATE_SCHEMA,
        "ok": True,
        "state": "awaiting_owner_discord_retirement_ack",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "owner_subject_sha256": owner_subject_sha256,
        "discord_token_install_receipt_sha256": install_sha256,
        "token_device": token_device,
        "token_inode": token_inode,
        # This proves active-lease absence.  A validated terminal recovery
        # receipt may still occupy the durable journal path.
        "process_lease_absent": True,
        "services_stopped_proven": True,
        "frame_schema": DISCORD_RETIREMENT_ACK_FRAME_SCHEMA,
        "expires_at_unix": current_time + RECOVERY_GATE_MAX_SECONDS,
    }
    return {**unsigned, "gate_sha256": _sha256_json(unsigned)}


def _read_discord_retirement_ack(
    *,
    gate: Mapping[str, Any],
    fd: int = ADMIN_FRAME_FD,
) -> Mapping[str, Any]:
    if fd != ADMIN_FRAME_FD or os.isatty(fd):
        _fail("discord_retirement_ack_fd_invalid")
    header = _read_exact(fd, _ACK_FRAME_HEADER.size)
    raw = bytearray()
    try:
        magic, size = _ACK_FRAME_HEADER.unpack(header)
        if magic != DISCORD_RETIREMENT_ACK_FRAME_MAGIC:
            _fail("discord_retirement_ack_magic_invalid")
        if not 1 <= size <= MAX_OWNER_APPROVAL_BYTES:
            _fail("discord_retirement_ack_bound_invalid")
        raw = _read_exact(fd, size)
        if os.read(fd, 1):
            _fail("discord_retirement_ack_trailing_data")
        value = _decode_mapping(
            bytes(raw),
            code="discord_retirement_ack_json_invalid",
        )
    finally:
        _zeroize(header)
        _zeroize(raw)
    if set(value) != _DISCORD_RETIREMENT_ACK_FIELDS:
        _fail("discord_retirement_ack_fields_invalid")
    unsigned = {key: item for key, item in value.items() if key != "ack_sha256"}
    now_unix = int(time.time())
    if (
        value["schema"] != DISCORD_RETIREMENT_ACK_SCHEMA
        or value["scope"] != "stop_and_retire_full_canary_discord_token"
        or value["release_sha"] != gate["release_sha"]
        or value["coordinator_input_sha256"] != gate["coordinator_input_sha256"]
        or value["owner_subject_sha256"] != gate["owner_subject_sha256"]
        or value["retirement_gate_sha256"] != gate["gate_sha256"]
        or value["discord_token_install_receipt_sha256"]
        != gate["discord_token_install_receipt_sha256"]
        or value["token_device"] != gate["token_device"]
        or value["token_inode"] != gate["token_inode"]
        or type(value["approved_at_unix"]) is not int
        or type(value["expires_at_unix"]) is not int
        or not value["approved_at_unix"] <= now_unix <= value["expires_at_unix"]
        or not 1
        <= value["expires_at_unix"] - value["approved_at_unix"]
        <= RECOVERY_GATE_MAX_SECONDS
        or value["expires_at_unix"] > gate["expires_at_unix"]
        or value["ack_sha256"] != _sha256_json(unsigned)
    ):
        _fail("discord_retirement_ack_not_bound")
    _digest(value["nonce_sha256"], "discord_retirement_ack_nonce_invalid")
    return copy.deepcopy(dict(value))


def _revalidate_discord_retirement_gate_source(
    *,
    coordinator_input: CoordinatorInput,
    gate: Mapping[str, Any],
) -> _DiscordInstallState | None:
    existing = _load_discord_retirement(coordinator_input)
    if existing is not None and existing[0]["state"] == "retirement_prepared":
        retirement, _snapshot, install_state = _load_prepared_discord_retirement_source(
            coordinator_input,
            require_terminal_install=False,
        )
        if (
            retirement["discord_token_install_receipt_sha256"]
            != gate["discord_token_install_receipt_sha256"]
            or retirement["token_device"] != gate["token_device"]
            or retirement["token_inode"] != gate["token_inode"]
        ):
            _fail("discord_token_retirement_gate_state_drifted")
        return install_state
    if existing is not None and existing[0]["state"] == "retired":
        retirement = existing[0]
        if (
            retirement["discord_token_install_receipt_sha256"]
            != gate["discord_token_install_receipt_sha256"]
            or retirement["token_device"] != gate["token_device"]
            or retirement["token_inode"] != gate["token_inode"]
            or any(
                os.path.lexists(path)
                for path in (
                    DISCORD_TOKEN_PATH,
                    DISCORD_TOKEN_STAGE_PATH,
                    DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
                )
            )
        ):
            _fail("discord_token_retirement_gate_state_drifted")
        return None
    if os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH):
        install_state = _load_discord_install_state(coordinator_input)
        if (
            install_state.receipt_sha256 != gate["discord_token_install_receipt_sha256"]
            or install_state.device != gate["token_device"]
            or install_state.inode != gate["token_inode"]
        ):
            _fail("discord_token_retirement_gate_state_drifted")
        return install_state
    _fail("discord_token_retirement_gate_state_drifted")


def stop_and_retire_discord_token(
    *,
    gate_emitter: Callable[[Mapping[str, Any]], None],
    ack_reader: Callable[..., Mapping[str, Any]] = (_read_discord_retirement_ack),
) -> Mapping[str, Any]:
    """Stop/prove services and idempotently finish the token retirement."""

    _require_root_linux()
    coordinator_input = load_coordinator_input()
    validate_dedicated_canary_host(coordinator_input.base_plan)
    gate = discord_retirement_gate()
    gate_emitter(gate)
    ack_reader(gate=gate)
    _load_terminal_recovery_journal(coordinator_input)
    _revalidate_discord_retirement_gate_source(
        coordinator_input=coordinator_input,
        gate=gate,
    )
    if os.path.lexists(DEFAULT_PLAN_PATH):
        active_plan = load_full_canary_plan()
        if active_plan.revision != coordinator_input.revision:
            _fail("discord_token_recovery_runtime_plan_drifted")
        stop_receipt = FullCanaryLifecycle(active_plan).stop(
            reason="operator_requested"
        )
        _validate_recovery_stop_receipt(stop_receipt, plan=active_plan)
    else:
        if any(
            os.path.lexists(path)
            for path in (
                DEFAULT_STAGED_PLAN_PATH,
                DEFAULT_WRITER_CONFIG_SOURCE,
                OWNER_APPROVAL_REQUEST_PATH,
                CANARY_BOOTSTRAP_CREDENTIAL_PATH,
            )
        ):
            _fail("discord_token_recovery_canonical_state_unresolved")
        if not _services_are_exactly_stopped_and_disabled():
            _fail("discord_token_recovery_services_not_stopped")
    with _lifecycle_lock():
        _load_terminal_recovery_journal(coordinator_input)
        if not _services_are_exactly_stopped_and_disabled():
            _fail("discord_token_recovery_services_not_stopped")
        final_state = _revalidate_discord_retirement_gate_source(
            coordinator_input=coordinator_input,
            gate=gate,
        )
        if final_state is not None:
            return _retire_discord_install_journal(
                coordinator_input=coordinator_input,
                install_state=final_state,
            )
        return _finish_prepared_discord_retirement_without_install_state(
            coordinator_input
        )


def _cli_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Root-only isolated full-canary coordinator"
    )
    parser.add_argument(
        "command",
        choices=(
            "publish-coordinator-input",
            "preflight-owner-launch",
            "preflight-recovery",
            "run",
            "recover",
            "finalize-recovery",
            "install-discord-token",
            "install-final-approval",
            "stop-and-retire-discord-token",
        ),
    )
    return parser


def _emit_frame(value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value) + b"\n"
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _unbound_failure(
    error: BaseException,
    *,
    command: str,
) -> Mapping[str, Any]:
    code, phase = _error_code_and_phase(
        error,
        fallback_phase="command",
    )
    coordinator_input: CoordinatorInput | None = None
    credential_approval: CredentialPrepareApproval | None = None
    try:
        raw = _stable_root_read(
            COORDINATOR_INPUT_PATH,
            maximum=MAX_COORDINATOR_INPUT_BYTES,
        )
        coordinator_input = CoordinatorInput.from_mapping(
            _decode_mapping(raw, code="coordinator_input_json_invalid")
        )
    except BaseException:
        pass
    if coordinator_input is not None:
        try:
            raw = _stable_root_read(
                CREDENTIAL_PREPARE_APPROVAL_PATH,
                maximum=MAX_OWNER_APPROVAL_BYTES,
            )
            credential_approval = CredentialPrepareApproval.from_mapping(
                _decode_mapping(
                    raw,
                    code="credential_prepare_approval_json_invalid",
                )
            )
        except BaseException:
            pass
    bootstrap_absent = not os.path.lexists(CANARY_BOOTSTRAP_CREDENTIAL_PATH)
    token_absent = not os.path.lexists(DISCORD_TOKEN_PATH) and not os.path.lexists(
        DISCORD_TOKEN_STAGE_PATH
    )
    token_receipt_absent = not os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH)
    retirement_terminal = True
    if os.path.lexists(DISCORD_TOKEN_RETIREMENT_RECEIPT_PATH):
        retirement_terminal = False
        if coordinator_input is not None:
            try:
                loaded_retirement = _load_discord_retirement(coordinator_input)
                retirement_terminal = bool(
                    loaded_retirement is not None
                    and loaded_retirement[0]["state"] == "retired"
                )
            except BaseException:
                # An invalid or drifting durable journal is recovery material,
                # never evidence that secret cleanup completed.
                retirement_terminal = False
    discord_token_logically_removed = (
        token_absent and token_receipt_absent and retirement_terminal
    )
    request_absent = not os.path.lexists(OWNER_APPROVAL_REQUEST_PATH)
    process_lease_absent = not os.path.lexists(COORDINATOR_PROCESS_LEASE_PATH)
    try:
        services_terminal = _services_are_exactly_stopped_and_disabled()
    except BaseException:
        services_terminal = False
    # ``run`` and ``recover`` may both have crossed a privileged SQL-session
    # boundary before an exception reaches this generic receipt builder.  A
    # generic failure has no closure witness, so it must never manufacture
    # one; only their specialized terminal records may prove closure.
    session_closed = command not in {"run", "recover"}
    complete = (
        not isinstance(error, CoordinatorCleanupBlocked)
        and bootstrap_absent
        and discord_token_logically_removed
        and request_absent
        and process_lease_absent
        and services_terminal
        and session_closed
    )
    unsigned = {
        "schema": COORDINATOR_FAILURE_SCHEMA,
        "ok": False,
        "phase": phase,
        "error_code": code,
        "release_sha": (
            None if coordinator_input is None else coordinator_input.revision
        ),
        "coordinator_input_sha256": (
            None if coordinator_input is None else coordinator_input.sha256
        ),
        "credential_prepare_approval_sha256": (
            None if credential_approval is None else credential_approval.sha256
        ),
        "owner_subject_sha256": (
            None
            if credential_approval is None
            else credential_approval.value["owner_subject_sha256"]
        ),
        "ephemeral_admin_username": (
            None
            if credential_approval is None
            else derive_ephemeral_admin_username(credential_approval.sha256)
        ),
        "full_canary_plan_sha256": None,
        "cleanup_status": "complete" if complete else "cleanup_blocked",
        "recovery_material_preserved": not complete,
        "admin_session_closed": session_closed,
        "coordinator_process_lease_retired": process_lease_absent,
        "bootstrap_login_password_disabled": False,
        # No bootstrap lease exists in an unbound/pre-admin command, so this
        # must not turn ambient absence into a fabricated cleanup action.
        "bootstrap_credential_removed": False,
        "discord_token_removed": discord_token_logically_removed,
        "services_enabled": False if services_terminal else None,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _execute_mutating_cli(
    *,
    command: str,
    operation: Callable[[], Mapping[str, Any]],
) -> int:
    with _SignalFence() as fence:
        try:
            receipt = operation()
        except BaseException as exc:
            fence.begin_cleanup()
            _emit_frame(_unbound_failure(exc, command=command))
            return 2
        fence.begin_cleanup()
        _emit_frame(receipt)
        return 0 if receipt.get("ok") is True else 2


def main(argv: Sequence[str] | None = None) -> int:
    command = _cli_parser().parse_args(argv).command
    try:
        if command == "publish-coordinator-input":
            return _execute_mutating_cli(
                command=command,
                operation=collect_and_publish_coordinator_input,
            )
        coordinator_input = load_coordinator_input()
        _attest_current_cli_process(
            coordinator_input,
            command=command,
        )
        if command == "preflight-owner-launch":
            _emit_frame(preflight_owner_launch())
            return 0
        if command == "preflight-recovery":
            _emit_frame(preflight_recovery())
            return 0
        if command == "install-discord-token":
            return _execute_mutating_cli(
                command=command,
                operation=lambda: install_discord_token(gate_emitter=_emit_frame),
            )
        if command == "install-final-approval":
            return _execute_mutating_cli(
                command=command,
                operation=lambda: install_final_owner_approval(
                    gate_emitter=_emit_frame
                ),
            )
        if command == "stop-and-retire-discord-token":
            return _execute_mutating_cli(
                command=command,
                operation=lambda: stop_and_retire_discord_token(
                    gate_emitter=_emit_frame,
                ),
            )
        if command == "recover":
            return _execute_mutating_cli(
                command=command,
                operation=lambda: recover_full_canary(gate_emitter=_emit_frame),
            )
        if command == "finalize-recovery":
            return _execute_mutating_cli(
                command=command,
                operation=finalize_recovery,
            )
        if command == "run":
            return _execute_mutating_cli(
                command=command,
                operation=lambda: run_full_canary(frame_emitter=_emit_frame),
            )
        raise AssertionError("unreachable coordinator command")
    except BaseException as exc:
        _emit_frame(_unbound_failure(exc, command=command))
        return 2


__all__ = [
    "ADMIN_FRAME_SCHEMA",
    "CANARY_DATABASE_HOST",
    "CANARY_DATABASE_NAME",
    "CANARY_DATABASE_PORT",
    "COORDINATOR_INPUT_PUBLICATION_SCHEMA",
    "COORDINATOR_INPUT_PATH",
    "COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH",
    "COORDINATOR_INPUT_SCHEMA",
    "CREDENTIAL_PREPARE_APPROVAL_PATH",
    "FINAL_APPROVAL_CANCEL_RECEIPT_FIELDS",
    "FINAL_APPROVAL_CANCEL_RECEIPT_SCHEMA",
    "FINAL_APPROVAL_FRAME_SCHEMA",
    "FINAL_APPROVAL_TRANSMIT_MARGIN_SECONDS",
    "OWNER_APPROVAL_REQUEST_PATH",
    "CoordinatorInput",
    "CoordinatorError",
    "OwnerApprovalRequest",
    "OpaqueDiscordTokenFrame",
    "OpaqueStdinAdminFrame",
    "VerifiedTLSBootstrapAdminSession",
    "collect_and_publish_coordinator_input",
    "collect_coordinator_input",
    "derive_ephemeral_admin_username",
    "discord_retirement_gate",
    "install_discord_token",
    "install_final_owner_approval",
    "finalize_recovery",
    "main",
    "preflight_owner_launch",
    "preflight_recovery",
    "publish_coordinator_input",
    "recover_full_canary",
    "run_full_canary",
    "stop_and_retire_discord_token",
]


if __name__ == "__main__":
    raise SystemExit(main())
