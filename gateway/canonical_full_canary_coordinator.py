#!/usr/bin/env python3
"""Root-only mechanical coordinator for one isolated full Muncho canary.

This module deliberately owns no task semantics.  It binds an exact reviewed
canary input, terminal writer-foundation truth, one session-bound final owner
approval, and the existing honest live driver.  It never accepts SQL, a host,
a database, a path, or a secret through argv or the environment, and it never
creates a database principal.
"""

from __future__ import annotations

import copy
import base64
import grp
import hashlib
import json
import os
import pwd
import re
import secrets
import signal
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
    FullCanaryOwnerApproval,
    FullCanaryPlan,
    _validate_artifact_source,
    _validate_edge_config,
    _validate_gateway_config,
    _validate_writer_config,
    _validate_writer_only_receipt,
    _read_stable_file,
    _release_binding,
    _validated_e2e_fixture,
    _validated_release_file,
    _MAX_HOST_IDENTITY_RECEIPT_BYTES,
    build_full_canary_plan,
    collect_service_state,
    evaluate_service_states,
    _lifecycle_lock,
    _require_root_linux,
    validate_dedicated_canary_host,
    mechanically_stop_full_canary_services,
)
from gateway.canonical_full_canary_live_driver import (
    HonestFullCanaryDriver,
    _atomic_stage_writer_config,
    _read_staged_writer_config,
    prepare_session_bound_plan,
)
from gateway.canonical_full_canary_e2e import (
    _validate_fixture as _validate_e2e_fixture_mapping,
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
SESSION_BOUND_APPROVAL_REQUEST_SCHEMA = (
    "muncho-full-canary-session-bound-owner-approval-request.v1"
)
SESSION_BOUND_COORDINATOR_RECEIPT_SCHEMA = (
    "muncho-full-canary-session-bound-coordinator-receipt.v1"
)
COORDINATOR_FAILURE_SCHEMA = (
    "muncho-full-canary-session-bound-coordinator-failure.v1"
)
DISCORD_RETIREMENT_GATE_SCHEMA = "muncho-full-canary-discord-token-retirement-gate.v1"
DISCORD_RETIREMENT_ACK_SCHEMA = "muncho-full-canary-owner-discord-retirement-ack.v1"

# Fixed owner/VM transport for the separately approved writer-foundation
# Phase-B.  This protocol runs while every live-canary service is stopped and
# before any Discord credential or legacy full-canary administrator exists.
# It is deliberately an enumerated, length-delimited state machine rather
# than a general RPC surface.
PHASE_B_OWNER_REQUEST_SCHEMA = "muncho-canonical-writer-phase-b-owner-request.v2"
PHASE_B_OWNER_RESPONSE_SCHEMA = "muncho-canonical-writer-phase-b-owner-response.v2"
PHASE_B_AUTHORITY_SCHEMA = "muncho-canonical-writer-phase-b-authority.v2"
PHASE_B_LOCAL_PREFLIGHT_SCHEMA = (
    "muncho-canonical-writer-phase-b-local-preflight.v1"
)
PHASE_B_APPLY_RECEIPT_SCHEMA = "muncho-canonical-writer-phase-b-apply-receipt.v1"
PHASE_B_APPLY_GATE_SCHEMA = "muncho-canonical-writer-phase-b-owner-gate.v2"
PHASE_B_LIVE_GATE_SCHEMA = "muncho-full-canary-phase-b-live-gate.v1"
PHASE_B_OWNER_FRAME_MAGIC = b"MPB1"
PHASE_B_OWNER_FRAME_SCHEMA = "MPB1-u32be-json-u32be-opaque.v1"
# Fresh execution consumes ten rounds (two authority rounds plus eight Cloud
# transitions).  The remaining six are the exact one-retry allowance for the
# three mutation boundaries and recovery/terminal observations; no other
# transition may consume them.
PHASE_B_MAX_ROUNDS = 16
PHASE_B_MAX_REQUEST_BYTES = 512 * 1024
PHASE_B_MAX_RESPONSE_BYTES = 512 * 1024
PHASE_B_CREDENTIAL_BYTES = 64
PHASE_B_OWNER_PUBLIC_KEY_PATH = (
    "/Users/emillomliev/.ssh/skyvision_mac_ops_ed25519.pub"
)
PHASE_B_OWNER_PUBLIC_KEY_FINGERPRINT = (
    "SHA256:7Ea5WNys9ui7FL/p0FlOnL1ZLr6NPFuewekwqRw/rdw"
)
PHASE_B_PINNED_APPROVAL_SOURCE_SHA256 = hashlib.sha256(
    PHASE_B_OWNER_PUBLIC_KEY_FINGERPRINT.encode("ascii")
).hexdigest()
PHASE_B_OWNER_PUBLIC_KEY_UID = 501
PHASE_B_OWNER_PUBLIC_KEY_GID = 20
PHASE_B_AUTHORITY_ROOT = Path("/etc/muncho/canonical-writer-phase-b")
PHASE_B_AUTHORITY_STAGE = Path(
    "/etc/muncho/.canonical-writer-phase-b.installing"
)
PHASE_B_PLAN_PATH = PHASE_B_AUTHORITY_ROOT / "plan.json"
PHASE_B_APPROVAL_PATH = PHASE_B_AUTHORITY_ROOT / "owner-approval.json"
PHASE_B_APPROVAL_SOURCE_PATH = (
    PHASE_B_AUTHORITY_ROOT / "owner-approval-source.json"
)
PHASE_B_AUTHORITY_RECEIPT_PATH = PHASE_B_AUTHORITY_ROOT / "authority-receipt.json"
PHASE_B_RESUME_APPROVAL_ROOT = PHASE_B_AUTHORITY_ROOT / "resume-approvals"

_PHASE_B_OWNER_FRAME_HEADER = struct.Struct(">4sII")
_PHASE_B_OWNER_OPERATIONS = frozenset({
    "authority_observe_initial",
    "authority_approve",
    "authority_resume_approve",
    "observe_initial",
    "observe_recovery",
    "temporary_create_or_rotate",
    "temporary_delete",
    "bootstrap_describe",
    "bootstrap_create_or_rotate",
    "observe_terminal",
})

COORDINATOR_INPUT_PATH = Path("/etc/muncho/full-canary/coordinator-input.json")
COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH = Path(
    "/etc/muncho/full-canary/coordinator-input-publication.json"
)
OBSOLETE_CREDENTIAL_PREPARE_APPROVAL_PATH = Path(
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
OBSOLETE_COORDINATOR_PROCESS_JOURNAL_PATH = Path(
    "/etc/muncho/full-canary/coordinator-process-lease.json"
)
DISCORD_TOKEN_PATH = DEFAULT_EDGE_TOKEN_PATH
DISCORD_TOKEN_STAGE_PATH = DISCORD_TOKEN_PATH.with_name(".discord-bot-token.installing")
OBSOLETE_BOOTSTRAP_CREDENTIAL_PATH = Path(
    "/etc/muncho/credentials/canonical-canary-bootstrap-db-password"
)
CANARY_DATABASE_CA_PATH = Path("/etc/muncho/trust/cloudsql-server-ca.pem")

CANARY_DATABASE_HOST = "10.91.0.3"
CANARY_DATABASE_PORT = 5432
CANARY_DATABASE_NAME = "muncho_canary_brain"
ADMIN_FRAME_FD = 0
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
DISCORD_RETIREMENT_ACK_FRAME_MAGIC = b"DRA1"
DISCORD_RETIREMENT_ACK_FRAME_SCHEMA = "DRA1-u32be-canonical-json-eof.v1"
DISCORD_RETIREMENT_GATE_MAX_SECONDS = 300

_TOKEN_FRAME_HEADER = struct.Struct("!4sI")
_FINAL_APPROVAL_FRAME_HEADER = struct.Struct("!4sI")
_ACK_FRAME_HEADER = struct.Struct("!4sI")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_TLS_NAME_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.europe-west3\.sql\.goog$"
)

_COORDINATOR_INPUT_FIELDS = frozenset({
    "schema",
    "revision",
    "writer_activation_plan",
    "writer_activation_receipt",
    "writer_activation_receipt_file_sha256",
    "identities",
    "writer_config",
    "artifacts",
    "coordinator_input_sha256",
})
_DISCORD_TOKEN_INSTALL_APPROVAL_FIELDS = frozenset({
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
    DEFAULT_STAGED_PLAN_PATH,
    DEFAULT_PLAN_PATH,
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

    if path != DISCORD_TOKEN_PATH:
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
    if path != DISCORD_TOKEN_PATH:
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
        writer_config_raw = _canonical_bytes(raw["writer_config"])
        try:
            _validate_writer_config(writer_config_raw, identities)
            _release_binding(writer_plan)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise CoordinatorError(
                "coordinator_input_writer_config_semantics_invalid"
            ) from exc
        return cls(
            value=raw,
            writer_activation_plan=writer_plan,
            identities=identities,
            artifacts=artifacts,
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


def _validated_base_e2e_fixture(
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    """Validate the published fixture before its in-memory session binding."""

    artifact = coordinator_input.artifacts["e2e_fixture"]
    raw = _read_exact_artifact(artifact, label="e2e_fixture")
    value = _decode_mapping(raw, code="coordinator_base_fixture_invalid")
    if "api_session_key_sha256" in value:
        _fail("coordinator_base_fixture_already_session_bound")
    candidate = copy.deepcopy(dict(value))
    candidate["api_session_key_sha256"] = "0" * 64
    try:
        validated = _validate_e2e_fixture_mapping(candidate)
    except BaseException as exc:
        raise CoordinatorError("coordinator_base_fixture_invalid") from exc
    validated.pop("api_session_key_sha256", None)
    release = _release_binding(coordinator_input.writer_activation_plan)
    if (
        validated != value
        or value.get("release_sha") != coordinator_input.revision
        or value.get("release_artifact_sha256") != release["artifact_sha256"]
    ):
        _fail("coordinator_base_fixture_release_drifted")
    return copy.deepcopy(dict(value))

def _phase_b_authority_file_source(
    path: Path,
    *,
    maximum: int,
    expected_uid: int,
    expected_gid: int,
    allowed_modes: frozenset[int],
) -> tuple[bytes, Mapping[str, Any]]:
    """Capture one semantically prevalidated fixed source by bytes and inode."""

    try:
        raw, item = _read_stable_file(
            path,
            maximum=maximum,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            allowed_modes=allowed_modes,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CoordinatorError(
            "phase_b_authority_source_unavailable",
            phase="phase_b_authority",
        ) from exc
    source = {
        "path": str(path),
        "file_sha256": _sha256_bytes(raw),
        "device": item.st_dev,
        "inode": item.st_ino,
        "uid": item.st_uid,
        "gid": item.st_gid,
        "mode": f"{stat.S_IMODE(item.st_mode):04o}",
        "size": item.st_size,
    }
    if any(type(source[name]) is not int or source[name] < 0 for name in (
        "device", "inode", "uid", "gid", "size"
    )):
        _fail("phase_b_authority_source_identity_invalid", phase="phase_b_authority")
    return raw, source


def _phase_b_owner_lineage(
    *,
    activation_approval: Any,
    native_approval: Any,
) -> Mapping[str, str]:
    """Bind Phase-B to the agreeing, already-validated owner receipts."""

    try:
        activation_owner_subject = _digest(
            activation_approval.value.get("owner_subject_sha256"),
            "phase_b_activation_owner_invalid",
        )
        native_owner_subject = _digest(
            native_approval.value.get("owner_subject_sha256"),
            "phase_b_native_owner_invalid",
        )
        activation_approval_source = _digest(
            activation_approval.value.get("approval_source_sha256"),
            "phase_b_activation_owner_invalid",
        )
        native_approval_source = _digest(
            native_approval.value.get("approval_source_sha256"),
            "phase_b_native_owner_invalid",
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise CoordinatorError(
            "phase_b_owner_authority_invalid",
            phase="phase_b_authority",
        ) from exc
    if (
        activation_owner_subject != native_owner_subject
        or activation_approval_source != native_approval_source
        or activation_approval_source != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
    ):
        _fail("phase_b_owner_authority_drifted", phase="phase_b_authority")
    return {
        "owner_subject_sha256": activation_owner_subject,
        "approval_source_sha256": activation_approval_source,
    }


def _phase_b_authority_provenance(
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    """Re-attest every historical root source and current staged intent.

    Historical owner approvals are evaluated at their authenticated receipt
    times.  Fresh Phase-B mutation authority is separate and is never inferred
    from an old TTL.  The durable source identities below make a byte-identical
    replacement detectable even when a digest would otherwise stay equal.
    """

    if not isinstance(coordinator_input, CoordinatorInput):
        raise TypeError("CoordinatorInput is required")
    try:
        from gateway import canonical_writer_activation as activation
        from gateway import canonical_writer_host_authority as host_authority
        from gateway.canonical_writer_config_collector import (
            EVIDENCE_ROOT as CONFIG_COLLECTOR_EVIDENCE_ROOT,
            load_config_collector_receipt,
        )

        _validate_coordinator_input_live(coordinator_input)
        writer_plan = activation.load_activation_plan(
            activation.DEFAULT_PLAN_PATH
        )
        if writer_plan.to_mapping() != coordinator_input.writer_activation_plan.to_mapping():
            _fail("phase_b_activation_plan_drifted", phase="phase_b_authority")
        writer_receipt = _validate_writer_only_receipt(
            coordinator_input.value["writer_activation_receipt"],
            plan=writer_plan,
        )
        activation_completed = int(writer_receipt["completed_at_unix"])
        activation_approval = host_authority.OwnerApprovalReceipt.from_mapping(
            writer_receipt["owner_approval_receipt"]
        )
        activation_approval.require(
            scope="activation",
            plan_sha256=writer_plan.sha256,
            now_unix=activation_completed,
        )
        if activation_approval.sha256 != writer_receipt[
            "owner_approval_receipt_sha256"
        ]:
            _fail("phase_b_activation_approval_drifted", phase="phase_b_authority")

        native_plan = host_authority.load_native_observation_plan()
        if native_plan.to_mapping() != writer_plan.native_observation_receipt["plan"]:
            _fail("phase_b_native_plan_drifted", phase="phase_b_authority")
        native_receipt = activation.load_durable_native_observation_receipt(
            native_plan
        )
        if native_receipt.to_mapping() != writer_plan.native_observation_receipt:
            _fail("phase_b_native_receipt_drifted", phase="phase_b_authority")
        native_approval, external_iam, external_evidence = (
            activation._load_durable_native_authority_chain(
                native_plan,
                native_receipt,
            )
        )
        owner_lineage = _phase_b_owner_lineage(
            activation_approval=activation_approval,
            native_approval=native_approval,
        )

        collector_sha256 = str(native_plan.value["config_collector_receipt_sha256"])
        collector = load_config_collector_receipt(
            revision=writer_plan.revision,
            receipt_sha256=collector_sha256,
            require_fresh=False,
        )
        collector_path = (
            CONFIG_COLLECTOR_EVIDENCE_ROOT
            / writer_plan.revision
            / f"{collector_sha256}.json"
        )
        collector_raw, collector_source = _phase_b_authority_file_source(
            collector_path,
            maximum=MAX_COORDINATOR_INPUT_BYTES,
            expected_uid=0,
            expected_gid=0,
            allowed_modes=frozenset({0o400}),
        )
        if collector_raw not in {
            _canonical_bytes(collector.to_mapping()),
            _canonical_bytes(collector.to_mapping()) + b"\n",
        }:
            _fail("phase_b_config_collector_drifted", phase="phase_b_authority")

        activation_receipt_path = _success_receipt_path(
            writer_plan,
            create_parent=False,
        )
        activation_approval_path = host_authority.owner_approval_receipt_path(
            activation_approval
        )
        native_receipt_path = activation._native_receipt_path(native_plan)
        native_approval_path = host_authority.owner_approval_receipt_path(
            native_approval
        )
        external_iam_path = Path(str(external_evidence["path"]))

        fixed_root_sources: dict[str, tuple[Path, int]] = {
            "coordinator_input": (COORDINATOR_INPUT_PATH, MAX_COORDINATOR_INPUT_BYTES),
            "activation_plan": (activation.DEFAULT_PLAN_PATH, MAX_COORDINATOR_INPUT_BYTES),
            "activation_receipt": (activation_receipt_path, MAX_COORDINATOR_INPUT_BYTES),
            "activation_owner_approval": (activation_approval_path, 64 * 1024),
            "native_plan": (activation.DEFAULT_NATIVE_PLAN_PATH, MAX_COORDINATOR_INPUT_BYTES),
            "native_receipt": (native_receipt_path, MAX_COORDINATOR_INPUT_BYTES),
            "native_owner_approval": (native_approval_path, 64 * 1024),
            "external_iam_receipt": (external_iam_path, 64 * 1024),
            "host_identity_receipt": (DEFAULT_HOST_IDENTITY_RECEIPT, _MAX_HOST_IDENTITY_RECEIPT_BYTES),
        }
        sources: dict[str, Mapping[str, Any]] = {
            "config_collector_receipt": collector_source,
        }
        source_raw: dict[str, bytes] = {
            "config_collector_receipt": collector_raw,
        }
        for label, (path, maximum) in fixed_root_sources.items():
            raw, source = _phase_b_authority_file_source(
                path,
                maximum=maximum,
                expected_uid=0,
                expected_gid=0,
                allowed_modes=frozenset({0o400}),
            )
            sources[label] = source
            source_raw[label] = raw

        for label, artifact in (
            ("gateway_config_intent", coordinator_input.artifacts["gateway_config"]),
            ("edge_config_intent", coordinator_input.artifacts["edge_config"]),
            ("fixture_intent", coordinator_input.artifacts["e2e_fixture"]),
        ):
            raw, source = _phase_b_authority_file_source(
                artifact.source_path,
                maximum=artifact.maximum_bytes,
                expected_uid=artifact.uid,
                expected_gid=artifact.gid,
                allowed_modes=frozenset({artifact.mode}),
            )
            if _sha256_bytes(raw) != artifact.sha256:
                _fail("phase_b_staged_intent_drifted", phase="phase_b_authority")
            sources[label] = source
            source_raw[label] = raw

        if source_raw["coordinator_input"] not in {
            _canonical_bytes(coordinator_input.value),
            _canonical_bytes(coordinator_input.value) + b"\n",
        }:
            _fail("phase_b_coordinator_input_drifted", phase="phase_b_authority")
        if source_raw["activation_plan"] not in {
            _canonical_bytes(writer_plan.to_mapping()),
            _canonical_bytes(writer_plan.to_mapping()) + b"\n",
        }:
            _fail("phase_b_activation_plan_drifted", phase="phase_b_authority")
        if source_raw["activation_receipt"] not in {
            _canonical_bytes(writer_receipt),
            _canonical_bytes(writer_receipt) + b"\n",
        }:
            _fail("phase_b_activation_receipt_drifted", phase="phase_b_authority")
        if source_raw["activation_owner_approval"] not in {
            _canonical_bytes(activation_approval.to_mapping()),
            _canonical_bytes(activation_approval.to_mapping()) + b"\n",
        }:
            _fail("phase_b_activation_approval_drifted", phase="phase_b_authority")
        if source_raw["native_plan"] not in {
            _canonical_bytes(native_plan.to_mapping()),
            _canonical_bytes(native_plan.to_mapping()) + b"\n",
        } or source_raw["native_receipt"] not in {
            _canonical_bytes(native_receipt.to_mapping()),
            _canonical_bytes(native_receipt.to_mapping()) + b"\n",
        }:
            _fail("phase_b_native_truth_drifted", phase="phase_b_authority")
        if source_raw["native_owner_approval"] not in {
            _canonical_bytes(native_approval.to_mapping()),
            _canonical_bytes(native_approval.to_mapping()) + b"\n",
        }:
            _fail("phase_b_native_approval_drifted", phase="phase_b_authority")
        if source_raw["external_iam_receipt"] not in {
            _canonical_bytes(external_iam.to_mapping()),
            _canonical_bytes(external_iam.to_mapping()) + b"\n",
        }:
            _fail("phase_b_external_iam_drifted", phase="phase_b_authority")

        host_value = _decode_mapping(
            source_raw["host_identity_receipt"].rstrip(b"\n"),
            code="phase_b_host_identity_receipt_invalid",
        )
        host_receipt_sha256 = _digest(
            host_value.get("receipt_sha256"),
            "phase_b_host_identity_receipt_invalid",
        )
        if external_iam.policy_sha256 != writer_plan.digests.external_iam_policy_sha256:
            _fail("phase_b_external_iam_drifted", phase="phase_b_authority")

        fixture = _validated_base_e2e_fixture(coordinator_input)
        authority_sources = {
            **sources,
            # This owner-local source is filled exactly once after the first
            # MPB1 response.  Keeping the placeholder in the initial context
            # makes the monotonic extension explicit and hash-visible.
            "owner_resume_public_key": None,
        }
        result = {
            "release_sha": writer_plan.revision,
            "coordinator_input_sha256": coordinator_input.sha256,
            **owner_lineage,
            "activation_plan_sha256": writer_plan.sha256,
            "writer_activation_receipt_sha256": writer_receipt["receipt_sha256"],
            "activation_owner_approval_sha256": activation_approval.sha256,
            "activation_approval_issued_at_unix": activation_approval.value[
                "approved_at_unix"
            ],
            "activation_approval_expires_at_unix": activation_approval.value[
                "expires_at_unix"
            ],
            "native_observation_plan_sha256": native_plan.sha256,
            "native_observation_receipt_sha256": native_receipt.sha256,
            "native_observation_approval_sha256": native_approval.sha256,
            "native_approval_issued_at_unix": native_approval.value[
                "approved_at_unix"
            ],
            "native_approval_expires_at_unix": native_approval.value[
                "expires_at_unix"
            ],
            "external_iam_policy_sha256": external_iam.policy_sha256,
            "external_iam_receipt_sha256": external_iam.sha256,
            "config_collector_receipt_sha256": collector.sha256,
            "gateway_config_intent_sha256": coordinator_input.artifacts[
                "gateway_config"
            ].sha256,
            "edge_config_intent_sha256": coordinator_input.artifacts[
                "edge_config"
            ].sha256,
            "fixture_intent_sha256": _sha256_json(fixture),
            "host_identity_receipt_sha256": host_receipt_sha256,
            "owner_resume_public_key_ed25519_hex": None,
            "owner_resume_key_id": None,
            "owner_resume_public_key_file_sha256": None,
            "owner_resume_public_fingerprint": None,
            "authority_sources": dict(sorted(authority_sources.items())),
        }
        return {
            **result,
            "authority_sources_sha256": _sha256_json(result["authority_sources"]),
        }
    except CoordinatorError:
        raise
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CoordinatorError(
            "phase_b_authority_chain_invalid",
            phase="phase_b_authority",
        ) from exc


class _PhaseBOwnerOperationError(CoordinatorError):
    def __init__(self, code: str, *, reconciliation_required: bool) -> None:
        self.reconciliation_required = reconciliation_required
        super().__init__(code, phase="phase_b_owner_transport")


def _read_phase_b_exact(size: int, *, maximum: int) -> bytearray:
    if (
        type(size) is not int
        or type(maximum) is not int
        or not 0 <= size <= maximum
    ):
        _fail("phase_b_owner_frame_bound_invalid", phase="phase_b_owner_transport")
    value = bytearray()
    while len(value) < size:
        try:
            chunk = os.read(ADMIN_FRAME_FD, size - len(value))
        except OSError as exc:
            _zeroize(value)
            raise CoordinatorError(
                "phase_b_owner_frame_read_failed",
                phase="phase_b_owner_transport",
            ) from exc
        if not chunk:
            _zeroize(value)
            _fail("phase_b_owner_frame_truncated", phase="phase_b_owner_transport")
        value.extend(chunk)
    return value


def _phase_b_no_secret_fields(value: Any) -> None:
    """Reject secret-like receipt keys without trying to classify semantics."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = key.casefold() if isinstance(key, str) else ""
            declarative_absence = lowered in {
                "secret_material_recorded",
                "password_or_digest_recorded",
                "content_or_digest_recorded",
            } and item is False
            if not isinstance(key, str) or (
                not declarative_absence
                and any(
                    marker in lowered
                    for marker in (
                        "password",
                        "secret",
                        "secret_digest",
                        "credential_digest",
                        "credential_value",
                    )
                )
            ):
                _fail(
                    "phase_b_owner_receipt_secret_field_forbidden",
                    phase="phase_b_owner_transport",
                )
            _phase_b_no_secret_fields(item)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for item in value:
            _phase_b_no_secret_fields(item)


_PHASE_B_OWNER_RESUME_AUTHORITY_FIELDS = frozenset({
    "public_key_ed25519_hex",
    "key_id",
    "public_key_file_sha256",
    "public_fingerprint",
    "public_key_source",
})
_PHASE_B_OWNER_PUBLIC_SOURCE_FIELDS = frozenset({
    "path",
    "file_sha256",
    "device",
    "inode",
    "uid",
    "gid",
    "mode",
    "size",
})


def _phase_b_ssh_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def _validate_phase_b_owner_resume_authority(value: Any) -> Mapping[str, Any]:
    """Validate and independently derive the fixed owner's public identity."""

    if not isinstance(value, Mapping) or set(value) != (
        _PHASE_B_OWNER_RESUME_AUTHORITY_FIELDS
    ):
        _fail("phase_b_owner_public_authority_invalid", phase="phase_b_authority")
    raw = copy.deepcopy(dict(value))
    source = raw["public_key_source"]
    public_hex = raw["public_key_ed25519_hex"]
    if (
        not isinstance(source, Mapping)
        or set(source) != _PHASE_B_OWNER_PUBLIC_SOURCE_FIELDS
        or not isinstance(public_hex, str)
        or re.fullmatch(r"[0-9a-f]{64}", public_hex) is None
        or raw["key_id"] != _sha256_bytes(bytes.fromhex(public_hex))
        or raw["public_key_file_sha256"] != source.get("file_sha256")
        or source.get("path") != PHASE_B_OWNER_PUBLIC_KEY_PATH
        or source.get("uid") != PHASE_B_OWNER_PUBLIC_KEY_UID
        or source.get("gid") != PHASE_B_OWNER_PUBLIC_KEY_GID
        or source.get("mode") != "0600"
        or type(source.get("device")) is not int
        or source["device"] < 0
        or type(source.get("inode")) is not int
        or source["inode"] <= 0
        or type(source.get("size")) is not int
        or not 1 <= source["size"] <= 4096
    ):
        _fail("phase_b_owner_public_authority_invalid", phase="phase_b_authority")
    _digest(
        raw["public_key_file_sha256"],
        "phase_b_owner_public_authority_invalid",
    )
    public_bytes = bytes.fromhex(public_hex)
    public_blob = _phase_b_ssh_string(b"ssh-ed25519") + _phase_b_ssh_string(
        public_bytes
    )
    fingerprint = "SHA256:" + base64.b64encode(
        hashlib.sha256(public_blob).digest()
    ).decode("ascii").rstrip("=")
    if (
        raw["public_fingerprint"] != fingerprint
        or fingerprint != PHASE_B_OWNER_PUBLIC_KEY_FINGERPRINT
    ):
        _fail("phase_b_owner_public_authority_invalid", phase="phase_b_authority")
    _phase_b_no_secret_fields(raw)
    return raw


class _FixedPhaseBVMProtocol:
    """One in-process, monotonic VM half of the fixed MPB1 exchange."""

    _RESPONSE_FIELDS = frozenset({
        "schema",
        "frame_schema",
        "ok",
        "operation",
        "sequence",
        "request_sha256",
        "idempotency_key",
        "authority_context_sha256",
        "phase_b_plan_sha256",
        "phase_b_approval_sha256",
        "credential_present",
        "credential_length",
        "result",
        "error_code",
        "completed_at_unix",
        "response_sha256",
    })

    def __init__(
        self,
        *,
        provenance: Mapping[str, Any],
        phase_b_plan_sha256: str | None,
        phase_b_approval_sha256: str | None,
        approval_expires_at_unix: int,
        sequence: int = 0,
        previous_response_sha256: str | None = None,
    ) -> None:
        if (
            not isinstance(provenance, Mapping)
            or type(approval_expires_at_unix) is not int
            or approval_expires_at_unix <= int(time.time())
            or type(sequence) is not int
            or not 0 <= sequence < PHASE_B_MAX_ROUNDS
        ):
            _fail("phase_b_owner_protocol_context_invalid", phase="phase_b_owner_transport")
        if phase_b_plan_sha256 is not None:
            _digest(phase_b_plan_sha256, "phase_b_owner_protocol_plan_invalid")
        if phase_b_approval_sha256 is not None:
            _digest(phase_b_approval_sha256, "phase_b_owner_protocol_approval_invalid")
        if previous_response_sha256 is not None:
            _digest(previous_response_sha256, "phase_b_owner_protocol_chain_invalid")
        self._provenance = copy.deepcopy(dict(provenance))
        self._authority_context_sha256 = _sha256_json(self._provenance)
        self._plan_sha256 = phase_b_plan_sha256
        self._approval_sha256 = phase_b_approval_sha256
        self._expires = approval_expires_at_unix
        self._sequence = sequence
        self._previous_response_sha256 = previous_response_sha256
        self._historical_resume_bound = False

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def previous_response_sha256(self) -> str | None:
        return self._previous_response_sha256

    @property
    def authority_context(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._provenance)

    def bind_owner_resume_authority(self, value: Any) -> Mapping[str, Any]:
        authority = _validate_phase_b_owner_resume_authority(value)
        sources = self._provenance.get("authority_sources")
        if (
            self._sequence != 1
            or self._plan_sha256 is not None
            or self._approval_sha256 is not None
            or not isinstance(sources, Mapping)
            or sources.get("owner_resume_public_key") is not None
            or any(
                self._provenance.get(name) is not None
                for name in (
                    "owner_resume_public_key_ed25519_hex",
                    "owner_resume_key_id",
                    "owner_resume_public_key_file_sha256",
                    "owner_resume_public_fingerprint",
                )
            )
        ):
            _fail(
                "phase_b_owner_protocol_rebind_forbidden",
                phase="phase_b_owner_transport",
            )
        updated_sources = copy.deepcopy(dict(sources))
        updated_sources["owner_resume_public_key"] = copy.deepcopy(
            dict(authority["public_key_source"])
        )
        self._provenance.update({
            "owner_resume_public_key_ed25519_hex": authority[
                "public_key_ed25519_hex"
            ],
            "owner_resume_key_id": authority["key_id"],
            "owner_resume_public_key_file_sha256": authority[
                "public_key_file_sha256"
            ],
            "owner_resume_public_fingerprint": authority[
                "public_fingerprint"
            ],
            "authority_sources": dict(sorted(updated_sources.items())),
        })
        self._provenance["authority_sources_sha256"] = _sha256_json(
            self._provenance["authority_sources"]
        )
        self._authority_context_sha256 = _sha256_json(self._provenance)
        return authority

    def bind_plan(self, plan_sha256: str) -> None:
        if self._plan_sha256 is not None or self._approval_sha256 is not None:
            _fail("phase_b_owner_protocol_rebind_forbidden", phase="phase_b_owner_transport")
        self._plan_sha256 = _digest(
            plan_sha256, "phase_b_owner_protocol_plan_invalid"
        )

    def bind_approval(
        self,
        *,
        approval_sha256: str,
        approval_expires_at_unix: int,
    ) -> None:
        if self._plan_sha256 is None or self._approval_sha256 is not None:
            _fail("phase_b_owner_protocol_rebind_forbidden", phase="phase_b_owner_transport")
        self._approval_sha256 = _digest(
            approval_sha256, "phase_b_owner_protocol_approval_invalid"
        )
        if (
            type(approval_expires_at_unix) is not int
            or approval_expires_at_unix <= int(time.time())
            or approval_expires_at_unix > self._expires
        ):
            _fail("phase_b_owner_protocol_expiry_invalid", phase="phase_b_owner_transport")
        self._expires = approval_expires_at_unix

    def bind_resume_approval(
        self,
        *,
        expected_previous_approval_sha256: str,
        approval_sha256: str,
        approval_expires_at_unix: int,
    ) -> None:
        if (
            self._plan_sha256 is None
            or self._approval_sha256
            != _digest(
                expected_previous_approval_sha256,
                "phase_b_owner_protocol_approval_invalid",
            )
            or self._sequence
            != (2 if self._historical_resume_bound else 1)
        ):
            _fail(
                "phase_b_owner_protocol_rebind_forbidden",
                phase="phase_b_owner_transport",
            )
        replacement = _digest(
            approval_sha256,
            "phase_b_owner_protocol_approval_invalid",
        )
        if (
            replacement == self._approval_sha256
            or type(approval_expires_at_unix) is not int
            or approval_expires_at_unix <= int(time.time())
        ):
            _fail(
                "phase_b_owner_protocol_expiry_invalid",
                phase="phase_b_owner_transport",
            )
        self._approval_sha256 = replacement
        self._expires = approval_expires_at_unix

    def bind_historical_resume_approval(
        self,
        *,
        expected_previous_approval_sha256: str,
        approval_sha256: str,
        approval_expires_at_unix: int,
    ) -> None:
        """Advance protocol chaining without granting expired mutation authority."""

        if (
            self._plan_sha256 is None
            or self._historical_resume_bound
            or self._approval_sha256
            != _digest(
                expected_previous_approval_sha256,
                "phase_b_owner_protocol_approval_invalid",
            )
            or self._sequence != 1
        ):
            _fail(
                "phase_b_owner_protocol_rebind_forbidden",
                phase="phase_b_owner_transport",
            )
        replacement = _digest(
            approval_sha256,
            "phase_b_owner_protocol_approval_invalid",
        )
        if (
            replacement == self._approval_sha256
            or type(approval_expires_at_unix) is not int
            or approval_expires_at_unix > int(time.time())
        ):
            _fail(
                "phase_b_owner_protocol_expiry_invalid",
                phase="phase_b_owner_transport",
            )
        # Keep the still-fresh outer apply-gate expiry solely as the MPB
        # transport deadline.  The historical approval itself never becomes
        # mutation authority; the next exchange must append a fresh successor.
        self._approval_sha256 = replacement
        self._historical_resume_bound = True

    def exchange(
        self,
        operation: str,
        *,
        payload: Mapping[str, Any],
        boundary_kind: str | None = None,
        boundary_ordinal: int | None = None,
        credential_expected: bool = False,
    ) -> tuple[Mapping[str, Any], bytearray | None]:
        if (
            operation not in _PHASE_B_OWNER_OPERATIONS
            or not isinstance(payload, Mapping)
            or boundary_kind not in {None, "temporary", "bootstrap"}
            or (
                boundary_kind is None
                and boundary_ordinal is not None
                or boundary_kind is not None
                and (type(boundary_ordinal) is not int or not 0 <= boundary_ordinal <= 7)
            )
            or type(credential_expected) is not bool
            or self._sequence >= PHASE_B_MAX_ROUNDS
        ):
            _fail("phase_b_owner_request_invalid", phase="phase_b_owner_transport")
        now_unix = int(time.time())
        if now_unix >= self._expires:
            _fail("phase_b_owner_approval_expired", phase="phase_b_owner_transport")
        unsigned = {
            "schema": PHASE_B_OWNER_REQUEST_SCHEMA,
            "frame_schema": PHASE_B_OWNER_FRAME_SCHEMA,
            "operation": operation,
            "sequence": self._sequence,
            "previous_response_sha256": self._previous_response_sha256,
            "authority_context_sha256": self._authority_context_sha256,
            "authority_context": copy.deepcopy(self._provenance),
            "phase_b_plan_sha256": self._plan_sha256,
            "phase_b_approval_sha256": self._approval_sha256,
            "boundary_kind": boundary_kind,
            "boundary_ordinal": boundary_ordinal,
            "credential_expected": credential_expected,
            "payload": copy.deepcopy(dict(payload)),
            "issued_at_unix": now_unix,
            "expires_at_unix": self._expires,
        }
        idempotency_projection = {
            "authority_context_sha256": self._authority_context_sha256,
            "phase_b_plan_sha256": self._plan_sha256,
            "phase_b_approval_sha256": self._approval_sha256,
            "operation": operation,
            "sequence": self._sequence,
            "previous_response_sha256": self._previous_response_sha256,
            "payload": unsigned["payload"],
        }
        unsigned["idempotency_key"] = _sha256_json(idempotency_projection)
        request = {**unsigned, "request_sha256": _sha256_json(unsigned)}
        _phase_b_no_secret_fields(request)
        if len(_canonical_bytes(request)) > PHASE_B_MAX_REQUEST_BYTES:
            _fail("phase_b_owner_request_oversized", phase="phase_b_owner_transport")
        frame_emitter = _ACTIVE_PHASE_B_FRAME_EMITTER
        if frame_emitter is None:
            _fail("phase_b_owner_transport_inactive", phase="phase_b_owner_transport")
        frame_emitter(request)

        header = _read_phase_b_exact(
            _PHASE_B_OWNER_FRAME_HEADER.size,
            maximum=_PHASE_B_OWNER_FRAME_HEADER.size,
        )
        receipt_raw: bytearray | None = None
        credential: bytearray | None = None
        try:
            magic, receipt_size, credential_size = _PHASE_B_OWNER_FRAME_HEADER.unpack(
                header
            )
            if (
                magic != PHASE_B_OWNER_FRAME_MAGIC
                or not 2 <= receipt_size <= PHASE_B_MAX_RESPONSE_BYTES
                or credential_size not in {0, PHASE_B_CREDENTIAL_BYTES}
            ):
                _fail("phase_b_owner_frame_invalid", phase="phase_b_owner_transport")
            receipt_raw = _read_phase_b_exact(
                receipt_size,
                maximum=PHASE_B_MAX_RESPONSE_BYTES,
            )
            if credential_size:
                credential = _read_phase_b_exact(
                    credential_size,
                    maximum=PHASE_B_CREDENTIAL_BYTES,
                )
            receipt = _decode_mapping(
                bytes(receipt_raw),
                code="phase_b_owner_response_invalid",
            )
            if set(receipt) != self._RESPONSE_FIELDS:
                _fail("phase_b_owner_response_invalid", phase="phase_b_owner_transport")
            response_sha = _digest(
                receipt["response_sha256"], "phase_b_owner_response_invalid"
            )
            response_unsigned = {
                key: copy.deepcopy(value)
                for key, value in receipt.items()
                if key != "response_sha256"
            }
            if (
                receipt["schema"] != PHASE_B_OWNER_RESPONSE_SCHEMA
                or receipt["frame_schema"] != PHASE_B_OWNER_FRAME_SCHEMA
                or receipt["operation"] != operation
                or receipt["sequence"] != self._sequence
                or receipt["request_sha256"] != request["request_sha256"]
                or receipt["idempotency_key"] != request["idempotency_key"]
                or receipt["authority_context_sha256"]
                != self._authority_context_sha256
                or receipt["phase_b_plan_sha256"] != self._plan_sha256
                or receipt["phase_b_approval_sha256"] != self._approval_sha256
                or receipt["response_sha256"] != _sha256_json(response_unsigned)
                or type(receipt["ok"]) is not bool
                or type(receipt["completed_at_unix"]) is not int
                or not now_unix <= receipt["completed_at_unix"] <= self._expires
                or type(receipt["credential_present"]) is not bool
                or type(receipt["credential_length"]) is not int
                or receipt["credential_length"] != credential_size
                or receipt["credential_present"] is not bool(credential_size)
                or (
                    receipt["ok"] is True
                    and receipt["credential_present"] is not credential_expected
                    or receipt["ok"] is False
                    and receipt["credential_present"] is not False
                )
                or not isinstance(receipt["result"], Mapping)
                or (
                    receipt["ok"] is True
                    and receipt["error_code"] is not None
                    or receipt["ok"] is False
                    and not isinstance(receipt["error_code"], str)
                )
            ):
                _fail("phase_b_owner_response_invalid", phase="phase_b_owner_transport")
            _phase_b_no_secret_fields(receipt)
            if credential is not None:
                if (
                    len(credential) != PHASE_B_CREDENTIAL_BYTES
                    or b"\x00" in credential
                    or any(value < 0x20 or value == 0x7F for value in credential)
                ):
                    _fail("phase_b_owner_credential_invalid", phase="phase_b_owner_transport")
            self._sequence += 1
            self._previous_response_sha256 = response_sha
            if receipt["ok"] is not True:
                reconciliation = receipt["result"].get(
                    "mutation_reconciliation_required"
                )
                if type(reconciliation) is not bool:
                    _fail("phase_b_owner_response_invalid", phase="phase_b_owner_transport")
                _zeroize(credential)
                credential = None
                raise _PhaseBOwnerOperationError(
                    str(receipt["error_code"]),
                    reconciliation_required=reconciliation,
                )
            result = copy.deepcopy(dict(receipt["result"]))
            returned = credential
            credential = None
            return result, returned
        finally:
            _zeroize(header)
            _zeroize(receipt_raw)
            _zeroize(credential)


class _FixedPhaseBTemporaryAdminBoundary:
    def __init__(self, protocol: _FixedPhaseBVMProtocol, ordinal: int) -> None:
        self._protocol = protocol
        self._ordinal = ordinal
        self._reconciliation_required = False
        self._owner_subject_sha256: str | None = None
        self._mutation_context_sha256: str | None = None
        self._authority_receipt: Mapping[str, Any] | None = None
        self._absence_receipt: Mapping[str, Any] | None = None

    def _exchange(
        self,
        operation: str,
        payload: Mapping[str, Any],
        *,
        credential_expected: bool = False,
    ) -> tuple[Mapping[str, Any], bytearray | None]:
        try:
            result = self._protocol.exchange(
                operation,
                payload=payload,
                boundary_kind="temporary",
                boundary_ordinal=self._ordinal,
                credential_expected=credential_expected,
            )
        except _PhaseBOwnerOperationError as exc:
            self._reconciliation_required = exc.reconciliation_required
            raise
        self._reconciliation_required = False
        return result

    def begin_mutation_observation(
        self,
        *,
        expected_owner_subject_sha256: str,
        expected_mutation_context_sha256: str,
    ) -> None:
        if self._owner_subject_sha256 is not None:
            _fail("phase_b_owner_boundary_replay_forbidden")
        self._owner_subject_sha256 = _digest(
            expected_owner_subject_sha256,
            "phase_b_owner_boundary_subject_invalid",
        )
        self._mutation_context_sha256 = _digest(
            expected_mutation_context_sha256,
            "phase_b_owner_boundary_context_invalid",
        )

    def create_or_rotate_recovery(self, username: str) -> bytearray:
        if self._owner_subject_sha256 is None or self._mutation_context_sha256 is None:
            _fail("phase_b_owner_boundary_observation_missing")
        result, credential = self._exchange(
            "temporary_create_or_rotate",
            {
                "username": username,
                "expected_owner_subject_sha256": self._owner_subject_sha256,
                "expected_mutation_context_sha256": self._mutation_context_sha256,
            },
            credential_expected=True,
        )
        if credential is None:
            _fail("phase_b_owner_credential_missing", phase="phase_b_owner_transport")
        authority = result.get("authority_receipt")
        if not isinstance(authority, Mapping):
            _zeroize(credential)
            _fail("phase_b_owner_response_invalid", phase="phase_b_owner_transport")
        self._authority_receipt = copy.deepcopy(dict(authority))
        return credential

    def mutation_reconciliation_required(self) -> bool:
        return self._reconciliation_required

    def require_current_authority(self, username: str) -> None:
        if self._authority_receipt is None:
            _fail("phase_b_owner_authority_missing")
        expected = hashlib.sha256(username.encode("ascii")).hexdigest()
        if self._authority_receipt.get("username_sha256") != expected:
            _fail("phase_b_owner_authority_drifted")

    def temporary_admin_authority_receipt(
        self, username: str
    ) -> Mapping[str, Any]:
        self.require_current_authority(username)
        assert self._authority_receipt is not None
        return copy.deepcopy(dict(self._authority_receipt))

    def delete_and_confirm_absent(self, username: str) -> None:
        result, _credential = self._exchange(
            "temporary_delete",
            {
                "username": username,
                "authority_receipt_sha256": (
                    None
                    if self._authority_receipt is None
                    else self._authority_receipt.get("receipt_sha256")
                ),
            },
        )
        receipt = result.get("absence_receipt")
        if not isinstance(receipt, Mapping):
            _fail("phase_b_owner_response_invalid", phase="phase_b_owner_transport")
        self._absence_receipt = copy.deepcopy(dict(receipt))

    def reconciliation_receipt(self) -> Mapping[str, Any]:
        if self._absence_receipt is None:
            _fail("phase_b_owner_absence_receipt_missing")
        return copy.deepcopy(dict(self._absence_receipt))


class _FixedPhaseBBootstrapLoginBoundary:
    def __init__(self, protocol: _FixedPhaseBVMProtocol, ordinal: int) -> None:
        self._protocol = protocol
        self._ordinal = ordinal
        self._reconciliation_required = False
        self._authority_receipt: Mapping[str, Any] | None = None

    def _exchange(
        self,
        operation: str,
        payload: Mapping[str, Any],
        *,
        credential_expected: bool = False,
    ) -> tuple[Mapping[str, Any], bytearray | None]:
        try:
            result = self._protocol.exchange(
                operation,
                payload=payload,
                boundary_kind="bootstrap",
                boundary_ordinal=self._ordinal,
                credential_expected=credential_expected,
            )
        except _PhaseBOwnerOperationError as exc:
            self._reconciliation_required = exc.reconciliation_required
            raise
        self._reconciliation_required = False
        return result

    def describe(self) -> Mapping[str, Any] | None:
        result, _credential = self._exchange("bootstrap_describe", {})
        resource = result.get("resource")
        if resource is not None and not isinstance(resource, Mapping):
            _fail("phase_b_owner_response_invalid", phase="phase_b_owner_transport")
        return None if resource is None else copy.deepcopy(dict(resource))

    def create_or_rotate_recovery(self) -> bytearray:
        result, credential = self._exchange(
            "bootstrap_create_or_rotate",
            {},
            credential_expected=True,
        )
        if credential is None:
            _fail("phase_b_owner_credential_missing", phase="phase_b_owner_transport")
        authority = result.get("authority_receipt")
        if not isinstance(authority, Mapping):
            _zeroize(credential)
            _fail("phase_b_owner_response_invalid", phase="phase_b_owner_transport")
        self._authority_receipt = copy.deepcopy(dict(authority))
        return credential

    def mutation_reconciliation_required(self) -> bool:
        return self._reconciliation_required

    def require_current_authority(self) -> None:
        if self._authority_receipt is None:
            _fail("phase_b_owner_authority_missing")

    def authority_receipt(self) -> Mapping[str, Any]:
        self.require_current_authority()
        assert self._authority_receipt is not None
        return copy.deepcopy(dict(self._authority_receipt))


class FixedPhaseBOwnerCloudSQLTransportBoundary:
    """Exact VM-side Cloud edge; construction is sealed to the zero-arg factory."""

    _CONSTRUCTION_TOKEN = object()

    def __init__(
        self,
        token: object,
        protocol: _FixedPhaseBVMProtocol,
        plan: Any,
        approval: Mapping[str, Any],
    ) -> None:
        if token is not self._CONSTRUCTION_TOKEN:
            _fail("phase_b_owner_boundary_construction_forbidden")
        self._protocol = protocol
        self._plan = plan
        self._approval = copy.deepcopy(dict(approval))
        self._temporary_ordinal = 0
        self._bootstrap_ordinal = 0

    def _require_plan(self, plan: Any) -> None:
        if (
            not hasattr(plan, "to_mapping")
            or plan.to_mapping() != self._plan.to_mapping()
        ):
            _fail("phase_b_owner_boundary_plan_changed")

    def temporary_admin_factory(self, plan: Any) -> Any:
        self._require_plan(plan)
        ordinal = self._temporary_ordinal
        if ordinal > 7:
            _fail("phase_b_owner_boundary_count_exceeded")
        self._temporary_ordinal += 1
        return _FixedPhaseBTemporaryAdminBoundary(self._protocol, ordinal)

    def bootstrap_login_factory(self, plan: Any) -> Any:
        self._require_plan(plan)
        ordinal = self._bootstrap_ordinal
        if ordinal > 7:
            _fail("phase_b_owner_boundary_count_exceeded")
        self._bootstrap_ordinal += 1
        return _FixedPhaseBBootstrapLoginBoundary(self._protocol, ordinal)

    def observe_initial(self, plan: Any) -> Mapping[str, Any]:
        self._require_plan(plan)
        result, _credential = self._protocol.exchange(
            "observe_initial",
            payload={
                "phase_b_plan": self._plan.to_mapping(),
                "phase_b_approval": copy.deepcopy(self._approval),
            },
        )
        observation = result.get("cloud_observation")
        if not isinstance(observation, Mapping):
            _fail("phase_b_owner_response_invalid")
        return copy.deepcopy(dict(observation))

    def observe_recovery(self, plan: Any) -> Mapping[str, Any]:
        self._require_plan(plan)
        result, _credential = self._protocol.exchange(
            "observe_recovery",
            payload={
                "phase_b_plan": self._plan.to_mapping(),
                "phase_b_approval": copy.deepcopy(self._approval),
            },
        )
        observation = result.get("cloud_observation")
        if not isinstance(observation, Mapping):
            _fail("phase_b_owner_response_invalid")
        return copy.deepcopy(dict(observation))

    def observe_terminal(
        self,
        plan: Any,
        *,
        bootstrap_resource: Mapping[str, Any],
        absence_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._require_plan(plan)
        result, _credential = self._protocol.exchange(
            "observe_terminal",
            payload={
                "bootstrap_resource": copy.deepcopy(dict(bootstrap_resource)),
                "absence_receipt": copy.deepcopy(dict(absence_receipt)),
            },
        )
        observation = result.get("cloud_observation")
        if not isinstance(observation, Mapping):
            _fail("phase_b_owner_response_invalid")
        return copy.deepcopy(dict(observation))


_ACTIVE_PHASE_B_PROTOCOL: _FixedPhaseBVMProtocol | None = None
_ACTIVE_PHASE_B_FRAME_EMITTER: Callable[[Mapping[str, Any]], None] | None = None


def build_fixed_phase_b_owner_cloud_sql_boundary(
) -> FixedPhaseBOwnerCloudSQLTransportBoundary:
    """Return the only production Phase-B Cloud boundary (no caller inputs)."""

    protocol = _ACTIVE_PHASE_B_PROTOCOL
    if protocol is None:
        _fail("phase_b_owner_transport_inactive", phase="phase_b_owner_transport")
    from gateway import canonical_writer_phase_b_runtime as runtime

    plan, approval_chain, _journal = runtime.load_fixed_phase_b_authority()
    if not approval_chain:
        _fail("phase_b_approval_chain_invalid", phase="phase_b_owner_transport")
    return FixedPhaseBOwnerCloudSQLTransportBoundary(
        FixedPhaseBOwnerCloudSQLTransportBoundary._CONSTRUCTION_TOKEN,
        protocol,
        plan,
        approval_chain[-1],
    )


def _collect_fixed_phase_b_local_preflight(
    coordinator_input: CoordinatorInput,
) -> Mapping[str, Any]:
    """Collect only the fixed VM half; Cloud truth arrives from the owner."""

    from gateway import canonical_writer_foundation as foundation
    from gateway import canonical_writer_foundation_phase_b as phase_b
    from gateway import canonical_writer_phase_b_runtime as runtime
    from gateway.canonical_writer_planner import load_release_manifest

    if coordinator_input.revision != coordinator_input.writer_activation_plan.revision:
        _fail("phase_b_local_preflight_release_drifted", phase="phase_b_preflight")
    session: Any = None
    try:
        session = runtime._PhaseBAdminSession(runtime._writer_session())
        artifacts = runtime._phase_b_artifacts(coordinator_input.revision)
        _manifest, raw_manifest = load_release_manifest(coordinator_input.revision)
        release_artifacts = dict(sorted({
            phase_b.PREFLIGHT_ARTIFACT_PATH: artifacts["phase_b_preflight"].sha256,
            phase_b.ROLE_ARTIFACT_PATH: artifacts["phase_b_role"].sha256,
        }.items()))
        observed_at = int(time.time())
        unsigned = {
            "schema": PHASE_B_LOCAL_PREFLIGHT_SCHEMA,
            "release_revision": coordinator_input.revision,
            "release_manifest_sha256": _sha256_bytes(raw_manifest),
            "release_artifacts": release_artifacts,
            "release_artifact_set_sha256": _sha256_json(release_artifacts),
            "database": runtime._database_identity(session),
            "foundation": runtime._database_preflight(session, artifacts),
            "credential": foundation.PersistentWriterSecretStore().observe(None),
            "services": runtime._collect_services(
                coordinator_input.revision,
                observed_at,
            ),
            "observed_at_unix": observed_at,
        }
        _phase_b_no_secret_fields(unsigned)
        return {
            **unsigned,
            "local_preflight_sha256": _sha256_json(unsigned),
        }
    except CoordinatorError:
        raise
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CoordinatorError(
            "phase_b_local_preflight_invalid",
            phase="phase_b_preflight",
        ) from exc
    finally:
        if session is not None:
            try:
                session.close()
            except BaseException as exc:
                raise CoordinatorError(
                    "phase_b_local_preflight_session_close_failed",
                    phase="phase_b_preflight",
                ) from exc


def _compose_fixed_phase_b_preflight(
    local: Mapping[str, Any],
    cloud: Mapping[str, Any],
) -> Any:
    from gateway import canonical_writer_foundation_phase_b as phase_b

    expected = frozenset({
        "schema",
        "release_revision",
        "release_manifest_sha256",
        "release_artifacts",
        "release_artifact_set_sha256",
        "database",
        "foundation",
        "credential",
        "services",
        "observed_at_unix",
        "local_preflight_sha256",
    })
    if not isinstance(local, Mapping) or set(local) != expected:
        _fail("phase_b_local_preflight_invalid", phase="phase_b_preflight")
    unsigned_local = {
        key: copy.deepcopy(value)
        for key, value in local.items()
        if key != "local_preflight_sha256"
    }
    if (
        local["schema"] != PHASE_B_LOCAL_PREFLIGHT_SCHEMA
        or local["local_preflight_sha256"] != _sha256_json(unsigned_local)
        or not isinstance(cloud, Mapping)
    ):
        _fail("phase_b_local_preflight_invalid", phase="phase_b_preflight")
    unsigned = {
        "schema": phase_b.PHASE_B_PREFLIGHT_SCHEMA,
        **{
            key: copy.deepcopy(value)
            for key, value in unsigned_local.items()
            if key != "schema"
        },
        "cloud_sql": copy.deepcopy(dict(cloud)),
    }
    try:
        return phase_b.PhaseBPreflight.from_mapping({
            **unsigned,
            "observation_sha256": _sha256_json(unsigned),
        })
    except (TypeError, ValueError) as exc:
        raise CoordinatorError(
            "phase_b_initial_preflight_invalid",
            phase="phase_b_preflight",
        ) from exc


def _phase_b_authority_sidecar(
    *,
    provenance: Mapping[str, Any],
    plan: Any,
    approval: Any,
) -> Mapping[str, Any]:
    if (
        not isinstance(provenance, Mapping)
        or not hasattr(plan, "sha256")
        or not hasattr(approval, "sha256")
    ):
        _fail("phase_b_authority_invalid", phase="phase_b_authority")
    approval_value = approval.to_mapping()
    expected_source = provenance.get("approval_source_sha256")
    if expected_source != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256:
        _fail("phase_b_authority_approval_source_missing", phase="phase_b_authority")
    if (
        approval_value.get("approval_source_sha256") != expected_source
        or approval_value.get("owner_subject_sha256")
        != provenance.get("owner_subject_sha256")
    ):
        _fail("phase_b_authority_approval_drifted", phase="phase_b_authority")
    unsigned = {
        "schema": PHASE_B_AUTHORITY_SCHEMA,
        "phase_b_plan_sha256": plan.sha256,
        "phase_b_approval_sha256": approval.sha256,
        "approval_source_sha256": approval_value["approval_source_sha256"],
        "owner_subject_sha256": approval_value["owner_subject_sha256"],
        "approval_issued_at_unix": approval_value["issued_at_unix"],
        "approval_expires_at_unix": approval_value["expires_at_unix"],
        **{
            key: copy.deepcopy(value)
            for key, value in provenance.items()
            if key not in {"approval_source_sha256", "owner_subject_sha256"}
        },
    }
    return {**unsigned, "authority_sha256": _sha256_json(unsigned)}


def _load_bound_phase_b_authority_context(
    *,
    plan: Any,
    initial_approval: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Recover the exact immutable MPB1 context from a validated sidecar."""

    raw = _stable_root_read(
        PHASE_B_AUTHORITY_RECEIPT_PATH,
        maximum=MAX_COORDINATOR_INPUT_BYTES,
    )
    sidecar = _decode_mapping(raw, code="phase_b_authority_receipt_invalid")
    unsigned = {
        key: copy.deepcopy(value)
        for key, value in sidecar.items()
        if key != "authority_sha256"
    }
    if (
        sidecar.get("schema") != PHASE_B_AUTHORITY_SCHEMA
        or sidecar.get("phase_b_plan_sha256") != getattr(plan, "sha256", None)
        or sidecar.get("phase_b_approval_sha256")
        != initial_approval.get("approval_sha256")
        or sidecar.get("authority_sha256") != _sha256_json(unsigned)
    ):
        _fail("phase_b_authority_receipt_invalid", phase="phase_b_authority")
    excluded = {
        "schema",
        "phase_b_plan_sha256",
        "phase_b_approval_sha256",
        "approval_issued_at_unix",
        "approval_expires_at_unix",
        "authority_sha256",
    }
    context = {
        key: copy.deepcopy(value)
        for key, value in sidecar.items()
        if key not in excluded
    }
    if (
        context.get("approval_source_sha256")
        != initial_approval.get("approval_source_sha256")
        or context.get("owner_subject_sha256")
        != initial_approval.get("owner_subject_sha256")
        or not isinstance(context.get("authority_sources"), Mapping)
        or context.get("authority_sources_sha256")
        != _sha256_json(context["authority_sources"])
    ):
        _fail("phase_b_authority_receipt_invalid", phase="phase_b_authority")
    _phase_b_no_secret_fields(context)
    return context


def _write_phase_b_stage_file(
    directory_fd: int,
    name: str,
    payload: bytes,
) -> None:
    if name not in {
        "plan.json",
        "owner-approval.json",
        "owner-approval-source.json",
        "authority-receipt.json",
    }:
        _fail("phase_b_authority_path_invalid", phase="phase_b_authority")
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
        dir_fd=directory_fd,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short Phase-B authority write")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.fchown(descriptor, 0, 0)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _persist_fixed_phase_b_authority(
    *,
    plan: Any,
    approval: Any,
    approval_source: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> None:
    """Atomically install the signed closure as one no-replace directory."""

    _require_root_linux()
    payloads = {
        "plan.json": _canonical_bytes(plan.to_mapping()) + b"\n",
        "owner-approval.json": _canonical_bytes(approval.to_mapping()) + b"\n",
        "owner-approval-source.json": _canonical_bytes(approval_source) + b"\n",
        "authority-receipt.json": _canonical_bytes(sidecar) + b"\n",
    }
    if any(len(payload) > MAX_COORDINATOR_INPUT_BYTES for payload in payloads.values()):
        _fail("phase_b_authority_oversized", phase="phase_b_authority")
    if PHASE_B_AUTHORITY_ROOT.exists():
        from gateway import canonical_writer_phase_b_runtime as runtime

        loaded_plan, loaded_approval, _journal = runtime.load_fixed_phase_b_authority()
        if (
            loaded_plan.to_mapping() != plan.to_mapping()
            or loaded_approval != approval.to_mapping()
        ):
            _fail("phase_b_authority_generation_conflict", phase="phase_b_authority")
        return
    parent = PHASE_B_AUTHORITY_ROOT.parent
    try:
        parent_item = parent.lstat()
        if (
            not stat.S_ISDIR(parent_item.st_mode)
            or stat.S_ISLNK(parent_item.st_mode)
            or parent_item.st_uid != 0
            or stat.S_IMODE(parent_item.st_mode) & 0o022
        ):
            _fail("phase_b_authority_parent_untrusted", phase="phase_b_authority")
        if PHASE_B_AUTHORITY_STAGE.exists():
            _fail("phase_b_authority_incomplete_install", phase="phase_b_authority")
        os.mkdir(PHASE_B_AUTHORITY_STAGE, 0o700)
        os.chown(PHASE_B_AUTHORITY_STAGE, 0, 0)
        stage_fd = os.open(
            PHASE_B_AUTHORITY_STAGE,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            for name, payload in payloads.items():
                _write_phase_b_stage_file(stage_fd, name, payload)
            os.mkdir("resume-approvals", 0o700, dir_fd=stage_fd)
            resume_fd = os.open(
                "resume-approvals",
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=stage_fd,
            )
            try:
                os.fchmod(resume_fd, 0o700)
                os.fchown(resume_fd, 0, 0)
                lock_fd = os.open(
                    ".lock",
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=resume_fd,
                )
                try:
                    os.fchmod(lock_fd, 0o600)
                    os.fchown(lock_fd, 0, 0)
                    os.fsync(lock_fd)
                finally:
                    os.close(lock_fd)
                os.fsync(resume_fd)
            finally:
                os.close(resume_fd)
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        os.chmod(PHASE_B_AUTHORITY_STAGE, 0o700)
        os.rename(PHASE_B_AUTHORITY_STAGE, PHASE_B_AUTHORITY_ROOT)
        parent_fd = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except CoordinatorError:
        raise
    except OSError as exc:
        raise CoordinatorError(
            "phase_b_authority_publication_failed",
            phase="phase_b_authority",
        ) from exc


def _require_phase_b_pristine_live_boundary() -> None:
    if (
        not _services_are_exactly_stopped_and_disabled()
        or os.path.lexists(DISCORD_TOKEN_PATH)
        or os.path.lexists(DISCORD_TOKEN_STAGE_PATH)
        or os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH)
        or os.path.lexists(OBSOLETE_CREDENTIAL_PREPARE_APPROVAL_PATH)
        or os.path.lexists(OBSOLETE_BOOTSTRAP_CREDENTIAL_PATH)
        or os.path.lexists(OBSOLETE_COORDINATOR_PROCESS_JOURNAL_PATH)
    ):
        _fail(
            "phase_b_requires_pristine_stopped_live_boundary",
            phase="phase_b_preflight",
        )


def preflight_phase_b_apply() -> Mapping[str, Any]:
    """Issue a fresh no-secret gate for initial apply or same-plan resume."""

    _require_root_linux()
    _harden_secret_process()
    coordinator_input = load_coordinator_input()
    _require_phase_b_pristine_live_boundary()
    provenance = dict(_phase_b_authority_provenance(coordinator_input))
    owner_subject_sha256 = _digest(
        provenance.get("owner_subject_sha256"),
        "phase_b_preflight_owner_invalid",
    )
    approval_source_sha256 = _digest(
        provenance.get("approval_source_sha256"),
        "phase_b_preflight_owner_invalid",
    )
    if approval_source_sha256 != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256:
        _fail("phase_b_preflight_owner_invalid", phase="phase_b_preflight")
    now_unix = int(time.time())
    authority_present = PHASE_B_AUTHORITY_ROOT.exists()
    inspection: Mapping[str, Any] | None = None
    if authority_present:
        from gateway import canonical_writer_phase_b_runtime as runtime

        try:
            inspection = runtime.inspect_fixed_phase_b_incomplete_head()
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise CoordinatorError(
                "phase_b_existing_authority_invalid",
                phase="phase_b_preflight",
            ) from exc
        expected_inspection_sha = inspection.get("inspection_sha256")
        unsigned_inspection = {
            key: copy.deepcopy(value)
            for key, value in inspection.items()
            if key != "inspection_sha256"
        }
        if (
            not isinstance(expected_inspection_sha, str)
            or expected_inspection_sha != _sha256_json(unsigned_inspection)
            or inspection.get("owner_subject_sha256")
            != owner_subject_sha256
            or inspection.get("approval_source_sha256")
            != approval_source_sha256
        ):
            _fail("phase_b_existing_authority_invalid", phase="phase_b_preflight")
        gate_state = "same_plan_resume_or_replay"
        if inspection.get("terminal") is True:
            expires_at = ((now_unix // 300) + 12) * 300
        elif inspection.get("fresh_head") is True:
            expires_at = min(
                ((now_unix // 300) + 12) * 300,
                int(inspection["approval_expires_at_unix"]),
            )
        else:
            expires_at = ((now_unix // 300) + 12) * 300
    else:
        # This is readiness only.  Mutation authority is created later by the
        # signed, session-bound MPB1 ``authority_approve`` exchange.
        gate_state = "initial_apply_ready"
        expires_at = now_unix + 900
    if expires_at <= now_unix:
        _fail("phase_b_apply_gate_expired", phase="phase_b_preflight")
    unsigned = {
        "schema": PHASE_B_APPLY_GATE_SCHEMA,
        "ok": True,
        "state": gate_state,
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "owner_subject_sha256": owner_subject_sha256,
        "approval_source_sha256": approval_source_sha256,
        "authority_present": authority_present,
        "phase_b_plan_sha256": (
            None if inspection is None else inspection["plan_sha256"]
        ),
        "phase_b_approval_sha256": (
            None if inspection is None else inspection["approval_sha256"]
        ),
        "phase_b_approval_sequence": (
            None if inspection is None else inspection["approval_sequence"]
        ),
        "phase_b_incomplete_state": (
            None if inspection is None else inspection["incomplete_state"]
        ),
        "phase_b_inspection_sha256": (
            None if inspection is None else inspection["inspection_sha256"]
        ),
        "phase_b_terminal": (
            False if inspection is None else inspection["terminal"]
        ),
        "phase_b_requires_reapproval": (
            False if inspection is None else inspection["requires_reapproval"]
        ),
        "issued_at_unix": now_unix,
        "expires_at_unix": expires_at,
    }
    _phase_b_no_secret_fields(unsigned)
    return {**unsigned, "gate_sha256": _sha256_json(unsigned)}


def apply_fixed_phase_b_foundation(
    *,
    frame_emitter: Callable[[Mapping[str, Any]], None],
) -> Mapping[str, Any]:
    """Author, execute and publish one fixed stopped-service Phase-B generation."""

    global _ACTIVE_PHASE_B_PROTOCOL, _ACTIVE_PHASE_B_FRAME_EMITTER
    _harden_secret_process()
    apply_gate = preflight_phase_b_apply()
    coordinator_input = load_coordinator_input()
    if apply_gate["coordinator_input_sha256"] != coordinator_input.sha256:
        _fail("phase_b_apply_gate_drifted", phase="phase_b_preflight")
    _require_phase_b_pristine_live_boundary()
    provenance = dict(_phase_b_authority_provenance(coordinator_input))
    if (
        provenance.get("owner_subject_sha256")
        != apply_gate.get("owner_subject_sha256")
        or provenance.get("approval_source_sha256")
        != apply_gate.get("approval_source_sha256")
    ):
        _fail("phase_b_apply_gate_drifted", phase="phase_b_preflight")

    from gateway import canonical_writer_foundation_phase_b as phase_b
    from gateway import canonical_writer_phase_b_runtime as runtime

    protocol: _FixedPhaseBVMProtocol | None = None
    _ACTIVE_PHASE_B_FRAME_EMITTER = frame_emitter
    try:
        terminal: Mapping[str, Any] | None = None
        if not PHASE_B_AUTHORITY_ROOT.exists():
            protocol = _FixedPhaseBVMProtocol(
                provenance=provenance,
                phase_b_plan_sha256=None,
                phase_b_approval_sha256=None,
                approval_expires_at_unix=int(apply_gate["expires_at_unix"]),
            )
            local = _collect_fixed_phase_b_local_preflight(coordinator_input)
            observed, credential = protocol.exchange(
                "authority_observe_initial",
                payload={"local_preflight": local},
            )
            _zeroize(credential)
            cloud = observed.get("cloud_observation")
            owner_authority = protocol.bind_owner_resume_authority(
                observed.get("owner_resume_authority")
            )
            provenance = dict(protocol.authority_context)
            if not isinstance(cloud, Mapping):
                _fail("phase_b_owner_response_invalid", phase="phase_b_authority")
            preflight = _compose_fixed_phase_b_preflight(local, cloud)
            plan = phase_b.build_phase_b_plan(
                preflight,
                owner_subject_sha256=provenance["owner_subject_sha256"],
                owner_resume_public_key_ed25519_hex=owner_authority[
                    "public_key_ed25519_hex"
                ],
                owner_resume_public_key_file_sha256=owner_authority[
                    "public_key_file_sha256"
                ],
            )
            protocol.bind_plan(plan.sha256)
            approved_result, credential = protocol.exchange(
                "authority_approve",
                payload={"phase_b_plan": plan.to_mapping()},
            )
            _zeroize(credential)
            approval_mapping = approved_result.get("phase_b_approval")
            approval_source_mapping = approved_result.get(
                "phase_b_approval_source"
            )
            if (
                not isinstance(approval_mapping, Mapping)
                or not isinstance(approval_source_mapping, Mapping)
            ):
                _fail("phase_b_owner_response_invalid", phase="phase_b_authority")
            try:
                approval = phase_b.PhaseBApproval.from_mapping(
                    approval_mapping,
                    plan=plan,
                    now_unix=int(time.time()),
                )
                approval_source = phase_b.validate_phase_b_source_authentication(
                    approval_source_mapping,
                    plan=plan,
                    approval=approval,
                )
            except (phase_b.PhaseBError, TypeError, ValueError) as exc:
                raise CoordinatorError(
                    "phase_b_owner_approval_invalid",
                    phase="phase_b_authority",
                ) from exc
            if (
                approval.value["approval_source_sha256"]
                != provenance["approval_source_sha256"]
                or approval.value["owner_subject_sha256"]
                != provenance["owner_subject_sha256"]
                or approval.value["expires_at_unix"]
                > apply_gate["expires_at_unix"]
            ):
                _fail("phase_b_owner_approval_invalid", phase="phase_b_authority")
            protocol.bind_approval(
                approval_sha256=approval.sha256,
                approval_expires_at_unix=approval.value["expires_at_unix"],
            )
            sidecar = _phase_b_authority_sidecar(
                provenance=provenance,
                plan=plan,
                approval=approval,
            )
            _persist_fixed_phase_b_authority(
                plan=plan,
                approval=approval,
                approval_source=approval_source,
                sidecar=sidecar,
            )
        else:
            plan, approval_values, _journal = (
                runtime.load_fixed_phase_b_authority()
            )
            if not approval_values:
                _fail("phase_b_approval_chain_invalid", phase="phase_b_authority")
            provenance = dict(
                _load_bound_phase_b_authority_context(
                    plan=plan,
                    initial_approval=approval_values[0],
                )
            )
            inspection = runtime.inspect_fixed_phase_b_incomplete_head()
            if (
                inspection.get("plan_sha256") != plan.sha256
                or inspection.get("plan_sha256")
                != apply_gate.get("phase_b_plan_sha256")
                or inspection.get("approval_sha256")
                != apply_gate.get("phase_b_approval_sha256")
                or inspection.get("approval_sequence")
                != apply_gate.get("phase_b_approval_sequence")
                or inspection.get("incomplete_state")
                != apply_gate.get("phase_b_incomplete_state")
                or inspection.get("terminal")
                is not apply_gate.get("phase_b_terminal")
                or inspection.get("requires_reapproval")
                is not apply_gate.get("phase_b_requires_reapproval")
            ):
                _fail("phase_b_apply_gate_drifted", phase="phase_b_preflight")
            if inspection["terminal"] is True:
                foundation = runtime.load_fixed_completed_phase_b_foundation()
                terminal = copy.deepcopy(dict(foundation["terminal_receipt"]))
                approval = phase_b.PhaseBApproval.from_mapping(
                    foundation["approval"],
                    plan=plan,
                    now_unix=int(terminal["terminal_at_unix"]),
                )
            else:
                head_mapping = approval_values[-1]
                head = phase_b.PhaseBApproval.from_mapping(
                    head_mapping,
                    plan=plan,
                    now_unix=int(head_mapping["issued_at_unix"]),
                )
                protocol = _FixedPhaseBVMProtocol(
                    provenance=provenance,
                    phase_b_plan_sha256=plan.sha256,
                    phase_b_approval_sha256=head.sha256,
                    approval_expires_at_unix=(
                        int(apply_gate["expires_at_unix"])
                        if inspection["requires_reapproval"] is True
                        else int(head.value["expires_at_unix"])
                    ),
                )
                if inspection["requires_reapproval"] is True:
                    current_chain = tuple(approval_values)
                    current_inspection = copy.deepcopy(dict(inspection))
                    approval = None
                    # Exactly two owner frames are sufficient and permitted:
                    # one may complete an expired source-only crash residue as
                    # non-authorizing history; the next must be a fresh head.
                    for resume_round in range(2):
                        current_head_mapping = current_chain[-1]
                        current_head = phase_b.PhaseBApproval.from_mapping(
                            current_head_mapping,
                            plan=plan,
                            now_unix=int(
                                current_head_mapping["issued_at_unix"]
                            ),
                        )
                        resumed_result, credential = protocol.exchange(
                            "authority_resume_approve",
                            payload={
                                "phase_b_plan": plan.to_mapping(),
                                "phase_b_approval_chain": [
                                    copy.deepcopy(dict(value))
                                    for value in current_chain
                                ],
                                "phase_b_incomplete_head": copy.deepcopy(
                                    current_inspection
                                ),
                            },
                        )
                        _zeroize(credential)
                        resumed_approval = resumed_result.get(
                            "phase_b_approval"
                        )
                        resumed_source = resumed_result.get(
                            "phase_b_approval_source"
                        )
                        if (
                            not isinstance(resumed_approval, Mapping)
                            or not isinstance(resumed_source, Mapping)
                        ):
                            _fail(
                                "phase_b_owner_response_invalid",
                                phase="phase_b_authority",
                            )
                        install_receipt = (
                            runtime.install_fixed_phase_b_resume_approval(
                                resumed_approval,
                                resumed_source,
                                expected_previous_approval_sha256=(
                                    current_head.sha256
                                ),
                            )
                        )
                        reloaded = runtime.load_fixed_phase_b_approval_chain(
                            require_fresh_head=False
                        )
                        if (
                            len(reloaded) != len(current_chain) + 1
                            or reloaded[-1] != resumed_approval
                            or not isinstance(install_receipt, Mapping)
                            or install_receipt.get("schema")
                            != (
                                "muncho-canonical-writer-phase-b-"
                                "resume-approval-install.v1"
                            )
                            or install_receipt.get("plan_sha256")
                            != plan.sha256
                            or install_receipt.get("previous_approval_sha256")
                            != current_head.sha256
                            or install_receipt.get("approval_sha256")
                            != resumed_approval.get("approval_sha256")
                        ):
                            _fail(
                                "phase_b_resume_approval_install_invalid",
                                phase="phase_b_authority",
                            )
                        installed = phase_b.PhaseBApproval.from_mapping(
                            reloaded[-1],
                            plan=plan,
                            now_unix=int(reloaded[-1]["issued_at_unix"]),
                        )
                        if install_receipt.get("mutation_authorized") is True:
                            if (
                                install_receipt.get("approval_fresh") is not True
                                or install_receipt.get(
                                    "requires_fresh_successor"
                                )
                                is not False
                            ):
                                _fail(
                                    "phase_b_resume_approval_install_invalid",
                                    phase="phase_b_authority",
                                )
                            approval = phase_b.PhaseBApproval.from_mapping(
                                reloaded[-1],
                                plan=plan,
                                now_unix=int(time.time()),
                            )
                            protocol.bind_resume_approval(
                                expected_previous_approval_sha256=(
                                    current_head.sha256
                                ),
                                approval_sha256=approval.sha256,
                                approval_expires_at_unix=approval.value[
                                    "expires_at_unix"
                                ],
                            )
                            break
                        if (
                            resume_round != 0
                            or install_receipt.get("approval_fresh") is not False
                            or install_receipt.get("mutation_authorized") is not False
                            or install_receipt.get(
                                "requires_fresh_successor"
                            )
                            is not True
                            or install_receipt.get(
                                "completed_trailing_source_only_residue"
                            )
                            is not True
                        ):
                            _fail(
                                "phase_b_resume_approval_install_invalid",
                                phase="phase_b_authority",
                            )
                        protocol.bind_historical_resume_approval(
                            expected_previous_approval_sha256=(
                                current_head.sha256
                            ),
                            approval_sha256=installed.sha256,
                            approval_expires_at_unix=installed.value[
                                "expires_at_unix"
                            ],
                        )
                        current_chain = tuple(reloaded)
                        current_inspection = copy.deepcopy(dict(
                            runtime.inspect_fixed_phase_b_incomplete_head()
                        ))
                        if (
                            current_inspection.get("plan_sha256")
                            != plan.sha256
                            or current_inspection.get("approval_sha256")
                            != installed.sha256
                            or current_inspection.get("approval_sequence")
                            != installed.sequence
                            or current_inspection.get(
                                "pending_source_sequence"
                            )
                            is not None
                            or current_inspection.get(
                                "pending_source_authentication_sha256"
                            )
                            is not None
                            or current_inspection.get("terminal") is not False
                            or current_inspection.get("resume_eligible") is not True
                            or current_inspection.get("requires_reapproval")
                            is not True
                            or current_inspection.get("mutation_authorized")
                            is not False
                        ):
                            _fail(
                                "phase_b_resume_inspection_invalid",
                                phase="phase_b_authority",
                            )
                    if approval is None:
                        _fail(
                            "phase_b_fresh_resume_approval_missing",
                            phase="phase_b_authority",
                        )
                else:
                    approval = phase_b.PhaseBApproval.from_mapping(
                        head_mapping,
                        plan=plan,
                        now_unix=int(time.time()),
                    )
                _ACTIVE_PHASE_B_PROTOCOL = protocol
                terminal = runtime.execute_fixed_phase_b()
        if terminal is None:
            if protocol is None:
                _fail(
                    "phase_b_owner_protocol_missing",
                    phase="phase_b_owner_transport",
                )
            _ACTIVE_PHASE_B_PROTOCOL = protocol
            terminal = runtime.execute_fixed_phase_b()
        readiness = runtime.publish_fixed_phase_b_readiness()
        readiness_mapping = readiness.to_mapping()
        unsigned = {
            "schema": PHASE_B_APPLY_RECEIPT_SCHEMA,
            "ok": True,
            "state": "terminal_ready",
            "release_sha": coordinator_input.revision,
            "coordinator_input_sha256": coordinator_input.sha256,
            "phase_b_plan_sha256": plan.sha256,
            "phase_b_approval_sha256": approval.sha256,
            "phase_b_terminal_receipt_sha256": terminal["receipt_sha256"],
            "phase_b_readiness_receipt_sha256": readiness_mapping[
                "receipt_sha256"
            ],
            "safe_to_start": True,
            "completed_at_unix": int(time.time()),
        }
        return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
    finally:
        _ACTIVE_PHASE_B_PROTOCOL = None
        _ACTIVE_PHASE_B_FRAME_EMITTER = None


def _load_fixed_phase_b_live_authority(
    coordinator_input: CoordinatorInput,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Load the exact terminal Phase-B generation and latest live anchor."""

    if not isinstance(coordinator_input, CoordinatorInput):
        raise TypeError("CoordinatorInput is required")
    from gateway import canonical_writer_phase_b_runtime as runtime

    try:
        foundation = runtime.load_fixed_completed_phase_b_foundation()
        anchor = runtime.load_fixed_phase_b_readiness_anchor()
    except CoordinatorError:
        raise
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise CoordinatorError(
            "phase_b_live_authority_invalid",
            phase="phase_b_live_preflight",
        ) from exc
    if (
        not isinstance(foundation, Mapping)
        or set(foundation)
        != {
            "plan",
            "approval",
            "generation",
            "terminal_receipt",
            "terminal_observation",
        }
        or not isinstance(anchor, Mapping)
    ):
        _fail("phase_b_live_authority_invalid", phase="phase_b_live_preflight")
    plan = foundation["plan"]
    approval = foundation["approval"]
    generation = foundation["generation"]
    terminal = foundation["terminal_receipt"]
    if not all(
        isinstance(item, Mapping)
        for item in (plan, approval, generation, terminal)
    ):
        _fail("phase_b_live_authority_invalid", phase="phase_b_live_preflight")
    expected = {
        "phase_b_release_revision": coordinator_input.revision,
        "phase_b_plan_sha256": plan.get("plan_sha256"),
        "phase_b_approval_sha256": approval.get("approval_sha256"),
        "phase_b_terminal_receipt_sha256": terminal.get("receipt_sha256"),
        "phase_b_foundation_generation_sha256": generation.get(
            "generation_sha256"
        ),
    }
    if (
        plan.get("release_revision") != coordinator_input.revision
        or approval.get("plan_sha256") != plan.get("plan_sha256")
        or approval.get("approval_source_sha256")
        != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
        or terminal.get("plan_sha256") != plan.get("plan_sha256")
        or any(anchor.get(name) != value for name, value in expected.items())
    ):
        _fail("phase_b_live_authority_invalid", phase="phase_b_live_preflight")
    for name in (
        "owner_subject_sha256",
        "approval_source_sha256",
        "approval_sha256",
    ):
        _digest(approval.get(name), "phase_b_live_authority_invalid")
    runtime.validate_fixed_phase_b_readiness_descendant(anchor)
    return copy.deepcopy(dict(foundation)), copy.deepcopy(dict(anchor))


def preflight_phase_b_live_run() -> Mapping[str, Any]:
    """Return the fresh read-only gate for Discord install and target run."""

    _require_root_linux()
    _harden_secret_process()
    coordinator_input = load_coordinator_input()
    if (
        not _services_are_exactly_stopped_and_disabled()
        or os.path.lexists(OBSOLETE_CREDENTIAL_PREPARE_APPROVAL_PATH)
        or os.path.lexists(OBSOLETE_BOOTSTRAP_CREDENTIAL_PATH)
        or os.path.lexists(OBSOLETE_COORDINATOR_PROCESS_JOURNAL_PATH)
        or os.path.lexists(DISCORD_TOKEN_PATH)
        or os.path.lexists(DISCORD_TOKEN_STAGE_PATH)
        or os.path.lexists(DISCORD_TOKEN_INSTALL_RECEIPT_PATH)
    ):
        _fail(
            "phase_b_live_boundary_not_pristine",
            phase="phase_b_live_preflight",
        )
    foundation, anchor = _load_fixed_phase_b_live_authority(coordinator_input)
    approval = foundation["approval"]
    now_unix = int(time.time())
    unsigned = {
        "schema": PHASE_B_LIVE_GATE_SCHEMA,
        "ok": True,
        "state": "phase_b_terminal_ready",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "owner_subject_sha256": approval["owner_subject_sha256"],
        "approval_source_sha256": approval["approval_source_sha256"],
        "phase_b_readiness_anchor": anchor,
        "phase_b_readiness_anchor_sha256": _sha256_json(anchor),
        "issued_at_unix": now_unix,
        "expires_at_unix": now_unix + 300,
    }
    return {**unsigned, "gate_sha256": _sha256_json(unsigned)}


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
        if (
            self.value["coordinator_input_sha256"] != coordinator_input.sha256
            or self.value["release_sha"] != coordinator_input.revision
            or self.value["approval_source_sha256"]
            != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
            or type(now_unix) is not int
            or not self.value["approved_at_unix"]
            <= now_unix
            <= self.value["expires_at_unix"]
        ):
            _fail("discord_token_install_approval_not_fresh_or_bound")


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
    if os.path.lexists(OBSOLETE_COORDINATOR_PROCESS_JOURNAL_PATH):
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
        if os.path.lexists(OBSOLETE_COORDINATOR_PROCESS_JOURNAL_PATH):
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
    _validated_base_e2e_fixture(value)
    from gateway.canonical_writer_planner import load_release_manifest

    try:
        _manifest, manifest_raw = load_release_manifest(value.revision)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CoordinatorError("coordinator_release_manifest_invalid") from exc
    if (
        _sha256_bytes(manifest_raw)
        != value.writer_activation_plan.digests.release_manifest_file_sha256
    ):
        _fail("coordinator_release_manifest_digest_drifted")


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
    if (
        approval.value["coordinator_input_sha256"] != coordinator_input.sha256
        or approval.value["release_sha"] != coordinator_input.revision
        or approval.value["approval_source_sha256"]
        != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
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


def install_discord_token(
    *,
    gate_emitter: Any,
    frame_reader: Any = OpaqueDiscordTokenFrame.read,
) -> Mapping[str, Any]:
    """Gate, read, and no-replace install the isolated Discord token."""

    _harden_secret_process()
    coordinator_input = load_coordinator_input()
    _consume_terminal_discord_retirement(coordinator_input)
    if os.path.lexists(OBSOLETE_COORDINATOR_PROCESS_JOURNAL_PATH):
        _fail("obsolete_coordinator_process_journal_requires_reconciliation")
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
            if os.path.lexists(OBSOLETE_COORDINATOR_PROCESS_JOURNAL_PATH):
                _fail("obsolete_coordinator_process_journal_requires_reconciliation")
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


def _read_exact(fd: int, size: int) -> bytearray:
    if type(fd) is not int or fd < 0 or type(size) is not int or size < 0:
        _fail("secret_frame_bounds_invalid", phase="credential_read")
    result = bytearray()
    while len(result) < size:
        try:
            chunk = os.read(fd, size - len(result))
        except OSError as exc:
            _zeroize(result)
            raise CoordinatorError(
                "secret_frame_read_failed", phase="credential_read"
            ) from exc
        if not chunk:
            _zeroize(result)
            _fail("secret_frame_truncated", phase="credential_read")
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


def _services_are_exactly_stopped_and_disabled() -> bool:
    states = {
        unit: collect_service_state(unit)
        for unit in (EDGE_UNIT_NAME, WRITER_UNIT_NAME, GATEWAY_UNIT_NAME)
    }
    checks = evaluate_service_states(states, phase="stopped")
    return bool(checks) and all(checks.values())


def _validated_activation_release_file(
    coordinator_input: CoordinatorInput,
    relative_path: Path,
    *,
    maximum_bytes: int,
) -> tuple[Path, bytes, str]:
    """Read one manifest-bound file without constructing a premature plan."""

    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.as_posix() != str(relative_path)
        or not 0 < maximum_bytes <= MAX_COORDINATOR_INPUT_BYTES
    ):
        _fail("coordinator_release_file_request_invalid", phase="process_identity")
    from gateway.canonical_writer_planner import load_release_manifest

    try:
        manifest, manifest_raw = load_release_manifest(coordinator_input.revision)
        release = _release_binding(coordinator_input.writer_activation_plan)
        mapping = manifest.to_mapping()
        if (
            _sha256_bytes(manifest_raw) != release["manifest_file_sha256"]
            or mapping.get("artifact_root") != release["artifact_root"]
            or mapping.get("interpreter") != release["interpreter"]
            or mapping.get("artifact_sha256") != release["artifact_sha256"]
        ):
            _fail("coordinator_release_manifest_digest_drifted")
        matches = [
            item
            for item in mapping.get("entries", [])
            if isinstance(item, Mapping)
            and item.get("path") == relative_path.as_posix()
        ]
        if len(matches) != 1 or matches[0].get("kind") != "file":
            _fail("coordinator_release_file_not_manifest_bound")
        expected_sha256 = _digest(
            matches[0].get("sha256"),
            "coordinator_release_file_digest_invalid",
        )
        path = Path(release["artifact_root"]) / relative_path
        raw = _stable_root_read(path, maximum=maximum_bytes)
        if _sha256_bytes(raw) != expected_sha256:
            _fail("coordinator_release_file_digest_drifted")
        return path, raw, expected_sha256
    except CoordinatorError:
        raise
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CoordinatorError(
            "coordinator_release_file_not_manifest_bound",
            phase="process_identity",
        ) from exc


def _sealed_coordinator_module_identity(
    coordinator_input: CoordinatorInput,
) -> tuple[str, str]:
    origin, module_sha256 = module_file_identity(__file__)
    release = _release_binding(coordinator_input.writer_activation_plan)
    artifact_root = Path(release["artifact_root"])
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
        sealed_path, _sealed_raw, sealed_sha256 = _validated_activation_release_file(
            coordinator_input,
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
    "preflight-phase-b-apply",
    "preflight-phase-b-live-run",
    "phase-b-apply",
    "run",
    "install-discord-token",
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
        str(_release_binding(coordinator_input.writer_activation_plan)["interpreter"]),
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
    release = _release_binding(coordinator_input.writer_activation_plan)
    expected_path = Path(release["interpreter"])
    artifact_root = Path(release["artifact_root"])
    try:
        relative = expected_path.relative_to(artifact_root)
        sealed_path, sealed_raw, sealed_sha256 = _validated_activation_release_file(
            coordinator_input,
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


def discord_retirement_gate(
    *,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    _require_root_linux()
    coordinator_input = load_coordinator_input()
    if os.path.lexists(OBSOLETE_COORDINATOR_PROCESS_JOURNAL_PATH):
        _fail("obsolete_coordinator_process_journal_requires_reconciliation")
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
        # The target session-bound coordinator owns no privileged process
        # lease.  Any artifact at the retired legacy path blocks this gate.
        "process_lease_absent": True,
        "services_stopped_proven": True,
        "frame_schema": DISCORD_RETIREMENT_ACK_FRAME_SCHEMA,
        "expires_at_unix": current_time + DISCORD_RETIREMENT_GATE_MAX_SECONDS,
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
        <= DISCORD_RETIREMENT_GATE_MAX_SECONDS
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
    """Mechanically stop and retire only the installed Discord lease."""

    _require_root_linux()
    mechanically_stop_full_canary_services()
    if not _services_are_exactly_stopped_and_disabled():
        _fail("discord_token_recovery_services_not_stopped")
    coordinator_input = load_coordinator_input()
    gate = discord_retirement_gate()
    gate_emitter(gate)
    ack_reader(gate=gate)
    with _lifecycle_lock():
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


def run_session_bound_full_canary(
    *,
    frame_emitter: Callable[[Mapping[str, Any]], None],
    final_approval_frame_reader: Callable[[], Mapping[str, Any]] = (
        _read_final_owner_approval_frame
    ),
    driver_factory: Callable[..., HonestFullCanaryDriver] = HonestFullCanaryDriver,
) -> Mapping[str, Any]:
    """Run the admin-free target canary from terminal Phase-B readiness.

    The only ephemeral secret created here is the loopback API session key
    owned by ``prepare_session_bound_plan``.  The writer configuration is
    staged unchanged, only the fixture receives the key digest, and the raw
    key is consumed once by the live driver.  Plan meaning, effort, and task
    decisions remain model-authored.
    """

    _require_root_linux()
    _harden_secret_process()
    coordinator_input = load_coordinator_input()
    if not _services_are_exactly_stopped_and_disabled():
        _fail("full_canary_services_not_initially_stopped", phase="live_preflight")
    foundation, readiness_anchor = _load_fixed_phase_b_live_authority(
        coordinator_input
    )
    phase_b_approval = foundation.get("approval")
    if (
        not isinstance(phase_b_approval, Mapping)
        or phase_b_approval.get("approval_source_sha256")
        != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
    ):
        _fail("phase_b_owner_lineage_not_pinned", phase="live_preflight")
    owner_subject_sha256 = _digest(
        phase_b_approval.get("owner_subject_sha256"),
        "phase_b_owner_subject_invalid",
    )
    phase_b_approval_sha256 = _digest(
        phase_b_approval.get("approval_sha256"),
        "phase_b_owner_approval_invalid",
    )
    base_fixture = _validated_base_e2e_fixture(coordinator_input)
    fixture_expiry = int(base_fixture["valid_until_unix_ms"]) // 1000
    writer_config = copy.deepcopy(dict(coordinator_input.value["writer_config"]))

    discord_install_receipt, discord_token_identity, discord_receipt_snapshot = (
        load_discord_token_install_receipt(coordinator_input)
    )
    prior_writer = _capture_writer_snapshot(coordinator_input.identities.writer_gid)
    writer_publication: _WriterPublication | None = None
    staged_plan_publication: _RootPublication | None = None
    runtime_plan_publication: _RootPublication | None = None
    prepared: Any = None
    plan: FullCanaryPlan | None = None
    live_result: Mapping[str, Any] | None = None
    token_retired = False
    primary: BaseException | None = None
    cleanup_errors: list[BaseException] = []

    def publish_staged_writer(path: Path, payload: bytes, **kwargs: Any) -> None:
        nonlocal writer_publication
        _atomic_stage_writer_config(path, payload, **kwargs)
        raw, item = _read_staged_writer_config(
            DEFAULT_WRITER_CONFIG_SOURCE,
            mode=0o440,
            uid=0,
            gid=coordinator_input.identities.writer_gid,
        )
        if raw != payload:
            _fail("staged_writer_publication_drifted")
        writer_publication = _WriterPublication(
            writer_gid=coordinator_input.identities.writer_gid,
            before=prior_writer,
            after=_WriterSnapshot(raw=raw, item=item),
        )

    def stale_writer_clearance() -> Mapping[str, Any]:
        raw, _item = _read_staged_writer_config(
            DEFAULT_WRITER_CONFIG_SOURCE,
            mode=0o440,
            uid=0,
            gid=coordinator_input.identities.writer_gid,
        )
        if not _services_are_exactly_stopped_and_disabled():
            _fail("staged_writer_config_unreconciled")
        return {
            "schema": "muncho-full-canary-stale-artifact-clearance.v1",
            "staged_path": str(DEFAULT_WRITER_CONFIG_SOURCE),
            "stale_sha256": _sha256_bytes(raw),
            "services_stopped": True,
        }

    def build_plan() -> FullCanaryPlan:
        nonlocal plan, staged_plan_publication
        writer_raw, _writer_item = _read_staged_writer_config(
            DEFAULT_WRITER_CONFIG_SOURCE,
            mode=0o440,
            uid=0,
            gid=coordinator_input.identities.writer_gid,
        )
        fixture_raw, _fixture_item = _read_staged_writer_config(
            DEFAULT_E2E_FIXTURE,
            mode=0o440,
            uid=0,
            gid=coordinator_input.identities.gateway_gid,
        )
        writer_artifact = ExactArtifact(
            source_path=DEFAULT_WRITER_CONFIG_SOURCE,
            target_path=DEFAULT_WRITER_CONFIG,
            sha256=_sha256_bytes(writer_raw),
            mode=0o440,
            uid=0,
            gid=coordinator_input.identities.writer_gid,
        )
        fixture_artifact = ExactArtifact(
            source_path=DEFAULT_E2E_FIXTURE,
            target_path=DEFAULT_E2E_FIXTURE,
            sha256=_sha256_bytes(fixture_raw),
            mode=0o440,
            uid=0,
            gid=coordinator_input.identities.gateway_gid,
        )
        plan = build_full_canary_plan(
            writer_activation_plan=coordinator_input.writer_activation_plan,
            writer_activation_receipt=coordinator_input.value[
                "writer_activation_receipt"
            ],
            writer_activation_receipt_file_sha256=coordinator_input.value[
                "writer_activation_receipt_file_sha256"
            ],
            phase_b_readiness_anchor=readiness_anchor,
            identities=coordinator_input.identities,
            writer_config=writer_artifact,
            gateway_config=coordinator_input.artifacts["gateway_config"],
            edge_config=coordinator_input.artifacts["edge_config"],
            e2e_fixture=fixture_artifact,
            host_identity_receipt=coordinator_input.artifacts[
                "host_identity_receipt"
            ],
        )
        validate_dedicated_canary_host(plan)
        if os.path.lexists(DEFAULT_STAGED_PLAN_PATH):
            _fail("staged_plan_target_not_fresh")
        staged_plan_publication = _publish_root_payload(
            DEFAULT_STAGED_PLAN_PATH,
            _canonical_bytes(plan.to_mapping()),
            expected_previous_sha256=None,
        )
        return plan

    def approve(final_plan: FullCanaryPlan) -> FullCanaryOwnerApproval:
        nonlocal runtime_plan_publication
        now_unix = int(time.time())
        hard_deadline = min(now_unix + 900, fixture_expiry)
        if hard_deadline - now_unix < 30:
            _fail("owner_approval_wait_window_exhausted")
        request_unsigned = {
            "schema": SESSION_BOUND_APPROVAL_REQUEST_SCHEMA,
            "ok": True,
            "state": "awaiting_session_bound_owner_approval",
            "release_sha": coordinator_input.revision,
            "coordinator_input_sha256": coordinator_input.sha256,
            "full_canary_plan_sha256": final_plan.sha256,
            "staged_plan_path": str(DEFAULT_STAGED_PLAN_PATH),
            "staged_plan_file_sha256": _sha256_bytes(
                _canonical_bytes(final_plan.to_mapping())
            ),
            "fixture_sha256": final_plan.artifacts["e2e_fixture"].sha256,
            "phase_b_readiness_anchor_sha256": _sha256_json(readiness_anchor),
            "phase_b_approval_sha256": phase_b_approval_sha256,
            "owner_subject_sha256": owner_subject_sha256,
            "approval_source_sha256": PHASE_B_PINNED_APPROVAL_SOURCE_SHA256,
            "requested_at_unix": now_unix,
            "owner_input_cutoff_unix": hard_deadline - 5,
            "approval_deadline_unix": hard_deadline,
            "approval_path": None,
            "final_approval_frame_schema": FINAL_APPROVAL_FRAME_SCHEMA,
        }
        request = {
            **request_unsigned,
            "request_sha256": _sha256_json(request_unsigned),
        }
        frame_emitter(request)
        try:
            approval = FullCanaryOwnerApproval.from_mapping(
                final_approval_frame_reader()
            )
        except (TypeError, ValueError) as exc:
            raise CoordinatorError(
                "final_owner_approval_contract_invalid",
                phase="final_approval",
            ) from exc
        approved_at = approval.value["approved_at_unix"]
        current = int(time.time())
        approval.require(plan_sha256=final_plan.sha256, now_unix=current)
        if (
            approval.value["owner_subject_sha256"] != owner_subject_sha256
            or approval.value["approval_source_sha256"]
            != PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
            or not now_unix <= approved_at <= request["owner_input_cutoff_unix"]
            or approval.value["expires_at_unix"] > hard_deadline
        ):
            _fail("final_owner_approval_not_bound", phase="final_approval")
        if os.path.lexists(DEFAULT_PLAN_PATH):
            _fail("runtime_plan_target_not_fresh")
        runtime_plan_publication = _publish_root_payload(
            DEFAULT_PLAN_PATH,
            _canonical_bytes(final_plan.to_mapping()),
            expected_previous_sha256=None,
        )
        return approval

    try:
        prepared = prepare_session_bound_plan(
            writer_config=writer_config,
            fixture=base_fixture,
            writer_gid=coordinator_input.identities.writer_gid,
            staged_writer_config=DEFAULT_WRITER_CONFIG_SOURCE,
            staged_fixture=DEFAULT_E2E_FIXTURE,
            fixture_gid=coordinator_input.identities.gateway_gid,
            plan_builder=build_plan,
            approval_provider=approve,
            writer=publish_staged_writer,
            prestage_reconciler=stale_writer_clearance,
        )
        plan = prepared.plan
        live_result = driver_factory(prepared).run()
        if not _services_are_exactly_stopped_and_disabled():
            _fail("full_canary_terminal_service_truth_unconfirmed")
    except BaseException as exc:
        primary = exc
    finally:
        if prepared is not None:
            try:
                prepared.discard_session_key()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if not _services_are_exactly_stopped_and_disabled():
            try:
                mechanically_stop_full_canary_services()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if _services_are_exactly_stopped_and_disabled():
            try:
                retirement = _retire_discord_token_lease(
                    coordinator_input=coordinator_input,
                    install_receipt=discord_install_receipt,
                    installed=discord_token_identity,
                    install_snapshot=discord_receipt_snapshot,
                )
                token_retired = bool(
                    retirement.get("state") == "retired"
                    and retirement.get("token_removed") is True
                    and retirement.get("install_receipt_removed") is True
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
        if primary is not None or cleanup_errors:
            errors = ([] if primary is None else [primary]) + cleanup_errors
            raise ExceptionGroup("session-bound full-canary failed", errors)

    assert plan is not None and prepared is not None and live_result is not None
    if not token_retired:
        _fail("discord_token_retirement_unconfirmed", phase="terminal_cleanup")
    unsigned = {
        "schema": SESSION_BOUND_COORDINATOR_RECEIPT_SCHEMA,
        "ok": True,
        "state": "verified_stopped_and_credentials_retired",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "full_canary_plan_sha256": plan.sha256,
        "owner_approval_sha256": prepared.approval.sha256,
        "phase_b_readiness_anchor_sha256": _sha256_json(readiness_anchor),
        "api_session_key_sha256": prepared.session_key_sha256,
        "fixture_sha256": prepared.fixture_sha256,
        "live_driver_result": copy.deepcopy(dict(live_result)),
        "live_driver_receipt_sha256": _sha256_json(live_result),
        "services_stopped": True,
        "discord_token_retired": True,
        "temporary_admin_created": False,
        "bootstrap_credential_created": False,
        "completed_at_unix": int(time.time()),
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


# The admin-free implementation is the sole live target.
run_full_canary = run_session_bound_full_canary


def _cli_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Root-only isolated full-canary coordinator"
    )
    parser.add_argument(
        "command",
        choices=(
            "publish-coordinator-input",
            "preflight-phase-b-apply",
            "preflight-phase-b-live-run",
            "phase-b-apply",
            "run",
            "install-discord-token",
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
    process_lease_absent = not os.path.lexists(
        OBSOLETE_COORDINATOR_PROCESS_JOURNAL_PATH
    )
    try:
        services_terminal = _services_are_exactly_stopped_and_disabled()
    except BaseException:
        services_terminal = False
    complete = (
        not isinstance(error, CoordinatorCleanupBlocked)
        and discord_token_logically_removed
        and process_lease_absent
        and services_terminal
    )
    unsigned = {
        "schema": COORDINATOR_FAILURE_SCHEMA,
        "ok": False,
        "phase": phase,
        "command": command,
        "error_code": code,
        "release_sha": (
            None if coordinator_input is None else coordinator_input.revision
        ),
        "coordinator_input_sha256": (
            None if coordinator_input is None else coordinator_input.sha256
        ),
        "cleanup_status": "complete" if complete else "cleanup_blocked",
        "discord_token_removed": discord_token_logically_removed,
        "services_stopped": services_terminal,
        "obsolete_process_journal_absent": process_lease_absent,
        "completed_at_unix": int(time.time()),
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _execute_mutating_cli(
    *,
    command: str,
    operation: Callable[[], Mapping[str, Any]],
) -> int:
    try:
        receipt = operation()
    except BaseException as exc:
        with _defer_termination_signals():
            _emit_frame(_unbound_failure(exc, command=command))
        return 2
    with _defer_termination_signals():
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
        if command == "preflight-phase-b-apply":
            _emit_frame(preflight_phase_b_apply())
            return 0
        if command == "preflight-phase-b-live-run":
            _emit_frame(preflight_phase_b_live_run())
            return 0
        if command == "install-discord-token":
            return _execute_mutating_cli(
                command=command,
                operation=lambda: install_discord_token(gate_emitter=_emit_frame),
            )
        if command == "phase-b-apply":
            return _execute_mutating_cli(
                command=command,
                operation=lambda: apply_fixed_phase_b_foundation(
                    frame_emitter=_emit_frame,
                ),
            )
        if command == "stop-and-retire-discord-token":
            return _execute_mutating_cli(
                command=command,
                operation=lambda: stop_and_retire_discord_token(
                    gate_emitter=_emit_frame,
                ),
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
    "CANARY_DATABASE_HOST",
    "CANARY_DATABASE_NAME",
    "CANARY_DATABASE_PORT",
    "COORDINATOR_INPUT_PUBLICATION_SCHEMA",
    "COORDINATOR_INPUT_PATH",
    "COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH",
    "COORDINATOR_INPUT_SCHEMA",
    "FINAL_APPROVAL_FRAME_SCHEMA",
    "CoordinatorInput",
    "CoordinatorError",
    "OpaqueDiscordTokenFrame",
    "collect_and_publish_coordinator_input",
    "collect_coordinator_input",
    "discord_retirement_gate",
    "install_discord_token",
    "main",
    "publish_coordinator_input",
    "run_full_canary",
    "stop_and_retire_discord_token",
]


if __name__ == "__main__":
    raise SystemExit(main())
