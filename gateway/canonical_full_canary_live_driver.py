#!/usr/bin/env python3
"""Root-only honest live driver for the isolated full Muncho canary.

The runtime lifecycle owns systemd installation/start/stop.  This module owns
only the live observation which cannot be fabricated by an offline fixture:

* a one-shot API session key is selected before the plan is built and only its
  digest is staged in the privileged writer configuration;
* a root AF_UNIX collector authenticates the gateway MainPID with
  ``SO_PEERCRED`` and maintains an append-only frame hash chain;
* the exact fixture prompt is submitted through the authenticated loopback API
  without a system-message override;
* the Discord edge journal is logically identical across the deliberately
  invalid private-target probe;
* the writer's reviewed one-shot projection exporter supplies post-revocation
  Canonical truth, and the edge journal supplies the signed public request and
  receipt; and
* evidence is written only at the plan-addressed root-owned path and passed to
  the packaged offline verifier before services are stopped in reverse order.

There is no Discord ingress claim here.  The source is explicitly the
authenticated loopback API.  This module contains no keyword classifier,
semantic router, task decomposition, or external effort choice.  The model
authors plan steps, criteria, revisions, decisions, and the adaptive xhigh
directive; this driver checks mechanical identities, state transitions, and
receipts only.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import http.client
import json
import os
import re
import secrets
import socket
import sqlite3
import stat
import struct
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway.canonical_full_canary_e2e import (
    CANONICAL_TRUTH_RECEIPT_SCHEMA,
    EVIDENCE_SCHEMA,
    MODEL_CALL_RECEIPT_SCHEMA,
    PRIVATE_DENIAL_RECEIPT_SCHEMA,
    REASONING_RECEIPT_SCHEMA,
    SOURCE_RECEIPT_SCHEMA,
    TASK_OUTCOME_RECEIPT_SCHEMA,
    verify_evidence,
)
from gateway.canonical_full_canary_runtime import (
    COLLECTOR_IDENTITY_SCHEMA,
    COLLECTOR_READINESS_SCHEMA,
    DEFAULT_API_SERVER_CONTROL_KEY,
    DEFAULT_APPROVAL_PATH,
    DEFAULT_CANARY_PRECLAIM_RECONCILIATION_PATH,
    DEFAULT_COLLECTOR_READINESS_PATH,
    DEFAULT_COLLECTOR_SOCKET,
    DEFAULT_EDGE_READINESS_PATH,
    DEFAULT_PLUGIN_READINESS_PATH,
    FULL_CANARY_RECEIPT_SCHEMA,
    PLUGIN_FRAME_SCHEMA,
    PLUGIN_READINESS_SCHEMA,
    BootstrapProvisioner,
    CollectorReadiness,
    FullCanaryLifecycle,
    FullCanaryOwnerApproval,
    FullCanaryPlan,
    _api_loopback_listener_identity,
    _validated_e2e_fixture,
    canonical_canary_bootstrap_authorization_sha256,
    expected_live_evidence_path,
    load_full_canary_approval,
    load_start_receipt,
    observe_canary_preclaim_reconciliation_generation,
    validate_canary_preclaim_reconciliation_receipt,
)
from gateway.canonical_writer_activation import ActivationExecutor, ActivationPlan
from gateway.canonical_writer_boundary import (
    harden_current_process_against_dumping,
)
from gateway.canonical_writer_readiness import (
    boot_identity,
    module_file_identity,
    process_start_time_ticks,
    readiness_receipt_sha256,
)
from gateway.discord_edge_writer_authority import (
    derive_routeback_edge_idempotency_key,
)


LIVE_DRIVER_SCHEMA = "muncho-full-canary-live-driver.v1"
COLLECTOR_ACK_SCHEMA = "muncho-canary-evidence-ack.v1"
COLLECTOR_CHAIN_SCHEMA = "muncho-canary-evidence-chain.v1"
COLLECTOR_ZERO_CHAIN_SHA256 = "0" * 64

DEFAULT_EDGE_JOURNAL = Path(
    "/var/lib/muncho-discord-egress/discord-edge-journal.sqlite3"
)
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_HTTP_BODY_BYTES = 2 * 1024 * 1024
MAX_SSE_BYTES = 8 * 1024 * 1024
MAX_PROJECTION_BYTES = 32 * 1024 * 1024
MAX_COLLECTED_FRAMES = 1024
MAX_STAGED_WRITER_CONFIG_BYTES = 512 * 1024
_FRAME_HEADER = struct.Struct("!I")
_PEER_CREDENTIALS = struct.Struct("3i")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class LiveCanaryError(RuntimeError):
    """Stable, secret-free live-driver failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise LiveCanaryError(code)


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
        raise LiveCanaryError("non_canonical_json") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(code)
    return value


def _safe_id(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        _fail(code)
    return value


def _strict_mapping(
    value: Any,
    fields: frozenset[str],
    code: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(code)
    return value


def _strict_json(raw: bytes, code: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _token: (_ for _ in ()).throw(ValueError("constant")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise LiveCanaryError(code) from exc
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    if not callable(getter):
        raise PermissionError("full_canary_live_driver_requires_posix_uid")
    return int(getter())


def _effective_gid() -> int:
    getter = getattr(os, "getegid", None)
    if not callable(getter):
        raise PermissionError("full_canary_live_driver_requires_posix_gid")
    return int(getter())


def _require_root_linux() -> None:
    if sys.platform != "linux":
        raise PermissionError("full_canary_live_driver_requires_linux")
    if _effective_uid() != 0:
        raise PermissionError("full_canary_live_driver_requires_uid_0")
    harden_current_process_against_dumping()


def _utc_ms(value: Any, code: str) -> int:
    if not isinstance(value, str) or value != value.strip():
        _fail(code)
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise LiveCanaryError(code) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        _fail(code)
    return int(parsed.timestamp() * 1000)


def _fixture_expiry_iso(fixture: Mapping[str, Any]) -> str:
    value = fixture.get("valid_until_unix_ms")
    if type(value) is not int or value <= 0:
        _fail("fixture_expiry_invalid")
    return dt.datetime.fromtimestamp(
        value / 1000,
        tz=dt.timezone.utc,
    ).isoformat(timespec="milliseconds")


def _stable_read(
    path: Path,
    *,
    maximum: int,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    expected_mode: int | None = None,
) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise LiveCanaryError("trusted_file_unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= maximum
        or expected_uid is not None
        and before.st_uid != expected_uid
        or expected_gid is not None
        and before.st_gid != expected_gid
        or expected_mode is not None
        and stat.S_IMODE(before.st_mode) != expected_mode
    ):
        _fail("trusted_file_identity_invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LiveCanaryError("trusted_file_unavailable") from exc
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
        raise LiveCanaryError("trusted_file_replaced") from exc

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
        _fail("trusted_file_replaced")
    return b"".join(chunks)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_root(
    path: Path,
    payload: bytes,
    *,
    mode: int,
    uid: int = 0,
    gid: int = 0,
) -> None:
    if not path.is_absolute() or ".." in path.parts or not payload:
        _fail("root_artifact_path_invalid")
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid != 0
        or stat.S_IMODE(parent.st_mode) & 0o022
    ):
        _fail("root_artifact_parent_invalid")
    if os.path.lexists(path):
        _fail("root_artifact_already_exists")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail("root_artifact_write_stalled")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_staged_writer_config(
    path: Path,
    *,
    mode: int,
    uid: int,
    gid: int,
) -> tuple[bytes, os.stat_result]:
    """Read one exact staged generation without following replacements."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise LiveCanaryError("staged_writer_config_unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != uid
        or before.st_gid != gid
        or stat.S_IMODE(before.st_mode) != mode
        or not 0 < before.st_size <= MAX_STAGED_WRITER_CONFIG_BYTES
    ):
        _fail("staged_writer_config_identity_invalid")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    chunks: list[bytes] = []
    total = 0
    try:
        opened = os.fstat(descriptor)
        while total <= MAX_STAGED_WRITER_CONFIG_BYTES:
            chunk = os.read(
                descriptor,
                min(
                    64 * 1024,
                    MAX_STAGED_WRITER_CONFIG_BYTES + 1 - total,
                ),
            )
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
        raise LiveCanaryError("staged_writer_config_replaced") from exc

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
        total > MAX_STAGED_WRITER_CONFIG_BYTES
        or total != before.st_size
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
        or identity(before) != identity(reachable)
    ):
        _fail("staged_writer_config_replaced")
    return b"".join(chunks), before


def _decode_existing_staged_writer_config(raw: bytes) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if not isinstance(key, str) or not key or key in result:
                _fail("staged_writer_config_invalid")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        _fail("staged_writer_config_invalid")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise LiveCanaryError("staged_writer_config_invalid") from exc
    if not isinstance(value, Mapping):
        _fail("staged_writer_config_invalid")
    return value


PrestageReconciler = Callable[[], None]


def _reconcile_existing_staged_writer_config(
    path: Path,
    payload: bytes,
    *,
    mode: int,
    uid: int,
    gid: int,
    reconciler: PrestageReconciler | None,
) -> str | None:
    """Preserve an interrupted plan source until fresh DB retirement proof."""

    if not os.path.lexists(path):
        return None
    original_raw, original = _read_staged_writer_config(
        path,
        mode=mode,
        uid=uid,
        gid=gid,
    )
    original_sha256 = _sha256_bytes(original_raw)
    if original_raw == payload:
        return original_sha256
    config = _decode_existing_staged_writer_config(original_raw)
    service = config.get("service")
    scope = config.get("canary_scope_preapproval")
    if (
        reconciler is None
        or not isinstance(service, Mapping)
        or type(service.get("writer_uid")) is not int
        or service.get("writer_uid", 0) <= 0
        or service.get("writer_gid") != gid
        or not isinstance(scope, Mapping)
    ):
        _fail("staged_writer_config_unreconciled")
    prior_receipt_generation = observe_canary_preclaim_reconciliation_generation(
        DEFAULT_CANARY_PRECLAIM_RECONCILIATION_PATH
    )
    try:
        reconciler()
        validate_canary_preclaim_reconciliation_receipt(
            source_config_path=path,
            source_config_raw=original_raw,
            writer_config=config,
            writer_uid=service["writer_uid"],
            writer_gid=gid,
            allowed_outcomes=frozenset({"retired", "claimed"}),
            prior_generation=prior_receipt_generation,
            require_fresh_generation=True,
        )
    except Exception as exc:
        raise LiveCanaryError("staged_writer_config_unreconciled") from exc
    current_raw, current = _read_staged_writer_config(
        path,
        mode=mode,
        uid=uid,
        gid=gid,
    )
    if (
        _prestage_file_identity(current) != _prestage_file_identity(original)
        or _sha256_bytes(current_raw) != original_sha256
    ):
        _fail("staged_writer_config_replaced")
    return original_sha256


def _prestage_file_identity(item: os.stat_result) -> tuple[int, ...]:
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


def _atomic_stage_writer_config(
    path: Path,
    payload: bytes,
    *,
    mode: int,
    uid: int,
    gid: int,
    expected_existing_sha256: str | None = None,
) -> None:
    """Replace only a reconciled prior staged config, never another file."""

    if (
        not path.is_absolute()
        or ".." in path.parts
        or not payload
        or len(payload) > MAX_STAGED_WRITER_CONFIG_BYTES
    ):
        _fail("staged_writer_config_path_invalid")
    if expected_existing_sha256 is not None:
        expected_existing_sha256 = _digest(
            expected_existing_sha256,
            "staged_writer_config_expected_digest_invalid",
        )
    try:
        parent = path.parent.lstat()
    except OSError as exc:
        raise LiveCanaryError("staged_writer_config_parent_invalid") from exc
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid != 0
        or stat.S_IMODE(parent.st_mode) & 0o022
    ):
        _fail("staged_writer_config_parent_invalid")
    if os.path.lexists(path):
        existing_raw, existing = _read_staged_writer_config(
            path,
            mode=mode,
            uid=uid,
            gid=gid,
        )
        existing_sha256 = _sha256_bytes(existing_raw)
        if existing_raw == payload:
            return
        if (
            expected_existing_sha256 is None
            or existing_sha256 != expected_existing_sha256
        ):
            _fail("staged_writer_config_unreconciled")
        temporary = path.with_name(
            f".{path.name}.stage.{os.getpid()}.{uuid.uuid4().hex}"
        )
        descriptor = -1
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                mode,
            )
            os.fchown(descriptor, uid, gid)
            os.fchmod(descriptor, mode)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    _fail("staged_writer_config_write_stalled")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            # Re-open and re-read the exact object immediately before
            # replacing it.  Metadata alone is insufficient: a root-side
            # rewrite must not race the DB reconciliation receipt.
            current_raw, current = _read_staged_writer_config(
                path,
                mode=mode,
                uid=uid,
                gid=gid,
            )
            if (
                current.st_dev != existing.st_dev
                or current.st_ino != existing.st_ino
                or current.st_uid != existing.st_uid
                or current.st_gid != existing.st_gid
                or current.st_mode != existing.st_mode
                or current.st_nlink != existing.st_nlink
                or _sha256_bytes(current_raw) != expected_existing_sha256
            ):
                _fail("staged_writer_config_replaced")
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return
    if expected_existing_sha256 is not None:
        _fail("staged_writer_config_replaced")
    _atomic_write_root(path, payload, mode=mode, uid=uid, gid=gid)


class SessionBoundPlan:
    """One plan with an atomically consumable in-memory session key."""

    __slots__ = (
        "_approval",
        "_consumed",
        "_lock",
        "_plan",
        "_session_key",
        "_session_key_sha256",
        "_writer_config_sha256",
    )

    def __init__(
        self,
        *,
        session_key: str,
        session_key_sha256: str,
        writer_config_sha256: str,
        plan: FullCanaryPlan,
        approval: FullCanaryOwnerApproval,
    ) -> None:
        if not isinstance(session_key, str) or not session_key:
            _fail("session_key_invalid")
        self._session_key: str | None = session_key
        self._session_key_sha256 = _digest(
            session_key_sha256,
            "session_key_digest_invalid",
        )
        self._writer_config_sha256 = _digest(
            writer_config_sha256,
            "writer_config_digest_invalid",
        )
        if not isinstance(plan, FullCanaryPlan):
            raise TypeError("full-canary plan is required")
        if not isinstance(approval, FullCanaryOwnerApproval):
            raise TypeError("full-canary owner approval is required")
        self._plan = plan
        self._approval = approval
        self._lock = threading.Lock()
        self._consumed = False

    @property
    def session_key(self) -> str | None:
        """Return the key only until the driver atomically consumes it."""

        with self._lock:
            return self._session_key

    @property
    def session_key_sha256(self) -> str:
        return self._session_key_sha256

    @property
    def writer_config_sha256(self) -> str:
        return self._writer_config_sha256

    @property
    def plan(self) -> FullCanaryPlan:
        return self._plan

    @property
    def approval(self) -> FullCanaryOwnerApproval:
        return self._approval

    @property
    def consumed(self) -> bool:
        with self._lock:
            return self._consumed

    def consume_session_key(self) -> str:
        with self._lock:
            if self._consumed or self._session_key is None:
                _fail("session_bound_plan_already_consumed")
            value = self._session_key
            self._session_key = None
            self._consumed = True
            return value

    def discard_session_key(self) -> None:
        """Irrevocably retire the key without returning or otherwise using it."""

        with self._lock:
            self._session_key = None
            self._consumed = True

    def __repr__(self) -> str:
        return (
            "SessionBoundPlan("
            f"session_key_sha256={self.session_key_sha256!r}, "
            f"writer_config_sha256={self.writer_config_sha256!r}, "
            f"plan={self.plan!r}, approval={self.approval!r}, "
            f"consumed={self.consumed!r})"
        )


PlanBuilder = Callable[[], FullCanaryPlan]
ApprovalProvider = Callable[[FullCanaryPlan], FullCanaryOwnerApproval]


def prepare_session_bound_plan(
    *,
    writer_config: Mapping[str, Any],
    fixture: Mapping[str, Any],
    writer_gid: int,
    bootstrap_sql_sha256: str,
    bootstrap_retire_sql_sha256: str,
    staged_writer_config: Path,
    plan_builder: PlanBuilder,
    approval_provider: ApprovalProvider,
    session_key_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
    writer: Callable[..., None] = _atomic_stage_writer_config,
    prestage_reconciler: PrestageReconciler | None = None,
    process_guard: Callable[[], None] = _require_root_linux,
) -> SessionBoundPlan:
    """Stage the final session digest before building and approving the plan.

    ``approval_provider`` is deliberately called only after ``plan_builder``.
    It represents the separate owner-approval gate and cannot be replaced by
    this function.  The raw session key never enters JSON, a plan, or a
    receipt; the returned object keeps it in memory for the one live request.
    """

    process_guard()
    raw_key = session_key_factory()
    if (
        not isinstance(raw_key, str)
        or not 24 <= len(raw_key) <= 256
        or raw_key != raw_key.strip()
        or _CONTROL_RE.search(raw_key) is not None
    ):
        _fail("session_key_invalid")
    session_digest = _sha256_bytes(raw_key.encode("utf-8", errors="strict"))
    config = copy.deepcopy(dict(writer_config))
    scope = config.get("canary_scope_preapproval")
    if not isinstance(scope, Mapping):
        _fail("writer_scope_missing")
    clean_scope = dict(scope)
    clean_scope["session_key_sha256"] = session_digest
    clean_scope["expires_at"] = _fixture_expiry_iso(fixture)
    config["canary_scope_preapproval"] = clean_scope
    clean_scope["provisioning_receipt_sha256"] = (
        canonical_canary_bootstrap_authorization_sha256(
            config,
            bootstrap_sql_sha256=bootstrap_sql_sha256,
            bootstrap_retire_sql_sha256=bootstrap_retire_sql_sha256,
        )
    )
    payload = _canonical_bytes(config)
    if raw_key.encode("utf-8") in payload:
        _fail("raw_session_key_persistence_forbidden")
    expected_existing_sha256 = _reconcile_existing_staged_writer_config(
        staged_writer_config,
        payload,
        mode=0o440,
        uid=0,
        gid=writer_gid,
        reconciler=prestage_reconciler,
    )
    writer(
        staged_writer_config,
        payload,
        mode=0o440,
        uid=0,
        gid=writer_gid,
        expected_existing_sha256=expected_existing_sha256,
    )
    writer_digest = _sha256_bytes(payload)
    plan = plan_builder()
    if (
        not isinstance(plan, FullCanaryPlan)
        or plan.artifacts["writer_config"].source_path != staged_writer_config
        or plan.artifacts["writer_config"].sha256 != writer_digest
    ):
        _fail("session_bound_plan_drifted")
    approval = approval_provider(plan)
    if not isinstance(approval, FullCanaryOwnerApproval):
        _fail("owner_approval_missing")
    approval.require(plan_sha256=plan.sha256, now_unix=int(time.time()))
    return SessionBoundPlan(
        session_key=raw_key,
        session_key_sha256=session_digest,
        writer_config_sha256=writer_digest,
        plan=plan,
        approval=approval,
    )


def wait_for_fresh_owner_approval(
    plan: FullCanaryPlan,
    *,
    path: Path = DEFAULT_APPROVAL_PATH,
    timeout_seconds: float = 900.0,
    poll_seconds: float = 0.1,
    loader: Callable[[Path], FullCanaryOwnerApproval] = (load_full_canary_approval),
    monotonic: Callable[[], float] = time.monotonic,
    now: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
    ready_callback: Callable[[], None] | None = None,
    process_guard: Callable[[], None] = _require_root_linux,
) -> FullCanaryOwnerApproval:
    """Wait boundedly for a newly published approval for this exact plan.

    The deployment orchestrator calls this as ``approval_provider``.  The raw
    session key remains only in the caller's stack while an owner atomically
    publishes a new root:root ``0400`` approval at the fixed path.
    """

    process_guard()
    if not isinstance(plan, FullCanaryPlan) or path != DEFAULT_APPROVAL_PATH:
        _fail("owner_approval_wait_binding_invalid")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 1 <= timeout_seconds <= 900
        or isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, (int, float))
        or not 0.01 <= poll_seconds <= 1.0
        or ready_callback is not None
        and not callable(ready_callback)
    ):
        _fail("owner_approval_wait_bounds_invalid")

    def identity() -> tuple[int, ...] | None:
        try:
            item = path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise LiveCanaryError("owner_approval_unavailable") from exc
        if (
            not stat.S_ISREG(item.st_mode)
            or stat.S_ISLNK(item.st_mode)
            or item.st_nlink != 1
            or item.st_uid != 0
            or item.st_gid != 0
            or stat.S_IMODE(item.st_mode) != 0o400
        ):
            _fail("owner_approval_identity_invalid")
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

    baseline = identity()
    wait_started_unix = int(now())
    deadline = monotonic() + float(timeout_seconds)
    if ready_callback is not None:
        # The exact pre-existing approval identity is captured before the
        # coordinator tells the owner which plan digest is awaiting approval.
        # This closes the publication race without persisting any secret.
        ready_callback()
    while monotonic() <= deadline:
        observed = identity()
        if observed is not None and observed != baseline:
            try:
                approval = loader(path)
                approved_at = approval.value["approved_at_unix"]
                current_unix = int(now())
                if approved_at < wait_started_unix:
                    _fail("owner_approval_not_fresh")
                approval.require(
                    plan_sha256=plan.sha256,
                    now_unix=current_unix,
                )
                # The loader's stable read is followed by another identity
                # comparison so an approval cannot be swapped after loading.
                if identity() != observed:
                    _fail("owner_approval_replaced")
                return approval
            except (LiveCanaryError, PermissionError, RuntimeError, ValueError):
                # A newly visible but incomplete/incorrect publication may be
                # replaced by the owner within the same bounded window.
                pass
        sleeper(float(poll_seconds))
    _fail("owner_approval_wait_timeout")


@dataclass(frozen=True)
class PeerIdentity:
    pid: int
    uid: int
    gid: int
    start_time_ticks: int


@dataclass(frozen=True)
class JournalSnapshot:
    record_count: int
    logical_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "logical_sha256": self.logical_sha256,
        }


@dataclass(frozen=True)
class CollectedFrame:
    value: Mapping[str, Any]
    sha256: str
    chain_head_sha256: str
    peer: PeerIdentity


@dataclass(frozen=True)
class _JournalFileIdentity:
    device: int
    inode: int
    uid: int
    gid: int
    mode: int


def _edge_journal_identity(
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int,
) -> _JournalFileIdentity:
    if path != DEFAULT_EDGE_JOURNAL:
        _fail("edge_journal_path_invalid")
    try:
        item = path.lstat()
    except OSError as exc:
        raise LiveCanaryError("edge_journal_unavailable") from exc
    if (
        not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid != expected_uid
        or item.st_gid != expected_gid
        or stat.S_IMODE(item.st_mode) != 0o600
    ):
        _fail("edge_journal_identity_invalid")
    return _JournalFileIdentity(
        device=item.st_dev,
        inode=item.st_ino,
        uid=item.st_uid,
        gid=item.st_gid,
        mode=item.st_mode,
    )


def _logical_journal_snapshot(
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int,
) -> JournalSnapshot:
    """Read a coherent, mutation-free logical snapshot of the live WAL DB."""

    before = _edge_journal_identity(
        path,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )
    uri = f"file:{path}?mode=ro"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        tables = {
            "discord_edge_journal_meta_v1": (
                "singleton",
                "marker_id",
                "schema_version",
            ),
            "discord_edge_idempotency_v1": (
                "idempotency_key",
                "request_envelope_sha256",
                "request_envelope_json",
                "request_id",
                "capability_id",
                "request_sha256",
                "content_sha256",
                "state",
                "receipt_json",
                "blocker_code",
                "created_at_unix_ms",
                "updated_at_unix_ms",
            ),
            "discord_edge_receipt_history_v1": (
                "idempotency_key",
                "sequence",
                "receipt_json",
                "recorded_at_unix_ms",
            ),
        }
        projection: dict[str, list[dict[str, Any]]] = {}
        for table, columns in tables.items():
            order = ",".join(columns)
            rows = connection.execute(
                f"SELECT {order} FROM {table} ORDER BY {order}"  # noqa: S608
            ).fetchall()
            projection[table] = [{name: row[name] for name in columns} for row in rows]
        connection.execute("COMMIT")
    except sqlite3.Error as exc:
        raise LiveCanaryError("edge_journal_snapshot_failed") from exc
    finally:
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                pass
    after = _edge_journal_identity(
        path,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )
    if after != before:
        _fail("edge_journal_replaced")
    records = projection["discord_edge_idempotency_v1"]
    return JournalSnapshot(
        record_count=len(records),
        logical_sha256=_sha256_json(projection),
    )


def _read_edge_routeback(
    path: Path,
    *,
    idempotency_key: str,
    expected_uid: int,
    expected_gid: int,
) -> Mapping[str, Any]:
    _safe_id(idempotency_key, "edge_idempotency_key_invalid")
    before = _edge_journal_identity(
        path,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        row = connection.execute(
            """
            SELECT request_envelope_json, receipt_json, state, blocker_code
              FROM discord_edge_idempotency_v1
             WHERE idempotency_key = ?
            """,
            (idempotency_key,),
        ).fetchone()
        histories = connection.execute(
            """
            SELECT receipt_json
              FROM discord_edge_receipt_history_v1
             WHERE idempotency_key = ?
             ORDER BY sequence
            """,
            (idempotency_key,),
        ).fetchall()
        connection.execute("COMMIT")
    except sqlite3.Error as exc:
        raise LiveCanaryError("edge_routeback_read_failed") from exc
    finally:
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                pass
    after = _edge_journal_identity(
        path,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )
    if after != before:
        _fail("edge_journal_replaced")
    if (
        row is None
        or row["state"] != "verified"
        or row["blocker_code"] is not None
        or not isinstance(row["request_envelope_json"], str)
        or not isinstance(row["receipt_json"], str)
        or not histories
        or histories[-1]["receipt_json"] != row["receipt_json"]
    ):
        _fail("edge_routeback_not_verified")
    request_raw = row["request_envelope_json"].encode("utf-8", errors="strict")
    receipt_raw = row["receipt_json"].encode("utf-8", errors="strict")
    request = _strict_json(request_raw, "edge_routeback_request_invalid")
    receipt = _strict_json(receipt_raw, "edge_routeback_receipt_invalid")
    if request_raw != _canonical_bytes(request) or receipt_raw != _canonical_bytes(
        receipt
    ):
        _fail("edge_routeback_not_canonical")
    return {
        "provenance": "live_discord_edge_signed_receipt",
        "discord_edge_request": copy.deepcopy(dict(request)),
        "discord_edge_receipt": copy.deepcopy(dict(receipt)),
    }


class RootEvidenceCollector:
    """Authenticated in-process root collector for one sealed canary run."""

    _FRAME_FIELDS = frozenset({
        "schema",
        "sequence",
        "event",
        "release_sha",
        "release_sha256",
        "canary_run_id",
        "case_id",
        "fixture_sha256",
        "collector_service_identity_sha256",
        "discord_edge_service_identity_sha256",
        "session_id",
        "turn_id",
        "observed_at_unix_ms",
        "payload",
    })

    def __init__(
        self,
        plan: FullCanaryPlan,
        *,
        journal_path: Path = DEFAULT_EDGE_JOURNAL,
        journal_snapshotter: Callable[..., JournalSnapshot] = (
            _logical_journal_snapshot
        ),
        now: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        module_path: Path | None = None,
    ) -> None:
        if not isinstance(plan, FullCanaryPlan):
            raise TypeError("full-canary plan is required")
        self.plan = plan
        self.fixture = _validated_e2e_fixture(plan)
        self.journal_path = journal_path
        self._snapshotter = journal_snapshotter
        self._now = now
        self._monotonic = monotonic
        self._module_path = Path(__file__) if module_path is None else module_path
        self._thread: threading.Thread | None = None
        self._listener: socket.socket | None = None
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._plugin_ready = threading.Event()
        self._session_end = threading.Event()
        self._lock = threading.RLock()
        self._error: BaseException | None = None
        self._frames: list[CollectedFrame] = []
        self._chain_head = COLLECTOR_ZERO_CHAIN_SHA256
        self._gateway_peer: PeerIdentity | None = None
        self._collector_readiness: Mapping[str, Any] | None = None
        self._collector_readiness_file_sha256: str | None = None
        self._private_before: JournalSnapshot | None = None
        self._private_after: JournalSnapshot | None = None
        self._created_runtime_paths: dict[Path, tuple[int, ...]] = {}

    @staticmethod
    def _runtime_path_identity(path: Path) -> tuple[int, ...]:
        item = path.lstat()
        return (
            item.st_dev,
            item.st_ino,
            stat.S_IFMT(item.st_mode),
            stat.S_IMODE(item.st_mode),
            item.st_nlink,
            item.st_uid,
            item.st_gid,
        )

    def _remember_runtime_path(self, path: Path) -> None:
        self._created_runtime_paths[path] = self._runtime_path_identity(path)

    @property
    def frames(self) -> tuple[CollectedFrame, ...]:
        with self._lock:
            return tuple(self._frames)

    @property
    def private_snapshots(self) -> tuple[JournalSnapshot, JournalSnapshot]:
        with self._lock:
            if self._private_before is None or self._private_after is None:
                _fail("private_probe_evidence_missing")
            return self._private_before, self._private_after

    @property
    def chain_head_sha256(self) -> str:
        with self._lock:
            return self._chain_head

    def start(self) -> None:
        _require_root_linux()
        with self._lock:
            if self._thread is not None:
                _fail("collector_start_replayed")
            for path in (
                DEFAULT_COLLECTOR_SOCKET,
                DEFAULT_COLLECTOR_READINESS_PATH,
                DEFAULT_PLUGIN_READINESS_PATH,
            ):
                if os.path.lexists(path):
                    _fail("collector_runtime_not_fresh")
            self._thread = threading.Thread(
                target=self._serve,
                name="muncho-full-canary-root-collector",
                daemon=True,
            )
            self._thread.start()

    def _wait_runtime_directory(self, timeout: float = 120.0) -> None:
        deadline = self._monotonic() + timeout
        while not self._closed.is_set() and self._monotonic() < deadline:
            try:
                item = DEFAULT_COLLECTOR_SOCKET.parent.lstat()
            except FileNotFoundError:
                self._closed.wait(0.05)
                continue
            if (
                stat.S_ISDIR(item.st_mode)
                and not stat.S_ISLNK(item.st_mode)
                and item.st_uid == 0
                and item.st_gid == self.plan.identities.gateway_gid
                and stat.S_IMODE(item.st_mode) == 0o750
            ):
                return
            _fail("collector_runtime_directory_invalid")
        _fail("collector_runtime_directory_timeout")

    def _bind_listener(self) -> socket.socket:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.set_inheritable(False)
        try:
            listener.bind(str(DEFAULT_COLLECTOR_SOCKET))
            os.chown(
                DEFAULT_COLLECTOR_SOCKET,
                0,
                self.plan.identities.gateway_gid,
            )
            os.chmod(DEFAULT_COLLECTOR_SOCKET, 0o660)
            listener.listen(4)
            listener.settimeout(0.25)
            self._remember_runtime_path(DEFAULT_COLLECTOR_SOCKET)
        except BaseException:
            listener.close()
            raise
        return listener

    def _wait_edge_readiness(self, timeout: float = 180.0) -> Mapping[str, Any]:
        deadline = self._monotonic() + timeout
        while not self._closed.is_set() and self._monotonic() < deadline:
            try:
                raw = _stable_read(
                    DEFAULT_EDGE_READINESS_PATH,
                    maximum=512 * 1024,
                    expected_uid=self.plan.identities.edge_uid,
                    expected_gid=self.plan.identities.edge_gid,
                    expected_mode=0o400,
                )
                value = _strict_json(raw, "edge_readiness_invalid")
                if (
                    value.get("version") == "muncho-discord-edge-readiness-v1"
                    and value.get("config_sha256")
                    == self.plan.artifacts["edge_config"].sha256
                    and type(value.get("edge_pid")) is int
                    and value["edge_pid"] > 1
                ):
                    return value
            except LiveCanaryError:
                pass
            self._closed.wait(0.05)
        _fail("edge_readiness_timeout")

    def _publish_collector_readiness(
        self,
        edge_readiness: Mapping[str, Any],
    ) -> None:
        if self._listener is None:
            _fail("collector_listener_missing")
        module_origin, module_sha256 = module_file_identity(self._module_path)
        boot_sha256, boottime_ns = boot_identity()
        pid = os.getpid()
        identity = {
            "schema": COLLECTOR_IDENTITY_SCHEMA,
            "release_sha": self.plan.revision,
            "collector_pid": pid,
            "collector_start_time_ticks": process_start_time_ticks(pid),
            "collector_uid": _effective_uid(),
            "collector_gid": _effective_gid(),
            "boot_id_sha256": boot_sha256,
            "module_origin": module_origin,
            "module_sha256": module_sha256,
        }
        socket_item = DEFAULT_COLLECTOR_SOCKET.lstat()
        unsigned = {
            "schema": COLLECTOR_READINESS_SCHEMA,
            "release_sha": self.plan.revision,
            "full_canary_plan_sha256": self.plan.sha256,
            "canary_run_id": self.fixture["canary_run_id"],
            "edge_pid": edge_readiness["edge_pid"],
            "edge_service_identity_sha256": readiness_receipt_sha256(edge_readiness),
            "collector_socket": {
                "path": str(DEFAULT_COLLECTOR_SOCKET),
                "device": socket_item.st_dev,
                "inode": socket_item.st_ino,
                "uid": socket_item.st_uid,
                "gid": socket_item.st_gid,
                "mode": "0660",
            },
            "service_identity": identity,
            "service_identity_sha256": _sha256_json(identity),
            "observed_at_unix": int(self._now()),
            "observed_at_boottime_ns": boottime_ns,
        }
        receipt = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
        payload = _canonical_bytes(receipt)
        _atomic_write_root(
            DEFAULT_COLLECTOR_READINESS_PATH,
            payload,
            mode=0o400,
        )
        self._remember_runtime_path(DEFAULT_COLLECTOR_READINESS_PATH)
        with self._lock:
            self._collector_readiness = receipt
            self._collector_readiness_file_sha256 = _sha256_bytes(payload)
        self._ready.set()

    @staticmethod
    def _peer(connection: socket.socket) -> PeerIdentity:
        peercred = getattr(socket, "SO_PEERCRED", None)
        if peercred is None:
            _fail("collector_peer_credentials_unavailable")
        try:
            raw = connection.getsockopt(
                socket.SOL_SOCKET,
                peercred,
                _PEER_CREDENTIALS.size,
            )
        except OSError as exc:
            raise LiveCanaryError("collector_peer_credentials_unavailable") from exc
        if len(raw) != _PEER_CREDENTIALS.size:
            _fail("collector_peer_credentials_unavailable")
        pid, uid, gid = _PEER_CREDENTIALS.unpack(raw)
        return PeerIdentity(pid, uid, gid, process_start_time_ticks(pid))

    @staticmethod
    def _receive_exact(connection: socket.socket, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            try:
                chunk = connection.recv(remaining)
            except (OSError, socket.timeout) as exc:
                raise LiveCanaryError("collector_frame_truncated") from exc
            if not chunk:
                _fail("collector_frame_truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _validate_common_frame(
        self,
        frame: Mapping[str, Any],
        peer: PeerIdentity,
    ) -> None:
        _strict_mapping(frame, self._FRAME_FIELDS, "collector_frame_invalid")
        expected_sequence = len(self._frames) + 1
        readiness = self._collector_readiness
        if (
            not isinstance(readiness, Mapping)
            or frame.get("schema") != PLUGIN_FRAME_SCHEMA
            or frame.get("sequence") != expected_sequence
            or frame.get("release_sha") != self.plan.revision
            or frame.get("release_sha256") != self.plan.release["artifact_sha256"]
            or frame.get("canary_run_id") != self.fixture["canary_run_id"]
            or frame.get("case_id") != self.fixture["case_id"]
            or frame.get("fixture_sha256") != self.plan.artifacts["e2e_fixture"].sha256
            or frame.get("collector_service_identity_sha256")
            != readiness.get("service_identity_sha256")
            or frame.get("discord_edge_service_identity_sha256")
            != readiness.get("edge_service_identity_sha256")
            or not isinstance(frame.get("event"), str)
            or not isinstance(frame.get("payload"), Mapping)
            or type(frame.get("observed_at_unix_ms")) is not int
            or not self.fixture["valid_from_unix_ms"]
            <= frame["observed_at_unix_ms"]
            <= self.fixture["valid_until_unix_ms"]
        ):
            _fail("collector_frame_binding_invalid")
        if expected_sequence == 1:
            payload = frame["payload"]
            if (
                frame["event"] != "plugin_ready"
                or frame["session_id"] is not None
                or frame["turn_id"] is not None
                or peer.uid != self.plan.identities.gateway_uid
                or peer.gid != self.plan.identities.gateway_gid
                or payload.get("gateway_pid") != peer.pid
            ):
                _fail("plugin_ready_peer_invalid")
            self._gateway_peer = peer
        elif self._gateway_peer != peer:
            _fail("collector_gateway_peer_rotated")

    def _publish_plugin_readiness(
        self,
        frame: Mapping[str, Any],
        frame_sha256: str,
        chain_head: str,
        peer: PeerIdentity,
    ) -> None:
        if (
            self._collector_readiness is None
            or self._collector_readiness_file_sha256 is None
        ):
            _fail("collector_readiness_missing")
        boot_sha256, boottime_ns = boot_identity()
        unsigned = {
            "schema": PLUGIN_READINESS_SCHEMA,
            "full_canary_plan_sha256": self.plan.sha256,
            "canary_run_id": self.fixture["canary_run_id"],
            "collector_readiness_file_sha256": (self._collector_readiness_file_sha256),
            "gateway_peer": {
                "pid": peer.pid,
                "start_time_ticks": peer.start_time_ticks,
                "uid": peer.uid,
                "gid": peer.gid,
            },
            "plugin_ready_frame": copy.deepcopy(dict(frame)),
            "plugin_ready_frame_sha256": frame_sha256,
            "collector_hash_chain_head_sha256": chain_head,
            "boot_id_sha256": boot_sha256,
            "observed_at_unix": int(self._now()),
            "observed_at_boottime_ns": boottime_ns,
        }
        receipt = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
        _atomic_write_root(
            DEFAULT_PLUGIN_READINESS_PATH,
            _canonical_bytes(receipt),
            mode=0o400,
        )
        self._remember_runtime_path(DEFAULT_PLUGIN_READINESS_PATH)
        self._plugin_ready.set()

    def _accept_frame(self, connection: socket.socket) -> None:
        connection.settimeout(4.0)
        peer = self._peer(connection)
        header = self._receive_exact(connection, _FRAME_HEADER.size)
        (size,) = _FRAME_HEADER.unpack(header)
        if not 1 < size <= MAX_FRAME_BYTES:
            _fail("collector_frame_size_invalid")
        body = self._receive_exact(connection, size)
        if self._peer(connection) != peer:
            _fail("collector_peer_rotated")
        frame = _strict_json(body, "collector_frame_invalid")
        if body != _canonical_bytes(frame):
            _fail("collector_frame_not_canonical")
        with self._lock:
            if len(self._frames) >= MAX_COLLECTED_FRAMES:
                _fail("collector_frame_limit_exceeded")
            self._validate_common_frame(frame, peer)
            frame_sha256 = _sha256_bytes(body)
            chain = {
                "schema": COLLECTOR_CHAIN_SCHEMA,
                "previous_sha256": self._chain_head,
                "sequence": frame["sequence"],
                "frame_sha256": frame_sha256,
                "peer_pid": peer.pid,
                "peer_start_time_ticks": peer.start_time_ticks,
            }
            chain_head = _sha256_json(chain)
            event = frame["event"]
            if event == "private_target_probe_ready":
                if self._private_before is not None:
                    _fail("private_probe_replayed")
                self._private_before = self._snapshotter(
                    self.journal_path,
                    expected_uid=self.plan.identities.edge_uid,
                    expected_gid=self.plan.identities.edge_gid,
                )
            elif event == "private_target_probe_result":
                if self._private_before is None or self._private_after is not None:
                    _fail("private_probe_order_invalid")
                after = self._snapshotter(
                    self.journal_path,
                    expected_uid=self.plan.identities.edge_uid,
                    expected_gid=self.plan.identities.edge_gid,
                )
                if after != self._private_before:
                    _fail("private_probe_changed_journal")
                self._private_after = after
            collected = CollectedFrame(
                value=copy.deepcopy(dict(frame)),
                sha256=frame_sha256,
                chain_head_sha256=chain_head,
                peer=peer,
            )
            self._frames.append(collected)
            self._chain_head = chain_head
            if frame["sequence"] == 1:
                self._publish_plugin_readiness(
                    frame,
                    frame_sha256,
                    chain_head,
                    peer,
                )
            if event == "session_end":
                self._session_end.set()
            ack = {
                "schema": COLLECTOR_ACK_SCHEMA,
                "sequence": frame["sequence"],
                "accepted": True,
                "frame_sha256": frame_sha256,
                "collector_receipt_sha256": chain_head,
            }
        payload = _canonical_bytes(ack)
        connection.sendall(_FRAME_HEADER.pack(len(payload)) + payload)

    def _serve(self) -> None:
        try:
            self._wait_runtime_directory()
            self._listener = self._bind_listener()
            edge = self._wait_edge_readiness()
            self._publish_collector_readiness(edge)
            while not self._closed.is_set():
                try:
                    connection, _address = self._listener.accept()
                except socket.timeout:
                    continue
                with connection:
                    self._accept_frame(connection)
        except BaseException as exc:
            with self._lock:
                self._error = exc
            self._ready.set()
            self._plugin_ready.set()
            self._session_end.set()

    def _wait_event(self, event: threading.Event, timeout: float, code: str) -> None:
        if not event.wait(timeout):
            _fail(code)
        with self._lock:
            if self._error is not None:
                raise LiveCanaryError(code) from self._error

    def wait_ready(self, timeout: float = 180.0) -> None:
        self._wait_event(self._ready, timeout, "collector_readiness_failed")

    def wait_plugin_ready(self, timeout: float = 180.0) -> None:
        self._wait_event(self._plugin_ready, timeout, "plugin_readiness_failed")

    def wait_session_end(self, timeout: float = 600.0) -> None:
        self._wait_event(self._session_end, timeout, "session_end_evidence_missing")

    def close(self) -> None:
        self._closed.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        replaced = False
        for path in (
            DEFAULT_PLUGIN_READINESS_PATH,
            DEFAULT_COLLECTOR_READINESS_PATH,
            DEFAULT_COLLECTOR_SOCKET,
        ):
            expected = self._created_runtime_paths.get(path)
            if expected is None:
                continue
            try:
                if self._runtime_path_identity(path) != expected:
                    replaced = True
                    continue
                path.unlink()
            except FileNotFoundError:
                pass
        if replaced:
            _fail("collector_runtime_artifact_replaced")


@dataclass(frozen=True)
class SSEConversation:
    session_id: str
    session_create_request_id: str
    chat_stream_request_id: str
    api_run_id: str
    api_message_id: str
    events: tuple[tuple[str, Mapping[str, Any]], ...]
    assistant_completed: Mapping[str, Any]
    run_completed: Mapping[str, Any]
    observed_at_unix_ms: int
    completed_at_unix_ms: int


class LoopbackCanaryClient:
    """Bounded authenticated client for the exact local canary API surface."""

    def __init__(
        self,
        *,
        control_key: str,
        session_key: str,
        connection_factory: Callable[..., http.client.HTTPConnection] = (
            http.client.HTTPConnection
        ),
        timeout: float = 600.0,
    ) -> None:
        if not control_key or _CONTROL_RE.search(control_key) is not None:
            _fail("api_control_key_invalid")
        if not session_key or _CONTROL_RE.search(session_key) is not None:
            _fail("session_key_invalid")
        self._control_key: str | None = control_key
        self._session_key: str | None = session_key
        self._secret_lock = threading.Lock()
        self._connection_factory = connection_factory
        self._timeout = timeout

    def _secret_values(self) -> tuple[str, str]:
        with self._secret_lock:
            if self._control_key is None or self._session_key is None:
                _fail("api_client_secrets_consumed")
            return self._control_key, self._session_key

    def clear_secrets(self) -> None:
        """Idempotently drop both raw credentials from client storage."""

        with self._secret_lock:
            self._control_key = None
            self._session_key = None

    @property
    def secrets_cleared(self) -> bool:
        with self._secret_lock:
            return self._control_key is None and self._session_key is None

    def _connection(self) -> http.client.HTTPConnection:
        return self._connection_factory("127.0.0.1", 8642, timeout=self._timeout)

    def _headers(self, *, session: bool = False, request_id: str) -> dict[str, str]:
        control_key, session_key = self._secret_values()
        headers = {
            "Authorization": f"Bearer {control_key}",
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        }
        if session:
            headers["X-Hermes-Session-Key"] = session_key
        return headers

    @staticmethod
    def _bounded_response(response: http.client.HTTPResponse) -> bytes:
        body = response.read(MAX_HTTP_BODY_BYTES + 1)
        if len(body) > MAX_HTTP_BODY_BYTES:
            _fail("api_response_oversized")
        return body

    def create_session(self, session_id: str, request_id: str) -> None:
        _safe_id(session_id, "api_session_id_invalid")
        connection = self._connection()
        body = _canonical_bytes({"id": session_id})
        try:
            connection.request(
                "POST",
                "/api/sessions",
                body=body,
                headers=self._headers(request_id=request_id),
            )
            response = connection.getresponse()
            raw = self._bounded_response(response)
        except OSError as exc:
            raise LiveCanaryError("api_session_create_failed") from exc
        finally:
            connection.close()
        if response.status != 201:
            _fail("api_session_create_rejected")
        value = _strict_json(raw, "api_session_create_invalid")
        session = value.get("session")
        if (
            value.get("object") != "hermes.session"
            or not isinstance(session, Mapping)
            or session.get("id") != session_id
            or session.get("source") != "api_server"
        ):
            _fail("api_session_create_invalid")

    @staticmethod
    def _parse_sse(
        response: http.client.HTTPResponse,
    ) -> list[tuple[str, Mapping[str, Any]]]:
        events: list[tuple[str, Mapping[str, Any]]] = []
        total = 0
        event_name: str | None = None
        data_lines: list[bytes] = []
        while True:
            line = response.readline(256 * 1024 + 1)
            total += len(line)
            if len(line) > 256 * 1024 or total > MAX_SSE_BYTES:
                _fail("api_sse_oversized")
            if not line:
                break
            if line in {b"\n", b"\r\n"}:
                if event_name is not None:
                    raw_data = b"\n".join(data_lines)
                    value = _strict_json(raw_data, "api_sse_event_invalid")
                    events.append((event_name, copy.deepcopy(dict(value))))
                event_name = None
                data_lines = []
                continue
            if line.startswith(b":"):
                continue
            clean = line.rstrip(b"\r\n")
            if clean.startswith(b"event: "):
                try:
                    event_name = clean[7:].decode("ascii", errors="strict")
                except UnicodeDecodeError as exc:
                    raise LiveCanaryError("api_sse_event_invalid") from exc
            elif clean.startswith(b"data: "):
                data_lines.append(clean[6:])
            else:
                _fail("api_sse_event_invalid")
        if event_name is not None:
            _fail("api_sse_truncated")
        return events

    def run(
        self,
        *,
        fixture: Mapping[str, Any],
        session_id: str | None = None,
    ) -> SSEConversation:
        session = session_id or f"api_canary_{uuid.uuid4().hex}"
        create_request_id = str(uuid.uuid4())
        stream_request_id = str(uuid.uuid4())
        self.create_session(session, create_request_id)
        prompt = fixture["task_policy"]["prompt"]
        body = _canonical_bytes({"message": prompt})
        _control_key, expected_session_key = self._secret_values()
        connection = self._connection()
        path = f"/api/sessions/{session}/chat/stream"
        try:
            connection.request(
                "POST",
                path,
                body=body,
                headers=self._headers(session=True, request_id=stream_request_id),
            )
            response = connection.getresponse()
            if response.status != 200:
                self._bounded_response(response)
                _fail("api_sse_rejected")
            if response.getheader("X-Hermes-Session-Key") != expected_session_key:
                _fail("api_session_key_echo_invalid")
            content_type = str(response.getheader("Content-Type") or "")
            if not content_type.startswith("text/event-stream"):
                _fail("api_sse_content_type_invalid")
            events = self._parse_sse(response)
        except OSError as exc:
            raise LiveCanaryError("api_sse_failed") from exc
        finally:
            connection.close()
        names = [name for name, _payload in events]
        if (
            names.count("run.started") != 1
            or names.count("assistant.completed") != 1
            or names.count("run.completed") != 1
            or names.count("done") != 1
            or names[-1:] != ["done"]
            or any(
                name in {"error", "run.failed", "run.partial", "run.cancelled"}
                for name in names
            )
        ):
            _fail("api_sse_terminal_invalid")
        payloads = [payload for _name, payload in events]
        sequences = [payload.get("seq") for payload in payloads]
        if sequences != list(range(1, len(events) + 1)):
            _fail("api_sse_sequence_invalid")
        run_ids = {payload.get("run_id") for payload in payloads}
        session_ids = {payload.get("session_id") for payload in payloads}
        if len(run_ids) != 1 or len(session_ids) != 1 or session_ids != {session}:
            _fail("api_sse_correlation_invalid")
        run_id = _safe_id(next(iter(run_ids)), "api_sse_correlation_invalid")
        assistant = next(
            payload for name, payload in events if name == "assistant.completed"
        )
        terminal = next(payload for name, payload in events if name == "run.completed")
        started = next(payload for name, payload in events if name == "run.started")
        message_id = _safe_id(
            terminal.get("message_id"),
            "api_sse_correlation_invalid",
        )
        if (
            assistant.get("message_id") != message_id
            or terminal.get("completed") is not True
            or terminal.get("partial") is not False
            or terminal.get("interrupted") is not False
            or terminal.get("failed") is not False
            or terminal.get("status") != "completed"
            or terminal.get("turn_exit_reason") != "text_response(finish_reason=stop)"
            or assistant.get("status") != "completed"
        ):
            _fail("api_sse_completion_not_honest")
        try:
            observed = int(float(started.get("ts")) * 1000)
            completed = int(float(terminal.get("ts")) * 1000)
        except (TypeError, ValueError, OverflowError) as exc:
            raise LiveCanaryError("api_sse_timestamp_invalid") from exc
        if completed < observed:
            _fail("api_sse_timestamp_invalid")
        return SSEConversation(
            session_id=session,
            session_create_request_id=create_request_id,
            chat_stream_request_id=stream_request_id,
            api_run_id=run_id,
            api_message_id=message_id,
            events=tuple(events),
            assistant_completed=copy.deepcopy(dict(assistant)),
            run_completed=copy.deepcopy(dict(terminal)),
            observed_at_unix_ms=observed,
            completed_at_unix_ms=completed,
        )


def _read_api_control_key(
    path: Path = DEFAULT_API_SERVER_CONTROL_KEY,
) -> tuple[str, str]:
    raw = _stable_read(
        path,
        maximum=4096,
        expected_uid=0,
        expected_gid=0,
        expected_mode=0o400,
    )
    if raw.endswith(b"\n"):
        raw = raw[:-1]
    try:
        value = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise LiveCanaryError("api_control_key_invalid") from exc
    if not value or value != value.strip() or _CONTROL_RE.search(value):
        _fail("api_control_key_invalid")
    item = path.lstat()
    # This receipt deliberately hashes metadata, never the secret value.
    provenance = {
        "path": str(path),
        "device": item.st_dev,
        "inode": item.st_ino,
        "uid": item.st_uid,
        "gid": item.st_gid,
        "mode": "0400",
        "size": item.st_size,
    }
    return value, _sha256_json(provenance)


def _verify_listener_binding(
    start_receipt: Mapping[str, Any],
    *,
    checker: Callable[[int], Mapping[str, Any]] = _api_loopback_listener_identity,
) -> None:
    identities = start_receipt.get("service_identity_receipts")
    expected = start_receipt.get("api_loopback_listener")
    if not isinstance(identities, Mapping) or not isinstance(expected, Mapping):
        _fail("api_listener_receipt_missing")
    gateway = identities.get("gateway")
    if not isinstance(gateway, Mapping) or not isinstance(
        gateway.get("receipt"), Mapping
    ):
        _fail("api_listener_receipt_missing")
    pid = gateway["receipt"].get("gateway_pid")
    if type(pid) is not int or pid <= 1 or dict(checker(pid)) != dict(expected):
        _fail("api_listener_not_owned_by_gateway")


def _read_projection_events(plan: FullCanaryPlan) -> list[Mapping[str, Any]]:
    activation = ActivationPlan.from_mapping(plan.writer_activation_plan)
    path = activation.paths.projection_export_path
    raw = _stable_read(
        path,
        maximum=MAX_PROJECTION_BYTES,
        expected_uid=activation.identities.writer_uid,
        expected_gid=activation.identities.projector_gid,
        expected_mode=0o640,
    )
    value = _strict_json(raw, "projection_export_invalid")
    events = value.get("events")
    if not isinstance(events, list) or any(
        not isinstance(item, Mapping) for item in events
    ):
        _fail("projection_export_invalid")
    if len(events) != len({str(item.get("event_id")) for item in events}):
        _fail("projection_export_duplicate_event")
    return [copy.deepcopy(dict(item)) for item in events]


def _run_projection_export(plan: FullCanaryPlan) -> list[Mapping[str, Any]]:
    """Use the already reviewed, plan-bound temporary writer exporter once."""

    activation = ActivationPlan.from_mapping(plan.writer_activation_plan)
    executor = ActivationExecutor(activation)
    # This private implementation is the exact executor already bound into the
    # approved writer-only ActivationPlan.  Calling it avoids adding a second
    # privileged exporter seam or a new model-visible operation.
    receipt = executor._run_projection_export()  # noqa: SLF001
    if not isinstance(receipt, Mapping) or type(receipt.get("event_count")) is not int:
        _fail("projection_export_receipt_invalid")
    events = _read_projection_events(plan)
    if receipt["event_count"] != len(events):
        _fail("projection_export_count_drifted")
    return events


def _frame_values(
    frames: Sequence[CollectedFrame],
    event: str,
) -> list[Mapping[str, Any]]:
    return [item.value for item in frames if item.value.get("event") == event]


def _one_frame(
    frames: Sequence[CollectedFrame],
    event: str,
) -> Mapping[str, Any]:
    values = _frame_values(frames, event)
    if len(values) != 1:
        _fail("collector_event_cardinality_invalid")
    return values[0]


def _event_payload(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    payload = value.get("payload")
    item = payload.get(key) if isinstance(payload, Mapping) else None
    if not isinstance(item, Mapping):
        _fail("canonical_projection_payload_invalid")
    return item


def _event_digest(value: Mapping[str, Any]) -> str:
    return _sha256_json(value)


def _projection_case_events(
    projection: Sequence[Mapping[str, Any]],
    case_id: str,
) -> list[Mapping[str, Any]]:
    events = [value for value in projection if value.get("case_id") == case_id]
    if not events:
        _fail("canonical_projection_case_missing")
    return events


_READBACK_FIELDS = frozenset({
    "request_id",
    "status",
    "events",
    "support_events",
    "view",
    "bounded",
    "event_count",
    "truncated",
    "candidate_cases_truncated",
    "support_incomplete_reasons",
    "missing_verification_event_ids",
})


def _validate_live_readback(
    readback: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    projection_events: Sequence[Mapping[str, Any]],
) -> tuple[bool, list[str]]:
    """Validate the real pre-revocation resume bundle against projection."""

    if set(readback) != _READBACK_FIELDS:
        _fail("canonical_live_readback_invalid")
    events = readback.get("events")
    support = readback.get("support_events")
    reasons = readback.get("support_incomplete_reasons")
    missing = readback.get("missing_verification_event_ids")
    if (
        readback.get("status") != "ok"
        or readback.get("view") != "resume_bundle"
        or readback.get("bounded") is not True
        or readback.get("truncated") is not False
        or readback.get("candidate_cases_truncated") is not False
        or payload.get("query_view") != "resume_bundle"
        or payload.get("query_limit") != 200
        or not isinstance(events, list)
        or not isinstance(support, list)
        or any(not isinstance(item, Mapping) for item in [*events, *support])
        or type(readback.get("event_count")) is not int
        or readback.get("event_count") != len(events)
        or not isinstance(reasons, list)
        or any(not isinstance(item, str) or not item for item in reasons)
        or not isinstance(missing, list)
        or any(not isinstance(item, str) or not item for item in missing)
    ):
        _fail("canonical_live_readback_invalid")
    # Any omission reason makes the readback unsuitable as proof.  These
    # booleans are therefore derived from the actual writer response, not
    # relabelled by the collector.
    support_incomplete = bool(reasons or missing)
    if support_incomplete:
        _fail("canonical_live_readback_incomplete")
    combined = [*events, *support]
    readback_by_id: dict[str, Mapping[str, Any]] = {}
    for item in combined:
        event_id = str(item.get("event_id") or "")
        if not event_id or event_id in readback_by_id:
            _fail("canonical_live_readback_invalid")
        readback_by_id[event_id] = item
    projection_by_id = {
        str(item.get("event_id") or ""): item for item in projection_events
    }
    if not readback_by_id or len(projection_by_id) != len(projection_events):
        _fail("canonical_live_readback_invalid")
    for event_id, item in readback_by_id.items():
        projected = projection_by_id.get(event_id)
        if projected is None or _sha256_json(projected) != _sha256_json(item):
            _fail("canonical_live_readback_projection_drift")
    post_readback = [
        item
        for event_id, item in projection_by_id.items()
        if event_id not in readback_by_id
    ]
    if (
        len(post_readback) != 1
        or post_readback[0].get("event_type") != "canary.scope.revoked"
    ):
        _fail("canonical_live_readback_projection_drift")
    return support_incomplete, list(missing)


def _normalize_scope_events(
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    expected = (
        ("canary.scope.preapproved", "canary_scope_preapproval"),
        ("canary.scope.claimed", "canary_scope_claim"),
        ("canary.scope.revoked", "canary_scope_revocation"),
    )
    normalized: list[Mapping[str, Any]] = []
    for event_type, payload_key in expected:
        matches = [value for value in events if value.get("event_type") == event_type]
        if len(matches) != 1:
            _fail("canonical_scope_event_cardinality_invalid")
        raw = matches[0]
        scope = dict(_event_payload(raw, payload_key))
        if event_type in {"canary.scope.preapproved", "canary.scope.claimed"}:
            expires_at = scope.pop("expires_at", None)
            scope["expires_at_unix_ms"] = _utc_ms(
                expires_at,
                "canonical_scope_expiry_invalid",
            )
        if event_type == "canary.scope.revoked":
            safety = raw.get("safety")
            if not isinstance(safety, Mapping):
                _fail("canonical_scope_tombstone_receipt_missing")
            scope["session_tombstone_recorded"] = safety.get(
                "session_tombstone_recorded"
            )
        normalized.append({
            "event_id": raw.get("event_id"),
            "event_type": event_type,
            "case_id": raw.get("case_id"),
            "occurred_at_unix_ms": _utc_ms(
                raw.get("occurred_at"),
                "canonical_scope_time_invalid",
            ),
            "readback_verified": True,
            "canonical_content_sha256": _event_digest(raw),
            "scope": scope,
        })
    revoked = normalized[-1]
    revoked_scope = revoked["scope"]
    retirement = {
        "grant_id": revoked_scope["grant_id"],
        "session_key_sha256": revoked_scope["session_key_sha256"],
        "capability_epoch_sha256": revoked_scope["capability_epoch_sha256"],
        "authority_active": False,
        "revocation_event_id": revoked["event_id"],
        "session_tombstone_commit_receipt_verified": (
            revoked_scope.get("session_tombstone_recorded") is True
        ),
        "observed_at_unix_ms": revoked["occurred_at_unix_ms"],
    }
    return normalized, retirement


def _normalize_plan_events(
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    plan_rows = [
        value for value in events if value.get("event_type") == "task.plan.updated"
    ]
    if not plan_rows:
        _fail("canonical_plan_projection_missing")
    try:
        plan_rows.sort(key=lambda value: int(_event_payload(value, "plan")["revision"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveCanaryError("canonical_plan_revision_invalid") from exc
    normalized: list[Mapping[str, Any]] = []
    completion_receipts: list[Mapping[str, Any]] = []
    previous: Mapping[str, str] | None = None
    previous_current: str | None = None
    for raw in plan_rows:
        plan = _event_payload(raw, "plan")
        steps_raw = plan.get("steps")
        criteria_raw = plan.get("success_criteria")
        cursor_raw = plan.get("resume_cursor")
        if (
            not isinstance(steps_raw, list)
            or not isinstance(criteria_raw, list)
            or not isinstance(cursor_raw, Mapping)
        ):
            _fail("canonical_plan_projection_invalid")
        steps: list[dict[str, Any]] = []
        for step in steps_raw:
            if not isinstance(step, Mapping):
                _fail("canonical_plan_projection_invalid")
            steps.append({
                "id": step.get("id"),
                "status": step.get("status"),
                "depends_on": copy.deepcopy(list(step.get("depends_on") or [])),
            })
        criterion_ids = [
            criterion.get("id") if isinstance(criterion, Mapping) else None
            for criterion in criteria_raw
        ]
        current = str(plan.get("current_step_id") or "") or None
        verification_ids = copy.deepcopy(list(plan.get("verification_event_ids") or []))
        projected_plan = {
            "plan_id": plan.get("plan_id"),
            "revision": plan.get("revision"),
            "state": plan.get("state"),
            "current_step_id": current,
            "resume_cursor": {
                "next_step_id": str(cursor_raw.get("next_step_id") or "") or None,
                **(
                    {"summary": cursor_raw["summary"]}
                    if isinstance(cursor_raw.get("summary"), str)
                    else {}
                ),
            },
            "steps": steps,
            "criterion_ids": criterion_ids,
            "verification_event_ids": verification_ids,
        }
        normalized.append({
            "event_id": raw.get("event_id"),
            "event_type": "task.plan.updated",
            "case_id": raw.get("case_id"),
            "readback_verified": True,
            "canonical_content_sha256": _event_digest(raw),
            "plan": projected_plan,
        })
        status = {str(step["id"]): str(step["status"]) for step in steps}
        if previous is not None:
            newly_completed = [
                step_id
                for step_id in status
                if previous.get(step_id) != "completed"
                and status[step_id] == "completed"
            ]
            if newly_completed != [previous_current]:
                _fail("canonical_plan_completion_transition_invalid")
            completion_receipts.append({
                "step_id": newly_completed[0],
                "completion_ordinal": len(completion_receipts) + 1,
                "tool_receipt_sha256": _event_digest(raw),
            })
        previous = status
        previous_current = current
    return normalized, completion_receipts


def _normalize_verification_events(
    events: Sequence[Mapping[str, Any]],
    verification_event_ids: Sequence[str],
) -> list[Mapping[str, Any]]:
    by_id = {
        str(value.get("event_id")): value
        for value in events
        if value.get("event_type") == "task.verification.recorded"
    }
    normalized: list[Mapping[str, Any]] = []
    for event_id in verification_event_ids:
        raw = by_id.get(str(event_id))
        if raw is None:
            _fail("canonical_verification_projection_missing")
        verification = dict(_event_payload(raw, "verification"))
        raw_receipt = verification.get("receipt")
        if not isinstance(raw_receipt, Mapping):
            _fail("canonical_verification_receipt_invalid")
        kind = raw_receipt.get("kind")
        verification["receipt"] = {
            "kind": kind,
            "sha256": _sha256_json(raw_receipt),
        }
        normalized.append({
            "event_id": raw.get("event_id"),
            "event_type": "task.verification.recorded",
            "case_id": raw.get("case_id"),
            "readback_verified": True,
            "canonical_content_sha256": _event_digest(raw),
            "verification": {
                name: verification.get(name)
                for name in (
                    "verification_id",
                    "plan_id",
                    "plan_revision",
                    "criterion_ids",
                    "outcome",
                    "receipt",
                )
            },
        })
    return normalized


def _normalize_routeback_event(
    events: Sequence[Mapping[str, Any]],
    fixture: Mapping[str, Any],
) -> Mapping[str, Any]:
    sent = [value for value in events if value.get("event_type") == "route_back.sent"]
    blocked = [
        value for value in events if value.get("event_type") == "route_back.blocked"
    ]
    if len(sent) != 1 or blocked:
        _fail("canonical_routeback_terminal_invalid")
    raw = sent[0]
    payload = raw.get("payload")
    if not isinstance(payload, Mapping):
        _fail("canonical_routeback_projection_invalid")
    routeback = payload.get("route_back")
    receipt = payload.get("receipt")
    if not isinstance(routeback, Mapping) or not isinstance(receipt, Mapping):
        _fail("canonical_routeback_projection_invalid")
    return {
        "event_id": raw.get("event_id"),
        "event_type": "route_back.sent",
        "case_id": raw.get("case_id"),
        "readback_verified": True,
        "canonical_content_sha256": _event_digest(raw),
        "canonical_idempotency_key": fixture["public_routeback"][
            "canonical_idempotency_key"
        ],
        "authorization_id": payload.get("authorization_id"),
        "target_ref": copy.deepcopy(dict(routeback.get("target_ref") or {})),
        "receipt": {
            name: receipt.get(name)
            for name in (
                "platform",
                "adapter_receipt",
                "receipt_readback_verified",
                "message_id",
                "channel_id",
                "content_sha256",
            )
        },
    }


def _normalize_model_calls(
    frames: Sequence[CollectedFrame],
    *,
    fixture: Mapping[str, Any],
    session_id: str,
    turn_id: str,
) -> list[Mapping[str, Any]]:
    pre_frames = _frame_values(frames, "pre_api_request")
    post_frames = _frame_values(frames, "post_api_request")
    post_by_id = {
        str(value["payload"].get("api_request_id")): value
        for value in post_frames
        if isinstance(value.get("payload"), Mapping)
    }
    if len(pre_frames) < 2 or len(pre_frames) != len(post_by_id):
        _fail("model_call_evidence_incomplete")
    result: list[Mapping[str, Any]] = []
    for expected_ordinal, pre in enumerate(pre_frames, start=1):
        payload = pre.get("payload")
        if not isinstance(payload, Mapping):
            _fail("model_call_evidence_invalid")
        post = post_by_id.get(str(payload.get("api_request_id")))
        post_payload = post.get("payload") if isinstance(post, Mapping) else None
        if (
            not isinstance(post_payload, Mapping)
            or payload.get("request_ordinal") != expected_ordinal
            or post_payload.get("request_ordinal") != expected_ordinal
            or pre.get("session_id") != session_id
            or pre.get("turn_id") != turn_id
            or post.get("session_id") != session_id
            or post.get("turn_id") != turn_id
        ):
            _fail("model_call_evidence_invalid")
        result.append({
            "schema": MODEL_CALL_RECEIPT_SCHEMA,
            "provenance": "live_gateway_model_adapter",
            "release_sha": fixture["release_sha"],
            "canary_run_id": fixture["canary_run_id"],
            "session_id": session_id,
            "turn_id": turn_id,
            "request_ordinal": expected_ordinal,
            "provider": payload.get("provider"),
            "api_mode": payload.get("api_mode"),
            "base_url": payload.get("base_url"),
            "model": payload.get("model"),
            "reasoning_effort": payload.get("reasoning_effort"),
            "api_request_sha256": payload.get("api_request_sha256"),
            "response_payload_sha256": post_payload.get("response_payload_sha256"),
            "response_model": post_payload.get("response_model"),
            "response_observed_at_unix_ms": post_payload.get(
                "response_observed_at_unix_ms"
            ),
            "assistant_tool_call_ids": copy.deepcopy(
                list(post_payload.get("assistant_tool_call_ids") or [])
            ),
        })
    return result


def _normalize_reasoning_directive(
    frames: Sequence[CollectedFrame],
    *,
    fixture: Mapping[str, Any],
    session_id: str,
    turn_id: str,
) -> Mapping[str, Any]:
    matches: list[Mapping[str, Any]] = []
    for frame in _frame_values(frames, "post_tool_call"):
        payload = frame.get("payload")
        if (
            isinstance(payload, Mapping)
            and payload.get("tool_name") == "todo"
            and isinstance(payload.get("reasoning_directive"), Mapping)
        ):
            matches.append(frame)
    if len(matches) != 1:
        _fail("model_reasoning_directive_invalid")
    frame = matches[0]
    payload = frame["payload"]
    directive = payload["reasoning_directive"]
    if directive.get("effort") != "xhigh":
        _fail("model_reasoning_directive_invalid")
    return {
        "schema": REASONING_RECEIPT_SCHEMA,
        "provenance": "live_gateway_assistant_tool_call",
        "release_sha": fixture["release_sha"],
        "canary_run_id": fixture["canary_run_id"],
        "session_id": session_id,
        "turn_id": turn_id,
        "tool_name": "todo",
        "tool_call_id": payload.get("tool_call_id"),
        "model_authored": True,
        "directive": {"effort": directive["effort"]},
        "reasoning_control": copy.deepcopy(
            dict(payload.get("reasoning_control") or {})
        ),
        "produced_by_model_call_ordinal": payload.get("produced_by_model_call_ordinal"),
        "applied_before_model_call_ordinal": 2,
        "todo_result_sha256": payload.get("result_sha256"),
    }


def assemble_live_evidence(
    *,
    plan: FullCanaryPlan,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
    start_receipt: Mapping[str, Any],
    start_receipt_file_sha256: str,
    frames: Sequence[CollectedFrame],
    conversation: SSEConversation,
    credential_provenance_sha256: str,
    private_before: JournalSnapshot,
    private_after: JournalSnapshot,
    projection_events: Sequence[Mapping[str, Any]],
    public_routeback: Mapping[str, Any],
    collector_chain_head_sha256: str,
    collected_at_unix_ms: int | None = None,
) -> Mapping[str, Any]:
    """Join independently authenticated receipts into the verifier contract."""

    if not frames or [item.value.get("sequence") for item in frames] != list(
        range(1, len(frames) + 1)
    ):
        _fail("collector_frame_sequence_invalid")
    chain_head = COLLECTOR_ZERO_CHAIN_SHA256
    for item in frames:
        if item.sha256 != _sha256_json(item.value):
            _fail("collector_frame_digest_invalid")
        chain_head = _sha256_json({
            "schema": COLLECTOR_CHAIN_SCHEMA,
            "previous_sha256": chain_head,
            "sequence": item.value["sequence"],
            "frame_sha256": item.sha256,
            "peer_pid": item.peer.pid,
            "peer_start_time_ticks": item.peer.start_time_ticks,
        })
        if item.chain_head_sha256 != chain_head:
            _fail("collector_hash_chain_invalid")
    startup = [item.value.get("event") for item in frames[:4]]
    if startup != [
        "plugin_ready",
        "canonical_scope_claim",
        "private_target_probe_ready",
        "private_target_probe_result",
    ]:
        _fail("collector_startup_sequence_invalid")
    claim_frame = frames[1].value
    claim = claim_frame.get("payload")
    if not isinstance(claim, Mapping) or claim.get("success") is not True:
        _fail("canonical_scope_claim_failed")
    first_pre = _frame_values(frames, "pre_api_request")
    if not first_pre:
        _fail("model_call_evidence_incomplete")
    session_id = _safe_id(claim_frame.get("session_id"), "session_binding_invalid")
    turn_id = _safe_id(first_pre[0].get("turn_id"), "turn_binding_invalid")
    if conversation.session_id != session_id:
        _fail("api_plugin_session_mismatch")
    for frame in frames[4:]:
        if (
            frame.value.get("session_id") != session_id
            or frame.value.get("turn_id") != turn_id
        ):
            _fail("collector_turn_binding_invalid")
    session_end = _one_frame(frames, "session_end")
    session_payload = session_end.get("payload")
    if (
        not isinstance(session_payload, Mapping)
        or session_payload.get("completed") is not True
        or session_payload.get("interrupted") is not False
    ):
        _fail("plugin_session_end_incomplete")
    readback_frame = _one_frame(frames, "canonical_case_readback")
    readback_payload = readback_frame.get("payload")
    if not isinstance(readback_payload, Mapping):
        _fail("canonical_live_readback_invalid")
    readback = readback_payload.get("readback")
    if not isinstance(readback, Mapping) or _sha256_json(
        readback
    ) != readback_payload.get("readback_sha256"):
        _fail("canonical_live_readback_invalid")

    case_events = _projection_case_events(projection_events, fixture["case_id"])
    support_incomplete, missing_verification_ids = _validate_live_readback(
        readback,
        payload=readback_payload,
        projection_events=case_events,
    )
    scope_events, retirement = _normalize_scope_events(case_events)
    plan_events, completion_receipts = _normalize_plan_events(case_events)
    final_plan = plan_events[-1]["plan"]
    verification_events = _normalize_verification_events(
        case_events,
        final_plan["verification_event_ids"],
    )
    passed_verification_ids = {
        item["event_id"]
        for item in verification_events
        if item["verification"].get("outcome") == "passed"
    }
    required_verification_ids = set(final_plan["verification_event_ids"])
    plan_projection_complete = bool(plan_events) and (
        final_plan.get("state") == "completed"
    )
    completion_receipts_satisfied = bool(required_verification_ids) and (
        required_verification_ids <= passed_verification_ids
    )
    if (
        support_incomplete
        or missing_verification_ids
        or not plan_projection_complete
        or not completion_receipts_satisfied
    ):
        _fail("canonical_live_readback_incomplete")
    routeback_event = _normalize_routeback_event(case_events, fixture)
    model_calls = _normalize_model_calls(
        frames,
        fixture=fixture,
        session_id=session_id,
        turn_id=turn_id,
    )
    reasoning = _normalize_reasoning_directive(
        frames,
        fixture=fixture,
        session_id=session_id,
        turn_id=turn_id,
    )
    identities = start_receipt.get("service_identity_receipts")
    collector_receipt = start_receipt.get("collector_readiness_receipt")
    if not isinstance(identities, Mapping) or not isinstance(
        collector_receipt, Mapping
    ):
        _fail("runtime_provenance_missing")
    try:
        gateway_identity = identities["gateway"]
        writer_identity = identities["writer"]
        edge_identity = identities["edge"]
        writer_readiness = gateway_identity["receipt"]
        gateway_sha256 = gateway_identity["sha256"]
        writer_sha256 = writer_identity["sha256"]
        edge_sha256 = edge_identity["sha256"]
    except (KeyError, TypeError) as exc:
        raise LiveCanaryError("runtime_provenance_missing") from exc
    private_result = frames[3].value.get("payload")
    if not isinstance(private_result, Mapping) or private_before != private_after:
        _fail("private_probe_evidence_invalid")
    now_ms = (
        int(time.time() * 1000)
        if collected_at_unix_ms is None
        else collected_at_unix_ms
    )
    if not fixture["valid_from_unix_ms"] <= now_ms <= fixture["valid_until_unix_ms"]:
        _fail("evidence_collection_outside_fixture_window")
    retirement = {**dict(retirement), "observed_at_unix_ms": now_ms}
    tool_frames = _frame_values(frames, "post_tool_call")
    final_content = conversation.assistant_completed.get("content")
    if not isinstance(final_content, str) or not final_content:
        _fail("final_response_missing")
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "fixture_sha256": fixture_sha256,
        "collected_at_unix_ms": now_ms,
        "runtime_provenance": {
            "execution_mode": "live_isolated_canary",
            "synthetic": False,
            "release_sha": fixture["release_sha"],
            "canary_run_id": fixture["canary_run_id"],
            "owner_discord_user_id": fixture["owner_discord_user_id"],
            "full_canary_start_receipt_sha256": start_receipt_file_sha256,
            "gateway_service_identity_sha256": gateway_sha256,
            "canonical_writer_service_identity_sha256": writer_sha256,
            "discord_edge_service_identity_sha256": edge_sha256,
            "collector_receipt_sha256": collector_receipt.get("receipt_sha256"),
        },
        "writer_readiness": copy.deepcopy(dict(writer_readiness)),
        "source_receipt": {
            "schema": SOURCE_RECEIPT_SCHEMA,
            "provenance": "live_gateway_authenticated_loopback_api",
            "release_sha": fixture["release_sha"],
            "canary_run_id": fixture["canary_run_id"],
            "session_id": session_id,
            "turn_id": turn_id,
            "platform": "api_server",
            "control_protocol": "authenticated_loopback_api_server.v1",
            "host": "127.0.0.1",
            "port": 8642,
            "session_create_endpoint": "/api/sessions",
            "chat_stream_endpoint": f"/api/sessions/{session_id}/chat/stream",
            "session_create_request_id": conversation.session_create_request_id,
            "chat_stream_request_id": conversation.chat_stream_request_id,
            "api_run_id": conversation.api_run_id,
            "api_message_id": conversation.api_message_id,
            "loopback_peer_verified": True,
            "credential_provenance_receipt_sha256": (credential_provenance_sha256),
            "session_key_sha256": claim.get("session_key_sha256"),
            "capability_epoch_sha256": claim.get("capability_epoch_sha256"),
            "message_content_sha256": fixture["task_policy"]["prompt_sha256"],
            "observed_at_unix_ms": conversation.observed_at_unix_ms,
        },
        "model_calls": model_calls,
        "reasoning_directive": reasoning,
        "canonical_truth": {
            "schema": CANONICAL_TRUTH_RECEIPT_SCHEMA,
            "provenance": "canonical_writer_live_readback",
            "release_sha": fixture["release_sha"],
            "canary_run_id": fixture["canary_run_id"],
            "writer_query_request_id": readback_payload.get("writer_request_id"),
            "observed_at_unix_ms": now_ms,
            "query_status": "CANONICAL_BRAIN_QUERY_PASS",
            "query_view": "resume_bundle",
            "case_id": fixture["case_id"],
            "support_incomplete": support_incomplete,
            "plan_projection_complete": plan_projection_complete,
            "completion_receipts_satisfied": completion_receipts_satisfied,
            "missing_verification_event_ids": missing_verification_ids,
            "scope_events": scope_events,
            "scope_retirement": retirement,
            "plan_events": plan_events,
            "verification_events": verification_events,
            "routeback_event": routeback_event,
        },
        "public_routeback": copy.deepcopy(dict(public_routeback)),
        "private_denial": {
            "schema": PRIVATE_DENIAL_RECEIPT_SCHEMA,
            "provenance": "live_gateway_to_discord_edge_probe",
            "release_sha": fixture["release_sha"],
            "canary_run_id": fixture["canary_run_id"],
            "session_id": session_id,
            "turn_id": turn_id,
            "observed_at_unix_ms": private_result.get("observed_at_unix_ms"),
            "discord_edge_service_identity_sha256": private_result.get(
                "discord_edge_service_identity_sha256"
            ),
            "socket_identity_sha256": private_result.get("socket_identity_sha256"),
            "attempt_frame_sha256": private_result.get("attempt_frame_sha256"),
            "attempted_operation": private_result.get("attempted_operation"),
            "attempted_target_type": private_result.get("attempted_target_type"),
            "connection_closed_without_response": private_result.get(
                "connection_closed_without_response"
            ),
            "signed_receipt_observed": private_result.get("signed_receipt_observed"),
            "journal_snapshot_before": private_before.to_mapping(),
            "journal_snapshot_after": private_after.to_mapping(),
        },
        "task_outcome": {
            "schema": TASK_OUTCOME_RECEIPT_SCHEMA,
            "provenance": "live_gateway_turn_completion",
            "release_sha": fixture["release_sha"],
            "canary_run_id": fixture["canary_run_id"],
            "session_id": session_id,
            "turn_id": turn_id,
            "api_run_id": conversation.api_run_id,
            "api_message_id": conversation.api_message_id,
            "case_id": fixture["case_id"],
            "plan_id": final_plan["plan_id"],
            "stream_terminal_event": "run.completed",
            "completed": conversation.run_completed.get("completed"),
            "partial": conversation.run_completed.get("partial"),
            "interrupted": conversation.run_completed.get("interrupted"),
            "failed": conversation.run_completed.get("failed"),
            "turn_exit_reason": conversation.run_completed.get("turn_exit_reason"),
            "completed_at_unix_ms": conversation.completed_at_unix_ms,
            "completed_steps": completion_receipts,
            "model_call_count": len(model_calls),
            "tool_call_count": len(tool_frames),
            "final_response_sha256": _sha256_bytes(
                final_content.encode("utf-8", errors="strict")
            ),
        },
    }
    # Recompute the collector chain independently from stored frames before
    # accepting the caller's terminal head.
    if chain_head != collector_chain_head_sha256:
        _fail("collector_hash_chain_terminal_mismatch")
    return evidence


def _write_evidence(
    plan: FullCanaryPlan, evidence: Mapping[str, Any]
) -> tuple[Path, str]:
    path = expected_live_evidence_path(plan)
    parent = path.parent
    # The plan-addressed root may be absent on the first run.  Build only this
    # fixed hierarchy with private root ownership; never accept a symlink or a
    # pre-existing evidence file.
    missing: list[Path] = []
    current = parent
    while not os.path.lexists(current):
        missing.append(current)
        current = current.parent
    anchor = current.lstat()
    if (
        not stat.S_ISDIR(anchor.st_mode)
        or stat.S_ISLNK(anchor.st_mode)
        or anchor.st_uid != 0
        or stat.S_IMODE(anchor.st_mode) & 0o022
    ):
        _fail("evidence_parent_invalid")
    for directory in reversed(missing):
        os.mkdir(directory, 0o700)
        os.chown(directory, 0, 0)
        os.chmod(directory, 0o700)
        _fsync_directory(directory.parent)
    payload = _canonical_bytes(evidence)
    _atomic_write_root(path, payload, mode=0o400)
    return path, _sha256_bytes(payload)


LifecycleFactory = Callable[[FullCanaryPlan], FullCanaryLifecycle]
CollectorFactory = Callable[[FullCanaryPlan], RootEvidenceCollector]
ClientFactory = Callable[[str, str], LoopbackCanaryClient]


class HonestFullCanaryDriver:
    """Execute one approved plan and fail closed at every evidence barrier."""

    def __init__(
        self,
        prepared: SessionBoundPlan,
        *,
        lifecycle_factory: LifecycleFactory | None = None,
        bootstrap_provisioner: BootstrapProvisioner | None = None,
        collector_factory: CollectorFactory = RootEvidenceCollector,
        client_factory: ClientFactory = lambda control, session: LoopbackCanaryClient(
            control_key=control,
            session_key=session,
        ),
        control_key_reader: Callable[[], tuple[str, str]] = _read_api_control_key,
        projection_exporter: Callable[
            [FullCanaryPlan], list[Mapping[str, Any]]
        ] = _run_projection_export,
        edge_routeback_reader: Callable[..., Mapping[str, Any]] = (
            _read_edge_routeback
        ),
        evidence_writer: Callable[
            [FullCanaryPlan, Mapping[str, Any]], tuple[Path, str]
        ] = _write_evidence,
        root_guard: Callable[[], None] = _require_root_linux,
    ) -> None:
        if not isinstance(prepared, SessionBoundPlan):
            raise TypeError("session-bound full-canary plan is required")
        if lifecycle_factory is not None and bootstrap_provisioner is not None:
            raise TypeError(
                "full-canary lifecycle factory and bootstrap provisioner "
                "are mutually exclusive"
            )
        if bootstrap_provisioner is not None and any(
            not callable(getattr(bootstrap_provisioner, name, None))
            for name in ("provision", "reconcile", "abort")
        ):
            raise TypeError("full-canary bootstrap provisioner contract is invalid")
        self.prepared = prepared
        self._lifecycle_factory = lifecycle_factory
        self._bootstrap_provisioner = bootstrap_provisioner
        self._collector_factory = collector_factory
        self._client_factory = client_factory
        self._control_key_reader = control_key_reader
        self._projection_exporter = projection_exporter
        self._edge_routeback_reader = edge_routeback_reader
        self._evidence_writer = evidence_writer
        self._root_guard = root_guard

    def run(self) -> Mapping[str, Any]:
        result: Mapping[str, Any] | None = None
        primary: BaseException | None = None
        try:
            try:
                self._root_guard()
            except BaseException:
                # Process hardening must precede local secret use.  If that
                # gate itself fails, retire the caller-owned key without
                # reading it so even the earliest failure cannot leave
                # reusable authority.
                self.prepared.discard_session_key()
                raise
            session_key = self.prepared.consume_session_key()
            try:
                result = self._run_consumed(session_key)
            finally:
                # The caller-owned holder was already cleared atomically; drop
                # the driver's last wrapper reference regardless of
                # validation/start.
                session_key = None
        except BaseException as exc:
            primary = exc
        provisioner = self._bootstrap_provisioner
        # The driver must never retain the owner-operated admin boundary after
        # this one attempt, including root-guard and fixture failures before a
        # lifecycle exists.  The concrete provisioner makes abort idempotent,
        # so this also safely closes sessions already retired by lifecycle.
        self._bootstrap_provisioner = None
        abort_error: BaseException | None = None
        if provisioner is not None:
            try:
                provisioner.abort()
            except BaseException as first_abort_error:
                # The provisioner contract is idempotent and the concrete
                # session marks itself closed only after close succeeds.  One
                # bounded retry covers a transient close failure even when the
                # driver failed before lifecycle.start() could perform its own
                # cleanup.
                try:
                    provisioner.abort()
                except BaseException as second_abort_error:
                    abort_error = ExceptionGroup(
                        "bootstrap admin-session abort retries failed",
                        [first_abort_error, second_abort_error],
                    )
        if primary is not None and abort_error is not None:
            raise ExceptionGroup(
                "full-canary live run and bootstrap admin-session abort failed",
                [primary, abort_error],
            ) from None
        if primary is not None:
            raise primary
        if abort_error is not None:
            raise abort_error
        if result is None:
            raise RuntimeError("full-canary live run produced no result")
        return result

    def _lifecycle(self, plan: FullCanaryPlan) -> FullCanaryLifecycle:
        if self._lifecycle_factory is not None:
            return self._lifecycle_factory(plan)
        if self._bootstrap_provisioner is None:
            # FullCanaryLifecycle's own default is deliberately blocked.  A
            # live database bootstrap is impossible unless the owner passes an
            # already-open provisioner explicitly.
            return FullCanaryLifecycle(plan)
        return FullCanaryLifecycle(
            plan,
            bootstrap_provisioner=self._bootstrap_provisioner,
        )

    def _run_consumed(self, session_key: str) -> Mapping[str, Any]:
        plan = self.prepared.plan
        fixture = _validated_e2e_fixture(plan)
        if (
            self.prepared.session_key_sha256
            != _sha256_bytes(session_key.encode("utf-8"))
            or fixture["release_artifact_sha256"] != plan.release["artifact_sha256"]
        ):
            _fail("prepared_plan_binding_invalid")
        lifecycle = self._lifecycle(plan)
        collector = self._collector_factory(plan)
        try:
            services_live = False
            stop_completed = False
            primary: BaseException | None = None
            result: Mapping[str, Any] | None = None
            control_key: str | None = None
            client: LoopbackCanaryClient | None = None
            collector.start()
            start_result = lifecycle.start(self.prepared.approval)
            services_live = True
            receipt_path = Path(str(start_result.get("receipt_path") or ""))
            loaded_start = load_start_receipt(receipt_path, plan=plan)
            _verify_listener_binding(loaded_start.value)
            collector.wait_plugin_ready()
            control_key, credential_provenance = self._control_key_reader()
            client = self._client_factory(control_key, session_key)
            clear_client_secrets = getattr(client, "clear_secrets", None)
            if not callable(clear_client_secrets):
                _fail("api_client_secret_cleanup_unavailable")
            try:
                conversation = client.run(fixture=fixture)
            finally:
                clear_client_secrets()
                control_key = None
                session_key = None
            # A listener swap during a long model/tool run must not inherit
            # the preflight's authority.
            _verify_listener_binding(loaded_start.value)
            # API run.completed is emitted only after the gateway's finally
            # block has durably revoked the exact capability epoch.
            collector.wait_session_end()
            projection = self._projection_exporter(plan)
            target_key = derive_routeback_edge_idempotency_key(
                case_id=fixture["case_id"],
                canonical_idempotency_key=fixture["public_routeback"][
                    "canonical_idempotency_key"
                ],
            )
            public_routeback = self._edge_routeback_reader(
                DEFAULT_EDGE_JOURNAL,
                idempotency_key=target_key,
                expected_uid=plan.identities.edge_uid,
                expected_gid=plan.identities.edge_gid,
            )
            before, after = collector.private_snapshots
            evidence = assemble_live_evidence(
                plan=plan,
                fixture=fixture,
                fixture_sha256=plan.artifacts["e2e_fixture"].sha256,
                start_receipt=loaded_start.value,
                start_receipt_file_sha256=loaded_start.file_sha256,
                frames=collector.frames,
                conversation=conversation,
                credential_provenance_sha256=credential_provenance,
                private_before=before,
                private_after=after,
                projection_events=projection,
                public_routeback=public_routeback,
                collector_chain_head_sha256=collector.chain_head_sha256,
            )
            evidence_path, evidence_sha256 = self._evidence_writer(plan, evidence)
            # Fast in-process contract check before invoking the independently
            # packaged verifier through the lifecycle's verify-and-stop gate.
            invariant = verify_evidence(
                fixture,
                evidence,
                start_receipt_sha256=loaded_start.file_sha256,
                fixture_sha256=plan.artifacts["e2e_fixture"].sha256,
                evidence_sha256=evidence_sha256,
            )
            verification = lifecycle.verify_and_stop(
                start_receipt_path=receipt_path,
                evidence_path=evidence_path,
                evidence_sha256=evidence_sha256,
            )
            stop_completed = True
            services_live = False
            result = {
                "schema": LIVE_DRIVER_SCHEMA,
                "ok": True,
                "release_sha": plan.revision,
                "full_canary_plan_sha256": plan.sha256,
                "canary_run_id": fixture["canary_run_id"],
                "evidence_path": str(evidence_path),
                "evidence_sha256": evidence_sha256,
                "offline_invariant_receipt": invariant,
                "lifecycle_verification_receipt": copy.deepcopy(dict(verification)),
                "discord_ingress_claimed": False,
            }
        except BaseException as exc:
            primary = exc
        finally:
            cleanup_error: BaseException | None = None
            if client is not None:
                try:
                    clear = getattr(client, "clear_secrets", None)
                    if callable(clear):
                        clear()
                except BaseException as exc:
                    cleanup_error = exc
            control_key = None
            session_key = None
            if services_live and not stop_completed:
                try:
                    lifecycle.stop(reason="verification_failed")
                    stop_completed = True
                except BaseException as exc:
                    cleanup_error = exc
            try:
                collector.close()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                else:
                    cleanup_error = ExceptionGroup(
                        "full-canary cleanup failed",
                        [cleanup_error, exc],
                    )
            if cleanup_error is not None:
                if primary is None:
                    primary = cleanup_error
                else:
                    primary = ExceptionGroup(
                        "full-canary live run and ordered stop failed",
                        [primary, cleanup_error],
                    )
        if primary is not None:
            raise RuntimeError("full-canary live driver failed closed") from primary
        assert result is not None
        return result


__all__ = [
    "CollectedFrame",
    "HonestFullCanaryDriver",
    "JournalSnapshot",
    "LIVE_DRIVER_SCHEMA",
    "LiveCanaryError",
    "LoopbackCanaryClient",
    "RootEvidenceCollector",
    "SSEConversation",
    "SessionBoundPlan",
    "assemble_live_evidence",
    "prepare_session_bound_plan",
    "wait_for_fresh_owner_approval",
]
