"""Packaged fixed-target production wiring for Canonical Writer Phase-B.

This module is an execution edge, not a policy engine.  The approved plan,
approval, release, database, credential, journal, Cloud SQL instance and six
service units are all compile-time fixed.  The only injected dependency is the
already-authorized owner-side Cloud SQL boundary: moving that authority onto
the canary VM would silently expand its IAM power.

Readiness publication is stricter still.  Its issuer calls
``_collect_fixed_phase_b_readiness_mapping`` from inside the closure-sealed
foundation boundary; ordinary callers can neither supply evidence nor select
collectors, paths, services, hosts, users, or databases.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import socket
import ssl
import stat
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from gateway import canonical_writer_foundation as foundation
from gateway import canonical_writer_foundation_phase_b as phase_b
from gateway.canonical_canary_host_identity import (
    collect_dedicated_canary_host_identity_receipt,
)
from gateway.canonical_writer_boundary import harden_current_process_against_dumping
from gateway.canonical_writer_db import (
    CredentialSource,
    PostgresServerError,
    WriterDBConfig,
    _open_postgres_session,
    collect_managed_cloudsqladmin_hba_receipt,
)
from gateway.canonical_writer_planner import load_release_manifest


PHASE_B_AUTHORITY_ROOT = Path("/etc/muncho/canonical-writer-phase-b")
PHASE_B_PLAN_PATH = PHASE_B_AUTHORITY_ROOT / "plan.json"
PHASE_B_APPROVAL_PATH = PHASE_B_AUTHORITY_ROOT / "owner-approval.json"
PHASE_B_APPROVAL_SOURCE_PATH = (
    PHASE_B_AUTHORITY_ROOT / "owner-approval-source.json"
)
PHASE_B_RESUME_APPROVAL_ROOT = PHASE_B_AUTHORITY_ROOT / "resume-approvals"
PHASE_B_RESUME_APPROVAL_LOCK_PATH = PHASE_B_RESUME_APPROVAL_ROOT / ".lock"
PHASE_B_AUTHORITY_RECEIPT_PATH = PHASE_B_AUTHORITY_ROOT / "authority-receipt.json"
PHASE_B_JOURNAL_ROOT = Path(
    "/var/lib/muncho/canonical-writer-phase-b/journal"
)
PHASE_B_RUNTIME_RECEIPT_PATH = Path(
    "/var/lib/muncho/canonical-writer-phase-b/runtime-receipt.json"
)
PHASE_B_FULL_CANARY_ANCHOR_PATH = Path(
    "/var/lib/muncho/canonical-writer-phase-b/full-canary-anchor.json"
)

_ROOT_UID = 0
_ROOT_GID = 0
_ANONYMOUS_SECRET_DIRECTORY = Path("/run")
_AUTHORITY_MODE = 0o400
_AUTHORITY_APPEND_DIRECTORY_MODE = 0o700
_AUTHORITY_LOCK_MODE = 0o600
PHASE_B_MAX_RESUME_APPROVALS = 32
_MAX_AUTHORITY_BYTES = 8 * 1024 * 1024
_MAX_SYSTEMD_BYTES = 128 * 1024
_MAX_CLOUD_BYTES = 8 * 1024 * 1024
_SYSTEMCTL = "/usr/bin/systemctl"
_SERVICE_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "MainPID",
    "FragmentPath",
    "DropInPaths",
    "TriggeredBy",
    "Triggers",
    "NextElapseUSecRealtime",
)
_PIDLESS_SERVICE_UNITS = frozenset({
    "muncho-canonical-writer-export.timer",
})
_TEMPORARY_ADMIN_RE = re.compile(r"^muncho_canary_admin_[0-9a-f]{16}$")
_OPERATION_NAME_RE = re.compile(r"^[A-Za-z0-9._~-]{1,256}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AUTHORITY_RECEIPT_SCHEMA = "muncho-canonical-writer-phase-b-authority.v2"
_AUTHORITY_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "phase_b_plan_sha256",
        "phase_b_approval_sha256",
        "approval_source_sha256",
        "owner_subject_sha256",
        "owner_resume_public_key_ed25519_hex",
        "owner_resume_key_id",
        "owner_resume_public_key_file_sha256",
        "owner_resume_public_fingerprint",
        "approval_issued_at_unix",
        "approval_expires_at_unix",
        "release_sha",
        "coordinator_input_sha256",
        "activation_plan_sha256",
        "writer_activation_receipt_sha256",
        "activation_owner_approval_sha256",
        "activation_approval_issued_at_unix",
        "activation_approval_expires_at_unix",
        "native_observation_plan_sha256",
        "native_observation_receipt_sha256",
        "native_observation_approval_sha256",
        "native_approval_issued_at_unix",
        "native_approval_expires_at_unix",
        "external_iam_policy_sha256",
        "external_iam_receipt_sha256",
        "config_collector_receipt_sha256",
        "gateway_config_intent_sha256",
        "edge_config_intent_sha256",
        "fixture_intent_sha256",
        "host_identity_receipt_sha256",
        "authority_sources",
        "authority_sources_sha256",
        "authority_sha256",
    }
)
_PR_SET_NO_NEW_PRIVS = 38
_RUNTIME_HANDOFF_SCHEMA = "muncho-canonical-writer-phase-b-runtime-handoff.v1"
_RUNTIME_HANDOFF_FIELDS = frozenset(
    {
        "schema",
        "foundation",
        "foundation_sha256",
        "readiness_chain",
        "current_readiness_receipt_sha256",
        "published_at_unix",
        "handoff_sha256",
    }
)
_RUNTIME_HANDOFF_DIRECTORY_MODE = 0o750
_RUNTIME_HANDOFF_FILE_MODE = 0o440
_RUNTIME_HANDOFF_STAGING_NAME = ".runtime-receipt.staging"
_FULL_CANARY_ANCHOR_SCHEMA = "muncho-canonical-writer-phase-b-full-canary-anchor.v1"
_FULL_CANARY_ANCHOR_FIELDS = frozenset(
    {
        "phase_b_release_revision",
        "phase_b_plan_sha256",
        "phase_b_approval_sha256",
        "phase_b_terminal_receipt_sha256",
        "phase_b_foundation_generation_sha256",
        "phase_b_readiness_receipt_sha256",
        "phase_b_readiness_handoff_file_sha256",
        "phase_b_readiness_sequence",
    }
)
_INSTALLED_ANCHOR_FIELDS = frozenset(
    {"schema", "anchor", "anchor_sha256", "installed_at_unix"}
)


class PhaseBRuntimeError(RuntimeError):
    """Secret-free fixed runtime failure."""

    def __init__(self, code: str) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]{2,95}", code) is None:
            code = "canonical_writer_phase_b_runtime_failed"
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise PhaseBRuntimeError(code)


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
        raise PhaseBRuntimeError("phase_b_runtime_value_not_canonical") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _openssh_ed25519_fingerprint(public_hex: str) -> str:
    try:
        raw = bytes.fromhex(public_hex)
    except (TypeError, ValueError) as exc:
        raise PhaseBRuntimeError("phase_b_runtime_owner_key_invalid") from exc
    if len(raw) != 32:
        _fail("phase_b_runtime_owner_key_invalid")
    algorithm = b"ssh-ed25519"
    blob = (
        struct.pack(">I", len(algorithm))
        + algorithm
        + struct.pack(">I", len(raw))
        + raw
    )
    return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode(
        "ascii"
    ).rstrip("=")


def _require_root_linux() -> None:
    getter = getattr(os, "geteuid", None)
    if not callable(getter) or getter() != _ROOT_UID:
        _fail("phase_b_runtime_requires_root")
    if sys.platform != "linux":
        _fail("phase_b_runtime_requires_linux")


def _linux_prctl(option: int, argument: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    result = int(libc.prctl(option, argument, 0, 0, 0))
    if result < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number or errno.EPERM, "prctl failed")
    return result


def _harden_phase_b_process() -> None:
    """Harden before importing owner transport, reading tokens, or DB secrets."""

    try:
        harden_current_process_against_dumping()
        _linux_prctl(_PR_SET_NO_NEW_PRIVS, 1)
    except (OSError, RuntimeError) as exc:
        raise PhaseBRuntimeError("phase_b_runtime_process_hardening_failed") from exc


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _authority_directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_trusted_absolute_directory(path: Path) -> int:
    """Walk an absolute directory one component at a time with openat()."""

    if not path.is_absolute() or path.name in {"", ".", ".."}:
        _fail("phase_b_runtime_authority_path_invalid")
    descriptor: int | None = None
    try:
        descriptor = os.open("/", _authority_directory_flags())
        for component in path.parts[1:]:
            if component in {"", ".", ".."}:
                _fail("phase_b_runtime_authority_path_invalid")
            child = os.open(
                component,
                _authority_directory_flags(),
                dir_fd=descriptor,
            )
            status = os.fstat(child)
            if (
                not stat.S_ISDIR(status.st_mode)
                or status.st_uid not in {_ROOT_UID, 0}
                or stat.S_IMODE(status.st_mode) & 0o022
            ):
                os.close(child)
                _fail("phase_b_runtime_authority_parent_untrusted")
            os.close(descriptor)
            descriptor = child
    except PhaseBRuntimeError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise PhaseBRuntimeError(
            "phase_b_runtime_authority_file_unavailable"
        ) from exc
    assert descriptor is not None
    return descriptor


def _read_json_at(directory_fd: int, name: str) -> Mapping[str, Any]:
    if name in {"", ".", ".."} or "/" in name:
        _fail("phase_b_runtime_authority_path_invalid")
    descriptor: int | None = None
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != _ROOT_UID
            or before.st_gid != _ROOT_GID
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != _AUTHORITY_MODE
            or not 1 <= before.st_size <= _MAX_AUTHORITY_BYTES
        ):
            _fail("phase_b_runtime_authority_file_untrusted")
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        size = 0
        while size <= _MAX_AUTHORITY_BYTES:
            chunk = os.read(
                descriptor,
                min(64 * 1024, _MAX_AUTHORITY_BYTES + 1 - size),
            )
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        reachable = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except PhaseBRuntimeError:
        raise
    except OSError as exc:
        raise PhaseBRuntimeError(
            "phase_b_runtime_authority_file_unavailable"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or _stat_identity(before) != _stat_identity(opened)
        or _stat_identity(before) != _stat_identity(after)
        or _stat_identity(before) != _stat_identity(reachable)
    ):
        _fail("phase_b_runtime_authority_file_changed")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PhaseBRuntimeError("phase_b_runtime_authority_json_invalid") from exc
    if not isinstance(value, dict) or _canonical_bytes(value) + b"\n" != raw:
        _fail("phase_b_runtime_authority_json_not_canonical")
    return value


def _read_fixed_root_json(path: Path) -> Mapping[str, Any]:
    """Read one fixed authority leaf through a held, trusted directory fd."""

    if path not in {
        PHASE_B_PLAN_PATH,
        PHASE_B_APPROVAL_PATH,
        PHASE_B_APPROVAL_SOURCE_PATH,
        PHASE_B_AUTHORITY_RECEIPT_PATH,
    }:
        _fail("phase_b_runtime_authority_path_invalid")
    directory = _open_trusted_absolute_directory(path.parent)
    try:
        return _read_json_at(directory, path.name)
    finally:
        os.close(directory)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate key")
        value[key] = item
    return value


def _recollect_fixed_authority_provenance(
    *,
    plan: phase_b.PhaseBPlan,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Re-open durable sources and re-bind the signed owner lineage.

    The sidecar is only an index.  It is never accepted as evidence about its
    own inputs: the coordinator's fixed semantic loaders re-open the original
    activation/native/collector/config/fixture/host files and return their
    current bytes plus inode identities.  A byte-identical replacement is
    therefore a new generation, not silent continuity.  Owner identity and
    source are re-derived from the independently validated activation/native
    approval chain and must agree with the signed Phase-B plan.  No historical
    bootstrap credential or preapproval file participates in durable reload.
    """

    try:
        from gateway import canonical_full_canary_coordinator as coordinator

        coordinator_input = coordinator.load_coordinator_input()
        provenance = dict(coordinator._phase_b_authority_provenance(coordinator_input))
        approval_source_sha256 = provenance.get("approval_source_sha256")
        owner_subject_sha256 = provenance.get("owner_subject_sha256")
        for value in (approval_source_sha256, owner_subject_sha256):
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                _fail("phase_b_runtime_authority_source_invalid")
        if (
            approval_source_sha256 != coordinator.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
            or owner_subject_sha256 != plan.owner_subject_sha256
            or authority.get("approval_source_sha256") != approval_source_sha256
            or authority.get("owner_subject_sha256") != owner_subject_sha256
        ):
            _fail("phase_b_runtime_authority_source_invalid")
        placeholder_sources = provenance.get("authority_sources")
        historical_sources = authority.get("authority_sources")
        if (
            not isinstance(placeholder_sources, Mapping)
            or placeholder_sources.get("owner_resume_public_key") is not None
            or not isinstance(historical_sources, Mapping)
            or set(historical_sources)
            != set(placeholder_sources)
            or any(
                provenance.get(name) is not None
                for name in (
                    "owner_resume_public_key_ed25519_hex",
                    "owner_resume_key_id",
                    "owner_resume_public_key_file_sha256",
                    "owner_resume_public_fingerprint",
                )
            )
        ):
            _fail("phase_b_runtime_authority_source_invalid")
        owner_source = historical_sources.get("owner_resume_public_key")
        owner_fields = {
            "path",
            "file_sha256",
            "device",
            "inode",
            "uid",
            "gid",
            "mode",
            "size",
        }
        if (
            not isinstance(owner_source, Mapping)
            or set(owner_source) != owner_fields
            or owner_source.get("path")
            != coordinator.PHASE_B_OWNER_PUBLIC_KEY_PATH
            or owner_source.get("file_sha256")
            != plan.value["owner_resume_public_key_file_sha256"]
            or owner_source.get("uid")
            != coordinator.PHASE_B_OWNER_PUBLIC_KEY_UID
            or owner_source.get("gid")
            != coordinator.PHASE_B_OWNER_PUBLIC_KEY_GID
            or owner_source.get("mode") != "0600"
            or type(owner_source.get("device")) is not int
            or owner_source["device"] < 0
            or type(owner_source.get("inode")) is not int
            or owner_source["inode"] <= 0
            or type(owner_source.get("size")) is not int
            or not 1 <= owner_source["size"] <= 4096
        ):
            _fail("phase_b_runtime_authority_source_invalid")
        rebound_sources = dict(placeholder_sources)
        rebound_sources["owner_resume_public_key"] = copy.deepcopy(
            dict(owner_source)
        )
        provenance.update(
            {
                "owner_resume_public_key_ed25519_hex": plan.value[
                    "owner_resume_public_key_ed25519_hex"
                ],
                "owner_resume_key_id": plan.value["owner_resume_key_id"],
                "owner_resume_public_key_file_sha256": plan.value[
                    "owner_resume_public_key_file_sha256"
                ],
                "owner_resume_public_fingerprint": (
                    _openssh_ed25519_fingerprint(
                        plan.value["owner_resume_public_key_ed25519_hex"]
                    )
                ),
                "authority_sources": dict(sorted(rebound_sources.items())),
            }
        )
        provenance["authority_sources_sha256"] = _sha256_json(
            provenance["authority_sources"]
        )
        return provenance
    except PhaseBRuntimeError:
        raise
    except BaseException as exc:
        raise PhaseBRuntimeError(
            "phase_b_runtime_authority_source_invalid"
        ) from exc


def _load_fixed_approval_chain_for_plan(
    plan: phase_b.PhaseBPlan,
    *,
    allow_trailing_source: bool = False,
    lock_held: bool = False,
) -> tuple[tuple[phase_b.PhaseBApproval, ...], tuple[Mapping[str, Any], ...], Mapping[str, Any] | None]:
    initial_mapping = _read_fixed_root_json(PHASE_B_APPROVAL_PATH)
    if type(initial_mapping.get("issued_at_unix")) is not int:
        _fail("phase_b_runtime_approval_chain_invalid")
    initial = phase_b.PhaseBApproval.from_mapping(
        initial_mapping,
        plan=plan,
        now_unix=int(initial_mapping["issued_at_unix"]),
    )
    if initial.sequence != 0:
        _fail("phase_b_runtime_approval_chain_invalid")
    initial_source = phase_b.validate_phase_b_source_authentication(
        _read_fixed_root_json(PHASE_B_APPROVAL_SOURCE_PATH),
        plan=plan,
        approval=initial,
    )
    directory = _open_trusted_absolute_directory(PHASE_B_RESUME_APPROVAL_ROOT)
    lock_fd: int | None = None
    try:
        status = os.fstat(directory)
        if (
            status.st_uid != _ROOT_UID
            or status.st_gid != _ROOT_GID
            or stat.S_IMODE(status.st_mode) != _AUTHORITY_APPEND_DIRECTORY_MODE
        ):
            _fail("phase_b_runtime_resume_approval_directory_untrusted")
        try:
            lock_status = os.stat(
                ".lock", dir_fd=directory, follow_symlinks=False
            )
            lock_fd = os.open(
                ".lock",
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory,
            )
            opened_lock = os.fstat(lock_fd)
            if not lock_held:
                fcntl.flock(lock_fd, fcntl.LOCK_SH)
            names = sorted(os.listdir(directory))
        except OSError as exc:
            raise PhaseBRuntimeError(
                "phase_b_runtime_resume_approval_directory_untrusted"
            ) from exc
        if (
            not stat.S_ISREG(lock_status.st_mode)
            or _stat_identity(lock_status) != _stat_identity(opened_lock)
            or lock_status.st_uid != _ROOT_UID
            or lock_status.st_gid != _ROOT_GID
            or lock_status.st_nlink != 1
            or stat.S_IMODE(lock_status.st_mode) != _AUTHORITY_LOCK_MODE
        ):
            _fail("phase_b_runtime_resume_approval_lock_untrusted")
        payload_names = [name for name in names if name != ".lock"]
        if len(payload_names) > PHASE_B_MAX_RESUME_APPROVALS * 2:
            _fail("phase_b_runtime_approval_chain_too_long")
        if any(
            re.fullmatch(r"[0-9]{8}\.(?:approval|source)\.json", name)
            is None
            for name in payload_names
        ):
            _fail("phase_b_runtime_approval_chain_forked")
        by_sequence: dict[int, set[str]] = {}
        for name in payload_names:
            sequence_text, kind, _suffix = name.split(".")
            sequence = int(sequence_text)
            if sequence <= 0:
                _fail("phase_b_runtime_approval_chain_forked")
            if sequence > PHASE_B_MAX_RESUME_APPROVALS:
                _fail("phase_b_runtime_approval_chain_too_long")
            by_sequence.setdefault(sequence, set()).add(kind)
        approvals: list[phase_b.PhaseBApproval] = [initial]
        sources: list[Mapping[str, Any]] = [initial_source]
        trailing_source: Mapping[str, Any] | None = None
        for sequence in sorted(by_sequence):
            if sequence != len(approvals):
                _fail("phase_b_runtime_approval_chain_sequence_missing")
            kinds = by_sequence[sequence]
            if kinds == {"source"} and allow_trailing_source and sequence == max(by_sequence):
                trailing_source = _read_json_at(
                    directory, f"{sequence:08d}.source.json"
                )
                break
            if kinds != {"approval", "source"}:
                _fail("phase_b_runtime_approval_chain_incomplete")
            approval_mapping = _read_json_at(
                directory, f"{sequence:08d}.approval.json"
            )
            candidate_values = [
                item.to_mapping() for item in approvals
            ] + [approval_mapping]
            approvals = list(
                phase_b.validate_phase_b_approval_chain(
                    candidate_values,
                    plan=plan,
                )
            )
            source = phase_b.validate_phase_b_source_authentication(
                _read_json_at(directory, f"{sequence:08d}.source.json"),
                plan=plan,
                approval=approvals[-1],
            )
            sources.append(source)
        return tuple(approvals), tuple(sources), trailing_source
    finally:
        if lock_fd is not None:
            if not lock_held:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(lock_fd)
        os.close(directory)


def load_fixed_phase_b_approval_chain(
    *,
    require_fresh_head: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    """Load the exact signed, source-authenticated immutable approval chain."""

    _require_root_linux()
    _harden_phase_b_process()
    plan = phase_b.PhaseBPlan.from_mapping(_read_fixed_root_json(PHASE_B_PLAN_PATH))
    approvals, _sources, trailing = _load_fixed_approval_chain_for_plan(plan)
    if trailing is not None:
        _fail("phase_b_runtime_approval_chain_incomplete")
    validated = phase_b.validate_phase_b_approval_chain(
        [item.to_mapping() for item in approvals],
        plan=plan,
        now_unix=int(time.time()),
        require_fresh_head=require_fresh_head,
    )
    return tuple(item.to_mapping() for item in validated)


def load_fixed_phase_b_authority(
    *,
    _approval_lock_held: bool = False,
    _allow_trailing_source: bool = False,
) -> tuple[
    phase_b.PhaseBPlan,
    tuple[Mapping[str, Any], ...],
    phase_b.AppendOnlyPhaseBJournal,
]:
    """Load the exact root-owned plan, approval and fixed durable journal."""

    _require_root_linux()
    if {
        PHASE_B_PLAN_PATH.parent,
        PHASE_B_APPROVAL_PATH.parent,
        PHASE_B_APPROVAL_SOURCE_PATH.parent,
        PHASE_B_AUTHORITY_RECEIPT_PATH.parent,
    } != {PHASE_B_AUTHORITY_ROOT}:
        _fail("phase_b_runtime_authority_path_invalid")
    directory = _open_trusted_absolute_directory(PHASE_B_AUTHORITY_ROOT)
    try:
        plan_mapping = _read_json_at(directory, PHASE_B_PLAN_PATH.name)
        authority = _read_json_at(directory, PHASE_B_AUTHORITY_RECEIPT_PATH.name)
    finally:
        os.close(directory)
    plan = phase_b.PhaseBPlan.from_mapping(plan_mapping)
    approvals, _sources, trailing_source = _load_fixed_approval_chain_for_plan(
        plan,
        allow_trailing_source=_allow_trailing_source,
        lock_held=_approval_lock_held,
    )
    if trailing_source is not None and not _allow_trailing_source:
        _fail("phase_b_runtime_approval_chain_incomplete")
    approved = approvals[0]
    if not isinstance(authority, Mapping) or set(authority) != _AUTHORITY_RECEIPT_FIELDS:
        _fail("phase_b_runtime_authority_receipt_invalid")
    unsigned_authority = {
        name: item for name, item in authority.items() if name != "authority_sha256"
    }
    digest_fields = _AUTHORITY_RECEIPT_FIELDS - {
        "schema",
        "approval_issued_at_unix",
        "approval_expires_at_unix",
        "activation_approval_issued_at_unix",
        "activation_approval_expires_at_unix",
        "native_approval_issued_at_unix",
        "native_approval_expires_at_unix",
        "release_sha",
        "owner_resume_public_fingerprint",
        "authority_sources",
        "authority_sha256",
    }
    time_fields = {
        "approval_issued_at_unix",
        "approval_expires_at_unix",
        "activation_approval_issued_at_unix",
        "activation_approval_expires_at_unix",
        "native_approval_issued_at_unix",
        "native_approval_expires_at_unix",
    }
    if (
        authority["schema"] != _AUTHORITY_RECEIPT_SCHEMA
        or authority["phase_b_plan_sha256"] != plan.sha256
        or authority["phase_b_approval_sha256"] != approved.sha256
        or authority["approval_source_sha256"]
        != approved.value["approval_source_sha256"]
        or authority["owner_subject_sha256"] != plan.owner_subject_sha256
        or authority["owner_resume_public_key_ed25519_hex"]
        != plan.value["owner_resume_public_key_ed25519_hex"]
        or authority["owner_resume_key_id"]
        != plan.value["owner_resume_key_id"]
        or authority["owner_resume_public_key_file_sha256"]
        != plan.value["owner_resume_public_key_file_sha256"]
        or authority["owner_resume_public_fingerprint"]
        != _openssh_ed25519_fingerprint(
            plan.value["owner_resume_public_key_ed25519_hex"]
        )
        or authority["approval_issued_at_unix"] != approved.value["issued_at_unix"]
        or authority["approval_expires_at_unix"] != approved.value["expires_at_unix"]
        or authority["release_sha"] != plan.revision
        or any(
            type(authority[name]) is not int or authority[name] < 0
            for name in time_fields
        )
        or any(
            not isinstance(authority[name], str)
            or _SHA256_RE.fullmatch(authority[name]) is None
            for name in digest_fields
        )
        or not isinstance(authority["authority_sources"], Mapping)
        or authority["authority_sources_sha256"]
        != _sha256_json(authority["authority_sources"])
        or not isinstance(authority["authority_sha256"], str)
        or authority["authority_sha256"] != _sha256_json(unsigned_authority)
    ):
        _fail("phase_b_runtime_authority_receipt_invalid")
    provenance = _recollect_fixed_authority_provenance(
        plan=plan,
        authority=authority,
    )
    provenance_fields = _AUTHORITY_RECEIPT_FIELDS - {
        "schema",
        "phase_b_plan_sha256",
        "phase_b_approval_sha256",
        "approval_issued_at_unix",
        "approval_expires_at_unix",
        "authority_sha256",
    }
    if set(provenance) != provenance_fields or any(
        authority[name] != provenance[name] for name in provenance_fields
    ):
        _fail("phase_b_runtime_authority_source_drifted")
    # Live execution validates freshness inside execute_approved_phase_b.
    # Durable reload deliberately evaluates the same immutable approval at the
    # authenticated terminal time, so normal expiry cannot erase a completed
    # foundation generation.
    return (
        plan,
        tuple(item.to_mapping() for item in approvals),
        phase_b.AppendOnlyPhaseBJournal(PHASE_B_JOURNAL_ROOT),
    )


@contextlib.contextmanager
def _locked_resume_approval_directory(
    *,
    exclusive: bool = True,
) -> Iterator[int]:
    directory = _open_trusted_absolute_directory(PHASE_B_RESUME_APPROVAL_ROOT)
    lock_fd: int | None = None
    try:
        status = os.fstat(directory)
        if (
            status.st_uid != _ROOT_UID
            or status.st_gid != _ROOT_GID
            or stat.S_IMODE(status.st_mode) != _AUTHORITY_APPEND_DIRECTORY_MODE
        ):
            _fail("phase_b_runtime_resume_approval_directory_untrusted")
        lock_fd = os.open(
            ".lock",
            os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory,
        )
        lock_status = os.fstat(lock_fd)
        reachable = os.stat(".lock", dir_fd=directory, follow_symlinks=False)
        if (
            _stat_identity(lock_status) != _stat_identity(reachable)
            or not stat.S_ISREG(lock_status.st_mode)
            or lock_status.st_uid != _ROOT_UID
            or lock_status.st_gid != _ROOT_GID
            or lock_status.st_nlink != 1
            or stat.S_IMODE(lock_status.st_mode) != _AUTHORITY_LOCK_MODE
        ):
            _fail("phase_b_runtime_resume_approval_lock_untrusted")
        fcntl.flock(lock_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield directory
    except PhaseBRuntimeError:
        raise
    except OSError as exc:
        raise PhaseBRuntimeError(
            "phase_b_runtime_resume_approval_lock_failed"
        ) from exc
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(directory)


def _write_authority_json_o_excl_at(
    directory_fd: int,
    name: str,
    value: Mapping[str, Any],
) -> None:
    if re.fullmatch(r"[0-9]{8}\.(?:approval|source)\.json", name) is None:
        _fail("phase_b_runtime_authority_path_invalid")
    payload = _canonical_bytes(value) + b"\n"
    if not 1 <= len(payload) <= _MAX_AUTHORITY_BYTES:
        _fail("phase_b_runtime_authority_file_invalid")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            _AUTHORITY_MODE,
            dir_fd=directory_fd,
        )
        os.fchown(descriptor, _ROOT_UID, _ROOT_GID)
        os.fchmod(descriptor, _AUTHORITY_MODE)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail("phase_b_runtime_authority_write_stalled")
            offset += written
        os.fsync(descriptor)
    except FileExistsError:
        _fail("phase_b_runtime_approval_chain_old_head")
    except PhaseBRuntimeError:
        raise
    except OSError as exc:
        raise PhaseBRuntimeError(
            "phase_b_runtime_authority_write_failed"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    os.fsync(directory_fd)
    if _read_json_at(directory_fd, name) != value:
        _fail("phase_b_runtime_authority_readback_failed")


def inspect_fixed_phase_b_incomplete_head() -> Mapping[str, Any]:
    """Return the exact current approval/journal head without mutation."""

    _require_root_linux()
    _harden_phase_b_process()
    with _locked_resume_approval_directory(exclusive=False):
        plan, approval_values, journal = load_fixed_phase_b_authority(
            _approval_lock_held=True,
            _allow_trailing_source=True,
        )
        approvals = phase_b.validate_phase_b_approval_chain(
            approval_values,
            plan=plan,
        )
        _same, _sources, trailing = _load_fixed_approval_chain_for_plan(
            plan,
            allow_trailing_source=True,
            lock_held=True,
        )
        entries = journal.load(plan)
    head = approvals[-1]
    pending_source = None
    if trailing is not None:
        pending_source = phase_b.validate_phase_b_pending_source_authentication(
            trailing,
            plan=plan,
            expected_sequence=head.sequence + 1,
            expected_previous_approval_sha256=head.sha256,
            expected_approval_source_sha256=str(
                head.value["approval_source_sha256"]
            ),
        )
    now = int(time.time())
    terminal = bool(entries and entries[-1].event == "terminal")
    fresh = (
        head.value["issued_at_unix"] <= now < head.value["expires_at_unix"]
    )
    unsigned = {
        "schema": "muncho-canonical-writer-phase-b-incomplete-head.v1",
        "plan_sha256": plan.sha256,
        "intent_sha256": plan.value["intent_sha256"],
        "owner_subject_sha256": plan.owner_subject_sha256,
        "approval_source_sha256": head.value["approval_source_sha256"],
        "approval_sequence": head.sequence,
        "approval_sha256": head.sha256,
        "approval_expires_at_unix": head.value["expires_at_unix"],
        "pending_source_sequence": (
            None if pending_source is None else pending_source["sequence"]
        ),
        "pending_source_authentication_sha256": (
            None
            if pending_source is None
            else pending_source["receipt_sha256"]
        ),
        "journal_entry_count": len(entries),
        "journal_head_sha256": entries[-1].sha256 if entries else None,
        "journal_head_event": entries[-1].event if entries else None,
        "journal_head_recorded_at_unix": (
            entries[-1].value["recorded_at_unix"] if entries else None
        ),
        "terminal": terminal,
        "incomplete_state": (
            "terminal"
            if terminal
            else "journal_incomplete"
            if entries
            else "authority_published_no_intent"
        ),
        "resume_eligible": not terminal,
        "fresh_head": fresh,
        "requires_reapproval": not terminal and (
            not fresh or pending_source is not None
        ),
        "mutation_authorized": not terminal and fresh and pending_source is None,
        "inspected_at_unix": now,
    }
    return {**unsigned, "inspection_sha256": _sha256_json(unsigned)}


def install_fixed_phase_b_resume_approval(
    approval_mapping: Mapping[str, Any],
    source_authentication_mapping: Mapping[str, Any],
    *,
    expected_previous_approval_sha256: str,
) -> Mapping[str, Any]:
    """Append exactly the next owner-signed same-plan resume approval."""

    _require_root_linux()
    _harden_phase_b_process()
    if _SHA256_RE.fullmatch(str(expected_previous_approval_sha256)) is None:
        _fail("phase_b_runtime_approval_chain_old_head")
    plan = phase_b.PhaseBPlan.from_mapping(_read_fixed_root_json(PHASE_B_PLAN_PATH))
    journal = phase_b.AppendOnlyPhaseBJournal(PHASE_B_JOURNAL_ROOT)
    now = int(time.time())
    with _locked_resume_approval_directory() as directory:
        approvals, sources, trailing = _load_fixed_approval_chain_for_plan(
            plan,
            allow_trailing_source=True,
            lock_held=True,
        )
        entries = journal.load(plan)
        if entries and entries[-1].event == "terminal":
            _fail("phase_b_runtime_resume_after_terminal_forbidden")
        head = approvals[-1]
        if head.sha256 != expected_previous_approval_sha256:
            if (
                head.value.get("previous_approval_sha256")
                == expected_previous_approval_sha256
                and head.to_mapping() == approval_mapping
                and sources[-1] == source_authentication_mapping
            ):
                phase_b.validate_phase_b_approval_chain(
                    [item.to_mapping() for item in approvals],
                    plan=plan,
                )
                if head.value["issued_at_unix"] > now:
                    _fail("phase_b_runtime_approval_chain_not_yet_valid")
                source = phase_b.validate_phase_b_source_authentication(
                    source_authentication_mapping,
                    plan=plan,
                    approval=head,
                )
                fresh = (
                    head.value["issued_at_unix"]
                    <= now
                    < head.value["expires_at_unix"]
                )
                unsigned = {
                    "schema": "muncho-canonical-writer-phase-b-resume-approval-install.v1",
                    "plan_sha256": plan.sha256,
                    "approval_sequence": head.sequence,
                    "previous_approval_sha256": expected_previous_approval_sha256,
                    "approval_sha256": head.sha256,
                    "source_authentication_sha256": source["receipt_sha256"],
                    "journal_head_sha256": entries[-1].sha256 if entries else None,
                    "already_installed": True,
                    "completed_trailing_source_only_residue": False,
                    "approval_fresh": fresh,
                    "mutation_authorized": fresh,
                    "requires_fresh_successor": not fresh,
                    "installed_at_unix": now,
                }
                return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
            _fail("phase_b_runtime_approval_chain_old_head")
        candidate_chain = phase_b.validate_phase_b_approval_chain(
            [item.to_mapping() for item in approvals] + [approval_mapping],
            plan=plan,
        )
        candidate = candidate_chain[-1]
        if candidate.value["issued_at_unix"] > now:
            _fail("phase_b_runtime_approval_chain_not_yet_valid")
        fresh = (
            candidate.value["issued_at_unix"]
            <= now
            < candidate.value["expires_at_unix"]
        )
        # A signed approval may expire after its source half was durably
        # published but before the matching approval half reached disk.  The
        # exact approval is then append-only historical repair, not mutation
        # authority.  A wholly new pair still requires a fresh head.
        if trailing is None and not fresh:
            _fail("phase_b_runtime_approval_expired")
        source = phase_b.validate_phase_b_source_authentication(
            source_authentication_mapping,
            plan=plan,
            approval=candidate,
        )
        if trailing is not None and trailing != source:
            _fail("phase_b_runtime_approval_chain_forked")
        source_name = f"{candidate.sequence:08d}.source.json"
        approval_name = f"{candidate.sequence:08d}.approval.json"
        if trailing is None:
            _write_authority_json_o_excl_at(directory, source_name, source)
        _write_authority_json_o_excl_at(
            directory, approval_name, candidate.to_mapping()
        )
        reloaded, _reloaded_sources, residue = (
            _load_fixed_approval_chain_for_plan(plan, lock_held=True)
        )
        if residue is not None or reloaded[-1].sha256 != candidate.sha256:
            _fail("phase_b_runtime_approval_chain_install_failed")
        unsigned = {
            "schema": "muncho-canonical-writer-phase-b-resume-approval-install.v1",
            "plan_sha256": plan.sha256,
            "approval_sequence": candidate.sequence,
            "previous_approval_sha256": head.sha256,
            "approval_sha256": candidate.sha256,
            "source_authentication_sha256": source["receipt_sha256"],
            "journal_head_sha256": entries[-1].sha256 if entries else None,
            "already_installed": False,
            "completed_trailing_source_only_residue": trailing is not None,
            "approval_fresh": fresh,
            "mutation_authorized": fresh,
            "requires_fresh_successor": not fresh,
            "installed_at_unix": now,
        }
        return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _validate_anonymous_secret_descriptor(descriptor: int) -> None:
    try:
        os.set_inheritable(descriptor, False)
        os.fchmod(descriptor, 0o400)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != _ROOT_UID
            or opened.st_gid != _ROOT_GID
            or opened.st_nlink != 0
            or stat.S_IMODE(opened.st_mode) != 0o400
            or os.get_inheritable(descriptor)
        ):
            _fail("phase_b_runtime_anonymous_secret_identity_invalid")
    except PhaseBRuntimeError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise PhaseBRuntimeError(
            "phase_b_runtime_anonymous_secret_identity_invalid"
        ) from exc


def _open_anonymous_secret_descriptor() -> int:
    """Return one root-owned, unlinked, close-on-exec anonymous inode."""

    creator = getattr(os, "memfd_create", None)
    if callable(creator):
        descriptor: int | None = None
        succeeded = False
        try:
            descriptor = creator(
                "muncho-phase-b-credential",
                getattr(os, "MFD_CLOEXEC", 0),
            )
            _validate_anonymous_secret_descriptor(descriptor)
            succeeded = True
            return descriptor
        except PhaseBRuntimeError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise PhaseBRuntimeError(
                "phase_b_runtime_anonymous_secret_unavailable"
            ) from exc
        finally:
            if descriptor is not None and not succeeded:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    required_flags = (
        getattr(os, "O_CLOEXEC", 0),
        getattr(os, "O_DIRECTORY", 0),
        getattr(os, "O_EXCL", 0),
        getattr(os, "O_NOFOLLOW", 0),
        getattr(os, "O_TMPFILE", 0),
    )
    if any(type(flag) is not int or flag == 0 for flag in required_flags):
        _fail("phase_b_runtime_anonymous_secret_unavailable")

    directory_descriptor: int | None = None
    descriptor: int | None = None
    succeeded = False
    try:
        try:
            directory_descriptor = _open_trusted_absolute_directory(
                _ANONYMOUS_SECRET_DIRECTORY
            )
        except PhaseBRuntimeError as exc:
            raise PhaseBRuntimeError(
                "phase_b_runtime_anonymous_secret_directory_invalid"
            ) from exc
        directory = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(directory.st_mode)
            or directory.st_uid != _ROOT_UID
            or directory.st_gid != _ROOT_GID
            or stat.S_IMODE(directory.st_mode) & 0o022
            or os.get_inheritable(directory_descriptor)
        ):
            _fail("phase_b_runtime_anonymous_secret_directory_invalid")
        descriptor = os.open(
            ".",
            os.O_RDWR | os.O_CLOEXEC | os.O_EXCL | os.O_TMPFILE,
            0o400,
            dir_fd=directory_descriptor,
        )
        _validate_anonymous_secret_descriptor(descriptor)
        succeeded = True
        return descriptor
    except PhaseBRuntimeError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise PhaseBRuntimeError(
            "phase_b_runtime_anonymous_secret_unavailable"
        ) from exc
    finally:
        if directory_descriptor is not None:
            try:
                os.close(directory_descriptor)
            except OSError:
                pass
        if descriptor is not None and not succeeded:
            try:
                os.close(descriptor)
            except OSError:
                pass


@contextlib.contextmanager
def _secret_descriptor(secret: bytearray) -> Iterator[int]:
    if not isinstance(secret, bytearray) or not 24 <= len(secret) <= 4096:
        _fail("phase_b_runtime_secret_invalid")
    descriptor = _open_anonymous_secret_descriptor()
    try:
        try:
            offset = 0
            while offset < len(secret):
                written = os.write(descriptor, secret[offset:])
                if written <= 0:
                    _fail("phase_b_runtime_secret_transport_failed")
                offset += written
            os.fsync(descriptor)
        except PhaseBRuntimeError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise PhaseBRuntimeError(
                "phase_b_runtime_secret_transport_failed"
            ) from exc
        yield descriptor
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _database_config(
    user: str,
    *,
    credential: CredentialSource,
    application_name: str,
) -> WriterDBConfig:
    return WriterDBConfig(
        host=foundation.SQL_HOST,
        tls_server_name=foundation.SQL_TLS_SERVER_NAME,
        port=foundation.SQL_PORT,
        database=foundation.SQL_DATABASE,
        user=user,
        ca_file=foundation.DATABASE_CA_PATH,
        credential=credential,
        connect_timeout_seconds=5.0,
        io_timeout_seconds=10.0,
        application_name=application_name,
    )


def _writer_session() -> Any:
    return _open_postgres_session(foundation._fixed_writer_config())


def _open_secret_session(user: str, secret: bytearray, *, purpose: str) -> Any:
    with _secret_descriptor(secret) as descriptor:
        config = _database_config(
            user,
            credential=CredentialSource(
                fd=descriptor,
                expected_uid=_ROOT_UID,
                expected_gid=_ROOT_GID,
                allowed_modes=frozenset({0o400}),
            ),
            application_name=purpose,
        )
        return _open_postgres_session(config)


def _query_json(
    session: Any,
    sql: str,
    *,
    expected_column: str,
) -> Mapping[str, Any]:
    try:
        result = session.query(sql, maximum_rows=1)
    except BaseException as exc:
        raise PhaseBRuntimeError("phase_b_runtime_database_query_failed") from exc
    if (
        tuple(getattr(result, "columns", ())) != (expected_column,)
        or len(tuple(getattr(result, "rows", ()))) != 1
        or len(tuple(result.rows[0])) != 1
        or not isinstance(result.rows[0][0], str)
    ):
        _fail("phase_b_runtime_database_result_invalid")
    try:
        value = json.loads(
            result.rows[0][0],
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise PhaseBRuntimeError("phase_b_runtime_database_json_invalid") from exc
    if not isinstance(value, dict):
        _fail("phase_b_runtime_database_json_invalid")
    return value


def _database_identity(session: Any) -> Mapping[str, Any]:
    result = session.query(
        """
SELECT SESSION_USER::text AS session_user,
       CURRENT_USER::text AS current_user,
       current_database()::text AS database_name,
       pg_catalog.current_setting('server_version_num')::int::text AS version_num,
       pg_catalog.pg_get_userbyid(database.datdba)::text AS database_owner,
       pg_catalog.pg_backend_pid()::text AS backend_pid,
       pg_catalog.pg_postmaster_start_time()::text AS postmaster_started
  FROM pg_catalog.pg_database AS database
 WHERE database.datname = current_database()
""",
        maximum_rows=1,
    )
    if (
        tuple(result.columns)
        != (
            "session_user",
            "current_user",
            "database_name",
            "version_num",
            "database_owner",
            "backend_pid",
            "postmaster_started",
        )
        or len(result.rows) != 1
        or any(not isinstance(item, str) for item in result.rows[0])
    ):
        _fail("phase_b_runtime_database_identity_invalid")
    (
        session_user,
        current_user,
        database_name,
        version_text,
        database_owner,
        backend_pid,
        postmaster_started,
    ) = result.rows[0]
    try:
        version = int(version_text)
    except (TypeError, ValueError) as exc:
        raise PhaseBRuntimeError(
            "phase_b_runtime_database_identity_invalid"
        ) from exc
    peer = str(getattr(session, "tls_peer_certificate_sha256", ""))
    identity = _sha256_json(
        {
            "backend_pid": backend_pid,
            "database": database_name,
            "postmaster_started": postmaster_started,
            "session_user": session_user,
            "tls_peer_certificate_sha256": peer,
        }
    )
    return {
        "project": foundation.PROJECT,
        "instance": foundation.SQL_INSTANCE,
        "host": foundation.SQL_HOST,
        "port": foundation.SQL_PORT,
        "database": database_name,
        "database_owner": database_owner,
        "postgres_version_num": version,
        "tls_server_name": foundation.SQL_TLS_SERVER_NAME,
        "tls_peer_certificate_sha256": peer,
        "session_user": session_user,
        "current_user": current_user,
        "session_identity_sha256": identity,
    }


def _phase_b_artifacts(revision: str) -> Mapping[str, foundation.SealedSQLArtifact]:
    artifacts = foundation._load_sealed_artifacts(revision)
    if "phase_b_preflight" not in artifacts or "phase_b_role" not in artifacts:
        _fail("phase_b_runtime_release_artifacts_missing")
    return artifacts


def _database_preflight(
    session: Any,
    artifacts: Mapping[str, foundation.SealedSQLArtifact],
) -> Mapping[str, Any]:
    artifact = artifacts["phase_b_preflight"]
    sql = artifact.payload.decode("utf-8", errors="strict")
    return _query_json(
        session,
        sql,
        expected_column="phase_b_database_preflight",
    )


def _split_systemd_list(value: str) -> list[str]:
    if not value:
        return []
    return sorted(item for item in value.split() if item)


def _collect_one_service(unit: str) -> Mapping[str, Any]:
    if unit not in phase_b.SERVICE_UNITS:
        _fail("phase_b_runtime_service_not_fixed")
    command = [
        _SYSTEMCTL,
        "show",
        "--no-pager",
        *(f"--property={name}" for name in _SERVICE_PROPERTIES),
        "--",
        unit,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=5,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise PhaseBRuntimeError("phase_b_runtime_systemd_failed") from exc
    if (
        completed.returncode != 0
        or completed.stderr
        or len(completed.stdout.encode("utf-8")) > _MAX_SYSTEMD_BYTES
    ):
        _fail("phase_b_runtime_systemd_failed")
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator != "=" or name not in _SERVICE_PROPERTIES or name in values:
            _fail("phase_b_runtime_systemd_invalid")
        values[name] = value
    # Empty array/time properties may be omitted by some supported systemd
    # serializers.  Only these exact empty values are normalized.
    for optional in ("TriggeredBy", "Triggers", "NextElapseUSecRealtime"):
        values.setdefault(optional, "")
    missing = set(_SERVICE_PROPERTIES) - set(values)
    if missing == {"MainPID"} and unit in _PIDLESS_SERVICE_UNITS:
        # Timer units do not own a service process, so supported systemd
        # versions omit MainPID even for an exact stopped observation.
        values["MainPID"] = "0"
    elif missing:
        _fail("phase_b_runtime_systemd_invalid")
    try:
        main_pid = int(values["MainPID"])
    except ValueError as exc:
        raise PhaseBRuntimeError("phase_b_runtime_systemd_invalid") from exc
    load_state = values["LoadState"]
    unit_file_state = values["UnitFileState"] or (
        "not-found" if load_state == "not-found" else ""
    )
    fragment = values["FragmentPath"] or None
    next_elapse_raw = values["NextElapseUSecRealtime"]
    next_elapse: int | None = None
    if next_elapse_raw and next_elapse_raw not in {"0", "n/a"}:
        try:
            next_elapse = int(next_elapse_raw)
        except ValueError as exc:
            raise PhaseBRuntimeError("phase_b_runtime_systemd_invalid") from exc
    return {
        "name": unit,
        "load_state": load_state,
        "active_state": values["ActiveState"],
        "sub_state": values["SubState"],
        "unit_file_state": unit_file_state,
        "main_pid": main_pid,
        "fragment_path": fragment,
        "drop_in_paths": _split_systemd_list(values["DropInPaths"]),
        "triggered_by": _split_systemd_list(values["TriggeredBy"]),
        "triggers": _split_systemd_list(values["Triggers"]),
        "next_elapse_unix_usec": next_elapse,
    }


def _collect_services(revision: str, observed_at_unix: int) -> Mapping[str, Any]:
    rows = [_collect_one_service(unit) for unit in phase_b.SERVICE_UNITS]
    unsigned = {
        "schema": "muncho-canonical-writer-phase-b-services-stopped.v1",
        "release_revision": revision,
        "services": rows,
        "services_stopped_and_disabled": True,
        "observed_at_unix": observed_at_unix,
    }
    return {**unsigned, "attestation_sha256": _sha256_json(unsigned)}


class _PhaseBAdminSession:
    """Authenticated TLS session plus the one sealed Phase-B SQL method."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self.username = str(session.username)
        self.tls_peer_certificate_sha256 = str(
            session.tls_peer_certificate_sha256
        )

    def query(self, sql: str, *, maximum_rows: int) -> Any:
        return self._session.query(sql, maximum_rows=maximum_rows)

    def close(self) -> None:
        self._session.close()

    def execute_phase_b_role_artifact(
        self,
        artifact: foundation.SealedSQLArtifact,
        *,
        bindings: Mapping[str, str],
    ) -> Mapping[str, Any]:
        expected = {
            "muncho.canonical_writer_phase_b_release_revision",
            "muncho.canonical_writer_phase_b_role_artifact_sha256",
            "muncho.canonical_writer_phase_b_initial_observation_sha256",
            "muncho.canonical_writer_phase_b_approved_plan_sha256",
        }
        if set(bindings) != expected or any(
            not isinstance(item, str)
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", item) is None
            for item in bindings.values()
        ):
            _fail("phase_b_runtime_role_binding_invalid")
        set_sql = "\n".join(
            f"SET {name} = '{value}';" for name, value in sorted(bindings.items())
        )
        reset_sql = "ROLLBACK;\n" + "\n".join(
            f"RESET {name};" for name in sorted(bindings)
        )
        try:
            result = self.query(set_sql, maximum_rows=0)
            if not str(result.command_tag).upper().startswith("SET"):
                _fail("phase_b_runtime_role_binding_failed")
            receipt = _query_json(
                self,
                artifact.payload.decode("utf-8", errors="strict"),
                expected_column="phase_b_role_receipt",
            )
        except BaseException as primary:
            try:
                self.query(reset_sql, maximum_rows=0)
            except BaseException as cleanup:
                raise ExceptionGroup(
                    "phase-b role execution and cleanup failed",
                    [primary, cleanup],
                ) from None
            raise
        result = self.query(reset_sql, maximum_rows=0)
        if not str(result.command_tag).upper().startswith("RESET"):
            _fail("phase_b_runtime_role_binding_cleanup_failed")
        return receipt


class _BootstrapSelfDisable:
    def disable_and_prove_denied(
        self,
        *,
        plan: phase_b.PhaseBPlan,
        provisional_password: bytearray,
        authority_receipt: Mapping[str, Any],
        hba_rejection_receipt: Mapping[str, Any],
        statement: str,
    ) -> Mapping[str, Any]:
        if statement != phase_b.SELF_DISABLE_SQL:
            _fail("phase_b_runtime_self_disable_statement_invalid")
        session = _open_secret_session(
            foundation.CANARY_BOOTSTRAP_LOGIN,
            provisional_password,
            purpose="muncho-phase-b-bootstrap-self-disable",
        )
        peer = str(session.tls_peer_certificate_sha256)
        try:
            result = session.query(statement, maximum_rows=0)
            if str(result.command_tag).upper() != "ALTER ROLE" or result.rows:
                _fail("phase_b_runtime_self_disable_unconfirmed")
        finally:
            session.close()
        denial_state = ""
        denied = False
        try:
            fresh = _open_secret_session(
                foundation.CANARY_BOOTSTRAP_LOGIN,
                provisional_password,
                purpose="muncho-phase-b-bootstrap-denial",
            )
        except PostgresServerError as exc:
            denial_state = exc.sqlstate
            denied = exc.sqlstate in {"28000", "28P01"}
        else:
            fresh.close()
        if not denied:
            _fail("phase_b_runtime_self_disable_denial_missing")
        unsigned = {
            "schema": phase_b.PHASE_B_SELF_DISABLE_SCHEMA,
            "plan_sha256": plan.sha256,
            "bootstrap_authority_receipt_sha256": authority_receipt[
                "receipt_sha256"
            ],
            "hba_rejection_receipt_sha256": hba_rejection_receipt[
                "receipt_sha256"
            ],
            "user": foundation.CANARY_BOOTSTRAP_LOGIN,
            "database": foundation.SQL_DATABASE,
            "tls_peer_certificate_sha256": peer,
            "authenticated_as_self": True,
            "statement_sha256": hashlib.sha256(statement.encode("ascii")).hexdigest(),
            "command_tag": "ALTER ROLE",
            "password_disabled": True,
            "login_remains_true": True,
            "fresh_denial_connection": True,
            "denial_sqlstate": denial_state,
            "password_or_digest_recorded": False,
        }
        return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _hba_collector(
    plan: phase_b.PhaseBPlan,
    secret: bytearray,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    with _secret_descriptor(secret) as descriptor:
        config = _database_config(
            foundation.CANARY_BOOTSTRAP_LOGIN,
            credential=CredentialSource(
                fd=descriptor,
                expected_uid=_ROOT_UID,
                expected_gid=_ROOT_GID,
                allowed_modes=frozenset({0o400}),
            ),
            application_name="muncho-phase-b-managed-hba",
        )
        receipt = collect_managed_cloudsqladmin_hba_receipt(config)
    unsigned = {
        "schema": phase_b.PHASE_B_HBA_RECEIPT_SCHEMA,
        "plan_sha256": plan.sha256,
        "bootstrap_authority_receipt_sha256": authority["receipt_sha256"],
        "host": receipt.host,
        "port": receipt.port,
        "tls_server_name": receipt.tls_server_name,
        "tls_peer_certificate_sha256": receipt.server_certificate_sha256,
        "user": receipt.user,
        "database": receipt.database,
        "rejected": True,
        "sqlstate": receipt.sqlstate,
        "observed_at_unix": receipt.observed_at_unix,
        "expires_at_unix": receipt.expires_at_unix,
        "secret_material_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _terminal_observation(
    session: Any,
    plan: phase_b.PhaseBPlan,
    *,
    bootstrap_resource: Mapping[str, Any],
    absence_receipt: Mapping[str, Any],
    cloud: Mapping[str, Any],
    observed_at_unix: int,
) -> Mapping[str, Any]:
    artifacts = _phase_b_artifacts(plan.revision)
    database_preflight = _database_preflight(session, artifacts)
    projection = phase_b._database_preflight_projection(database_preflight)
    observed_foundation = foundation.observe_foundation(
        plan.revision,
        session,
        _artifacts=artifacts,
    ).to_mapping()
    services = _collect_services(plan.revision, observed_at_unix)
    if cloud.get("user_state_authority") == "postgres_pg_roles":
        cloud_sql = {
            **copy.deepcopy(dict(cloud)),
            "temporary_admin_absent": True,
            "temporary_admin_username_sha256": hashlib.sha256(
                plan.temporary_admin_username.encode("ascii")
            ).hexdigest(),
            "observed_at_unix": observed_at_unix,
        }
    else:
        cloud_sql = {
            "project": foundation.PROJECT,
            "instance": foundation.SQL_INSTANCE,
            "bootstrap_resource": copy.deepcopy(dict(bootstrap_resource)),
            "temporary_admin_absent": True,
            "temporary_admin_username_sha256": hashlib.sha256(
                plan.temporary_admin_username.encode("ascii")
            ).hexdigest(),
            "user_inventory": copy.deepcopy(cloud["user_inventory"]),
            "user_inventory_sha256": cloud["user_inventory_sha256"],
            "user_operations_quiescent": cloud["user_operations_quiescent"],
            "relevant_user_operations": copy.deepcopy(
                cloud["relevant_user_operations"]
            ),
            "operation_ledger_sha256": cloud["operation_ledger_sha256"],
            "observed_at_unix": observed_at_unix,
        }
    unsigned = {
        "schema": phase_b.PHASE_B_TERMINAL_OBSERVATION_SCHEMA,
        "plan_sha256": plan.sha256,
        "foundation_observation": observed_foundation,
        "database_preflight": database_preflight,
        "session_identity_sha256": _database_identity(session)[
            "session_identity_sha256"
        ],
        "writer_ping_identity_sha256": projection[
            "writer_ping_identity_sha256"
        ],
        "event_log_identity_sha256": projection["event_log_identity_sha256"],
        "legacy_archive_identity_sha256": projection[
            "legacy_archive_identity_sha256"
        ],
        "cross_database_acl_sha256": projection["cross_database_acl_sha256"],
        "bootstrap_connect_acl": {
            "database": foundation.SQL_DATABASE,
            "grantee": foundation.CANARY_BOOTSTRAP_ROLE,
            "grantor": foundation.DATABASE_OWNER_ROLE,
            "privilege": "CONNECT",
            "grantable": False,
        },
        "temporary_admin_references": [],
        "cloud_sql": cloud_sql,
        "services": services,
        "observed_at_unix": observed_at_unix,
    }
    return {**unsigned, "observation_sha256": _sha256_json(unsigned)}


def build_phase_b_dependencies(
    plan: phase_b.PhaseBPlan,
    cloud: Any,
) -> phase_b.PhaseBDependencies:
    """Compose real fixed VM adapters with one owner-authorized Cloud edge."""

    if not isinstance(plan, phase_b.PhaseBPlan):
        raise TypeError("PhaseBPlan is required")
    from gateway.canonical_full_canary_coordinator import (
        FixedPhaseBOwnerCloudSQLTransportBoundary,
    )

    if type(cloud) is not FixedPhaseBOwnerCloudSQLTransportBoundary:
        raise TypeError("fixed Phase-B owner Cloud SQL boundary is required")

    def pristine(session: Any) -> Mapping[str, Any]:
        now = int(time.time())
        artifacts = _phase_b_artifacts(plan.revision)
        revision = plan.revision
        _manifest, raw_manifest = load_release_manifest(revision)
        release_artifacts = {
            phase_b.PREFLIGHT_ARTIFACT_PATH: artifacts[
                "phase_b_preflight"
            ].sha256,
            phase_b.ROLE_ARTIFACT_PATH: artifacts["phase_b_role"].sha256,
        }
        unsigned = {
            "schema": phase_b.PHASE_B_PREFLIGHT_SCHEMA,
            "release_revision": revision,
            "release_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "release_artifacts": dict(sorted(release_artifacts.items())),
            "release_artifact_set_sha256": _sha256_json(
                dict(sorted(release_artifacts.items()))
            ),
            "database": _database_identity(session),
            "foundation": _database_preflight(session, artifacts),
            "credential": foundation.PersistentWriterSecretStore().observe(None),
            "services": _collect_services(revision, now),
            "cloud_sql": cloud.observe_initial(plan),
            "observed_at_unix": now,
        }
        return {**unsigned, "observation_sha256": _sha256_json(unsigned)}

    def recovery(
        session: Any,
        plan: phase_b.PhaseBPlan,
        _events: frozenset[str],
    ) -> Mapping[str, Any]:
        now = int(time.time())
        unsigned = {
            "schema": phase_b.PHASE_B_RECOVERY_SCHEMA,
            "plan_sha256": plan.sha256,
            "database": _database_identity(session),
            "database_preflight": _database_preflight(
                session, _phase_b_artifacts(plan.revision)
            ),
            "credential": foundation.PersistentWriterSecretStore().observe(None),
            "services": _collect_services(plan.revision, now),
            "cloud_sql": cloud.observe_recovery(plan),
            "observed_at_unix": now,
        }
        return {**unsigned, "observation_sha256": _sha256_json(unsigned)}

    def admin_factory(
        _plan: phase_b.PhaseBPlan,
        username: str,
        secret: bytearray,
    ) -> _PhaseBAdminSession:
        return _PhaseBAdminSession(
            _open_secret_session(
                username,
                secret,
                purpose="muncho-phase-b-temporary-admin",
            )
        )

    def predelete(
        session: Any,
        plan: phase_b.PhaseBPlan,
        role: Mapping[str, Any],
        authority: Mapping[str, Any],
        self_disable: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        artifacts = _phase_b_artifacts(plan.revision)
        database_preflight = _database_preflight(session, artifacts)
        auto_membership = role["temporary_auto_membership"]
        unsigned = {
            "schema": phase_b.PHASE_B_PREDELETE_SCHEMA,
            "plan_sha256": plan.sha256,
            "foundation_observation": foundation.observe_foundation(
                plan.revision,
                session,
                _artifacts=artifacts,
            ).to_mapping(),
            "database_preflight": database_preflight,
            "bootstrap_connect_acl": {
                "database": foundation.SQL_DATABASE,
                "grantee": foundation.CANARY_BOOTSTRAP_ROLE,
                "grantor": foundation.DATABASE_OWNER_ROLE,
                "privilege": "CONNECT",
                "grantable": False,
            },
            "temporary_auto_membership": copy.deepcopy(auto_membership),
            "other_temporary_admin_references": [],
            "role_receipt_sha256": role["receipt_sha256"],
            "bootstrap_authority_receipt_sha256": authority["receipt_sha256"],
            "self_disable_receipt_sha256": self_disable["receipt_sha256"],
            "temporary_admin_delete_required": True,
            "preterminal": True,
            "safe_to_start": False,
            "secret_material_recorded": False,
        }
        return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}

    def terminal(
        session: Any,
        plan: phase_b.PhaseBPlan,
        resource: Mapping[str, Any],
        absence: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        now = int(time.time())
        cloud_observation = cloud.observe_terminal(
            plan,
            bootstrap_resource=resource,
            absence_receipt=absence,
        )
        return _terminal_observation(
            session,
            plan,
            bootstrap_resource=resource,
            absence_receipt=absence,
            cloud=cloud_observation,
            observed_at_unix=now,
        )

    def temporary_factory(requested: phase_b.PhaseBPlan) -> Any:
        if requested.to_mapping() != plan.to_mapping():
            _fail("phase_b_runtime_plan_changed")
        return cloud.temporary_admin_factory(plan)

    def bootstrap_factory(requested: phase_b.PhaseBPlan) -> Any:
        if requested.to_mapping() != plan.to_mapping():
            _fail("phase_b_runtime_plan_changed")
        return cloud.bootstrap_login_factory(plan)

    def writer_factory() -> Any:
        return _PhaseBAdminSession(_writer_session())

    def services(requested: phase_b.PhaseBPlan, _transition: str) -> Mapping[str, Any]:
        if requested.to_mapping() != plan.to_mapping():
            _fail("phase_b_runtime_plan_changed")
        return _collect_services(plan.revision, int(time.time()))

    return phase_b.PhaseBDependencies(
        writer_session_factory=writer_factory,
        pristine_preflight_collector=pristine,
        recovery_collector=recovery,
        temporary_admin_factory=temporary_factory,
        bootstrap_login_factory=bootstrap_factory,
        admin_session_factory=admin_factory,
        bootstrap_self_disable=_BootstrapSelfDisable(),
        hba_collector=_hba_collector,
        predelete_collector=predelete,
        terminal_collector=terminal,
        services_collector=services,
    )


def execute_fixed_phase_b() -> Mapping[str, Any]:
    """Execute/resume only the root-owned approved Phase-B generation."""

    _require_root_linux()
    _harden_phase_b_process()
    # The owner transport is imported only after process hardening.  Requiring
    # the exact coordinator-owned class prevents a structurally compatible
    # caller object from injecting a second Cloud authority surface.
    from gateway.canonical_full_canary_coordinator import (
        FixedPhaseBOwnerCloudSQLTransportBoundary,
        build_fixed_phase_b_owner_cloud_sql_boundary,
    )

    # Hold a shared approval-generation lock through every mutation.  A newer
    # signed head cannot publish after this process selected an older one.
    with _locked_resume_approval_directory(exclusive=False):
        cloud = build_fixed_phase_b_owner_cloud_sql_boundary()
        if type(cloud) is not FixedPhaseBOwnerCloudSQLTransportBoundary:
            _fail("phase_b_runtime_owner_transport_untrusted")
        plan, approval, journal = load_fixed_phase_b_authority(
            _approval_lock_held=True
        )
        artifacts = _phase_b_artifacts(plan.revision)
        dependencies = build_phase_b_dependencies(plan, cloud)
        return phase_b.execute_approved_phase_b(
            plan,
            approval=approval,
            role_artifact=artifacts["phase_b_role"],
            journal=journal,
            dependencies=dependencies,
        )


def _fixed_completed_foundation() -> phase_b.PhaseBDurableFoundation:
    plan, approval, journal = load_fixed_phase_b_authority()
    return phase_b.load_durable_phase_b_foundation(
        plan,
        approval=approval,
        journal=journal,
    )


def load_fixed_completed_phase_b_foundation() -> Mapping[str, Any]:
    """Load only an authenticated terminal generation, independent of expiry.

    This boundary is evidence replay, never mutation authorization.  The
    immutable approval is structurally reloaded at its issue time and the
    durable foundation loader rebinds it to the authenticated terminal time.
    Missing/incomplete terminal state fails closed; callers must not use this
    function as a substitute for fresh approval of unfinished work.
    """

    _require_root_linux()
    _harden_phase_b_process()
    return copy.deepcopy(dict(_fixed_completed_foundation().to_mapping()))


def _collect_fixed_cloud_readiness(
    foundation_value: phase_b.PhaseBDurableFoundation,
    *,
    observed_at_unix: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Collect fixed Cloud SQL evidence through the VM metadata identity.

    This function intentionally has no token, URL, project, instance, login or
    evidence input.  The GCE identity must possess the already-reviewed
    read-only Cloud SQL permissions; absence of that authority fails closed.
    """

    # Host and OAuth-scope attestation must complete before a metadata token
    # exists in this process.
    host_identity = _collect_fixed_host_identity(observed_at_unix)
    token = _metadata_access_token()
    snapshot = _collect_stable_cloud_snapshot(token)
    instance_projection = copy.deepcopy(dict(snapshot.instance_projection))
    operations = copy.deepcopy(list(snapshot.relevant_user_operations))
    historical = foundation_value.terminal_observation["cloud_sql"]
    if (
        instance_projection != _FIXED_INSTANCE_PROJECTION
        or hashlib.sha256(
            _canonical_bytes(instance_projection) + b"\n"
        ).hexdigest()
        != _FIXED_INSTANCE_PROJECTION_SHA256
        or operations != historical.get("relevant_user_operations")
        or _sha256_json(operations) != historical.get("operation_ledger_sha256")
        or any(row[2] != "DONE" or row[4] is not True for row in operations)
    ):
        _fail("phase_b_runtime_cloud_readiness_invalid")
    cloud = {
        "project": foundation.PROJECT,
        "instance": foundation.SQL_INSTANCE,
        "instance_projection": instance_projection,
        "instance_projection_sha256": hashlib.sha256(
            _canonical_bytes(instance_projection) + b"\n"
        ).hexdigest(),
        "user_state_authority": "postgres_pg_roles",
        "bootstrap_role_present": True,
        "bootstrap_login_present": True,
        "temporary_admin_absent": True,
        "temporary_admin_username_sha256": hashlib.sha256(
            foundation_value.plan.temporary_admin_username.encode("ascii")
        ).hexdigest(),
        "temporary_admin_users": [],
        "user_operations_quiescent": True,
        "relevant_user_operations": operations,
        "operation_ledger_sha256": _sha256_json(operations),
        "observed_at_unix": observed_at_unix,
    }
    return cloud, host_identity


def _collect_fixed_phase_b_readiness_mapping(
    foundation_value: phase_b.PhaseBDurableFoundation,
    *,
    current_release_revision: str,
    observed_at_unix: int,
) -> Mapping[str, Any]:
    """Gather all fixed evidence; never accept a caller-authored observation."""

    _require_root_linux()
    _harden_phase_b_process()
    if (
        not isinstance(foundation_value, phase_b.PhaseBDurableFoundation)
        or _REVISION_RE.fullmatch(current_release_revision) is None
        or type(observed_at_unix) is not int
        or observed_at_unix <= 0
    ):
        _fail("phase_b_runtime_readiness_input_invalid")
    plan = foundation_value.plan
    session = _PhaseBAdminSession(_writer_session())
    try:
        cloud, host_identity = _collect_fixed_cloud_readiness(
            foundation_value,
            observed_at_unix=observed_at_unix,
        )
        historical_bootstrap_resource = foundation_value.terminal_observation[
            "cloud_sql"
        ]["bootstrap_resource"]
        terminal = _terminal_observation(
            session,
            plan,
            bootstrap_resource=historical_bootstrap_resource,
            absence_receipt={
                "evidence_sha256": cloud["operation_ledger_sha256"]
            },
            cloud=cloud,
            observed_at_unix=observed_at_unix,
        )
    finally:
        session.close()
    credential = foundation.PersistentWriterSecretStore().observe(None)
    services = _collect_services(current_release_revision, observed_at_unix)
    continuity = phase_b.build_phase_b_terminal_bootstrap_continuity(
        foundation_value,
        bootstrap_resource=historical_bootstrap_resource,
        operation_ledger_sha256=cloud["operation_ledger_sha256"],
        observed_at_unix=observed_at_unix,
    )
    unsigned = {
        "schema": phase_b.PHASE_B_READINESS_OBSERVATION_SCHEMA,
        "current_release_revision": current_release_revision,
        "foundation_generation_sha256": foundation_value.generation_sha256,
        "foundation_terminal_receipt_sha256": foundation_value.terminal_receipt[
            "receipt_sha256"
        ],
        "terminal_observation": terminal,
        "host_identity": host_identity,
        "cloud_sql": {
            "observation": cloud,
            "observed_at_unix": observed_at_unix,
        },
        "credential": {
            "identity": credential,
            "observed_at_unix": observed_at_unix,
        },
        "services": services,
        "bootstrap_runtime_continuity": continuity.to_mapping(),
        "observed_at_unix": observed_at_unix,
    }
    return {**unsigned, "observation_sha256": _sha256_json(unsigned)}


def _publish_fixed_phase_b_readiness_at(
    *,
    current_release_revision: str,
    now_unix: int,
) -> phase_b.PhaseBReadinessReceipt:
    """Private exact-time seam used by the zero-input production callsite."""

    _require_root_linux()
    _harden_phase_b_process()
    current = now_unix
    if type(current) is not int or current <= 0:
        _fail("phase_b_runtime_readiness_time_invalid")
    durable = _fixed_completed_foundation()
    observation = phase_b._collect_trusted_readiness_observation(
        durable,
        current_release_revision=current_release_revision,
        now_unix=current,
    )
    writer = phase_b._PhaseBReadinessWriterBoundary()
    receipt = writer.publish(
        durable,
        observation=observation,
        current_release_revision=current_release_revision,
        now_unix=current,
    )
    # Hold the current-head reader lock across handoff publication.  A second
    # publisher can therefore only win before this comparison (which fails us)
    # or after the complete root-owned handoff is durable.
    reader = phase_b.AppendOnlyPhaseBReadinessJournal()
    with reader._journal._shared_lock(durable):
        latest = reader._journal._latest_fresh_unlocked(
            durable,
            expected_current_release_revision=current_release_revision,
            now_unix=current,
        )
        if latest.to_mapping() != receipt.to_mapping():
            _fail("phase_b_runtime_readiness_not_current_head")
        chain = reader._journal._load_unlocked(durable)
        if not chain or chain[-1].to_mapping() != latest.to_mapping():
            _fail("phase_b_runtime_readiness_not_current_head")
        _publish_runtime_handoff(
            durable,
            receipts=chain,
            published_at_unix=current,
        )
    return receipt


def _runtime_handoff_mapping(
    durable: phase_b.PhaseBDurableFoundation,
    *,
    receipts: Sequence[phase_b.PhaseBReadinessReceipt],
    published_at_unix: int,
) -> dict[str, Any]:
    if not receipts:
        _fail("phase_b_runtime_readiness_chain_missing")
    foundation_mapping = durable.to_mapping()
    receipt_mappings = [receipt.to_mapping() for receipt in receipts]
    unsigned = {
        "schema": _RUNTIME_HANDOFF_SCHEMA,
        "foundation": foundation_mapping,
        "foundation_sha256": _sha256_json(foundation_mapping),
        "readiness_chain": receipt_mappings,
        "current_readiness_receipt_sha256": receipt_mappings[-1][
            "receipt_sha256"
        ],
        "published_at_unix": published_at_unix,
    }
    return {**unsigned, "handoff_sha256": _sha256_json(unsigned)}


def _open_runtime_handoff_directory() -> int:
    descriptor = _open_trusted_absolute_directory(
        PHASE_B_RUNTIME_RECEIPT_PATH.parent
    )
    status = os.fstat(descriptor)
    if (
        status.st_uid != _ROOT_UID
        or status.st_gid != foundation.WRITER_GID
        or stat.S_IMODE(status.st_mode) != _RUNTIME_HANDOFF_DIRECTORY_MODE
    ):
        os.close(descriptor)
        _fail("phase_b_runtime_handoff_directory_untrusted")
    return descriptor


def _publish_runtime_handoff(
    durable: phase_b.PhaseBDurableFoundation,
    *,
    receipts: Sequence[phase_b.PhaseBReadinessReceipt],
    published_at_unix: int,
) -> None:
    _require_root_linux()
    mapping = _runtime_handoff_mapping(
        durable,
        receipts=receipts,
        published_at_unix=published_at_unix,
    )
    _publish_runtime_owned_json(
        mapping,
        final_path=PHASE_B_RUNTIME_RECEIPT_PATH,
        staging_name=_RUNTIME_HANDOFF_STAGING_NAME,
    )


def _publish_runtime_owned_json(
    mapping: Mapping[str, Any],
    *,
    final_path: Path,
    staging_name: str,
) -> None:
    if (
        final_path not in {
            PHASE_B_RUNTIME_RECEIPT_PATH,
            PHASE_B_FULL_CANARY_ANCHOR_PATH,
        }
        or not staging_name.startswith(".")
        or "/" in staging_name
    ):
        _fail("phase_b_runtime_handoff_path_invalid")
    payload = _canonical_bytes(mapping) + b"\n"
    if len(payload) > _MAX_AUTHORITY_BYTES:
        _fail("phase_b_runtime_handoff_too_large")
    directory = _open_runtime_handoff_directory()
    descriptor: int | None = None
    try:
        try:
            staging_status = os.stat(
                staging_name,
                dir_fd=directory,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            if (
                not stat.S_ISREG(staging_status.st_mode)
                or staging_status.st_uid != _ROOT_UID
                or staging_status.st_gid != foundation.WRITER_GID
                or staging_status.st_nlink != 1
                or stat.S_IMODE(staging_status.st_mode)
                != _RUNTIME_HANDOFF_FILE_MODE
            ):
                _fail("phase_b_runtime_handoff_staging_untrusted")
            os.unlink(staging_name, dir_fd=directory)
            os.fsync(directory)
        descriptor = os.open(
            staging_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            _RUNTIME_HANDOFF_FILE_MODE,
            dir_fd=directory,
        )
        os.fchmod(descriptor, _RUNTIME_HANDOFF_FILE_MODE)
        os.fchown(descriptor, _ROOT_UID, foundation.WRITER_GID)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail("phase_b_runtime_handoff_write_failed")
            offset += written
        os.fsync(descriptor)
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != _ROOT_UID
            or status.st_gid != foundation.WRITER_GID
            or status.st_nlink != 1
            or stat.S_IMODE(status.st_mode) != _RUNTIME_HANDOFF_FILE_MODE
            or status.st_size != len(payload)
        ):
            _fail("phase_b_runtime_handoff_staging_untrusted")
        os.close(descriptor)
        descriptor = None
        os.replace(
            staging_name,
            final_path.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
        )
        os.fsync(directory)
        final = os.stat(
            final_path.name,
            dir_fd=directory,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_uid != _ROOT_UID
            or final.st_gid != foundation.WRITER_GID
            or final.st_nlink != 1
            or stat.S_IMODE(final.st_mode) != _RUNTIME_HANDOFF_FILE_MODE
            or final.st_size != len(payload)
        ):
            _fail("phase_b_runtime_handoff_file_untrusted")
    except PhaseBRuntimeError:
        raise
    except OSError as exc:
        raise PhaseBRuntimeError("phase_b_runtime_handoff_write_failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def publish_fixed_phase_b_readiness() -> phase_b.PhaseBReadinessReceipt:
    """Collect and publish fixed startup authority with no caller input."""

    _require_root_linux()
    _harden_phase_b_process()
    return _publish_fixed_phase_b_readiness_at(
        current_release_revision=_current_release_revision(),
        now_unix=int(time.time()),
    )


def _read_runtime_owned_json(path: Path) -> Mapping[str, Any]:
    if path not in {PHASE_B_RUNTIME_RECEIPT_PATH, PHASE_B_FULL_CANARY_ANCHOR_PATH}:
        _fail("phase_b_runtime_handoff_path_invalid")
    directory = _open_runtime_handoff_directory()
    descriptor: int | None = None
    try:
        name = path.name
        before = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != _ROOT_UID
            or before.st_gid != foundation.WRITER_GID
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != _RUNTIME_HANDOFF_FILE_MODE
            or not 1 <= before.st_size <= _MAX_AUTHORITY_BYTES
        ):
            _fail("phase_b_runtime_handoff_file_untrusted")
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory,
        )
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        size = 0
        while size <= _MAX_AUTHORITY_BYTES:
            chunk = os.read(
                descriptor,
                min(64 * 1024, _MAX_AUTHORITY_BYTES + 1 - size),
            )
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        reachable = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except PhaseBRuntimeError:
        raise
    except OSError as exc:
        raise PhaseBRuntimeError("phase_b_runtime_handoff_unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or _stat_identity(before) != _stat_identity(opened)
        or _stat_identity(before) != _stat_identity(after)
        or _stat_identity(before) != _stat_identity(reachable)
    ):
        _fail("phase_b_runtime_handoff_changed")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PhaseBRuntimeError("phase_b_runtime_handoff_json_invalid") from exc
    if not isinstance(value, dict) or _canonical_bytes(value) + b"\n" != raw:
        _fail("phase_b_runtime_handoff_json_not_canonical")
    return value


def _read_runtime_handoff_json() -> Mapping[str, Any]:
    return _read_runtime_owned_json(PHASE_B_RUNTIME_RECEIPT_PATH)


def _validate_runtime_handoff(
    value: Any,
    *,
    current_release_revision: str,
    now_unix: int,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RUNTIME_HANDOFF_FIELDS:
        _fail("phase_b_runtime_handoff_invalid")
    unsigned_handoff = {
        name: item for name, item in value.items() if name != "handoff_sha256"
    }
    if (
        value["schema"] != _RUNTIME_HANDOFF_SCHEMA
        or value["handoff_sha256"] != _sha256_json(unsigned_handoff)
        or type(value["published_at_unix"]) is not int
        or value["published_at_unix"] > now_unix
    ):
        _fail("phase_b_runtime_handoff_invalid")

    durable = value["foundation"]
    if not isinstance(durable, Mapping) or set(durable) != {
        "plan",
        "approval",
        "generation",
        "terminal_receipt",
        "terminal_observation",
    }:
        _fail("phase_b_runtime_handoff_foundation_invalid")
    if value["foundation_sha256"] != _sha256_json(durable):
        _fail("phase_b_runtime_handoff_foundation_invalid")
    plan = phase_b.PhaseBPlan.from_mapping(durable["plan"])
    terminal_receipt = phase_b._hashed_mapping(
        durable["terminal_receipt"],
        fields=phase_b._TERMINAL_RECEIPT_FIELDS,
        digest_field="receipt_sha256",
        code="phase_b_runtime_handoff_foundation_invalid",
    )
    approval = phase_b.PhaseBApproval.from_mapping(
        durable["approval"],
        plan=plan,
        now_unix=terminal_receipt["terminal_at_unix"],
    )
    historical_terminal = phase_b._validate_terminal_observation(
        durable["terminal_observation"],
        plan=plan,
        execution_preflight_session_sha256=None,
    )
    generation = phase_b._hashed_mapping(
        durable["generation"],
        fields=phase_b._DURABLE_FOUNDATION_FIELDS,
        digest_field="generation_sha256",
        code="phase_b_runtime_handoff_foundation_invalid",
    )
    if (
        terminal_receipt["schema"] != phase_b.PHASE_B_TERMINAL_RECEIPT_SCHEMA
        or terminal_receipt["ok"] is not True
        or terminal_receipt["state"] != "terminal"
        or terminal_receipt["safe_to_start"] is not True
        or terminal_receipt["release_revision"] != plan.revision
        or terminal_receipt["plan_sha256"] != plan.sha256
        or terminal_receipt["approval_sha256"] != approval.sha256
        or terminal_receipt["terminal_observation_sha256"]
        != historical_terminal["observation_sha256"]
        or terminal_receipt["services_attestation_sha256"]
        != historical_terminal["services"]["attestation_sha256"]
        or terminal_receipt["temporary_admin_absent"] is not True
        or terminal_receipt["bootstrap_login_retained"] is not True
        or terminal_receipt["bootstrap_login_password_disabled"] is not True
        or terminal_receipt["secret_material_recorded"] is not False
        or generation["schema"] != phase_b.PHASE_B_DURABLE_FOUNDATION_SCHEMA
        or generation["foundation_release_revision"] != plan.revision
        or generation["plan_sha256"] != plan.sha256
        or generation["approval_sha256"] != approval.sha256
        or generation["terminal_receipt_sha256"]
        != terminal_receipt["receipt_sha256"]
        or generation["terminal_observation_sha256"]
        != historical_terminal["observation_sha256"]
        or generation["terminal_at_unix"] != terminal_receipt["terminal_at_unix"]
    ):
        _fail("phase_b_runtime_handoff_foundation_invalid")

    raw_chain = value["readiness_chain"]
    if not isinstance(raw_chain, list) or not 1 <= len(raw_chain) <= 4096:
        _fail("phase_b_runtime_handoff_readiness_invalid")
    chain: list[dict[str, Any]] = []
    previous: str | None = None
    for sequence, item in enumerate(raw_chain):
        parsed = phase_b._hashed_mapping(
            item,
            fields=phase_b._READINESS_RECEIPT_FIELDS,
            digest_field="receipt_sha256",
            code="phase_b_runtime_handoff_readiness_invalid",
        )
        if (
            parsed["schema"] != phase_b.PHASE_B_READINESS_RECEIPT_SCHEMA
            or parsed["ok"] is not True
            or parsed["state"] != "ready"
            or parsed["safe_to_start"] is not True
            or parsed["current_release_revision"] != current_release_revision
            or parsed["foundation_generation_sha256"]
            != generation["generation_sha256"]
            or parsed["foundation_terminal_receipt_sha256"]
            != terminal_receipt["receipt_sha256"]
            or parsed["sequence"] != sequence
            or parsed["previous_receipt_sha256"] != previous
            or parsed["secret_material_recorded"] is not False
            or not isinstance(parsed["readiness_observation"], Mapping)
            or parsed["readiness_observation_sha256"]
            != parsed["readiness_observation"].get("observation_sha256")
        ):
            _fail("phase_b_runtime_handoff_readiness_chain_invalid")
        previous = parsed["receipt_sha256"]
        chain.append(parsed)
    receipt = chain[-1]
    observation = phase_b._hashed_mapping(
        receipt["readiness_observation"],
        fields=phase_b._READINESS_OBSERVATION_FIELDS,
        digest_field="observation_sha256",
        code="phase_b_runtime_handoff_readiness_invalid",
    )
    observed_at = observation["observed_at_unix"]
    if (
        value["current_readiness_receipt_sha256"] != receipt["receipt_sha256"]
        or receipt["readiness_observation_sha256"]
        != observation["observation_sha256"]
        or type(receipt["issued_at_unix"]) is not int
        or type(receipt["expires_at_unix"]) is not int
        or not receipt["issued_at_unix"] <= value["published_at_unix"] <= now_unix
        or not now_unix < receipt["expires_at_unix"]
        or value["published_at_unix"] != receipt["issued_at_unix"]
        or receipt["expires_at_unix"] - receipt["observed_at_unix"]
        != phase_b.PHASE_B_READINESS_MAX_AGE_SECONDS
        or receipt["observed_at_unix"] != observed_at
        or observation["schema"] != phase_b.PHASE_B_READINESS_OBSERVATION_SCHEMA
        or observation["current_release_revision"] != current_release_revision
        or observation["foundation_generation_sha256"]
        != generation["generation_sha256"]
        or observation["foundation_terminal_receipt_sha256"]
        != terminal_receipt["receipt_sha256"]
    ):
        _fail("phase_b_runtime_handoff_readiness_invalid")

    current_terminal = phase_b._validate_terminal_observation(
        observation["terminal_observation"],
        plan=plan,
        execution_preflight_session_sha256=(
            historical_terminal["session_identity_sha256"]
        ),
        services_release_revision=current_release_revision,
    )
    host = phase_b._validate_readiness_host_identity(
        observation["host_identity"],
        observed_at_unix=observed_at,
    )
    cloud_evidence = phase_b._strict_mapping(
        observation["cloud_sql"],
        frozenset({"observation", "observed_at_unix"}),
        "phase_b_runtime_handoff_readiness_invalid",
    )
    cloud = phase_b._validate_readiness_runtime_cloud(
        cloud_evidence["observation"],
        plan=plan,
        observed_at_unix=observed_at,
    )
    operations = phase_b._validate_terminal_cloud_operations(
        cloud["relevant_user_operations"]
    )
    services = phase_b._validate_services(
        observation["services"],
        release_revision=current_release_revision,
    )
    credential_evidence = phase_b._strict_mapping(
        observation["credential"],
        frozenset({"identity", "observed_at_unix"}),
        "phase_b_runtime_handoff_readiness_invalid",
    )
    credential = phase_b._validate_credential(credential_evidence["identity"])
    phase_b._require_same_credential(plan.preflight.value["credential"], credential)
    historical_cloud = historical_terminal["cloud_sql"]
    bootstrap = phase_b._validate_bootstrap_resource(
        historical_cloud["bootstrap_resource"]
    )
    terminal_cloud = current_terminal["cloud_sql"]
    if (
        operations != historical_cloud["relevant_user_operations"]
        or cloud["operation_ledger_sha256"]
        != historical_cloud["operation_ledger_sha256"]
        or terminal_cloud != cloud
        or services != current_terminal["services"]
        or cloud_evidence["observed_at_unix"] != observed_at
        or credential_evidence["observed_at_unix"] != observed_at
        or services["observed_at_unix"] != observed_at
        or host["observed_at_unix"] != observed_at
    ):
        _fail("phase_b_runtime_handoff_readiness_invalid")
    continuity = phase_b._hashed_mapping(
        observation["bootstrap_runtime_continuity"],
        fields=phase_b._BOOTSTRAP_CONTINUITY_FIELDS,
        digest_field="continuity_sha256",
        code="phase_b_runtime_handoff_readiness_invalid",
    )
    if (
        continuity["schema"] != phase_b.PHASE_B_BOOTSTRAP_CONTINUITY_SCHEMA
        or continuity["source_kind"] != "phase_b_terminal"
        or continuity["foundation_generation_sha256"]
        != generation["generation_sha256"]
        or continuity["source_receipt_sha256"]
        != terminal_receipt["receipt_sha256"]
        or continuity["bootstrap_resource_sha256"] != _sha256_json(bootstrap)
        or continuity["operation_ledger_sha256"]
        != cloud["operation_ledger_sha256"]
        or continuity["observed_at_unix"] != observed_at
    ):
        _fail("phase_b_runtime_handoff_readiness_invalid")
    return receipt


def _validate_full_canary_anchor(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FULL_CANARY_ANCHOR_FIELDS:
        _fail("phase_b_runtime_full_canary_anchor_invalid")
    anchor = copy.deepcopy(dict(value))
    if (
        _REVISION_RE.fullmatch(str(anchor["phase_b_release_revision"])) is None
        or any(
            _SHA256_RE.fullmatch(str(anchor[name])) is None
            for name in _FULL_CANARY_ANCHOR_FIELDS
            if name not in {"phase_b_release_revision", "phase_b_readiness_sequence"}
        )
        or type(anchor["phase_b_readiness_sequence"]) is not int
        or anchor["phase_b_readiness_sequence"] < 0
    ):
        _fail("phase_b_runtime_full_canary_anchor_invalid")
    return anchor


def _runtime_handoff_prefix(
    handoff: Mapping[str, Any],
    *,
    sequence: int,
) -> dict[str, Any]:
    chain = handoff["readiness_chain"]
    if not isinstance(chain, list) or not 0 <= sequence < len(chain):
        _fail("phase_b_runtime_full_canary_anchor_invalid")
    prefix = copy.deepcopy(chain[: sequence + 1])
    current = prefix[-1]
    unsigned = {
        "schema": _RUNTIME_HANDOFF_SCHEMA,
        "foundation": copy.deepcopy(handoff["foundation"]),
        "foundation_sha256": handoff["foundation_sha256"],
        "readiness_chain": prefix,
        "current_readiness_receipt_sha256": current["receipt_sha256"],
        "published_at_unix": current["issued_at_unix"],
    }
    return {**unsigned, "handoff_sha256": _sha256_json(unsigned)}


def _validate_handoff_descends_from_anchor(
    handoff: Mapping[str, Any],
    *,
    anchor: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    validated_anchor = _validate_full_canary_anchor(anchor)
    revision = validated_anchor["phase_b_release_revision"]
    current = _validate_runtime_handoff(
        handoff,
        current_release_revision=revision,
        now_unix=now_unix,
    )
    durable = handoff["foundation"]
    plan = durable["plan"]
    approval = durable["approval"]
    generation = durable["generation"]
    terminal = durable["terminal_receipt"]
    sequence = validated_anchor["phase_b_readiness_sequence"]
    chain = handoff["readiness_chain"]
    if sequence >= len(chain):
        _fail("phase_b_runtime_readiness_anchor_rolled_back")
    anchor_receipt = chain[sequence]
    prefix = _runtime_handoff_prefix(handoff, sequence=sequence)
    prefix_file_sha256 = hashlib.sha256(
        _canonical_bytes(prefix) + b"\n"
    ).hexdigest()
    if (
        validated_anchor["phase_b_plan_sha256"] != plan["plan_sha256"]
        or validated_anchor["phase_b_approval_sha256"]
        != approval["approval_sha256"]
        or validated_anchor["phase_b_terminal_receipt_sha256"]
        != terminal["receipt_sha256"]
        or validated_anchor["phase_b_foundation_generation_sha256"]
        != generation["generation_sha256"]
        or validated_anchor["phase_b_readiness_receipt_sha256"]
        != anchor_receipt["receipt_sha256"]
        or validated_anchor["phase_b_readiness_handoff_file_sha256"]
        != prefix_file_sha256
        or any(
            item["current_release_revision"] != revision
            or item["foundation_generation_sha256"]
            != generation["generation_sha256"]
            or item["foundation_terminal_receipt_sha256"]
            != terminal["receipt_sha256"]
            for item in chain[sequence:]
        )
    ):
        _fail("phase_b_runtime_readiness_anchor_forked")
    return current


def validate_fixed_phase_b_readiness_descendant(
    anchor: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Read-only root/live-driver validation of the plan-bound chain anchor."""

    return _validate_handoff_descends_from_anchor(
        _read_runtime_handoff_json(),
        anchor=anchor,
        now_unix=int(time.time()),
    )


def validate_fixed_phase_b_readiness_lineage(
    anchor: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate an expired anchor's immutable lineage without authorizing start.

    The returned historical head is evidence only.  Callers must publish a
    fresh readiness receipt and use ``validate_fixed_phase_b_readiness_descendant``
    before any privileged/model-running service starts.
    """

    _require_root_linux()
    _harden_phase_b_process()
    handoff = _read_runtime_handoff_json()
    published_at = handoff.get("published_at_unix")
    if type(published_at) is not int or published_at <= 0:
        _fail("phase_b_runtime_handoff_readiness_invalid")
    return _validate_handoff_descends_from_anchor(
        handoff,
        anchor=anchor,
        now_unix=published_at,
    )


def load_fixed_phase_b_readiness_anchor() -> Mapping[str, Any]:
    """Return the exact latest durable readiness anchor for plan construction.

    This is intentionally a zero-input, root-only read boundary.  Callers do
    not synthesize a prefix or select an older readiness generation: the
    anchor is derived from the current authenticated handoff after its full
    foundation/readiness chain has been revalidated for the running release.
    """

    _require_root_linux()
    _harden_phase_b_process()
    revision = _current_release_revision()
    handoff = _read_runtime_handoff_json()
    current = _validate_runtime_handoff(
        handoff,
        current_release_revision=revision,
        now_unix=int(time.time()),
    )
    chain = handoff["readiness_chain"]
    if not isinstance(chain, list) or not chain:
        _fail("phase_b_runtime_handoff_readiness_invalid")
    sequence = len(chain) - 1
    prefix = _runtime_handoff_prefix(handoff, sequence=sequence)
    prefix_file_sha256 = hashlib.sha256(
        _canonical_bytes(prefix) + b"\n"
    ).hexdigest()
    foundation_value = handoff["foundation"]
    anchor = _validate_full_canary_anchor(
        {
            "phase_b_release_revision": revision,
            "phase_b_plan_sha256": foundation_value["plan"]["plan_sha256"],
            "phase_b_approval_sha256": foundation_value["approval"][
                "approval_sha256"
            ],
            "phase_b_terminal_receipt_sha256": foundation_value[
                "terminal_receipt"
            ]["receipt_sha256"],
            "phase_b_foundation_generation_sha256": foundation_value[
                "generation"
            ]["generation_sha256"],
            "phase_b_readiness_receipt_sha256": current["receipt_sha256"],
            "phase_b_readiness_handoff_file_sha256": prefix_file_sha256,
            "phase_b_readiness_sequence": sequence,
        }
    )
    # Reuse the public descendant contract so the loader and all consumers
    # share one fork/rollback decision.
    _validate_handoff_descends_from_anchor(
        handoff,
        anchor=anchor,
        now_unix=int(time.time()),
    )
    return anchor


def install_fixed_phase_b_full_canary_anchor(
    anchor: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Persist only an already-current, plan-bound anchor for the non-root writer."""

    _require_root_linux()
    _harden_phase_b_process()
    validated = _validate_full_canary_anchor(anchor)
    validate_fixed_phase_b_readiness_descendant(validated)
    installed_at = int(time.time())
    unsigned = {
        "schema": _FULL_CANARY_ANCHOR_SCHEMA,
        "anchor": validated,
        "installed_at_unix": installed_at,
    }
    mapping = {**unsigned, "anchor_sha256": _sha256_json(unsigned)}
    _publish_runtime_owned_json(
        mapping,
        final_path=PHASE_B_FULL_CANARY_ANCHOR_PATH,
        staging_name=".full-canary-anchor.staging",
    )
    if _read_runtime_owned_json(PHASE_B_FULL_CANARY_ANCHOR_PATH) != mapping:
        _fail("phase_b_runtime_full_canary_anchor_readback_failed")
    return mapping


def _load_installed_full_canary_anchor() -> Mapping[str, Any]:
    value = _read_runtime_owned_json(PHASE_B_FULL_CANARY_ANCHOR_PATH)
    if not isinstance(value, Mapping) or set(value) != _INSTALLED_ANCHOR_FIELDS:
        _fail("phase_b_runtime_full_canary_anchor_invalid")
    unsigned = {
        name: item for name, item in value.items() if name != "anchor_sha256"
    }
    if (
        value["schema"] != _FULL_CANARY_ANCHOR_SCHEMA
        or value["anchor_sha256"] != _sha256_json(unsigned)
        or type(value["installed_at_unix"]) is not int
    ):
        _fail("phase_b_runtime_full_canary_anchor_invalid")
    return _validate_full_canary_anchor(value["anchor"])


def validate_fixed_phase_b_runtime_readiness() -> Mapping[str, Any]:
    """Validate the fixed root-published current-head handoff as the writer UID."""

    getuid = getattr(os, "geteuid", None)
    getgid = getattr(os, "getegid", None)
    if (
        not callable(getuid)
        or not callable(getgid)
        or getuid() != foundation.WRITER_UID
        or getgid() != foundation.WRITER_GID
    ):
        _fail("phase_b_runtime_handoff_requires_writer_identity")
    if sys.platform != "linux":
        _fail("phase_b_runtime_requires_linux")
    harden_current_process_against_dumping()
    anchor = _load_installed_full_canary_anchor()
    if anchor["phase_b_release_revision"] != _current_release_revision():
        _fail("phase_b_runtime_full_canary_anchor_release_mismatch")
    return _validate_handoff_descends_from_anchor(
        _read_runtime_handoff_json(),
        anchor=anchor,
        now_unix=int(time.time()),
    )


_METADATA_TOKEN_PATH = (
    "/computeMetadata/v1/instance/service-accounts/default/token"
)
_METADATA_SCOPES_PATH = (
    "/computeMetadata/v1/instance/service-accounts/default/scopes"
)


def _read_fixed_metadata_leaf(path: str, *, maximum_bytes: int) -> bytes:
    if path not in {_METADATA_TOKEN_PATH, _METADATA_SCOPES_PATH}:
        _fail("phase_b_runtime_metadata_path_forbidden")
    request = urllib.request.Request(
        "http://169.254.169.254" + path,
        headers={
            "Host": "metadata.google.internal",
            "Metadata-Flavor": "Google",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            raw = response.read(maximum_bytes + 1)
            flavor = response.headers.get("Metadata-Flavor")
    except (OSError, urllib.error.URLError) as exc:
        raise PhaseBRuntimeError("phase_b_runtime_metadata_unavailable") from exc
    if not raw or len(raw) > maximum_bytes or flavor != "Google":
        _fail("phase_b_runtime_metadata_invalid")
    return raw


def _metadata_oauth_scopes() -> list[str]:
    raw = _read_fixed_metadata_leaf(_METADATA_SCOPES_PATH, maximum_bytes=16 * 1024)
    try:
        text = raw.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise PhaseBRuntimeError("phase_b_runtime_metadata_scopes_invalid") from exc
    scopes = text.splitlines()
    if (
        scopes != sorted(set(scopes))
        or scopes != list(phase_b.PHASE_B_VM_OAUTH_SCOPES)
    ):
        _fail("phase_b_runtime_metadata_scopes_invalid")
    return scopes


def _collect_fixed_host_identity(observed_at_unix: int) -> Mapping[str, Any]:
    try:
        receipt = dict(
            collect_dedicated_canary_host_identity_receipt(
                observed_at_unix=observed_at_unix,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise PhaseBRuntimeError("phase_b_runtime_host_identity_invalid") from exc
    scopes = _metadata_oauth_scopes()
    return {
        **receipt,
        "oauth_scopes": scopes,
        "oauth_scopes_sha256": _sha256_json(scopes),
    }


def _metadata_access_token() -> str:
    raw = _read_fixed_metadata_leaf(_METADATA_TOKEN_PATH, maximum_bytes=64 * 1024)
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseBRuntimeError("phase_b_runtime_metadata_invalid") from exc
    token = value.get("access_token") if isinstance(value, dict) else None
    expires = value.get("expires_in") if isinstance(value, dict) else None
    if (
        not isinstance(token, str)
        or not 20 <= len(token) <= 16 * 1024
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in token)
        or type(expires) is not int
        or expires < 60
    ):
        _fail("phase_b_runtime_metadata_invalid")
    return token


def _cloud_get(token: str, url: str) -> Mapping[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "sqladmin.googleapis.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.fragment
        or not parsed.path.startswith(
            f"/sql/v1beta4/projects/{foundation.PROJECT}/"
        )
    ):
        _fail("phase_b_runtime_cloud_url_forbidden")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer " + token,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read(_MAX_CLOUD_BYTES + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise PhaseBRuntimeError("phase_b_runtime_cloud_api_unavailable") from exc
    if len(raw) > _MAX_CLOUD_BYTES:
        _fail("phase_b_runtime_cloud_api_invalid")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PhaseBRuntimeError("phase_b_runtime_cloud_api_invalid") from exc
    if not isinstance(value, dict):
        _fail("phase_b_runtime_cloud_api_invalid")
    return value


_FIXED_INSTANCE_PROJECTION = {
    "backendType": "SECOND_GEN",
    "connectionName": (
        "adventico-ai-platform:europe-west3:muncho-canary-pg18-v2"
    ),
    "databaseVersion": "POSTGRES_18",
    "ipAddresses": [{"ipAddress": "10.91.0.3", "type": "PRIVATE"}],
    "name": foundation.SQL_INSTANCE,
    "project": foundation.PROJECT,
    "region": "europe-west3",
    "state": "RUNNABLE",
}
_FIXED_INSTANCE_PROJECTION_SHA256 = (
    "c7979c4b0a97724a0ac6ac3217a67977a9584622546143aef093b146b7061139"
)


def _cloud_sql_instance(token: str) -> Mapping[str, Any]:
    payload = _cloud_get(
        token,
        "https://sqladmin.googleapis.com/sql/v1beta4/projects/"
        f"{foundation.PROJECT}/instances/{foundation.SQL_INSTANCE}",
    )
    projection = {
        name: copy.deepcopy(payload.get(name))
        for name in _FIXED_INSTANCE_PROJECTION
    }
    if (
        payload.get("kind") != "sql#instance"
        or projection != _FIXED_INSTANCE_PROJECTION
        or hashlib.sha256(_canonical_bytes(projection) + b"\n").hexdigest()
        != _FIXED_INSTANCE_PROJECTION_SHA256
    ):
        _fail("phase_b_runtime_cloud_instance_invalid")
    return projection


def _cloud_sql_operations(token: str) -> tuple[list[Any], ...]:
    base = (
        f"https://sqladmin.googleapis.com/sql/v1beta4/projects/"
        f"{foundation.PROJECT}/operations"
    )
    rows: dict[str, list[Any]] = {}
    page_token: str | None = None
    visited: set[str] = set()
    for _page in range(100):
        query: dict[str, Any] = {
            "instance": foundation.SQL_INSTANCE,
            "maxResults": 100,
        }
        if page_token is not None:
            query["pageToken"] = page_token
        payload = _cloud_get(token, base + "?" + urllib.parse.urlencode(query))
        if payload.get("kind") != "sql#operationsList":
            _fail("phase_b_runtime_cloud_operations_invalid")
        items = payload.get("items", [])
        if not isinstance(items, list):
            _fail("phase_b_runtime_cloud_operations_invalid")
        for item in items:
            if not isinstance(item, Mapping):
                _fail("phase_b_runtime_cloud_operations_invalid")
            operation_type = item.get("operationType")
            if operation_type not in {"CREATE_USER", "UPDATE_USER", "DELETE_USER"}:
                continue
            name = item.get("name")
            status_value = item.get("status")
            actor = item.get("user")
            if (
                not isinstance(name, str)
                or _OPERATION_NAME_RE.fullmatch(name) is None
                or name in rows
                or status_value not in {"PENDING", "RUNNING", "DONE"}
                or not isinstance(actor, str)
                or not actor
            ):
                _fail("phase_b_runtime_cloud_operations_invalid")
            succeeded = status_value == "DONE" and item.get("error") is None
            rows[name] = [
                name,
                operation_type,
                status_value,
                hashlib.sha256(actor.encode("ascii", errors="strict")).hexdigest(),
                succeeded,
            ]
        next_token = payload.get("nextPageToken")
        if next_token is None:
            return tuple(rows[name] for name in sorted(rows))
        if (
            not isinstance(next_token, str)
            or not next_token
            or len(next_token) > 4096
            or next_token in visited
        ):
            _fail("phase_b_runtime_cloud_operations_invalid")
        visited.add(next_token)
        page_token = next_token
    _fail("phase_b_runtime_cloud_operations_invalid")


@dataclass(frozen=True)
class _CloudReadSnapshot:
    instance_projection: Mapping[str, Any]
    relevant_user_operations: tuple[list[Any], ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "instance_projection": copy.deepcopy(
                dict(self.instance_projection)
            ),
            "relevant_user_operations": copy.deepcopy(
                list(self.relevant_user_operations)
            ),
        }


def _collect_cloud_snapshot_once(token: str) -> _CloudReadSnapshot:
    """Bound one exact instance GET between identical operation end fences."""

    operations_before = _cloud_sql_operations(token)
    instance = _cloud_sql_instance(token)
    operations_after = _cloud_sql_operations(token)
    if operations_before != operations_after:
        _fail("phase_b_runtime_cloud_snapshot_raced")
    return _CloudReadSnapshot(
        instance_projection=instance,
        relevant_user_operations=operations_after,
    )


def _collect_stable_cloud_snapshot(token: str) -> _CloudReadSnapshot:
    """Require two bounded, identical snapshots before authorizing startup."""

    first = _collect_cloud_snapshot_once(token)
    second = _collect_cloud_snapshot_once(token)
    if first.to_mapping() != second.to_mapping():
        _fail("phase_b_runtime_cloud_snapshot_raced")
    return second


def _current_release_revision() -> str:
    executable = Path(sys.executable).resolve(strict=True)
    release_base = Path("/opt/muncho-canary-releases")
    try:
        relative = executable.relative_to(release_base)
    except ValueError as exc:
        raise PhaseBRuntimeError("phase_b_runtime_release_not_sealed") from exc
    revision = relative.parts[0] if relative.parts else ""
    if _REVISION_RE.fullmatch(revision) is None:
        _fail("phase_b_runtime_release_not_sealed")
    return revision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect and publish fixed Canonical Writer Phase-B readiness",
    )
    # No arguments are intentionally registered.  The release revision is
    # derived from the sealed interpreter path and every other target is fixed.
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv)
    try:
        receipt = publish_fixed_phase_b_readiness()
    except BaseException as exc:
        code = (
            exc.code
            if isinstance(exc, (PhaseBRuntimeError, phase_b.PhaseBError))
            else "phase_b_runtime_failed"
        )
        print(
            json.dumps(
                {"ok": False, "error_code": code},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            receipt.to_mapping(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by installed CLI.
    raise SystemExit(main())


__all__ = [
    "PHASE_B_APPROVAL_PATH",
    "PHASE_B_AUTHORITY_ROOT",
    "PHASE_B_JOURNAL_ROOT",
    "PHASE_B_PLAN_PATH",
    "PhaseBRuntimeError",
    "execute_fixed_phase_b",
    "inspect_fixed_phase_b_incomplete_head",
    "install_fixed_phase_b_full_canary_anchor",
    "install_fixed_phase_b_resume_approval",
    "load_fixed_completed_phase_b_foundation",
    "load_fixed_phase_b_readiness_anchor",
    "load_fixed_phase_b_approval_chain",
    "main",
    "publish_fixed_phase_b_readiness",
    "validate_fixed_phase_b_readiness_descendant",
    "validate_fixed_phase_b_readiness_lineage",
    "validate_fixed_phase_b_runtime_readiness",
]
