#!/usr/bin/env python3
"""Rotate the dedicated-canary boot receipt through an owner-approved gate.

The stopped release publisher intentionally seals ``host-identity.json`` with
no-replace semantics.  A legitimate VM reboot changes only ``boot_id`` and
therefore needs an equally explicit append-only transition rather than an
operator deleting the stale receipt.  This source-side command provides that
transition while every canary service remains absent or disabled.

The command has no model, database, credential, network, service-control, or
IAM mutation authority.  It observes the fixed VM, validates an exact prior
receipt and owner-supplied IAM-policy digest, archives the prior bytes, writes
a durable tombstone, publishes a freshly collected receipt, and returns an
append-only completion receipt.  Every write is no-replace and fsync/readback
verified; retries resume the same deterministic transaction after any crash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypedDict

from scripts.canary import writer_release as _release
from scripts.canary.runtime_units import CANARY_RUNTIME_UNITS


HOST_RECEIPT_ROTATION_PLAN_SCHEMA = (
    "muncho-full-canary-host-identity-rotation-plan.v1"
)
HOST_RECEIPT_ROTATION_TOMBSTONE_SCHEMA = (
    "muncho-full-canary-host-identity-rotation-tombstone.v1"
)
HOST_RECEIPT_ROTATION_RECEIPT_SCHEMA = (
    "muncho-full-canary-host-identity-rotation-receipt.v1"
)
HOST_RECEIPT_ROTATION_FAILURE_SCHEMA = (
    "muncho-full-canary-host-identity-rotation-failure.v1"
)

DEFAULT_ROTATION_ROOT = Path(
    "/etc/muncho/full-canary/host-identity-rotations"
)
_ROTATION_DIRECTORY_MODE = 0o700
_RECEIPT_MODE = 0o400
_MAX_HOST_RECEIPT_BYTES = 16 * 1024
_MAX_ROTATION_ARTIFACT_BYTES = 128 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_ROTATION_ARTIFACT_NAMES = frozenset({
    "intent.json",
    "prior-host-identity.json",
    "fresh-host-identity.json",
    "tombstone.json",
    "completion.json",
})
_ROTATION_FILES = _ROTATION_ARTIFACT_NAMES | frozenset(
    f".{name}{_release._PUBLICATION_CANDIDATE_SUFFIX}"
    for name in _ROTATION_ARTIFACT_NAMES
)
Clock = Callable[[], float]
HostObserver = Callable[[], Mapping[str, str]]
HostReceiptCollector = Callable[[int], Mapping[str, Any]]
PathExists = Callable[[Path], bool]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError) as exc:
        raise ValueError("host receipt rotation value is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _require_digest(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is not an exact SHA-256")
    return value


def _rotation_id(
    *,
    revision: str,
    external_iam_policy_sha256: str,
    prior_file_sha256: str,
    prior_receipt_sha256: str,
    prior_boot_id_sha256: str,
    current_boot_id_sha256: str,
) -> str:
    return _sha256_json({
        "external_iam_policy_sha256": external_iam_policy_sha256,
        "prior_boot_id_sha256": prior_boot_id_sha256,
        "prior_host_identity_receipt_file_sha256": prior_file_sha256,
        "prior_host_identity_receipt_sha256": prior_receipt_sha256,
        "release_revision": revision,
        "target_boot_id_sha256": current_boot_id_sha256,
    })


def _transaction_paths(rotation_id: str) -> dict[str, Path]:
    _require_digest(rotation_id, label="host receipt rotation id")
    root = DEFAULT_ROTATION_ROOT / rotation_id
    return {
        "root": root,
        "intent": root / "intent.json",
        "archive": root / "prior-host-identity.json",
        "fresh_candidate": root / "fresh-host-identity.json",
        "tombstone": root / "tombstone.json",
        "completion": root / "completion.json",
    }


def _validate_rotation_parent() -> None:
    if DEFAULT_ROTATION_ROOT.parent != _release.DEFAULT_HOST_RECEIPT_PATH.parent:
        raise RuntimeError("host receipt rotation root is not fixed beside receipt")
    _release._validate_host_receipt_parent()
    if os.path.lexists(DEFAULT_ROTATION_ROOT):
        _release._validate_root_directory(
            DEFAULT_ROTATION_ROOT,
            exact_mode=_ROTATION_DIRECTORY_MODE,
        )


def _rotation_lock_path() -> Path:
    expected = DEFAULT_ROTATION_ROOT.with_name(f"{DEFAULT_ROTATION_ROOT.name}.lock")
    shared = _release._host_receipt_lease_path()
    if shared != expected:
        raise RuntimeError("host receipt rotation lock is not the shared lease")
    return shared


def _acquire_rotation_lock() -> int:
    """Acquire the fixed lock that spans the complete destructive transition."""

    _validate_rotation_parent()
    _rotation_lock_path()
    return _release._acquire_host_receipt_lease()


def _validate_transaction_directory(path: Path) -> None:
    _release._validate_root_directory(path, exact_mode=_ROTATION_DIRECTORY_MODE)
    entries = frozenset(os.listdir(path))
    if not entries <= _ROTATION_FILES:
        raise RuntimeError("host receipt rotation transaction has extra entries")


def _create_exact_directory(path: Path) -> None:
    try:
        os.mkdir(path, _ROTATION_DIRECTORY_MODE)
    except FileExistsError:
        _release._validate_root_directory(
            path,
            exact_mode=_ROTATION_DIRECTORY_MODE,
        )
        return
    os.chown(
        path,
        _release._BUILD_OWNER_UID,
        _release._BUILD_OWNER_GID,
        follow_symlinks=False,
    )
    os.chmod(path, _ROTATION_DIRECTORY_MODE, follow_symlinks=False)
    _release._validate_root_directory(path, exact_mode=_ROTATION_DIRECTORY_MODE)
    _release._fsync_directory(path.parent)


def _create_transaction_namespace(rotation_id: str) -> dict[str, Path]:
    paths = _transaction_paths(rotation_id)
    _validate_rotation_parent()
    if not os.path.lexists(DEFAULT_ROTATION_ROOT):
        _create_exact_directory(DEFAULT_ROTATION_ROOT)
    _release._validate_root_directory(
        DEFAULT_ROTATION_ROOT,
        exact_mode=_ROTATION_DIRECTORY_MODE,
    )
    if not os.path.lexists(paths["root"]):
        _create_exact_directory(paths["root"])
    _validate_transaction_directory(paths["root"])
    return paths


def _read_exact_file(path: Path, *, maximum_bytes: int) -> bytes:
    return _release._read_stable_root_file(
        path,
        maximum_bytes=maximum_bytes,
        exact_mode=_RECEIPT_MODE,
    )


def _read_committed_rotation_artifact(
    path: Path,
    *,
    maximum_bytes: int,
) -> bytes:
    scratch = _release._publication_candidate_path(path)
    if not os.path.lexists(scratch):
        return _read_exact_file(path, maximum_bytes=maximum_bytes)
    scratch_raw = _release._read_stable_publication_file(
        scratch,
        maximum_bytes=maximum_bytes,
        exact_mode=_RECEIPT_MODE,
        allowed_nlinks=frozenset({1, 2}),
    )
    if not os.path.lexists(path):
        return scratch_raw
    final_raw = _release._read_stable_publication_file(
        path,
        maximum_bytes=maximum_bytes,
        exact_mode=_RECEIPT_MODE,
        allowed_nlinks=frozenset({2}),
    )
    scratch_item = os.lstat(scratch)
    final_item = os.lstat(path)
    if (
        scratch_raw != final_raw
        or scratch_item.st_dev != final_item.st_dev
        or scratch_item.st_ino != final_item.st_ino
    ):
        raise RuntimeError("host receipt rotation publication diverged")
    return final_raw


def _read_current_host_receipt_raw() -> bytes:
    """Read the current name, including a crashed hardlink cleanup state."""

    path = _release.DEFAULT_HOST_RECEIPT_PATH
    candidate = _release._publication_candidate_path(path)
    if not os.path.lexists(candidate):
        return _read_exact_file(path, maximum_bytes=_MAX_HOST_RECEIPT_BYTES)
    candidate_raw = _release._read_stable_publication_file(
        candidate,
        maximum_bytes=_MAX_HOST_RECEIPT_BYTES,
        exact_mode=_RECEIPT_MODE,
        allowed_nlinks=frozenset({2}),
    )
    current_raw = _release._read_stable_publication_file(
        path,
        maximum_bytes=_MAX_HOST_RECEIPT_BYTES,
        exact_mode=_RECEIPT_MODE,
        allowed_nlinks=frozenset({2}),
    )
    candidate_item = os.lstat(candidate)
    current_item = os.lstat(path)
    if (
        candidate_raw != current_raw
        or candidate_item.st_dev != current_item.st_dev
        or candidate_item.st_ino != current_item.st_ino
    ):
        raise RuntimeError("current host receipt publication diverged")
    return current_raw


def _write_no_replace(path: Path, raw: bytes) -> None:
    _release._publish_bytes_no_replace(
        path,
        raw,
        maximum_bytes=_MAX_ROTATION_ARTIFACT_BYTES,
        exact_mode=_RECEIPT_MODE,
    )


def _publish_or_validate_exact_file(
    path: Path,
    raw: bytes,
    *,
    maximum_bytes: int,
) -> bytes:
    _write_no_replace(path, raw)
    existing = _read_exact_file(path, maximum_bytes=maximum_bytes)
    if existing != raw:
        raise RuntimeError("host receipt rotation artifact readback diverged")
    return existing


def _decode_mapping(raw: bytes, *, label: str) -> dict[str, Any]:
    if not raw or raw.endswith(b"\n") or len(raw) > _MAX_ROTATION_ARTIFACT_BYTES:
        raise RuntimeError(f"{label} framing is invalid")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_release._reject_duplicate_json_pairs,
            parse_constant=lambda _item: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != raw:
        raise RuntimeError(f"{label} is not a canonical mapping")
    return value


def _validate_prior_receipt(
    raw: bytes,
    *,
    current_host: Mapping[str, str],
    expected_file_sha256: str,
    expected_receipt_sha256: str,
    expected_boot_id_sha256: str,
) -> dict[str, Any]:
    if _sha256_bytes(raw) != expected_file_sha256:
        raise RuntimeError("prior host receipt file digest diverged")
    value = _decode_mapping(raw, label="prior host identity receipt")
    prior_host = {
        name: value.get(name) for name in _release._HOST_OBSERVATION_FIELDS
    }
    receipt = _release._validate_host_receipt_mapping(
        value,
        plan={"dedicated_host": prior_host},
    )
    if (
        receipt["receipt_sha256"] != expected_receipt_sha256
        or receipt["boot_id_sha256"] != expected_boot_id_sha256
        or receipt["boot_id_sha256"] == current_host["boot_id_sha256"]
        or any(
            receipt[name] != current_host[name]
            for name in _release._HOST_OBSERVATION_FIELDS
            if name != "boot_id_sha256"
        )
    ):
        raise RuntimeError("prior host receipt is not the exact stale boot")
    return receipt


def _validate_current_receipt(
    raw: bytes,
    *,
    current_host: Mapping[str, str],
) -> dict[str, Any]:
    value = _decode_mapping(raw, label="current host identity receipt")
    return _release._validate_host_receipt_mapping(
        value,
        plan={"dedicated_host": dict(current_host)},
    )


def _prior_raw_for_plan(
    paths: Mapping[str, Path],
    *,
    expected_file_sha256: str,
) -> bytes:
    archive = paths["archive"]
    current = _release.DEFAULT_HOST_RECEIPT_PATH
    if os.path.lexists(archive):
        raw = _read_committed_rotation_artifact(
            archive,
            maximum_bytes=_MAX_HOST_RECEIPT_BYTES,
        )
        if _sha256_bytes(raw) != expected_file_sha256:
            raise RuntimeError("archived prior host receipt diverged")
        return raw
    if not os.path.lexists(current):
        raise RuntimeError("prior host receipt is absent without archive")
    raw = _read_exact_file(current, maximum_bytes=_MAX_HOST_RECEIPT_BYTES)
    if _sha256_bytes(raw) != expected_file_sha256:
        raise RuntimeError("current host receipt is not the expected prior receipt")
    return raw


def _safe_service_states(
    *,
    runner: _release.Runner = _release._runner,
) -> list[dict[str, Any]]:
    states = _release._collect_service_states(runner=runner)
    if (
        [state.get("unit") for state in states] != list(CANARY_RUNTIME_UNITS)
        or any(
            state["state"] not in {"absent", "disabled_inactive"}
            for state in states
        )
    ):
        raise RuntimeError("host receipt rotation requires stopped services")
    return states


def plan_host_receipt_rotation(
    revision: str,
    *,
    external_iam_policy_sha256: str,
    expected_prior_file_sha256: str,
    expected_prior_receipt_sha256: str,
    expected_prior_boot_id_sha256: str,
    expected_current_boot_id_sha256: str,
    runner: _release.Runner = _release._runner,
    host_observer: HostObserver = _release._default_host_observer,
) -> dict[str, Any]:
    """Return the exact read-only owner approval surface for one reboot."""

    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("host receipt rotation revision is invalid")
    external_iam_policy_sha256 = _require_digest(
        external_iam_policy_sha256,
        label="external IAM policy digest",
    )
    expected_prior_file_sha256 = _require_digest(
        expected_prior_file_sha256,
        label="prior host receipt file digest",
    )
    expected_prior_receipt_sha256 = _require_digest(
        expected_prior_receipt_sha256,
        label="prior host receipt digest",
    )
    expected_prior_boot_id_sha256 = _require_digest(
        expected_prior_boot_id_sha256,
        label="prior boot digest",
    )
    expected_current_boot_id_sha256 = _require_digest(
        expected_current_boot_id_sha256,
        label="current boot digest",
    )
    _validate_rotation_parent()
    current_host = _release._validate_host_observation(host_observer())
    if current_host["boot_id_sha256"] != expected_current_boot_id_sha256:
        raise RuntimeError("current boot identity does not match owner intent")
    rotation_id = _rotation_id(
        revision=revision,
        external_iam_policy_sha256=external_iam_policy_sha256,
        prior_file_sha256=expected_prior_file_sha256,
        prior_receipt_sha256=expected_prior_receipt_sha256,
        prior_boot_id_sha256=expected_prior_boot_id_sha256,
        current_boot_id_sha256=expected_current_boot_id_sha256,
    )
    paths = _transaction_paths(rotation_id)
    if os.path.lexists(paths["root"]):
        _validate_transaction_directory(paths["root"])
    prior_raw = _prior_raw_for_plan(
        paths,
        expected_file_sha256=expected_prior_file_sha256,
    )
    prior = _validate_prior_receipt(
        prior_raw,
        current_host=current_host,
        expected_file_sha256=expected_prior_file_sha256,
        expected_receipt_sha256=expected_prior_receipt_sha256,
        expected_boot_id_sha256=expected_prior_boot_id_sha256,
    )
    current_path = _release.DEFAULT_HOST_RECEIPT_PATH
    current_is_fresh = False
    if os.path.lexists(current_path):
        current_raw = _read_current_host_receipt_raw()
        if _sha256_bytes(current_raw) != expected_prior_file_sha256:
            if not os.path.lexists(paths["archive"]):
                raise RuntimeError("fresh host receipt exists without prior archive")
            if not os.path.lexists(paths["intent"]):
                raise RuntimeError("fresh host receipt exists without matching intent")
            _validate_current_receipt(current_raw, current_host=current_host)
            current_is_fresh = True
    elif not os.path.lexists(paths["archive"]):
        raise RuntimeError("host receipt and its archive are both absent")

    service_states = _safe_service_states(runner=runner)
    unsigned: dict[str, Any] = {
        "schema": HOST_RECEIPT_ROTATION_PLAN_SCHEMA,
        "release_revision": revision,
        "external_iam_policy_sha256": external_iam_policy_sha256,
        "rotation_id": rotation_id,
        "rotation_root": str(paths["root"]),
        "intent_path": str(paths["intent"]),
        "prior_archive_path": str(paths["archive"]),
        "tombstone_path": str(paths["tombstone"]),
        "completion_path": str(paths["completion"]),
        "host_identity_receipt_path": str(current_path),
        "prior_host_identity_receipt_file_sha256": expected_prior_file_sha256,
        "prior_host_identity_receipt_sha256": expected_prior_receipt_sha256,
        "prior_boot_id_sha256": expected_prior_boot_id_sha256,
        "prior_observed_at_unix": prior["observed_at_unix"],
        "target_host": current_host,
        "target_boot_id_sha256": expected_current_boot_id_sha256,
        "service_states": service_states,
        "invariants": {
            "services_started": False,
            "service_units_changed": False,
            "iam_mutated": False,
            "prior_receipt_archived_no_replace": True,
            "prior_receipt_tombstoned_before_retirement": True,
            "fresh_receipt_collected_on_target_boot": True,
        },
    }
    plan = {**unsigned, "plan_sha256": _sha256_json(unsigned)}
    if current_is_fresh:
        intent_raw = _read_committed_rotation_artifact(
            paths["intent"],
            maximum_bytes=_MAX_ROTATION_ARTIFACT_BYTES,
        )
        if intent_raw != _canonical_bytes(plan):
            raise RuntimeError("fresh host receipt intent diverged")
    return plan


def _validate_exact_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema",
        "release_revision",
        "external_iam_policy_sha256",
        "rotation_id",
        "rotation_root",
        "intent_path",
        "prior_archive_path",
        "tombstone_path",
        "completion_path",
        "host_identity_receipt_path",
        "prior_host_identity_receipt_file_sha256",
        "prior_host_identity_receipt_sha256",
        "prior_boot_id_sha256",
        "prior_observed_at_unix",
        "target_host",
        "target_boot_id_sha256",
        "service_states",
        "invariants",
        "plan_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RuntimeError("host receipt rotation plan is incomplete")
    plan = dict(value)
    if (
        plan["schema"] != HOST_RECEIPT_ROTATION_PLAN_SCHEMA
        or _REVISION_RE.fullmatch(str(plan["release_revision"])) is None
        or type(plan["prior_observed_at_unix"]) is not int
        or plan["prior_observed_at_unix"] < 0
    ):
        raise RuntimeError("host receipt rotation plan is invalid")
    for name in (
        "external_iam_policy_sha256",
        "rotation_id",
        "prior_host_identity_receipt_file_sha256",
        "prior_host_identity_receipt_sha256",
        "prior_boot_id_sha256",
        "target_boot_id_sha256",
        "plan_sha256",
    ):
        _require_digest(str(plan[name]), label=name)
    paths = _transaction_paths(str(plan["rotation_id"]))
    if (
        plan["rotation_root"] != str(paths["root"])
        or plan["intent_path"] != str(paths["intent"])
        or plan["prior_archive_path"] != str(paths["archive"])
        or plan["tombstone_path"] != str(paths["tombstone"])
        or plan["completion_path"] != str(paths["completion"])
        or plan["host_identity_receipt_path"]
        != str(_release.DEFAULT_HOST_RECEIPT_PATH)
        or plan["target_boot_id_sha256"]
        != plan["target_host"].get("boot_id_sha256")
    ):
        raise RuntimeError("host receipt rotation plan paths diverged")
    _release._validate_host_observation(plan["target_host"])
    if plan["invariants"] != {
        "services_started": False,
        "service_units_changed": False,
        "iam_mutated": False,
        "prior_receipt_archived_no_replace": True,
        "prior_receipt_tombstoned_before_retirement": True,
        "fresh_receipt_collected_on_target_boot": True,
    }:
        raise RuntimeError("host receipt rotation invariants diverged")
    expected_id = _rotation_id(
        revision=str(plan["release_revision"]),
        external_iam_policy_sha256=str(plan["external_iam_policy_sha256"]),
        prior_file_sha256=str(plan["prior_host_identity_receipt_file_sha256"]),
        prior_receipt_sha256=str(plan["prior_host_identity_receipt_sha256"]),
        prior_boot_id_sha256=str(plan["prior_boot_id_sha256"]),
        current_boot_id_sha256=str(plan["target_boot_id_sha256"]),
    )
    unsigned = {name: item for name, item in plan.items() if name != "plan_sha256"}
    if plan["rotation_id"] != expected_id or plan["plan_sha256"] != _sha256_json(unsigned):
        raise RuntimeError("host receipt rotation plan digest diverged")
    return plan


def _tombstone(plan: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "schema": HOST_RECEIPT_ROTATION_TOMBSTONE_SCHEMA,
        "state": "prior_boot_archived_before_retirement",
        "release_revision": plan["release_revision"],
        "external_iam_policy_sha256": plan["external_iam_policy_sha256"],
        "rotation_id": plan["rotation_id"],
        "plan_sha256": plan["plan_sha256"],
        "prior_host_identity_receipt_path": plan["host_identity_receipt_path"],
        "prior_archive_path": plan["prior_archive_path"],
        "prior_host_identity_receipt_file_sha256": (
            plan["prior_host_identity_receipt_file_sha256"]
        ),
        "prior_host_identity_receipt_sha256": (
            plan["prior_host_identity_receipt_sha256"]
        ),
        "prior_boot_id_sha256": plan["prior_boot_id_sha256"],
        "target_boot_id_sha256": plan["target_boot_id_sha256"],
        "services_started": False,
        "iam_mutated": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _completion_receipt(
    plan: Mapping[str, Any],
    *,
    tombstone_sha256: str,
    fresh_raw: bytes,
    fresh_receipt: Mapping[str, Any],
    service_states_after: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    unsigned = {
        "schema": HOST_RECEIPT_ROTATION_RECEIPT_SCHEMA,
        "ok": True,
        "state": "target_boot_receipt_published_services_stopped",
        "release_revision": plan["release_revision"],
        "external_iam_policy_sha256": plan["external_iam_policy_sha256"],
        "rotation_id": plan["rotation_id"],
        "plan_sha256": plan["plan_sha256"],
        "prior_archive_path": plan["prior_archive_path"],
        "prior_host_identity_receipt_file_sha256": (
            plan["prior_host_identity_receipt_file_sha256"]
        ),
        "prior_host_identity_receipt_sha256": (
            plan["prior_host_identity_receipt_sha256"]
        ),
        "prior_boot_id_sha256": plan["prior_boot_id_sha256"],
        "tombstone_path": plan["tombstone_path"],
        "tombstone_receipt_sha256": tombstone_sha256,
        "host_identity_receipt_path": plan["host_identity_receipt_path"],
        "host_identity_receipt_file_sha256": _sha256_bytes(fresh_raw),
        "host_identity_receipt_sha256": fresh_receipt["receipt_sha256"],
        "target_boot_id_sha256": plan["target_boot_id_sha256"],
        "fresh_observed_at_unix": fresh_receipt["observed_at_unix"],
        "service_states_before": plan["service_states"],
        "service_states_after": list(service_states_after),
        "services_started": False,
        "service_units_changed": False,
        "iam_mutated": False,
        "completion_path": plan["completion_path"],
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _validate_completion(
    raw: bytes,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    value = _decode_mapping(raw, label="host receipt rotation completion")
    if value != expected:
        raise RuntimeError("host receipt rotation completion diverged")
    return value


def _remove_exact_prior_current(
    *,
    expected_prior_raw: bytes,
    archive_path: Path,
) -> None:
    current_path = _release.DEFAULT_HOST_RECEIPT_PATH
    if not os.path.lexists(current_path):
        return
    current_raw = _read_current_host_receipt_raw()
    if current_raw != expected_prior_raw:
        return
    if os.path.lexists(_release._publication_candidate_path(current_path)):
        _release._write_host_receipt_no_replace(
            current_path,
            _decode_mapping(current_raw, label="prior host identity receipt"),
        )
        current_raw = _read_exact_file(
            current_path,
            maximum_bytes=_MAX_HOST_RECEIPT_BYTES,
        )
        if current_raw != expected_prior_raw:
            raise RuntimeError("prior host receipt recovery diverged")
    archive_raw = _read_exact_file(
        archive_path,
        maximum_bytes=_MAX_HOST_RECEIPT_BYTES,
    )
    if archive_raw != expected_prior_raw:
        raise RuntimeError("prior archive changed before retirement")
    before = os.lstat(current_path)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != _release._BUILD_OWNER_UID
        or before.st_gid != _release._BUILD_OWNER_GID
        or stat.S_IMODE(before.st_mode) != _RECEIPT_MODE
    ):
        raise RuntimeError("prior host receipt changed before retirement")
    os.unlink(current_path)
    _release._fsync_directory(current_path.parent)


def _publish_or_reuse_fresh_candidate(
    plan: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    host_receipt_collector: HostReceiptCollector,
    clock: Clock,
    prior_raw: bytes,
) -> tuple[bytes, dict[str, Any]]:
    """Commit the fresh receipt before the prior current name is retired."""

    candidate_path = paths["fresh_candidate"]
    candidate_scratch = _release._publication_candidate_path(candidate_path)
    if os.path.lexists(candidate_path) or os.path.lexists(candidate_scratch):
        candidate_raw = _release._read_complete_publication_candidate(
            candidate_path,
            maximum_bytes=_MAX_HOST_RECEIPT_BYTES,
            exact_mode=_RECEIPT_MODE,
        )
        if candidate_raw is None:
            if not os.path.lexists(candidate_path):
                candidate_raw = b""
            else:
                candidate_raw = _read_exact_file(
                    candidate_path,
                    maximum_bytes=_MAX_HOST_RECEIPT_BYTES,
                )
        if candidate_raw:
            _publish_or_validate_exact_file(
                candidate_path,
                candidate_raw,
                maximum_bytes=_MAX_HOST_RECEIPT_BYTES,
            )
            candidate_raw = _read_exact_file(
                candidate_path,
                maximum_bytes=_MAX_HOST_RECEIPT_BYTES,
            )
            return candidate_raw, _validate_current_receipt(
                candidate_raw,
                current_host=plan["target_host"],
            )

    current_path = _release.DEFAULT_HOST_RECEIPT_PATH
    if os.path.lexists(current_path):
        current_raw = _read_current_host_receipt_raw()
        if current_raw != prior_raw:
            fresh_receipt = _validate_current_receipt(
                current_raw,
                current_host=plan["target_host"],
            )
            fresh_raw = current_raw
        else:
            fresh_raw = b""
            fresh_receipt = {}
    else:
        fresh_raw = b""
        fresh_receipt = {}

    if not fresh_raw:
        observed_at_unix = int(clock())
        if observed_at_unix < 0:
            raise ValueError("host receipt rotation time is invalid")
        fresh_receipt = _release._validate_host_receipt_mapping(
            host_receipt_collector(observed_at_unix),
            plan={"dedicated_host": plan["target_host"]},
        )
        fresh_raw = _canonical_bytes(fresh_receipt)
        if len(fresh_raw) > _MAX_HOST_RECEIPT_BYTES:
            raise RuntimeError("fresh host identity receipt exceeds its bound")

    _publish_or_validate_exact_file(
        candidate_path,
        fresh_raw,
        maximum_bytes=_MAX_HOST_RECEIPT_BYTES,
    )
    committed = _read_exact_file(
        candidate_path,
        maximum_bytes=_MAX_HOST_RECEIPT_BYTES,
    )
    if committed != fresh_raw:
        raise RuntimeError("fresh host receipt candidate readback diverged")
    return committed, _validate_current_receipt(
        committed,
        current_host=plan["target_host"],
    )


def _apply_host_receipt_rotation_locked(
    revision: str,
    approved_plan_sha256: str,
    *,
    external_iam_policy_sha256: str,
    expected_prior_file_sha256: str,
    expected_prior_receipt_sha256: str,
    expected_prior_boot_id_sha256: str,
    expected_current_boot_id_sha256: str,
    runner: _release.Runner = _release._runner,
    host_observer: HostObserver = _release._default_host_observer,
    host_receipt_collector: HostReceiptCollector = (
        _release._default_host_receipt_collector
    ),
    clock: Clock = time.time,
) -> dict[str, Any]:
    """Apply or resume one exact append-only host receipt transition."""

    _release._require_root_linux()
    approved_plan_sha256 = _require_digest(
        approved_plan_sha256,
        label="approved host receipt rotation plan",
    )
    plan = _validate_exact_plan(plan_host_receipt_rotation(
        revision,
        external_iam_policy_sha256=external_iam_policy_sha256,
        expected_prior_file_sha256=expected_prior_file_sha256,
        expected_prior_receipt_sha256=expected_prior_receipt_sha256,
        expected_prior_boot_id_sha256=expected_prior_boot_id_sha256,
        expected_current_boot_id_sha256=expected_current_boot_id_sha256,
        runner=runner,
        host_observer=host_observer,
    ))
    if plan["plan_sha256"] != approved_plan_sha256:
        raise PermissionError("host receipt rotation approval does not match plan")
    paths = _create_transaction_namespace(str(plan["rotation_id"]))
    intent_raw = _canonical_bytes(plan)
    _publish_or_validate_exact_file(
        paths["intent"],
        intent_raw,
        maximum_bytes=_MAX_ROTATION_ARTIFACT_BYTES,
    )

    prior_raw = _prior_raw_for_plan(
        paths,
        expected_file_sha256=str(plan["prior_host_identity_receipt_file_sha256"]),
    )
    _validate_prior_receipt(
        prior_raw,
        current_host=plan["target_host"],
        expected_file_sha256=str(plan["prior_host_identity_receipt_file_sha256"]),
        expected_receipt_sha256=str(plan["prior_host_identity_receipt_sha256"]),
        expected_boot_id_sha256=str(plan["prior_boot_id_sha256"]),
    )
    _publish_or_validate_exact_file(
        paths["archive"],
        prior_raw,
        maximum_bytes=_MAX_HOST_RECEIPT_BYTES,
    )
    fresh_raw, fresh_receipt = _publish_or_reuse_fresh_candidate(
        plan,
        paths,
        host_receipt_collector=host_receipt_collector,
        clock=clock,
        prior_raw=prior_raw,
    )
    tombstone = _tombstone(plan)
    tombstone_raw = _canonical_bytes(tombstone)
    _publish_or_validate_exact_file(
        paths["tombstone"],
        tombstone_raw,
        maximum_bytes=_MAX_ROTATION_ARTIFACT_BYTES,
    )
    _remove_exact_prior_current(
        expected_prior_raw=prior_raw,
        archive_path=paths["archive"],
    )

    current_path = _release.DEFAULT_HOST_RECEIPT_PATH
    if os.path.lexists(current_path):
        current_raw = _read_current_host_receipt_raw()
        if current_raw != fresh_raw:
            raise RuntimeError("fresh host identity receipt diverged from candidate")
        fresh_receipt = _validate_current_receipt(
            current_raw, current_host=plan["target_host"]
        )
        _release._write_host_receipt_no_replace(current_path, fresh_receipt)
        current_raw = _read_exact_file(
            current_path,
            maximum_bytes=_MAX_HOST_RECEIPT_BYTES,
        )
        if current_raw != fresh_raw:
            raise RuntimeError("fresh host identity recovery diverged")
    else:
        _release._write_host_receipt_no_replace(current_path, fresh_receipt)
        readback = _read_exact_file(
            current_path,
            maximum_bytes=_MAX_HOST_RECEIPT_BYTES,
        )
        if readback != fresh_raw:
            raise RuntimeError("fresh host identity receipt readback diverged")
        fresh_raw = readback
        fresh_receipt = _validate_current_receipt(
            fresh_raw,
            current_host=plan["target_host"],
        )

    service_states_after = _safe_service_states(runner=runner)
    if service_states_after != plan["service_states"]:
        raise RuntimeError("service state drifted during host receipt rotation")
    completion = _completion_receipt(
        plan,
        tombstone_sha256=str(tombstone["receipt_sha256"]),
        fresh_raw=fresh_raw,
        fresh_receipt=fresh_receipt,
        service_states_after=service_states_after,
    )
    completion_raw = _canonical_bytes(completion)
    _publish_or_validate_exact_file(
        paths["completion"],
        completion_raw,
        maximum_bytes=_MAX_ROTATION_ARTIFACT_BYTES,
    )
    return _validate_completion(
        _read_exact_file(
            paths["completion"],
            maximum_bytes=_MAX_ROTATION_ARTIFACT_BYTES,
        ),
        expected=completion,
    )


def apply_host_receipt_rotation(
    revision: str,
    approved_plan_sha256: str,
    *,
    external_iam_policy_sha256: str,
    expected_prior_file_sha256: str,
    expected_prior_receipt_sha256: str,
    expected_prior_boot_id_sha256: str,
    expected_current_boot_id_sha256: str,
    runner: _release.Runner = _release._runner,
    host_observer: HostObserver = _release._default_host_observer,
    host_receipt_collector: HostReceiptCollector = (
        _release._default_host_receipt_collector
    ),
    clock: Clock = time.time,
) -> dict[str, Any]:
    """Apply one exact transition under the fixed whole-rotation lease."""

    _release._require_root_linux()
    descriptor = _acquire_rotation_lock()
    try:
        return _apply_host_receipt_rotation_locked(
            revision,
            approved_plan_sha256,
            external_iam_policy_sha256=external_iam_policy_sha256,
            expected_prior_file_sha256=expected_prior_file_sha256,
            expected_prior_receipt_sha256=expected_prior_receipt_sha256,
            expected_prior_boot_id_sha256=expected_prior_boot_id_sha256,
            expected_current_boot_id_sha256=expected_current_boot_id_sha256,
            runner=runner,
            host_observer=host_observer,
            host_receipt_collector=host_receipt_collector,
            clock=clock,
        )
    finally:
        os.close(descriptor)


class _ExactParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise ValueError("invalid host receipt rotation CLI arguments")


class _StoreOnce(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        if getattr(namespace, self.dest, None) is not None:
            raise ValueError("host receipt rotation CLI option was repeated")
        setattr(namespace, self.dest, values)


def _revision(value: str) -> str:
    if _REVISION_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("invalid revision")
    return value


def _digest(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("invalid digest")
    return value


def _add_exact_intent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--revision", required=True, type=_revision, action=_StoreOnce)
    parser.add_argument(
        "--external-iam-policy-sha256",
        required=True,
        type=_digest,
        action=_StoreOnce,
    )
    parser.add_argument(
        "--expected-prior-file-sha256",
        required=True,
        type=_digest,
        action=_StoreOnce,
    )
    parser.add_argument(
        "--expected-prior-receipt-sha256",
        required=True,
        type=_digest,
        action=_StoreOnce,
    )
    parser.add_argument(
        "--expected-prior-boot-id-sha256",
        required=True,
        type=_digest,
        action=_StoreOnce,
    )
    parser.add_argument(
        "--expected-current-boot-id-sha256",
        required=True,
        type=_digest,
        action=_StoreOnce,
    )


def _parser() -> argparse.ArgumentParser:
    parser = _ExactParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_ExactParser,
    )
    plan = subparsers.add_parser("plan", allow_abbrev=False)
    _add_exact_intent_arguments(plan)
    apply = subparsers.add_parser("apply", allow_abbrev=False)
    _add_exact_intent_arguments(apply)
    apply.add_argument(
        "--approved-plan-sha256",
        required=True,
        type=_digest,
        action=_StoreOnce,
    )
    return parser


class _RotationIntentKwargs(TypedDict):
    external_iam_policy_sha256: str
    expected_prior_file_sha256: str
    expected_prior_receipt_sha256: str
    expected_prior_boot_id_sha256: str
    expected_current_boot_id_sha256: str


def _intent_kwargs(arguments: argparse.Namespace) -> _RotationIntentKwargs:
    return {
        "external_iam_policy_sha256": arguments.external_iam_policy_sha256,
        "expected_prior_file_sha256": arguments.expected_prior_file_sha256,
        "expected_prior_receipt_sha256": arguments.expected_prior_receipt_sha256,
        "expected_prior_boot_id_sha256": arguments.expected_prior_boot_id_sha256,
        "expected_current_boot_id_sha256": arguments.expected_current_boot_id_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.command == "plan":
            result = plan_host_receipt_rotation(
                arguments.revision,
                **_intent_kwargs(arguments),
            )
        else:
            result = apply_host_receipt_rotation(
                arguments.revision,
                arguments.approved_plan_sha256,
                **_intent_kwargs(arguments),
            )
        sys.stdout.buffer.write(_canonical_bytes(result) + b"\n")
        sys.stdout.buffer.flush()
        return 0
    except BaseException as exc:
        code = (
            "invalid_host_receipt_rotation_arguments"
            if isinstance(exc, (ValueError, argparse.ArgumentError))
            else "host_receipt_rotation_failed"
        )
        unsigned = {
            "schema": HOST_RECEIPT_ROTATION_FAILURE_SCHEMA,
            "ok": False,
            "error_code": code,
        }
        failure = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
        sys.stdout.buffer.write(_canonical_bytes(failure) + b"\n")
        sys.stdout.buffer.flush()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ROTATION_ROOT",
    "HOST_RECEIPT_ROTATION_FAILURE_SCHEMA",
    "HOST_RECEIPT_ROTATION_PLAN_SCHEMA",
    "HOST_RECEIPT_ROTATION_RECEIPT_SCHEMA",
    "HOST_RECEIPT_ROTATION_TOMBSTONE_SCHEMA",
    "apply_host_receipt_rotation",
    "main",
    "plan_host_receipt_rotation",
]
