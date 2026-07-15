#!/usr/bin/env python3
"""Read-only, package-bound host authority collector for production cutover.

The collector accepts one canonical public request on stdin.  It never accepts
an executable, path override, credential, or private key.  It verifies the
immutable release package, reads back all nineteen fixed staged host files,
compares their exact target pre-state, and validates the already-authored
mechanical topology/transition/cron plan.  It performs no install, chmod,
chown, systemd, database, or secret operation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway import canonical_writer_production_cutover as cutover
from gateway import production_cron_migration
from gateway.production_capability_prerequisites import (
    validate_production_capability_topology,
)
from ops.muncho.runtime import mechanical_job_rail
from scripts.canary import package_production_cutover_artifacts as package


REQUEST_SCHEMA = "muncho-production-cutover-host-authority-request.v1"
RECEIPT_SCHEMA = "muncho-production-cutover-host-authority.v1"
MAX_INPUT = 16 * 1024 * 1024
MAX_FILE = 4 * 1024 * 1024
MAX_AGE_SECONDS = 900
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_FIELDS = frozenset({
    "schema",
    "release_revision",
    "initial_collector_receipt",
    "release_manifest_sha256",
    "gateway_target_identity",
    "writer_target_identity",
    "connector_target_identity",
    "host_transition",
    "capability_topology",
    "cron_continuity_plan",
    "secret_material_recorded",
    "secret_digest_recorded",
    "request_sha256",
})
_RECEIPT_FIELDS = frozenset({
    "schema",
    "release_revision",
    "request_sha256",
    "initial_collector_receipt_sha256",
    "release_manifest_sha256",
    "host_artifact_contract_sha256",
    "gateway_target_identity",
    "writer_target_identity",
    "connector_target_identity",
    "host_transition",
    "capability_topology",
    "cron_continuity_plan",
    "readback_file_count",
    "readback_files",
    "readback_set_sha256",
    "observed_at_unix",
    "source_boot_id_sha256",
    "secret_material_recorded",
    "secret_digest_recorded",
    "receipt_sha256",
})


class HostAuthorityError(RuntimeError):
    """Stable, secret-free host authority failure."""


def _canonical(value: Any) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise HostAuthorityError("host_authority_json_invalid") from exc
    if len(raw) > MAX_INPUT:
        raise HostAuthorityError("host_authority_json_oversized")
    return raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _decode(raw: bytes) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise HostAuthorityError("host_authority_duplicate_key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except HostAuthorityError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise HostAuthorityError("host_authority_json_invalid") from exc
    if not isinstance(value, Mapping) or raw != _canonical(value):
        raise HostAuthorityError("host_authority_json_not_canonical")
    return value


def _boot_id(path: Path = BOOT_ID_PATH) -> str:
    try:
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= 256
        ):
            raise HostAuthorityError("host_authority_boot_identity_invalid")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            raw = os.read(descriptor, 257)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except HostAuthorityError:
        raise
    except OSError as exc:
        raise HostAuthorityError("host_authority_boot_unavailable") from exc
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        len(raw) != before.st_size
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
        or not raw.strip()
    ):
        raise HostAuthorityError("host_authority_boot_changed")
    return _sha(raw.strip())


def _physical_path(logical: Path, filesystem_root: Path) -> Path:
    if not logical.is_absolute() or ".." in logical.parts:
        raise HostAuthorityError("host_authority_path_invalid")
    if filesystem_root == Path("/"):
        return logical
    try:
        root = filesystem_root.resolve(strict=True)
    except OSError as exc:
        raise HostAuthorityError("host_authority_test_root_invalid") from exc
    if not root.is_dir():
        raise HostAuthorityError("host_authority_test_root_invalid")
    return root.joinpath(*logical.parts[1:])


def _read_regular(
    logical: Path,
    *,
    filesystem_root: Path,
    expected_uid: int | None,
    expected_gid: int | None,
    expected_mode: int | None,
) -> tuple[bytes, os.stat_result]:
    path = _physical_path(logical, filesystem_root)
    descriptor: int | None = None
    try:
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= MAX_FILE
            or (expected_uid is not None and before.st_uid != expected_uid)
            or (expected_gid is not None and before.st_gid != expected_gid)
            or (
                expected_mode is not None
                and stat.S_IMODE(before.st_mode) != expected_mode
            )
        ):
            raise HostAuthorityError("host_authority_file_identity_invalid")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = MAX_FILE + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        reachable = path.lstat()
    except HostAuthorityError:
        raise
    except OSError as exc:
        raise HostAuthorityError("host_authority_file_unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identity = lambda item: (
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
    if (
        len(raw) != before.st_size
        or len(raw) > MAX_FILE
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
        or identity(before) != identity(reachable)
    ):
        raise HostAuthorityError("host_authority_file_changed")
    return raw, before


def _target_pre_state(logical: Path, *, filesystem_root: Path) -> Mapping[str, Any]:
    path = _physical_path(logical, filesystem_root)
    try:
        path.lstat()
    except FileNotFoundError:
        return {
            "state": "absent",
            "sha256": None,
            "uid": None,
            "gid": None,
            "mode": None,
        }
    except OSError as exc:
        raise HostAuthorityError("host_authority_file_unavailable") from exc
    raw, state = _read_regular(
        logical,
        filesystem_root=filesystem_root,
        expected_uid=None,
        expected_gid=None,
        expected_mode=None,
    )
    return {
        "state": "present",
        "sha256": _sha(raw),
        "uid": state.st_uid,
        "gid": state.st_gid,
        "mode": stat.S_IMODE(state.st_mode),
    }


def build_host_authority_request(
    *,
    initial_collector_receipt: Mapping[str, Any],
    release_manifest_sha256: str,
    gateway_target_identity: Mapping[str, Any],
    writer_target_identity: Mapping[str, Any],
    connector_target_identity: Mapping[str, Any],
    host_transition: Mapping[str, Any],
    capability_topology: Mapping[str, Any],
    cron_continuity_plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    revision = initial_collector_receipt.get("release_revision")
    unsigned = {
        "schema": REQUEST_SCHEMA,
        "release_revision": revision,
        "initial_collector_receipt": copy.deepcopy(dict(initial_collector_receipt)),
        "release_manifest_sha256": release_manifest_sha256,
        "gateway_target_identity": copy.deepcopy(dict(gateway_target_identity)),
        "writer_target_identity": copy.deepcopy(dict(writer_target_identity)),
        "connector_target_identity": copy.deepcopy(dict(connector_target_identity)),
        "host_transition": copy.deepcopy(dict(host_transition)),
        "capability_topology": copy.deepcopy(dict(capability_topology)),
        "cron_continuity_plan": copy.deepcopy(dict(cron_continuity_plan)),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    value = {**unsigned, "request_sha256": _sha(_canonical(unsigned))}
    validate_host_authority_request(value)
    return value


def validate_host_authority_request(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REQUEST_FIELDS:
        raise HostAuthorityError("host_authority_request_fields_invalid")
    unsigned = {key: item for key, item in value.items() if key != "request_sha256"}
    revision = value.get("release_revision")
    initial = value.get("initial_collector_receipt")
    if (
        value.get("schema") != REQUEST_SCHEMA
        or not isinstance(revision, str)
        or package.REVISION.fullmatch(revision) is None
        or not isinstance(initial, Mapping)
        or initial.get("release_revision") != revision
        or _SHA256.fullmatch(str(value.get("release_manifest_sha256"))) is None
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("request_sha256") != _sha(_canonical(unsigned))
    ):
        raise HostAuthorityError("host_authority_request_identity_invalid")
    return copy.deepcopy(dict(value))


def _validate_transition_and_plan(
    request: Mapping[str, Any],
    *,
    initial: Mapping[str, Any],
) -> None:
    try:
        gateway_pre = cutover.ServiceObservation.from_mapping(initial["gateway_before"])
        writer_pre = cutover.ServiceObservation.from_mapping(initial["writer_before"])
        connector_pre = cutover.ServiceObservation.from_mapping(
            initial["connector_before"]
        )
        gateway_target = cutover._validate_target_service_identity(
            request["gateway_target_identity"], unit=cutover.GATEWAY_UNIT
        )
        writer_target = cutover._validate_target_service_identity(
            request["writer_target_identity"], unit=cutover.WRITER_UNIT
        )
        connector_target = cutover._validate_target_service_identity(
            request["connector_target_identity"], unit=cutover.CONNECTOR_UNIT
        )
        topology = validate_production_capability_topology(
            request["capability_topology"]
        )
        cutover._validate_host_transition(
            request["host_transition"],
            gateway_pre=gateway_pre,
            writer_pre=writer_pre,
            connector_pre=connector_pre,
            gateway_target=gateway_target,
            writer_target=writer_target,
            connector_target=connector_target,
            capability_topology=topology,
        )
        inventory = production_cron_migration.validate_inventory(
            initial["cron_inventory"]
        )
        host_facts = mechanical_job_rail.validate_host_facts(
            initial["mechanical_job_host_facts"]
        )
        rail = mechanical_job_rail.validate_package_manifest(
            initial["mechanical_job_package"],
            revision=request["release_revision"],
            host_facts_sha256=host_facts["host_facts_sha256"],
        )
        continuity = production_cron_migration.validate_owner_approved_plan(
            inventory,
            request["cron_continuity_plan"],
            rail["manifest_sha256"],
        )
        if continuity["cutover_executable"] is not True:
            raise ValueError("cron continuity is not executable")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise HostAuthorityError("host_authority_plan_invalid") from exc


def collect_host_authority(
    value: Mapping[str, Any],
    *,
    release_root: Path | None = None,
    filesystem_root: Path = Path("/"),
    unit_inputs: Mapping[str, Any] | None = None,
    require_root: bool = True,
    staged_uid: int | None = None,
    staged_gid: int | None = None,
    clock: Callable[[], float] = time.time,
    boot_reader: Callable[[], str] = _boot_id,
) -> Mapping[str, Any]:
    """Collect the complete fixed host authority without mutating host state."""

    request = validate_host_authority_request(value)
    revision = str(request["release_revision"])
    fixed_release = cutover.PRODUCTION_RELEASE_BASE / f"hermes-agent-{revision[:12]}"
    release = fixed_release if release_root is None else release_root
    if require_root:
        if (
            not sys.platform.startswith("linux")
            or os.geteuid() != 0  # windows-footgun: ok — Linux production/canary boundary
            or filesystem_root != Path("/")
            or release != fixed_release
        ):
            raise HostAuthorityError("host_authority_requires_linux_root")
        inputs = package.load_fixed_unit_inputs(revision=revision)
        trusted_uid = 0
        trusted_gid = 0
    else:
        if unit_inputs is None or staged_uid is None or staged_gid is None:
            raise HostAuthorityError("host_authority_test_inputs_required")
        inputs = package._unit_inputs(unit_inputs, revision=revision)
        trusted_uid = staged_uid
        trusted_gid = staged_gid

    # Import lazily to keep the owner launcher free to use this module as a
    # typed helper without an import cycle.
    from scripts.canary import production_cutover_owner_launcher as owner

    initial = owner.validate_initial_collector_receipt(
        request["initial_collector_receipt"],
        release_revision=revision,
        now_unix=int(clock()),
    )
    manifest = package.verify_release_artifacts(
        release,
        revision,
        release_address=fixed_release,
        unit_inputs=inputs,
    )
    contract = manifest.get("host_artifact_contract")
    packaged_inputs = manifest.get("unit_inputs")
    transition = request["host_transition"]
    if (
        manifest.get("manifest_sha256") != request["release_manifest_sha256"]
        or not isinstance(contract, Mapping)
        or contract.get("schema") != package.HOST_ARTIFACT_CONTRACT_SCHEMA
        or contract.get("required_file_count") != len(package.HOST_ARTIFACT_TARGETS)
        or contract.get("all_files_require_readback") is not True
        or _SHA256.fullmatch(str(contract.get("contract_sha256"))) is None
        or not isinstance(packaged_inputs, Mapping)
        or packaged_inputs.get("writer_capability_public_key_id")
        != transition["discord_key_foundation"]["writer"]["public_key_id"]
        or packaged_inputs.get("operational_edge_key_foundation_sha256")
        != transition["operational_edge_key_foundation_sha256"]
        or packaged_inputs.get("operational_edge_receipt_public_key_ids")
        != transition["operational_edge_receipt_public_key_ids"]
        or packaged_inputs.get("release_owner_uid")
        != transition["release_owner_uid"]
        or packaged_inputs.get("release_owner_gid")
        != transition["release_owner_gid"]
    ):
        raise HostAuthorityError("host_authority_release_contract_invalid")
    files = request["host_transition"].get("files")
    contract_files = contract.get("files")
    if (
        not isinstance(files, Mapping)
        or set(files) != set(package.HOST_ARTIFACT_TARGETS)
        or not isinstance(contract_files, Mapping)
        or set(contract_files) != set(package.HOST_ARTIFACT_TARGETS)
    ):
        raise HostAuthorityError("host_authority_file_set_invalid")

    boot_before = boot_reader()
    readback: list[Mapping[str, Any]] = []
    staged_identities: list[tuple[Path, str]] = []
    for name in sorted(package.HOST_ARTIFACT_TARGETS):
        item = files[name]
        bound = contract_files[name]
        if (
            not isinstance(item, Mapping)
            or not isinstance(bound, Mapping)
            or set(item)
            != {
                "staged_path",
                "target_path",
                "sha256",
                "uid",
                "gid",
                "mode",
                "pre",
            }
            or bound.get("staged_path") != item.get("staged_path")
            or bound.get("target_path") != item.get("target_path")
            or bound.get("required_readback") is not True
            or bound.get("actual_sha256_bound_by") != RECEIPT_SCHEMA
            or _SHA256.fullmatch(str(item.get("sha256"))) is None
            or (
                bound.get("package_sha256") is not None
                and bound.get("package_sha256") != item.get("sha256")
            )
        ):
            raise HostAuthorityError("host_authority_file_binding_invalid")
        staged_raw, staged_state = _read_regular(
            Path(str(item["staged_path"])),
            filesystem_root=filesystem_root,
            expected_uid=trusted_uid,
            expected_gid=trusted_gid,
            expected_mode=0o400,
        )
        if _sha(staged_raw) != item["sha256"]:
            raise HostAuthorityError("host_authority_staged_digest_invalid")
        observed_pre = _target_pre_state(
            Path(str(item["target_path"])), filesystem_root=filesystem_root
        )
        if observed_pre != item["pre"]:
            raise HostAuthorityError("host_authority_target_pre_state_drifted")
        readback.append({
            "name": name,
            "sha256": item["sha256"],
            "size": len(staged_raw),
            "staged_uid": staged_state.st_uid,
            "staged_gid": staged_state.st_gid,
            "staged_mode": stat.S_IMODE(staged_state.st_mode),
            "target_pre": observed_pre,
        })
        staged_identities.append((
            Path(str(item["staged_path"])),
            str(item["sha256"]),
        ))
    # Reject a mixed snapshot assembled while one of the sequentially read
    # staged files changed.  Every fixed file is read and hashed a second time
    # before the aggregate receipt is authored.
    for staged_path, digest in staged_identities:
        repeated, _state = _read_regular(
            staged_path,
            filesystem_root=filesystem_root,
            expected_uid=trusted_uid,
            expected_gid=trusted_gid,
            expected_mode=0o400,
        )
        if _sha(repeated) != digest:
            raise HostAuthorityError("host_authority_staged_digest_invalid")
    _validate_transition_and_plan(request, initial=initial)
    boot_after = boot_reader()
    if (
        boot_before != boot_after
        or _SHA256.fullmatch(boot_after or "") is None
        or boot_after != initial["source_boot_id_sha256"]
    ):
        raise HostAuthorityError("host_authority_boot_changed")
    observed_at = int(clock())
    if (
        observed_at < initial["observed_at_unix"]
        or observed_at - initial["observed_at_unix"] > MAX_AGE_SECONDS
    ):
        raise HostAuthorityError("host_authority_initial_receipt_stale")
    unsigned = {
        "schema": RECEIPT_SCHEMA,
        "release_revision": revision,
        "request_sha256": request["request_sha256"],
        "initial_collector_receipt_sha256": initial["receipt_sha256"],
        "release_manifest_sha256": manifest["manifest_sha256"],
        "host_artifact_contract_sha256": contract["contract_sha256"],
        "gateway_target_identity": copy.deepcopy(request["gateway_target_identity"]),
        "writer_target_identity": copy.deepcopy(request["writer_target_identity"]),
        "connector_target_identity": copy.deepcopy(
            request["connector_target_identity"]
        ),
        "host_transition": copy.deepcopy(request["host_transition"]),
        "capability_topology": copy.deepcopy(request["capability_topology"]),
        "cron_continuity_plan": copy.deepcopy(request["cron_continuity_plan"]),
        "readback_file_count": len(readback),
        "readback_files": readback,
        "readback_set_sha256": _sha(_canonical({"files": readback})),
        "observed_at_unix": observed_at,
        "source_boot_id_sha256": boot_after,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha(_canonical(unsigned))}


def validate_host_authority_receipt(
    value: Any,
    *,
    host_authority_request: Mapping[str, Any],
    initial_collector_receipt: Mapping[str, Any],
    release_revision: str,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    current = int(time.time()) if now_unix is None else now_unix
    if not isinstance(value, Mapping) or set(value) != _RECEIPT_FIELDS:
        raise HostAuthorityError("host_authority_receipt_fields_invalid")
    request = validate_host_authority_request(host_authority_request)
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    readback = value.get("readback_files")
    transition = value.get("host_transition")
    transition_files = (
        transition.get("files") if isinstance(transition, Mapping) else None
    )
    if (
        value.get("schema") != RECEIPT_SCHEMA
        or value.get("release_revision") != release_revision
        or request.get("release_revision") != release_revision
        or request.get("initial_collector_receipt")
        != initial_collector_receipt
        or value.get("request_sha256") != request.get("request_sha256")
        or value.get("initial_collector_receipt_sha256")
        != initial_collector_receipt.get("receipt_sha256")
        or value.get("release_manifest_sha256")
        != request.get("release_manifest_sha256")
        or value.get("gateway_target_identity")
        != request.get("gateway_target_identity")
        or value.get("writer_target_identity")
        != request.get("writer_target_identity")
        or value.get("connector_target_identity")
        != request.get("connector_target_identity")
        or value.get("host_transition") != request.get("host_transition")
        or value.get("capability_topology")
        != request.get("capability_topology")
        or value.get("cron_continuity_plan")
        != request.get("cron_continuity_plan")
        or value.get("source_boot_id_sha256")
        != initial_collector_receipt.get("source_boot_id_sha256")
        or type(value.get("readback_file_count")) is not int
        or value.get("readback_file_count") != len(package.HOST_ARTIFACT_TARGETS)
        or not isinstance(readback, list)
        or len(readback) != len(package.HOST_ARTIFACT_TARGETS)
        or not isinstance(transition_files, Mapping)
        or _SHA256.fullmatch(str(value.get("release_manifest_sha256"))) is None
        or _SHA256.fullmatch(str(value.get("host_artifact_contract_sha256"))) is None
        or _SHA256.fullmatch(str(value.get("readback_set_sha256"))) is None
        or type(value.get("observed_at_unix")) is not int
        or not current - MAX_AGE_SECONDS <= value["observed_at_unix"] <= current + 30
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("receipt_sha256") != _sha(_canonical(unsigned))
    ):
        raise HostAuthorityError("host_authority_receipt_identity_invalid")
    expected_names = sorted(package.HOST_ARTIFACT_TARGETS)
    for expected_name, item in zip(expected_names, readback, strict=True):
        transition_item = transition_files.get(expected_name)
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "name",
                "sha256",
                "size",
                "staged_uid",
                "staged_gid",
                "staged_mode",
                "target_pre",
            }
            or item.get("name") != expected_name
            or not isinstance(transition_item, Mapping)
            or item.get("sha256") != transition_item.get("sha256")
            or type(item.get("size")) is not int
            or not 0 < item["size"] <= MAX_FILE
            or type(item.get("staged_uid")) is not int
            or item["staged_uid"] < 0
            or type(item.get("staged_gid")) is not int
            or item["staged_gid"] < 0
            or item.get("staged_mode") != 0o400
            or item.get("target_pre") != transition_item.get("pre")
        ):
            raise HostAuthorityError("host_authority_readback_invalid")
    if value.get("readback_set_sha256") != _sha(_canonical({"files": readback})):
        raise HostAuthorityError("host_authority_readback_invalid")
    request_like = {
        "release_revision": release_revision,
        "gateway_target_identity": value["gateway_target_identity"],
        "writer_target_identity": value["writer_target_identity"],
        "connector_target_identity": value["connector_target_identity"],
        "host_transition": value["host_transition"],
        "capability_topology": value["capability_topology"],
        "cron_continuity_plan": value["cron_continuity_plan"],
    }
    _validate_transition_and_plan(
        request_like,
        initial=initial_collector_receipt,
    )
    return copy.deepcopy(dict(value))


def compose_full_authority_receipt(
    *,
    initial_collector_receipt: Mapping[str, Any],
    host_authority_request: Mapping[str, Any],
    host_authority_receipt: Mapping[str, Any],
    release_revision: str,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Compose the two independently validated read-only receipts."""

    from scripts.canary import production_cutover_owner_launcher as owner

    current = int(time.time()) if now_unix is None else now_unix
    initial = owner.validate_initial_collector_receipt(
        initial_collector_receipt,
        release_revision=release_revision,
        now_unix=current,
    )
    host = validate_host_authority_receipt(
        host_authority_receipt,
        host_authority_request=host_authority_request,
        initial_collector_receipt=initial,
        release_revision=release_revision,
        now_unix=current,
    )
    observed_at = max(initial["observed_at_unix"], host["observed_at_unix"])
    unsigned = {
        "schema": owner.COLLECTOR_SCHEMA,
        "release_revision": release_revision,
        "target": copy.deepcopy(initial["target"]),
        "artifacts": copy.deepcopy(initial["artifacts"]),
        "gateway_before": copy.deepcopy(initial["gateway_before"]),
        "writer_before": copy.deepcopy(initial["writer_before"]),
        "connector_before": copy.deepcopy(initial["connector_before"]),
        "gateway_target_identity": copy.deepcopy(host["gateway_target_identity"]),
        "writer_target_identity": copy.deepcopy(host["writer_target_identity"]),
        "connector_target_identity": copy.deepcopy(host["connector_target_identity"]),
        "host_transition": copy.deepcopy(host["host_transition"]),
        "capability_topology": copy.deepcopy(host["capability_topology"]),
        "initial_snapshot": copy.deepcopy(initial["initial_snapshot"]),
        "cron_inventory": copy.deepcopy(initial["cron_inventory"]),
        "cron_continuity_plan": copy.deepcopy(host["cron_continuity_plan"]),
        "mechanical_job_host_facts": copy.deepcopy(
            initial["mechanical_job_host_facts"]
        ),
        "mechanical_job_package": copy.deepcopy(initial["mechanical_job_package"]),
        "observed_at_unix": observed_at,
        "source_boot_id_sha256": initial["source_boot_id_sha256"],
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    full = {**unsigned, "receipt_sha256": _sha(_canonical(unsigned))}
    return owner.validate_collector_receipt(
        full,
        release_revision=release_revision,
        now_unix=current,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise HostAuthorityError("host_authority_arguments_forbidden")
    raw = sys.stdin.buffer.read(MAX_INPUT + 1)
    try:
        if not raw or len(raw) > MAX_INPUT or raw.endswith(b"\n"):
            raise HostAuthorityError("host_authority_input_invalid")
        receipt = collect_host_authority(_decode(raw))
    except (HostAuthorityError, OSError, TypeError, ValueError):
        print(
            '{"error_code":"production_host_authority_failed","ok":false}',
            file=sys.stderr,
        )
        return 2
    print(_canonical(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
