#!/usr/bin/env python3
"""Digest-bound boot-disk expansion for the isolated Muncho canary.

The first mutation in this gate is the one-way expansion of the existing,
identity-pinned boot disk from 20 GB to 40 GB.  Live evidence proved that the
pinned Debian image expands its root partition/filesystem during boot, not
immediately while the VM remains online.  ``host_storage_boot`` therefore owns
the separate exact stop/start receipt when expansion is required.  This module
still grants no guest command or repair authority, and final readiness binds
that boot receipt to a fresh read-only filesystem postflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from scripts.canary.foundation import PREFLIGHT_MAX_AGE_SECONDS, PROJECT, ZONE, PlanStep
from scripts.canary.host import BOOT_DISK_TYPE, IMAGE, IMAGE_PROJECT, VM_NAME


STORAGE_PREFLIGHT_SCHEMA = "muncho-isolated-canary-host-storage-preflight.v1"
STORAGE_PLAN_SCHEMA = "muncho-isolated-canary-host-storage-plan.v1"
STORAGE_APPLY_RECEIPT_SCHEMA = "muncho-isolated-canary-host-storage-apply-receipt.v1"
STORAGE_READINESS_RECEIPT_SCHEMA = (
    "muncho-isolated-canary-host-storage-readiness-receipt.v2"
)
LEGACY_BOOT_RECEIPT_SCHEMA = "muncho-isolated-canary-storage-boot-expansion-receipt.v1"
LEGACY_READINESS_RECEIPT_SCHEMA = (
    "muncho-isolated-canary-host-storage-readiness-receipt.v1"
)
LEGACY_BOOT_OPERATION_PLAN_SHA256 = (
    "1d0f35d8c9de478278e515823709abeb894c5890e68210e3c0346ef66406cad9"
)
LEGACY_APPLY_RECEIPT_SHA256 = (
    "b2ada08f473cf67dee9c738852373d19d9dba473fdc8ee3cdf834f14d951dbd5"
)
LEGACY_BOOT_RECEIPT_SHA256 = (
    "7430a8e859c6f24261ef89182dc49c64c2f5832b50e8918bdae8008b8c0a0cb8"
)
LEGACY_READINESS_RECEIPT_SHA256 = (
    "de2da85103d578073b795828fbad2db2a602d63cbb079352e9c90e74d6400777"
)
LEGACY_OWNER_RECEIPT_ROOT = Path(
    "/Users/emillomliev/.hermes/owner-approvals/canary-storage"
)
LEGACY_APPLY_RECEIPT_PATH = LEGACY_OWNER_RECEIPT_ROOT / (
    f"apply-{LEGACY_APPLY_RECEIPT_SHA256}.json"
)
LEGACY_BOOT_RECEIPT_PATH = LEGACY_OWNER_RECEIPT_ROOT / (
    f"boot-expansion-{LEGACY_BOOT_RECEIPT_SHA256}.json"
)
LEGACY_READINESS_RECEIPT_PATH = LEGACY_OWNER_RECEIPT_ROOT / (
    f"readiness-{LEGACY_READINESS_RECEIPT_SHA256}.json"
)
_OWNER_UID = 501
_OWNER_GID = 20
_LEGACY_RECEIPT_MAX_BYTES = 16 * 1024
OWNER_ACCOUNT = "lomliev@adventico.com"
TRUSTED_GCLOUD = (
    "/Users/emillomliev/.hermes/trusted/google-cloud-sdk-569.0.0/bin/gcloud"
)
TRUSTED_PYTHON = (
    "/Users/emillomliev/.local/share/uv/python/"
    "cpython-3.11.15-macos-aarch64-none/bin/python3.11"
)
VM_INSTANCE_ID = "9153645328899914617"
DISK_NAME = VM_NAME
DISK_ID = "4195397669213846393"
BOOT_DEVICE_NAME = "persistent-disk-0"
SOURCE_BOOT_DISK_SIZE_GB = 20
TARGET_BOOT_DISK_SIZE_GB = 40
MINIMUM_PACKAGING_FREE_BYTES = 8 * 1024 * 1024 * 1024
MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES = 39_000_000_000
MAXIMUM_TARGET_ROOT_FILESYSTEM_BYTES = TARGET_BOOT_DISK_SIZE_GB * 1024**3
ROOT_SOURCE = "/dev/sda1"
ROOT_FILESYSTEM = "ext4"
ROOT_MOUNTPOINT = "/"
RESIZE_STEP = "resize_canary_boot_disk_to_40gb"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PREFLIGHT_FIELDS = {
    "schema",
    "ok",
    "state",
    "collected_at_unix",
    "plan_sha256",
    "vm_instance_id",
    "disk_id",
    "disk_size_gb",
    "root_source",
    "root_filesystem",
    "root_mountpoint",
    "root_size_bytes",
    "root_available_bytes",
    "normal_host_preflight_ok",
    "host_report_sha256",
    "satisfied_steps",
    "ready_for_packaging",
    "checks",
    "report_sha256",
}
_APPLY_RECEIPT_FIELDS = {
    "schema",
    "ok",
    "plan_sha256",
    "approved_plan_sha256",
    "preflight_report_sha256",
    "completed_at_unix",
    "mutation_performed",
    "receipts",
    "filesystem_ready",
    "ready_for_packaging",
    "requires_post_apply_attestation",
    "receipt_sha256",
}
_LEGACY_BOOT_FIELDS = {
    "schema",
    "ok",
    "operation_plan_sha256",
    "parent_storage_plan_sha256",
    "parent_apply_receipt_sha256",
    "vm_instance_id",
    "disk_id",
    "pre_instance_sha256",
    "stopped_instance_sha256",
    "started_instance_sha256",
    "stop_receipt",
    "start_receipt",
    "completed_at_unix",
    "filesystem_ready",
    "requires_storage_postflight",
    "opens_runtime_gate",
    "secret_material_recorded",
    "receipt_sha256",
}
_LEGACY_READINESS_FIELDS = {
    "schema",
    "ok",
    "plan_sha256",
    "apply_receipt_sha256",
    "postflight_report_sha256",
    "attested_at_unix",
    "vm_instance_id",
    "disk_id",
    "disk_size_gb",
    "root_source",
    "root_filesystem",
    "root_mountpoint",
    "root_size_bytes",
    "root_available_bytes",
    "filesystem_ready",
    "ready_for_packaging",
    "opens_runtime_gate",
    "receipt_sha256",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _seal(unsigned: Mapping[str, object], *, digest_key: str) -> dict[str, object]:
    value = dict(unsigned)
    value[digest_key] = _sha256_json(value)
    return value


def _require_sealed(
    value: Mapping[str, object], *, digest_key: str, label: str
) -> None:
    digest = value.get(digest_key)
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise RuntimeError(f"{label} digest is invalid")
    unsigned = dict(value)
    del unsigned[digest_key]
    if _sha256_json(unsigned) != digest:
        raise RuntimeError(f"{label} digest mismatch")


@dataclass(frozen=True)
class HostStorageSpec:
    project: str = PROJECT
    zone: str = ZONE
    owner_account: str = OWNER_ACCOUNT
    vm_name: str = VM_NAME
    vm_instance_id: str = VM_INSTANCE_ID
    disk_name: str = DISK_NAME
    disk_id: str = DISK_ID
    boot_device_name: str = BOOT_DEVICE_NAME
    source_size_gb: int = SOURCE_BOOT_DISK_SIZE_GB
    target_size_gb: int = TARGET_BOOT_DISK_SIZE_GB
    disk_type: str = BOOT_DISK_TYPE
    source_image_project: str = IMAGE_PROJECT
    source_image: str = IMAGE
    root_source: str = ROOT_SOURCE
    root_filesystem: str = ROOT_FILESYSTEM
    root_mountpoint: str = ROOT_MOUNTPOINT
    minimum_target_root_filesystem_bytes: int = MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES
    minimum_packaging_free_bytes: int = MINIMUM_PACKAGING_FREE_BYTES

    def validate(self) -> None:
        if self != HostStorageSpec():
            raise ValueError("host storage shape is fork-pinned")


@dataclass(frozen=True)
class HostStoragePlan:
    schema: str
    spec: HostStorageSpec
    architecture: Mapping[str, object]
    steps: tuple[PlanStep, ...]

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "spec": asdict(self.spec),
            "architecture": dict(self.architecture),
            "steps": [asdict(step) for step in self.steps],
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.payload())

    def report(self) -> dict[str, object]:
        return {**self.payload(), "plan_sha256": self.sha256}


def build_plan(spec: HostStorageSpec | None = None) -> HostStoragePlan:
    exact = HostStorageSpec() if spec is None else spec
    exact.validate()
    return HostStoragePlan(
        schema=STORAGE_PLAN_SCHEMA,
        spec=exact,
        architecture={
            "phase": "host_storage_expansion",
            "single_mutation": True,
            "one_way_expansion_only": True,
            "creates_disk": False,
            "deletes_disk": False,
            "creates_snapshot": False,
            "mutates_iam": False,
            "mutates_guest": False,
            # This approved-plan field means image-owned expansion rather than
            # an operator guest command. Live proof narrowed it to boot-time;
            # keep the payload byte-stable and bind host_storage_boot instead.
            "public_image_automatic_root_expansion": True,
            "requires_fresh_read_only_preflight": True,
            "requires_fresh_read_only_postflight": True,
            "opens_runtime_gate": False,
        },
        steps=(
            PlanStep(
                RESIZE_STEP,
                (
                    "gcloud",
                    "compute",
                    "disks",
                    "resize",
                    exact.disk_name,
                    f"--project={exact.project}",
                    f"--zone={exact.zone}",
                    f"--account={exact.owner_account}",
                    f"--size={exact.target_size_gb}GB",
                    "--quiet",
                ),
            ),
        ),
    )


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run only the reviewed resize through the sealed owner gcloud runtime."""

    expected = build_plan().steps[0].argv
    if tuple(argv) != expected:
        raise RuntimeError("storage mutation argv is not exact")

    # Imported lazily so the pure plan builder remains lightweight and the
    # mutation path reuses the same byte-stable SDK/config identity boundary as
    # the read-only IAP collector.  A bare PATH-resolved gcloud is never used.
    from scripts.canary.full_canary_owner_launcher import (
        GcloudOwnerAccessToken,
        PinnedGcloudConfiguration,
        TrustedGcloudExecutable,
        _owner_gcloud_environment,
    )

    executable = TrustedGcloudExecutable(
        candidates=(TRUSTED_GCLOUD,),
        python_candidates=(TRUSTED_PYTHON,),
    )
    configuration = PinnedGcloudConfiguration()
    owner_identity = GcloudOwnerAccessToken(
        gcloud_executable=executable,
        gcloud_configuration=configuration,
    )
    account = owner_identity.account_for_read_only_preflight()
    if account != OWNER_ACCOUNT:
        raise RuntimeError("storage mutation owner identity is not exact")
    prefix = executable.trusted_command_prefix()
    environment = _owner_gcloud_environment(configuration, prefix[0])
    try:
        completed = subprocess.run(
            (*prefix, *tuple(argv)[1:]),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environment),
            shell=False,
            timeout=600,
        )
    finally:
        executable.trusted_command_prefix()
        configuration.assert_stable()
        owner_identity.require_stable()
    if (
        not isinstance(completed.stdout, bytes)
        or not isinstance(completed.stderr, bytes)
        or len(completed.stdout) > 64 * 1024
        or len(completed.stderr) > 64 * 1024
    ):
        raise RuntimeError("storage mutation output is invalid")
    try:
        stdout = completed.stdout.decode("utf-8", errors="strict")
        stderr = completed.stderr.decode("utf-8", errors="strict")
    except UnicodeError:
        raise RuntimeError("storage mutation output is invalid") from None
    return subprocess.CompletedProcess(
        args=list(argv),
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _require_fresh_report(
    report: Mapping[str, object],
    *,
    plan: HostStoragePlan,
    now_unix: int,
) -> None:
    if set(report) != _PREFLIGHT_FIELDS:
        raise RuntimeError("storage preflight fields are not exact")
    _require_sealed(report, digest_key="report_sha256", label="storage preflight")
    collected_at = report.get("collected_at_unix")
    if (
        report.get("schema") != STORAGE_PREFLIGHT_SCHEMA
        or report.get("plan_sha256") != plan.sha256
        or report.get("vm_instance_id") != plan.spec.vm_instance_id
        or report.get("disk_id") != plan.spec.disk_id
        or not isinstance(report.get("host_report_sha256"), str)
        or _SHA256.fullmatch(str(report.get("host_report_sha256"))) is None
        or type(collected_at) is not int
        or now_unix < collected_at
        or now_unix - collected_at > PREFLIGHT_MAX_AGE_SECONDS
    ):
        raise RuntimeError("storage preflight is stale or not exact")


def _successful_check_names(report: Mapping[str, object]) -> set[str]:
    raw_checks = report.get("checks")
    if not isinstance(raw_checks, list):
        raise RuntimeError("storage preflight check inventory is invalid")
    names: set[str] = set()
    for item in raw_checks:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"name", "passed", "detail"}
            or not isinstance(item.get("name"), str)
            or item.get("passed") is not True
            or not isinstance(item.get("detail"), str)
            or not item["detail"]
            or item["name"] in names
        ):
            raise RuntimeError("storage preflight check inventory is invalid")
        names.add(item["name"])
    return names


def execute_plan(
    plan: HostStoragePlan,
    *,
    approved_plan_sha256: str,
    preflight: Mapping[str, object],
    runner: Runner = _runner,
    now_unix: int | None = None,
) -> dict[str, object]:
    """Apply only the exact disk expansion; never claim filesystem readiness."""

    if approved_plan_sha256 != plan.sha256:
        raise RuntimeError("approved storage plan digest mismatch")
    now = int(time.time()) if now_unix is None else now_unix
    if type(now) is not int or now < 0:
        raise RuntimeError("storage apply time is invalid")
    _require_fresh_report(preflight, plan=plan, now_unix=now)
    if preflight.get("ok") is not True:
        raise RuntimeError("storage preflight did not pass")
    expected_checks = {
        "host.lifecycle_state_exact",
        "instance.identity_exact",
        "disk.identity_exact",
        "disk.size_transition_exact",
        "filesystem.root_identity_exact",
        "filesystem.capacity_matches_disk_state",
        "filesystem.packaging_headroom_if_target",
    }
    if _successful_check_names(preflight) != expected_checks:
        raise RuntimeError("storage preflight check set is not exact")
    raw_satisfied = preflight.get("satisfied_steps")
    if raw_satisfied not in ([], [RESIZE_STEP]):
        raise RuntimeError("storage preflight satisfied-step inventory is invalid")
    state = preflight.get("state")
    if (state, raw_satisfied) not in (
        ("source_ready", []),
        ("target_ready", [RESIZE_STEP]),
    ):
        raise RuntimeError("storage preflight transition state is invalid")

    mutation_performed = not raw_satisfied
    if mutation_performed:
        completed = runner(plan.steps[0].argv)
        step_receipt = {
            "name": RESIZE_STEP,
            "result": "resized" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        }
        ok = completed.returncode == 0
    else:
        step_receipt = {"name": RESIZE_STEP, "result": "verified_existing"}
        ok = True

    unsigned = {
        "schema": STORAGE_APPLY_RECEIPT_SCHEMA,
        "ok": ok,
        "plan_sha256": plan.sha256,
        "approved_plan_sha256": approved_plan_sha256,
        "preflight_report_sha256": preflight["report_sha256"],
        "completed_at_unix": now,
        "mutation_performed": mutation_performed,
        "receipts": [step_receipt],
        "filesystem_ready": False,
        "ready_for_packaging": False,
        "requires_post_apply_attestation": ok,
    }
    return _seal(unsigned, digest_key="receipt_sha256")


def _legacy_command_receipt_exact(value: object) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"returncode", "stdout_sha256", "stderr_sha256"}
        and value.get("returncode") == 0
        and isinstance(value.get("stdout_sha256"), str)
        and _SHA256.fullmatch(str(value.get("stdout_sha256"))) is not None
        and isinstance(value.get("stderr_sha256"), str)
        and _SHA256.fullmatch(str(value.get("stderr_sha256"))) is not None
    )


def _validate_legacy_live_receipts(
    *,
    plan: HostStoragePlan,
    apply_receipt: Mapping[str, object],
    legacy_boot_receipt: Mapping[str, object],
    legacy_readiness_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact completed live transition without inventing evidence."""

    if set(legacy_boot_receipt) != _LEGACY_BOOT_FIELDS:
        raise RuntimeError("legacy storage boot receipt fields are not exact")
    if set(legacy_readiness_receipt) != _LEGACY_READINESS_FIELDS:
        raise RuntimeError("legacy storage readiness receipt fields are not exact")
    _require_sealed(
        legacy_boot_receipt,
        digest_key="receipt_sha256",
        label="legacy storage boot receipt",
    )
    _require_sealed(
        legacy_readiness_receipt,
        digest_key="receipt_sha256",
        label="legacy storage readiness receipt",
    )
    apply_time = apply_receipt.get("completed_at_unix")
    boot_time = legacy_boot_receipt.get("completed_at_unix")
    readiness_time = legacy_readiness_receipt.get("attested_at_unix")
    pre_instance = legacy_boot_receipt.get("pre_instance_sha256")
    stopped_instance = legacy_boot_receipt.get("stopped_instance_sha256")
    started_instance = legacy_boot_receipt.get("started_instance_sha256")
    legacy_root_size = legacy_readiness_receipt.get("root_size_bytes")
    legacy_root_available = legacy_readiness_receipt.get("root_available_bytes")
    if (
        legacy_boot_receipt.get("schema") != LEGACY_BOOT_RECEIPT_SCHEMA
        or apply_receipt.get("receipt_sha256") != LEGACY_APPLY_RECEIPT_SHA256
        or legacy_boot_receipt.get("receipt_sha256") != LEGACY_BOOT_RECEIPT_SHA256
        or legacy_boot_receipt.get("ok") is not True
        or legacy_boot_receipt.get("operation_plan_sha256")
        != LEGACY_BOOT_OPERATION_PLAN_SHA256
        or legacy_boot_receipt.get("parent_storage_plan_sha256") != plan.sha256
        or legacy_boot_receipt.get("parent_apply_receipt_sha256")
        != apply_receipt.get("receipt_sha256")
        or legacy_boot_receipt.get("vm_instance_id") != plan.spec.vm_instance_id
        or legacy_boot_receipt.get("disk_id") != plan.spec.disk_id
        or not all(
            isinstance(value, str) and _SHA256.fullmatch(value) is not None
            for value in (pre_instance, stopped_instance, started_instance)
        )
        or pre_instance != started_instance
        or stopped_instance == pre_instance
        or not _legacy_command_receipt_exact(legacy_boot_receipt.get("stop_receipt"))
        or not _legacy_command_receipt_exact(legacy_boot_receipt.get("start_receipt"))
        or legacy_boot_receipt.get("filesystem_ready") is not False
        or legacy_boot_receipt.get("requires_storage_postflight") is not True
        or legacy_boot_receipt.get("opens_runtime_gate") is not False
        or legacy_boot_receipt.get("secret_material_recorded") is not False
        or legacy_readiness_receipt.get("schema") != LEGACY_READINESS_RECEIPT_SCHEMA
        or legacy_readiness_receipt.get("receipt_sha256")
        != LEGACY_READINESS_RECEIPT_SHA256
        or legacy_readiness_receipt.get("ok") is not True
        or legacy_readiness_receipt.get("plan_sha256") != plan.sha256
        or legacy_readiness_receipt.get("apply_receipt_sha256")
        != apply_receipt.get("receipt_sha256")
        or not isinstance(legacy_readiness_receipt.get("postflight_report_sha256"), str)
        or _SHA256.fullmatch(
            str(legacy_readiness_receipt.get("postflight_report_sha256"))
        )
        is None
        or legacy_readiness_receipt.get("vm_instance_id") != plan.spec.vm_instance_id
        or legacy_readiness_receipt.get("disk_id") != plan.spec.disk_id
        or legacy_readiness_receipt.get("disk_size_gb") != plan.spec.target_size_gb
        or legacy_readiness_receipt.get("root_source") != plan.spec.root_source
        or legacy_readiness_receipt.get("root_filesystem") != plan.spec.root_filesystem
        or legacy_readiness_receipt.get("root_mountpoint") != plan.spec.root_mountpoint
        or type(legacy_root_size) is not int
        or not plan.spec.minimum_target_root_filesystem_bytes
        <= legacy_root_size
        <= MAXIMUM_TARGET_ROOT_FILESYSTEM_BYTES
        or type(legacy_root_available) is not int
        or not plan.spec.minimum_packaging_free_bytes
        <= legacy_root_available
        <= legacy_root_size
        or legacy_readiness_receipt.get("filesystem_ready") is not True
        or legacy_readiness_receipt.get("ready_for_packaging") is not True
        or legacy_readiness_receipt.get("opens_runtime_gate") is not False
        or type(apply_time) is not int
        or type(boot_time) is not int
        or type(readiness_time) is not int
        or not 0 <= apply_time <= boot_time <= readiness_time
    ):
        raise RuntimeError("legacy live storage receipts are not exact")
    return {
        "boot_completed_at_unix": boot_time,
        "readiness_attested_at_unix": readiness_time,
        "legacy_boot_receipt_sha256": LEGACY_BOOT_RECEIPT_SHA256,
        "legacy_readiness_receipt_sha256": LEGACY_READINESS_RECEIPT_SHA256,
    }


def build_readiness_receipt(
    plan: HostStoragePlan,
    *,
    apply_receipt: Mapping[str, object],
    postflight: Mapping[str, object],
    boot_preflight: Mapping[str, object] | None = None,
    boot_receipt: Mapping[str, object] | None = None,
    legacy_boot_receipt: Mapping[str, object] | None = None,
    legacy_readiness_receipt: Mapping[str, object] | None = None,
    now_unix: int | None = None,
) -> dict[str, object]:
    """Bind target filesystem facts to the apply and any required boot."""

    now = int(time.time()) if now_unix is None else now_unix
    if type(now) is not int or now < 0:
        raise RuntimeError("storage readiness time is invalid")
    _require_sealed(
        apply_receipt,
        digest_key="receipt_sha256",
        label="storage apply receipt",
    )
    if set(apply_receipt) != _APPLY_RECEIPT_FIELDS:
        raise RuntimeError("storage apply receipt fields are not exact")
    completed_at = apply_receipt.get("completed_at_unix")
    raw_step_receipts = apply_receipt.get("receipts")
    mutation_performed = apply_receipt.get("mutation_performed")
    steps_exact = False
    if isinstance(raw_step_receipts, list) and len(raw_step_receipts) == 1:
        step = raw_step_receipts[0]
        if mutation_performed is True:
            steps_exact = bool(
                isinstance(step, Mapping)
                and set(step)
                == {
                    "name",
                    "result",
                    "returncode",
                    "stdout_sha256",
                    "stderr_sha256",
                }
                and step.get("name") == RESIZE_STEP
                and step.get("result") == "resized"
                and step.get("returncode") == 0
                and isinstance(step.get("stdout_sha256"), str)
                and _SHA256.fullmatch(str(step.get("stdout_sha256"))) is not None
                and isinstance(step.get("stderr_sha256"), str)
                and _SHA256.fullmatch(str(step.get("stderr_sha256"))) is not None
            )
        elif mutation_performed is False:
            steps_exact = step == {
                "name": RESIZE_STEP,
                "result": "verified_existing",
            }
    if (
        apply_receipt.get("schema") != STORAGE_APPLY_RECEIPT_SCHEMA
        or apply_receipt.get("ok") is not True
        or apply_receipt.get("plan_sha256") != plan.sha256
        or apply_receipt.get("approved_plan_sha256") != plan.sha256
        or apply_receipt.get("requires_post_apply_attestation") is not True
        or apply_receipt.get("filesystem_ready") is not False
        or apply_receipt.get("ready_for_packaging") is not False
        or type(mutation_performed) is not bool
        or not steps_exact
        or type(completed_at) is not int
        or completed_at < 0
        or completed_at > now
    ):
        raise RuntimeError("storage apply receipt is not terminal-success exact")
    boot_expansion_required = mutation_performed is True
    validated_boot: Mapping[str, object] | None = None
    boot_plan_sha256: str | None = None
    boot_preflight_sha256: str | None = None
    boot_receipt_sha256: str | None = None
    legacy_boot_sha256: str | None = None
    legacy_readiness_sha256: str | None = None
    boot_evidence_kind = "not_required_existing_target"
    preboot_boot_id_evidence: bool | None = None
    preboot_service_state_evidence: bool | None = None
    follow_on_stopped_runtime_gate_required = False
    terminal_mutation_time = completed_at
    normal_boot_requested = boot_preflight is not None or boot_receipt is not None
    legacy_boot_requested = (
        legacy_boot_receipt is not None or legacy_readiness_receipt is not None
    )
    if normal_boot_requested and legacy_boot_requested:
        raise RuntimeError(
            "storage readiness boot evidence modes are mutually exclusive"
        )
    if boot_expansion_required:
        if legacy_boot_requested:
            if legacy_boot_receipt is None or legacy_readiness_receipt is None:
                raise RuntimeError(
                    "legacy storage reconciliation receipt pair is incomplete"
                )
            legacy = _validate_legacy_live_receipts(
                plan=plan,
                apply_receipt=apply_receipt,
                legacy_boot_receipt=legacy_boot_receipt,
                legacy_readiness_receipt=legacy_readiness_receipt,
            )
            terminal_mutation_time = int(legacy["readiness_attested_at_unix"])
            boot_plan_sha256 = LEGACY_BOOT_OPERATION_PLAN_SHA256
            boot_receipt_sha256 = str(legacy["legacy_boot_receipt_sha256"])
            legacy_boot_sha256 = str(legacy["legacy_boot_receipt_sha256"])
            legacy_readiness_sha256 = str(legacy["legacy_readiness_receipt_sha256"])
            boot_evidence_kind = "legacy_live_receipt_reconciliation"
            preboot_boot_id_evidence = False
            preboot_service_state_evidence = False
            follow_on_stopped_runtime_gate_required = True
        elif boot_preflight is None or boot_receipt is None:
            raise RuntimeError("storage readiness requires the exact boot receipt")
        else:
            # Lazy import avoids a module cycle: the boot contract is deliberately
            # an adjunct to the byte-stable, already-approved resize plan.
            from scripts.canary import host_storage_boot

            boot_plan = host_storage_boot.build_plan()
            validated_boot = host_storage_boot.validate_receipt(
                plan=boot_plan,
                preflight=boot_preflight,
                receipt=boot_receipt,
                now_unix=now,
            )
            if validated_boot.get("storage_apply_receipt_sha256") != apply_receipt.get(
                "receipt_sha256"
            ):
                raise RuntimeError("storage boot is not bound to the resize transition")
            terminal_mutation_time = validated_boot["completed_at_unix"]
            boot_plan_sha256 = boot_plan.sha256
            boot_preflight_sha256 = str(boot_preflight["report_sha256"])
            boot_receipt_sha256 = str(validated_boot["receipt_sha256"])
            boot_evidence_kind = "journaled_boot_receipt"
            preboot_boot_id_evidence = True
            preboot_service_state_evidence = True
    elif normal_boot_requested or legacy_boot_requested:
        raise RuntimeError("storage readiness has an unexpected boot receipt")

    _require_fresh_report(postflight, plan=plan, now_unix=now)
    collected_at = postflight.get("collected_at_unix")
    if (
        postflight.get("ok") is not True
        or postflight.get("state") != "target_ready"
        or postflight.get("satisfied_steps") != [RESIZE_STEP]
        or postflight.get("ready_for_packaging") is not True
        or type(collected_at) is not int
        or collected_at < terminal_mutation_time
        or postflight.get("disk_size_gb") != plan.spec.target_size_gb
        or postflight.get("root_source") != plan.spec.root_source
        or postflight.get("root_filesystem") != plan.spec.root_filesystem
        or postflight.get("root_mountpoint") != plan.spec.root_mountpoint
    ):
        raise RuntimeError("storage postflight is not target-ready exact")
    if _successful_check_names(postflight) != {
        "host.lifecycle_state_exact",
        "instance.identity_exact",
        "disk.identity_exact",
        "disk.size_transition_exact",
        "filesystem.root_identity_exact",
        "filesystem.capacity_matches_disk_state",
        "filesystem.packaging_headroom_if_target",
    }:
        raise RuntimeError("storage postflight check set is not exact")
    root_size = postflight.get("root_size_bytes")
    root_available = postflight.get("root_available_bytes")
    if (
        type(root_size) is not int
        or not plan.spec.minimum_target_root_filesystem_bytes
        <= root_size
        <= MAXIMUM_TARGET_ROOT_FILESYSTEM_BYTES
        or type(root_available) is not int
        or not plan.spec.minimum_packaging_free_bytes <= root_available <= root_size
    ):
        raise RuntimeError("storage postflight capacity is not ready")

    unsigned = {
        "schema": STORAGE_READINESS_RECEIPT_SCHEMA,
        "ok": True,
        "plan_sha256": plan.sha256,
        "apply_receipt_sha256": apply_receipt["receipt_sha256"],
        "postflight_report_sha256": postflight["report_sha256"],
        "attested_at_unix": now,
        "vm_instance_id": plan.spec.vm_instance_id,
        "disk_id": plan.spec.disk_id,
        "disk_size_gb": plan.spec.target_size_gb,
        "root_source": plan.spec.root_source,
        "root_filesystem": plan.spec.root_filesystem,
        "root_mountpoint": plan.spec.root_mountpoint,
        "root_size_bytes": root_size,
        "root_available_bytes": root_available,
        "boot_expansion_required": boot_expansion_required,
        "boot_plan_sha256": boot_plan_sha256,
        "boot_preflight_report_sha256": boot_preflight_sha256,
        "boot_receipt_sha256": boot_receipt_sha256,
        "boot_evidence_kind": boot_evidence_kind,
        "legacy_boot_receipt_sha256": legacy_boot_sha256,
        "legacy_readiness_receipt_sha256": legacy_readiness_sha256,
        "preboot_boot_id_evidence": preboot_boot_id_evidence,
        "preboot_service_state_evidence": preboot_service_state_evidence,
        "follow_on_stopped_runtime_gate_required": (
            follow_on_stopped_runtime_gate_required
        ),
        "filesystem_ready": True,
        "ready_for_packaging": True,
        "opens_runtime_gate": False,
    }
    return _seal(unsigned, digest_key="receipt_sha256")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("plan", "apply", "attest"), nargs="?", default="plan"
    )
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--apply-receipt", type=Path)
    parser.add_argument("--postflight", type=Path)
    parser.add_argument("--boot-preflight", type=Path)
    parser.add_argument("--boot-receipt", type=Path)
    parser.add_argument("--legacy-boot-receipt", type=Path)
    parser.add_argument("--legacy-readiness-receipt", type=Path)
    return parser


def _load(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise SystemExit("receipt input must be a JSON object")
    return value


def _load_exact_legacy_owner_receipt(
    path: Path,
    *,
    expected_path: Path,
) -> Mapping[str, object]:
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(expected_path)
        or path != expected_path
    ):
        raise SystemExit("legacy receipt path is not the fixed owner archive")
    parent = os.lstat(expected_path.parent)
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid != _OWNER_UID
        or parent.st_gid != _OWNER_GID
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise SystemExit("legacy receipt archive is not owner-only exact")
    item = os.lstat(path)
    if (
        not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != _OWNER_UID
        or item.st_gid != _OWNER_GID
        or stat.S_IMODE(item.st_mode) != 0o600
        or item.st_nlink != 1
        or not 0 < item.st_size <= _LEGACY_RECEIPT_MAX_BYTES
    ):
        raise SystemExit("legacy receipt file identity is not owner-only exact")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        raw = os.read(descriptor, _LEGACY_RECEIPT_MAX_BYTES + 1)
        if os.read(descriptor, 1) or len(raw) != opened.st_size:
            raise SystemExit("legacy receipt framing is invalid")
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_uid,
            opened.st_gid,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_uid,
            item.st_gid,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        ) or (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_uid,
            opened.st_gid,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ):
            raise SystemExit("legacy receipt changed during read")
    finally:
        os.close(descriptor)

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate key")
            result[name] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        raise SystemExit("legacy receipt is invalid JSON") from None
    if not isinstance(value, Mapping):
        raise SystemExit("legacy receipt must be a JSON object")
    return value


def main() -> int:
    args = _parser().parse_args()
    plan = build_plan()
    if args.command == "plan":
        print(json.dumps(plan.report(), indent=2, sort_keys=True))
        return 0
    if args.command == "apply":
        if args.preflight is None or not args.approved_plan_sha256:
            raise SystemExit("apply requires --preflight and --approved-plan-sha256")
        result = execute_plan(
            plan,
            approved_plan_sha256=args.approved_plan_sha256,
            preflight=_load(args.preflight),
        )
    else:
        if args.apply_receipt is None or args.postflight is None:
            raise SystemExit("attest requires --apply-receipt and --postflight")
        normal_requested = (
            args.boot_preflight is not None or args.boot_receipt is not None
        )
        legacy_requested = (
            args.legacy_boot_receipt is not None
            or args.legacy_readiness_receipt is not None
        )
        if normal_requested and legacy_requested:
            raise SystemExit("boot receipt modes are mutually exclusive")
        if normal_requested and (
            args.boot_preflight is None or args.boot_receipt is None
        ):
            raise SystemExit("journaled boot evidence requires both receipt inputs")
        if legacy_requested and (
            args.legacy_boot_receipt is None or args.legacy_readiness_receipt is None
        ):
            raise SystemExit("legacy reconciliation requires both receipt inputs")
        apply_receipt = (
            _load_exact_legacy_owner_receipt(
                args.apply_receipt,
                expected_path=LEGACY_APPLY_RECEIPT_PATH,
            )
            if legacy_requested
            else _load(args.apply_receipt)
        )
        result = build_readiness_receipt(
            plan,
            apply_receipt=apply_receipt,
            postflight=_load(args.postflight),
            boot_preflight=(
                _load(args.boot_preflight) if args.boot_preflight is not None else None
            ),
            boot_receipt=(
                _load(args.boot_receipt) if args.boot_receipt is not None else None
            ),
            legacy_boot_receipt=(
                _load_exact_legacy_owner_receipt(
                    args.legacy_boot_receipt,
                    expected_path=LEGACY_BOOT_RECEIPT_PATH,
                )
                if args.legacy_boot_receipt is not None
                else None
            ),
            legacy_readiness_receipt=(
                _load_exact_legacy_owner_receipt(
                    args.legacy_readiness_receipt,
                    expected_path=LEGACY_READINESS_RECEIPT_PATH,
                )
                if args.legacy_readiness_receipt is not None
                else None
            ),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DISK_ID",
    "DISK_NAME",
    "BOOT_DEVICE_NAME",
    "HostStoragePlan",
    "HostStorageSpec",
    "LEGACY_APPLY_RECEIPT_PATH",
    "LEGACY_APPLY_RECEIPT_SHA256",
    "LEGACY_BOOT_OPERATION_PLAN_SHA256",
    "LEGACY_BOOT_RECEIPT_PATH",
    "LEGACY_BOOT_RECEIPT_SCHEMA",
    "LEGACY_BOOT_RECEIPT_SHA256",
    "LEGACY_READINESS_RECEIPT_PATH",
    "LEGACY_READINESS_RECEIPT_SCHEMA",
    "LEGACY_READINESS_RECEIPT_SHA256",
    "MINIMUM_PACKAGING_FREE_BYTES",
    "MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES",
    "RESIZE_STEP",
    "SOURCE_BOOT_DISK_SIZE_GB",
    "STORAGE_APPLY_RECEIPT_SCHEMA",
    "STORAGE_PLAN_SCHEMA",
    "STORAGE_PREFLIGHT_SCHEMA",
    "STORAGE_READINESS_RECEIPT_SCHEMA",
    "TARGET_BOOT_DISK_SIZE_GB",
    "VM_INSTANCE_ID",
    "build_plan",
    "build_readiness_receipt",
    "execute_plan",
]
