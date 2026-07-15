#!/usr/bin/env python3
"""Read-only disk and guest-filesystem evidence for canary storage expansion."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

from scripts.canary.foundation import PREFLIGHT_MAX_AGE_SECONDS, PROJECT, ZONE
from scripts.canary.full_canary_owner_launcher import (
    GcloudOwnerAccessToken,
    IapStoppedReleaseTransport,
    PinnedGcloudConfiguration,
    PinnedGoogleComputeKnownHosts,
    TrustedGcloudExecutable,
)
from scripts.canary.host import HostSpec, VM_NAME
from scripts.canary.host import build_plan as build_host_plan
from scripts.canary.host_preflight import _vm_exact_for_boot_disk_size
from scripts.canary.host_preflight import collect as collect_host
from scripts.canary.host_preflight import evaluate as evaluate_host
from scripts.canary.host_storage import (
    BOOT_DEVICE_NAME,
    DISK_ID,
    DISK_NAME,
    MAXIMUM_TARGET_ROOT_FILESYSTEM_BYTES,
    MINIMUM_PACKAGING_FREE_BYTES,
    MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES,
    OWNER_ACCOUNT,
    RESIZE_STEP,
    ROOT_FILESYSTEM,
    ROOT_MOUNTPOINT,
    ROOT_SOURCE,
    SOURCE_BOOT_DISK_SIZE_GB,
    STORAGE_PREFLIGHT_SCHEMA,
    TARGET_BOOT_DISK_SIZE_GB,
    TRUSTED_GCLOUD,
    TRUSTED_PYTHON,
    VM_INSTANCE_ID,
    _seal,
    _sha256_json,
    build_plan,
)


SOURCE_ROOT_MINIMUM_BYTES = 18_000_000_000
SOURCE_ROOT_MAXIMUM_BYTES = SOURCE_BOOT_DISK_SIZE_GB * 1024**3
_GUEST_MAXIMUM_OUTPUT_BYTES = 64 * 1024
_HOST_CHECK_NAMES = {
    "network.complete_exact",
    "network.preflight_fresh",
    "image.exact_ready",
    "resource.vm_absent_or_exact_running",
}
_STORAGE_CHECK_NAMES = {
    "host.lifecycle_state_exact",
    "instance.identity_exact",
    "disk.identity_exact",
    "disk.size_transition_exact",
    "filesystem.root_identity_exact",
    "filesystem.capacity_matches_disk_state",
    "filesystem.packaging_headroom_if_target",
}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _ends_with(value: object, suffix: str) -> bool:
    return isinstance(value, str) and value.endswith(suffix)


def _instance_exact(raw: object) -> bool:
    if not isinstance(raw, Mapping):
        return False
    disks = raw.get("disks")
    if not isinstance(disks, list) or len(disks) != 1:
        return False
    attached = disks[0]
    return bool(
        isinstance(attached, Mapping)
        and raw.get("id") == VM_INSTANCE_ID
        and raw.get("name") == VM_NAME
        and raw.get("status") == "RUNNING"
        and _ends_with(raw.get("zone"), f"/zones/{ZONE}")
        and attached.get("boot") is True
        and attached.get("autoDelete") is True
        and attached.get("deviceName") == BOOT_DEVICE_NAME
        and attached.get("mode") == "READ_WRITE"
        and attached.get("type") == "PERSISTENT"
        and _ends_with(attached.get("source"), f"/disks/{DISK_NAME}")
    )


def _disk_exact(raw: object) -> bool:
    if not isinstance(raw, Mapping):
        return False
    return bool(
        raw.get("id") == DISK_ID
        and raw.get("name") == DISK_NAME
        and raw.get("status") == "READY"
        and _ends_with(raw.get("zone"), f"/zones/{ZONE}")
        and _ends_with(raw.get("type"), "/diskTypes/pd-balanced")
        and _ends_with(
            raw.get("sourceImage"),
            "/projects/debian-cloud/global/images/debian-12-bookworm-v20260609",
        )
        and raw.get("architecture") == "X86_64"
        and str(raw.get("physicalBlockSizeBytes")) == "4096"
        and raw.get("users")
        == [
            "https://www.googleapis.com/compute/v1/projects/"
            f"{PROJECT}/zones/{ZONE}/instances/{VM_NAME}"
        ]
    )


def _host_check_state(report: object) -> tuple[bool, set[str], set[str]]:
    if not isinstance(report, Mapping):
        return False, set(), set()
    raw_checks = report.get("checks")
    if not isinstance(raw_checks, list) or len(raw_checks) != len(_HOST_CHECK_NAMES):
        return False, set(), set()
    passed: set[str] = set()
    failed: set[str] = set()
    for item in raw_checks:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"name", "passed", "detail"}
            or item.get("name") not in _HOST_CHECK_NAMES
            or type(item.get("passed")) is not bool
            or not isinstance(item.get("detail"), str)
            or not item["detail"]
            or item["name"] in passed | failed
        ):
            return False, set(), set()
        (passed if item["passed"] else failed).add(item["name"])
    exact_inventory = passed | failed == _HOST_CHECK_NAMES
    return exact_inventory, passed, failed


def _host_lifecycle_exact(
    report: object,
    *,
    instance: object,
    disk: object,
    disk_size_gb: int | None,
    collected_at_unix: object,
) -> tuple[bool, bool]:
    inventory_exact, passed, failed = _host_check_state(report)
    if not isinstance(report, Mapping):
        return False, False
    nested_time = report.get("collected_at_unix")
    fresh = bool(
        type(collected_at_unix) is int
        and type(nested_time) is int
        and 0 <= collected_at_unix - nested_time <= PREFLIGHT_MAX_AGE_SECONDS
    )
    try:
        spec = HostSpec(
            sql_private_ip=str(report.get("sql_private_ip") or ""),
            network_plan_sha256=str(report.get("network_plan_sha256") or ""),
        )
        expected_host_plan_sha256 = build_host_plan(spec).sha256
    except ValueError:
        return False, False
    common = bool(
        inventory_exact
        and fresh
        and report.get("schema") == "muncho-isolated-canary-host-preflight.v1"
        and report.get("plan_sha256") == expected_host_plan_sha256
    )
    target_shape_exact = _vm_exact_for_boot_disk_size(
        instance,
        disk,
        spec,
        boot_disk_size_gb=TARGET_BOOT_DISK_SIZE_GB,
    )
    transitional_source_shape_exact = _vm_exact_for_boot_disk_size(
        instance,
        disk,
        spec,
        boot_disk_size_gb=SOURCE_BOOT_DISK_SIZE_GB,
    )
    target_exact = bool(
        common
        and target_shape_exact
        and disk_size_gb == TARGET_BOOT_DISK_SIZE_GB
        and report.get("ok") is True
        and report.get("satisfied_steps") == ["create_isolated_canary_vm"]
        and passed == _HOST_CHECK_NAMES
        and not failed
    )
    transitional_source_exact = bool(
        common
        and transitional_source_shape_exact
        and disk_size_gb == SOURCE_BOOT_DISK_SIZE_GB
        and report.get("ok") is False
        and report.get("satisfied_steps") == []
        and failed == {"resource.vm_absent_or_exact_running"}
        and passed == _HOST_CHECK_NAMES - {"resource.vm_absent_or_exact_running"}
    )
    return target_exact or transitional_source_exact, target_exact


def _root_record(raw: object) -> Mapping[str, object] | None:
    if not isinstance(raw, Mapping) or set(raw) != {"filesystems"}:
        return None
    filesystems = raw.get("filesystems")
    if not isinstance(filesystems, list) or len(filesystems) != 1:
        return None
    item = filesystems[0]
    if not isinstance(item, Mapping):
        return None
    return item


def _nonnegative_json_integer(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    return None


def evaluate(evidence: Mapping[str, object]) -> dict[str, object]:
    plan = build_plan()
    instance = evidence.get("instance")
    disk = evidence.get("disk")
    host_report = evidence.get("host_report")
    root = _root_record(evidence.get("guest_root"))
    raw_size = disk.get("sizeGb") if isinstance(disk, Mapping) else None
    try:
        disk_size = int(raw_size) if isinstance(raw_size, str) else raw_size
    except ValueError:
        disk_size = None
    if type(disk_size) is not int:
        disk_size = None
    instance_exact = _instance_exact(instance)
    disk_exact = _disk_exact(disk)
    size_exact = disk_size in {
        SOURCE_BOOT_DISK_SIZE_GB,
        TARGET_BOOT_DISK_SIZE_GB,
    }
    root_source = root.get("source") if root else None
    root_filesystem = root.get("fstype") if root else None
    root_mountpoint = root.get("target") if root else None
    root_size = _nonnegative_json_integer(root.get("size")) if root else None
    root_available = _nonnegative_json_integer(root.get("avail")) if root else None
    root_identity = bool(
        root_source == ROOT_SOURCE
        and root_filesystem == ROOT_FILESYSTEM
        and root_mountpoint == ROOT_MOUNTPOINT
        and type(root_size) is int
        and type(root_available) is int
        and 0 <= root_available <= root_size
    )
    source_capacity = bool(
        root_identity
        and SOURCE_ROOT_MINIMUM_BYTES <= root_size <= SOURCE_ROOT_MAXIMUM_BYTES
    )
    target_capacity = bool(
        root_identity
        and MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES
        <= root_size
        <= MAXIMUM_TARGET_ROOT_FILESYSTEM_BYTES
    )
    capacity_matches = bool(
        (disk_size == SOURCE_BOOT_DISK_SIZE_GB and source_capacity)
        or (disk_size == TARGET_BOOT_DISK_SIZE_GB and target_capacity)
    )
    packaging_headroom = bool(
        disk_size != TARGET_BOOT_DISK_SIZE_GB
        or (root_identity and root_available >= MINIMUM_PACKAGING_FREE_BYTES)
    )
    host_lifecycle, normal_host_exact = _host_lifecycle_exact(
        host_report,
        instance=instance,
        disk=disk,
        disk_size_gb=disk_size,
        collected_at_unix=evidence.get("collected_at_unix"),
    )
    if (
        instance_exact
        and disk_exact
        and disk_size == SOURCE_BOOT_DISK_SIZE_GB
        and root_identity
        and source_capacity
        and host_lifecycle
        and not normal_host_exact
    ):
        state = "source_ready"
    elif (
        instance_exact
        and disk_exact
        and disk_size == TARGET_BOOT_DISK_SIZE_GB
        and root_identity
        and host_lifecycle
        and normal_host_exact
        and target_capacity
        and packaging_headroom
    ):
        state = "target_ready"
    elif (
        instance_exact
        and disk_exact
        and disk_size == TARGET_BOOT_DISK_SIZE_GB
        and root_identity
        and host_lifecycle
        and normal_host_exact
    ):
        state = "transition_pending"
    else:
        state = "invalid"

    checks = [
        Check(
            "host.lifecycle_state_exact",
            host_lifecycle,
            "normal host preflight must differ only by the reviewed 20 GB source or pass exact 40 GB",
        ),
        Check(
            "instance.identity_exact",
            instance_exact,
            "the running VM ID, name, zone, and sole boot attachment must be exact",
        ),
        Check(
            "disk.identity_exact",
            disk_exact,
            "the persistent disk ID, image, type, user, and block shape must be exact",
        ),
        Check(
            "disk.size_transition_exact",
            size_exact,
            "the disk must be the one reviewed 20 GB source or exact 40 GB target",
        ),
        Check(
            "filesystem.root_identity_exact",
            root_identity,
            "the read-only guest fact must describe ext4 /dev/sda1 mounted at root",
        ),
        Check(
            "filesystem.capacity_matches_disk_state",
            capacity_matches,
            "root capacity must match the source or fully expanded target state",
        ),
        Check(
            "filesystem.packaging_headroom_if_target",
            packaging_headroom,
            "the 40 GB target must expose at least 8 GiB of available root space",
        ),
    ]
    ok = state in {"source_ready", "target_ready"} and all(
        check.passed for check in checks
    )
    host_report_sha256 = None
    if isinstance(host_report, Mapping):
        host_report_sha256 = _sha256_json(dict(host_report))
    unsigned = {
        "schema": STORAGE_PREFLIGHT_SCHEMA,
        "ok": ok,
        "state": state,
        "collected_at_unix": evidence.get("collected_at_unix"),
        "plan_sha256": plan.sha256,
        "vm_instance_id": instance.get("id") if isinstance(instance, Mapping) else None,
        "disk_id": disk.get("id") if isinstance(disk, Mapping) else None,
        "disk_size_gb": disk_size,
        "root_source": root_source,
        "root_filesystem": root_filesystem,
        "root_mountpoint": root_mountpoint,
        "root_size_bytes": root_size,
        "root_available_bytes": root_available,
        "normal_host_preflight_ok": normal_host_exact,
        "host_report_sha256": host_report_sha256,
        "satisfied_steps": [RESIZE_STEP] if state == "target_ready" else [],
        "ready_for_packaging": state == "target_ready",
        "checks": [asdict(check) for check in checks],
    }
    return _seal(unsigned, digest_key="report_sha256")


CommandRunner = Callable[[Sequence[str]], object]
GuestRunner = Callable[[Sequence[str]], object]


def _guest_command() -> tuple[str, ...]:
    return (
        "/usr/bin/findmnt",
        "--json",
        "--bytes",
        "--output=SOURCE,FSTYPE,SIZE,AVAIL,TARGET",
        "/",
    )


def _sealed_owner_transport() -> tuple[
    GcloudOwnerAccessToken,
    IapStoppedReleaseTransport,
]:
    executable = TrustedGcloudExecutable(
        candidates=(TRUSTED_GCLOUD,),
        python_candidates=(TRUSTED_PYTHON,),
    )
    configuration = PinnedGcloudConfiguration()
    owner_identity = GcloudOwnerAccessToken(
        gcloud_executable=executable,
        gcloud_configuration=configuration,
    )
    transport = IapStoppedReleaseTransport(
        owner_identity,
        gcloud_executable=executable,
        gcloud_configuration=configuration,
        known_hosts=PinnedGoogleComputeKnownHosts(),
    )
    return owner_identity, transport


def _run_guest_json(
    argv: Sequence[str],
    *,
    owner_identity: GcloudOwnerAccessToken | None = None,
    transport: IapStoppedReleaseTransport | None = None,
) -> object:
    if tuple(argv) != _guest_command():
        raise RuntimeError("guest storage command is not exact")
    if (owner_identity is None) != (transport is None):
        raise RuntimeError("guest storage transport boundary is incomplete")
    if owner_identity is None or transport is None:
        owner_identity, transport = _sealed_owner_transport()
    account = owner_identity.account_for_read_only_preflight()
    if account != OWNER_ACCOUNT:
        raise RuntimeError("guest storage owner identity is not exact")
    completed = transport._run_remote(
        _guest_command(),
        account=account,
        timeout_seconds=90.0,
        maximum_output_bytes=_GUEST_MAXIMUM_OUTPUT_BYTES,
    )
    if (
        completed.returncode != 0
        or not completed.stdout
        or not isinstance(completed.stdout, bytes)
        or len(completed.stdout) > _GUEST_MAXIMUM_OUTPUT_BYTES
    ):
        raise RuntimeError("read-only guest storage evidence unavailable")
    try:
        value = json.loads(completed.stdout.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError):
        raise RuntimeError("read-only guest storage evidence invalid") from None
    return value


def collect(
    *,
    run_json: CommandRunner | None = None,
    run_guest_json: GuestRunner | None = None,
) -> dict[str, object]:
    if (run_json is None) != (run_guest_json is None):
        raise RuntimeError("storage collector injection boundary is incomplete")
    if run_json is None or run_guest_json is None:
        owner_identity, transport = _sealed_owner_transport()
        run_json = owner_identity.run_canary_iam_read_only_json

        def run_guest_json(argv: Sequence[str]) -> object:
            return _run_guest_json(
                argv,
                owner_identity=owner_identity,
                transport=transport,
            )

    host_evidence = collect_host(run_json=run_json)
    return {
        "collected_at_unix": int(time.time()),
        "host_report": evaluate_host(host_evidence),
        "instance": host_evidence.get("planned_vm"),
        "disk": host_evidence.get("planned_vm_disk"),
        "guest_root": run_guest_json(_guest_command()),
    }


def main() -> int:
    print(json.dumps(evaluate(collect()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SOURCE_ROOT_MAXIMUM_BYTES",
    "SOURCE_ROOT_MINIMUM_BYTES",
    "collect",
    "evaluate",
]
