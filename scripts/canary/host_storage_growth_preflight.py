#!/usr/bin/env python3
"""Read-only exact evidence for the isolated canary 40 -> 80 GB gate."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence, cast

from gateway.canonical_writer_host_authority import ExternalIAMReceipt
from scripts.canary import storage_growth_contract as contract
from scripts.canary.host_storage import _seal, _sha256_json
from scripts.canary.host_storage_boot import BOOT_ID_COMMAND
from scripts.canary.host_storage_growth import (
    _service_states_exact,
    build_plan,
)
from scripts.canary.host_storage_preflight import (
    _disk_exact,
    _instance_exact,
)
from scripts.canary.writer_release import (
    DEFAULT_EVIDENCE_BASE,
    DEFAULT_HOST_RECEIPT_PATH,
    DEFAULT_SYSTEMCTL_EXECUTABLE,
    _SERVICE_PROPERTIES,
    _parse_service_observation,
)
from scripts.canary.runtime_units import CANARY_RUNTIME_UNITS


_BOOT_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_MAX_GUEST_OUTPUT_BYTES = 256 * 1024
HOST_RECEIPT_COMMAND = ("/usr/bin/cat", str(DEFAULT_HOST_RECEIPT_PATH))
STOPPED_RELEASE_RECEIPT_PATH = (
    DEFAULT_EVIDENCE_BASE
    / contract.CURRENT_STOPPED_RELEASE_SHA
    / "stopped-release-publication.json"
)
STOPPED_RELEASE_RECEIPT_COMMAND = (
    "/usr/bin/cat",
    str(STOPPED_RELEASE_RECEIPT_PATH),
)
_RECEIPT_BINDING_FIELDS = {
    "source",
    "current_stopped_release_sha",
    "current_host_receipt_file_sha256",
    "current_host_receipt_sha256",
    "current_stopped_release_receipt_file_sha256",
    "current_stopped_release_receipt_sha256",
}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _string_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping) or any(
        not isinstance(name, str) for name in value
    ):
        return None
    return cast(Mapping[str, object], value)


def _integer(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    return None


def _root_record(raw: object) -> Mapping[str, object] | None:
    value = _string_mapping(raw)
    if value is None or set(value) != {"filesystems"}:
        return None
    filesystems = value.get("filesystems")
    if not isinstance(filesystems, list) or len(filesystems) != 1:
        return None
    return _string_mapping(filesystems[0])


def _instance_for_status(raw: object, *, status: str) -> bool:
    value = _string_mapping(raw)
    if value is None or value.get("status") != status:
        return False
    service_accounts = value.get("serviceAccounts")
    account = (
        _string_mapping(service_accounts[0])
        if isinstance(service_accounts, list) and len(service_accounts) == 1
        else None
    )
    if (
        account is None
        or account.get("email") != contract.RUNTIME_SERVICE_ACCOUNT
        or account.get("scopes") != list(contract.RUNTIME_SCOPES)
    ):
        return False
    normalized = dict(value)
    normalized["status"] = "RUNNING"
    return _instance_exact(normalized)


def _attachment_exact(raw: object) -> bool:
    value = _string_mapping(raw)
    if value is None:
        return False
    disks = value.get("disks")
    if not isinstance(disks, list) or len(disks) != 1:
        return False
    attached = _string_mapping(disks[0])
    plan = build_plan()
    source = attached.get("source") if attached is not None else None
    return bool(
        attached is not None
        and attached.get("boot") is True
        and attached.get("autoDelete") is True
        and attached.get("deviceName") == plan.spec.boot_device_name
        and attached.get("mode") == "READ_WRITE"
        and attached.get("type") == "PERSISTENT"
        and isinstance(source, str)
        and source.endswith(f"/disks/{plan.spec.disk_name}")
    )


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeError("storage growth receipt is not canonical JSON") from None


def _decode_receipt(
    raw: bytes, *, label: str, trailing_newline: bool
) -> dict[str, object]:
    if not raw or len(raw) > _MAX_GUEST_OUTPUT_BYTES:
        raise RuntimeError(f"storage growth {label} output is invalid")
    payload = raw[:-1] if trailing_newline and raw.endswith(b"\n") else raw
    if trailing_newline and not raw.endswith(b"\n"):
        raise RuntimeError(f"storage growth {label} framing is invalid")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for name, item in pairs:
            if name in value:
                raise ValueError("duplicate key")
            value[name] = item
        return value

    try:
        value = json.loads(
            payload.decode("ascii", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _item: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        raise RuntimeError(f"storage growth {label} is invalid JSON") from None
    if not isinstance(value, dict) or _canonical_bytes(value) != payload:
        raise RuntimeError(f"storage growth {label} is not canonical")
    return value


def _validate_live_receipts(
    host_raw: bytes,
    stopped_raw: bytes,
) -> dict[str, object]:
    """Validate exact current receipt bytes read from the running VM."""

    host = _decode_receipt(
        host_raw,
        label="host identity receipt",
        trailing_newline=False,
    )
    stopped = _decode_receipt(
        stopped_raw,
        label="stopped release receipt",
        trailing_newline=True,
    )
    host_unsigned = dict(host)
    host_sha256 = host_unsigned.pop("receipt_sha256", None)
    stopped_unsigned = dict(stopped)
    stopped_sha256 = stopped_unsigned.pop("receipt_sha256", None)
    dedicated = _string_mapping(stopped.get("dedicated_host"))
    if (
        not isinstance(host_sha256, str)
        or _sha256_json(host_unsigned) != host_sha256
        or not isinstance(stopped_sha256, str)
        or _sha256_json(stopped_unsigned) != stopped_sha256
        or stopped.get("release_revision")
        != contract.CURRENT_STOPPED_RELEASE_SHA
        or stopped.get("host_identity_receipt_sha256") != host_sha256
        or dedicated is None
        or dict(dedicated) != {name: host.get(name) for name in dedicated}
    ):
        raise RuntimeError("storage growth live canonical receipts are invalid")
    bindings: dict[str, object] = {
        "source": "fresh_running_vm_receipt_bytes",
        "current_stopped_release_sha": contract.CURRENT_STOPPED_RELEASE_SHA,
        "current_host_receipt_file_sha256": hashlib.sha256(host_raw).hexdigest(),
        "current_host_receipt_sha256": host_sha256,
        "current_stopped_release_receipt_file_sha256": hashlib.sha256(
            stopped_raw
        ).hexdigest(),
        "current_stopped_release_receipt_sha256": stopped_sha256,
    }
    if (
        bindings["current_host_receipt_file_sha256"]
        != contract.CURRENT_HOST_RECEIPT_FILE_SHA256
        or bindings["current_host_receipt_sha256"]
        != contract.CURRENT_HOST_RECEIPT_SHA256
        or bindings["current_stopped_release_receipt_file_sha256"]
        != contract.CURRENT_STOPPED_RELEASE_RECEIPT_FILE_SHA256
        or bindings["current_stopped_release_receipt_sha256"]
        != contract.CURRENT_STOPPED_RELEASE_RECEIPT_SHA256
        or stopped.get("host_identity_receipt_file_sha256")
        != bindings["current_host_receipt_file_sha256"]
        or stopped.get("host_identity_receipt_sha256")
        != bindings["current_host_receipt_sha256"]
    ):
        raise RuntimeError("storage growth live canonical receipts are not exact")
    return bindings


def _validate_receipt_bindings(
    raw: object, *, running: bool
) -> dict[str, object] | None:
    value = _string_mapping(raw)
    if value is None or set(value) != _RECEIPT_BINDING_FIELDS:
        return None
    source = value.get("source")
    if source not in {
        "fresh_running_vm_receipt_bytes",
        "durable_signed_source_snapshot_for_stopped_vm",
    } or (running and source != "fresh_running_vm_receipt_bytes"):
        return None
    expected = {
        "current_stopped_release_sha": contract.CURRENT_STOPPED_RELEASE_SHA,
        "current_host_receipt_file_sha256": (
            contract.CURRENT_HOST_RECEIPT_FILE_SHA256
        ),
        "current_host_receipt_sha256": contract.CURRENT_HOST_RECEIPT_SHA256,
        "current_stopped_release_receipt_file_sha256": (
            contract.CURRENT_STOPPED_RELEASE_RECEIPT_FILE_SHA256
        ),
        "current_stopped_release_receipt_sha256": (
            contract.CURRENT_STOPPED_RELEASE_RECEIPT_SHA256
        ),
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        return None
    return dict(value)


def _validate_external_iam(
    raw: object, *, now_unix: object
) -> dict[str, object] | None:
    value = _string_mapping(raw)
    if value is None or type(now_unix) is not int:
        return None
    try:
        receipt = ExternalIAMReceipt.from_mapping(
            cast(Mapping[str, Any], value)
        )
        receipt.require_fresh(now_unix, minimum_remaining_seconds=720)
    except (TypeError, ValueError):
        return None
    if (
        receipt.value.get("source_approval_sha256") != build_plan().sha256
        or receipt.policy_sha256 != contract.EXTERNAL_IAM_POLICY_SHA256
    ):
        return None
    return {
        "external_iam_receipt": receipt.to_mapping(),
        "external_iam_receipt_sha256": receipt.sha256,
        "external_iam_policy_sha256": receipt.policy_sha256,
        "external_iam_collected_at_unix": receipt.value["collected_at_unix"],
        "external_iam_expires_at_unix": receipt.value["expires_at_unix"],
    }


def evaluate(evidence: Mapping[str, object]) -> dict[str, object]:
    """Evaluate exact cloud, receipt, IAM, root, boot, and systemd facts."""

    plan = build_plan()
    instance = evidence.get("instance")
    disk = evidence.get("disk")
    guest = evidence.get("guest")
    instance_map = _string_mapping(instance)
    disk_map = _string_mapping(disk)
    guest_map = _string_mapping(guest)
    status = instance_map.get("status") if instance_map is not None else None
    collected_at = evidence.get("collected_at_unix")
    receipt_bindings = _validate_receipt_bindings(
        evidence.get("canonical_receipts"),
        running=status == "RUNNING",
    )
    external_iam = _validate_external_iam(
        evidence.get("external_iam_receipt"),
        now_unix=collected_at,
    )
    canonical_receipts_exact = receipt_bindings is not None
    iam_exact = external_iam is not None
    status_exact = status in {"RUNNING", "TERMINATED"}
    instance_exact = bool(
        status_exact
        and _instance_for_status(instance, status=str(status))
        and _attachment_exact(instance)
    )
    disk_identity = _disk_exact(disk)
    disk_size = _integer(disk_map.get("sizeGb")) if disk_map is not None else None
    disk_size_exact = disk_size in {contract.SOURCE_SIZE_GB, contract.TARGET_SIZE_GB}

    boot_id: object = None
    root: Mapping[str, object] | None = None
    services: list[dict[str, object]] | None = None
    if status == "RUNNING" and guest_map is not None:
        boot_id = guest_map.get("boot_id_sha256")
        root = _root_record(guest_map.get("root"))
        services = _service_states_exact(guest_map.get("service_states"))
    root_source = root.get("source") if root else None
    root_filesystem = root.get("fstype") if root else None
    root_mountpoint = root.get("target") if root else None
    root_size = _integer(root.get("size")) if root else None
    root_available = _integer(root.get("avail")) if root else None
    root_identity = bool(
        root_source == plan.spec.root_source
        and root_filesystem == plan.spec.root_filesystem
        and root_mountpoint == plan.spec.root_mountpoint
        and type(root_size) is int
        and type(root_available) is int
        and 0 <= root_available <= root_size
    )
    source_capacity = bool(
        root_identity
        and isinstance(root_size, int)
        and isinstance(root_available, int)
        and contract.MINIMUM_SOURCE_ROOT_FILESYSTEM_BYTES
        <= root_size
        <= contract.MAXIMUM_SOURCE_ROOT_FILESYSTEM_BYTES
        and root_available >= contract.MINIMUM_SOURCE_FREE_BYTES
    )
    target_capacity = bool(
        root_identity
        and isinstance(root_size, int)
        and isinstance(root_available, int)
        and contract.MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES
        <= root_size
        <= contract.MAXIMUM_TARGET_ROOT_FILESYSTEM_BYTES
        and root_available >= contract.MINIMUM_FREE_BYTES
    )
    boot_exact = bool(
        isinstance(boot_id, str) and re.fullmatch(r"[0-9a-f]{64}", boot_id) is not None
    )
    services_exact = services is not None
    running_common = bool(
        status == "RUNNING"
        and instance_exact
        and disk_identity
        and disk_size_exact
        and canonical_receipts_exact
        and iam_exact
        and boot_exact
        and services_exact
        and root_identity
    )
    if running_common and disk_size == contract.SOURCE_SIZE_GB and source_capacity:
        state = "source_ready"
        ok = True
    elif running_common and disk_size == contract.TARGET_SIZE_GB and source_capacity:
        state = "resize_complete_boot_required"
        ok = False
    elif running_common and disk_size == contract.TARGET_SIZE_GB and target_capacity:
        state = "target_ready"
        ok = True
    elif (
        status == "TERMINATED"
        and instance_exact
        and disk_identity
        and disk_size == contract.TARGET_SIZE_GB
        and guest is None
        and canonical_receipts_exact
        and iam_exact
    ):
        state = "terminated_after_growth_intent"
        ok = False
    else:
        state = "invalid"
        ok = False

    resource_exact = bool(instance_exact and disk_identity)
    capacity_exact = bool(
        (disk_size == contract.SOURCE_SIZE_GB and source_capacity)
        or (
            disk_size == contract.TARGET_SIZE_GB
            and (source_capacity or target_capacity)
        )
    )
    checks = [
        Check(
            "resource.project_zone_exact",
            resource_exact,
            "the identity-pinned instance and disk URLs must remain in the exact project and zone",
        ),
        Check(
            "resource.instance_attachment_exact",
            instance_exact and _attachment_exact(instance),
            "the sole persistent-disk-0 attachment must remain boot, auto-delete, and read-write",
        ),
        Check(
            "resource.disk_identity_and_size_exact",
            disk_identity and disk_size_exact,
            "the exact disk ID may be only the historical 40 GB source or approved 80 GB target",
        ),
        Check(
            "runtime.units_inventory_exact",
            services_exact,
            "the complete fixed twelve-unit canary runtime superset must be observed",
        ),
        Check(
            "runtime.units_stopped_exact",
            services_exact,
            "every canary unit must be absent or disabled and inactive",
        ),
        Check(
            "canonical.receipts_exact",
            canonical_receipts_exact,
            "the current host and stopped-release receipts must be freshly validated and exact",
        ),
        Check(
            "iam.policy_live_exact",
            iam_exact,
            "the current external IAM projection must be exact and retain at least twelve minutes",
        ),
        Check(
            "filesystem.root_identity_exact",
            root_identity,
            "the read-only guest fact must describe ext4 /dev/sda1 mounted at root",
        ),
        Check(
            "filesystem.capacity_and_headroom_exact",
            capacity_exact,
            "root must be source-sized, or at least 84 GB with at least 32 GiB free at target",
        ),
    ]
    if state == "terminated_after_growth_intent":
        # No guest command is attempted against a stopped VM.  The failed guest
        # checks make this unusable as an initial preflight while still giving
        # the journaled executor an exact identity-pinned recovery observation.
        services_value: list[dict[str, object]] = []
        service_digest = _sha256_json([])
    else:
        services_value = services or []
        service_digest = _sha256_json(services_value)
    unsigned = {
        "schema": contract.STORAGE_GROWTH_PREFLIGHT_SCHEMA,
        "ok": ok,
        "state": state,
        "collected_at_unix": collected_at,
        "plan_sha256": plan.sha256,
        "project": plan.spec.project,
        "zone": plan.spec.zone,
        "owner_account": plan.spec.owner_account,
        "vm_name": instance_map.get("name") if instance_map is not None else None,
        "vm_instance_id": (
            instance_map.get("id") if instance_map is not None else None
        ),
        "disk_name": disk_map.get("name") if disk_map is not None else None,
        "disk_id": disk_map.get("id") if disk_map is not None else None,
        "boot_device_name": plan.spec.boot_device_name,
        "instance_status": status,
        "disk_size_gb": disk_size,
        "boot_id_sha256": boot_id,
        "root_source": root_source,
        "root_filesystem": root_filesystem,
        "root_mountpoint": root_mountpoint,
        "root_size_bytes": root_size,
        "root_available_bytes": root_available,
        "canonical_receipt_source": (
            receipt_bindings["source"] if receipt_bindings is not None else None
        ),
        "current_stopped_release_sha": (
            receipt_bindings["current_stopped_release_sha"]
            if receipt_bindings is not None
            else None
        ),
        "current_host_receipt_file_sha256": (
            receipt_bindings["current_host_receipt_file_sha256"]
            if receipt_bindings is not None
            else None
        ),
        "current_host_receipt_sha256": (
            receipt_bindings["current_host_receipt_sha256"]
            if receipt_bindings is not None
            else None
        ),
        "current_stopped_release_receipt_file_sha256": (
            receipt_bindings["current_stopped_release_receipt_file_sha256"]
            if receipt_bindings is not None
            else None
        ),
        "current_stopped_release_receipt_sha256": (
            receipt_bindings["current_stopped_release_receipt_sha256"]
            if receipt_bindings is not None
            else None
        ),
        "external_iam_receipt_sha256": (
            external_iam["external_iam_receipt_sha256"]
            if external_iam is not None
            else None
        ),
        "external_iam_receipt": (
            external_iam["external_iam_receipt"] if external_iam is not None else None
        ),
        "external_iam_policy_sha256": (
            external_iam["external_iam_policy_sha256"]
            if external_iam is not None
            else None
        ),
        "external_iam_collected_at_unix": (
            external_iam["external_iam_collected_at_unix"]
            if external_iam is not None
            else None
        ),
        "external_iam_expires_at_unix": (
            external_iam["external_iam_expires_at_unix"]
            if external_iam is not None
            else None
        ),
        "runtime_units": list(CANARY_RUNTIME_UNITS),
        "service_states": services_value,
        "service_states_sha256": service_digest,
        "checks": [asdict(check) for check in checks],
    }
    return _seal(unsigned, digest_key="report_sha256")


def _boot_id_sha256(raw: bytes) -> str:
    try:
        value = raw.decode("ascii", errors="strict").strip()
    except UnicodeError:
        raise RuntimeError("storage growth boot identity output is invalid") from None
    if _BOOT_ID.fullmatch(value) is None:
        raise RuntimeError("storage growth boot identity output is invalid")
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _root_command() -> tuple[str, ...]:
    return (
        "/usr/bin/findmnt",
        "--json",
        "--bytes",
        "--output=SOURCE,FSTYPE,SIZE,AVAIL,TARGET",
        "/",
    )


def _service_command(unit: str) -> tuple[str, ...]:
    if unit not in CANARY_RUNTIME_UNITS:
        raise RuntimeError("storage growth service unit is not exact")
    return (
        str(DEFAULT_SYSTEMCTL_EXECUTABLE),
        "show",
        "--no-pager",
        *(f"--property={name}" for name in _SERVICE_PROPERTIES),
        unit,
    )


def _decode_json(raw: bytes, label: str) -> object:
    if not raw or len(raw) > _MAX_GUEST_OUTPUT_BYTES:
        raise RuntimeError(f"storage growth {label} output is invalid")
    try:
        return json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError):
        raise RuntimeError(f"storage growth {label} output is invalid") from None


GuestRunner = Callable[[Sequence[str]], bytes]
JsonRunner = Callable[[Sequence[str]], object]


def _collect_guest(run_guest: GuestRunner) -> dict[str, object]:
    boot = run_guest(BOOT_ID_COMMAND)
    root = _decode_json(run_guest(_root_command()), "root")
    canonical_receipts = _validate_live_receipts(
        run_guest(HOST_RECEIPT_COMMAND),
        run_guest(STOPPED_RELEASE_RECEIPT_COMMAND),
    )
    states: list[dict[str, object]] = []
    for unit in CANARY_RUNTIME_UNITS:
        raw = run_guest(_service_command(unit))
        if not raw or len(raw) > _MAX_GUEST_OUTPUT_BYTES:
            raise RuntimeError("storage growth service output is invalid")
        try:
            rendered = raw.decode("utf-8", errors="strict")
        except UnicodeError:
            raise RuntimeError("storage growth service output is invalid") from None
        states.append(_parse_service_observation(unit, rendered))
    return {
        "boot_id_sha256": _boot_id_sha256(boot),
        "root": root,
        "service_states": states,
        "canonical_receipts": canonical_receipts,
    }


def collect(
    *,
    run_json: JsonRunner,
    run_guest: GuestRunner,
    external_iam_receipt: Mapping[str, object],
    stopped_receipt_snapshot: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Collect through the release-bound launcher's injected trusted readers."""

    if not callable(run_json) or not callable(run_guest):
        raise RuntimeError("storage growth trusted collector boundary is incomplete")
    if not isinstance(external_iam_receipt, Mapping):
        raise RuntimeError("storage growth external IAM receipt is unavailable")
    project = f"--project={build_plan().spec.project}"
    zone = f"--zone={build_plan().spec.zone}"
    instance = run_json((
        "gcloud",
        "compute",
        "instances",
        "describe",
        contract.VM_NAME,
        zone,
        project,
        "--format=json",
    ))
    disk = run_json((
        "gcloud",
        "compute",
        "disks",
        "describe",
        contract.VM_NAME,
        zone,
        project,
        "--format=json",
    ))
    instance_map = _string_mapping(instance)
    status = instance_map.get("status") if instance_map is not None else None
    guest = _collect_guest(run_guest) if status == "RUNNING" else None
    if guest is not None:
        canonical_receipts = guest.pop("canonical_receipts")
    else:
        canonical_receipts = _validate_receipt_bindings(
            stopped_receipt_snapshot,
            running=False,
        )
        if canonical_receipts is not None:
            canonical_receipts["source"] = (
                "durable_signed_source_snapshot_for_stopped_vm"
            )
    return evaluate({
        "collected_at_unix": int(time.time()),
        "instance": instance,
        "disk": disk,
        "guest": guest,
        "canonical_receipts": canonical_receipts,
        "external_iam_receipt": dict(external_iam_receipt),
    })


def main() -> int:
    raise SystemExit(
        "storage growth preflight is available only through the release-bound "
        "full_canary_owner_launcher.py"
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "collect",
    "evaluate",
    "HOST_RECEIPT_COMMAND",
    "STOPPED_RELEASE_RECEIPT_COMMAND",
]
