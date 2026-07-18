from __future__ import annotations

import copy
import json
from typing import Mapping, Sequence, cast

import pytest

from gateway import canonical_writer_host_authority as host_authority
from scripts.canary import host_storage, host_storage_boot
from scripts.canary import host_storage_growth as growth
from scripts.canary import host_storage_growth_preflight as preflight
from scripts.canary import storage_growth_contract as contract


NOW = 1_800_000_000
BOOT_1 = "1" * 64


def _instance(status: str = "RUNNING") -> dict[str, object]:
    return {
        "id": contract.VM_INSTANCE_ID,
        "name": contract.VM_NAME,
        "status": status,
        "zone": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{contract.PROJECT}/zones/{contract.ZONE}"
        ),
        "serviceAccounts": [
            {
                "email": contract.RUNTIME_SERVICE_ACCOUNT,
                "scopes": list(contract.RUNTIME_SCOPES),
            }
        ],
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "deviceName": contract.BOOT_DEVICE_NAME,
                "mode": "READ_WRITE",
                "type": "PERSISTENT",
                "source": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{contract.PROJECT}/zones/{contract.ZONE}/disks/"
                    f"{contract.DISK_NAME}"
                ),
            }
        ],
    }


def _disk(size_gb: int) -> dict[str, object]:
    return {
        "id": contract.DISK_ID,
        "name": contract.DISK_NAME,
        "sizeGb": str(size_gb),
        "status": "READY",
        "zone": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{contract.PROJECT}/zones/{contract.ZONE}"
        ),
        "type": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{contract.PROJECT}/zones/{contract.ZONE}/diskTypes/"
            f"{contract.DISK_TYPE}"
        ),
        "sourceImage": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{contract.SOURCE_IMAGE_PROJECT}/global/images/"
            f"{contract.SOURCE_IMAGE}"
        ),
        "architecture": "X86_64",
        "physicalBlockSizeBytes": "4096",
        "users": [
            "https://www.googleapis.com/compute/v1/projects/"
            f"{contract.PROJECT}/zones/{contract.ZONE}/instances/"
            f"{contract.VM_NAME}"
        ],
    }


def _service_states() -> list[dict[str, object]]:
    return [
        {
            "unit": unit,
            "state": "absent",
            "properties": {
                "LoadState": "not-found",
                "ActiveState": "inactive",
                "SubState": "dead",
                "UnitFileState": "",
                "MainPID": "0",
                "FragmentPath": "",
                "DropInPaths": "",
            },
        }
        for unit in growth.CANARY_RUNTIME_UNITS
    ]


def _canonical_receipt_bindings(*, stopped: bool = False) -> dict[str, object]:
    return {
        "source": (
            "durable_signed_source_snapshot_for_stopped_vm"
            if stopped
            else "fresh_running_vm_receipt_bytes"
        ),
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


def _external_iam_receipt(collected_at: int) -> dict[str, object]:
    return {
        "schema": host_authority.EXTERNAL_IAM_RECEIPT_SCHEMA,
        "project": host_authority._CANARY_PROJECT,
        "zone": host_authority._CANARY_ZONE,
        "instance": host_authority._CANARY_INSTANCE,
        "service_account": host_authority._CANARY_SERVICE_ACCOUNT,
        "scopes": list(host_authority._CANARY_SCOPES),
        "roles": list(host_authority._CANARY_ROLES),
        "permissions": list(host_authority._CANARY_PERMISSIONS),
        "foundation_plan_sha256": (
            "105c496970fa3c8f5a2fc2bbcc9a6dc2a5cbf8c712d494017ad383bf6702bb16"
        ),
        "host_plan_sha256": (
            "3601d637bfcef6176efff7c8b35f34d8499c70c866fb7bf681349c0f6b03afef"
        ),
        "foundation_report_sha256": "1" * 64,
        "host_report_sha256": "2" * 64,
        "source_approval_sha256": contract.canonical_plan_sha256(),
        "collected_at_unix": collected_at,
        "expires_at_unix": collected_at + host_authority.EXTERNAL_IAM_TTL_SECONDS,
    }


def _evidence(
    *,
    disk_size: int,
    root_size: int,
    root_available: int,
    boot: str = BOOT_1,
    status: str = "RUNNING",
    collected_at: int = NOW,
) -> dict[str, object]:
    guest = None
    if status == "RUNNING":
        guest = {
            "boot_id_sha256": boot,
            "root": {
                "filesystems": [
                    {
                        "source": contract.ROOT_SOURCE,
                        "fstype": contract.ROOT_FILESYSTEM,
                        "size": root_size,
                        "avail": root_available,
                        "target": contract.ROOT_MOUNTPOINT,
                    }
                ]
            },
            "service_states": _service_states(),
        }
    return {
        "collected_at_unix": collected_at,
        "instance": _instance(status),
        "disk": _disk(disk_size),
        "guest": guest,
        "canonical_receipts": _canonical_receipt_bindings(
            stopped=status == "TERMINATED"
        ),
        "external_iam_receipt": _external_iam_receipt(collected_at),
    }


def _report(
    *,
    disk_size: int,
    root_size: int,
    root_available: int,
    boot: str = BOOT_1,
    status: str = "RUNNING",
    collected_at: int = NOW,
) -> dict[str, object]:
    return preflight.evaluate(
        _evidence(
            disk_size=disk_size,
            root_size=root_size,
            root_available=root_available,
            boot=boot,
            status=status,
            collected_at=collected_at,
        )
    )


def _source_report(*, collected_at: int = NOW) -> dict[str, object]:
    return _report(
        disk_size=contract.SOURCE_SIZE_GB,
        root_size=42_025_213_952,
        root_available=3_284_721_664,
        collected_at=collected_at,
    )


def _pending_report(
    *, collected_at: int, status: str = "RUNNING"
) -> dict[str, object]:
    return _report(
        disk_size=contract.TARGET_SIZE_GB,
        root_size=42_025_213_952,
        root_available=3_284_721_664,
        status=status,
        collected_at=collected_at,
    )


def _target_report(
    *,
    collected_at: int,
    boot: str = BOOT_1,
) -> dict[str, object]:
    return _report(
        disk_size=contract.TARGET_SIZE_GB,
        root_size=contract.MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES,
        root_available=40_000_000_000,
        boot=boot,
        collected_at=collected_at,
    )


def test_historical_20_to_40_contract_remains_separate() -> None:
    historical = host_storage.build_plan()
    growth_plan = growth.build_plan()

    assert host_storage.SOURCE_BOOT_DISK_SIZE_GB == 20
    assert host_storage.TARGET_BOOT_DISK_SIZE_GB == 40
    assert growth_plan.spec.source_size_gb == 40
    assert growth_plan.spec.target_size_gb == 80
    assert historical.sha256 != growth_plan.sha256


def test_growth_facade_renders_only_the_canonical_plan() -> None:
    plan = growth.build_plan()

    assert plan.report() == contract.canonical_plan_report()
    assert growth.CANARY_RUNTIME_UNITS == host_storage_boot.CANARY_RUNTIME_UNITS
    assert [step.name for step in plan.steps] == [
        contract.RESIZE_STEP,
        contract.STOP_STEP,
        contract.START_STEP,
    ]
    assert plan.architecture["local_journal_authority"] is False
    assert plan.architecture["guest_command_authority"] is False
    assert plan.architecture["shell_authority"] is False
    assert plan.architecture["cleanup_authority"] is False
    assert plan.architecture["delete_authority"] is False
    assert plan.architecture["snapshot_authority"] is False
    assert plan.architecture["opens_runtime_gate"] is False


def test_plan_and_preflight_direct_entrypoints_fail_closed() -> None:
    with pytest.raises(SystemExit, match="passkey-v2 owner-gate executor"):
        growth.main()
    with pytest.raises(SystemExit, match="release-bound full_canary_owner_launcher"):
        preflight.main()


@pytest.mark.parametrize(
    ("report", "state", "ok"),
    [
        (_source_report(), "source_ready", True),
        (_pending_report(collected_at=NOW), "resize_complete_boot_required", False),
        (_target_report(collected_at=NOW), "target_ready", True),
        (
            _pending_report(collected_at=NOW, status="TERMINATED"),
            "terminated_after_growth_intent",
            False,
        ),
    ],
)
def test_read_only_preflight_recognizes_only_bounded_states(
    report: dict[str, object], state: str, ok: bool
) -> None:
    assert report["state"] == state
    assert report["ok"] is ok
    assert report["plan_sha256"] == contract.canonical_plan_sha256()
    assert report["vm_instance_id"] == contract.VM_INSTANCE_ID
    assert report["disk_id"] == contract.DISK_ID


def test_initial_gate_accepts_only_a_fresh_exact_source_observation() -> None:
    plan = growth.build_plan()
    source = _source_report()

    assert growth._require_initial_preflight(
        source,
        plan=plan,
        now_unix=NOW,
    ) == source
    for invalid in (
        _pending_report(collected_at=NOW),
        _target_report(collected_at=NOW),
        _source_report(collected_at=NOW - contract.PREFLIGHT_MAX_AGE_SECONDS - 1),
    ):
        with pytest.raises(RuntimeError, match="not the exact 40 GB source"):
            growth._require_initial_preflight(
                invalid,
                plan=plan,
                now_unix=NOW,
            )


def test_preflight_rejects_missing_or_active_runtime_unit() -> None:
    evidence = _evidence(
        disk_size=contract.SOURCE_SIZE_GB,
        root_size=42_025_213_952,
        root_available=3_284_721_664,
    )
    missing = copy.deepcopy(evidence)
    missing["guest"]["service_states"].pop()  # type: ignore[index]
    active = copy.deepcopy(evidence)
    active["guest"]["service_states"][0]["properties"]["ActiveState"] = "active"  # type: ignore[index]

    assert preflight.evaluate(missing)["state"] == "invalid"
    assert preflight.evaluate(active)["state"] == "invalid"


def test_preflight_rejects_receipt_pin_drift() -> None:
    evidence = _evidence(
        disk_size=contract.SOURCE_SIZE_GB,
        root_size=42_025_213_952,
        root_available=3_284_721_664,
    )
    evidence["canonical_receipts"]["current_host_receipt_sha256"] = "0" * 64  # type: ignore[index]

    report = preflight.evaluate(evidence)

    assert report["state"] == "invalid"
    assert report["current_host_receipt_sha256"] is None


def test_preflight_requires_target_capacity_and_headroom() -> None:
    too_small = _report(
        disk_size=contract.TARGET_SIZE_GB,
        root_size=contract.MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES - 1,
        root_available=40_000_000_000,
    )
    low_free = _report(
        disk_size=contract.TARGET_SIZE_GB,
        root_size=contract.MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES,
        root_available=contract.MINIMUM_FREE_BYTES - 1,
    )

    assert too_small["state"] == "invalid"
    assert low_free["state"] == "invalid"


def test_collector_accepts_only_injected_readers_and_fixed_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="trusted collector boundary"):
        preflight.collect(
            run_json=lambda _argv: {},
            run_guest=None,  # type: ignore[arg-type]
            external_iam_receipt=_external_iam_receipt(NOW),
        )

    monkeypatch.setattr(preflight.time, "time", lambda: NOW)
    monkeypatch.setattr(
        preflight,
        "_validate_live_receipts",
        lambda _host, _stopped: _canonical_receipt_bindings(),
    )
    commands: list[tuple[str, ...]] = []

    def run_json(argv: Sequence[str]) -> object:
        logical = tuple(argv)
        commands.append(logical)
        return _instance() if "instances" in logical else _disk(contract.SOURCE_SIZE_GB)

    def run_guest(argv: Sequence[str]) -> bytes:
        logical = tuple(argv)
        commands.append(logical)
        if logical == host_storage_boot.BOOT_ID_COMMAND:
            return b"11111111-1111-1111-1111-111111111111\n"
        if logical in {
            preflight.HOST_RECEIPT_COMMAND,
            preflight.STOPPED_RELEASE_RECEIPT_COMMAND,
        }:
            return b"test-only-receipt"
        if logical[0] == "/usr/bin/findmnt":
            return json.dumps(
                {
                    "filesystems": [
                        {
                            "source": contract.ROOT_SOURCE,
                            "fstype": contract.ROOT_FILESYSTEM,
                            "size": 42_025_213_952,
                            "avail": 3_284_721_664,
                            "target": contract.ROOT_MOUNTPOINT,
                        }
                    ]
                }
            ).encode()
        unit = logical[-1]
        state = next(item for item in _service_states() if item["unit"] == unit)
        properties = cast(Mapping[str, object], state["properties"])
        return "\n".join(
            f"{name}={properties[name]}"
            for name in host_storage_boot._SERVICE_PROPERTIES
        ).encode()

    report = preflight.collect(
        run_json=run_json,
        run_guest=run_guest,
        external_iam_receipt=_external_iam_receipt(NOW),
    )

    assert report["state"] == "source_ready"
    assert len(commands) == 2 + 4 + len(growth.CANARY_RUNTIME_UNITS)
    assert all(
        command[0] not in {"sh", "bash", "/bin/sh", "/bin/bash"}
        for command in commands
    )
