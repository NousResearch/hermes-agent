#!/usr/bin/env python3
"""Pure compatibility facade for the reviewed 40 -> 80 GB canary plan.

This module intentionally has no approval author, journal, mutation runner, or
recovery engine.  Production mutation authority lives only in the split
passkey-v2 owner-gate executor.  The owner launcher imports this facade solely
to render the canonical plan and validate read-only source observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from scripts.canary import storage_growth_contract as contract
from scripts.canary import storage_growth_evidence as evidence
from scripts.canary.foundation import PlanStep
from scripts.canary.host_storage import _sha256_json
from scripts.canary.runtime_units import CANARY_RUNTIME_UNITS
from scripts.canary.writer_release import (
    _SERVICE_PROPERTIES,
    _parse_service_observation,
)


PROJECT = contract.PROJECT
ZONE = contract.ZONE
OWNER_ACCOUNT = contract.OWNER_ACCOUNT
VM_NAME = contract.VM_NAME
VM_INSTANCE_ID = contract.VM_INSTANCE_ID
DISK_NAME = contract.DISK_NAME
DISK_ID = contract.DISK_ID
BOOT_DEVICE_NAME = contract.BOOT_DEVICE_NAME
SOURCE_BOOT_DISK_SIZE_GB = contract.SOURCE_SIZE_GB
TARGET_BOOT_DISK_SIZE_GB = contract.TARGET_SIZE_GB
MINIMUM_SOURCE_ROOT_FILESYSTEM_BYTES = (
    contract.MINIMUM_SOURCE_ROOT_FILESYSTEM_BYTES
)
MAXIMUM_SOURCE_ROOT_FILESYSTEM_BYTES = (
    contract.MAXIMUM_SOURCE_ROOT_FILESYSTEM_BYTES
)
MINIMUM_SOURCE_FREE_BYTES = contract.MINIMUM_SOURCE_FREE_BYTES
MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES = (
    contract.MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES
)
MAXIMUM_TARGET_ROOT_FILESYSTEM_BYTES = (
    contract.MAXIMUM_TARGET_ROOT_FILESYSTEM_BYTES
)
MINIMUM_FREE_BYTES = contract.MINIMUM_FREE_BYTES
STORAGE_GROWTH_PLAN_SCHEMA = contract.STORAGE_GROWTH_PLAN_SCHEMA
STORAGE_GROWTH_PREFLIGHT_SCHEMA = contract.STORAGE_GROWTH_PREFLIGHT_SCHEMA
AUTHORITATIVE_JOURNAL_ROOT = contract.AUTHORITATIVE_JOURNAL_ROOT
AUTHORITATIVE_EXECUTOR_UID = contract.AUTHORITATIVE_EXECUTOR_UID
AUTHORITATIVE_EXECUTOR_GID = contract.AUTHORITATIVE_EXECUTOR_GID
RESIZE_STEP = contract.RESIZE_STEP
STOP_STEP = contract.STOP_STEP
START_STEP = contract.START_STEP


@dataclass(frozen=True)
class HostStorageGrowthSpec:
    project: str = contract.PROJECT
    zone: str = contract.ZONE
    owner_account: str = contract.OWNER_ACCOUNT
    vm_name: str = contract.VM_NAME
    vm_instance_id: str = contract.VM_INSTANCE_ID
    disk_name: str = contract.DISK_NAME
    disk_id: str = contract.DISK_ID
    boot_device_name: str = contract.BOOT_DEVICE_NAME
    source_size_gb: int = contract.SOURCE_SIZE_GB
    target_size_gb: int = contract.TARGET_SIZE_GB
    disk_type: str = contract.DISK_TYPE
    source_image_project: str = contract.SOURCE_IMAGE_PROJECT
    source_image: str = contract.SOURCE_IMAGE
    root_source: str = contract.ROOT_SOURCE
    root_filesystem: str = contract.ROOT_FILESYSTEM
    root_mountpoint: str = contract.ROOT_MOUNTPOINT
    minimum_target_root_filesystem_bytes: int = (
        contract.MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES
    )
    minimum_free_bytes: int = contract.MINIMUM_FREE_BYTES

    def validate(self) -> None:
        if self != HostStorageGrowthSpec():
            raise ValueError("host storage growth shape is fork-pinned")
        if asdict(self) != contract.canonical_plan_payload()["spec"]:
            raise ValueError("host storage growth contract drift")


@dataclass(frozen=True)
class HostStorageGrowthPlan:
    schema: str
    spec: HostStorageGrowthSpec
    architecture: Mapping[str, object]
    steps: tuple[PlanStep, ...]

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "spec": asdict(self.spec),
            "architecture": dict(self.architecture),
            "steps": [
                {"name": step.name, "argv": list(step.argv)}
                for step in self.steps
            ],
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.payload())

    def report(self) -> dict[str, object]:
        return {**self.payload(), "plan_sha256": self.sha256}


def build_plan(
    spec: HostStorageGrowthSpec | None = None,
) -> HostStorageGrowthPlan:
    exact = HostStorageGrowthSpec() if spec is None else spec
    exact.validate()
    payload = contract.canonical_plan_payload()
    plan = HostStorageGrowthPlan(
        schema=str(payload["schema"]),
        spec=exact,
        architecture=dict(payload["architecture"]),
        steps=tuple(
            PlanStep(
                str(step["name"]),
                tuple(str(item) for item in step["argv"]),
            )
            for step in payload["steps"]
        ),
    )
    if plan.report() != contract.canonical_plan_report():
        raise ValueError("host storage growth contract serialization drift")
    return plan


def _service_states_exact(raw: object) -> list[dict[str, object]] | None:
    if not isinstance(raw, list) or len(raw) != len(CANARY_RUNTIME_UNITS):
        return None
    result: list[dict[str, object]] = []
    for unit, item in zip(CANARY_RUNTIME_UNITS, raw, strict=True):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"unit", "state", "properties"}
            or item.get("unit") != unit
            or item.get("state") not in {"absent", "disabled_inactive"}
        ):
            return None
        properties = item.get("properties")
        if (
            not isinstance(properties, Mapping)
            or set(properties) != set(_SERVICE_PROPERTIES)
            or any(
                not isinstance(properties[name], str)
                for name in _SERVICE_PROPERTIES
            )
        ):
            return None
        rendered = "\n".join(
            f"{name}={properties[name]}" for name in _SERVICE_PROPERTIES
        )
        try:
            parsed = _parse_service_observation(unit, rendered)
        except RuntimeError:
            return None
        if parsed != dict(item):
            return None
        result.append(parsed)
    return result


def _require_initial_preflight(
    value: Mapping[str, object],
    *,
    plan: HostStorageGrowthPlan,
    now_unix: int,
) -> dict[str, object]:
    if plan.report() != contract.canonical_plan_report():
        raise RuntimeError("storage growth plan contract is not exact")
    try:
        report = evidence.validate_initial_observation(
            value,
            now_unix=now_unix,
        )
    except evidence.StorageGrowthEvidenceError as exc:
        raise RuntimeError(
            "storage growth preflight is not the exact 40 GB source"
        ) from None
    return dict(report)


def main() -> int:
    raise SystemExit(
        "storage growth is available only through the passkey-v2 "
        "owner-gate executor"
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITATIVE_EXECUTOR_GID",
    "AUTHORITATIVE_EXECUTOR_UID",
    "AUTHORITATIVE_JOURNAL_ROOT",
    "CANARY_RUNTIME_UNITS",
    "HostStorageGrowthPlan",
    "HostStorageGrowthSpec",
    "MINIMUM_FREE_BYTES",
    "MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES",
    "RESIZE_STEP",
    "SOURCE_BOOT_DISK_SIZE_GB",
    "START_STEP",
    "STOP_STEP",
    "STORAGE_GROWTH_PLAN_SCHEMA",
    "STORAGE_GROWTH_PREFLIGHT_SCHEMA",
    "TARGET_BOOT_DISK_SIZE_GB",
    "build_plan",
    "main",
]
