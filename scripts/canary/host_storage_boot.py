#!/usr/bin/env python3
"""Owner-approved boot transition for the pinned canary disk expansion.

The pinned Debian image expands its root partition and ext4 filesystem during
boot.  Expanding the persistent disk while the VM is online is therefore only
the first half of the transition.  This module owns the second half: one exact
stop/start of the identity-pinned canary, after a sealed report proves that all
canary runtime units are absent or disabled and inactive.

There is no operator-supplied command, shell, repair command, service start, or
filesystem mutation surface.  Retries are state-aware: a VM already stopped by
the approved transaction is only started, and a VM already on a different boot
is only attested.  A separate storage postflight remains responsible for
proving that the boot actually expanded the partition and filesystem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from scripts.canary.foundation import PREFLIGHT_MAX_AGE_SECONDS, PlanStep
from scripts.canary.host_storage import (
    DISK_ID,
    DISK_NAME,
    OWNER_ACCOUNT,
    TARGET_BOOT_DISK_SIZE_GB,
    TRUSTED_GCLOUD,
    TRUSTED_PYTHON,
    VM_INSTANCE_ID,
    _APPLY_RECEIPT_FIELDS,
    _require_fresh_report,
    _require_sealed,
    _seal,
    _sha256_json,
    build_plan as build_storage_plan,
)
from scripts.canary.host_storage_preflight import (
    SOURCE_ROOT_MAXIMUM_BYTES,
    SOURCE_ROOT_MINIMUM_BYTES,
    _STORAGE_CHECK_NAMES,
    _disk_exact,
    _instance_exact,
)
from scripts.canary.host_storage_preflight import collect as collect_storage
from scripts.canary.host_storage_preflight import evaluate as evaluate_storage
from scripts.canary.host_storage_boot_journal import (
    DEFAULT_JOURNAL_ROOT,
    StorageBootJournal,
)
from scripts.canary.runtime_units import CANARY_RUNTIME_UNITS
from scripts.canary.writer_release import (
    DEFAULT_SYSTEMCTL_EXECUTABLE,
    _SERVICE_PROPERTIES,
    _parse_service_observation,
)


BOOT_EXPANSION_PLAN_SCHEMA = "muncho-isolated-canary-storage-boot-plan.v1"
BOOT_EXPANSION_PREFLIGHT_SCHEMA = "muncho-isolated-canary-storage-boot-preflight.v1"
BOOT_EXPANSION_RECEIPT_SCHEMA = "muncho-isolated-canary-storage-boot-receipt.v1"
BOOT_EXPANSION_INTENT_SCHEMA = "muncho-isolated-canary-storage-boot-intent.v1"
BOOT_EXPANSION_STOP_SCHEMA = "muncho-isolated-canary-storage-boot-stop.v1"
BOOT_EXPANSION_START_SCHEMA = "muncho-isolated-canary-storage-boot-start.v1"
STOP_STEP = "stop_identity_pinned_canary_for_storage_boot"
START_STEP = "start_identity_pinned_canary_for_storage_boot"
BOOT_ID_COMMAND = ("/usr/bin/cat", "/proc/sys/kernel/random/boot_id")
_BOOT_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_GUEST_OUTPUT_BYTES = 64 * 1024
_TRANSACTION_MAX_AGE_SECONDS = 60 * 60
_PREFLIGHT_FIELDS = {
    "schema",
    "ok",
    "state",
    "collected_at_unix",
    "plan_sha256",
    "storage_plan_sha256",
    "storage_apply_receipt_sha256",
    "storage_report_sha256",
    "vm_instance_id",
    "disk_id",
    "disk_size_gb",
    "prior_boot_id_sha256",
    "runtime_units",
    "service_states",
    "service_states_sha256",
    "checks",
    "report_sha256",
}
_RECEIPT_FIELDS = {
    "schema",
    "ok",
    "plan_sha256",
    "approved_plan_sha256",
    "preflight_report_sha256",
    "storage_report_sha256",
    "storage_apply_receipt_sha256",
    "transaction_id",
    "journal_intent_sha256",
    "journal_stop_sha256",
    "journal_start_sha256",
    "completed_at_unix",
    "execution_state",
    "mutation_performed",
    "receipts",
    "vm_instance_id",
    "disk_id",
    "disk_size_gb",
    "prior_boot_id_sha256",
    "current_boot_id_sha256",
    "runtime_units",
    "service_states_before_sha256",
    "service_states_after_sha256",
    "initial_observation_sha256",
    "initial_observation_collected_at_unix",
    "stopped_observation_sha256",
    "stopped_observation_collected_at_unix",
    "started_observation_sha256",
    "started_observation_collected_at_unix",
    "stop_command_completed_at_unix",
    "start_command_completed_at_unix",
    "boot_changed",
    "requires_storage_postflight",
    "opens_runtime_gate",
    "receipt_sha256",
}
_CHECK_NAMES = {
    "storage.transition_pending_exact",
    "runtime.units_inventory_exact",
    "runtime.units_stopped_exact",
    "host.prior_boot_identity_exact",
}


@dataclass(frozen=True)
class StorageBootPlan:
    schema: str
    storage_plan_sha256: str
    architecture: Mapping[str, object]
    steps: tuple[PlanStep, ...]

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "storage_plan_sha256": self.storage_plan_sha256,
            "architecture": dict(self.architecture),
            "steps": [asdict(step) for step in self.steps],
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.payload())

    def report(self) -> dict[str, object]:
        return {**self.payload(), "plan_sha256": self.sha256}


def build_plan() -> StorageBootPlan:
    """Return the sole approved stop/start shape for this one pinned host."""

    storage_plan = build_storage_plan()
    common = (
        f"--project={storage_plan.spec.project}",
        f"--zone={storage_plan.spec.zone}",
        f"--account={storage_plan.spec.owner_account}",
        "--quiet",
    )
    return StorageBootPlan(
        schema=BOOT_EXPANSION_PLAN_SCHEMA,
        storage_plan_sha256=storage_plan.sha256,
        architecture={
            "phase": "host_storage_boot_expansion",
            "identity_pinned_vm": VM_INSTANCE_ID,
            "identity_pinned_disk": DISK_ID,
            "requires_40gb_disk_with_source_sized_root": True,
            "requires_all_runtime_units_absent_or_disabled_inactive": True,
            "conditional_exact_stop_start": True,
            "crash_retry_state_aware": True,
            "append_only_owner_journal": str(DEFAULT_JOURNAL_ROOT),
            "intent_fsynced_before_stop": True,
            "observation_chronology_bound": True,
            "guest_command_authority": False,
            "shell_authority": False,
            "starts_canary_runtime_units": False,
            "requires_storage_postflight": True,
            "opens_runtime_gate": False,
        },
        steps=(
            PlanStep(
                STOP_STEP,
                (
                    "gcloud",
                    "compute",
                    "instances",
                    "stop",
                    storage_plan.spec.vm_name,
                    *common,
                ),
            ),
            PlanStep(
                START_STEP,
                (
                    "gcloud",
                    "compute",
                    "instances",
                    "start",
                    storage_plan.spec.vm_name,
                    *common,
                ),
            ),
        ),
    )


def _service_states_exact(raw: object) -> list[dict[str, object]] | None:
    if not isinstance(raw, list) or len(raw) != len(CANARY_RUNTIME_UNITS):
        return None
    validated: list[dict[str, object]] = []
    for expected_unit, item in zip(CANARY_RUNTIME_UNITS, raw, strict=True):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"unit", "state", "properties"}
            or item.get("unit") != expected_unit
            or item.get("state") not in {"absent", "disabled_inactive"}
        ):
            return None
        properties = item.get("properties")
        if not isinstance(properties, Mapping) or set(properties) != set(
            _SERVICE_PROPERTIES
        ):
            return None
        if any(not isinstance(properties[name], str) for name in _SERVICE_PROPERTIES):
            return None
        rendered = "\n".join(
            f"{name}={properties[name]}" for name in _SERVICE_PROPERTIES
        )
        try:
            parsed = _parse_service_observation(expected_unit, rendered)
        except RuntimeError:
            return None
        if parsed != dict(item):
            return None
        validated.append(parsed)
    return validated


def _storage_transition_pending_exact(
    report: Mapping[str, object], *, now_unix: int
) -> bool:
    storage_plan = build_storage_plan()
    try:
        _require_fresh_report(report, plan=storage_plan, now_unix=now_unix)
    except RuntimeError:
        return False
    raw_checks = report.get("checks")
    if not isinstance(raw_checks, list) or len(raw_checks) != len(_STORAGE_CHECK_NAMES):
        return False
    checks: dict[str, bool] = {}
    for item in raw_checks:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"name", "passed", "detail"}
            or item.get("name") not in _STORAGE_CHECK_NAMES
            or type(item.get("passed")) is not bool
            or not isinstance(item.get("detail"), str)
            or not item["detail"]
            or item["name"] in checks
        ):
            return False
        checks[str(item["name"])] = bool(item["passed"])
    required_passes = _STORAGE_CHECK_NAMES - {
        "filesystem.capacity_matches_disk_state",
        "filesystem.packaging_headroom_if_target",
    }
    root_size = report.get("root_size_bytes")
    root_available = report.get("root_available_bytes")
    return bool(
        set(checks) == _STORAGE_CHECK_NAMES
        and all(checks[name] for name in required_passes)
        and checks["filesystem.capacity_matches_disk_state"] is False
        and report.get("schema") == "muncho-isolated-canary-host-storage-preflight.v1"
        and report.get("ok") is False
        and report.get("state") == "transition_pending"
        and report.get("plan_sha256") == storage_plan.sha256
        and report.get("vm_instance_id") == VM_INSTANCE_ID
        and report.get("disk_id") == DISK_ID
        and report.get("disk_size_gb") == TARGET_BOOT_DISK_SIZE_GB
        and report.get("normal_host_preflight_ok") is True
        and report.get("satisfied_steps") == []
        and report.get("ready_for_packaging") is False
        and type(root_size) is int
        and SOURCE_ROOT_MINIMUM_BYTES <= root_size <= SOURCE_ROOT_MAXIMUM_BYTES
        and type(root_available) is int
        and 0 <= root_available <= root_size
    )


def build_preflight(
    *,
    storage_apply_receipt: Mapping[str, object],
    storage_report: Mapping[str, object],
    prior_boot_id_sha256: str,
    service_states: object,
    now_unix: int | None = None,
) -> dict[str, object]:
    """Seal the exact pre-reboot disk, boot, and stopped-unit evidence."""

    now = int(time.time()) if now_unix is None else now_unix
    if type(now) is not int or now < 0:
        raise RuntimeError("storage boot preflight time is invalid")
    plan = build_plan()
    try:
        _require_sealed(
            storage_apply_receipt,
            digest_key="receipt_sha256",
            label="storage apply receipt",
        )
    except RuntimeError:
        apply_exact = False
    else:
        apply_exact = bool(
            set(storage_apply_receipt) == _APPLY_RECEIPT_FIELDS
            and storage_apply_receipt.get("schema")
            == "muncho-isolated-canary-host-storage-apply-receipt.v1"
            and storage_apply_receipt.get("ok") is True
            and storage_apply_receipt.get("plan_sha256") == plan.storage_plan_sha256
            and storage_apply_receipt.get("approved_plan_sha256")
            == plan.storage_plan_sha256
            and storage_apply_receipt.get("mutation_performed") is True
            and storage_apply_receipt.get("requires_post_apply_attestation") is True
            and type(storage_apply_receipt.get("completed_at_unix")) is int
            and 0 <= storage_apply_receipt["completed_at_unix"] <= now
        )
    storage_exact = _storage_transition_pending_exact(storage_report, now_unix=now)
    if (
        apply_exact
        and type(storage_report.get("collected_at_unix")) is int
        and storage_report["collected_at_unix"]
        < storage_apply_receipt["completed_at_unix"]
    ):
        storage_exact = False
    services = _service_states_exact(service_states)
    boot_exact = bool(
        isinstance(prior_boot_id_sha256, str)
        and _SHA256.fullmatch(prior_boot_id_sha256) is not None
    )
    checks = [
        {
            "name": "storage.transition_pending_exact",
            "passed": storage_exact and apply_exact,
            "detail": "the pinned 40 GB disk must still expose the exact source-sized root",
        },
        {
            "name": "runtime.units_inventory_exact",
            "passed": services is not None,
            "detail": "the complete fixed canary runtime unit inventory must be present",
        },
        {
            "name": "runtime.units_stopped_exact",
            "passed": services is not None,
            "detail": "every canary runtime unit must be absent or disabled and inactive",
        },
        {
            "name": "host.prior_boot_identity_exact",
            "passed": boot_exact,
            "detail": "the current boot UUID is retained only as a SHA-256 digest",
        },
    ]
    ok = all(item["passed"] is True for item in checks)
    unsigned = {
        "schema": BOOT_EXPANSION_PREFLIGHT_SCHEMA,
        "ok": ok,
        "state": "boot_required" if ok else "invalid",
        "collected_at_unix": now,
        "plan_sha256": plan.sha256,
        "storage_plan_sha256": plan.storage_plan_sha256,
        "storage_apply_receipt_sha256": storage_apply_receipt.get("receipt_sha256"),
        "storage_report_sha256": storage_report.get("report_sha256"),
        "vm_instance_id": storage_report.get("vm_instance_id"),
        "disk_id": storage_report.get("disk_id"),
        "disk_size_gb": storage_report.get("disk_size_gb"),
        "prior_boot_id_sha256": prior_boot_id_sha256,
        "runtime_units": list(CANARY_RUNTIME_UNITS),
        "service_states": services if services is not None else [],
        "service_states_sha256": _sha256_json(services or []),
        "checks": checks,
    }
    return _seal(unsigned, digest_key="report_sha256")


def _require_preflight(
    value: Mapping[str, object],
    *,
    plan: StorageBootPlan,
    now_unix: int,
    require_fresh: bool,
) -> None:
    if set(value) != _PREFLIGHT_FIELDS:
        raise RuntimeError("storage boot preflight fields are not exact")
    _require_sealed(
        value,
        digest_key="report_sha256",
        label="storage boot preflight",
    )
    collected_at = value.get("collected_at_unix")
    services = _service_states_exact(value.get("service_states"))
    checks = value.get("checks")
    if (
        value.get("schema") != BOOT_EXPANSION_PREFLIGHT_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "boot_required"
        or value.get("plan_sha256") != plan.sha256
        or value.get("storage_plan_sha256") != plan.storage_plan_sha256
        or not isinstance(value.get("storage_apply_receipt_sha256"), str)
        or _SHA256.fullmatch(str(value.get("storage_apply_receipt_sha256"))) is None
        or not isinstance(value.get("storage_report_sha256"), str)
        or _SHA256.fullmatch(str(value.get("storage_report_sha256"))) is None
        or value.get("vm_instance_id") != VM_INSTANCE_ID
        or value.get("disk_id") != DISK_ID
        or value.get("disk_size_gb") != TARGET_BOOT_DISK_SIZE_GB
        or not isinstance(value.get("prior_boot_id_sha256"), str)
        or _SHA256.fullmatch(str(value.get("prior_boot_id_sha256"))) is None
        or value.get("runtime_units") != list(CANARY_RUNTIME_UNITS)
        or services is None
        or value.get("service_states_sha256") != _sha256_json(services)
        or not isinstance(checks, list)
        or {item.get("name") for item in checks if isinstance(item, Mapping)}
        != _CHECK_NAMES
        or len(checks) != len(_CHECK_NAMES)
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "passed", "detail"}
            or item.get("passed") is not True
            or not isinstance(item.get("detail"), str)
            or not item["detail"]
            for item in checks
        )
        or type(collected_at) is not int
        or collected_at < 0
        or collected_at > now_unix
        or (require_fresh and now_unix - collected_at > PREFLIGHT_MAX_AGE_SECONDS)
    ):
        raise RuntimeError("storage boot preflight is not exact")


def _instance_identity_exact(raw: object, *, status: str) -> bool:
    if not isinstance(raw, Mapping) or raw.get("status") != status:
        return False
    normalized = dict(raw)
    normalized["status"] = "RUNNING"
    return _instance_exact(normalized)


def _disk_target_exact(raw: object) -> bool:
    if not _disk_exact(raw) or not isinstance(raw, Mapping):
        return False
    try:
        return int(str(raw.get("sizeGb"))) == TARGET_BOOT_DISK_SIZE_GB
    except ValueError:
        return False


def _require_observation(raw: Mapping[str, object]) -> dict[str, object]:
    if set(raw) != {
        "collected_at_unix",
        "instance",
        "disk",
        "boot_id_sha256",
        "service_states",
    }:
        raise RuntimeError("storage boot observation fields are not exact")
    instance = raw.get("instance")
    status = instance.get("status") if isinstance(instance, Mapping) else None
    collected_at = raw.get("collected_at_unix")
    if type(collected_at) is not int or collected_at < 0:
        raise RuntimeError("storage boot observation time is invalid")
    if status not in {"RUNNING", "TERMINATED"}:
        raise RuntimeError("storage boot observation lifecycle is not stable")
    if not _instance_identity_exact(
        instance, status=str(status)
    ) or not _disk_target_exact(raw.get("disk")):
        raise RuntimeError("storage boot observation identity is not exact")
    services = None
    boot_id = raw.get("boot_id_sha256")
    if status == "RUNNING":
        services = _service_states_exact(raw.get("service_states"))
        if (
            not isinstance(boot_id, str)
            or _SHA256.fullmatch(boot_id) is None
            or services is None
        ):
            raise RuntimeError("storage boot running observation is incomplete")
    elif boot_id is not None or raw.get("service_states") is not None:
        raise RuntimeError("storage boot stopped observation has guest evidence")
    return {
        "status": status,
        "collected_at_unix": collected_at,
        "observation_sha256": _sha256_json(dict(raw)),
        "boot_id_sha256": boot_id,
        "service_states": services,
    }


def _require_live_observation(
    raw: Mapping[str, object],
    *,
    now_unix: int,
    strictly_after_unix: int | None = None,
    command_completed_at_unix: int | None = None,
) -> dict[str, object]:
    value = _require_observation(raw)
    collected_at = value["collected_at_unix"]
    if (
        type(now_unix) is not int
        or type(collected_at) is not int
        or collected_at > now_unix
        or now_unix - collected_at > PREFLIGHT_MAX_AGE_SECONDS
        or (strictly_after_unix is not None and collected_at <= strictly_after_unix)
        or (
            command_completed_at_unix is not None
            and collected_at < command_completed_at_unix
        )
    ):
        raise RuntimeError("storage boot observation is stale or out of order")
    return value


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
Observer = Callable[[], Mapping[str, object]]
Clock = Callable[[], int]
Checkpoint = Callable[[str], None]


def _runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    plan = build_plan()
    allowed = {step.argv for step in plan.steps}
    logical = tuple(argv)
    if logical not in allowed:
        raise RuntimeError("storage boot mutation argv is not exact")

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
    identity = GcloudOwnerAccessToken(
        gcloud_executable=executable,
        gcloud_configuration=configuration,
    )
    if identity.account_for_read_only_preflight() != OWNER_ACCOUNT:
        raise RuntimeError("storage boot owner identity is not exact")
    prefix = executable.trusted_command_prefix()
    environment = _owner_gcloud_environment(configuration, prefix[0])
    try:
        completed = subprocess.run(
            (*prefix, *logical[1:]),
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
        identity.require_stable()
    if (
        not isinstance(completed.stdout, bytes)
        or not isinstance(completed.stderr, bytes)
        or len(completed.stdout) > _MAX_GUEST_OUTPUT_BYTES
        or len(completed.stderr) > _MAX_GUEST_OUTPUT_BYTES
    ):
        raise RuntimeError("storage boot mutation output is invalid")
    try:
        stdout = completed.stdout.decode("utf-8", errors="strict")
        stderr = completed.stderr.decode("utf-8", errors="strict")
    except UnicodeError:
        raise RuntimeError("storage boot mutation output is invalid") from None
    return subprocess.CompletedProcess(logical, completed.returncode, stdout, stderr)


def _step_receipt(
    name: str, *, result: str, completed: subprocess.CompletedProcess[str] | None
) -> dict[str, object]:
    if completed is None:
        return {"name": name, "result": result}
    return {
        "name": name,
        "result": result,
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
    }


def _step_receipt_exact(
    value: object,
    *,
    name: str,
    result: str,
    command_evidenced: bool,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not command_evidenced:
        return dict(value) == {"name": name, "result": result}
    return bool(
        set(value)
        == {
            "name",
            "result",
            "returncode",
            "stdout_sha256",
            "stderr_sha256",
        }
        and value.get("name") == name
        and value.get("result") == result
        and value.get("returncode") == 0
        and isinstance(value.get("stdout_sha256"), str)
        and _SHA256.fullmatch(str(value.get("stdout_sha256"))) is not None
        and isinstance(value.get("stderr_sha256"), str)
        and _SHA256.fullmatch(str(value.get("stderr_sha256"))) is not None
    )


def _transaction_id(plan: StorageBootPlan, preflight: Mapping[str, object]) -> str:
    return _sha256_json({
        "plan_sha256": plan.sha256,
        "approved_plan_sha256": plan.sha256,
        "preflight_report_sha256": preflight["report_sha256"],
        "storage_apply_receipt_sha256": preflight["storage_apply_receipt_sha256"],
        "storage_report_sha256": preflight["storage_report_sha256"],
        "vm_instance_id": VM_INSTANCE_ID,
        "disk_id": DISK_ID,
        "disk_size_gb": TARGET_BOOT_DISK_SIZE_GB,
        "prior_boot_id_sha256": preflight["prior_boot_id_sha256"],
        "service_states_before_sha256": preflight["service_states_sha256"],
    })


def _build_intent(
    *,
    plan: StorageBootPlan,
    preflight: Mapping[str, object],
    transaction_id: str,
    observation: Mapping[str, object],
    published_at_unix: int,
) -> dict[str, object]:
    unsigned = {
        "schema": BOOT_EXPANSION_INTENT_SCHEMA,
        "transaction_id": transaction_id,
        "plan_sha256": plan.sha256,
        "approved_plan_sha256": plan.sha256,
        "preflight_report_sha256": preflight["report_sha256"],
        "storage_apply_receipt_sha256": preflight["storage_apply_receipt_sha256"],
        "storage_report_sha256": preflight["storage_report_sha256"],
        "vm_instance_id": VM_INSTANCE_ID,
        "disk_id": DISK_ID,
        "disk_size_gb": TARGET_BOOT_DISK_SIZE_GB,
        "prior_boot_id_sha256": preflight["prior_boot_id_sha256"],
        "runtime_units": list(CANARY_RUNTIME_UNITS),
        "service_states_before_sha256": preflight["service_states_sha256"],
        "initial_observation_sha256": observation["observation_sha256"],
        "initial_service_states_sha256": _sha256_json(observation["service_states"]),
        "initial_observation_collected_at_unix": observation["collected_at_unix"],
        "published_at_unix": published_at_unix,
        "expires_at_unix": published_at_unix + _TRANSACTION_MAX_AGE_SECONDS,
        "state": "intent_published_before_stop",
    }
    return _seal(unsigned, digest_key="intent_sha256")


def _require_intent(
    value: Mapping[str, object],
    *,
    plan: StorageBootPlan,
    preflight: Mapping[str, object],
    transaction_id: str,
    now_unix: int,
    require_unexpired: bool = True,
) -> dict[str, object]:
    expected_fields = {
        "schema",
        "transaction_id",
        "plan_sha256",
        "approved_plan_sha256",
        "preflight_report_sha256",
        "storage_apply_receipt_sha256",
        "storage_report_sha256",
        "vm_instance_id",
        "disk_id",
        "disk_size_gb",
        "prior_boot_id_sha256",
        "runtime_units",
        "service_states_before_sha256",
        "initial_observation_sha256",
        "initial_service_states_sha256",
        "initial_observation_collected_at_unix",
        "published_at_unix",
        "expires_at_unix",
        "state",
        "intent_sha256",
    }
    if set(value) != expected_fields:
        raise RuntimeError("storage boot journal intent fields are not exact")
    _require_sealed(value, digest_key="intent_sha256", label="storage boot intent")
    published = value.get("published_at_unix")
    expires = value.get("expires_at_unix")
    observed = value.get("initial_observation_collected_at_unix")
    if (
        value.get("schema") != BOOT_EXPANSION_INTENT_SCHEMA
        or value.get("transaction_id") != transaction_id
        or value.get("plan_sha256") != plan.sha256
        or value.get("approved_plan_sha256") != plan.sha256
        or value.get("preflight_report_sha256") != preflight.get("report_sha256")
        or value.get("storage_apply_receipt_sha256")
        != preflight.get("storage_apply_receipt_sha256")
        or value.get("storage_report_sha256") != preflight.get("storage_report_sha256")
        or value.get("vm_instance_id") != VM_INSTANCE_ID
        or value.get("disk_id") != DISK_ID
        or value.get("disk_size_gb") != TARGET_BOOT_DISK_SIZE_GB
        or value.get("prior_boot_id_sha256") != preflight.get("prior_boot_id_sha256")
        or value.get("runtime_units") != list(CANARY_RUNTIME_UNITS)
        or value.get("service_states_before_sha256")
        != preflight.get("service_states_sha256")
        or not isinstance(value.get("initial_observation_sha256"), str)
        or _SHA256.fullmatch(str(value.get("initial_observation_sha256"))) is None
        or value.get("initial_service_states_sha256")
        != preflight.get("service_states_sha256")
        or type(observed) is not int
        or type(published) is not int
        or type(expires) is not int
        or type(now_unix) is not int
        or now_unix < published
        or not preflight["collected_at_unix"] <= observed <= published
        or expires != published + _TRANSACTION_MAX_AGE_SECONDS
        or (require_unexpired and now_unix > expires)
        or value.get("state") != "intent_published_before_stop"
    ):
        raise RuntimeError("storage boot journal intent is stale or unrelated")
    return dict(value)


def _build_transition_record(
    *,
    schema: str,
    digest_key: str,
    transaction_id: str,
    intent: Mapping[str, object],
    result: str,
    step_receipt: Mapping[str, object],
    command_completed_at_unix: int | None,
    observation: Mapping[str, object] | None,
    recorded_at_unix: int,
) -> dict[str, object]:
    unsigned = {
        "schema": schema,
        "transaction_id": transaction_id,
        "intent_sha256": intent["intent_sha256"],
        "result": result,
        "step_receipt": dict(step_receipt),
        "command_completed_at_unix": command_completed_at_unix,
        "observation_sha256": (
            observation["observation_sha256"] if observation is not None else None
        ),
        "observation_collected_at_unix": (
            observation["collected_at_unix"] if observation is not None else None
        ),
        "boot_id_sha256": (
            observation["boot_id_sha256"] if observation is not None else None
        ),
        "service_states_sha256": (
            _sha256_json(observation["service_states"])
            if observation is not None and observation["service_states"] is not None
            else None
        ),
        "recorded_at_unix": recorded_at_unix,
    }
    return _seal(unsigned, digest_key=digest_key)


def _require_transition_record(
    value: Mapping[str, object],
    *,
    schema: str,
    digest_key: str,
    transaction_id: str,
    intent: Mapping[str, object],
    allowed_results: set[str],
) -> dict[str, object]:
    expected = {
        "schema",
        "transaction_id",
        "intent_sha256",
        "result",
        "step_receipt",
        "command_completed_at_unix",
        "observation_sha256",
        "observation_collected_at_unix",
        "boot_id_sha256",
        "service_states_sha256",
        "recorded_at_unix",
        digest_key,
    }
    if set(value) != expected:
        raise RuntimeError("storage boot journal transition fields are not exact")
    _require_sealed(value, digest_key=digest_key, label="storage boot transition")
    result = value.get("result")
    observed = value.get("observation_collected_at_unix")
    command_time = value.get("command_completed_at_unix")
    recorded = value.get("recorded_at_unix")
    if schema == BOOT_EXPANSION_STOP_SCHEMA and digest_key == "stop_sha256":
        expected_name = STOP_STEP
        result_shapes = {
            "stopped": (True, True, False),
            "verified_terminated_after_intent": (False, True, False),
            "unobserved_stop_authorized_by_intent": (False, False, False),
        }
    elif schema == BOOT_EXPANSION_START_SCHEMA and digest_key == "start_sha256":
        expected_name = START_STEP
        result_shapes = {
            "started": (True, True, True),
            "verified_new_boot_after_intent": (False, True, True),
        }
    else:
        raise RuntimeError("storage boot journal transition schema is invalid")
    shape = result_shapes.get(str(result))
    if shape is None:
        raise RuntimeError("storage boot journal transition result is invalid")
    command_evidenced, observation_evidenced, running_observation = shape
    observation_digest = value.get("observation_sha256")
    boot_id = value.get("boot_id_sha256")
    service_states_digest = value.get("service_states_sha256")
    if (
        value.get("schema") != schema
        or value.get("transaction_id") != transaction_id
        or value.get("intent_sha256") != intent.get("intent_sha256")
        or result not in allowed_results
        or not _step_receipt_exact(
            value.get("step_receipt"),
            name=expected_name,
            result=str(result),
            command_evidenced=command_evidenced,
        )
        or type(recorded) is not int
        or recorded < intent["published_at_unix"]
        or (command_evidenced and type(command_time) is not int)
        or (not command_evidenced and command_time is not None)
        or (
            command_evidenced
            and not intent["published_at_unix"] <= command_time <= recorded
        )
        or (observation_evidenced and type(observed) is not int)
        or (not observation_evidenced and observed is not None)
        or (
            observation_evidenced
            and (
                observed <= intent["initial_observation_collected_at_unix"]
                or observed > recorded
                or (command_time is not None and observed < command_time)
            )
        )
        or (
            observation_evidenced
            and (
                not isinstance(observation_digest, str)
                or _SHA256.fullmatch(observation_digest) is None
            )
        )
        or (not observation_evidenced and observation_digest is not None)
        or (
            running_observation
            and (
                not isinstance(boot_id, str)
                or _SHA256.fullmatch(boot_id) is None
                or boot_id == intent.get("prior_boot_id_sha256")
                or not isinstance(service_states_digest, str)
                or _SHA256.fullmatch(service_states_digest) is None
            )
        )
        or (
            not running_observation
            and (boot_id is not None or service_states_digest is not None)
        )
    ):
        raise RuntimeError("storage boot journal transition is invalid")
    return dict(value)


def execute_plan(
    plan: StorageBootPlan,
    *,
    approved_plan_sha256: str,
    preflight: Mapping[str, object],
    observer: Observer | None = None,
    runner: Runner = _runner,
    journal: StorageBootJournal | None = None,
    clock: Clock | None = None,
    checkpoint: Checkpoint | None = None,
    now_unix: int | None = None,
) -> dict[str, object]:
    """Run or recover the exact journaled boot without inventing causality."""

    if approved_plan_sha256 != plan.sha256:
        raise RuntimeError("approved storage boot plan digest mismatch")
    if clock is None:
        clock = lambda: int(time.time()) if now_unix is None else now_unix
    checkpoint = checkpoint or (lambda _name: None)
    now = clock()
    if type(now) is not int or now < 0:
        raise RuntimeError("storage boot execution time is invalid")
    if observer is None:
        observer = _sealed_observer()
    journal = journal or StorageBootJournal()
    _require_preflight(preflight, plan=plan, now_unix=now, require_fresh=False)
    transaction_id = _transaction_id(plan, preflight)
    completion = journal.read(transaction_id, "completion")
    if completion is not None:
        return validate_receipt(
            plan=plan,
            preflight=preflight,
            receipt=completion,
            now_unix=now,
            journal=journal,
        )

    current_raw = observer()
    observed_now = clock()
    current = _require_live_observation(current_raw, now_unix=observed_now)
    prior_boot = str(preflight["prior_boot_id_sha256"])
    intent = journal.read(transaction_id, "intent")
    if intent is None:
        if current["status"] != "RUNNING" or current["boot_id_sha256"] != prior_boot:
            raise RuntimeError(
                "storage boot recovery requires a durable pre-stop intent"
            )
        if (
            _sha256_json(current["service_states"])
            != preflight["service_states_sha256"]
        ):
            raise RuntimeError("storage boot live service state diverged before intent")
        published_at = clock()
        _require_preflight(
            preflight,
            plan=plan,
            now_unix=published_at,
            require_fresh=True,
        )
        if (
            published_at < current["collected_at_unix"]
            or published_at - current["collected_at_unix"] > PREFLIGHT_MAX_AGE_SECONDS
        ):
            raise RuntimeError("storage boot intent chronology is invalid")
        intent = journal.publish(
            transaction_id,
            "intent",
            _build_intent(
                plan=plan,
                preflight=preflight,
                transaction_id=transaction_id,
                observation=current,
                published_at_unix=published_at,
            ),
        )
        checkpoint("intent_published")
    intent = _require_intent(
        intent,
        plan=plan,
        preflight=preflight,
        transaction_id=transaction_id,
        now_unix=clock(),
    )

    stop_record = journal.read(transaction_id, "stop")
    if stop_record is not None:
        stop_record = _require_transition_record(
            stop_record,
            schema=BOOT_EXPANSION_STOP_SCHEMA,
            digest_key="stop_sha256",
            transaction_id=transaction_id,
            intent=intent,
            allowed_results={
                "stopped",
                "verified_terminated_after_intent",
                "unobserved_stop_authorized_by_intent",
            },
        )

    if current["status"] == "RUNNING" and current["boot_id_sha256"] == prior_boot:
        if stop_record is not None:
            raise RuntimeError("storage boot journal progress contradicts live boot")
        stopped = runner(plan.steps[0].argv)
        stop_command_time = clock()
        checkpoint("stop_command_completed")
        if stopped.returncode != 0:
            raise RuntimeError("storage boot stop failed")
        stop_raw = observer()
        stop_observed_now = clock()
        current = _require_live_observation(
            stop_raw,
            now_unix=stop_observed_now,
            strictly_after_unix=intent["initial_observation_collected_at_unix"],
            command_completed_at_unix=stop_command_time,
        )
        if current["status"] != "TERMINATED":
            raise RuntimeError("storage boot stop did not reach terminated state")
        stop_record = journal.publish(
            transaction_id,
            "stop",
            _build_transition_record(
                schema=BOOT_EXPANSION_STOP_SCHEMA,
                digest_key="stop_sha256",
                transaction_id=transaction_id,
                intent=intent,
                result="stopped",
                step_receipt=_step_receipt(
                    STOP_STEP, result="stopped", completed=stopped
                ),
                command_completed_at_unix=stop_command_time,
                observation=current,
                recorded_at_unix=clock(),
            ),
        )
        checkpoint("stop_recorded")
    elif current["status"] == "TERMINATED":
        if stop_record is None:
            if (
                current["collected_at_unix"]
                <= intent["initial_observation_collected_at_unix"]
            ):
                raise RuntimeError("storage boot terminated observation is stale")
            stop_record = journal.publish(
                transaction_id,
                "stop",
                _build_transition_record(
                    schema=BOOT_EXPANSION_STOP_SCHEMA,
                    digest_key="stop_sha256",
                    transaction_id=transaction_id,
                    intent=intent,
                    result="verified_terminated_after_intent",
                    step_receipt=_step_receipt(
                        STOP_STEP,
                        result="verified_terminated_after_intent",
                        completed=None,
                    ),
                    command_completed_at_unix=None,
                    observation=current,
                    recorded_at_unix=clock(),
                ),
            )
            checkpoint("stop_recorded")
    elif current["boot_id_sha256"] != prior_boot and stop_record is None:
        stop_record = journal.publish(
            transaction_id,
            "stop",
            _build_transition_record(
                schema=BOOT_EXPANSION_STOP_SCHEMA,
                digest_key="stop_sha256",
                transaction_id=transaction_id,
                intent=intent,
                result="unobserved_stop_authorized_by_intent",
                step_receipt=_step_receipt(
                    STOP_STEP,
                    result="unobserved_stop_authorized_by_intent",
                    completed=None,
                ),
                command_completed_at_unix=None,
                observation=None,
                recorded_at_unix=clock(),
            ),
        )
        checkpoint("stop_recorded")

    assert stop_record is not None
    start_record = journal.read(transaction_id, "start")
    if start_record is not None:
        start_record = _require_transition_record(
            start_record,
            schema=BOOT_EXPANSION_START_SCHEMA,
            digest_key="start_sha256",
            transaction_id=transaction_id,
            intent=intent,
            allowed_results={"started", "verified_new_boot_after_intent"},
        )
        if current["status"] == "RUNNING" and (
            current["boot_id_sha256"] != start_record["boot_id_sha256"]
            or _sha256_json(current["service_states"])
            != start_record["service_states_sha256"]
        ):
            raise RuntimeError(
                "storage boot terminal journal contradicts live observation"
            )

    if current["status"] == "TERMINATED":
        if start_record is not None:
            raise RuntimeError("storage boot start journal contradicts terminated VM")
        intent = _require_intent(
            intent,
            plan=plan,
            preflight=preflight,
            transaction_id=transaction_id,
            now_unix=clock(),
        )
        started = runner(plan.steps[1].argv)
        start_command_time = clock()
        checkpoint("start_command_completed")
        if started.returncode != 0:
            raise RuntimeError("storage boot start failed")
        baseline = stop_record.get("observation_collected_at_unix")
        if type(baseline) is not int:
            baseline = intent["initial_observation_collected_at_unix"]
        start_raw = observer()
        start_observed_now = clock()
        current = _require_live_observation(
            start_raw,
            now_unix=start_observed_now,
            strictly_after_unix=baseline,
            command_completed_at_unix=start_command_time,
        )
        if current["status"] != "RUNNING" or current["boot_id_sha256"] == prior_boot:
            raise RuntimeError("storage boot did not establish a new boot")
        start_record = journal.publish(
            transaction_id,
            "start",
            _build_transition_record(
                schema=BOOT_EXPANSION_START_SCHEMA,
                digest_key="start_sha256",
                transaction_id=transaction_id,
                intent=intent,
                result="started",
                step_receipt=_step_receipt(
                    START_STEP, result="started", completed=started
                ),
                command_completed_at_unix=start_command_time,
                observation=current,
                recorded_at_unix=clock(),
            ),
        )
        checkpoint("start_recorded")
    elif current["boot_id_sha256"] != prior_boot and start_record is None:
        baseline = stop_record.get("observation_collected_at_unix")
        if type(baseline) is not int:
            baseline = intent["initial_observation_collected_at_unix"]
        if current["collected_at_unix"] <= baseline:
            raise RuntimeError("storage boot recovered observation is stale")
        start_record = journal.publish(
            transaction_id,
            "start",
            _build_transition_record(
                schema=BOOT_EXPANSION_START_SCHEMA,
                digest_key="start_sha256",
                transaction_id=transaction_id,
                intent=intent,
                result="verified_new_boot_after_intent",
                step_receipt=_step_receipt(
                    START_STEP,
                    result="verified_new_boot_after_intent",
                    completed=None,
                ),
                command_completed_at_unix=None,
                observation=current,
                recorded_at_unix=clock(),
            ),
        )
        checkpoint("start_recorded")

    assert start_record is not None
    current_boot = start_record["boot_id_sha256"]
    after_services_sha = start_record["service_states_sha256"]
    if (
        not isinstance(current_boot, str)
        or current_boot == prior_boot
        or not isinstance(after_services_sha, str)
        or _SHA256.fullmatch(after_services_sha) is None
    ):
        raise RuntimeError("storage boot terminal observation is incomplete")
    stop_result = stop_record["result"]
    start_result = start_record["result"]
    if stop_result == "stopped" and start_result == "started":
        execution_state = "stop_start_completed"
    elif start_result == "started":
        execution_state = "resumed_from_terminated"
    else:
        execution_state = "verified_completed_before_retry"
    receipts = [stop_record["step_receipt"], start_record["step_receipt"]]
    mutation_performed = any(
        item.get("result") in {"stopped", "started"}
        for item in receipts
        if isinstance(item, Mapping)
    )
    completed_at = clock()
    unsigned = {
        "schema": BOOT_EXPANSION_RECEIPT_SCHEMA,
        "ok": True,
        "plan_sha256": plan.sha256,
        "approved_plan_sha256": approved_plan_sha256,
        "preflight_report_sha256": preflight["report_sha256"],
        "storage_report_sha256": preflight["storage_report_sha256"],
        "storage_apply_receipt_sha256": preflight["storage_apply_receipt_sha256"],
        "transaction_id": transaction_id,
        "journal_intent_sha256": intent["intent_sha256"],
        "journal_stop_sha256": stop_record["stop_sha256"],
        "journal_start_sha256": start_record["start_sha256"],
        "completed_at_unix": completed_at,
        "execution_state": execution_state,
        "mutation_performed": mutation_performed,
        "receipts": receipts,
        "vm_instance_id": VM_INSTANCE_ID,
        "disk_id": DISK_ID,
        "disk_size_gb": TARGET_BOOT_DISK_SIZE_GB,
        "prior_boot_id_sha256": prior_boot,
        "current_boot_id_sha256": current_boot,
        "runtime_units": list(CANARY_RUNTIME_UNITS),
        "service_states_before_sha256": preflight["service_states_sha256"],
        "service_states_after_sha256": after_services_sha,
        "initial_observation_sha256": intent["initial_observation_sha256"],
        "initial_observation_collected_at_unix": intent[
            "initial_observation_collected_at_unix"
        ],
        "stopped_observation_sha256": stop_record["observation_sha256"],
        "stopped_observation_collected_at_unix": stop_record[
            "observation_collected_at_unix"
        ],
        "started_observation_sha256": start_record["observation_sha256"],
        "started_observation_collected_at_unix": start_record[
            "observation_collected_at_unix"
        ],
        "stop_command_completed_at_unix": stop_record["command_completed_at_unix"],
        "start_command_completed_at_unix": start_record["command_completed_at_unix"],
        "boot_changed": True,
        "requires_storage_postflight": True,
        "opens_runtime_gate": False,
    }
    receipt = _seal(unsigned, digest_key="receipt_sha256")
    published = journal.publish(transaction_id, "completion", receipt)
    checkpoint("completion_recorded")
    return validate_receipt(
        plan=plan,
        preflight=preflight,
        receipt=published,
        now_unix=clock(),
        journal=journal,
    )


def validate_receipt(
    *,
    plan: StorageBootPlan,
    preflight: Mapping[str, object],
    receipt: Mapping[str, object],
    now_unix: int,
    journal: StorageBootJournal | None = None,
) -> dict[str, object]:
    """Validate the complete preflight → new-boot receipt relationship."""

    _require_preflight(preflight, plan=plan, now_unix=now_unix, require_fresh=False)
    if set(receipt) != _RECEIPT_FIELDS:
        raise RuntimeError("storage boot receipt fields are not exact")
    _require_sealed(receipt, digest_key="receipt_sha256", label="storage boot receipt")
    completed_at = receipt.get("completed_at_unix")
    prior = receipt.get("prior_boot_id_sha256")
    current = receipt.get("current_boot_id_sha256")
    raw_steps = receipt.get("receipts")
    transaction_id = _transaction_id(plan, preflight)
    if (
        receipt.get("schema") != BOOT_EXPANSION_RECEIPT_SCHEMA
        or receipt.get("ok") is not True
        or receipt.get("plan_sha256") != plan.sha256
        or receipt.get("approved_plan_sha256") != plan.sha256
        or receipt.get("preflight_report_sha256") != preflight.get("report_sha256")
        or receipt.get("storage_report_sha256")
        != preflight.get("storage_report_sha256")
        or receipt.get("storage_apply_receipt_sha256")
        != preflight.get("storage_apply_receipt_sha256")
        or receipt.get("transaction_id") != transaction_id
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in (
                receipt.get("journal_intent_sha256"),
                receipt.get("journal_stop_sha256"),
                receipt.get("journal_start_sha256"),
            )
        )
        or receipt.get("vm_instance_id") != VM_INSTANCE_ID
        or receipt.get("disk_id") != DISK_ID
        or receipt.get("disk_size_gb") != TARGET_BOOT_DISK_SIZE_GB
        or receipt.get("runtime_units") != list(CANARY_RUNTIME_UNITS)
        or receipt.get("service_states_before_sha256")
        != preflight.get("service_states_sha256")
        or not isinstance(receipt.get("service_states_after_sha256"), str)
        or _SHA256.fullmatch(str(receipt.get("service_states_after_sha256"))) is None
        or prior != preflight.get("prior_boot_id_sha256")
        or not isinstance(current, str)
        or _SHA256.fullmatch(current) is None
        or current == prior
        or receipt.get("boot_changed") is not True
        or receipt.get("requires_storage_postflight") is not True
        or receipt.get("opens_runtime_gate") is not False
        or receipt.get("execution_state")
        not in {
            "stop_start_completed",
            "resumed_from_terminated",
            "verified_completed_before_retry",
        }
        or type(receipt.get("mutation_performed")) is not bool
        or type(completed_at) is not int
        or completed_at < preflight["collected_at_unix"]
        or completed_at > now_unix
        or not isinstance(raw_steps, list)
        or len(raw_steps) != 2
        or [item.get("name") for item in raw_steps if isinstance(item, Mapping)]
        != [STOP_STEP, START_STEP]
    ):
        raise RuntimeError("storage boot receipt is not exact")
    state = receipt["execution_state"]
    mutation = receipt["mutation_performed"]
    compact_results = [item.get("result") for item in raw_steps]
    evidenced_mutation = any(
        result in {"stopped", "started"} for result in compact_results
    )
    if state == "stop_start_completed":
        invariant = compact_results == ["stopped", "started"]
    elif state == "resumed_from_terminated":
        invariant = compact_results == [
            "verified_terminated_after_intent",
            "started",
        ]
    else:
        invariant = compact_results in [
            ["stopped", "verified_new_boot_after_intent"],
            [
                "verified_terminated_after_intent",
                "verified_new_boot_after_intent",
            ],
            [
                "unobserved_stop_authorized_by_intent",
                "verified_new_boot_after_intent",
            ],
        ]
    if not invariant or mutation is not evidenced_mutation:
        raise RuntimeError("storage boot receipt retry state is inconsistent")
    stop_result = str(compact_results[0])
    start_result = str(compact_results[1])
    if not _step_receipt_exact(
        raw_steps[0],
        name=STOP_STEP,
        result=stop_result,
        command_evidenced=stop_result == "stopped",
    ) or not _step_receipt_exact(
        raw_steps[1],
        name=START_STEP,
        result=start_result,
        command_evidenced=start_result == "started",
    ):
        raise RuntimeError("storage boot receipt step is invalid")
    initial_at = receipt.get("initial_observation_collected_at_unix")
    stopped_at = receipt.get("stopped_observation_collected_at_unix")
    started_at = receipt.get("started_observation_collected_at_unix")
    stop_command_at = receipt.get("stop_command_completed_at_unix")
    start_command_at = receipt.get("start_command_completed_at_unix")
    initial_digest = receipt.get("initial_observation_sha256")
    stopped_digest = receipt.get("stopped_observation_sha256")
    started_digest = receipt.get("started_observation_sha256")
    stop_observation_evidenced = stop_result != "unobserved_stop_authorized_by_intent"
    stop_command_evidenced = stop_result == "stopped"
    start_command_evidenced = start_result == "started"
    if (
        type(initial_at) is not int
        or initial_at < preflight["collected_at_unix"]
        or not isinstance(initial_digest, str)
        or _SHA256.fullmatch(initial_digest) is None
        or (stop_observation_evidenced and type(stopped_at) is not int)
        or (not stop_observation_evidenced and stopped_at is not None)
        or (
            stop_observation_evidenced
            and (
                not isinstance(stopped_digest, str)
                or _SHA256.fullmatch(stopped_digest) is None
            )
        )
        or (not stop_observation_evidenced and stopped_digest is not None)
        or type(started_at) is not int
        or not isinstance(started_digest, str)
        or _SHA256.fullmatch(started_digest) is None
        or started_at <= initial_at
        or (stopped_at is not None and not initial_at < stopped_at < started_at)
        or (stop_command_evidenced and type(stop_command_at) is not int)
        or (not stop_command_evidenced and stop_command_at is not None)
        or (stop_command_evidenced and not initial_at <= stop_command_at <= stopped_at)
        or (start_command_evidenced and type(start_command_at) is not int)
        or (not start_command_evidenced and start_command_at is not None)
        or (
            start_command_evidenced
            and not (stopped_at if stopped_at is not None else initial_at)
            <= start_command_at
            <= started_at
        )
        or completed_at < started_at
    ):
        raise RuntimeError("storage boot receipt chronology is invalid")
    if journal is not None:
        raw_intent = journal.read(transaction_id, "intent")
        raw_stop = journal.read(transaction_id, "stop")
        raw_start = journal.read(transaction_id, "start")
        completion = journal.read(transaction_id, "completion")
        if raw_intent is None or raw_stop is None or raw_start is None:
            raise RuntimeError("storage boot receipt journal chain is incomplete")
        intent = _require_intent(
            raw_intent,
            plan=plan,
            preflight=preflight,
            transaction_id=transaction_id,
            now_unix=now_unix,
            require_unexpired=False,
        )
        stop = _require_transition_record(
            raw_stop,
            schema=BOOT_EXPANSION_STOP_SCHEMA,
            digest_key="stop_sha256",
            transaction_id=transaction_id,
            intent=intent,
            allowed_results={
                "stopped",
                "verified_terminated_after_intent",
                "unobserved_stop_authorized_by_intent",
            },
        )
        start = _require_transition_record(
            raw_start,
            schema=BOOT_EXPANSION_START_SCHEMA,
            digest_key="start_sha256",
            transaction_id=transaction_id,
            intent=intent,
            allowed_results={"started", "verified_new_boot_after_intent"},
        )
        if (
            completion is None
            or intent.get("intent_sha256") != receipt.get("journal_intent_sha256")
            or stop.get("stop_sha256") != receipt.get("journal_stop_sha256")
            or start.get("start_sha256") != receipt.get("journal_start_sha256")
            or receipt.get("initial_observation_sha256")
            != intent.get("initial_observation_sha256")
            or receipt.get("initial_observation_collected_at_unix")
            != intent.get("initial_observation_collected_at_unix")
            or receipt.get("stopped_observation_sha256")
            != stop.get("observation_sha256")
            or receipt.get("stopped_observation_collected_at_unix")
            != stop.get("observation_collected_at_unix")
            or receipt.get("stop_command_completed_at_unix")
            != stop.get("command_completed_at_unix")
            or receipt.get("started_observation_sha256")
            != start.get("observation_sha256")
            or receipt.get("started_observation_collected_at_unix")
            != start.get("observation_collected_at_unix")
            or receipt.get("start_command_completed_at_unix")
            != start.get("command_completed_at_unix")
            or receipt.get("current_boot_id_sha256") != start.get("boot_id_sha256")
            or receipt.get("service_states_after_sha256")
            != start.get("service_states_sha256")
            or receipt.get("receipts")
            != [stop.get("step_receipt"), start.get("step_receipt")]
            or stop.get("recorded_at_unix") > start.get("recorded_at_unix")
            or start.get("recorded_at_unix") > completed_at
            or completion != dict(receipt)
        ):
            raise RuntimeError("storage boot receipt journal chain is invalid")
    return dict(receipt)


def _boot_id_sha256(raw: bytes) -> str:
    try:
        value = raw.decode("ascii", errors="strict").strip()
    except UnicodeError:
        raise RuntimeError("storage boot identity output is invalid") from None
    if _BOOT_ID.fullmatch(value) is None:
        raise RuntimeError("storage boot identity output is invalid")
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _collect_guest_state(
    transport: object, *, account: str
) -> tuple[str, list[dict[str, object]]]:
    boot_completed = transport._run_remote(
        BOOT_ID_COMMAND,
        account=account,
        timeout_seconds=90.0,
        maximum_output_bytes=_MAX_GUEST_OUTPUT_BYTES,
    )
    if boot_completed.returncode != 0 or not isinstance(boot_completed.stdout, bytes):
        raise RuntimeError("storage boot identity is unavailable")
    states: list[dict[str, object]] = []
    for unit in CANARY_RUNTIME_UNITS:
        command = _service_observation_command(unit)
        completed = transport._run_remote(
            command,
            account=account,
            timeout_seconds=90.0,
            maximum_output_bytes=_MAX_GUEST_OUTPUT_BYTES,
        )
        if completed.returncode != 0 or not isinstance(completed.stdout, bytes):
            raise RuntimeError("storage boot service state is unavailable")
        try:
            raw = completed.stdout.decode("utf-8", errors="strict")
        except UnicodeError:
            raise RuntimeError("storage boot service state is invalid") from None
        states.append(_parse_service_observation(unit, raw))
    return _boot_id_sha256(boot_completed.stdout), states


def _service_observation_command(unit: str) -> tuple[str, ...]:
    if unit not in CANARY_RUNTIME_UNITS:
        raise RuntimeError("storage boot service unit is not exact")
    return (
        str(DEFAULT_SYSTEMCTL_EXECUTABLE),
        "show",
        "--no-pager",
        *(f"--property={name}" for name in _SERVICE_PROPERTIES),
        unit,
    )


def collect_preflight(
    *, storage_apply_receipt: Mapping[str, object]
) -> dict[str, object]:
    """Collect the fixed live preflight without changing guest or cloud state."""

    from scripts.canary.host_storage_preflight import (
        _run_guest_json,
        _sealed_owner_transport,
    )

    owner, transport = _sealed_owner_transport()
    account = owner.account_for_read_only_preflight()
    if account != OWNER_ACCOUNT:
        raise RuntimeError("storage boot owner identity is not exact")
    storage_evidence = collect_storage(
        run_json=owner.run_canary_iam_read_only_json,
        run_guest_json=lambda argv: _run_guest_json(
            argv,
            owner_identity=owner,
            transport=transport,
        ),
    )
    boot_id, services = _collect_guest_state(transport, account=account)
    return build_preflight(
        storage_apply_receipt=storage_apply_receipt,
        storage_report=evaluate_storage(storage_evidence),
        prior_boot_id_sha256=boot_id,
        service_states=services,
    )


def _sealed_observer() -> Observer:
    from scripts.canary.host_storage_preflight import _sealed_owner_transport

    owner, transport = _sealed_owner_transport()
    account = owner.account_for_read_only_preflight()
    if account != OWNER_ACCOUNT:
        raise RuntimeError("storage boot owner identity is not exact")
    storage_plan = build_storage_plan()
    instance_command = (
        "gcloud",
        "compute",
        "instances",
        "describe",
        storage_plan.spec.vm_name,
        f"--zone={storage_plan.spec.zone}",
        f"--project={storage_plan.spec.project}",
        "--format=json",
    )
    disk_command = (
        "gcloud",
        "compute",
        "disks",
        "describe",
        storage_plan.spec.disk_name,
        f"--zone={storage_plan.spec.zone}",
        f"--project={storage_plan.spec.project}",
        "--format=json",
    )

    def observe() -> Mapping[str, object]:
        instance = owner.run_canary_iam_read_only_json(instance_command)
        disk = owner.run_canary_iam_read_only_json(disk_command)
        status = instance.get("status") if isinstance(instance, Mapping) else None
        boot_id: str | None = None
        services: list[dict[str, object]] | None = None
        if status == "RUNNING":
            boot_id, services = _collect_guest_state(transport, account=account)
        return {
            "collected_at_unix": int(time.time()),
            "instance": instance,
            "disk": disk,
            "boot_id_sha256": boot_id,
            "service_states": services,
        }

    return observe


def _load(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise SystemExit("storage boot input must be a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("plan", "preflight", "apply"),
        nargs="?",
        default="plan",
    )
    parser.add_argument("--storage-apply-receipt", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--approved-plan-sha256")
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = build_plan()
    if args.command == "plan":
        result = plan.report()
    elif args.command == "preflight":
        if args.storage_apply_receipt is None:
            raise SystemExit("preflight requires --storage-apply-receipt")
        result = collect_preflight(
            storage_apply_receipt=_load(args.storage_apply_receipt)
        )
    else:
        if args.preflight is None or not args.approved_plan_sha256:
            raise SystemExit("apply requires --preflight and --approved-plan-sha256")
        result = execute_plan(
            plan,
            approved_plan_sha256=args.approved_plan_sha256,
            preflight=_load(args.preflight),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOT_EXPANSION_PLAN_SCHEMA",
    "BOOT_EXPANSION_PREFLIGHT_SCHEMA",
    "BOOT_EXPANSION_RECEIPT_SCHEMA",
    "BOOT_ID_COMMAND",
    "CANARY_RUNTIME_UNITS",
    "START_STEP",
    "STOP_STEP",
    "StorageBootPlan",
    "build_plan",
    "build_preflight",
    "collect_preflight",
    "execute_plan",
    "main",
    "validate_receipt",
]
