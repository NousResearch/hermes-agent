#!/usr/bin/env python3
"""Read-only exact attestation for phase 3 of the Muncho canary."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from scripts.canary.foundation import PREFLIGHT_MAX_AGE_SECONDS, PROJECT, ZONE
from scripts.canary.host import (
    IMAGE,
    IMAGE_PROJECT,
    VM_NAME,
    HostSpec,
    build_plan,
)
from scripts.canary.network_boundary import (
    NetworkBoundarySpec,
    build_plan as build_network_plan,
)
from scripts.canary.network_preflight import collect as collect_network
from scripts.canary.network_preflight import evaluate as evaluate_network


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _ends_with(value: object, suffix: str) -> bool:
    return isinstance(value, str) and value.endswith(suffix)


def _metadata_exact(raw: object) -> bool:
    if not isinstance(raw, Mapping):
        return False
    items = raw.get("items") or []
    if not isinstance(items, list):
        return False
    values = {
        str(item.get("key")): str(item.get("value"))
        for item in items
        if isinstance(item, Mapping) and item.get("key")
    }
    return values == {
        "disable-legacy-endpoints": "TRUE",
        "enable-oslogin": "TRUE",
    }


def _vm_exact(raw: object, disk: object, spec: HostSpec) -> bool:
    if not isinstance(raw, Mapping) or not isinstance(disk, Mapping):
        return False
    service_accounts = raw.get("serviceAccounts")
    interfaces = raw.get("networkInterfaces")
    disks = raw.get("disks")
    scheduling = raw.get("scheduling")
    shielded = raw.get("shieldedInstanceConfig")
    tags = raw.get("tags")
    if not all(
        isinstance(value, expected)
        for value, expected in (
            (service_accounts, list),
            (interfaces, list),
            (disks, list),
            (scheduling, Mapping),
            (shielded, Mapping),
            (tags, Mapping),
        )
    ):
        return False
    if len(service_accounts) != 1 or not isinstance(service_accounts[0], Mapping):
        return False
    expected_scopes = {
        "https://www.googleapis.com/auth/logging.write",
        "https://www.googleapis.com/auth/monitoring.write",
    }
    account = service_accounts[0]
    if len(interfaces) != 1 or not isinstance(interfaces[0], Mapping):
        return False
    interface = interfaces[0]
    access_configs = interface.get("accessConfigs")
    if not isinstance(access_configs, list) or len(access_configs) != 1:
        return False
    access = access_configs[0]
    if not isinstance(access, Mapping):
        return False
    if len(disks) != 1 or not isinstance(disks[0], Mapping):
        return False
    attached_disk = disks[0]
    labels = raw.get("labels") or {}
    return bool(
        raw.get("name") == spec.vm_name
        and raw.get("status") == "RUNNING"
        and _ends_with(raw.get("zone"), f"/zones/{spec.zone}")
        and _ends_with(raw.get("machineType"), f"/machineTypes/{spec.machine_type}")
        and raw.get("canIpForward") is False
        and raw.get("deletionProtection") is False
        and labels == {}
        and not (raw.get("resourcePolicies") or [])
        and set(tags.get("items") or []) == {"iap-ssh"}
        and _metadata_exact(raw.get("metadata"))
        and account.get("email") == spec.service_account_email
        and set(account.get("scopes") or []) == expected_scopes
        and _ends_with(interface.get("network"), f"/networks/{spec.network}")
        and _ends_with(interface.get("subnetwork"), f"/subnetworks/{spec.subnet}")
        and interface.get("stackType") in {None, "IPV4_ONLY"}
        and bool(interface.get("networkIP"))
        and access.get("type") == "ONE_TO_ONE_NAT"
        and access.get("networkTier") == "PREMIUM"
        and bool(access.get("natIP"))
        and attached_disk.get("boot") is True
        and attached_disk.get("autoDelete") is True
        and scheduling.get("automaticRestart") is True
        and scheduling.get("onHostMaintenance") == "MIGRATE"
        and scheduling.get("preemptible") is False
        and scheduling.get("provisioningModel") == "STANDARD"
        and shielded.get("enableSecureBoot") is True
        and shielded.get("enableVtpm") is True
        and shielded.get("enableIntegrityMonitoring") is True
        and disk.get("name") == spec.vm_name
        and str(disk.get("sizeGb")) == str(spec.boot_disk_size_gb)
        and _ends_with(disk.get("type"), f"/diskTypes/{spec.boot_disk_type}")
        and _ends_with(
            disk.get("sourceImage"),
            f"/projects/{spec.image_project}/global/images/{spec.image}",
        )
    )


def _network_complete(
    report: object, *, collected_at_unix: object
) -> tuple[bool, bool, str, str]:
    if not isinstance(report, Mapping):
        return False, False, "", ""
    sql_ip = str(report.get("sql_private_ip") or "")
    try:
        expected_plan = build_network_plan(NetworkBoundarySpec(sql_private_ip=sql_ip))
    except ValueError:
        return False, False, "", sql_ip
    expected_steps = [step.name for step in expected_plan.steps]
    raw_checks = report.get("checks")
    check_items = raw_checks if isinstance(raw_checks, list) else []
    required_checks = {
        "foundation.complete_exact",
        "foundation.preflight_fresh",
        "sql.exact_private_ready",
        "identity.target_service_account_exact",
        "firewall.iap_absent_or_exact",
        "firewall.allow_absent_or_exact",
        "firewall.deny_absent_or_exact",
        "firewall.effective_surface_exact",
    }
    passed_checks = {
        str(item.get("name"))
        for item in check_items
        if isinstance(item, Mapping) and item.get("passed") is True
    }
    complete = bool(
        report.get("schema") == "muncho-isolated-canary-network-preflight.v1"
        and report.get("ok") is True
        and report.get("plan_sha256") == expected_plan.sha256
        and report.get("satisfied_steps") == expected_steps
        and passed_checks == required_checks
    )
    network_collected = report.get("collected_at_unix")
    fresh = bool(
        type(collected_at_unix) is int
        and type(network_collected) is int
        and 0 <= collected_at_unix - network_collected <= PREFLIGHT_MAX_AGE_SECONDS
    )
    return complete, fresh, expected_plan.sha256, sql_ip


def evaluate(evidence: Mapping[str, object]) -> dict[str, object]:
    network_complete, network_fresh, network_sha, sql_ip = _network_complete(
        evidence.get("network_report"),
        collected_at_unix=evidence.get("collected_at_unix"),
    )
    plan = None
    spec = None
    if network_sha and sql_ip:
        spec = HostSpec(sql_private_ip=sql_ip, network_plan_sha256=network_sha)
        plan = build_plan(spec)
    instances = evidence.get("instances")
    instance_items = instances if isinstance(instances, list) else []
    matching = [
        item
        for item in instance_items
        if isinstance(item, Mapping) and item.get("name") == VM_NAME
    ]
    vm_present = bool(matching)
    vm_exact = bool(
        spec
        and len(matching) == 1
        and _vm_exact(evidence.get("planned_vm"), evidence.get("planned_vm_disk"), spec)
    )
    image = evidence.get("image")
    image_exact = bool(
        isinstance(image, Mapping)
        and image.get("name") == IMAGE
        and image.get("status") == "READY"
        and not image.get("deprecated")
    )
    checks = [
        Check(
            "network.complete_exact",
            network_complete,
            "all phase-2 rules must be live and exactly attested before VM creation",
        ),
        Check(
            "network.preflight_fresh",
            network_fresh,
            "the nested phase-2 attestation must be fresh",
        ),
        Check(
            "image.exact_ready",
            image_exact,
            "the immutable Debian image must exist and be READY",
        ),
        Check(
            "resource.vm_absent_or_exact_running",
            not vm_present or vm_exact,
            "the canary VM must be absent or match the complete exact running shape",
        ),
    ]
    satisfied_steps = ["create_isolated_canary_vm"] if vm_exact else []
    return {
        "schema": "muncho-isolated-canary-host-preflight.v1",
        "ok": bool(plan) and all(check.passed for check in checks),
        "collected_at_unix": evidence.get("collected_at_unix"),
        "network_plan_sha256": network_sha or None,
        "sql_private_ip": sql_ip,
        "plan_sha256": plan.sha256 if plan else None,
        "satisfied_steps": satisfied_steps,
        "checks": [asdict(check) for check in checks],
    }


def _run_json(argv: Sequence[str]) -> object:
    completed = subprocess.run(
        list(argv), check=False, capture_output=True, text=True, timeout=60
    )
    if completed.returncode != 0:
        raise RuntimeError(f"read-only gcloud command failed: {argv[1:3]}")
    return json.loads(completed.stdout or "null")


def collect() -> dict[str, object]:
    project_flag = f"--project={PROJECT}"
    network_report = evaluate_network(collect_network())
    instances = _run_json(
        (
            "gcloud",
            "compute",
            "instances",
            "list",
            project_flag,
            "--format=json",
        )
    )
    instance_names = {
        str(item.get("name"))
        for item in (instances if isinstance(instances, list) else [])
        if isinstance(item, Mapping) and item.get("name")
    }
    evidence: dict[str, object] = {
        "collected_at_unix": int(time.time()),
        "network_report": network_report,
        "instances": instances,
        "image": _run_json(
            (
                "gcloud",
                "compute",
                "images",
                "describe",
                IMAGE,
                f"--project={IMAGE_PROJECT}",
                "--format=json",
            )
        ),
    }
    if VM_NAME in instance_names:
        evidence["planned_vm"] = _run_json(
            (
                "gcloud",
                "compute",
                "instances",
                "describe",
                VM_NAME,
                f"--zone={ZONE}",
                project_flag,
                "--format=json",
            )
        )
        evidence["planned_vm_disk"] = _run_json(
            (
                "gcloud",
                "compute",
                "disks",
                "describe",
                VM_NAME,
                f"--zone={ZONE}",
                project_flag,
                "--format=json",
            )
        )
    return evidence


def main() -> int:
    report = evaluate(collect())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
