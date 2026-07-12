#!/usr/bin/env python3
"""Digest-bound phase 3 host creation for the isolated Muncho canary."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from scripts.canary.foundation import (
    NETWORK,
    PREFLIGHT_MAX_AGE_SECONDS,
    PROJECT,
    SERVICE_ACCOUNT_NAME,
    SUBNET,
    ZONE,
    PlanStep,
)
from scripts.canary.network_boundary import (
    NetworkBoundarySpec,
    build_plan as build_network_plan,
)


VM_NAME = "muncho-canary-v2-01"
IMAGE_PROJECT = "debian-cloud"
IMAGE = "debian-12-bookworm-v20260609"
MACHINE_TYPE = "e2-medium"
BOOT_DISK_SIZE_GB = 20
BOOT_DISK_TYPE = "pd-balanced"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class HostSpec:
    sql_private_ip: str
    network_plan_sha256: str
    project: str = PROJECT
    zone: str = ZONE
    network: str = NETWORK
    subnet: str = SUBNET
    service_account_name: str = SERVICE_ACCOUNT_NAME
    vm_name: str = VM_NAME
    image_project: str = IMAGE_PROJECT
    image: str = IMAGE
    machine_type: str = MACHINE_TYPE
    boot_disk_size_gb: int = BOOT_DISK_SIZE_GB
    boot_disk_type: str = BOOT_DISK_TYPE

    @property
    def service_account_email(self) -> str:
        return f"{self.service_account_name}@{self.project}.iam.gserviceaccount.com"

    def validate(self) -> None:
        try:
            address = ipaddress.ip_address(self.sql_private_ip)
        except ValueError as exc:
            raise ValueError("SQL private IP is invalid") from exc
        if address.version != 4 or not address.is_private:
            raise ValueError("SQL address must be private IPv4")
        if not _SHA256.fullmatch(self.network_plan_sha256):
            raise ValueError("network plan digest must be lowercase SHA-256")
        expected_network_plan = build_network_plan(
            NetworkBoundarySpec(
                sql_private_ip=self.sql_private_ip,
                project=self.project,
                network=self.network,
                service_account_name=self.service_account_name,
            )
        )
        if self.network_plan_sha256 != expected_network_plan.sha256:
            raise ValueError("network plan digest does not match exact SQL endpoint")
        if (
            self.project != PROJECT
            or self.zone != ZONE
            or self.network != NETWORK
            or self.subnet != SUBNET
            or self.service_account_name != SERVICE_ACCOUNT_NAME
            or self.vm_name != VM_NAME
            or self.image_project != IMAGE_PROJECT
            or self.image != IMAGE
            or self.machine_type != MACHINE_TYPE
            or self.boot_disk_size_gb != BOOT_DISK_SIZE_GB
            or self.boot_disk_type != BOOT_DISK_TYPE
        ):
            raise ValueError("host shape is production-pinned")


@dataclass(frozen=True)
class HostPlan:
    schema: str
    spec: HostSpec
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
        encoded = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def report(self) -> dict[str, object]:
        return {**self.payload(), "plan_sha256": self.sha256}


def build_plan(spec: HostSpec) -> HostPlan:
    spec.validate()
    return HostPlan(
        schema="muncho-isolated-canary-host-plan.v1",
        spec=spec,
        architecture={
            "phase": 3,
            "creates_vm": True,
            "creates_network_rules": False,
            "creates_secret_manager_resources": False,
            "requires_live_complete_network_attestation": True,
            "single_host_step": True,
            "requires_post_apply_attestation": True,
        },
        steps=(
            PlanStep(
                "create_isolated_canary_vm",
                (
                    "gcloud",
                    "compute",
                    "instances",
                    "create",
                    spec.vm_name,
                    f"--project={spec.project}",
                    f"--zone={spec.zone}",
                    f"--machine-type={spec.machine_type}",
                    f"--network={spec.network}",
                    f"--subnet={spec.subnet}",
                    "--stack-type=IPV4_ONLY",
                    "--network-tier=PREMIUM",
                    f"--image-project={spec.image_project}",
                    f"--image={spec.image}",
                    f"--boot-disk-size={spec.boot_disk_size_gb}GB",
                    f"--boot-disk-type={spec.boot_disk_type}",
                    f"--service-account={spec.service_account_email}",
                    "--scopes=logging-write,monitoring-write",
                    "--tags=iap-ssh",
                    "--metadata=disable-legacy-endpoints=TRUE,enable-oslogin=TRUE",
                    "--shielded-secure-boot",
                    "--shielded-vtpm",
                    "--shielded-integrity-monitoring",
                    "--no-can-ip-forward",
                    "--no-deletion-protection",
                    "--maintenance-policy=MIGRATE",
                    "--provisioning-model=STANDARD",
                    "--restart-on-failure",
                    "--quiet",
                ),
            ),
        ),
    )


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), check=False, capture_output=True, text=True, timeout=1800
    )


def execute_plan(
    plan: HostPlan,
    *,
    approved_plan_sha256: str,
    preflight: Mapping[str, object],
    runner: Runner = _runner,
    now_unix: int | None = None,
) -> dict[str, object]:
    if approved_plan_sha256 != plan.sha256:
        raise RuntimeError("approved host plan digest mismatch")
    if preflight.get("schema") != "muncho-isolated-canary-host-preflight.v1":
        raise RuntimeError("host preflight schema mismatch")
    if preflight.get("ok") is not True or preflight.get("plan_sha256") != plan.sha256:
        raise RuntimeError("host preflight did not pass for exact plan")
    if preflight.get("network_plan_sha256") != plan.spec.network_plan_sha256:
        raise RuntimeError("host preflight network plan mismatch")
    if preflight.get("sql_private_ip") != plan.spec.sql_private_ip:
        raise RuntimeError("host preflight SQL endpoint mismatch")
    collected_at = preflight.get("collected_at_unix")
    now = int(time.time()) if now_unix is None else now_unix
    if (
        type(collected_at) is not int
        or now < collected_at
        or now - collected_at > PREFLIGHT_MAX_AGE_SECONDS
    ):
        raise RuntimeError("host preflight is stale")
    raw_satisfied = preflight.get("satisfied_steps")
    if not isinstance(raw_satisfied, list) or any(
        not isinstance(item, str) for item in raw_satisfied
    ):
        raise RuntimeError("host preflight satisfied-step inventory is invalid")
    allowed_steps = {step.name for step in plan.steps}
    satisfied = set(raw_satisfied)
    if not satisfied <= allowed_steps:
        raise RuntimeError("host preflight contains unknown satisfied step")

    receipts: list[dict[str, object]] = []
    for step in plan.steps:
        if step.name in satisfied:
            receipts.append({"name": step.name, "result": "verified_existing"})
            continue
        completed = runner(step.argv)
        receipts.append(
            {
                "name": step.name,
                "result": "created" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
            }
        )
        if completed.returncode != 0:
            return {
                "schema": "muncho-isolated-canary-host-receipt.v1",
                "ok": False,
                "plan_sha256": plan.sha256,
                "receipts": receipts,
            }
    return {
        "schema": "muncho-isolated-canary-host-receipt.v1",
        "ok": True,
        "requires_post_apply_attestation": True,
        "plan_sha256": plan.sha256,
        "receipts": receipts,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "apply"), nargs="?", default="plan")
    parser.add_argument("--sql-private-ip", required=True)
    parser.add_argument("--network-plan-sha256", required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--approved-plan-sha256")
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = build_plan(
        HostSpec(
            sql_private_ip=args.sql_private_ip,
            network_plan_sha256=args.network_plan_sha256,
        )
    )
    if args.command == "plan":
        print(json.dumps(plan.report(), indent=2, sort_keys=True))
        return 0
    if args.preflight is None or not args.approved_plan_sha256:
        raise SystemExit("apply requires --preflight and --approved-plan-sha256")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    receipt = execute_plan(
        plan,
        approved_plan_sha256=args.approved_plan_sha256,
        preflight=preflight,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
